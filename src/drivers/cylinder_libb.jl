using KernelAbstractions

# =====================================================================
# Schäfer-Turek-style cylinder drivers with Bouzidi LI-BB.
#
# Uses the Ginzburg-exact V2 LI-BB kernel (pre-phase Bouzidi + TRT),
# combined with Zou-He inlet (parabolic profile) and an extrapolation
# pressure outlet (`NeqExtrapolationPressure`). Top and bottom walls are handled by the fused kernel's
# default halfway-BB fallback at the y domain edges.
#
# Drag / lift measured via the MEA (momentum-exchange) formula
# adapted to LI-BB: for each fluid cell adjacent to the cylinder and
# each flagged cut link q, the momentum transferred per step is
# 2 · c_q · f_q(post-collision). We sum over all flagged fluid links
# and average over a window of the final steps.
# =====================================================================

"""
    compute_drag_libb_2d(f_pre, q_wall, Nx, Ny)
        -> (Fx, Fy)

Simple halfway-BB MEA drag (2·c_q·f_q at the fluid cell). Underestimates
Cd when q_w ≠ 0.5; kept for the old scaffold/test. Prefer
`compute_drag_libb_mei_2d` for quantitative Schäfer-Turek-style
benchmarks.
"""
function compute_drag_libb_2d(f_post, q_wall, Nx::Int, Ny::Int)
    f = Array(f_post)
    qw = Array(q_wall)
    cxv = (0, 1, 0, -1,  0, 1, -1, -1,  1)
    cyv = (0, 0, 1,  0, -1, 1,  1, -1, -1)
    Fx = 0.0; Fy = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx
        for q in 2:9
            if qw[i, j, q] > 0
                Fx += 2.0 * Float64(cxv[q]) * Float64(f[i, j, q])
                Fy += 2.0 * Float64(cyv[q]) * Float64(f[i, j, q])
            end
        end
    end
    return (Fx = Fx, Fy = Fy)
end

"""
    compute_drag_libb_mei_2d(f_pre, q_wall, uw_link_x, uw_link_y, Nx, Ny)
        -> (Fx, Fy)

Mei-Luo-Shyy 2002 (J. Comput. Phys. 161, 680) momentum-exchange drag
for LI-BB. Consistent with the V2 pre-phase Bouzidi substitution: for
each cut link q on a fluid cell, the force per link per step is

    F_link = c_q · (f_q_pre + f_q̄_bouzidi)

where `f_q_pre = f_pre[i,j,q]` (pop q pre-step at the fluid cell,
i.e. post-coll from the previous step) and `f_q̄_bouzidi` is the
arriving pop computed via the same `_libb_branch` used in the
kernel's ApplyLiBBPrePhase brick (lag-1 Bouzidi estimate). Reduces to
`2·c_q·f_q` at q_w = 0.5 stationary, matching the halfway formula.

`f_pre` must be the `f_in` array at step N (= `f_out` from step N-1
BEFORE the boundary patches of step N are applied; in the driver we
pass the full `f_in` each sampling step). `uw_link_x/y` are the per-
link wall-velocity arrays (zero for a stationary cylinder).
"""
function compute_drag_libb_mei_2d(f_pre, q_wall, uw_link_x, uw_link_y,
                                    Nx::Int, Ny::Int)
    backend = KernelAbstractions.get_backend(f_pre)
    if backend isa KernelAbstractions.CPU
        return _compute_drag_libb_mei_2d_host(f_pre, q_wall, uw_link_x,
                                                uw_link_y, Nx, Ny)
    else
        return _compute_drag_libb_mei_2d_gpu_cached(f_pre, q_wall, uw_link_x,
                                                      uw_link_y, Nx, Ny)
    end
end

function _compute_drag_libb_mei_2d_host(f_pre, q_wall, uw_link_x, uw_link_y,
                                          Nx::Int, Ny::Int)
    f = Array(f_pre)
    qw = Array(q_wall)
    uwx = Array(uw_link_x); uwy = Array(uw_link_y)
    cxv = (0, 1, 0, -1,  0, 1, -1, -1,  1)
    cyv = (0, 0, 1,  0, -1, 1,  1, -1, -1)
    Fx = 0.0; Fy = 0.0
    @inbounds for j in 1:Ny, i in 1:Nx
        for q in 2:9
            q_w = qw[i, j, q]
            q_w > 0 || continue
            qbar = _D2Q9_OPP[q]
            im = i - cxv[q]; jm = j - cyv[q]
            fp_q_back = (1 <= im <= Nx && 1 <= jm <= Ny) ?
                Float64(f[im, jm, q]) : Float64(f[i, j, qbar])
            fq_here = Float64(f[i, j, q])
            fqbar_here = Float64(f[i, j, qbar])
            w_q = (q in (2, 3, 4, 5)) ? 1.0/9.0 : 1.0/36.0
            cu = Float64(cxv[q]) * uwx[i, j, q] +
                 Float64(cyv[q]) * uwy[i, j, q]
            δ = -6.0 * w_q * cu
            arriving = if q_w ≤ 0.5
                2q_w * fq_here + (1 - 2q_w) * fp_q_back + δ
            else
                inv2q = 1.0 / (2q_w)
                inv2q * fq_here + (1 - inv2q) * fqbar_here + inv2q * δ
            end
            F_link = fq_here + arriving
            Fx += Float64(cxv[q]) * F_link
            Fy += Float64(cyv[q]) * F_link
        end
    end
    return (Fx = Fx, Fy = Fy)
end

# Cache: per-q_wall device buffer of cut-link list + per-link Fx/Fy
# scratch arrays. Indexed by objectid(q_wall) so repeated calls with
# the same array reuse the upload (the q_wall geometry is static for
# a given run). For multi-cylinder / dynamic geometry, use the
# explicit GPU API in `kernels/drag_gpu.jl`.
const _DRAG_GPU_2D_CACHE = IdDict{Any, Any}()

function _compute_drag_libb_mei_2d_gpu_cached(f_pre, q_wall, uw_x, uw_y,
                                                Nx::Int, Ny::Int)
    cache = get!(_DRAG_GPU_2D_CACHE, q_wall) do
        T = eltype(f_pre)
        backend = KernelAbstractions.get_backend(f_pre)
        qw_h = Array(q_wall)
        links = build_cut_link_list_2d(qw_h; backend=backend)
        Fx_l = KernelAbstractions.allocate(backend, T, links.Nlinks)
        Fy_l = KernelAbstractions.allocate(backend, T, links.Nlinks)
        Fx_buf = zeros(T, links.Nlinks)
        Fy_buf = zeros(T, links.Nlinks)
        (links, Fx_l, Fy_l, Fx_buf, Fy_buf)
    end
    links, Fx_l, Fy_l, Fx_buf, Fy_buf = cache
    compute_drag_libb_mei_2d_gpu!(Fx_l, Fy_l, links, f_pre, uw_x, uw_y, Nx, Ny)
    copyto!(Fx_buf, Fx_l)
    copyto!(Fy_buf, Fy_l)
    return (Fx = Float64(sum(Fx_buf)), Fy = Float64(sum(Fy_buf)))
end

"""
    rebuild_inlet_outlet_libb_2d!(f_out, f_in, u_profile, ρ_out, ν, Nx, Ny)

Legacy wrapper for the hardcoded (Zou-He velocity west + extrapolation
pressure outlet east) rebuild. Prefer the modular `apply_bc_rebuild_2d!(f_out,
f_in, BCSpec2D(west=ZouHeVelocity(u_profile),
east=NeqExtrapolationPressure(ρ_out)), ν, Nx, Ny)` in new code — it decouples
BC choice from the driver. With the default half-way south/north walls,
both inlet and outlet rebuild all rows `1:Ny`.
"""
function rebuild_inlet_outlet_libb_2d!(f_out, f_in, u_profile, ρ_out::Real,
                                        ν::Real, Nx::Int, Ny::Int)
    bc = BCSpec2D(; west = ZouHeVelocity(u_profile),
                    east = NeqExtrapolationPressure(eltype(f_out)(ρ_out)))
    apply_bc_rebuild_2d!(f_out, f_in, bc, ν, Nx, Ny)
end

"""
    run_cylinder_libb_2d(; Nx=300, Ny=80, cx=Nx÷4, cy=Ny÷2, radius=10,
                          u_in=0.04, ν=0.04, max_steps=50000,
                          avg_window=5000, inlet=:parabolic, ρ_out=1.0,
                          stop=:none, tol=1e-6, check_every=nothing)

2D cylinder with LI-BB V2 boundary condition. Parabolic / uniform
Zou-He velocity inlet (reconstructed from pre-step populations) and an
extrapolation pressure outlet (`NeqExtrapolationPressure`, rebuilt from the
post-collision column Nx-1, including rows 1 and Ny); both bypass the kernel's
halfway-BB fallback at the non-wall boundaries. Top / bottom channel walls are
halfway-BB via the kernel's fallback at j=1 and j=Ny, so they sit at
y = -1/2 and y = Ny - 1/2 (node j at y = j - 1) and the channel height is Ny.

Arguments:
- `u_in`: for the `:parabolic` inlet, the centreline velocity U_m of the
  channel parabola, i.e. 3/2 of the delivered mean velocity (the sampled
  peak is `u_in · Ny² / (Ny² + 1/2)`); uniform u for `:uniform`.
- `inlet`: `:parabolic` (Schäfer-Turek 2D convention) or `:uniform`.
- `ρ_out`: outlet density imposed by the east pressure outlet (default 1).
- `stop`: `:none` (default) runs exactly `max_steps` steps and computes the same
  values as without the option. `:cd_change` samples the drag coefficient every
  `check_every` steps and stops as soon as two consecutive samples satisfy
  `|Cd_k - Cd_(k-1)| <= tol * |Cd_k|`. `max_steps` stays the hard cap.
- `tol`: relative tolerance of the `:cd_change` rule (default `1e-6`).
- `check_every`: steps between two samples. Default `nothing`: one flow-through,
  `round(Int, Nx / u_ref)` steps.

The `:parabolic` profile vanishes on the half-way walls and its row sum is
exactly `u_ref · Ny`, `u_ref = (2/3) u_in`:
`u(j) = 4 u_in (j - 1/2)(Ny + 1/2 - j) / (Ny² + 1/2)`. The Zou-He rebuild
imposes it on every row `1:Ny`, rows 1 and Ny included.

Returns a NamedTuple with:
- `ρ`, `ux`, `uy`, `is_solid`, `q_wall`, `u_ref`, `D`, `Fx`, `Fy`
- `Cd = 2·Fx / (u_ref² · D)` with `u_ref = u_mean = 2/3 · u_max`
  (parabolic) or `u_ref = u_in` (uniform).
- `Cl` analogously.
- `converged::Bool`: `true` if the `:cd_change` rule was met (always `false` with
  `stop = :none`); `stop_step`: the number of steps run.
- `cd_history`: vector of `(step, Cd)` samples taken every `check_every` steps
  (empty with `stop = :none`). These are instantaneous values, not window means.

`Cd`, `Cl`, `Fx`, `Fy` are the mean over the last `avg_window` steps of `max_steps`.
A run stopped inside that window averages the samples taken so far; a run stopped
before it has none, so they take the values of the last `cd_history` sample.

Reference for Schäfer-Turek 2D-1 (Re=20, blockage ≈ 25%): Cd ≈ 5.58.
"""
function run_cylinder_libb_2d(; Nx::Int=300, Ny::Int=80,
                                cx::Union{Nothing,Real}=nothing,
                                cy::Union{Nothing,Real}=nothing,
                                radius::Real=10,
                                u_in::Real=0.04, ν::Real=0.04,
                                max_steps::Int=50_000,
                                avg_window::Int=5_000,
                                inlet::Symbol=:parabolic,
                                ρ_out::Real=1.0,
                                stop::Symbol=:none,
                                tol::Real=1e-6,
                                check_every::Union{Nothing,Integer}=nothing,
                                backend=KernelAbstractions.CPU(),
                                T::Type{<:AbstractFloat}=Float64)
    stop in (:none, :cd_change) ||
        throw(ArgumentError("stop must be :none or :cd_change, got $(stop)"))
    tol >= 0 || throw(ArgumentError("tol must be non-negative"))
    cx = isnothing(cx) ? Nx ÷ 4 : Float64(cx)
    cy = isnothing(cy) ? Ny ÷ 2 : Float64(cy)

    q_wall_h, is_solid_h = precompute_q_wall_cylinder(Nx, Ny, cx, cy, radius; FT=T)

    u_prof_h = if inlet === :parabolic
        # Half-way channel walls at y = -1/2 and Ny - 1/2 (node j at y = j - 1):
        # the parabola vanishes on them and its row sum is exactly
        # (2/3) u_in Ny, i.e. a mean velocity of u_ref over the height Ny.
        _halfway_poiseuille_profile(T, Ny, (2 / 3) * Float64(u_in))
    elseif inlet === :uniform
        fill(T(u_in), Ny)
    else
        error("unknown inlet $(inlet); expected :parabolic or :uniform")
    end
    u_ref = inlet === :parabolic ? (2 / 3) * Float64(u_in) : Float64(u_in)

    # Host-side initial populations
    f_in_h  = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_in_h[i, j, q] = KrakeNN.equilibrium(D2Q9(), one(T), u_prof_h[j],
                                               zero(T), q)
    end

    # Device allocations
    q_wall    = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    is_solid  = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    uw_x      = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    uw_y      = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_in      = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_out     = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    ρ         = KernelAbstractions.allocate(backend, T, Nx, Ny)
    ux        = KernelAbstractions.allocate(backend, T, Nx, Ny)
    uy        = KernelAbstractions.allocate(backend, T, Nx, Ny)
    u_profile = KernelAbstractions.allocate(backend, T, Ny)
    copyto!(q_wall, q_wall_h); copyto!(is_solid, is_solid_h)
    fill!(uw_x, zero(T)); fill!(uw_y, zero(T))
    copyto!(f_in, f_in_h); fill!(f_out, zero(T))
    fill!(ρ, one(T)); fill!(ux, zero(T)); fill!(uy, zero(T))
    copyto!(u_profile, u_prof_h)

    Fx_sum = 0.0; Fy_sum = 0.0; n_avg = 0

    D = 2 * Float64(radius)
    check_n = isnothing(check_every) ? max(1, round(Int, Nx / u_ref)) : Int(check_every)
    check_n >= 1 || throw(ArgumentError("check_every must be >= 1"))
    cd_history = Tuple{Int,Float64}[]
    last_Fx = NaN; last_Fy = NaN
    converged = false
    stop_step = max_steps

    for step in 1:max_steps
        fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                 q_wall, uw_x, uw_y, Nx, Ny, T(ν))
        # Zou-He velocity inlet at i=1 and extrapolation outlet at i=Nx (both j=1..Ny).
        rebuild_inlet_outlet_libb_2d!(f_out, f_in, u_profile, ρ_out, ν, Nx, Ny)

        if step > max_steps - avg_window
            drag = compute_drag_libb_mei_2d(f_out, q_wall, uw_x, uw_y, Nx, Ny)
            Fx_sum += drag.Fx; Fy_sum += drag.Fy
            n_avg  += 1
        end

        if stop === :cd_change && step % check_n == 0
            drag = compute_drag_libb_mei_2d(f_out, q_wall, uw_x, uw_y, Nx, Ny)
            last_Fx = drag.Fx; last_Fy = drag.Fy
            push!(cd_history, (step, 2.0 * drag.Fx / (u_ref^2 * D)))
            if length(cd_history) >= 2
                cd_new = cd_history[end][2]; cd_old = cd_history[end - 1][2]
                if isfinite(cd_new) && abs(cd_new - cd_old) <= tol * abs(cd_new)
                    converged = true
                    stop_step = step
                    f_in, f_out = f_out, f_in
                    break
                end
            end
        end

        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    Fx_avg = n_avg > 0 ? Fx_sum / n_avg : last_Fx
    Fy_avg = n_avg > 0 ? Fy_sum / n_avg : last_Fy
    Cd     = 2.0 * Fx_avg / (u_ref^2 * D)
    Cl     = 2.0 * Fy_avg / (u_ref^2 * D)

    return (; ρ = Array(ρ), ux = Array(ux), uy = Array(uy),
             Cd, Cl, Fx = Fx_avg, Fy = Fy_avg,
             q_wall = Array(q_wall), is_solid = Array(is_solid),
             u_ref, D, inlet, converged, stop_step, cd_history)
end
