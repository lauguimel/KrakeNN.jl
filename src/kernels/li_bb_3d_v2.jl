# =====================================================================
# TRT + LI-BB 3D kernel (D3Q19), Ginzburg-exact via DSL.
#
# Same recipe as li_bb_2d_v2.jl, ported to D3Q19:
#   PullHalfwayBB_3D → SolidInert_3D | ApplyLiBBPrePhase_3D →
#                      Moments_3D → CollideTRTDirect_3D →
#                      WriteMoments_3D
#
# Reuses `_libb_branch` from li_bb_2d.jl (the Bouzidi formula is
# stencil-independent).
# =====================================================================

const _TRT_LIBB_V2_SPEC_3D = LBMSpec(
    PullHalfwayBB_3D(), SolidInert_3D(),
    ApplyLiBBPrePhase_3D(),
    Moments_3D(), CollideTRTDirect_3D(),
    WriteMoments_3D();
    stencil = :D3Q19,
)

"""
    fused_trt_libb_v2_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                q_wall, uw_x, uw_y, uw_z,
                                Nx, Ny, Nz, ν; Λ=3/16)

Single D3Q19 step: pull-stream + SolidInert on solid cells, or
ApplyLiBBPrePhase + TRT collision on fluid cells. Ginzburg-exact for
halfway-BB + TRT Λ = 3/16 on Couette; handles arbitrary q_w ∈ (0, 1]
via the full Bouzidi formula.
"""
function fused_trt_libb_v2_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                      q_wall, uw_link_x, uw_link_y, uw_link_z,
                                      Nx, Ny, Nz, ν; Λ::Real=3/16)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    s_plus, s_minus = trt_rates(ν; Λ=Λ)
    kernel! = build_lbm_kernel(backend, _TRT_LIBB_V2_SPEC_3D)
    kernel!(f_out, ρ, ux, uy, uz, f_in, is_solid,
            q_wall, uw_link_x, uw_link_y, uw_link_z,
            Nx, Ny, Nz, ET(s_plus), ET(s_minus);
            ndrange=(Nx, Ny, Nz))
end

# TRT + LI-BB 3D step with a per-cell Guo body-force field fused into the
# collision (3D analogue of `_TRT_LIBB_V2_GUO_FIELD_SPEC`). Same pull /
# solid / LI-BB pre-phase as `_TRT_LIBB_V2_SPEC_3D`; only the collision
# brick carries the Guo force. Used by the VE sphere driver to inject the
# polymer body force F_poly = ∇·τ_p at the solvent rate (replacing the
# standalone re-relaxed Hermite source).
const _TRT_LIBB_V2_GUO_FIELD_SPEC_3D = LBMSpec(
    PullHalfwayBB_3D(), SolidInert_3D(),
    ApplyLiBBPrePhase_3D(),
    Moments_3D(), CollideTRTDirectGuoField_3D(),
    WriteMoments_3D();
    stencil = :D3Q19,
)

# Variant of the Guo-field spec with PERIODIC z-wrap in the pull-stream brick
# (x/y faces stay no-slip halfway-BB). Selected when the caller declares
# periodic_z=true — e.g. the planar-extensional driver whose FVFD polymer side
# is fully z-periodic, so the LBM solvent side must agree on z. The collision /
# solid / cut-link bricks are identical; only PullHalfwayBB_3D's z-treatment
# differs (distinct brick TYPE → distinct cached kernel, no runtime branch).
const _TRT_LIBB_V2_GUO_FIELD_SPEC_Z_PERIODIC_3D = LBMSpec(
    PullHalfwayBB_3D{true}(), SolidInert_3D(),
    ApplyLiBBPrePhase_3D(),
    Moments_3D(), CollideTRTDirectGuoField_3D(),
    WriteMoments_3D();
    stencil = :D3Q19,
)

"""
    fused_trt_libb_v2_guo_field_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                          q_wall, uw_x, uw_y, uw_z,
                                          Fx_field, Fy_field, Fz_field,
                                          Nx, Ny, Nz, ν; Λ=3/16)

D3Q19 LI-BB V2 solvent step with a per-cell Guo force field. Identical
pull-stream / solid handling / cut-link pre-phase to
`fused_trt_libb_v2_step_3d!`, with the collision brick replaced by the
TRT+Guo-field brick. The force is fused into the collision and consumed
EXACTLY ONCE (no post-collision re-relaxation). 3D analogue of the 2D
`fused_trt_libb_v2_guo_field_step!` used by the validated VE cylinder.
"""
function fused_trt_libb_v2_guo_field_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                                q_wall, uw_link_x, uw_link_y, uw_link_z,
                                                Fx_field, Fy_field, Fz_field,
                                                Nx, Ny, Nz, ν; Λ::Real=3/16,
                                                periodic_z::Bool=false)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    s_plus, s_minus = trt_rates(ν; Λ=Λ)
    # periodic_z=false (DEFAULT) keeps the bit-identical no-slip-z spec used by
    # the VE sphere driver; =true selects the z-wrap variant for the extensional
    # driver. Val-dispatch on a compile-time flag (M1 scheme) → no runtime branch.
    spec = periodic_z ? _TRT_LIBB_V2_GUO_FIELD_SPEC_Z_PERIODIC_3D :
                        _TRT_LIBB_V2_GUO_FIELD_SPEC_3D
    kernel! = build_lbm_kernel(backend, spec)
    kernel!(f_out, ρ, ux, uy, uz, f_in, is_solid,
            q_wall, uw_link_x, uw_link_y, uw_link_z,
            Nx, Ny, Nz, ET(s_plus), ET(s_minus),
            Fx_field, Fy_field, Fz_field;
            ndrange=(Nx, Ny, Nz))
end

"""
    precompute_q_wall_sphere_3d(Nx, Ny, Nz, cx, cy, cz, R; FT=Float64)
        -> (q_wall, is_solid)

Analytical sub-cell `q_w` for a sphere of radius `R` centred at
`(cx, cy, cz)` on an `Nx×Ny×Nz` D3Q19 grid with unit lattice spacing.
For every fluid node whose link `q` crosses the sphere, solves the
quadratic `|x_f + t · c_q − c|² = R²` for the smallest positive
`t ∈ (0, 1]` and stores that as `q_wall[i,j,k,q]`.
"""
function precompute_q_wall_sphere_3d(Nx::Int, Ny::Int, Nz::Int,
                                      cx::Real, cy::Real, cz::Real,
                                      R::Real;
                                      FT::Type{<:AbstractFloat}=Float64)
    cxT, cyT, czT, RT = FT(cx), FT(cy), FT(cz), FT(R)
    R² = RT * RT
    is_solid = zeros(Bool, Nx, Ny, Nz)
    q_wall = zeros(FT, Nx, Ny, Nz, 19)
    cxs = velocities_x(D3Q19())
    cys = velocities_y(D3Q19())
    czs = velocities_z(D3Q19())

    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        xf = FT(i - 1); yf = FT(j - 1); zf = FT(k - 1)
        dx_f = xf - cxT; dy_f = yf - cyT; dz_f = zf - czT
        if dx_f * dx_f + dy_f * dy_f + dz_f * dz_f ≤ R²
            is_solid[i, j, k] = true
            continue
        end
        for q in 2:19
            cqx = FT(cxs[q]); cqy = FT(cys[q]); cqz = FT(czs[q])
            xn = xf + cqx; yn = yf + cqy; zn = zf + cqz
            dx_n = xn - cxT; dy_n = yn - cyT; dz_n = zn - czT
            if dx_n * dx_n + dy_n * dy_n + dz_n * dz_n > R²
                continue
            end
            a = cqx * cqx + cqy * cqy + cqz * cqz
            b = FT(2) * (dx_f * cqx + dy_f * cqy + dz_f * cqz)
            c = dx_f * dx_f + dy_f * dy_f + dz_f * dz_f - R²
            disc = b * b - FT(4) * a * c
            disc < zero(FT) && continue
            sd = sqrt(disc)
            t1 = (-b - sd) / (FT(2) * a)
            t2 = (-b + sd) / (FT(2) * a)
            t = t1 > zero(FT) ? t1 : t2
            if t > zero(FT) && t ≤ one(FT)
                q_wall[i, j, k, q] = t
            end
        end
    end
    return q_wall, is_solid
end

"""
    compute_drag_libb_3d(f_post, q_wall, Nx, Ny, Nz)
        -> (Fx, Fy, Fz)

D3Q19 MEA drag/lift for a stationary LI-BB wall. Same halfway-BB
convention as the 2D version (`F = 2·c_q·f_q(post-coll)`).
"""
function compute_drag_libb_3d(f_post, q_wall, Nx::Int, Ny::Int, Nz::Int)
    backend = KernelAbstractions.get_backend(f_post)
    if backend isa KernelAbstractions.CPU
        return _compute_drag_libb_3d_host(f_post, q_wall, Nx, Ny, Nz)
    else
        return _compute_drag_libb_3d_gpu_cached(f_post, q_wall, Nx, Ny, Nz)
    end
end

function _compute_drag_libb_3d_host(f_post, q_wall, Nx::Int, Ny::Int, Nz::Int)
    f = Array(f_post)
    qw = Array(q_wall)
    cxs = velocities_x(D3Q19())
    cys = velocities_y(D3Q19())
    czs = velocities_z(D3Q19())
    Fx = 0.0; Fy = 0.0; Fz = 0.0
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        for q in 2:19
            if qw[i, j, k, q] > 0
                fv = Float64(f[i, j, k, q])
                Fx += 2.0 * Float64(cxs[q]) * fv
                Fy += 2.0 * Float64(cys[q]) * fv
                Fz += 2.0 * Float64(czs[q]) * fv
            end
        end
    end
    return (Fx = Fx, Fy = Fy, Fz = Fz)
end

# Cache: per-q_wall device cut-link list + scratch for the 3D drag.
const _DRAG_GPU_3D_CACHE = IdDict{Any, Any}()

function _compute_drag_libb_3d_gpu_cached(f_post, q_wall, Nx::Int, Ny::Int, Nz::Int)
    cache = get!(_DRAG_GPU_3D_CACHE, q_wall) do
        T = eltype(f_post)
        backend = KernelAbstractions.get_backend(f_post)
        qw_h = Array(q_wall)
        links = build_cut_link_list_3d(qw_h; backend=backend)
        Fx_l = KernelAbstractions.allocate(backend, T, links.Nlinks)
        Fy_l = KernelAbstractions.allocate(backend, T, links.Nlinks)
        Fz_l = KernelAbstractions.allocate(backend, T, links.Nlinks)
        Fx_buf = zeros(T, links.Nlinks)
        Fy_buf = zeros(T, links.Nlinks)
        Fz_buf = zeros(T, links.Nlinks)
        (links, Fx_l, Fy_l, Fz_l, Fx_buf, Fy_buf, Fz_buf)
    end
    links, Fx_l, Fy_l, Fz_l, Fx_buf, Fy_buf, Fz_buf = cache
    compute_drag_libb_3d_gpu!(Fx_l, Fy_l, Fz_l, links, f_post, Nx, Ny, Nz)
    copyto!(Fx_buf, Fx_l)
    copyto!(Fy_buf, Fy_l)
    copyto!(Fz_buf, Fz_l)
    return (Fx = Float64(sum(Fx_buf)), Fy = Float64(sum(Fy_buf)),
            Fz = Float64(sum(Fz_buf)))
end

"""
    rebuild_inlet_outlet_libb_3d!(f_out, f_in, u_in, ρ_out, ν, Nx, Ny, Nz)

Legacy wrapper for the hardcoded 3D (ZouHe-velocity west + ZouHe-pressure
east) rebuild. Prefer the modular `apply_bc_rebuild_3d!(f_out, f_in,
BCSpec3D(west=ZouHeVelocity(fill(u_in, Ny)), east=ZouHePressure(ρ_out)),
ν, Nx, Ny, Nz)` in new code.
"""
function rebuild_inlet_outlet_libb_3d!(f_out, f_in, u_in::Real, ρ_out::Real,
                                         ν::Real, Nx::Int, Ny::Int, Nz::Int)
    T = eltype(f_out)
    # Build a device profile matching backend for the ZouHeVelocity BC.
    backend = KernelAbstractions.get_backend(f_out)
    prof = KernelAbstractions.allocate(backend, T, Ny)
    fill!(prof, T(u_in))
    bc = BCSpec3D(; west = ZouHeVelocity(prof),
                    east = ZouHePressure(T(ρ_out)))
    apply_bc_rebuild_3d!(f_out, f_in, bc, ν, Nx, Ny, Nz)
    return
    return nothing
end

"""
    run_sphere_libb_3d(; Nx=120, Ny=60, Nz=60, radius=8, u_in=0.04,
                        ν=…, max_steps=20000, avg_window=5000,
                        inlet=:uniform, ρ_out=1.0)

3D sphere in an x-flow. Halfway-BB on y/z walls (kernel fallback),
Zou-He velocity inlet at i=1 + Zou-He pressure outlet at i=Nx
reconstructed pre-collision via the modular BC system
(`apply_bc_rebuild_3d!`).

- `inlet=:uniform`: inlet velocity u = (u_in, 0, 0) constant over y, z.
- `inlet=:parabolic`: inlet u(y, z) = u_in · 16 · y·(H-y)·z·(D-z) /
  (H·D)² (parabolic in both y and z). `u_in` is the centerline maximum.
- `inlet=:parabolic_y`: parabolic in y only (rectangular duct,
  infinite in z direction).

Drag integrated via `compute_drag_libb_3d` over `avg_window` steps.
Reference: free-stream sphere Re=20, Cd ≈ 2.6 (Clift et al. 1978;
confinement-adjusted upward for ducted setups).
"""
function run_sphere_libb_3d(; Nx::Int=120, Ny::Int=60, Nz::Int=60,
                              cx::Union{Nothing,Real}=nothing,
                              cy::Union{Nothing,Real}=nothing,
                              cz::Union{Nothing,Real}=nothing,
                              radius::Real=8,
                              u_in::Real=0.04, ν::Real=0.04,
                              max_steps::Int=20_000,
                              avg_window::Int=5_000,
                              inlet::Symbol=:uniform,
                              ρ_out::Real=1.0,
                              backend=KernelAbstractions.CPU(),
                              T::Type{<:AbstractFloat}=Float64)
    cx = isnothing(cx) ? Nx ÷ 4 : Float64(cx)
    cy = isnothing(cy) ? Ny ÷ 2 : Float64(cy)
    cz = isnothing(cz) ? Nz ÷ 2 : Float64(cz)

    # Precompute on CPU (analytic geometry, not kernel-launched)
    q_wall_h, is_solid_h = precompute_q_wall_sphere_3d(Nx, Ny, Nz, cx, cy, cz,
                                                        radius; FT=T)
    # Inlet profile matrix (Ny, Nz): normal-into-domain velocity per cell.
    u_profile_h = zeros(T, Ny, Nz)
    if inlet === :uniform
        fill!(u_profile_h, T(u_in))
    elseif inlet === :parabolic
        # Parabolic in both y and z, peak = u_in at (Ny/2, Nz/2).
        # Normalised so max(u) = u_in.
        Hy = T(Ny - 1); Hz = T(Nz - 1)
        for k in 1:Nz, j in 1:Ny
            yy = T(j - 1); zz = T(k - 1)
            u_profile_h[j, k] = T(16) * T(u_in) *
                                yy * (Hy - yy) * zz * (Hz - zz) /
                                (Hy^2 * Hz^2)
        end
    elseif inlet === :parabolic_y
        Hy = T(Ny - 1)
        for j in 1:Ny
            yy = T(j - 1)
            val = T(4) * T(u_in) * yy * (Hy - yy) / Hy^2
            for k in 1:Nz
                u_profile_h[j, k] = val
            end
        end
    else
        error("unknown inlet $(inlet); expected :uniform|:parabolic|:parabolic_y")
    end

    f_in_h = zeros(T, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 1:19
        f_in_h[i, j, k, q] = Kraken.equilibrium(D3Q19(), one(T),
                                                 u_profile_h[j, k],
                                                 zero(T), zero(T), q)
    end

    # Allocate on the target backend
    q_wall   = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny, Nz)
    uw_x     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    uw_y     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    uw_z     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    f_in     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    f_out    = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    ρ        = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    ux       = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    uy       = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    uz       = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    u_profile = KernelAbstractions.allocate(backend, T, Ny, Nz)

    copyto!(q_wall, q_wall_h)
    copyto!(is_solid, is_solid_h)
    fill!(uw_x, zero(T)); fill!(uw_y, zero(T)); fill!(uw_z, zero(T))
    copyto!(f_in, f_in_h)
    fill!(ρ, one(T));  fill!(ux, zero(T))
    fill!(uy, zero(T)); fill!(uz, zero(T))
    copyto!(u_profile, u_profile_h)

    bcspec = BCSpec3D(; west = ZouHeVelocity(u_profile),
                        east = ZouHePressure(T(ρ_out)))

    Fx_sum = 0.0; Fy_sum = 0.0; Fz_sum = 0.0; n_avg = 0

    for step in 1:max_steps
        fused_trt_libb_v2_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                    q_wall, uw_x, uw_y, uw_z,
                                    Nx, Ny, Nz, T(ν))
        # Pre-collision Zou-He rebuild (modular BC dispatch)
        apply_bc_rebuild_3d!(f_out, f_in, bcspec, ν, Nx, Ny, Nz)

        if step > max_steps - avg_window
            drag = compute_drag_libb_3d(f_out, q_wall, Nx, Ny, Nz)
            Fx_sum += drag.Fx; Fy_sum += drag.Fy; Fz_sum += drag.Fz
            n_avg += 1
        end

        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    Fx_avg = Fx_sum / n_avg
    Fy_avg = Fy_sum / n_avg
    Fz_avg = Fz_sum / n_avg
    D = 2 * Float64(radius)
    A = π * Float64(radius)^2        # cross-section area
    # Cd reference velocity: for parabolic inlets the convention is
    # u_mean — for a doubly parabolic duct, u_mean = (4/9)·u_max; for
    # y-only parabolic, u_mean = (2/3)·u_max.
    u_ref = inlet === :parabolic   ? (4/9) * Float64(u_in) :
            inlet === :parabolic_y ? (2/3) * Float64(u_in) :
                                       Float64(u_in)
    Cd = 2.0 * Fx_avg / (u_ref^2 * A)

    return (; ρ = Array(ρ), ux = Array(ux), uy = Array(uy), uz = Array(uz),
             Cd, Fx = Fx_avg, Fy = Fy_avg, Fz = Fz_avg,
             q_wall = Array(q_wall), is_solid = Array(is_solid),
             u_ref, D, A)
end

# GNF variants retain the existing LI-BB pre-phase and Guo field conventions.
const _TRT_GNF_LIBB_GUO_FIELD_SPEC_3D = LBMSpec(
    PullHalfwayBB_3D(), SolidInert_3D(),
    ApplyLiBBPrePhase_3D(),
    Moments_3D(), CollideTRTGNFGuoField_3D(),
    WriteMoments_3D();
    stencil = :D3Q19,
)

const _TRT_GNF_LIBB_GUO_FIELD_SPEC_Z_PERIODIC_3D = LBMSpec(
    PullHalfwayBB_3D{true}(), SolidInert_3D(),
    ApplyLiBBPrePhase_3D(),
    Moments_3D(), CollideTRTGNFGuoField_3D(),
    WriteMoments_3D();
    stencil = :D3Q19,
)

"""
    fused_trt_gnf_libb_guo_field_step_3d!(f_out, f_in, rho, ux, uy, uz, is_solid,
        q_wall, uw_x, uw_y, uw_z, Fx, Fy, Fz, sp_field, sm_field,
        Nx, Ny, Nz, rheology; Λ=3/16, periodic_z=false)

Advance D3Q19 populations with LI-BB, local GNF TRT relaxation, and Guo forcing.
The written velocity includes `F/(2rho)`. Strain uses the Guo-corrected stress
moment and the lagged even relaxation time `1/sp_field`; the fluid-cell rate
fields are then updated in place. Initialize them to the chosen starting
viscosity's TRT rates before the first call. Solid-cell rates remain untouched.

Supported models are isothermal Newtonian, PowerLaw, CarreauYasuda, and Cross;
parameters are converted to `eltype(f_in)` on the host. `periodic_z=true` wraps
z links while retaining the existing x/y halfway-BB fallback.
"""
function fused_trt_gnf_libb_guo_field_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                             q_wall, uw_link_x, uw_link_y, uw_link_z,
                                             Fx_field, Fy_field, Fz_field,
                                             sp_field, sm_field, Nx, Ny, Nz, rheology;
                                             Λ::Real=3/16, periodic_z::Bool=false)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    model = _gnf3d_rheology(ET, rheology)
    spec = periodic_z ? _TRT_GNF_LIBB_GUO_FIELD_SPEC_Z_PERIODIC_3D :
                        _TRT_GNF_LIBB_GUO_FIELD_SPEC_3D
    kernel! = build_lbm_kernel(backend, spec)
    # Canonical arguments first; collision-only arguments follow declaration order.
    kernel!(f_out, ρ, ux, uy, uz, f_in, is_solid,
            q_wall, uw_link_x, uw_link_y, uw_link_z, Nx, Ny, Nz,
            Fx_field, Fy_field, Fz_field, model, sp_field, sm_field, ET(Λ);
            ndrange=(Nx, Ny, Nz))
end
