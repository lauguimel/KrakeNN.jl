# --- Steady GNF driver: periodic in x, body force, walls in y, solid mask ---
#
# Runs `gnf_step!` until the estimated distance to the fixed point (default) or
# the per-step change of the velocity field falls below `tol`, and returns the
# fixed point (f*, τ*) with diagnostics.
# Everything is in lattice units: the caller converts its physical case.

"""
    gnf_rest_state(backend, T, Nx, Ny, tau0) -> (f, tau)

Populations at rest (ρ = 1, u = 0) on every node and a uniform relaxation time.
"""
function gnf_rest_state(backend, ::Type{T}, Nx::Int, Ny::Int, tau0) where {T}
    w = gnf_feq_cell(one(T), zero(T), zero(T))
    f_cpu = Array{T}(undef, Nx, Ny, 9)
    for q in 1:9
        f_cpu[:, :, q] .= w[q]
    end
    f = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    copyto!(f, f_cpu)
    tau = KernelAbstractions.allocate(backend, T, Nx, Ny)
    fill!(tau, T(tau0))
    return f, tau
end

_gnf_field(backend, ::Type{T}, v::Number, Nx, Ny) where {T} =
    (a = KernelAbstractions.allocate(backend, T, Nx, Ny); fill!(a, T(v)); a)
function _gnf_field(backend, ::Type{T}, v::AbstractMatrix, Nx, Ny) where {T}
    size(v) == (Nx, Ny) || throw(DimensionMismatch("force field must be $(Nx)×$(Ny), got $(size(v))"))
    a = KernelAbstractions.allocate(backend, T, Nx, Ny)
    copyto!(a, T.(Array(v)))
    return a
end

# Least-squares slope of log(y) against x (y > 0).
function _gnf_log_slope(x, y)
    lx = Float64.(x); ly = log.(Float64.(y))
    mx = sum(lx) / length(lx); my = sum(ly) / length(ly)
    return sum((lx .- mx) .* (ly .- my)) / sum((lx .- mx) .^ 2)
end

# Contraction per step ρ and distance estimate r2 ρ²/(1 − ρ²) from the
# two-step changes r2 of the last `nfit` checks of `history` (entries
# (step, r_u, r_f, r_tau, r_u2)): for e_k = ρ e_(k−1), u^k − u^(k−2) =
# e_k (1 − ρ⁻²). (NaN, Inf) with fewer than 3 usable checks; distance Inf when
# ρ ≥ 1 (no contraction measured); distance 0 when nothing changed (r2 = 0).
function _gnf_distance_estimate(history, nfit::Int)
    isempty(history) && return (NaN, Inf)
    r_last = history[end][5]
    r_last == 0 && return (0.0, 0.0)
    win = history[max(1, end - nfit + 1):end]
    pts = [(h[1], h[5]) for h in win if h[5] > 0]
    length(pts) < 3 && return (NaN, Inf)
    ρ = exp(_gnf_log_slope(first.(pts), last.(pts)))
    (isfinite(ρ) && ρ < 1) || return (ρ, Inf)
    return (ρ, r_last * ρ^2 / (1 - ρ^2))
end

"""
    run_gnf_periodic_steady(; is_solid, closure, Fx, Fy=0, backend=CPU(), T=Float64,
                              stop=:distance, tol=1e-10, fit_checks=20,
                              max_steps=1_000_000, check_every=1000,
                              f_init=nothing, tau_init=nothing, fused=false,
                              max_wall=Inf, verbose=false) -> NamedTuple

Steady GNF flow on an `Nx × Ny` D2Q9 lattice, periodic in x, half-way
bounce-back walls at y = 1/2 and y = Ny + 1/2 (node (i, j) at (i − 1/2, j − 1/2)),
arbitrary solid mask `is_solid` (full bounce-back swap), Guo body force
`(Fx, Fy)` (numbers or `Nx × Ny` fields), viscosity `ν = gnf_viscosity(closure, I1)`
with `I1 = S:S` from the non-equilibrium stress and the τ of the previous step.
All quantities in lattice units. The closure is first converted to `T`
(`gnf_convert_closure`), so that split and fused paths compute in the same type.

Starts from `(f_init, tau_init)` if given (arrays on any backend, post-collision
populations), else from rest with `τ = 3ν(I1 = 0) + 1/2`. Every `check_every`
steps it measures the relative changes over the fluid nodes

    r_u  = ‖u^k − u^(k−1)‖₂ / ‖u^k‖₂   (one step),  r_f, r_τ (same for f, τ)
    r_u2 = ‖u^k − u^(k−2)‖₂ / ‖u^k‖₂   (two steps)

# Stopping criteria (`stop`)

- `:distance` (default): stop when the estimated relative distance of `u` to the
  fixed point, `dist_est = r_u2 ρ²/(1 − ρ²)`, is below `tol`. `ρ` is the
  contraction per step, `exp` of the least-squares slope of `log r_u2` against
  the step over the last `fit_checks` checks. For a linear contraction of
  factor ρ per step, e_k = u^k − u* satisfies u^k − u^(k−2) = e_k (1 − ρ⁻²),
  hence the estimate (≈ r_u ρ/(1 − ρ) while the iteration contracts). With
  ρ ≈ 0.9999 (obstacle channel, m = 64) the distance is ≈ 10⁴ r_u. No stop before 3
  checks, nor while ρ ≥ 1 (`dist_est = Inf`).

  Why two steps: in Float64 the one-step change stops decreasing at a floor
  set by a neutral period-2 mode of the lattice (uy alternating in time and in
  y, fed by round-off: amplitude 1e-13 on the 4×64 Poiseuille of test T2, where
  it holds r_u at 1e-11 while r_u2 reaches 4e-15; measured 2026-10-19). That
  mode is not part of the fixed point and cancels in u^k − u^(k−2). On the
  obstacle channel the one-step floor is r_u ≈ 1–3e-14, which caps the one-step
  estimate at ≈ 2e-10.
- `:step_change`: stop when `r_u < tol`. Cheaper to reach,
  but it bounds the change of one step, not the error: with ρ ≈ 0.9999,
  `r_u < 1e-10` leaves a distance of about 1e-6.

In both cases the run also stops when a value is not finite, at `max_steps`, or
after `max_wall` seconds (then `converged = false`).

Returns `(; f, tau, I1, gamma_dot, nu, rho, ux, uy, steps, converged, stop, r_u,
r_u2, r_f, r_tau, contraction, dist_est, history, wall_time, tau_min, tau_max, rho_min,
rho_max, u_max, finite)`. `f`, `tau`, `I1`, `gamma_dot = √(2 I1)`, `nu` stay on
`backend` (post-collision state and the values of the last step), `rho`, `ux`,
`uy` are host arrays, `history` is a vector of `(step, r_u, r_f, r_tau, r_u2)` tuples,
and `contraction`, `dist_est` are the last estimates (computed for both
criteria).
"""
function run_gnf_periodic_steady(; is_solid::AbstractMatrix{Bool}, closure, Fx, Fy=0,
                                 backend=KernelAbstractions.CPU(), T::Type=Float64,
                                 stop::Symbol=:distance, tol=1e-10, fit_checks::Integer=20,
                                 max_steps::Integer=1_000_000,
                                 check_every::Integer=1000,
                                 f_init=nothing, tau_init=nothing, fused::Bool=false,
                                 max_wall=Inf, verbose::Bool=false)
    stop in (:distance, :step_change) ||
        throw(ArgumentError("stop must be :distance or :step_change, got :$stop"))
    check_every >= 2 || throw(ArgumentError("check_every must be ≥ 2 (two-step change)"))
    closure = gnf_convert_closure(T, closure)
    Nx, Ny = size(is_solid)
    solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    copyto!(solid, Array(is_solid))
    fluidw = KernelAbstractions.allocate(backend, T, Nx, Ny)
    copyto!(fluidw, T.(.!Array(is_solid)))
    Fxa = _gnf_field(backend, T, Fx, Nx, Ny)
    Fya = _gnf_field(backend, T, Fy, Nx, Ny)
    tau_rest = 3 * gnf_viscosity(closure, zero(T)) + T(0.5)

    if f_init === nothing
        tau0 = tau_init === nothing ? tau_rest : tau_init
        tau0 isa Number || throw(ArgumentError("tau_init must be a number when f_init is not given"))
        f_in, tau = gnf_rest_state(backend, T, Nx, Ny, tau0)
    else
        size(f_init) == (Nx, Ny, 9) || throw(DimensionMismatch("f_init must be $(Nx)×$(Ny)×9"))
        f_in = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
        copyto!(f_in, T.(Array(f_init)))
        tau = KernelAbstractions.allocate(backend, T, Nx, Ny)
        if tau_init isa Number
            fill!(tau, T(tau_init))
        elseif tau_init === nothing
            fill!(tau, tau_rest)
        else
            copyto!(tau, T.(Array(tau_init)))
        end
    end
    f_out = similar(f_in); copyto!(f_out, f_in)
    f_prev = similar(f_in)
    f_prev2 = similar(f_in)
    tau_prev = similar(tau)
    I1 = KernelAbstractions.zeros(backend, T, Nx, Ny)
    nu = KernelAbstractions.zeros(backend, T, Nx, Ny)
    rho = KernelAbstractions.zeros(backend, T, Nx, Ny)
    ux = KernelAbstractions.zeros(backend, T, Nx, Ny)
    uy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    uxp = similar(ux); uyp = similar(uy); rhop = similar(rho)
    uxp2 = similar(ux); uyp2 = similar(uy)
    fluid3 = reshape(fluidw, Nx, Ny, 1)

    history = Tuple{Int,Float64,Float64,Float64,Float64}[]
    r_u = Inf; r_f = Inf; r_tau = Inf; r_u2 = Inf
    contraction = NaN; dist_est = Inf
    converged = false; finite = true
    steps = 0
    KernelAbstractions.synchronize(backend)
    t0 = time()
    for step in 1:max_steps
        check = step % check_every == 0
        (step + 1) % check_every == 0 && copyto!(f_prev2, f_in)   # state k − 2
        if check
            copyto!(f_prev, f_in)
            copyto!(tau_prev, tau)
        end
        gnf_step!(f_out, f_in, tau, I1, nu, solid, closure, Fxa, Fya; fused=fused)
        f_in, f_out = f_out, f_in
        steps = step
        if check
            gnf_macroscopic_post_2d!(rho, ux, uy, f_in, solid, Fxa, Fya)
            gnf_macroscopic_post_2d!(rhop, uxp, uyp, f_prev, solid, Fxa, Fya)
            gnf_macroscopic_post_2d!(rhop, uxp2, uyp2, f_prev2, solid, Fxa, Fya)
            du2 = sum(fluidw .* ((ux .- uxp) .^ 2 .+ (uy .- uyp) .^ 2))
            dd2 = sum(fluidw .* ((ux .- uxp2) .^ 2 .+ (uy .- uyp2) .^ 2))
            u2 = sum(fluidw .* (ux .^ 2 .+ uy .^ 2))
            df2 = sum(fluid3 .* (f_in .- f_prev) .^ 2)
            f2 = sum(fluid3 .* f_in .^ 2)
            dt2 = sum(fluidw .* (tau .- tau_prev) .^ 2)
            t2 = sum(fluidw .* tau .^ 2)
            r_u = sqrt(Float64(du2) / Float64(u2))
            r_f = sqrt(Float64(df2) / Float64(f2))
            r_tau = sqrt(Float64(dt2) / Float64(t2))
            r_u2 = sqrt(Float64(dd2) / Float64(u2))
            push!(history, (step, r_u, r_f, r_tau, r_u2))
            if !(isfinite(r_u) && isfinite(r_f) && isfinite(r_tau) && isfinite(r_u2))
                finite = false
                break
            end
            contraction, dist_est = _gnf_distance_estimate(history, Int(fit_checks))
            if verbose && (length(history) % 20 == 1)
                @info "run_gnf_periodic_steady" step r_u r_u2 r_f r_tau contraction dist_est
            end
            if (stop === :distance ? dist_est : r_u) < tol
                converged = true
                break
            end
            time() - t0 > max_wall && break
        end
    end
    KernelAbstractions.synchronize(backend)
    wall = time() - t0

    gnf_macroscopic_post_2d!(rho, ux, uy, f_in, solid, Fxa, Fya)
    gamma_dot = gnf_shear_rate.(I1)
    rho_h = Array(rho); ux_h = Array(ux); uy_h = Array(uy); tau_h = Array(tau)
    fl = .!Array(is_solid)
    umag = sqrt.(ux_h .^ 2 .+ uy_h .^ 2)
    return (; f=f_in, tau=tau, I1=I1, gamma_dot=gamma_dot, nu=nu,
            rho=rho_h, ux=ux_h, uy=uy_h, steps=steps, converged=converged, stop=stop,
            r_u=r_u, r_u2=r_u2, r_f=r_f, r_tau=r_tau, contraction=contraction, dist_est=dist_est,
            history=history, wall_time=wall,
            tau_min=minimum(tau_h[fl]), tau_max=maximum(tau_h[fl]),
            rho_min=minimum(rho_h[fl]), rho_max=maximum(rho_h[fl]),
            u_max=maximum(umag[fl]), finite=finite)
end
