# Steady GNF adjoint (part 5/5): diagnostics and adjoint solvers.
# Split from src/ad/gnf_adjoint.jl, which holds the overview of the method.


# ============================================================
# Diagnostics: power iteration, truncated adjoint series, real step
# ============================================================

"""
    gnf_power_iteration(op; iters=50, seed=1) -> (; mu, ratio, history)

Power iteration on Q P Aᵀ P (Q = I − ĉĉᵀ removes the mass eigenvalue 1):
`mu` = last Rayleigh quotient, `ratio` = last ‖y‖/‖x‖, `history` of both.
"""
function gnf_power_iteration(op::AbstractGNFStepOperator; iters::Integer=50, seed::Integer=1)
    n = op.n
    xh = _gnf_pseudo_randn(seed, n)
    x = similar(op.chat); copyto!(x, xh)
    x .*= op.active
    x .-= dot(op.chat, x) .* op.chat
    x ./= norm(x)
    y = similar(x)
    hist = Tuple{Float64,Float64}[]
    mu = NaN; ratio = NaN
    for it in 1:iters
        gnf_vjp!(y, op, x)
        y .-= dot(op.chat, y) .* op.chat
        mu = dot(x, y); ratio = norm(y)
        push!(hist, (mu, ratio))
        x .= y ./ ratio
    end
    return (; mu, ratio, history=hist)
end

# Deterministic pseudo-normal numbers (xorshift, sum of four uniforms) to seed
# the power iteration without a Random dependency in src.
function _gnf_pseudo_randn(seed::Integer, n::Integer)
    s = UInt64(seed) * 0x9E3779B97F4A7C15 + 0x0000000001234567
    out = zeros(n)
    for i in 1:n
        acc = 0.0
        for _ in 1:4
            s ⊻= s << 13; s ⊻= s >> 7; s ⊻= s << 17
            acc += (s >> 11) * (1.0 / 9007199254740992.0)
        end
        out[i] = (acc - 2.0) * sqrt(3.0)
    end
    return out
end

"""
    gnf_adjoint_series(op, b, Ns) -> Dict(N => λ_N)

Truncated Neumann series of the adjoint, λ_N = Q Σ_{n<N} (P Aᵀ P)ⁿ P b with
Q = I − ĉĉᵀ (computed as Σ (Q P Aᵀ P)ⁿ Q P b), for every N in `Ns`.
"""
function gnf_adjoint_series(op::AbstractGNFStepOperator, b, Ns)
    Nmax = maximum(Ns)
    x = similar(op.chat); x .= op.active .* b
    x .-= dot(op.chat, x) .* op.chat
    acc = copy(x)
    y = similar(x)
    out = Dict{Int,typeof(x)}()
    1 in Ns && (out[1] = copy(acc))
    for it in 1:Nmax-1
        gnf_vjp!(y, op, x)
        y .-= dot(op.chat, y) .* op.chat
        x, y = y, x
        acc .+= x
        (it + 1) in Ns && (out[it+1] = copy(acc))
    end
    return out
end

# ============================================================
# Fixed-point and K-step adjoint solvers
# ============================================================

# y = M^K x with M = Q P Aᵀ P (adjoint iteration operator, Q = I − ĉĉᵀ), or with
# `transposed = true` y = (Mᵀ)^K x, Mᵀ = P A P Q. Buffers u, v distinct from x, y.
function _gnf_power!(y, op::AbstractGNFStepOperator, x, K::Integer, u, v; transposed::Bool=false)
    copyto!(u, x)
    for _ in 1:K
        if transposed
            u .-= dot(op.chat, u) .* op.chat
            gnf_jvp!(v, op, u)
        else
            gnf_vjp!(v, op, u)
            v .-= dot(op.chat, v) .* op.chat
        end
        u, v = v, u
    end
    copyto!(y, u)
    return y
end

"""
    gnf_adjoint_power!(y, op, x, K) -> y      y = M^K x,     M  = Q P Aᵀ P
    gnf_tangent_power!(y, op, x, K) -> y      y = (Mᵀ)^K x,  Mᵀ = P A P Q

K products of the adjoint iteration operator M (Q = I − ĉĉᵀ removes the mass
mode) or of its transpose; ⟨M^K l, v⟩ = ⟨l, (Mᵀ)^K v⟩ is the K-step form of T4c.
"""
gnf_adjoint_power!(y, op::AbstractGNFStepOperator, x, K::Integer) =
    _gnf_power!(y, op, x, K, similar(x), similar(x))

"""
    gnf_tangent_power!(y, op, x, K) -> y

Apply K powers of the transposed adjoint iteration operator:
`y = (Mᵀ)^K x`, with `Mᵀ = P A P Q` and `Q = I − ĉĉᵀ` removing the mass mode.
`op::AbstractGNFStepOperator` supplies the linearised step, active mask P and gauge ĉ;
`x` and `y` are flat vectors of length `op.n` on its backend, and `K` is the
nonnegative number of products. Allocates two work vectors and returns the
updated `y`. Products run on the host or device backend of `op`.
"""
gnf_tangent_power!(y, op::AbstractGNFStepOperator, x, K::Integer) =
    _gnf_power!(y, op, x, K, similar(x), similar(x); transposed=true)

"""
    GNFKStepAdjoint(op, K)

Callable `A(y, x)`: y = x − M^K x (K VJPs, M = Q P Aᵀ P), the operator of the
K-step adjoint system of `gnf_adjoint_kstep_solve`.
"""
struct GNFKStepAdjoint{TO<:AbstractGNFStepOperator, TV<:AbstractVector{Float64}}
    op::TO
    K::Int
    u::TV
    v::TV
    w::TV
end
GNFKStepAdjoint(op::AbstractGNFStepOperator, K::Integer) =
    GNFKStepAdjoint(op, Int(K), similar(op.chat), similar(op.chat), similar(op.chat))

function (A::GNFKStepAdjoint)(y, x)
    _gnf_power!(A.w, A.op, x, A.K, A.u, A.v)
    y .= x .- A.w
    return y
end

# b' = Q P b and the projected warm start (zero if λ0 === nothing).
function _gnf_projected_rhs_start(op::AbstractGNFStepOperator, b, λ0)
    bp = similar(op.chat); bp .= op.active .* b
    bp .-= dot(op.chat, bp) .* op.chat
    λ = similar(bp)
    if λ0 === nothing
        fill!(λ, 0.0)
    else
        λ .= op.active .* λ0
        λ .-= dot(op.chat, λ) .* op.chat
    end
    return bp, λ
end

# Explicit residual ‖b' − (I − M) λ‖ / ‖b'‖ (one VJP).
function _gnf_true_relres(op::AbstractGNFStepOperator, bp, λ)
    y = similar(bp)
    gnf_vjp!(y, op, λ)
    y .-= dot(op.chat, y) .* op.chat
    y .= bp .- λ .+ y
    return norm(y) / max(norm(bp), 1e-300)
end

"""
    gnf_adjoint_fixed_point(op, b; tol=1e-6, check_every=1000, fit_checks=20,
                            maxiter=10^7, λ0=nothing, max_wall=Inf, snapshots=())
        -> (λ, η, info)

Adjoint by the plain fixed-point iteration λ ← M λ + b' (M = Q P Aᵀ P,
b' = Q P b), i.e. the truncated series continued to convergence, with the stop
rule of the forward driver (`run_gnf_periodic_steady`, `stop = :distance`):
every `check_every` iterations r2 = ‖λ_k − λ_(k−2)‖/‖λ_k‖ (the period-2 mode
μ ≈ −1 cancels in it), contraction ρ = exp of the slope of log r2 over the last
`fit_checks` checks. Converged when the distance estimate r2 ρ²/(1 − ρ²) < `tol`
and the explicit residual ‖b' − (I − M)λ̄‖/‖b'‖ of the mean λ̄ of the last two
iterates is ≤ `tol` (as `gnf_adjoint_kstep_solve`); the explicit residual (one
VJP, counted in `t_total`) is evaluated only at checks where the distance
estimate is below `tol`, and the iteration goes on while it is above `tol`.
Returns λ̄ (removes the period-2 oscillation), η = 0 and info
`(; converged, iters, relres, dist_est, contraction, ...)` in the layout of
`gnf_gmres!`, with `relres` the explicit residual of the returned λ,
`relres_true` the explicit residuals evaluated at checks and `relres_iter` the
distance estimate at each check. Stops only at a check (or after `maxiter` or
`max_wall`, unconverged). `snapshots`: distance tolerances above `tol` at which
the mean iterate is kept as `(; tol, iters, t, x)`.
"""
function gnf_adjoint_fixed_point(op::AbstractGNFStepOperator, b; tol::Real=1e-6, check_every::Integer=1000,
                                 fit_checks::Integer=20, maxiter::Integer=10^7, λ0=nothing,
                                 max_wall::Real=Inf, snapshots=())
    check_every >= 2 || throw(ArgumentError("check_every must be ≥ 2 (two-step change)"))
    t0 = time(); t_diag = 0.0
    bp, λ = _gnf_projected_rhs_start(op, b, λ0)
    y = similar(bp); λp = similar(bp); λp2 = similar(bp); xm = similar(bp)
    history = Tuple{Int,Float64,Float64,Float64,Float64}[]
    relres_iter = Float64[]; t_iter = Float64[]; relres_true = Tuple{Int,Float64}[]
    snap_tols = sort(collect(Float64, snapshots); rev=true)
    snaps = NamedTuple[]; next_snap = 1
    ρ = NaN; dist = Inf; converged = false; k = 0; relres = NaN
    while true
        k += 1
        (k + 1) % check_every == 0 && copyto!(λp2, λ)      # λ_(k−2) at the next check
        check = k % check_every == 0
        check && copyto!(λp, λ)                            # λ_(k−1)
        gnf_vjp!(y, op, λ)
        s = dot(op.chat, y)
        λ .= y .- s .* op.chat .+ bp
        check || continue
        nl = max(norm(λ), 1e-300)
        y .= λ .- λp; r1 = norm(y) / nl
        y .= λ .- λp2; r2 = norm(y) / nl
        push!(history, (k, r1, NaN, NaN, r2))
        ρ, dist = _gnf_distance_estimate(history, Int(fit_checks))
        push!(relres_iter, dist); push!(t_iter, time() - t0 - t_diag)
        while next_snap <= length(snap_tols) && dist <= snap_tols[next_snap]
            td = time()
            xs = similar(λ); xs .= 0.5 .* (λ .+ λp); _gnf_sync(xs)
            push!(snaps, (; tol=snap_tols[next_snap], iters=k, t=td - t0 - t_diag, x=xs))
            next_snap += 1
            t_diag += time() - td
        end
        isfinite(r2) || break
        if dist < tol
            # explicit one-step residual of the mean iterate (one VJP)
            xm .= 0.5 .* (λ .+ λp)
            relres = _gnf_true_relres(op, bp, xm)
            push!(relres_true, (k, relres))
            relres <= tol && (converged = true; break)
        end
        (k >= maxiter || time() - t0 - t_diag > max_wall) && break
    end
    λ .= 0.5 .* (λ .+ λp)
    _gnf_sync(λ)
    t_total = time() - t0 - t_diag
    if !converged
        td = time()
        relres = _gnf_true_relres(op, bp, λ)
        t_diag += time() - td
    end
    info = (; converged, iters=k, cycles=0, relres, relres_cycles=[relres], relres_true, relres_iter, t_iter,
            t_total, t_op=t_total, t_orth=0.0, t_diag, n_apply=k, restart=0, orth=:none,
            deflate=0, ritz=ComplexF64[], snaps, dist_est=dist, contraction=ρ, history,
            check_every=Int(check_every))
    return λ, 0.0, info
end

"""
    gnf_adjoint_kstep_solve(op, b; K=1001, tol=1e-6, restart=50, orth=:cgs2,
                            maxiter=10_000, λ0=nothing, max_wall=Inf, snapshots=())
        -> (λ, η, info)

Adjoint by GMRES on the K-step system (I − M^K) λ = Σ_(j<K) M^j b'
(M = Q P Aᵀ P, b' = Q P b): the system (I − M)λ = b' left-multiplied by the
K-term truncated series Σ_(j<K) M^j, so it has the same solution. One GMRES
iteration costs K VJPs (`GNFKStepAdjoint`), which amortises the
orthogonalisation. K must be odd: the period-2 mode μ ≈ −1 gives 1 − μ^K ≈ 0
for even K. Convergence: explicit residual ‖b' − (I − M)λ‖/‖b'‖ ≤ `tol`
(`info.relres`, as for `gnf_adjoint_solve`); GMRES first runs to `tol` on the
K-step residual, then, while the explicit one is above `tol`, continues from
its iterate with the K-step tolerance scaled down by the measured ratio
(`refine` rounds at most; the explicit residual, one VJP, counts in
`t_total`). `snapshots` apply to the K-step residual of the first round;
`info.t_rhs` = cost of the right-hand side (K − 1 VJPs, included in `t_total`
and in the snapshot times). Returns η = 0.
"""
function gnf_adjoint_kstep_solve(op::AbstractGNFStepOperator, b; K::Integer=1001, tol::Real=1e-6,
                                 restart::Integer=50, orth::Symbol=:cgs2, maxiter::Integer=10_000,
                                 λ0=nothing, max_wall::Real=Inf, snapshots=(), verbose::Bool=false,
                                 refine::Integer=3)
    isodd(K) || throw(ArgumentError("K must be odd (period-2 mode μ ≈ −1), got $K"))
    t0 = time()
    bp, x = _gnf_projected_rhs_start(op, b, λ0)
    rhs = copy(bp); u = copy(bp); v = similar(bp)
    for _ in 1:K-1
        gnf_vjp!(v, op, u)
        v .-= dot(op.chat, v) .* op.chat
        u, v = v, u
        rhs .+= u
    end
    _gnf_sync(rhs)
    t_rhs = time() - t0
    A = GNFKStepAdjoint(op, K)
    # GMRES on the K-step residual, then the explicit residual of (I − M)λ = b';
    # if it is above `tol`, continue from x with the K-step tolerance scaled down
    # by the measured ratio (at most `refine` extra rounds)
    tol_k = Float64(tol); rounds = 0; t_true = 0.0
    gi = nothing; iters = 0; n_apply = 0; t_gm = 0.0; t_op = 0.0; t_orth = 0.0; t_gdiag = 0.0
    snaps = NamedTuple[]; relres = Inf; relres_hist = Float64[]
    while true
        gr = gnf_gmres!(x, A, rhs; tol=tol_k, restart, maxiter=maxiter - iters, orth,
                        max_wall=max_wall - t_rhs - t_gm - t_true, verbose,
                        snapshots=rounds == 0 ? snapshots : ())
        for s in gr.snaps
            push!(snaps, (; tol=s.tol, iters=s.iters + iters, t=s.t + t_rhs + t_gm + t_true, x=s.x))
        end
        iters += gr.iters; n_apply += gr.n_apply; t_gm += gr.t_total; t_op += gr.t_op
        t_orth += gr.t_orth; t_gdiag += gr.t_diag
        gi = gr
        td = time()
        relres = _gnf_true_relres(op, bp, x)
        t_true += time() - td
        push!(relres_hist, relres)
        (relres <= tol || !gr.converged || rounds >= refine || iters >= maxiter ||
         t_rhs + t_gm + t_true > max_wall) && break
        tol_k *= 0.5 * tol / relres
        rounds += 1
    end
    converged = gi.converged && relres <= tol
    info = merge(gi, (; converged, iters, n_apply, t_op, t_orth, relres, relres_kstep=gi.relres,
                      relres_true_rounds=relres_hist, tol_kstep=tol_k, rounds,
                      t_total=t_rhs + t_gm + t_true, t_rhs, t_diag=t_gdiag, K=Int(K),
                      n_vjp=(K - 1) + K * n_apply + length(relres_hist), snaps))
    return x, 0.0, info
end

# Solver selection for `gnf_steady_adjoint` (`solver` keyword).
function _gnf_solve_adjoint(op::AbstractGNFStepOperator, b, solver::Symbol; kwargs...)
    solver === :gmres && return gnf_adjoint_solve(op, b; kwargs...)
    solver === :fixed_point && return gnf_adjoint_fixed_point(op, b; kwargs...)
    solver === :kstep && return gnf_adjoint_kstep_solve(op, b; kwargs...)
    throw(ArgumentError("solver must be :gmres, :fixed_point or :kstep, got :$solver"))
end

"""
    gnf_step_state(w, is_solid, closure, Fx, Fy) -> w'

One real GNF step (`gnf_step!`, split path) applied to a flat state vector, on
the backend of `w`; `is_solid`, `Fx`, `Fy` device arrays (Nx, Ny). The closure
should hold numbers of `eltype(w)`.
"""
function gnf_step_state(w, is_solid, closure, Fx, Fy)
    Nx, Ny = size(is_solid)
    f, tau = gnf_unpack_state(w, Nx, Ny)
    fo = similar(f)
    I1 = similar(tau); nu = similar(tau)
    gnf_step!(fo, f, tau, I1, nu, is_solid, closure, Fx, Fy)
    return gnf_pack_state(fo, tau)
end
