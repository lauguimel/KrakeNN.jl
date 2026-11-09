# Steady GNF adjoint (part 4/5): observations and parameter gradients.
# Split from src/ad/gnf_adjoint.jl, which holds the overview of the method.


# ============================================================
# Observation, adjoint solve, parameter gradient
# ============================================================

"""
    GNFVelocityObs(node, weight, comp, target, scale)

Velocity observations: value m is `scale · Σ_k weight[k, m] · u_comp[node[k, m]]`
(`comp[m]` = 1 for ux, 2 for uy; `node` = linear node index, 0 = unused slot),
with u the post-collision lattice velocity (0 at solid nodes). Objective
J = ½ Σ_m (value_m − target_m)².
"""
struct GNFVelocityObs
    node::Matrix{Int}
    weight::Matrix{Float64}
    comp::Vector{Int}
    target::Vector{Float64}
    scale::Float64
end

"""
    gnf_obs_values(obs, ux, uy) -> values

Evaluate the weighted velocity observations defined by `obs::GNFVelocityObs`
from the post-collision lattice velocity fields `ux` and `uy` (Nx × Ny host
matrices). Each value is `obs.scale` times the weighted sum of the component
selected by `obs.comp` at the nodes in `obs.node`; node index 0 is ignored.
Returns a host `Vector{Float64}` with one entry per observation. Computed on
the host; the input velocity fields must support host indexing.
"""
function gnf_obs_values(obs::GNFVelocityObs, ux::AbstractMatrix, uy::AbstractMatrix)
    M = length(obs.target)
    v = zeros(M)
    for mi in 1:M
        s = 0.0
        for kk in axes(obs.node, 1)
            nd = obs.node[kk, mi]
            nd == 0 && continue
            s += obs.weight[kk, mi] * (obs.comp[mi] == 1 ? ux[nd] : uy[nd])
        end
        v[mi] = obs.scale * s
    end
    return v
end

"""
    gnf_obs_objective(obs, ux, uy) -> J

J = ½ Σ (value − target)² from host velocity fields (post-collision, lattice units).
"""
gnf_obs_objective(obs::GNFVelocityObs, ux, uy) =
    0.5 * sum(abs2, gnf_obs_values(obs, ux, uy) .- obs.target)

"""
    gnf_obs_grad_u(obs, lin) -> (J, gux, guy)

Objective J = ½ Σ (value − target)² at the linearisation state and its
cotangent on the lattice velocity, gux = ∂J/∂u_x and guy = ∂J/∂u_y (host
Nx × Ny, zero where no observation reads).
"""
function gnf_obs_grad_u(obs::GNFVelocityObs, lin::GNFLinearisation)
    Nx, Ny = lin.Nx, lin.Ny
    v = gnf_obs_values(obs, lin.ux, lin.uy)
    res = v .- obs.target
    J = 0.5 * sum(abs2, res)
    gx = zeros(Nx, Ny); gy = zeros(Nx, Ny)
    for mi in eachindex(res)
        for kk in axes(obs.node, 1)
            nd = obs.node[kk, mi]
            nd == 0 && continue
            gval = res[mi] * obs.scale * obs.weight[kk, mi]
            obs.comp[mi] == 1 ? (gx[nd] += gval) : (gy[nd] += gval)
        end
    end
    return J, gx, gy
end

"""
    gnf_velocity_rhs(lin, gux, guy) -> b

Adjoint right-hand side b = (∂u/∂w)ᵀ g_u for a cotangent (gux, guy) on the
post-collision lattice velocity of the state (as `gnf_post_moments`; host
vector of length 10 N, nonzero on fluid populations only):
∂u_x/∂f_q = (c_xq − u_x)/ρ. Values of g_u at solid nodes are ignored (u ≡ 0 there).
"""
function gnf_velocity_rhs(lin::GNFLinearisation, gux::AbstractMatrix, guy::AbstractMatrix)
    Nx, Ny = lin.Nx, lin.Ny; N = Nx * Ny
    (size(gux) == (Nx, Ny) && size(guy) == (Nx, Ny)) ||
        throw(DimensionMismatch("g_u components must be $(Nx)×$(Ny)"))
    gx = Float64.(Array(gux)); gy = Float64.(Array(guy))
    b = zeros(10N)
    for c in 1:N
        lin.is_solid[c] && continue
        (gx[c] == 0 && gy[c] == 0) && continue
        ρ = lin.rho[c]; ux = lin.ux[c]; uy = lin.uy[c]
        for q in 1:9
            b[c + N * (q - 1)] = (gx[c] * (_GNF_CX[q] - ux) + gy[c] * (_GNF_CY[q] - uy)) / ρ
        end
    end
    return b
end

"""
    gnf_obs_rhs(obs, lin) -> (J, b)

Objective at the linearisation state and its gradient b = ∂J/∂w:
`gnf_obs_grad_u` then `gnf_velocity_rhs`.
"""
function gnf_obs_rhs(obs::GNFVelocityObs, lin::GNFLinearisation)
    J, gx, gy = gnf_obs_grad_u(obs, lin)
    return J, gnf_velocity_rhs(lin, gx, gy)
end

"""
    gnf_adjoint_solve(op, b; tol=1e-6, restart=1000, maxiter=100_000, orth=:mgs2,
                      deflate=0, λ0=nothing, max_wall=Inf, ritz=false, verbose=false)
        -> (λ, η, info)

Solves the gauge-bordered adjoint system for the right-hand side `b` = ∂J/∂w
(device vector, length n; projected on the active space here). `λ0`: warm
start (projected on the active space and on ĉ⊥). Returns λ (length n, λ ⊥ ĉ up
to the solver tolerance), the border unknown η and the `gnf_gmres!` info.
"""
function gnf_adjoint_solve(op::AbstractGNFStepOperator, b; tol::Real=1e-6, restart::Integer=1000,
                           maxiter::Integer=100_000, orth::Symbol=:mgs2, deflate::Integer=0,
                           λ0=nothing, max_wall::Real=Inf, ritz::Bool=false,
                           verbose::Bool=false, snapshots=())
    n = op.n
    bb = similar(op.chat, n + 1)
    view(bb, 1:n) .= op.active .* b
    fill!(view(bb, n+1:n+1), 0.0)
    x = similar(bb); fill!(x, 0.0)
    if λ0 !== nothing
        xn = view(x, 1:n)
        xn .= op.active .* λ0
        s = dot(op.chat, xn)
        xn .-= s .* op.chat
    end
    A = GNFBorderedAdjoint(op)
    info = gnf_gmres!(x, A, bb; tol, restart, maxiter, orth, deflate, max_wall, ritz, verbose,
                      snapshots)
    λ = x[1:n]
    η = Array(x[n+1:n+1])[1]
    return λ, η, info
end

"""
    gnf_parameter_gradient(lin, λ, θ, closure_of) -> (grad, s)

dJ/dθ = Σ_c s_c ∂ν_c/∂θ with s_c = (∂(f', τ')_c/∂ν)ᵀ λ_c (ν column of the
blocks) and ∂ν_c/∂θ = ForwardDiff gradient of `gnf_viscosity(closure_of(θ), I1_c)`.
`closure_of(θ)` returns the closure in lattice units; `λ` any backend.
"""
function gnf_parameter_gradient(lin::GNFLinearisation, λ, θ, closure_of)
    s = gnf_nu_sensitivity(lin, λ)
    return _gnf_theta_gradient(lin, s, θ, closure_of), s
end

"""
    gnf_nu_sensitivity(lin, λ) -> s

Per-cell cotangent of ν_lat, s_c = (∂(f', τ')_c/∂ν)ᵀ λ_c (ν column of the
blocks), host Nx × Ny, zero at solid nodes; `λ` any backend.
"""
function gnf_nu_sensitivity(lin::GNFLinearisation, λ)
    λh = Float64.(Array(λ))
    Nx, Ny = lin.Nx, lin.Ny
    s = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        lin.is_solid[i, j] && continue
        acc = 0.0
        for r in 1:10
            acc += lin.B[i, j, r, 11] * λh[_spxwy_lin(i, j, r, Nx, Ny)]
        end
        s[i, j] = acc
    end
    return s
end

# Σ_c s_c ∂ν_c/∂θ, ∂ν_c/∂θ by ForwardDiff through `gnf_viscosity(closure_of(θ), I1_c)`.
function _gnf_theta_gradient(lin::GNFLinearisation, s, θ, closure_of)
    Nx, Ny = lin.Nx, lin.Ny
    P = length(θ)
    θs = SVector{P,Float64}(Tuple(θ))
    grad = zeros(P)
    for j in 1:Ny, i in 1:Nx
        lin.is_solid[i, j] && continue
        I1c = lin.I1[i, j]
        gθ = ForwardDiff.gradient(t -> gnf_viscosity(closure_of(t), I1c), θs)
        for p in 1:P
            grad[p] += s[i, j] * gθ[p]
        end
    end
    return grad
end

_gnf_split_gu(gu::Tuple{<:AbstractMatrix,<:AbstractMatrix}) = gu
_gnf_split_gu(gu::NamedTuple{(:ux, :uy)}) = (gu.ux, gu.uy)
function _gnf_split_gu(gu::AbstractArray{<:Real,3})
    size(gu, 3) == 2 || throw(DimensionMismatch("g_u must be Nx×Ny×2, got $(size(gu))"))
    g = Array(gu)
    return (g[:, :, 1], g[:, :, 2])
end

"""
    gnf_steady_adjoint(lin, op, gu; m=nothing, nu0_lat=nothing, solver=:kstep, kwargs...) -> (s, stats)
    gnf_steady_adjoint(f, tau, gu; is_solid, closure, Fx, Fy=0, backend=CPU(),
                       T=Float64, m=nothing, nu0_lat=nothing, kwargs...) -> (s, stats)

Generic steady adjoint. For any objective L of the
lattice velocity at the fixed point w* = (f, tau) of the GNF step with
`closure` (lattice units), `gu` = ∂L/∂u_lat: an Nx × Ny × 2 array (component 1
= ∂L/∂u_x, 2 = ∂L/∂u_y), a tuple `(gux, guy)` or a named tuple `(; ux, uy)`
of Nx × Ny arrays, u = post-collision lattice velocity (`gnf_post_moments`,
0 at solid nodes, where `gu` is ignored). Builds b = (∂u/∂w)ᵀ g_u
(`gnf_velocity_rhs`), solves the adjoint system with `solver` = `:kstep`
(default: GMRES on the K-step system, `gnf_adjoint_kstep_solve`, defaults
K=1001 (must be odd), restart=50, orth=:cgs2), `:gmres` (gauge-bordered GMRES,
`gnf_adjoint_solve`) or `:fixed_point` (`gnf_adjoint_fixed_point`), `kwargs`
passed on (tol, restart, orth, deflate, max_wall, K, …; `λ0` = initial adjoint
state for a warm start, a device vector of length `op.n` such as `stats.λ` of
the previous call),
and returns

- `s`: per-cell cotangent of ν_lat (Nx × Ny host, zero at solid nodes,
  `gnf_nu_sensitivity`), so that dL/dθ = Σ_c s_c ∂ν_lat,c/∂θ
  = Σ_c s_c ν0,lat ∂η*_c/∂θ, plus any explicit θ dependence of L;
- `stats`: `(; converged, iters, relres, t_rhs, t_solve, t_s, t_lin, t_copy,
  λ, eta, info, I1_lat, I1star, lin, op)`; `I1star` = I1_lat m⁴/ν0,lat² when
  `m` and `nu0_lat` are given (length unit L = m lattice nodes, kinematic
  viscosity unit ν0 = nu0_lat, time unit L²/ν0), else `nothing`. The second
  form linearises first (`gnf_linearise`, `GNFStepOperator` on `backend`).
"""
function gnf_steady_adjoint(lin::GNFLinearisation, op::GNFStepOperator, gu;
                            m=nothing, nu0_lat=nothing, solver::Symbol=:kstep,
                            t_lin=0.0, t_copy=0.0, kwargs...)
    t0 = time()
    gux, guy = _gnf_split_gu(gu)
    bh = gnf_velocity_rhs(lin, gux, guy)
    b = similar(op.chat, op.n); copyto!(b, bh); _gnf_sync(b)
    t1 = time()
    λ, η, info = _gnf_solve_adjoint(op, b, solver; kwargs...)
    t2 = time()
    s = gnf_nu_sensitivity(lin, λ)
    t3 = time()
    I1star = (m === nothing || nu0_lat === nothing) ? nothing : lin.I1 .* (m^4 / nu0_lat^2)
    stats = (; converged=info.converged, iters=info.iters, relres=info.relres,
             t_rhs=t1 - t0, t_solve=t2 - t1, t_s=t3 - t2, t_lin, t_copy,
             λ, eta=η, info, I1_lat=lin.I1, I1star, lin, op)
    return s, stats
end

function gnf_steady_adjoint(f::AbstractArray, tau::AbstractArray, gu; is_solid, closure, Fx, Fy=0,
                            backend=KernelAbstractions.CPU(), T::Type=Float64, kwargs...)
    t0 = time()
    lin = gnf_linearise(f, tau, is_solid, closure, Fx, Fy)
    t1 = time()
    op = GNFStepOperator(lin; backend, T)
    _gnf_sync(op.chat)
    t2 = time()
    return gnf_steady_adjoint(lin, op, gu; t_lin=t1 - t0, t_copy=t2 - t1, kwargs...)
end

"""
    gnf_steady_gradient(f, tau, θ, obs; is_solid, closure_of, Fx, Fy=0,
                        backend=CPU(), T=Float64, kwargs...) -> NamedTuple

Value and parameter gradient of J = ½ Σ (u_obs − target)² at the fixed point
w* = (f, tau) of the GNF step with closure `closure_of(θ)` (lattice units):
thin wrapper of `gnf_steady_adjoint` with g_u = ∂J/∂u (`gnf_obs_grad_u`),
then dJ/dθ = Σ_c s_c ∂ν_c/∂θ; `kwargs` go to `gnf_steady_adjoint` (default
`solver=:kstep`; `λ0` for a warm start). Returns `(; J, grad, λ, η, info, lin,
op, s, t_lin, t_copy, t_solve, t_grad, t_total)`.
"""
function gnf_steady_gradient(f, tau, θ, obs::GNFVelocityObs; is_solid, closure_of, Fx, Fy=0,
                             backend=KernelAbstractions.CPU(), T::Type=Float64, kwargs...)
    t0 = time()
    lin = gnf_linearise(f, tau, is_solid, closure_of(θ), Fx, Fy)
    t1 = time()
    op = GNFStepOperator(lin; backend, T)
    J, gux, guy = gnf_obs_grad_u(obs, lin)
    t2 = time()
    s, st = gnf_steady_adjoint(lin, op, (gux, guy); kwargs...)
    t3 = time()
    grad = _gnf_theta_gradient(lin, s, θ, closure_of)
    t4 = time()
    return (; J, grad, λ=st.λ, η=st.eta, info=st.info, lin, op, s, t_lin=t1 - t0,
            t_copy=t2 - t1 + st.t_rhs, t_solve=st.t_solve + st.t_s, t_grad=t4 - t3,
            t_total=t4 - t0)
end
