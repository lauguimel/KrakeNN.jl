module KrakeNNLinearSolveExt

# LinearSolve.jl backend for the factorize-once linear-solve seam (issue #8).
# Loaded by `using LinearSolve` (weakdep). Implements the `_ls_factorize` /
# `_ls_solve!` ext-indirection stubs declared in
# src/solve/linear_solve_frontend.jl (KrakeNNOptimExt pattern — no method
# overwriting), which back the `PoissonLinearSolve` tag of the seam.

using KrakeNN
using LinearSolve
using SparseArrays

import KrakeNN: _ls_factorize, _ls_solve!

# Default-algorithm choice mirrors the built-in CPU seam: Cholesky for SPD,
# LU otherwise. An explicit `tag.alg` always wins.
function _ls_default_alg(spd::Bool)
    return spd ? CHOLMODFactorization() : UMFPACKFactorization()
end

"""
    _ls_factorize(tag::PoissonLinearSolve, A; spd=true, pin_k0=0) -> LinearSolveCache

LinearSolve.jl method of the factorize-once seam ([`KrakeNN.lin_factorize`](@ref)
routes here for a `PoissonLinearSolve` tag). Pins reference DOF `pin_k0` (value
0) at factorize time via `KrakeNN.pin_reference_dof` — same pinning contract as
the CPU seam — then caches a `LinearSolve.init` problem in the
`LinearSolveCache.factor` slot. The numeric factorization happens on the first
`solve!` and is reused for every later RHS (`alias_A=true`: the cache holds the
assembled operator, never copies it).
"""
function _ls_factorize(tag::KrakeNN.PoissonLinearSolve,
                       A::SparseMatrixCSC{Float64,Int};
                       spd::Bool = true, pin_k0::Int = 0)
    A_unpinned = A
    if pin_k0 > 0
        A_used, _ = KrakeNN.pin_reference_dof(A, zeros(Float64, size(A, 1)),
                                             pin_k0, 0.0)
    else
        A_used = A
    end

    alg = tag.alg === nothing ? _ls_default_alg(spd) : tag.alg
    prob = LinearProblem(A_used, zeros(Float64, size(A_used, 1)))
    linsolve = init(prob, alg; tag.kwargs...)

    return KrakeNN.LinearSolveCache(tag, linsolve, A_used, A_unpinned,
                                   Int(pin_k0), spd)
end

"""
    _ls_solve!(tag::PoissonLinearSolve, cache, b) -> Vector{Float64}

Per-RHS solve of the LinearSolve.jl seam backend ([`KrakeNN.lin_solve!`](@ref)
routes here). Reuses the cached `LinearSolve.init` problem — NEVER
re-factorizes for direct algorithms — and applies the consistent RHS pinning
from `cache.A_unpinned` when `cache.pin_k0 > 0`, exactly like the CPU seam.
Returns a fresh `Vector{Float64}` (the internal solution buffer is copied so
successive solves do not alias).
"""
function _ls_solve!(::KrakeNN.PoissonLinearSolve, cache::KrakeNN.LinearSolveCache,
                    b::AbstractVector)
    bvec = Vector{Float64}(b)
    if cache.pin_k0 > 0
        _, bvec = KrakeNN.pin_reference_dof(cache.A_unpinned, bvec,
                                           cache.pin_k0, 0.0)
    end
    cache.factor.b = bvec
    sol = solve!(cache.factor)
    return copy(sol.u)
end

end # module KrakeNNLinearSolveExt
