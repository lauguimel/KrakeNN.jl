# LinearSolve.jl / CUDSS front-ends for the factorize-once linear-solve seam.
#
# WHY THIS EXISTS
# ---------------
# The seam in linear_solve.jl has two built-in backends: CPU CHOLMOD/UMFPACK
# (here) and the manual-load cuDSS file for GPU jobs. Issue #8 adds two
# PACKAGE-EXTENSION backends behind [weakdeps] so `using KrakeNN` stays free of
# any solver-backend dependency (ADR rule; precedent: KrakeNNOptimExt):
#
#   * `PoissonLinearSolve` — a new `LinearSolveBackend` tag that routes
#     `lin_factorize` / `lin_solve!` through LinearSolve.jl. Methods live in
#     `ext/KrakeNNLinearSolveExt.jl`, loaded by `using LinearSolve`.
#   * `CUDABackendTag` host-matrix entry + CuSparse methods via cuDSS. Methods
#     live in `ext/KrakeNNCUDSSExt.jl`, loaded by `using CUDSS` (CUDA is already
#     a strong dep, listed as an extra trigger).
#
# This file is CPU-only and dependency-free: it defines the tag, the
# ext-indirection stubs (the KrakeNNOptimExt `_fit_lbfgs` pattern: the stub
# errors with a load hint, the ext adds the typed method — no method
# overwriting), and the assembled-direct driver `solve_poisson_direct` that is
# the drop-in alternative to `solve_poisson_mg` on the SAME discretization
# (the poisson.jl assemblers; MG parity receipt: test/analytical/poisson_mg_mms.jl).

using SparseArrays

# Assemblers + the seam live in poisson.jl (which tail-includes linear_solve.jl).
# Pull them in if standalone-included.
if !isdefined(@__MODULE__, :assemble_poisson_dirichlet)
    include(joinpath(@__DIR__, "poisson.jl"))
end

"""
    PoissonLinearSolve(; alg=nothing, kwargs...) <: LinearSolveBackend

Backend tag routing the factorize-once seam ([`lin_factorize`](@ref) /
[`lin_solve!`](@ref)) through LinearSolve.jl. Requires `using LinearSolve`
(weakdep — activates `ext/KrakeNNLinearSolveExt.jl`); without it the seam
methods raise a documented load-hint error.

  * `alg`     a LinearSolve.jl algorithm instance (e.g. `KLUFactorization()`),
              or `nothing` (default) to pick `CHOLMODFactorization()` when
              `spd=true` and `UMFPACKFactorization()` otherwise — mirroring the
              built-in CPU seam semantics.
  * `kwargs`  forwarded to `LinearSolve.init` (e.g. `abstol`, `reltol` for
              iterative algorithms).

The ext caches a `LinearSolve.init` problem in the `LinearSolveCache.factor`
slot, so repeated [`lin_solve!`](@ref) calls reuse the factorization —
same factorize-once contract as `CPUBackendTag()`. Usable anywhere a
`LinearSolveBackend` tag is accepted, including [`solve_poisson_direct`](@ref).
Receipt: `test/analytical/poisson_linearsolve_mms.jl`.
"""
struct PoissonLinearSolve{A, K<:NamedTuple} <: LinearSolveBackend
    alg::A
    kwargs::K
end
PoissonLinearSolve(; alg = nothing, kwargs...) =
    PoissonLinearSolve(alg, NamedTuple(kwargs))

const _LINEARSOLVE_LOAD_ERROR =
    "Load LinearSolve to enable the PoissonLinearSolve front-end: `using " * "LinearSolve`"
const _CUDSS_LOAD_ERROR =
    "Load CUDA and CUDSS to enable the GPU direct backend: `using " * "CUDA, CUDSS`"

# Ext-indirection stubs (KrakeNNOptimExt pattern): ext/KrakeNNLinearSolveExt.jl
# adds the typed methods; these fallbacks raise the load hint.
_ls_factorize(args...; kwargs...) = error(_LINEARSOLVE_LOAD_ERROR)
_ls_solve!(args...; kwargs...) = error(_LINEARSOLVE_LOAD_ERROR)
# ext/KrakeNNCUDSSExt.jl adds the host-matrix CUDA method of this stub.
_cudss_factorize(args...; kwargs...) = error(_CUDSS_LOAD_ERROR)

# Seam methods for the LinearSolve tag: pure delegation to the ext stubs.
function lin_factorize(tag::PoissonLinearSolve, A::SparseMatrixCSC{Float64,Int};
                       spd::Bool = true, pin_k0::Int = 0)
    return _ls_factorize(tag, A; spd = spd, pin_k0 = pin_k0)
end
lin_solve!(tag::PoissonLinearSolve, cache::LinearSolveCache, b::AbstractVector) =
    _ls_solve!(tag, cache, b)

# Host-matrix entry for the CUDA tag (pin on CPU, upload, cuDSS-factorize).
# The CuSparseMatrixCSR methods live entirely in ext/KrakeNNCUDSSExt.jl.
function lin_factorize(tag::CUDABackendTag, A::SparseMatrixCSC{Float64,Int};
                       spd::Bool = true, pin_k0::Int = 0)
    return _cudss_factorize(tag, A; spd = spd, pin_k0 = pin_k0)
end

# RHS input normalisation: `solve_poisson_direct` mirrors `solve_poisson_mg`
# in accepting either a Function (x,y) or an N x N cell-centred array.
_poisson_rhs_function(f::Function, ::Int) = f
function _poisson_rhs_function(f::AbstractMatrix{<:Real}, N::Int)
    size(f) == (N, N) || throw(ArgumentError("RHS array must have size (N, N)"))
    # assemble_* samples ONLY exact cell centres ((i-0.5)/N, (j-0.5)/N), so the
    # inverse index map is exact.
    return (x, y) -> Float64(f[round(Int, x * N + 0.5), round(Int, y * N + 0.5)])
end

"""
    solve_poisson_direct(f, N; bc=:dirichlet, method=PoissonLinearSolve(), k0=1)
        -> Matrix{Float64}

Solve `-∇²u = f` on the unit square (`N x N` cell-centred grid) with an
ASSEMBLED sparse direct/Krylov solve — the drop-in alternative to the
matrix-free [`solve_poisson_mg`](@ref) on the SAME discretization (the
5-point operators of `assemble_poisson_dirichlet` /
`assemble_poisson_neumann_unpinned`; MG↔assembled parity receipt:
`test/analytical/poisson_mg_mms.jl`).

  * `f`       `Function (x,y)` sampled at cell centres, or an `N x N` array.
  * `bc`      `:dirichlet` (homogeneous ghost-0) or `:neumann` (all-Neumann,
              singular — regularised by pinning DOF `k0` to 0, so the returned
              field is the zero-anchored gauge; subtract the mean before
              comparing to an exact solution).
  * `method`  any `LinearSolveBackend` tag: [`PoissonLinearSolve`](@ref)
              (default; requires `using LinearSolve`), `CPUBackendTag()`
              (built-in CHOLMOD), or `CUDABackendTag()` (cuDSS on GPU;
              requires `using CUDA, CUDSS`).
  * `k0`      reference DOF pinned for `bc=:neumann` (ignored for Dirichlet).

Routes through the factorize-once seam and consumes the cache immediately
(single-RHS convenience, like [`solve_poisson`](@ref)); loops that reuse the
operator should call `lin_factorize` once and `lin_solve!` per RHS. The result
is always gathered to a host `Matrix{Float64}` (layout `u[i, j]`,
`k = i + (j-1)N`). Registered in `src/KrakeNN.jl` (exported by `using KrakeNN`).
Receipts: `test/analytical/poisson_linearsolve_mms.jl` (CPU) and
`test/analytical/poisson_cudss_gpu.jl` (GPU, gated on `CUDA.functional()`).
"""
function solve_poisson_direct(f, N::Integer;
                              bc::Symbol = :dirichlet,
                              method::LinearSolveBackend = PoissonLinearSolve(),
                              k0::Integer = 1)
    N = _check_grid_size(N)
    ffun = _poisson_rhs_function(f, N)

    if bc === :dirichlet
        A, b = assemble_poisson_dirichlet(N, ffun)
        pin = 0
    elseif bc === :neumann
        A, b = assemble_poisson_neumann_unpinned(N, ffun)
        pin = Int(k0)
    else
        throw(ArgumentError("bc must be :dirichlet or :neumann"))
    end

    cache = lin_factorize(A; backend = method, spd = true, pin_k0 = pin)
    u = lin_solve!(cache, b)
    return reshape(Vector{Float64}(Array(u)), N, N)
end
