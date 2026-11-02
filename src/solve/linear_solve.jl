# Backend-parametric linear-solve seam with factorize-once caching.
#
# WHY THIS EXISTS
# ---------------
# The pressure-Poisson matrix on a fixed grid is GEOMETRY-ONLY: it is constant
# across every SIMPLE outer iteration (the lid-driven cavity ran ~3000 of them).
# Re-factorizing it each call is pure waste. This module factorizes the constant
# operator ONCE and reuses the factorization for every right-hand side.
#
# THE SEAM
# --------
# A `LinearSolveCache` holds a backend tag plus a cached factorization object.
# Two functions form the entire contract:
#
#   lin_factorize(A; backend, spd, pin_k0) -> LinearSolveCache
#       Factorize the (constant) operator A ONCE for the given backend.
#         * spd=true  -> Cholesky        (SPD operators: viscous / pressure Lap)
#         * spd=false -> LU / LDLᵀ        (non-symmetric or indefinite fallback)
#         * pin_k0>0  -> the operator was pinned at reference dof k0 (singular
#                        all-Neumann pressure); the RHS is pinned per-solve.
#
#   lin_solve!(cache, b) -> x
#       Reuse the cached factorization to solve A x = b for a fresh RHS b.
#       Called once per outer iteration; NEVER re-factorizes.
#
# Dispatch is keyed on the backend tag (a singleton type). The CPU method lives
# here and uses ONLY LinearAlgebra + SparseArrays (CHOLMOD / UMFPACK). The CUDA
# method (`CUDABackendTag`) lives in `linear_solve_cuda.jl` and is loaded ONLY
# inside a job that has `using CUDA, CUDSS` — it implements the SAME two
# functions for `CuSparseMatrixCSR`, so a GPU run drops in with no call-site
# change. Keep this file CUDA-free so it stays include-able on a CPU-only box.
#
# DESIGN NOTE — pinning lives in the cache, not the call site.
# The singular pressure operator is pinned (row/col k0 replaced by identity) at
# factorize time via `pin_reference_dof`. The cache remembers k0 and the ORIGINAL
# unpinned matrix so each `lin_solve!` can pin the RHS consistently. This keeps
# the seam identical for pinned and non-pinned operators.

using LinearAlgebra
using SparseArrays

# `pin_reference_dof` lives in poisson.jl. Pull it in if not already present so
# this file is standalone-include-able.
if !isdefined(@__MODULE__, :pin_reference_dof)
    include(joinpath(@__DIR__, "poisson.jl"))
end

# --------------------------------------------------------------------------
# Backend tags. These are the DISPATCH KEY for the seam. `CPUBackendTag` is the
# default and is handled here; `CUDABackendTag` is handled in linear_solve_cuda.jl
# (loaded only under `using CUDA, CUDSS`). New backends = new tag + two methods.
# --------------------------------------------------------------------------
abstract type LinearSolveBackend end
struct CPUBackendTag <: LinearSolveBackend end
struct CUDABackendTag <: LinearSolveBackend end

"""
    LinearSolveCache

Holds a factorize-once cache for a CONSTANT sparse operator.

Fields:
  `backend`  the backend tag (`CPUBackendTag()` / `CUDABackendTag()`).
  `factor`   the cached factorization object (CHOLMOD/UMFPACK on CPU, a CUDSS
             solver on GPU). Built ONCE; reused for every RHS.
  `A`        the operator actually factorized (already pinned if `pin_k0 > 0`).
  `A_unpinned` the ORIGINAL operator before pinning (== `A` when not pinned);
             used to pin the RHS consistently per solve.
  `pin_k0`   reference dof for a singular (all-Neumann/periodic) operator, or 0.
  `spd`      whether a Cholesky factorization was used.
  `work`     backend-owned scratch (preallocated device RHS buffers for cuDSS),
             or `nothing` (CPU / LinearSolve backends). The six-argument
             constructor leaves it `nothing`.

[`lin_refactorize!`](@ref) updates `A` / `A_unpinned` IN PLACE (same sparsity
pattern) and refactorizes `factor` reusing its symbolic analysis.
"""
struct LinearSolveCache{B<:LinearSolveBackend, F, M, W}
    backend::B
    factor::F
    A::M
    A_unpinned::M
    pin_k0::Int
    spd::Bool
    work::W
end
LinearSolveCache(backend::LinearSolveBackend, factor, A, A_unpinned,
                 pin_k0::Integer, spd::Bool) =
    LinearSolveCache(backend, factor, A, A_unpinned, Int(pin_k0), spd, nothing)

# --------------------------------------------------------------------------
# Public entry points. Default `backend = CPUBackendTag()`; pass a tag to route.
# --------------------------------------------------------------------------

"""
    lin_factorize(A; backend=CPUBackendTag(), spd=true, pin_k0=0) -> LinearSolveCache

Factorize the constant operator `A` ONCE for `backend` and return a reusable
cache. `spd=true` selects Cholesky; `spd=false` selects an LU/LDLᵀ fallback.
`pin_k0>0` pins reference dof `k0` (singular all-Neumann/periodic operator) at
factorize time via `pin_reference_dof`; the RHS is then pinned consistently per
solve inside [`lin_solve!`](@ref).

This is one of the TWO functions of the backend-parametric factorize-once seam
(dispatch keyed on the `LinearSolveBackend` singleton tag):

  * `CPUBackendTag()` (default) — this file; `SparseMatrixCSC{Float64,Int}`,
    CHOLMOD Cholesky / LDLᵀ / UMFPACK LU. LinearAlgebra + SparseArrays only.
  * `CUDABackendTag()` — `linear_solve_cuda.jl`, loaded ONLY under
    `using CUDA, CUDSS`; same two functions for `CuSparseMatrixCSR{Float64,Int32}`
    (cuDSS). Because `pin_reference_dof` is a CPU routine, the CUDA method takes
    the ALREADY-PINNED matrix as `A` plus an `A_unpinned` keyword; `pin_k0>0`
    there only drives the per-solve RHS pinning (done on device).

Measured win (issue #8, Aqua A100): the factorize-once cuDSS pressure solve runs
~30x faster than CPU CHOLMOD at 1M DOF, with the amortized back-substitution at
4.7 ms/solve (`benchmarks/krk/inc_ns/poisson_gpu_bench.jl`). Registered in
`src/Kraken.jl` THROUGH `solve/poisson.jl` (which tail-includes this file —
never include it directly there); exported by `using Kraken`.
"""
function lin_factorize(A::SparseMatrixCSC{Float64,Int};
                       backend::LinearSolveBackend = CPUBackendTag(),
                       spd::Bool = true,
                       pin_k0::Integer = 0)
    return lin_factorize(backend, A; spd = spd, pin_k0 = Int(pin_k0))
end

"""
    lin_solve!(cache, b) -> x

Solve `A x = b` reusing the cached factorization built by
[`lin_factorize`](@ref). NEVER re-factorizes — this is the per-outer-iteration
call of the factorize-once seam (the pressure-Poisson operator on a fixed grid is
geometry-only and constant across every SIMPLE iteration; only `b` changes).

For a pinned operator (`cache.pin_k0 > 0`) the RHS is adjusted per solve to
enforce `x[k0] = 0` consistently with the pinned matrix (the cache keeps the
ORIGINAL unpinned operator for this), so the seam looks identical for pinned and
non-pinned operators at the call site.

Returns a `Vector{Float64}` on the CPU backend; the CUDA method
(`linear_solve_cuda.jl`, loaded under `using CUDA, CUDSS`) accepts a host or
device `b` and returns a `CuVector{Float64}`.
"""
function lin_solve!(cache::LinearSolveCache, b::AbstractVector;
                    transpose::Bool = false)
    transpose || return lin_solve!(cache.backend, cache, b)
    x = _lin_solution_buffer(cache.backend, cache, b)
    _lin_solve_into!(cache.backend, x, cache, b, true)
    return x
end

"""
    lin_solve!(cache, B::AbstractMatrix; transpose=false) -> X

Multiple right-hand sides: solve `A X = B` (or `Aᵀ X = B`) for every column of
`B` with ONE call on the cached factors (cuDSS solves all columns together).
Returns a fresh array (`Matrix{Float64}` on CPU, `CuMatrix{Float64}` on CUDA);
use the in-place form `lin_solve!(X, cache, B)` to reuse storage.
"""
function lin_solve!(cache::LinearSolveCache, B::AbstractMatrix;
                    transpose::Bool = false)
    size(B, 1) == size(cache.A, 1) ||
        throw(DimensionMismatch("B has $(size(B, 1)) rows, operator has $(size(cache.A, 1))"))
    X = _lin_solution_buffer(cache.backend, cache, B)
    _lin_solve_into!(cache.backend, X, cache, B, transpose)
    return X
end

"""
    lin_solve!(x, cache, b; transpose=false) -> x

In-place solve of `A x = b` (`transpose=true`: `Aᵀ x = b`) into the
caller-owned `x`, reusing the cached factors. `x` and `b` are both vectors or
both matrices (multiple right-hand sides, one column each) of the same size;
`b` is never modified. Pinned caches (`pin_k0 > 0`) pin row `k0` of every
right-hand side to 0, as the allocating form does.

On the cuDSS backend this is the allocation-free path: with `x` a
`CuArray{Float64}` and `b` on device, no device memory is allocated per call
(`CUDA.@allocated == 0`; the RHS is staged in buffers preallocated by
`lin_factorize` when pinning or a host `b` requires it). Transposed solves of
a non-symmetric (LU) cuDSS factorization need CUDSS.jl >= 1.0 and raise an
error otherwise (older versions silently ignore the transpose request); see
[`lin_supports_transpose`](@ref). For `spd=true` caches `Aᵀ = A` and the
transposed solve is the normal solve on every backend.
"""
function lin_solve!(x::AbstractVecOrMat, cache::LinearSolveCache,
                    b::AbstractVecOrMat; transpose::Bool = false)
    size(x) == size(b) ||
        throw(DimensionMismatch("x has size $(size(x)), b has size $(size(b))"))
    size(b, 1) == size(cache.A, 1) ||
        throw(DimensionMismatch("b has $(size(b, 1)) rows, operator has $(size(cache.A, 1))"))
    _lin_solve_into!(cache.backend, x, cache, b, transpose)
    return x
end

"""
    lin_refactorize!(cache) -> cache
    lin_refactorize!(cache, A_new) -> cache

Numeric refactorization of a cached operator whose VALUES changed but whose
sparsity pattern did not (Newton iterations, adjoint after the last Newton
step). The symbolic analysis done by [`lin_factorize`](@ref) is reused
(`lu!` / `cholesky!` / `ldlt!`), which is where the time goes on large
operators (cuDSS 1024² saddle point: analysis 45 s vs refactorization 1.2 s).

  * `lin_refactorize!(cache)`: the new values were already written into
    `cache.A_unpinned` (`=== cache.A` when not pinned), e.g. by an assembly
    kernel writing `cache.A.nzVal` directly on GPU.
  * `lin_refactorize!(cache, A_new)`: `A_new` (UNPINNED, same pattern as the
    matrix given to `lin_factorize`) has its values COPIED into
    `cache.A_unpinned` — the matrix object passed to `lin_factorize` is
    therefore overwritten on CPU, as with `lu!`. On cuDSS `A_new` may be a host
    `SparseMatrixCSC` or a device `CuSparseMatrixCSR`.

Pinned caches (`pin_k0 > 0`) re-pin from the new values. On cuDSS a pinned
cache can only be refactorized from a host `SparseMatrixCSC` (pinning is a CPU
routine). A pattern change raises an `ArgumentError`: call `lin_factorize`
again. A CPU `spd=false` cache factorized by LDLᵀ (symmetric input) refuses
non-symmetric new values, which LDLᵀ would silently mis-factorize.
"""
lin_refactorize!(cache::LinearSolveCache) =
    (_lin_refactorize!(cache.backend, cache, nothing); cache)
lin_refactorize!(cache::LinearSolveCache, A_new) =
    (_lin_refactorize!(cache.backend, cache, A_new); cache)

"""
    lin_supports_transpose(cache) -> Bool

Whether `lin_solve!(...; transpose=true)` can run on this cache. Always true on
the CPU backend and for `spd=true` caches; on cuDSS LU caches it requires
CUDSS.jl >= 1.0. False for the LinearSolve.jl front-end.
"""
lin_supports_transpose(cache::LinearSolveCache) =
    _lin_supports_transpose(cache.backend, cache)

# --------------------------------------------------------------------------
# Backend hooks of the extended seam. Backends without support raise an
# explicit error; nothing silently falls back to a different operator.
# --------------------------------------------------------------------------
_lin_backend_name(tag::LinearSolveBackend) = string(nameof(typeof(tag)))

_lin_solution_buffer(::LinearSolveBackend, cache::LinearSolveCache,
                     b::AbstractVecOrMat) = Array{Float64}(undef, size(b))

_lin_supports_transpose(::LinearSolveBackend, cache::LinearSolveCache) = false

function _lin_refactorize!(tag::LinearSolveBackend, cache::LinearSolveCache, A_new)
    throw(ArgumentError("lin_refactorize! is not supported by the " *
                        _lin_backend_name(tag) * " backend; call lin_factorize again"))
end

# Generic in-place fallback: column-wise through the backend's allocating
# solve. Transposed solves are refused (no silent A \ b).
function _lin_solve_into!(tag::LinearSolveBackend, x::AbstractVecOrMat,
                          cache::LinearSolveCache, b::AbstractVecOrMat,
                          transpose::Bool)
    transpose && throw(ArgumentError("transposed solve is not supported by the " *
                                     _lin_backend_name(tag) * " backend"))
    if b isa AbstractVector
        copyto!(x, lin_solve!(tag, cache, b))
    else
        for j in axes(b, 2)
            copyto!(view(x, :, j), lin_solve!(tag, cache, b[:, j]))
        end
    end
    return x
end

# Minimum CUDSS.jl version with a working native transposed solve. CUDSS.jl
# 0.6.5 + libcudss 0.7.1 accept `solve_mode = 'T'` and silently return A \ b
# (measured on Aqua, J0 2026-11-02); CUDSS.jl 1.0 + libcudss 1.0 solve Aᵀ x = b
# to 1e-12. Kept here (CUDA-free) so the gate is testable on CPU.
const CUDSS_TRANSPOSE_MIN_VERSION = v"1.0.0"

_cudss_transpose_supported(ver::VersionNumber) = ver >= CUDSS_TRANSPOSE_MIN_VERSION

function _cudss_check_transpose(ver::VersionNumber)
    _cudss_transpose_supported(ver) && return nothing
    error("transposed cuDSS solve requires CUDSS.jl >= $(CUDSS_TRANSPOSE_MIN_VERSION) " *
          "(loaded: $ver). Older CUDSS.jl/libcudss ignore solve_mode='T' and " *
          "return the solution of A x = b, so Kraken refuses instead. Upgrade to " *
          "CUDA.jl 6 + CUDSS.jl 1, or factorize transpose(A) as a separate cache.")
end

# --------------------------------------------------------------------------
# CPU method (CHOLMOD for SPD, LU/LDLᵀ fallback). LinearAlgebra + SparseArrays
# only. This is the reference path; the CUDA method mirrors it exactly.
# --------------------------------------------------------------------------
function lin_factorize(::CPUBackendTag, A::SparseMatrixCSC{Float64,Int};
                       spd::Bool = true, pin_k0::Int = 0)
    A_unpinned = A
    if pin_k0 > 0
        Apin, _ = pin_reference_dof(A, zeros(Float64, size(A, 1)), pin_k0, 0.0)
        A_used = Apin
    else
        A_used = A
    end

    if spd
        factor = cholesky(Symmetric(A_used); check = true)
    else
        # Non-symmetric / indefinite fallback. LDLᵀ only when the matrix is
        # GENUINELY symmetric (symmetric indefinite); otherwise LU directly.
        # NOTE: `ldlt(Symmetric(A))` on a non-symmetric A does NOT throw — it
        # silently factorizes the symmetrized (upper-triangle) matrix and
        # returns wrong fields (seen as O(1e2) residuals on advection-diffusion
        # systems). The symmetry check is O(nnz), negligible vs factorization.
        factor = if issymmetric(A_used)
            try
                ldlt(Symmetric(A_used))
            catch
                lu(A_used)   # symmetric but LDLᵀ-unfactorizable (e.g. zero pivot)
            end
        else
            lu(A_used)
        end
    end

    return LinearSolveCache(CPUBackendTag(), factor, A_used, A_unpinned,
                            Int(pin_k0), spd)
end

# Row k0 of every RHS column set to the pin value 0 (equivalent to
# `pin_reference_dof(A, b, k0, 0.0)`: the column correction vanishes at 0).
function _pin_rhs_rows!(b::AbstractVecOrMat, k0::Int)
    if b isa AbstractVector
        fill!(view(b, k0:k0), 0.0)
    else
        fill!(view(b, k0:k0, :), 0.0)
    end
    return b
end

_cpu_factor_symmetric(F) = F isa SparseArrays.CHOLMOD.Factor

_lin_supports_transpose(::CPUBackendTag, cache::LinearSolveCache) = true

function _lin_solve_into!(::CPUBackendTag, x::AbstractVecOrMat,
                          cache::LinearSolveCache, b::AbstractVecOrMat,
                          transpose::Bool)
    rhs = Array{Float64}(b)          # copy: the caller's b is never modified
    cache.pin_k0 > 0 && _pin_rhs_rows!(rhs, cache.pin_k0)
    F = cache.factor
    # CHOLMOD factors (Cholesky / LDLᵀ) are symmetric: Aᵀ = A.
    sol = (transpose && !_cpu_factor_symmetric(F)) ?
          LinearAlgebra.transpose(F) \ rhs : F \ rhs
    copyto!(x, sol)
    return x
end

function _check_same_pattern(A::SparseMatrixCSC, B::SparseMatrixCSC, what::String)
    (size(A) == size(B) && A.colptr == B.colptr && rowvals(A) == rowvals(B)) ||
        throw(ArgumentError("lin_refactorize!: $what has a different sparsity " *
                            "pattern than the factorized operator; call lin_factorize again"))
    return nothing
end

function _lin_refactorize!(::CPUBackendTag, cache::LinearSolveCache, A_new)
    Au = cache.A_unpinned
    if A_new !== nothing && A_new !== Au
        A_new isa SparseMatrixCSC ||
            throw(ArgumentError("lin_refactorize! on the CPU backend takes a " *
                                "SparseMatrixCSC, got $(typeof(A_new))"))
        _check_same_pattern(Au, A_new, "the new matrix")
        copyto!(nonzeros(Au), nonzeros(A_new))
    end
    if cache.pin_k0 > 0
        Apin, _ = pin_reference_dof(Au, zeros(Float64, size(Au, 1)),
                                    cache.pin_k0, 0.0)
        _check_same_pattern(cache.A, Apin, "the re-pinned matrix")
        copyto!(nonzeros(cache.A), nonzeros(Apin))
    end
    F = cache.factor
    A = cache.A
    if F isa SparseArrays.CHOLMOD.Factor
        if cache.spd
            cholesky!(F, Symmetric(A); check = true)
        else
            # LDLᵀ was chosen because A was symmetric at factorize time.
            issymmetric(A) ||
                throw(ArgumentError("lin_refactorize!: the cache was factorized " *
                                    "by LDLᵀ (symmetric input) but the new values " *
                                    "are not symmetric; call lin_factorize again"))
            ldlt!(F, Symmetric(A); check = true)
        end
    else
        lu!(F, A)                    # UMFPACK, reuses the symbolic analysis
    end
    return cache
end

function lin_solve!(::CPUBackendTag, cache::LinearSolveCache, b::AbstractVector)
    bvec = Vector{Float64}(b)
    if cache.pin_k0 > 0
        _, bpin = pin_reference_dof(cache.A_unpinned, bvec, cache.pin_k0, 0.0)
        return cache.factor \ bpin
    else
        return cache.factor \ bvec
    end
end
