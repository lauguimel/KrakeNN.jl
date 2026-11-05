module KrakenCUDSSExt

# cuDSS (GPU direct sparse) backend for the factorize-once linear-solve seam
# (issue #8). Loaded by `using CUDSS` (weakdep trigger). CUDA is a strong dep
# of Kraken, usable here directly (Julia >= 1.11 lets extensions use the
# parent's strong deps — do NOT also list CUDA in [weakdeps], that breaks the
# ext precompile env). Zero impact when CUDA/CUDSS are absent.
#
# ARCHITECTURE NOTE — CUDSS without LinearSolve. This ext routes through
# Kraken's OWN seam (`lin_factorize`/`lin_solve!` on `CUDABackendTag`), not
# through LinearSolve.jl's CUDSS bindings: the two exts stay independent
# (either can load without the other) and the GPU path mirrors the proven
# manual-load file `src/solve/linear_solve_cuda.jl` (Aqua GPU jobs), which it
# supersedes for package users. The manual file attaches its methods to the
# INCLUDING module (bench scripts' Main), so both can coexist without method
# clashes.

#
# C0 additions (all opt-in, the two-function seam is unchanged):
#   * `lin_refactorize!` -> `lu!` / `cholesky!` on the cached CudssSolver,
#     reusing the symbolic analysis (the dominant cost on large operators);
#   * allocation-free in-place `lin_solve!(x, cache, b)`: RHS staging buffers
#     are allocated ONCE by `lin_factorize` (cache `work` slot) and the
#     reference-DOF pin is a device-side fill, no scalar host write;
#   * multiple right-hand sides (`CuMatrix` columns, one cuDSS solve);
#   * native transposed solve (`solve_mode = 'T'`) gated on CUDSS.jl >= 1.0;
#     older versions raise instead of silently returning A \ b.
#
# CUDA.jl 5 / 6 compatibility: CUDA.jl 6 moved CUSPARSE into the `cuSPARSE`
# package (still reachable as `CUDA.cuSPARSE`; `CUDA.CUSPARSE` is a deprecated
# binding there). CUDSS.jl 1.x requires CUDA.jl 6; 0.3-0.6 require CUDA.jl 5.

using Kraken
using CUDA
@static if isdefined(CUDA, :cuSPARSE)
    using CUDA.cuSPARSE: CuSparseMatrixCSR
else
    using CUDA.CUSPARSE: CuSparseMatrixCSR
end
using CUDSS
using SparseArrays
using LinearAlgebra

import Kraken: lin_factorize, lin_solve!, _cudss_factorize, _lin_solve_into!,
               _lin_refactorize!, _lin_solution_buffer, _lin_supports_transpose,
               _fv_device_csr

# cuDSS operates on CSR; host CSC operators are converted on upload.
const _CUDSS_GPU_MAT = CuSparseMatrixCSR{Float64,Int32}

# Loaded CUDSS.jl version (fixed for a precompiled extension).
const _CUDSS_VERSION = pkgversion(CUDSS)

# Device scratch owned by a cuDSS LinearSolveCache (`work` slot). `b` is
# allocated by `lin_factorize`; `B` (multi-RHS staging) is (re)allocated on
# the first solve with a new column count, then reused.
mutable struct _CudssWork
    b::CuVector{Float64}
    B::CuMatrix{Float64}
end
_CudssWork(n::Integer) = _CudssWork(CuVector{Float64}(undef, n),
                                    CuMatrix{Float64}(undef, n, 0))

"""
    lin_factorize(A::CuSparseMatrixCSR; backend=CUDABackendTag(), spd=true,
                  pin_k0=0, A_unpinned=A) -> LinearSolveCache

Device-matrix entry of the factorize-once seam (generic keyword form, mirrors
the host `SparseMatrixCSC` entry in src/solve/linear_solve.jl). Because
`Kraken.pin_reference_dof` is a CPU CSC routine, callers pass the
ALREADY-PINNED matrix as `A` plus the unpinned matrix via `A_unpinned`;
`pin_k0 > 0` then only drives the per-solve RHS pinning (done on device).
Prefer the host-matrix entry (`lin_factorize(A_csc; backend=CUDABackendTag(),
pin_k0=k0)`) which handles pinning and upload for you.
"""
function lin_factorize(A::_CUDSS_GPU_MAT;
                       backend::Kraken.LinearSolveBackend = Kraken.CUDABackendTag(),
                       spd::Bool = true, pin_k0::Integer = 0,
                       A_unpinned::_CUDSS_GPU_MAT = A)
    return lin_factorize(backend, A; spd = spd, pin_k0 = Int(pin_k0),
                         A_unpinned = A_unpinned)
end

# CUDA `lin_factorize`: cuDSS symbolic + numeric factorization done ONCE.
# Mirror of src/solve/linear_solve_cuda.jl (see that file's header for the
# CUDSS.jl API notes). `spd=false` goes straight to LU — cuDSS LDLᵀ silently
# mis-factorizes non-symmetric matrices, same trap as the CPU seam.
function lin_factorize(::Kraken.CUDABackendTag, A::_CUDSS_GPU_MAT;
                       spd::Bool = true, pin_k0::Int = 0,
                       A_unpinned::_CUDSS_GPU_MAT = A)
    factor = spd ? cholesky(A; view = 'F') : lu(A)
    return Kraken.LinearSolveCache(Kraken.CUDABackendTag(), factor, A,
                                   A_unpinned, Int(pin_k0), spd,
                                   _CudssWork(size(A, 1)))
end

"""
    _cudss_factorize(::CUDABackendTag, A::SparseMatrixCSC; spd=true, pin_k0=0)
        -> LinearSolveCache

Host-matrix cuDSS factorize (the `lin_factorize(A_csc; backend=
CUDABackendTag())` route declared in src/solve/linear_solve_frontend.jl):
pins reference DOF `pin_k0` on the CPU (geometry-only, done once), uploads
BOTH matrices as `CuSparseMatrixCSR{Float64,Int32}`, and factorizes with cuDSS
(Cholesky for `spd=true`, LU otherwise). This is what makes
[`Kraken.solve_poisson_direct`](@ref) run on GPU with `method=CUDABackendTag()`
and no call-site change.
"""
function _cudss_factorize(::Kraken.CUDABackendTag, A::SparseMatrixCSC{Float64,Int};
                          spd::Bool = true, pin_k0::Int = 0)
    if pin_k0 > 0
        Apin, _ = Kraken.pin_reference_dof(A, zeros(Float64, size(A, 1)),
                                           pin_k0, 0.0)
        A_dev = _CUDSS_GPU_MAT(Apin)
        A_unpinned_dev = _CUDSS_GPU_MAT(A)
    else
        A_dev = _CUDSS_GPU_MAT(A)
        A_unpinned_dev = A_dev
    end
    return lin_factorize(Kraken.CUDABackendTag(), A_dev;
                         spd = spd, pin_k0 = pin_k0,
                         A_unpinned = A_unpinned_dev)
end

"""
    lin_solve!(::CUDABackendTag, cache, b) -> CuVector{Float64}

cuDSS per-RHS solve of the factorize-once seam: reuses the cached symbolic +
numeric factors (`ldiv!`), accepts a host or device `b` (never modified),
applies the reference-DOF RHS pinning on device when `cache.pin_k0 > 0` (pin
value 0, so only the pinned entry is forced — no host round-trip in the outer
loop), and returns a FRESH `CuVector{Float64}` (one device allocation per
call; the in-place `lin_solve!(x, cache, b)` has none). Callers needing host
data `Array(...)` it back (as `Kraken.solve_poisson_direct` does).
"""
function lin_solve!(::Kraken.CUDABackendTag, cache::Kraken.LinearSolveCache,
                    b::AbstractVector)
    x = CuVector{Float64}(undef, length(b))
    return _lin_solve_into!(Kraken.CUDABackendTag(), x, cache, b, false)
end

_lin_solution_buffer(::Kraken.CUDABackendTag, cache::Kraken.LinearSolveCache,
                     b::AbstractVecOrMat) = CuArray{Float64}(undef, size(b))

_lin_supports_transpose(::Kraken.CUDABackendTag, cache::Kraken.LinearSolveCache) =
    cache.spd || Kraken._cudss_transpose_supported(_CUDSS_VERSION)

# Staging buffer of the right shape (vector: preallocated; matrix: reused
# while the column count is unchanged).
_cudss_stage(work::_CudssWork, ::AbstractVector) = work.b
# Caches built by the six-argument constructor (manual-load
# src/solve/linear_solve_cuda.jl) carry no work slot: stage in a fresh buffer.
_cudss_stage(::Nothing, b::AbstractVecOrMat) = CuArray{Float64}(undef, size(b))
function _cudss_stage(work::_CudssWork, b::AbstractMatrix)
    if size(work.B) != size(b)
        work.B = CuMatrix{Float64}(undef, size(b))
    end
    return work.B
end

function _cudss_copy_rhs!(buf::CuArray{Float64}, b)
    if b isa CUDA.AnyCuArray
        buf .= b                                  # device kernel (any eltype)
    elseif b isa Array{Float64}
        copyto!(buf, b)                           # host -> device copy
    else
        copyto!(buf, Array{Float64}(b))           # generic host input
    end
    return buf
end

# The RHS handed to cuDSS: `b` itself when it is a device Float64 array that
# needs no pinning and does not share memory with `x`; otherwise a staged copy
# (the caller's `b` is never modified).
function _cudss_rhs(cache::Kraken.LinearSolveCache, x::CuArray{Float64},
                    b::AbstractVecOrMat)
    if b isa CuArray{Float64} && cache.pin_k0 == 0 && pointer(b) != pointer(x)
        return b
    end
    buf = _cudss_copy_rhs!(_cudss_stage(cache.work, b), b)
    cache.pin_k0 > 0 && Kraken._pin_rhs_rows!(buf, cache.pin_k0)
    return buf
end

# In-place cuDSS solve (vector or multi-RHS matrix). Allocation-free on the
# device when `x` and `b` are `CuArray{Float64}`: `ldiv!` writes into `x`, the
# staging buffers are preallocated, the pin is a one-element device fill.
function _lin_solve_into!(::Kraken.CUDABackendTag, x::AbstractVecOrMat,
                          cache::Kraken.LinearSolveCache, b::AbstractVecOrMat,
                          transpose::Bool)
    x isa CuArray{Float64} ||
        throw(ArgumentError("the cuDSS in-place solve needs a CuArray{Float64} " *
                            "solution buffer, got $(typeof(x))"))
    # spd caches are Cholesky factors of a symmetric matrix: Aᵀ = A. cuDSS does
    # not support solve_mode='T' for SPD/HPD factors anyway.
    use_transpose = transpose && !cache.spd
    use_transpose && Kraken._cudss_check_transpose(_CUDSS_VERSION)
    rhs = _cudss_rhs(cache, x, b)
    F = cache.factor
    if use_transpose
        cudss_set(F, "solve_mode", 'T')
        try
            ldiv!(x, F, rhs)
        finally
            cudss_set(F, "solve_mode", 'N')
        end
    else
        ldiv!(x, F, rhs)                  # reuse symbolic + numeric factors
    end
    return x
end

# --------------------------------------------------------------------------
# Refactorization (same pattern, new values) reusing the symbolic analysis.
# --------------------------------------------------------------------------
function _cudss_pattern_error(what::String)
    throw(ArgumentError("lin_refactorize!: $what has a different sparsity pattern " *
                        "than the factorized operator; call lin_factorize again"))
end

# Copy the values of a host CSC matrix into a device CSR matrix with the same
# pattern. CSR(A) == CSC(Aᵀ), so the CSR value order is that of sparse(Aᵀ).
function _cudss_upload_values!(dev::_CUDSS_GPU_MAT, A::SparseMatrixCSC, what::String)
    size(dev) == size(A) || _cudss_pattern_error(what)
    At = sparse(transpose(A))
    (nnz(At) == length(dev.nzVal) && Array(dev.rowPtr) == At.colptr &&
     Array(dev.colVal) == rowvals(At)) || _cudss_pattern_error(what)
    copyto!(dev.nzVal, nonzeros(At))
    return dev
end

function _cudss_refactor_numeric!(cache::Kraken.LinearSolveCache)
    cache.spd ? cholesky!(cache.factor, cache.A) : lu!(cache.factor, cache.A)
    return cache
end

const _CUDSS_PINNED_DEVICE_REFACTOR =
    "lin_refactorize!: a pinned cuDSS cache (pin_k0 > 0) can only be refactorized " *
    "from a host SparseMatrixCSC (pinning is a CPU routine): lin_refactorize!(cache, A_csc)"

# Values already written into cache.A (e.g. `cache.A.nzVal` by an assembly
# kernel). No device allocation.
function _lin_refactorize!(::Kraken.CUDABackendTag, cache::Kraken.LinearSolveCache,
                           ::Nothing)
    cache.pin_k0 > 0 && throw(ArgumentError(_CUDSS_PINNED_DEVICE_REFACTOR))
    return _cudss_refactor_numeric!(cache)
end

# New device matrix with the same pattern: its values are copied into cache.A.
# When it shares the pattern arrays of cache.A (or IS cache.A) no check and no
# device allocation happen; otherwise the pattern is compared on device.
function _lin_refactorize!(::Kraken.CUDABackendTag, cache::Kraken.LinearSolveCache,
                           A_new::_CUDSS_GPU_MAT)
    cache.pin_k0 > 0 && throw(ArgumentError(_CUDSS_PINNED_DEVICE_REFACTOR))
    A = cache.A
    if A_new !== A
        size(A_new) == size(A) && length(A_new.nzVal) == length(A.nzVal) ||
            _cudss_pattern_error("the new matrix")
        same_arrays = A_new.rowPtr === A.rowPtr && A_new.colVal === A.colVal
        same_arrays || (A_new.rowPtr == A.rowPtr && A_new.colVal == A.colVal) ||
            _cudss_pattern_error("the new matrix")
        copyto!(A.nzVal, A_new.nzVal)
    end
    return _cudss_refactor_numeric!(cache)
end

# New host matrix (UNPINNED): pinned on the CPU when needed, values uploaded.
function _lin_refactorize!(::Kraken.CUDABackendTag, cache::Kraken.LinearSolveCache,
                           A_new::SparseMatrixCSC)
    A_host = SparseMatrixCSC{Float64,Int}(A_new)
    if cache.pin_k0 > 0
        Apin, _ = Kraken.pin_reference_dof(A_host, zeros(Float64, size(A_host, 1)),
                                           cache.pin_k0, 0.0)
        _cudss_upload_values!(cache.A, Apin, "the re-pinned matrix")
        _cudss_upload_values!(cache.A_unpinned, A_host, "the new matrix")
    else
        _cudss_upload_values!(cache.A, A_host, "the new matrix")
    end
    return _cudss_refactor_numeric!(cache)
end

function _lin_refactorize!(::Kraken.CUDABackendTag, cache::Kraken.LinearSolveCache,
                           A_new)
    throw(ArgumentError("lin_refactorize! on the cuDSS backend takes a " *
                        "CuSparseMatrixCSR{Float64,Int32} or a SparseMatrixCSC, " *
                        "got $(typeof(A_new))"))
end

# --------------------------------------------------------------------------
# Device CSR matrix for the FV Newton Jacobian (C2, src/methods/inc_ns/
# fv_newton.jl): pattern built on the host, values written in place on device
# by the coloured dual assembly, refactorized by `lin_refactorize!(cache)`.
# --------------------------------------------------------------------------
function _fv_device_csr(::Kraken.CUDABackendTag, rowptr::AbstractVector,
                        colval::AbstractVector, n::Integer)
    nzval = CUDA.zeros(Float64, length(colval))
    J = _CUDSS_GPU_MAT(CuVector{Int32}(Int32.(rowptr)), CuVector{Int32}(Int32.(colval)),
                       nzval, (Int(n), Int(n)))
    return J, J.nzVal
end

end # module KrakenCUDSSExt
