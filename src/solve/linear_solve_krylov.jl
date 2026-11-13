# Krylov branch of the linear-solve seam: restarted, right-preconditioned GMRES
# (`gnf_gmres!`) on the ASSEMBLED Jacobian, with one of two preconditioners.
#
# Same seam as the direct backends (`lin_factorize` / `lin_refactorize!` /
# `lin_solve!(...; transpose)`), so the FV Newton and adjoint drivers switch
# branch with one keyword (`linsolve = KrylovBackendTag()`). The matrix is a
# plain CSR triple (`KrylovCSRMatrix`) on the device of the assembler (Array or
# CuArray). This file is CUDA-free: the CPU methods (ILU(0), triangular solves,
# UMFPACK Schur LU) live here, SpMV and the block kernels are KernelAbstractions
# kernels; the CuArray methods (cuSPARSE `ilu02!`, SpSV, cuDSS Schur LU) live
# in `ext/KrakeNNCUDSSExt.jl` (`using CUDA, CUDSS`).
#
# P1 (`precond = :ilu0`) = ILU(0) of the full saddle-point Jacobian (interleaved
# u, v, [w,] p unknowns per cell; Rhie–Chow gives every continuity row a
# non-zero pressure diagonal, so the zero-pivot breakdown of a pure saddle point
# does not occur structurally). Pivots are checked after every factorization:
# a zero or non-finite pivot raises, small pivots are counted and reported.
#
# P2 (`precond = :simple`) = block SIMPLE. With the unknowns split into
# velocity-like (U: every field but pressure, scalars included), pressure (P)
# and extras (X: the flow-rate border), J = [A G; D C] on U ∪ P, and
#     S̃ = C − D diag(A)⁻¹ G          (SIMPLE Schur approximation),
# one application z = M⁻¹ v is
#     û = Ã⁻¹ v_U                     (Ã = ILU(0) of A),
#     p = S̃⁻¹ (v_P − D û)             (LU of S̃: UMFPACK / cuDSS; or ILU(0)),
#     u = û − diag(A)⁻¹ G p,   z_X = v_X,
# i.e. M = [Ã 0; D S̃] [I diag(A)⁻¹G; 0 I], the extras being left to GMRES
# (identity in M). Block patterns and the S̃ pattern are built once on the host;
# values are refreshed on device after each refactorization (S̃ by one thread
# per pressure row, no stored triple products).
#
# P3 (`precond = :mg`) = the same block SIMPLE with Ã⁻¹ and S̃⁻¹ each replaced by
# one aggregation-multigrid cycle (src/solve/linear_solve_multigrid.jl;
# `schur = :auto` resolves to `:mg` for P3 and to `:lu` for P2; `schur = :lu`
# with P3 keeps the exact Schur LU, `schur = :mg` with P2 uses multigrid for S̃
# only). The aggregates follow the structured grid, so P3 needs the geometry
# of the matrix (`krylov_csr_matrix(...; geometry)`, set by the FV Newton
# solver).
#
# Flow-rate border (imposed mean velocity, extra unknown F_x in the last row and
# column), P1 only: `border = :krylov` (default) factorizes the LOCAL block only
# — the border row and column are removed from the ILU and its diagonal set to
# 1, so M = blockdiag(ILU(J_loc), 1) exactly — and leaves the coupling to GMRES.
# `border = :ilu` keeps the border inside the ILU(0). P2 always treats the
# border as an extra (identity in M).
#
# Transposed solve (adjoint): the transposed pattern and the permutation
# `perm` (Jᵀ[q] = J[perm[q]]) are built once on the host; after each
# refactorization Jᵀ is gathered on device. P1 also gathers (LU)ᵀ = UᵀLᵀ and
# runs GMRES on Jᵀ with (UᵀLᵀ)⁻¹; P2 builds a second SIMPLE preconditioner from
# the blocks of Jᵀ (its Schur matrix is S̃ᵀ, factorized on its own: no
# transposed factor solve is needed).

using LinearAlgebra
using SparseArrays
using KernelAbstractions

"""
    KrylovBackendTag(; rtol=1e-6, rtol_transpose=1e-10, restart=50, maxiter=2000,
                     orth=:cgs2, border=:krylov, pivot_tol=1e-10, strict=true,
                     trisolve=:persistent, precond=:auto, schur=:auto)

Linear-solve backend: restarted GMRES (`gnf_gmres!`, restart length `restart`)
with a right preconditioner on the assembled matrix. `precond = :auto`
(default) is resolved per matrix by [`lin_factorize`](@ref): `:vanka` (P4,
below) for a 3D isothermal FV Jacobian (block layout of 4 fields per cell with
the pressure in slot 4, and a cell grid with more than one cell along z, as
set by `fv_newton_solver` on a 3D problem without energy equation), `:ilu0`
(P1) otherwise, so 2D problems, 3D problems with an energy equation and bare
matrices keep ILU(0). Explicit choices: `precond = :ilu0` (P1,
ILU(0) of the whole matrix) or `precond = :simple` (P2, block SIMPLE: ILU(0)
of the velocity block, Schur approximation S̃ = C − D diag(A)⁻¹ G factorized by
LU (`schur = :lu`: UMFPACK on CPU, cuDSS on GPU) or by ILU(0) (`schur =
:ilu0`)). P2 needs the block layout of the matrix (`krylov_csr_matrix(...;
nfields, pslot, nlocal)`, set by the FV Newton solver). A solve stops when the
TRUE relative residual ‖b − A x‖/‖b‖ ≤ `rtol` (forward) or `rtol_transpose`
(`transpose = true`, the adjoint), or after `maxiter` iterations; with
`strict = true` a non-converged solve raises. Inside Newton this is an inexact
Newton step with a fixed forcing term `rtol`.

`border` selects the flow-rate border treatment of P1 (`:krylov` =
preconditioner on the local block only, `:ilu` = border kept in the ILU(0));
P2 leaves every unknown beyond `nlocal` to GMRES. `pivot_tol` is the
relative pivot size |Uᵢᵢ| / maxⱼ|Aᵢⱼ| below which a pivot is counted as small
([`lin_krylov_pivot`](@ref)); an exactly zero or non-finite pivot always raises.
`trisolve` (GPU only): `:persistent` keeps the cuSPARSE triangular-solve
analysis between solves, `:simple` redoes it at every application (slow,
fallback). Works on CPU (`to = Array`) and, with `using CUDA, CUDSS`, on
`CuArray`.

`precond = :mg` (P3) is block SIMPLE with one aggregation-multigrid cycle for
the velocity block and (`schur = :auto` → `:mg`) for S̃; `schur = :auto` means
`:lu` for P2. It needs the block layout and the grid geometry of the matrix
(both set by `fv_newton_solver`). Multigrid options: `mg_cycle` (`:W` or
`:V`), `mg_smoother` (`:l1jacobi`, `:jacobi` or `:ilu0`), `mg_omega` (smoother
damping, 0 = smoother default: 0.7 for `:jacobi`, 1 otherwise), `mg_sweeps`
(pre- and post-smoothing sweeps), `mg_cycles` (cycles per block application),
`mg_coarse` (largest coarsest level, solved by sparse LU), `mg_levels`
(maximum number of levels), `mg_scale` (coarse-correction scaling).

Outer block structure and Schur approximation of P2 / P3
(src/solve/linear_solve_block_schur.jl; defaults = the SIMPLE above):
`block` (`:simple`, `:lower` or `:upper` block triangular), `schur_approx`
(`:simple` S̃ = C − D diag(A)⁻¹ G, `:simplec` with the l1 row sum of A, `:lsc`
least-squares commutator with diag(A) scaling, `:pcd` algebraic pressure
convection–diffusion), `schur_q` (`:auto`, `:diag`, `:rowsum`, or `:scalar`
= `schur_qscale` · I, e.g. the cell volume (a velocity mass matrix): the Q of
S̃ = C − D Q⁻¹ G and of the LSC middle term), `schur_shift` (σ ≥ 0: S̃ − σ diag(D Q⁻¹ G)), `lsc_alpha`
(weight of C in the LSC middle term), `ablock` (velocity-block solver:
`:auto` = ILU(0) for P2 and multigrid for P3, or `:ilu0`, `:mg`, `:lu` exact
sparse LU).

`precond = :vanka` (P4, src/solve/linear_solve_mg_monolithic.jl) is a
monolithic geometric multigrid on the coupled velocity–pressure block (all
unknowns of a cell coarsened together on 2 × 2 × 2 blocks, restriction by
sums, prolongation `vanka_prolong = :linear` (trilinear, default) or
`:constant`, Galerkin coarse operators R J P of the full saddle-point matrix)
with an additive Vanka smoother (one local saddle-point solve per fluid
pressure unknown, damping `vanka_omega`, default 0.5: with constant
prolongation ω = 0.5 passes the D2 gate of the 3D benchmarks, ω = 0.7 fails
it on the body-force duct); the extras are left to
GMRES. It takes the same `mg_*` options as P3 (`mg_smoother` and `mg_omega`
unused) and needs the block layout and the grid geometry. P3L, the setting of
the 3D benchmarks before P4, stays available as `precond = :mg, block =
:upper, schur_approx = :lsc, mg_cycles = 4`.
"""
Base.@kwdef struct KrylovBackendTag <: LinearSolveBackend
    rtol::Float64 = 1e-6
    rtol_transpose::Float64 = 1e-10
    restart::Int = 50
    maxiter::Int = 2000
    orth::Symbol = :cgs2
    border::Symbol = :krylov
    pivot_tol::Float64 = 1e-10
    strict::Bool = true
    trisolve::Symbol = :persistent
    precond::Symbol = :auto
    schur::Symbol = :auto
    mg_cycle::Symbol = :W
    mg_smoother::Symbol = :l1jacobi
    mg_omega::Float64 = 0.0
    mg_sweeps::Int = 2
    mg_cycles::Int = 1
    mg_coarse::Int = 2000
    mg_levels::Int = 20
    mg_scale::Float64 = 1.0
    block::Symbol = :simple
    schur_approx::Symbol = :simple
    schur_q::Symbol = :auto
    schur_shift::Float64 = 0.0
    lsc_alpha::Float64 = 1.0
    ablock::Symbol = :auto
    schur_qscale::Float64 = 0.0
    vanka_omega::Float64 = 0.5
    vanka_prolong::Symbol = :linear
end

# Preconditioner actually used for the matrix `A` (`precond = :auto` resolved):
# P4 on a 3D isothermal FV Jacobian (4 fields per cell, pressure in slot 4,
# geometry with Nz > 1), P1 otherwise (2D, 3D with energy, bare matrices).
function _krylov_precond_kind(tag::KrylovBackendTag, A)
    tag.precond === :auto || return tag.precond
    return (A.nfields == 4 && A.pslot == 4 && A.nlocal > 0 && A.geom !== nothing &&
            A.geom.dims[3] > 1) ? :vanka : :ilu0
end

# Copy of `tag` with the preconditioner `precond` (the resolved tag kept in the cache).
function _krylov_with_precond(tag::KrylovBackendTag, precond::Symbol)
    names = fieldnames(KrylovBackendTag)
    nt = NamedTuple{names}(ntuple(i -> getfield(tag, i), length(names)))
    return KrylovBackendTag(; merge(nt, (; precond))...)
end

# Schur treatment actually used (`:auto` resolved per preconditioner).
_krylov_schur_kind(tag::KrylovBackendTag) =
    tag.schur === :auto ? (tag.precond === :mg ? :mg : :lu) : tag.schur

"""
    KrylovCSRMatrix(n, rowptr, colval, nzval, border[, nfields, pslot, nlocal])

Square CSR matrix (1-based `Int32` indices, sorted columns, every diagonal
stored) on the device of its arrays; the matrix type of the Krylov branch.
`border` is the index of the flow-rate border unknown, or 0. Block layout (used
by P2 only; 0 = unknown): unknowns `1:nlocal` are interleaved with `nfields`
fields per cell, field `pslot` being the pressure; unknowns beyond `nlocal` are
extras. `geom` (used by P3 only; `nothing` = unknown) is the host geometry of
the cells: `(; dims = (Nx, Ny, Nz), solid::Vector{Bool})`, cell
`c = i + (j-1) Nx + (k-1) Nx Ny` being cell `(i, j, k)`.
"""
struct KrylovCSRMatrix{VI<:AbstractVector{Int32},VV<:AbstractVector{Float64}}
    n::Int
    rowptr::VI
    colval::VI
    nzval::VV
    border::Int
    nfields::Int
    pslot::Int
    nlocal::Int
    geom::Any
end
KrylovCSRMatrix(n, rowptr, colval, nzval, border) =
    KrylovCSRMatrix(n, rowptr, colval, nzval, border, 0, 0, 0, nothing)
KrylovCSRMatrix(n, rowptr, colval, nzval, border, nfields, pslot, nlocal) =
    KrylovCSRMatrix(n, rowptr, colval, nzval, border, nfields, pslot, nlocal, nothing)
Base.size(A::KrylovCSRMatrix) = (A.n, A.n)
Base.size(A::KrylovCSRMatrix, d::Integer) = d <= 2 ? A.n : 1
SparseArrays.nnz(A::KrylovCSRMatrix) = length(A.colval)

"""
    krylov_csr_matrix(to, rowptr, colval, n; border=0, nfields=0, pslot=0, nlocal=0,
                      geometry=nothing) -> KrylovCSRMatrix

Upload a host CSR pattern (1-based, sorted columns) with zero values to the
device of `to` (`Array` or `CuArray`). `nfields`, `pslot`, `nlocal`: block
layout for the P2/P3 preconditioners; `geometry = (; dims = (Nx, Ny, Nz),
solid)`: cell grid for the P3 aggregates (see [`KrylovCSRMatrix`](@ref)).
"""
function krylov_csr_matrix(to, rowptr::AbstractVector, colval::AbstractVector,
                           n::Integer; border::Integer = 0, nfields::Integer = 0,
                           pslot::Integer = 0, nlocal::Integer = 0, geometry = nothing)
    length(rowptr) == n + 1 || throw(DimensionMismatch("rowptr has length $(length(rowptr)), expected $(n + 1)"))
    0 <= nlocal <= n || throw(ArgumentError("nlocal = $nlocal outside 0:$n"))
    (nfields == 0 || (1 <= pslot <= nfields && nlocal % nfields == 0)) ||
        throw(ArgumentError("inconsistent block layout (nfields = $nfields, pslot = $pslot, nlocal = $nlocal)"))
    geom = nothing
    if geometry !== nothing
        dims = ntuple(d -> Int(geometry.dims[d]), 3)
        solid = Vector{Bool}(vec(collect(geometry.solid)))
        length(solid) == prod(dims) ||
            throw(ArgumentError("geometry: $(length(solid)) solid flags for $(prod(dims)) cells"))
        (nfields == 0 || nlocal == nfields * prod(dims)) ||
            throw(ArgumentError("geometry: $(prod(dims)) cells but nlocal / nfields = $(nlocal ÷ max(nfields, 1))"))
        geom = (; dims, solid)
    end
    rp = to(Vector{Int32}(rowptr))
    cv = to(Vector{Int32}(colval))
    nz = to(zeros(Float64, length(colval)))
    return KrylovCSRMatrix(Int(n), rp, cv, nz, Int(border), Int(nfields), Int(pslot), Int(nlocal), geom)
end

# One preconditioned operator (forward J with LU, or Jᵀ with UᵀLᵀ).
# `first_unit`: the lower solve has a unit diagonal (forward: L unit, U
# non-unit); false for the transpose (Uᵀ non-unit lower, Lᵀ unit upper).
mutable struct _KrylovSide{VI,VV}
    n::Int
    rowptr::VI
    colval::VI
    diag::VI
    A::VV            # operator values (the assembled J, or the gathered Jᵀ)
    F::VV            # ILU(0) factor values, same pattern
    first_unit::Bool
    t0::VV
    t1::VV
    t2::VV
    t3::VV
    z::VV
    handle::Any      # backend triangular-solve state (GPU), or nothing
    simple::Any      # P2 block preconditioner of this operator, or nothing (P1)
end

# `factor = false`: no ILU of this operator (P2 sides: F is empty, only the
# operator values and the work vectors are kept).
function _KrylovSide(rowptr, colval, diag, A, first_unit::Bool; factor::Bool = true)
    n = length(rowptr) - 1
    F = factor ? similar(A) : similar(A, 0)
    vec() = fill!(similar(A, n), 0.0)
    return _KrylovSide(n, rowptr, colval, diag, A, F, first_unit,
                       vec(), vec(), vec(), vec(), vec(), nothing, nothing)
end

mutable struct KrylovFactor{S,VI,VV,VM}
    tag::KrylovBackendTag
    fwd::S
    tr::Union{Nothing,S}
    perm::Union{Nothing,VI}
    tstale::Bool
    bpos::VI                     # factor positions zeroed by border = :krylov
    bdiag::Int                   # position of the border diagonal, or 0
    ratio::VV                    # per-row pivot ratio |Uᵢᵢ| / max|Aᵢⱼ| (−1 = non-finite)
    ratio_host::Vector{Float64}
    basis::VM                    # GMRES basis, reused by every solve
    y::VV
    rhs::VV
    work::NamedTuple             # GMRES work vectors (r, hbuf, d), reused by every solve
    pivot::NamedTuple
    log::Vector{NamedTuple}
    n_factor::Int
end

_dev_int(template, h::Vector{Int32}) = copyto!(similar(template, Int32, length(h)), h)

function _krylov_diag_positions(rowptr::Vector{Int32}, colval::Vector{Int32}, n::Int)
    d = zeros(Int32, n)
    @inbounds for i in 1:n, q in rowptr[i]:(rowptr[i+1]-1)
        colval[q] == i && (d[i] = q)
    end
    i0 = findfirst(==(0), d)
    i0 === nothing || throw(ArgumentError("row $i0 has no stored diagonal (ILU(0) needs every diagonal)"))
    return d
end

function _krylov_check_sorted(rowptr::Vector{Int32}, colval::Vector{Int32}, n::Int)
    @inbounds for i in 1:n, q in rowptr[i]:(rowptr[i+1]-2)
        colval[q] < colval[q+1] ||
            throw(ArgumentError("row $i of the CSR matrix has unsorted or repeated columns"))
    end
end

"""
    lin_factorize(A::KrylovCSRMatrix; backend=KrylovBackendTag(), spd=false, pin_k0=0)

Build the preconditioner of `A` (Krylov branch): ILU(0) (P1), block SIMPLE
(P2), block SIMPLE with multigrid blocks (P3) or monolithic Vanka multigrid
(P4), per `backend.precond`; `precond = :auto` (default) gives P4 on a 3D
isothermal FV Jacobian and P1 otherwise, and the cache keeps the resolved tag
(`cache.backend.precond`). `pin_k0 > 0` is not supported (pin the operator
rows instead, as the FV Newton problems do).
"""
function lin_factorize(A::KrylovCSRMatrix; backend::LinearSolveBackend = KrylovBackendTag(),
                       spd::Bool = false, pin_k0::Integer = 0)
    backend isa KrylovBackendTag ||
        throw(ArgumentError("a KrylovCSRMatrix needs backend = KrylovBackendTag(), got $(_lin_backend_name(backend))"))
    pin_k0 == 0 || throw(ArgumentError("KrylovBackendTag does not support pin_k0 > 0"))
    tag = backend.precond === :auto ? _krylov_with_precond(backend, _krylov_precond_kind(backend, A)) : backend
    tag.border in (:krylov, :ilu) || throw(ArgumentError("border must be :krylov or :ilu, got :$(tag.border)"))
    tag.trisolve in (:persistent, :simple) ||
        throw(ArgumentError("trisolve must be :persistent or :simple, got :$(tag.trisolve)"))
    tag.precond in (:ilu0, :simple, :mg, :vanka) ||
        throw(ArgumentError("precond must be :auto, :ilu0 (P1), :simple (P2), :mg (P3) or :vanka (P4), got :$(tag.precond)"))
    tag.vanka_omega > 0 || throw(ArgumentError("vanka_omega must be > 0, got $(tag.vanka_omega)"))
    tag.vanka_prolong in (:linear, :constant) ||
        throw(ArgumentError("vanka_prolong must be :linear or :constant, got :$(tag.vanka_prolong)"))
    tag.schur in (:auto, :lu, :ilu0, :mg) ||
        throw(ArgumentError("schur must be :auto, :lu, :ilu0 or :mg, got :$(tag.schur)"))
    tag.mg_cycle in (:V, :W) || throw(ArgumentError("mg_cycle must be :V or :W, got :$(tag.mg_cycle)"))
    tag.mg_smoother in (:jacobi, :l1jacobi, :ilu0) ||
        throw(ArgumentError("mg_smoother must be :jacobi, :l1jacobi or :ilu0, got :$(tag.mg_smoother)"))
    (tag.mg_sweeps >= 0 && tag.mg_cycles >= 1 && tag.mg_coarse >= 1 && tag.mg_levels >= 1 &&
     tag.mg_omega >= 0) ||
        throw(ArgumentError("multigrid options: need mg_sweeps >= 0, mg_cycles >= 1, mg_coarse >= 1, " *
                            "mg_levels >= 1, mg_omega >= 0"))
    _bs_check_tag(tag)
    simple = tag.precond in (:simple, :mg, :vanka)     # block preconditioner (P2/P3/P4): no ILU of J
    simple && A.nfields == 0 &&
        throw(ArgumentError("precond = :$(tag.precond) needs the block layout of the matrix " *
                            "(krylov_csr_matrix(...; nfields, pslot, nlocal))"))
    simple && (tag.precond === :vanka || _krylov_ablock_kind(tag) === :mg ||
               _krylov_schur_kind(tag) === :mg) && A.geom === nothing &&
        throw(ArgumentError("multigrid (precond = :mg / :vanka, ablock = :mg or schur = :mg) needs the grid geometry of the " *
                            "matrix (krylov_csr_matrix(...; geometry))"))
    n = A.n
    hrp = Vector{Int32}(Array(A.rowptr))
    hcv = Vector{Int32}(Array(A.colval))
    _krylov_check_sorted(hrp, hcv, n)
    hd = _krylov_diag_positions(hrp, hcv, n)
    bpos_h = Int32[]
    bdiag = 0
    if A.border > 0 && tag.border === :krylov && !simple
        b = A.border
        bdiag = Int(hd[b])
        for i in 1:n, q in hrp[i]:(hrp[i+1]-1)
            q == bdiag && continue
            (i == b || hcv[q] == b) && push!(bpos_h, Int32(q))
        end
    end
    fwd = _KrylovSide(A.rowptr, A.colval, _dev_int(A.colval, hd), A.nzval, true; factor = !simple)
    simple && (fwd.simple = _krylov_block_build(hrp, hcv, A.nfields, A.pslot, A.nlocal, fwd.A, tag, A.geom))
    nr = simple ? 0 : n
    ratio = fill!(similar(A.nzval, nr), 0.0)
    basis = similar(A.nzval, n, tag.restart + 1)
    y = fill!(similar(A.nzval, n), 0.0)
    rhs = fill!(similar(A.nzval, n), 0.0)
    work = (; r = fill!(similar(A.nzval, n), 0.0), hbuf = fill!(similar(A.nzval, tag.restart + 1), 0.0),
            d = fill!(similar(A.nzval, tag.restart + 1), 0.0))
    f = KrylovFactor(tag, fwd, nothing, nothing, true, _dev_int(A.colval, bpos_h), bdiag,
                     ratio, zeros(Float64, nr), basis, y, rhs, work,
                     (; min_ratio = NaN, n_small = 0, row_min = 0), NamedTuple[], 0)
    _krylov_factor!(f)
    return LinearSolveCache(tag, f, A, A, 0, false, nothing)
end

# ---- device-generic kernels (KernelAbstractions: Array and CuArray) ----

@kernel function _krylov_zero_kernel!(F, @Const(pos))
    t = @index(Global)
    @inbounds F[pos[t]] = 0.0
end

@kernel function _krylov_pivot_kernel!(ratio, @Const(rowptr), @Const(diag), @Const(A), @Const(F))
    i = @index(Global)
    @inbounds begin
        amax = 0.0
        bad = false
        for q in rowptr[i]:(rowptr[i+1]-Int32(1))
            amax = max(amax, abs(A[q]))
            isfinite(F[q]) || (bad = true)
        end
        u = abs(F[diag[i]])
        ratio[i] = bad ? -1.0 : (amax > 0 ? u / amax : u)
    end
end

@kernel function _krylov_take_kernel!(dst, @Const(src), @Const(idx))
    t = @index(Global)
    @inbounds dst[t] = src[idx[t]]
end

@kernel function _krylov_put_kernel!(dst, @Const(src), @Const(idx))
    t = @index(Global)
    @inbounds dst[idx[t]] = src[t]
end

@kernel function _krylov_spmv_kernel!(w, @Const(rowptr), @Const(colval), @Const(A), @Const(z))
    i = @index(Global)
    @inbounds begin
        s = 0.0
        for q in rowptr[i]:(rowptr[i+1]-Int32(1))
            s += A[q] * z[colval[q]]
        end
        w[i] = s
    end
end

@kernel function _krylov_gather_kernel!(At, Ft, @Const(A), @Const(F), @Const(perm))
    q = @index(Global)
    @inbounds begin
        p = perm[q]
        At[q] = A[p]
        Ft[q] = F[p]
    end
end

# Pivot report of an ILU(0) factor held in `side.F` (raises on a zero or
# non-finite pivot). `label` names the factorized operator in the message.
_krylov_pivot_report!(ratio, ratio_host, side::_KrylovSide, tol::Float64, label::String) =
    _krylov_ratio_report!(ratio, ratio_host, side.rowptr, side.diag, side.A, side.F, side.n, tol,
                          "ILU(0) of $label")

# Same report on any (A, F) pair with one pattern; F = A gives the diagonal
# ratio |aᵢᵢ| / maxⱼ|aᵢⱼ| of A itself (P3: the Jacobi smoothers need aᵢᵢ ≠ 0).
function _krylov_ratio_report!(ratio, ratio_host, rowptr, diag, A, F, n::Int, tol::Float64,
                               label::String)
    be = KernelAbstractions.get_backend(A)
    _krylov_pivot_kernel!(be, 256)(ratio, rowptr, diag, A, F; ndrange = n)
    KernelAbstractions.synchronize(be)
    copyto!(ratio_host, ratio)
    rmin = Inf; imin = 0; nsmall = 0
    @inbounds for i in eachindex(ratio_host)
        r = ratio_host[i]
        r < rmin && (rmin = r; imin = i)
        0 <= r < tol && (nsmall += 1)
    end
    if rmin <= 0
        what = rmin < 0 ? "a non-finite factor entry" : "a zero pivot"
        throw(ErrorException("$label failed: $what in row $imin " *
                             "(size $n); use another preconditioner " *
                             "(precond = :ilu0 / :simple / :mg) or a direct backend for this case."))
    end
    return (; min_ratio = rmin, n_small = nsmall, row_min = imin)
end

function _krylov_factor!(f::KrylovFactor)
    side = f.fwd
    if side.simple !== nothing
        f.pivot = _simple_refresh!(side.simple, f.tag)
    else
        be = KernelAbstractions.get_backend(side.A)
        copyto!(side.F, side.A)
        if f.bdiag > 0
            nb = length(f.bpos)
            nb > 0 && _krylov_zero_kernel!(be, 256)(side.F, f.bpos; ndrange = nb)
            fill!(view(side.F, f.bdiag:f.bdiag), 1.0)
            KernelAbstractions.synchronize(be)
        end
        _krylov_ilu0!(side)
        f.pivot = _krylov_pivot_report!(f.ratio, f.ratio_host, side, f.tag.pivot_tol,
                                        "the full matrix (P1)")
        _krylov_rebuild!(side, f.tag)
    end
    f.tstale = true
    f.n_factor += 1
    return f
end

function _krylov_transpose_pattern(rowptr::Vector{Int32}, colval::Vector{Int32}, n::Int)
    nz = length(colval)
    cnt = zeros(Int, n)
    @inbounds for q in 1:nz
        cnt[colval[q]] += 1
    end
    rpt = Vector{Int32}(undef, n + 1)
    rpt[1] = 1
    @inbounds for i in 1:n
        rpt[i+1] = rpt[i] + cnt[i]
    end
    next = Vector{Int}(rpt[1:n])
    cvt = Vector{Int32}(undef, nz)
    perm = Vector{Int32}(undef, nz)
    @inbounds for i in 1:n, q in rowptr[i]:(rowptr[i+1]-1)
        j = colval[q]
        p = next[j]
        cvt[p] = i
        perm[p] = q
        next[j] = p + 1
    end
    return rpt, cvt, perm
end

# Transposed side, built on first use and refreshed after each refactorization.
function _krylov_tside!(f::KrylovFactor)
    side = f.fwd
    if f.tr === nothing
        hrp = Vector{Int32}(Array(side.rowptr))
        hcv = Vector{Int32}(Array(side.colval))
        rpt, cvt, perm = _krylov_transpose_pattern(hrp, hcv, side.n)
        hdt = _krylov_diag_positions(rpt, cvt, side.n)
        At = similar(side.A)
        simple = side.simple !== nothing
        f.tr = _KrylovSide(_dev_int(side.colval, rpt), _dev_int(side.colval, cvt),
                           _dev_int(side.colval, hdt), At, false; factor = !simple)
        if simple
            s = side.simple
            f.tr.simple = _krylov_block_build(rpt, cvt, s.nfields, s.pslot, s.nlocal, At, f.tag, s.geom)
        end
        f.perm = _dev_int(side.colval, perm)
        f.tstale = true
    end
    if f.tstale
        tr = f.tr
        be = KernelAbstractions.get_backend(side.A)
        if tr.simple !== nothing
            _krylov_take_kernel!(be, 256)(tr.A, side.A, f.perm; ndrange = length(tr.A))
            KernelAbstractions.synchronize(be)
            _simple_refresh!(tr.simple, f.tag)
        else
            _krylov_gather_kernel!(be, 256)(tr.A, tr.F, side.A, side.F, f.perm; ndrange = length(tr.A))
            KernelAbstractions.synchronize(be)
            _krylov_rebuild!(tr, f.tag)
        end
        f.tstale = false
    end
    return f.tr
end

# ---- backend hooks: CPU methods here, CuArray methods in the CUDSS extension ----

_krylov_noback(side) = throw(ArgumentError("KrylovBackendTag has no method for arrays of type " *
    "$(typeof(side.A)); on GPU load `using CUDA, CUDSS` (cuSPARSE ILU(0) / SpSV)"))
_krylov_ilu0!(side::_KrylovSide) = _krylov_noback(side)
_krylov_rebuild!(side::_KrylovSide, tag) = _krylov_noback(side)
_krylov_precond!(z, side::_KrylovSide, v) = _krylov_noback(side)

# SpMV on any KernelAbstractions backend (one thread per row): no workspace,
# no device allocation. The CPU method below is a plain loop.
function _krylov_matvec!(w, side::_KrylovSide, z)
    be = KernelAbstractions.get_backend(side.A)
    _krylov_spmv_kernel!(be, 256)(w, side.rowptr, side.colval, side.A, z; ndrange = side.n)
    KernelAbstractions.synchronize(be)
    return w
end

# Preconditioner application of a side: P2 when it carries a block SIMPLE
# preconditioner, P1 (ILU(0) triangular solves) otherwise.
_krylov_apply!(z, side::_KrylovSide, v) =
    side.simple === nothing ? _krylov_precond!(z, side, v) : _simple_apply!(z, side.simple, v)

# ILU(0), IKJ variant (Saad, Iterative Methods, Alg. 10.4), in place on F.
function _krylov_ilu0!(side::_KrylovSide{<:Vector,<:Vector})
    rp = side.rowptr; cv = side.colval; d = side.diag; F = side.F
    iw = zeros(Int32, side.n)
    @inbounds for i in 1:side.n
        for q in rp[i]:(rp[i+1]-1)
            iw[cv[q]] = q
        end
        for q in rp[i]:(d[i]-1)
            k = cv[q]
            F[q] /= F[d[k]]
            lik = F[q]
            for qq in (d[k]+1):(rp[k+1]-1)
                p = iw[cv[qq]]
                p != 0 && (F[p] -= lik * F[qq])
            end
        end
        for q in rp[i]:(rp[i+1]-1)
            iw[cv[q]] = 0
        end
    end
    return side
end

_krylov_rebuild!(side::_KrylovSide{<:Vector,<:Vector}, tag) = side

function _krylov_lower_cpu!(y, side, b, unit::Bool)
    rp = side.rowptr; cv = side.colval; d = side.diag; F = side.F
    @inbounds for i in 1:side.n
        s = b[i]
        for q in rp[i]:(d[i]-1)
            s -= F[q] * y[cv[q]]
        end
        y[i] = unit ? s : s / F[d[i]]
    end
    return y
end

function _krylov_upper_cpu!(x, side, y, unit::Bool)
    rp = side.rowptr; cv = side.colval; d = side.diag; F = side.F
    @inbounds for i in side.n:-1:1
        s = y[i]
        for q in (d[i]+1):(rp[i+1]-1)
            s -= F[q] * x[cv[q]]
        end
        x[i] = unit ? s : s / F[d[i]]
    end
    return x
end

function _krylov_precond!(z, side::_KrylovSide{<:Vector,<:Vector}, v)
    side.t0 .= v
    _krylov_lower_cpu!(side.t1, side, side.t0, side.first_unit)
    _krylov_upper_cpu!(side.t2, side, side.t1, !side.first_unit)
    z .= side.t2
    return z
end

function _krylov_matvec!(w, side::_KrylovSide{<:Vector,<:Vector}, z)
    rp = side.rowptr; cv = side.colval; A = side.A
    @inbounds for i in 1:side.n
        s = 0.0
        for q in rp[i]:(rp[i+1]-1)
            s += A[q] * z[cv[q]]
        end
        w[i] = s
    end
    return w
end

# w = A M⁻¹ v (right preconditioning: GMRES iterates on y, x = M⁻¹ y)
struct _KrylovOp{S}
    side::S
end
function (op::_KrylovOp)(w, v)
    s = op.side
    _krylov_apply!(s.z, s, v)
    _krylov_matvec!(w, s, s.z)
    return w
end

# ---- P2: block SIMPLE preconditioner ----

# Blocks of one operator (J, or the gathered Jᵀ), all referring to the values
# `J` by position: A (U × U, compact CSR with its own values for the ILU(0)),
# G (U × P), D (P × U), and the Schur approximation S̃ (P × P).
mutable struct _KrylovSimple{VI,VV,SA}
    nfields::Int
    pslot::Int
    nlocal::Int
    nu::Int
    np::Int
    J::VV                        # operator values (not owned)
    uidx::VI                     # global index of each U unknown
    pidx::VI                     # global index of each P unknown
    Ablk::SA                     # _KrylovSide (ILU(0), P2) or _KrylovMG (P3) of A
    posA::VI                     # J position of each A entry
    posAd::VI                    # J position of the diagonal of each U row
    Grp::VI; Gcv::VI; posG::VI   # G: rows U, compact P columns, J positions
    Drp::VI; Dcv::VI; posD::VI   # D: rows P, compact U columns, J positions
    Srp::VI; Scv::VI; posC::VI   # S̃ pattern; J position of the C entry, or 0
    Sval::VV
    schur::Any                   # LU object (`:lu`), _KrylovSide (`:ilu0`) or _KrylovMG (`:mg`)
    schurkind::Symbol
    geom::Any                    # host cell geometry (P3), or nothing
    ratio::VV
    ratio_host::Vector{Float64}
    vu::VV; uh::VV; u::VV        # U work vectors
    vp::VV; rp::VV; p::VV        # P work vectors
    info::NamedTuple
    bs::Any                      # block structure / Schur approximation data (linear_solve_block_schur.jl), or nothing
end

# Values of the velocity block, and x ≈ A⁻¹ v with the velocity-block solver
# (the _KrylovMG methods are in linear_solve_multigrid.jl).
_simple_avals(b::_KrylovSide) = b.A
_simple_ablk_solve!(x, b::_KrylovSide, v) = _krylov_precond!(x, b, v)

function _simple_build(hrp::Vector{Int32}, hcv::Vector{Int32}, nfields::Int, pslot::Int,
                       nlocal::Int, J, tag::KrylovBackendTag, geom = nothing)
    n = length(hrp) - 1
    kind = zeros(Int8, n)                      # 1 = U, 2 = P, 0 = extra
    cidx = zeros(Int32, n)
    nu = 0; np = 0
    @inbounds for i in 1:nlocal
        if (i - 1) % nfields + 1 == pslot
            np += 1; kind[i] = 2; cidx[i] = np
        else
            nu += 1; kind[i] = 1; cidx[i] = nu
        end
    end
    (nu > 0 && np > 0) || throw(ArgumentError("precond = :simple needs velocity and pressure unknowns"))
    uidx = Int32[i for i in 1:n if kind[i] == 1]
    pidx = Int32[i for i in 1:n if kind[i] == 2]
    Arp = Int32[1]; Acv = Int32[]; posA = Int32[]; Adiag = zeros(Int32, nu); posAd = zeros(Int32, nu)
    Grp = Int32[1]; Gcv = Int32[]; posG = Int32[]
    @inbounds for (a, r) in enumerate(uidx)
        for q in hrp[r]:(hrp[r+1]-1)
            c = hcv[q]; k = kind[c]
            if k == 1
                push!(Acv, cidx[c]); push!(posA, q)
                c == r && (Adiag[a] = length(Acv); posAd[a] = q)
            elseif k == 2
                push!(Gcv, cidx[c]); push!(posG, q)
            end
        end
        Adiag[a] == 0 && throw(ArgumentError("velocity row $r has no stored diagonal"))
        push!(Arp, length(Acv) + 1); push!(Grp, length(Gcv) + 1)
    end
    Drp = Int32[1]; Dcv = Int32[]; posD = Int32[]
    Srp = Int32[1]; Scv = Int32[]; posC = Int32[]
    mark = zeros(Int32, np); cpos = zeros(Int32, np); cols = Int32[]
    @inbounds for (i, r) in enumerate(pidx)
        empty!(cols)
        d0 = length(Dcv) + 1
        for q in hrp[r]:(hrp[r+1]-1)
            c = hcv[q]; k = kind[c]
            if k == 1
                push!(Dcv, cidx[c]); push!(posD, q)
            elseif k == 2
                kc = cidx[c]
                mark[kc] != i && (mark[kc] = i; push!(cols, kc))
                cpos[kc] = q
            end
        end
        for q in d0:length(Dcv), qq in Grp[Dcv[q]]:(Grp[Dcv[q]+1]-1)
            kc = Gcv[qq]
            mark[kc] != i && (mark[kc] = i; push!(cols, kc); cpos[kc] = 0)
        end
        push!(Drp, length(Dcv) + 1)
        sort!(cols)
        for kc in cols
            push!(Scv, kc); push!(posC, cpos[kc])
        end
        push!(Srp, length(Scv) + 1)
    end
    devi(h) = copyto!(similar(J, Int32, length(h)), h)
    vec(m) = fill!(similar(J, m), 0.0)
    schurkind = _krylov_schur_kind(tag)
    akind = _krylov_ablock_kind(tag)
    if akind === :mg
        # P3: aggregation multigrid of A, aggregates per (2×2×2 block, component, class)
        Avals = vec(length(Acv))
        Ablk = _mg_build(Arp, Acv, Avals,
                         _mg_nodes(Int32[(r - 1) ÷ nfields + 1 for r in uidx],
                                   Int32[(r - 1) % nfields + 1 for r in uidx], geom), tag)
    elseif akind === :lu
        # exact velocity block (linear_solve_block_schur.jl)
        Ablk = _KrylovALU(nu, devi(Arp), devi(Acv), devi(Adiag), vec(length(Acv)), nothing)
    else
        Ablk = _KrylovSide(devi(Arp), devi(Acv), devi(Adiag), vec(length(Acv)), true)
    end
    Sval = vec(length(Scv))
    Srp_d = devi(Srp); Scv_d = devi(Scv)
    schur = nothing
    if schurkind === :ilu0
        schur = _KrylovSide(Srp_d, Scv_d, devi(_krylov_diag_positions(Srp, Scv, np)), Sval, true)
    elseif schurkind === :mg
        schur = _mg_build(Srp, Scv, Sval,
                          _mg_nodes(Int32[(r - 1) ÷ nfields + 1 for r in pidx],
                                    fill(Int32(pslot), np), geom), tag)
    end
    info = (; nu, np, nnz_A = length(Acv), nnz_G = length(Gcv), nnz_D = length(Dcv),
            nnz_S = length(Scv), schur = schurkind,
            mg_A = Ablk isa _KrylovMG ? Ablk.info : nothing,
            mg_S = schurkind === :mg ? schur.info : nothing,
            ablock = akind, block = tag.block, schur_approx = tag.schur_approx,
            schur_q = _bs_qkind(tag), schur_shift = tag.schur_shift)
    bs = _bs_active(tag) ? _bs_build(hrp, hcv, nfields, pslot, nlocal, nu, np, J, tag) : nothing
    return _KrylovSimple(nfields, pslot, nlocal, nu, np, J, devi(uidx), devi(pidx), Ablk,
                         devi(posA), devi(posAd), devi(Grp), devi(Gcv), devi(posG),
                         devi(Drp), devi(Dcv), devi(posD), Srp_d, Scv_d, devi(posC), Sval,
                         schur, schurkind, geom, vec(nu), zeros(Float64, nu),
                         vec(nu), vec(nu), vec(nu), vec(np), vec(np), vec(np), info, bs)
end

# S̃ = C − D diag(A)⁻¹ G, one thread per pressure row (sorted S̃ row, binary
# search of each product column; no stored triple products).
@kernel function _simple_schur_kernel!(Sval, @Const(J), @Const(Srp), @Const(Scv), @Const(posC),
                                       @Const(Drp), @Const(Dcv), @Const(posD),
                                       @Const(Grp), @Const(Gcv), @Const(posG), @Const(posAd))
    i = @index(Global)
    @inbounds begin
        s0 = Srp[i]; s1 = Srp[i+1] - Int32(1)
        for s in s0:s1
            c = posC[s]
            Sval[s] = c > 0 ? J[c] : 0.0
        end
        for q in Drp[i]:(Drp[i+1]-Int32(1))
            j = Dcv[q]
            dj = J[posD[q]] / J[posAd[j]]
            for qq in Grp[j]:(Grp[j+1]-Int32(1))
                k = Gcv[qq]
                lo = s0; hi = s1
                while lo < hi
                    mid = (lo + hi) >> Int32(1)
                    if Scv[mid] < k
                        lo = mid + Int32(1)
                    else
                        hi = mid
                    end
                end
                Sval[lo] -= dj * J[posG[qq]]
            end
        end
    end
end

# rp = vp − D û
@kernel function _simple_dmul_kernel!(rp, @Const(vp), @Const(J), @Const(Drp), @Const(Dcv),
                                      @Const(posD), @Const(uh))
    i = @index(Global)
    @inbounds begin
        s = vp[i]
        for q in Drp[i]:(Drp[i+1]-Int32(1))
            s -= J[posD[q]] * uh[Dcv[q]]
        end
        rp[i] = s
    end
end

# u = û − diag(A)⁻¹ G p
@kernel function _simple_gcorr_kernel!(u, @Const(uh), @Const(J), @Const(Grp), @Const(Gcv),
                                       @Const(posG), @Const(posAd), @Const(p))
    a = @index(Global)
    @inbounds begin
        s = 0.0
        for q in Grp[a]:(Grp[a+1]-Int32(1))
            s += J[posG[q]] * p[Gcv[q]]
        end
        u[a] = uh[a] - s / J[posAd[a]]
    end
end

# New values of J: ILU(0) of A (pivot report), S̃ assembled and factorized.
function _simple_refresh!(s::_KrylovSimple, tag::KrylovBackendTag)
    be = KernelAbstractions.get_backend(s.J)
    A = s.Ablk
    Avals = _simple_avals(A)
    _krylov_take_kernel!(be, 256)(Avals, s.J, s.posA; ndrange = length(Avals))
    s.bs === nothing &&
        _simple_schur_kernel!(be, 256)(s.Sval, s.J, s.Srp, s.Scv, s.posC, s.Drp, s.Dcv, s.posD,
                                       s.Grp, s.Gcv, s.posG, s.posAd; ndrange = s.np)
    KernelAbstractions.synchronize(be)
    s.bs === nothing || _bs_schur!(s)        # Q choice, shift, PCD F_p (linear_solve_block_schur.jl)
    if A isa _KrylovSide
        copyto!(A.F, A.A)
        _krylov_ilu0!(A)
        pivot = _krylov_pivot_report!(s.ratio, s.ratio_host, A, tag.pivot_tol,
                                      "the velocity block (P2)")
        _krylov_rebuild!(A, tag)
    elseif A isa _KrylovALU
        pivot = _krylov_ratio_report!(s.ratio, s.ratio_host, A.rowptr, A.diag, A.A, A.A,
                                      A.n, tag.pivot_tol, "the velocity-block diagonal (exact LU)")
        _bs_alu_refresh!(A)
    else
        # P3: diagonal ratio of A (the Jacobi smoothers divide by it), then the hierarchy
        L1 = A.levels[1]
        pivot = _krylov_ratio_report!(s.ratio, s.ratio_host, L1.rowptr, L1.diag, L1.A, L1.A,
                                      L1.n, tag.pivot_tol, "the velocity-block diagonal (P3)")
        _mg_refresh!(A, tag)
    end
    if s.schurkind === :ilu0
        S = s.schur::_KrylovSide
        copyto!(S.F, S.A)
        _krylov_ilu0!(S)
        _krylov_rebuild!(S, tag)
    elseif s.schurkind === :mg
        _mg_refresh!(s.schur, tag)
    elseif s.schur === nothing
        s.schur = _simple_schur_lu(s.Srp, s.Scv, s.Sval, s.np)
    else
        _simple_schur_refactor!(s.schur)
    end
    return pivot
end

function _simple_apply!(z, s::_KrylovSimple, v)
    s.bs === nothing || return _bs_apply!(z, s, v)           # linear_solve_block_schur.jl
    be = KernelAbstractions.get_backend(z)
    z .= v                                                   # extras: identity
    _krylov_take_kernel!(be, 256)(s.vu, z, s.uidx; ndrange = s.nu)
    _krylov_take_kernel!(be, 256)(s.vp, z, s.pidx; ndrange = s.np)
    KernelAbstractions.synchronize(be)
    _simple_ablk_solve!(s.uh, s.Ablk, s.vu)                  # û = Ã⁻¹ v_U (ILU(0) or MG)
    _simple_dmul_kernel!(be, 256)(s.rp, s.vp, s.J, s.Drp, s.Dcv, s.posD, s.uh; ndrange = s.np)
    KernelAbstractions.synchronize(be)
    if s.schur isa _KrylovSide
        _krylov_precond!(s.p, s.schur, s.rp)                 # p ≈ S̃⁻¹ r_P (ILU(0))
    elseif s.schurkind === :mg
        _mg_apply!(s.p, s.schur, s.rp)                       # p ≈ S̃⁻¹ r_P (one MG cycle)
    else
        _simple_schur_solve!(s.p, s.schur, s.rp)             # p = S̃⁻¹ r_P (LU)
    end
    _simple_gcorr_kernel!(be, 256)(s.u, s.uh, s.J, s.Grp, s.Gcv, s.posG, s.posAd, s.p; ndrange = s.nu)
    KernelAbstractions.synchronize(be)
    _krylov_put_kernel!(be, 256)(z, s.u, s.uidx; ndrange = s.nu)
    _krylov_put_kernel!(be, 256)(z, s.p, s.pidx; ndrange = s.np)
    KernelAbstractions.synchronize(be)
    return z
end

# Schur LU hooks: UMFPACK here (host arrays), cuDSS in the CUDSS extension.
_simple_schur_lu(Srp, Scv, Sval, np) =
    throw(ArgumentError("schur = :lu has no method for arrays of type $(typeof(Sval)); " *
                        "on GPU load `using CUDA, CUDSS`, or use schur = :ilu0"))

# S̃ is held in CSC form (the transposed CSR pattern), refreshed by a gather.
struct _SchurCPU{F}
    S::SparseMatrixCSC{Float64,Int}
    F::F
    perm::Vector{Int}
    Sval::Vector{Float64}
end

function _simple_schur_lu(Srp::Vector{Int32}, Scv::Vector{Int32}, Sval::Vector{Float64}, np::Int)
    rpt, cvt, perm32 = _krylov_transpose_pattern(Srp, Scv, np)
    perm = Vector{Int}(perm32)
    S = SparseMatrixCSC{Float64,Int}(np, np, Vector{Int}(rpt), Vector{Int}(cvt), Sval[perm])
    return _SchurCPU(S, lu(S), perm, Sval)
end

function _simple_schur_refactor!(c::_SchurCPU)
    nz = nonzeros(c.S)
    @inbounds for q in eachindex(nz)
        nz[q] = c.Sval[c.perm[q]]
    end
    lu!(c.F, c.S)
    return c
end

_simple_schur_solve!(x, c::_SchurCPU, b) = ldiv!(x, c.F, b)

"""
    lin_krylov_blocks(cache) -> NamedTuple or nothing

Block sizes of a P2 / P3 (`precond = :simple` / `:mg`) Krylov cache: `(; nu,
np, nnz_A, nnz_G, nnz_D, nnz_S, schur, mg_A, mg_S)` (velocity-like and pressure
unknown counts, stored entries of each block and of the Schur approximation
S̃, the Schur treatment used, and the multigrid hierarchy of A and of S̃ —
`(; levels, sizes, nnz, operator_complexity, grid_complexity, cycle,
smoother, omega, sweeps, cycles, scale)` — or `nothing`), and the block options
in use (`ablock`, `block`, `schur_approx`, `schur_q`, `schur_shift`); `nothing`
for P1.
"""
lin_krylov_blocks(cache::LinearSolveCache) =
    cache.factor.fwd.simple === nothing ? nothing : cache.factor.fwd.simple.info

# ---- seam methods ----

_krylov_ishost(a) = a isa Array || (a isa SubArray && parent(a) isa Array)

function _krylov_copy!(dst, src)
    if _krylov_ishost(dst) == _krylov_ishost(src)
        dst .= src
    elseif _krylov_ishost(src)
        copyto!(dst, src isa Array ? src : collect(src))
    else
        dst .= Array(src)
    end
    return dst
end

function _krylov_solve_vec!(f::KrylovFactor, x, b, transpose::Bool)
    side = transpose ? _krylov_tside!(f) : f.fwd
    tol = transpose ? f.tag.rtol_transpose : f.tag.rtol
    _krylov_copy!(f.rhs, b)
    fill!(f.y, 0.0)
    t0 = time()
    info = gnf_gmres!(f.y, _KrylovOp(side), f.rhs; tol, restart = f.tag.restart,
                      maxiter = f.tag.maxiter, orth = f.tag.orth, basis = f.basis,
                      work = f.work)
    _krylov_apply!(side.z, side, f.y)
    _krylov_copy!(x, side.z)
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(side.z))
    seconds = time() - t0
    push!(f.log, (; transpose, iters = info.iters, cycles = info.cycles,
                  relres = info.relres, converged = info.converged, seconds))
    if f.tag.strict && !info.converged
        throw(ErrorException("GMRES ($(transpose ? "transposed" : "forward") solve) did not " *
                             "reach rtol = $tol in $(info.iters) iterations " *
                             "(relative residual $(info.relres))"))
    end
    return x
end

function _lin_solve_into!(::KrylovBackendTag, x::AbstractVecOrMat, cache::LinearSolveCache,
                          b::AbstractVecOrMat, transpose::Bool)
    f = cache.factor
    if b isa AbstractVector
        _krylov_solve_vec!(f, x, b, transpose)
    else
        for j in axes(b, 2)
            _krylov_solve_vec!(f, view(x, :, j), view(b, :, j), transpose)
        end
    end
    return x
end

function lin_solve!(tag::KrylovBackendTag, cache::LinearSolveCache, b::AbstractVector)
    x = _lin_solution_buffer(tag, cache, b)
    return _lin_solve_into!(tag, x, cache, b, false)
end

_lin_solution_buffer(::KrylovBackendTag, cache::LinearSolveCache, b::AbstractVecOrMat) =
    similar(cache.A.nzval, size(b))

_lin_supports_transpose(::KrylovBackendTag, cache::LinearSolveCache) = true

function _lin_refactorize!(::KrylovBackendTag, cache::LinearSolveCache, A_new)
    A = cache.A
    if A_new !== nothing
        A_new isa KrylovCSRMatrix && A_new.n == A.n && length(A_new.colval) == length(A.colval) ||
            throw(ArgumentError("lin_refactorize!: the new matrix must be a KrylovCSRMatrix with the same pattern"))
        if A_new.nzval !== A.nzval
            Array(A_new.rowptr) == Array(A.rowptr) && Array(A_new.colval) == Array(A.colval) ||
                throw(ArgumentError("lin_refactorize!: sparsity pattern changed; call lin_factorize again"))
            copyto!(A.nzval, A_new.nzval)
        end
    end
    _krylov_factor!(cache.factor)
    return cache
end

"""
    lin_krylov_log(cache) -> Vector{NamedTuple}

Per-solve record of a Krylov cache: `(; transpose, iters, cycles, relres,
converged, seconds)`, one entry per right-hand side, in call order.
"""
lin_krylov_log(cache::LinearSolveCache) = cache.factor.log

"""
    lin_krylov_pivot(cache) -> NamedTuple

Pivot report of the last ILU(0) factorization: `(; min_ratio, n_small,
row_min)` with the ratio |Uᵢᵢ| / maxⱼ|Aᵢⱼ| per row (`n_small` rows below the
tag's `pivot_tol`).
"""
lin_krylov_pivot(cache::LinearSolveCache) = cache.factor.pivot
