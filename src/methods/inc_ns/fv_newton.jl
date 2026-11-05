# Newton solver on the steady colocated FV residual of fv_steady_residual.jl.
# Roadmap "differentiable FV + cuDSS", capability C2:
#   (a) sparsity pattern of dR/dU built directly from the known stencil (CPU,
#       once per geometry) and a column colouring of it (SparseMatrixColorings,
#       CPU, once);
#   (b) the Jacobian VALUES assembled by evaluating the C1 residual kernel on
#       ForwardDiff duals, chunk by chunk of colours, and decompressed straight
#       into the value vector of the solver's sparse matrix (CSC on CPU,
#       CSR on cuDSS) — three generic KernelAbstractions kernels, the same code
#       on the CPU and GPU backends;
#   (c) Newton with Armijo backtracking and an optional bounded pseudo-transient
#       continuation; the linear systems go through the factorize-once seam
#       (src/solve/linear_solve.jl): ONE `lin_factorize` (symbolic analysis +
#       numeric factorization) on the first Newton step, then
#       `lin_refactorize!` (numeric only, same pattern) and the in-place
#       allocation-free `lin_solve!`.
# The per-iteration GPU loop allocates nothing on the device (residual norms are
# reduced by a kernel into a preallocated buffer of per-thread partial sums,
# copied to a preallocated host buffer).

import ForwardDiff
import SparseMatrixColorings
using SparseArrays
using KernelAbstractions

# ------------------------------------------------------------------ pattern
"""
    fv_jacobian_pattern(prob::FVSteadyProblem2D) -> SparseMatrixCSC{Bool,Int}

Structural sparsity pattern of `∂R/∂U` for the steady FV residual of
[`fv_residual!`](@ref), built directly from the stencil (no tracing; CPU, once
per geometry, host `prob`). It follows the residual's own access paths through
the face classification (walls, outlets, periodic wrap, solid cells), so it is a
superset of the support for every state, viscosity field and value of `conv`
(Stokes and Navier–Stokes share one pattern, which keeps the cuDSS analysis
valid across a Stokes → Navier–Stokes continuation). For a fluid cell `P`:

- momentum row `u`: `u` at `P` and its face neighbours; `v` at `P`, its face
  neighbours and the east/west neighbours of its north/south neighbours (the
  `∂v/∂x` cross term); `p` at `P`, its face neighbours and the second
  neighbours along each axis (Rhie–Chow mean-gradient term);
- momentum row `v`: the same with `u` and `v` swapped and the cross term taken
  through the east/west neighbours;
- continuity row: `u` at `P` and its east/west neighbours, `v` at `P` and its
  north/south neighbours, `p` as in the momentum rows;
- pinned continuity row: `p_P` only; solid cell: identity.

Interior rows carry 23, 23 and 15 entries (20.3 per row on a large grid). The
pattern is the Newtonian one: a viscosity that depends on the state (C3)
widens it.
"""
function fv_jacobian_pattern(prob::FVSteadyProblem2D)
    g = prob.grid
    tag = Array(prob.tag)
    Nx, Ny = g.Nx, g.Ny
    n = fv_nunknowns(g)
    I = Int[]; J = Int[]
    sizehint!(I, 21 * n); sizehint!(J, 21 * n)
    # neighbour through face d, or (0, 0) when the face is not interior
    function nb(i, j, d)
        i == 0 && return (0, 0)
        kind, ii, jj = _fv_face_kind(g, tag, i, j, d)
        return kind == _FV_F_INT ? (ii, jj) : (0, 0)
    end
    push(r, c, comp) = (c[1] != 0 && (push!(I, r); push!(J, fv_index(g, c[1], c[2], comp))); nothing)
    for j in 1:Ny, i in 1:Nx
        P = (i, j)
        ru = fv_index(g, i, j, 1); rv = ru + 1; rc = ru + 2
        if tag[i, j] == FV_CELL_SOLID
            push!(I, ru, rv, rc); push!(J, ru, rv, rc)
            continue
        end
        E = nb(i, j, 1); W = nb(i, j, 2); N = nb(i, j, 3); S = nb(i, j, 4)
        star = (P, E, W, N, S)
        axis2 = (nb(E..., 1), nb(W..., 2), nb(N..., 3), nb(S..., 4))
        # cross terms: Ru reads v at the E/W neighbours of N and S, Rv reads u
        # at the N/S neighbours of E and W
        cross_u = (nb(N..., 1), nb(N..., 2), nb(S..., 1), nb(S..., 2))
        cross_v = (nb(E..., 3), nb(E..., 4), nb(W..., 3), nb(W..., 4))
        for c in star
            push(ru, c, 1); push(ru, c, 2); push(ru, c, 3)
            push(rv, c, 1); push(rv, c, 2); push(rv, c, 3)
        end
        for c in axis2
            push(ru, c, 3); push(rv, c, 3)
        end
        for c in cross_u
            push(ru, c, 2)
        end
        for c in cross_v
            push(rv, c, 1)
        end
        if tag[i, j] == FV_CELL_PINNED
            push(rc, P, 3)
        else
            for c in (P, E, W)
                push(rc, c, 1)
            end
            for c in (P, N, S)
                push(rc, c, 2)
            end
            for c in star
                push(rc, c, 3)
            end
            for c in axis2
                push(rc, c, 3)
            end
        end
    end
    return sparse(I, J, trues(length(I)), n, n, |)
end

"""
    fv_jacobian_coloring(S::SparseMatrixCSC) -> (colors::Vector{Int}, ncolors::Int)

Column colouring of the pattern `S` for compressed forward-mode assembly
(columns of one colour share no row): SparseMatrixColorings greedy colouring
with the largest-first order, CPU, once per geometry.
"""
function fv_jacobian_coloring(S::SparseMatrixCSC)
    prob = SparseMatrixColorings.ColoringProblem(; structure = :nonsymmetric,
                                                 partition = :column)
    res = SparseMatrixColorings.coloring(S, prob,
        SparseMatrixColorings.GreedyColoringAlgorithm(SparseMatrixColorings.LargestFirst()))
    return SparseMatrixColorings.column_colors(res), SparseMatrixColorings.ncolors(res)
end

# ------------------------------------------------------------------ assembly
"""
    FVJacobianTag

ForwardDiff tag of the duals used by the coloured Jacobian assembly.
"""
struct FVJacobianTag end

"""
    FVJacobianAssembler

Device buffers of the coloured dual assembly of `∂R/∂U` (build with
[`fv_jacobian_assembler`](@ref)): per stored entry, in the storage order of the
target matrix, its row `nzrow` and column `nzcol`; the colour of every unknown;
the dual state / residual buffers `Ud`, `Rd` (`Dual{FVJacobianTag,T,chunk}`);
`diag`, the storage position of each row's diagonal entry (pseudo-transient
shift). `ncolors / chunk` residual evaluations fill the whole matrix.
"""
struct FVJacobianAssembler{N,VI,VD}
    n::Int
    nnz::Int
    ncolors::Int
    nzrow::VI
    nzcol::VI
    colors::VI
    diag::VI
    Ud::VD
    Rd::VD
end

fv_jacobian_chunk(::FVJacobianAssembler{N}) where {N} = N

# (rows, cols) of the stored entries of S in CSC order (`:csc`) or in the order
# of the CSR arrays of the same matrix (`:csr`); also the CSR rowptr / colval.
function _fv_storage_order(S::SparseMatrixCSC, order::Symbol)
    n = size(S, 1)
    if order === :csc
        rows = copy(rowvals(S))
        cols = similar(rows)
        for c in 1:n, q in nzrange(S, c)
            cols[q] = c
        end
        return rows, cols, nothing, nothing
    elseif order === :csr
        St = sparse(transpose(S))           # CSC of Sᵀ == CSR of S
        rowptr = St.colptr
        cols = copy(rowvals(St))
        rows = similar(cols)
        for r in 1:n, q in nzrange(St, r)
            rows[q] = r
        end
        return rows, cols, rowptr, cols
    else
        throw(ArgumentError("storage order must be :csc or :csr, got :$order"))
    end
end

function _fv_diag_positions(rows, cols, n)
    d = zeros(Int, n)
    for q in eachindex(rows)
        rows[q] == cols[q] && (d[rows[q]] = q)
    end
    all(>(0), d) || error("fv Jacobian pattern lacks a diagonal entry")
    return d
end

"""
    fv_jacobian_assembler(S, colors, ncolors; to = Array, order = :csc,
                          chunk = 4, T = Float64) -> FVJacobianAssembler

Assembler of the coloured Jacobian of pattern `S` (from
[`fv_jacobian_pattern`](@ref)) with the colouring of
[`fv_jacobian_coloring`](@ref). `to` moves the index arrays to the device of the
state (`Array`, `CuArray`); `order` is the storage order of the target value
vector (`:csc` for a `SparseMatrixCSC`, `:csr` for a CSR device matrix);
`chunk` is the number of colours per dual evaluation (J0, A100/H100: 4 is the
fastest of 2/4/8/16 on Float64 duals).
"""
function fv_jacobian_assembler(S::SparseMatrixCSC, colors::AbstractVector{<:Integer},
                               ncolors::Integer; to = Array, order::Symbol = :csc,
                               chunk::Int = 4, T::Type = Float64)
    n = size(S, 1)
    rows, cols, _, _ = _fv_storage_order(S, order)
    diag = _fv_diag_positions(rows, cols, n)
    D = ForwardDiff.Dual{FVJacobianTag,T,chunk}
    i32(v) = to(Int32.(v))
    Ud = to(Vector{D}(undef, n))
    Rd = to(Vector{D}(undef, n))
    return FVJacobianAssembler{chunk,typeof(i32(rows)),typeof(Ud)}(
        n, length(rows), Int(ncolors), i32(rows), i32(cols), i32(colors), i32(diag), Ud, Rd)
end

@kernel function _fv_seed_kernel!(Ud, @Const(U), @Const(colors), offset, ::Val{N}) where {N}
    k = @index(Global)
    @inbounds begin
        c = colors[k] - offset
        T = eltype(U)
        Ud[k] = ForwardDiff.Dual{FVJacobianTag}(U[k],
                    ForwardDiff.Partials(ntuple(m -> m == c ? one(T) : zero(T), Val(N))))
    end
end

@kernel function _fv_decompress_kernel!(nzval, @Const(nzrow), @Const(nzcol), @Const(colors),
                                        @Const(Rd), offset, ::Val{N}) where {N}
    q = @index(Global)
    @inbounds begin
        c = colors[nzcol[q]] - offset
        if 1 <= c <= N
            nzval[q] = ForwardDiff.partials(Rd[nzrow[q]])[c]
        end
    end
end

@kernel function _fv_diag_shift_kernel!(nzval, @Const(diag), sigma)
    c = @index(Global)          # cell index; shift the two momentum rows
    @inbounds begin
        b = 3 * (c - 1)
        nzval[diag[b + 1]] += sigma
        nzval[diag[b + 2]] += sigma
    end
end

"""
    fv_assemble_jacobian!(nzval, asm, U, prob; sync = true) -> nzval

Write the values of `∂R/∂U` at the state `U` into `nzval` (the value vector of
the sparse matrix whose storage order `asm` was built for: `nonzeros(J)` of a
`SparseMatrixCSC`, `J.nzVal` of a cuDSS CSR matrix). `ceil(ncolors / chunk)`
passes of three kernels (seed, C1 residual kernel on duals, decompression); the
residual is the C1 kernel itself, so the matrix is the exact Jacobian of the
residual evaluated by [`fv_residual!`](@ref). Allocation-free on the device.
"""
function fv_assemble_jacobian!(nzval, asm::FVJacobianAssembler{N}, U,
                               prob::FVSteadyProblem2D; sync::Bool = true) where {N}
    length(U) == asm.n || throw(DimensionMismatch("length(U) = $(length(U)), expected $(asm.n)"))
    length(nzval) == asm.nnz || throw(DimensionMismatch("length(nzval) = $(length(nzval)), expected $(asm.nnz)"))
    backend = KernelAbstractions.get_backend(U)
    seed! = _fv_seed_kernel!(backend, 256)
    dec! = _fv_decompress_kernel!(backend, 256)
    for offset in 0:N:(asm.ncolors - 1)
        seed!(asm.Ud, U, asm.colors, offset, Val(N); ndrange = asm.n)
        fv_residual!(asm.Rd, asm.Ud, prob; sync = false)
        dec!(nzval, asm.nzrow, asm.nzcol, asm.colors, asm.Rd, offset, Val(N); ndrange = asm.nnz)
    end
    sync && KernelAbstractions.synchronize(backend)
    return nzval
end

# ------------------------------------------------------------------ residual norms
@kernel function _fv_norm_partials_kernel!(Pt, @Const(R), ncell, per)
    t = @index(Global)
    T = eltype(Pt)
    m2 = zero(T); c2 = zero(T); mi = zero(T); ci = zero(T)
    c0 = (t - 1) * per + 1
    c1 = min(t * per, ncell)
    @inbounds for c in c0:c1
        b = 3 * (c - 1)
        ru = R[b + 1]; rv = R[b + 2]; rc = R[b + 3]
        m2 += ru * ru + rv * rv
        c2 += rc * rc
        mi = max(mi, abs(ru), abs(rv))
        ci = max(ci, abs(rc))
    end
    @inbounds begin
        Pt[1, t] = m2; Pt[2, t] = c2; Pt[3, t] = mi; Pt[4, t] = ci
    end
end

struct _FVNormBuffers{D,H}
    dev::D
    host::H
    per::Int
end

function _FVNormBuffers(to, ncell::Int; per::Int = 256)
    nt = cld(ncell, per)
    return _FVNormBuffers(to(zeros(Float64, 4, nt)), zeros(Float64, 4, nt), per)
end

# (‖R_momentum‖₂, ‖R_continuity‖₂, ‖R‖∞): one kernel + one copy of the
# 4 × ncell/256 partial sums to the host; deterministic, no device allocation.
function _fv_norms(nb::_FVNormBuffers, R)
    ncell = length(R) ÷ 3
    backend = KernelAbstractions.get_backend(R)
    _fv_norm_partials_kernel!(backend, 64)(nb.dev, R, ncell, nb.per; ndrange = size(nb.dev, 2))
    KernelAbstractions.synchronize(backend)
    H = nb.dev isa Array ? nb.dev : copyto!(nb.host, nb.dev)
    m2 = 0.0; c2 = 0.0; inf = 0.0
    @inbounds for t in axes(H, 2)
        m2 += H[1, t]; c2 += H[2, t]
        inf = max(inf, H[3, t], H[4, t])
    end
    return sqrt(m2), sqrt(c2), inf
end

# ------------------------------------------------------------------ solver
"""
    FVNewtonSolver

State of the Newton solver of the steady FV residual (build with
[`fv_newton_solver`](@ref)): the problem on its device, the
[`FVJacobianAssembler`](@ref), the Jacobian matrix `J` (a
`SparseMatrixCSC{Float64,Int}` on the CPU seam, a `CuSparseMatrixCSR{Float64,Int32}`
on cuDSS) whose values the assembler overwrites in place, the linear-solve
cache (`nothing` until the first factorization), work vectors and counters:
`n_factorize` (symbolic analysis + factorization, 1 after any number of
Newton iterations on one geometry) and `n_refactorize` (numeric only).
"""
mutable struct FVNewtonSolver{P,A,M,V,NB,B}
    prob::P
    asm::A
    J::M
    nzval::V
    linsolve::B
    cache::Union{Nothing,LinearSolveCache}
    R::V
    dU::V
    Ut::V
    norms::NB
    n_factorize::Int
    n_refactorize::Int
    t_pattern::Float64
    t_coloring::Float64
    # (shift, ΣU, ΣU²) of the last assembled J and of the J held by `cache`;
    # NaN = unknown/stale. Lets the adjoint refuse factors that are not the
    # unshifted J at the given U (finalize = false).
    asm_sig::NTuple{3,Float64}
    fac_sig::NTuple{3,Float64}
end

# Reuse the norm workspace for the two state checksums. GPU `sum` allocates
# reduction storage; these fixed partial sums keep finalization allocation-free.
@kernel function _fv_state_sig_kernel!(Pt, @Const(U), ncell, per)
    t = @index(Global)
    su = 0.0; sq = 0.0
    k0 = 3 * (t - 1) * per + 1
    k1 = 3 * min(t * per, ncell)
    @inbounds for k in k0:k1
        u = Float64(U[k])
        su += u
        sq += u * u
    end
    @inbounds begin
        Pt[1, t] = su
        Pt[2, t] = sq
    end
end

function _fv_state_sig(nb::_FVNormBuffers, shift, U)
    ncell = length(U) ÷ 3
    backend = KernelAbstractions.get_backend(U)
    _fv_state_sig_kernel!(backend, 64)(nb.dev, U, ncell, nb.per; ndrange = size(nb.dev, 2))
    KernelAbstractions.synchronize(backend)
    H = nb.dev isa Array ? nb.dev : copyto!(nb.host, nb.dev)
    su = 0.0; sq = 0.0
    @inbounds for t in axes(H, 2)
        su += H[1, t]
        sq += H[2, t]
    end
    return Float64(shift), su, sq
end

"""
    _fv_device_csr(tag, rowptr, colval, n) -> (J, nzval)

Backend hook: build the device CSR matrix with the given (host, 1-based)
pattern and zero values, and return it with its value vector. Implemented for
`CUDABackendTag()` by `ext/KrakenCUDSSExt.jl` (`using CUDA, CUDSS`).
"""
_fv_device_csr(tag::LinearSolveBackend, rowptr, colval, n) =
    throw(ArgumentError("no FV Jacobian matrix for the $(_lin_backend_name(tag)) backend " *
                        "(CUDABackendTag needs `using CUDA, CUDSS`)"))

"""
    fv_newton_solver(prob::FVSteadyProblem2D; to = Array,
                     linsolve = CPUBackendTag(), chunk = 4) -> FVNewtonSolver

Set up the Newton solver for the HOST problem `prob`: pattern
([`fv_jacobian_pattern`](@ref)) and colouring ([`fv_jacobian_coloring`](@ref))
on the CPU, then the problem, the assembler and the Jacobian on the device of
`to` (`Array` with `CPUBackendTag()`, `CuArray` with `CUDABackendTag()`).
Nothing is factorized here: the first [`fv_newton!`](@ref) step does the one
symbolic analysis.
"""
function fv_newton_solver(prob::FVSteadyProblem2D; to = Array,
                          linsolve::LinearSolveBackend = CPUBackendTag(),
                          chunk::Int = 4)
    t_pattern = @elapsed S = fv_jacobian_pattern(prob)
    t_coloring = @elapsed colors, ncol = fv_jacobian_coloring(S)
    n = size(S, 1)
    if linsolve isa CPUBackendTag
        J = SparseMatrixCSC{Float64,Int}(n, n, copy(S.colptr), copy(rowvals(S)),
                                         zeros(Float64, nnz(S)))
        nzval = nonzeros(J)
        order = :csc
    else
        _, _, rowptr, colval = _fv_storage_order(S, :csr)
        J, nzval = _fv_device_csr(linsolve, rowptr, colval, n)
        order = :csr
    end
    asm = fv_jacobian_assembler(S, colors, ncol; to, order, chunk)
    dprob = fv_adapt(to, prob)
    vec0() = to(zeros(Float64, n))
    return FVNewtonSolver(dprob, asm, J, nzval, linsolve, nothing,
                          vec0(), vec0(), vec0(), _FVNormBuffers(to, n ÷ 3), 0, 0,
                          t_pattern, t_coloring,
                          (NaN, NaN, NaN), (NaN, NaN, NaN))
end

_fv_sync(U) = KernelAbstractions.synchronize(KernelAbstractions.get_backend(U))

"""
    fv_jacobian!(solver, U; shift = 0.0) -> solver.J

Assemble `∂R/∂U` at `U` into `solver.J` (values only; pattern fixed).
`shift > 0` adds `shift` to the diagonal of the two momentum rows of every cell
(pseudo-transient term `h²/Δτ`).
"""
function fv_jacobian!(s::FVNewtonSolver, U; shift::Real = 0.0)
    fv_assemble_jacobian!(s.nzval, s.asm, U, s.prob; sync = shift == 0)
    s.asm_sig = _fv_state_sig(s.norms, shift, U)
    if shift != 0
        backend = KernelAbstractions.get_backend(U)
        _fv_diag_shift_kernel!(backend, 256)(s.nzval, s.asm.diag, Float64(shift);
                                             ndrange = s.asm.n ÷ 3)
        KernelAbstractions.synchronize(backend)
    end
    return s.J
end

# factorize on first use (the only symbolic analysis), refactorize afterwards
function _fv_factor!(s::FVNewtonSolver)
    if s.cache === nothing
        s.cache = lin_factorize(s.J; backend = s.linsolve, spd = false)
        s.n_factorize += 1
    else
        lin_refactorize!(s.cache)
        s.n_refactorize += 1
    end
    s.fac_sig = s.asm_sig
    return s.cache
end

"""
    fv_newton!(solver, U; atol = 1e-12, rtol = 1e-13, maxit = 30,
               linesearch = true, armijo = 1e-4, alpha_min = 2.0^-10,
               dtau0 = nothing, dtau_max = 1e8, verbose = false) -> NamedTuple

Solve `R(U) = 0` in place by Newton's method, `U` on the solver's device (e.g.
`to(fv_state(...))`). Each iteration: residual → coloured dual assembly of `J`
→ factorization (first iteration: [`lin_factorize`](@ref), i.e. symbolic
analysis + numeric factorization; afterwards [`lin_refactorize!`](@ref),
numeric only) → in-place [`lin_solve!`](@ref) of `J δ = R` → Armijo
backtracking `U ← U − α δ` on `‖R‖₂`.

Stops when `‖R‖∞ ≤ atol` or `‖R‖₂ ≤ rtol ‖R₀‖₂`, or after `maxit` iterations,
or when two consecutive line searches reach `alpha_min` without decrease
(round-off floor; flagged `stagnated`).

Pseudo-transient continuation (off by default): with `dtau0` set, the momentum
rows get the diagonal shift `h²/Δτ_k`, `Δτ_k = dtau0 ‖R₀‖₂ / ‖R_k‖₂` (switched
evolution relaxation); once `Δτ_k ≥ dtau_max` the shift is dropped for good,
so the last iterations are pure Newton (quadratic convergence).

Returns `(converged, stagnated, iterations, history, mom, cont, inf, alpha,
shift, t_assembly, t_factor, t_solve, t_linesearch, n_factorize, n_refactorize)`:
`history[k]` = `‖R‖₂` before iteration `k` (`history[end]` final), `mom`/`cont`
the separate momentum / continuity 2-norms, `inf` the max norm, timings in
seconds per iteration (`t_factor[1]` includes the analysis). The cache in
`solver.cache` holds the factors of the last assembled Jacobian (shifted if
continuation was still active): call [`fv_newton_finalize!`](@ref) before using
it for an adjoint.
"""
function fv_newton!(s::FVNewtonSolver, U; atol::Real = 1e-12, rtol::Real = 1e-13,
                    maxit::Int = 30, linesearch::Bool = true, armijo::Real = 1e-4,
                    alpha_min::Real = 2.0^-10, dtau0 = nothing, dtau_max::Real = 1e8,
                    verbose::Bool = false)
    p = s.prob
    h = Float64(p.grid.h)
    R = s.R; dU = s.dU; Ut = s.Ut
    hist = Float64[]; mom = Float64[]; cont = Float64[]; inf = Float64[]
    alphas = Float64[]; shifts = Float64[]
    ta = Float64[]; tf = Float64[]; ts = Float64[]; tl = Float64[]
    fv_residual!(R, U, p)
    m, c, ri = _fv_norms(s.norms, R)
    r = hypot(m, c)
    r0 = r
    push!(hist, r); push!(mom, m); push!(cont, c); push!(inf, ri)
    ptc = dtau0 !== nothing
    converged = ri <= atol || r <= rtol * r0
    stagnated = false
    nfail = 0
    it = 0
    while !converged && it < maxit
        it += 1
        sigma = 0.0
        if ptc
            dtau = Float64(dtau0) * r0 / r
            dtau >= dtau_max ? (ptc = false) : (sigma = h * h / dtau)
        end
        t0 = time_ns()
        fv_jacobian!(s, U; shift = sigma)
        t1 = time_ns()
        _fv_factor!(s)
        _fv_sync(U)
        t2 = time_ns()
        lin_solve!(dU, s.cache, R)
        _fv_sync(U)
        t3 = time_ns()
        alpha = 1.0
        local mt, ct, rit, rt
        while true
            Ut .= U .- alpha .* dU
            fv_residual!(R, Ut, p)
            mt, ct, rit = _fv_norms(s.norms, R)
            rt = hypot(mt, ct)
            (!linesearch || rt <= (1 - armijo * alpha) * r || alpha <= alpha_min) && break
            alpha /= 2
        end
        t4 = time_ns()
        nfail = (linesearch && rt > (1 - armijo * alpha) * r) ? nfail + 1 : 0
        U .= Ut                         # R already holds R(Ut)
        r = rt
        push!(hist, rt); push!(mom, mt); push!(cont, ct); push!(inf, rit)
        push!(alphas, alpha); push!(shifts, sigma)
        push!(ta, (t1 - t0) / 1e9); push!(tf, (t2 - t1) / 1e9)
        push!(ts, (t3 - t2) / 1e9); push!(tl, (t4 - t3) / 1e9)
        verbose && @info "fv_newton" it res = rt inf = rit alpha shift = sigma
        converged = rit <= atol || rt <= rtol * r0
        if nfail >= 2
            stagnated = true
            break
        end
    end
    return (; converged, stagnated, iterations = it, history = hist, mom, cont, inf,
            alpha = alphas, shift = shifts, t_assembly = ta, t_factor = tf,
            t_solve = ts, t_linesearch = tl,
            n_factorize = s.n_factorize, n_refactorize = s.n_refactorize)
end

"""
    fv_newton_finalize!(solver, U) -> solver

Re-assemble `∂R/∂U` at the converged `U` without any continuation shift and
refactorize (factorize if nothing was factorized yet), so that `solver.J` and
`solver.cache` describe the exact Jacobian at `U` — the starting point of the
discrete adjoint `Jᵀ λ = ∂ℐ/∂U` (C4: `lin_solve!(λ, solver.cache, g;
transpose = true)`).
"""
function fv_newton_finalize!(s::FVNewtonSolver, U)
    fv_jacobian!(s, U)
    _fv_factor!(s)
    _fv_sync(U)
    return s
end

# ------------------------------------------------------------------ driver
"""
    solve_incns_fv_newton(; problem, U0 = nothing, to = Array,
                          linsolve = CPUBackendTag(), chunk = 4, newton...) -> NamedTuple

Steady incompressible (Navier–)Stokes by Newton on the colocated FV residual
(C1 + C2 of the FV Newton roadmap). `problem` is a host
[`FVSteadyProblem2D`](@ref) (from [`fv_steady_problem`](@ref)); `U0` an initial
host state (default rest); `to`/`linsolve` select the device (`CuArray` +
`CUDABackendTag()` for cuDSS). Remaining keywords go to [`fv_newton!`](@ref).
Returns `(u, v, p, U, converged, iterations, residual_history, stats, solver)`
with host fields and the device state `U`.
"""
function solve_incns_fv_newton(; problem::FVSteadyProblem2D, U0 = nothing, to = Array,
                               linsolve::LinearSolveBackend = CPUBackendTag(),
                               chunk::Int = 4, newton...)
    s = fv_newton_solver(problem; to, linsolve, chunk)
    U = to(U0 === nothing ? zeros(Float64, fv_nunknowns(problem)) : Vector{Float64}(U0))
    stats = fv_newton!(s, U; newton...)
    u, v, p = fv_fields(problem, U)
    return (; u, v, p, U, converged = stats.converged, iterations = stats.iterations,
            residual_history = stats.history, stats, solver = s)
end
