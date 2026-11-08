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
# Dependencies are declared by equation, using the residual's face traversal.
# Each entry is (unknown component, tuple of cells); missing neighbours are zero.
function _fv_dependency_stencil(g, tag, i, j)
    function nb(i, j, d)
        i == 0 && return (0, 0)
        kind, ii, jj = _fv_face_kind(g, tag, i, j, d)
        return kind == _FV_F_INT ? (ii, jj) : (0, 0)
    end
    P = (i, j)
    E = nb(i, j, 1); W = nb(i, j, 2); N = nb(i, j, 3); S = nb(i, j, 4)
    star = (P, E, W, N, S)
    axis2 = (nb(E..., 1), nb(W..., 2), nb(N..., 3), nb(S..., 4))
    cross_u = (nb(N..., 1), nb(N..., 2), nb(S..., 1), nb(S..., 2))
    cross_v = (nb(E..., 3), nb(E..., 4), nb(W..., 3), nb(W..., 4))
    return (; P, E, W, N, S, star, axis2, cross_u, cross_v)
end
function _fv_equation_dependencies(::Val{1}, s, tag)
    tag == FV_CELL_SOLID && return ((1, (s.P,)),)
    return ((1, s.star), (2, (s.star..., s.cross_u...)), (3, (s.star..., s.axis2...)))
end
function _fv_equation_dependencies(::Val{2}, s, tag)
    tag == FV_CELL_SOLID && return ((2, (s.P,)),)
    return ((1, (s.star..., s.cross_v...)), (2, s.star), (3, (s.star..., s.axis2...)))
end
function _fv_equation_dependencies(::Val{3}, s, tag)
    tag in (FV_CELL_SOLID, FV_CELL_PINNED) && return ((3, (s.P,)),)
    return ((1, (s.P, s.E, s.W)), (2, (s.P, s.N, s.S)), (3, (s.star..., s.axis2...)))
end
# T1 scalar rows are identities for every cell tag, including solids and pins.
_fv_equation_dependencies(::Val{K}, s, tag) where {K} = ((K, (s.P,)),)

# The energy row reads both signs of every face flux, including the GNF face
# gradient, plus the cell-gradient heating and the second upwind temperature.
# Pins only replace continuity: their energy row keeps its complete support.
function _fv_problem_dependencies(prob, equation, s, tag)
    if prob.energy !== nothing && equation == 4 && tag != FV_CELL_SOLID
        uv = (s.star..., s.cross_u..., s.cross_v...)
        return ((1, uv), (2, uv), (3, (s.star..., s.axis2...)),
                (4, (s.star..., s.axis2...)))
    end
    return _fv_equation_dependencies(Val(equation), s, tag)
end

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

An immersed ghost reads the opposite fluid neighbour O. O already belongs to
the same axial pair used by the Green–Gauss gradient: replacing a wall face
value with L/Q or linear pressure does not widen that gradient's support.
The momentum star and the normal-velocity continuity star also already
contain O. Extrapolating tangential gradients from O to a nominal immersed
face reads O's tangential neighbours, already in the cross-term support of
the opposite interior face. Thus this pattern covers all immersed closure
combinations and remains inside the existing radius-two (5 × 5) box.
"""
function fv_jacobian_pattern(prob::FVSteadyProblem2D)
    g = prob.grid
    tag = Array(prob.tag)
    Nx, Ny = g.Nx, g.Ny
    n = fv_nunknowns(prob)
    I = Int[]; J = Int[]
    sizehint!(I, 21 * n); sizehint!(J, 21 * n)
    for j in 1:Ny, i in 1:Nx
        stencil = _fv_dependency_stencil(g, tag, i, j)
        for equation in 1:_fv_nfields(g)
            row = fv_index(g, i, j, equation)
            for (component, cells) in _fv_problem_dependencies(prob, equation, stencil, tag[i, j])
                for cell in cells
                    cell[1] == 0 && continue
                    push!(I, row)
                    push!(J, fv_index(g, cell[1], cell[2], component))
                end
            end
        end
    end
    _fv_mass_pattern!(I, J, prob)
    _fv_flow_pattern!(I, J, prob)
    return sparse(I, J, trues(length(I)), n, n, |)
end

"""
    fv_jacobian_coloring(S::SparseMatrixCSC[, prob]) -> (colors::Vector{Int}, ncolors::Int)

Column colouring of the pattern `S` for compressed forward-mode assembly
(columns of one colour share no row): SparseMatrixColorings greedy colouring
with the largest-first order, CPU, once per geometry. Pass `prob` for M3 to
exclude directly assembled pair rows while retaining local lambda columns,
and for mean-flow control to exclude both the analytic force row and column.
"""
function fv_jacobian_coloring(S::SparseMatrixCSC)
    prob = SparseMatrixColorings.ColoringProblem(; structure = :nonsymmetric,
                                                 partition = :column)
    res = SparseMatrixColorings.coloring(S, prob,
        SparseMatrixColorings.GreedyColoringAlgorithm(SparseMatrixColorings.LargestFirst()))
    return SparseMatrixColorings.column_colors(res), SparseMatrixColorings.ncolors(res)
end

# Pair rows are affine and assembled directly. Their long support must never
# participate in column coloring; lambda columns retain their local FV rows.
function fv_jacobian_coloring(S::SparseMatrixCSC, prob::FVSteadyProblem2D)
    n = size(S,2) - (prob.flowrate !== nothing)
    colors, count = fv_jacobian_coloring(S[1:fv_nunknowns(prob.grid), 1:n])
    prob.flowrate === nothing || push!(colors,0)
    return colors, count
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
                                        @Const(Rd), offset, ::Val{N}, nlocal) where {N}
    q = @index(Global)
    @inbounds begin
        c = colors[nzcol[q]] - offset
        if 1 <= c <= N && nzrow[q] <= nlocal
            nzval[q] = ForwardDiff.partials(Rd[nzrow[q]])[c]
        end
    end
end

@kernel function _fv_diag_shift_kernel!(nzval, @Const(diag), sigma, layout)
    c = @index(Global)          # cell index; shift the two momentum rows
    @inbounds begin
        b = _fv_cell_base(layout, c)
        nzval[diag[b + _fv_velocity_slot(layout, 1)]] += sigma
        nzval[diag[b + _fv_velocity_slot(layout, 2)]] += sigma
    end
end

"""
    fv_assemble_jacobian!(nzval, asm, U, prob; sync = true) -> nzval

Write the values of `∂R/∂U` at the state `U` into `nzval` (the value vector of
the sparse matrix whose storage order `asm` was built for: `nonzeros(J)` of a
`SparseMatrixCSC`, `J.nzVal` of a cuDSS CSR matrix). `ceil(ncolors / chunk)`
passes of three kernels (seed, C1 residual kernel on duals, decompression); the
residual is the C1 kernel itself, so the matrix is the exact Jacobian of the
residual evaluated by [`fv_residual!`](@ref). M3 pair rows are omitted from
compressed differentiation and filled from their lowered affine coefficients.
Allocation-free on the device.
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
        fv_residual!(asm.Rd, asm.Ud, prob; sync = false, mass_rows = false)
        dec!(nzval, asm.nzrow, asm.nzcol, asm.colors, asm.Rd, offset, Val(N), fv_nunknowns(prob.grid); ndrange = asm.nnz)
    end
    _fv_mass_jacobian!(nzval, asm, prob, backend)
    _fv_flow_jacobian!(nzval, asm, prob, backend)
    sync && KernelAbstractions.synchronize(backend)
    return nzval
end

# ------------------------------------------------------------------ residual norms
@kernel function _fv_norm_partials_kernel!(Pt, @Const(R), cells, per)
    t = @index(Global)
    ncell, layout = _fv_norm_cells(cells)
    nlocal = _fv_nfields(layout) * ncell
    T = eltype(Pt)
    m2 = zero(T); c2 = zero(T); mi = zero(T); ci = zero(T)
    c0 = (t - 1) * per + 1
    c1 = min(t * per, ncell + length(R) - nlocal)
    @inbounds for c in c0:c1
        if c <= ncell
            b = _fv_cell_base(layout, c)
            ru = R[b + _fv_velocity_slot(layout, 1)]; rv = R[b + _fv_velocity_slot(layout, 2)]; rc = R[b + _fv_pressure_slot(layout)]
            m2 += ru * ru + rv * rv
            c2 += rc * rc
            mi = max(mi, abs(ru), abs(rv))
            ci = max(ci, abs(rc))
            for scalar in 1:_fv_nscalars(layout)
                ci = max(ci, abs(R[b + _fv_scalar_slot(layout, scalar)]))
            end
        else
            rc = R[nlocal + c - ncell]
            c2 += rc * rc
            ci = max(ci, abs(rc))
        end
    end
    @inbounds begin
        Pt[1, t] = m2; Pt[2, t] = c2; Pt[3, t] = mi; Pt[4, t] = ci
    end
end

# Keep the original Int kernel argument valid for compiler/unit callers.
@inline _fv_norm_cells(ncell::Int) = (ncell, FVFieldLayout{0}())
@inline _fv_norm_cells(cells::Tuple{Int,<:FVFieldLayout}) = cells

struct _FVEquationNormBuffers{D,H,V}
    dev::D
    host::H
    scales::V
    host_scales::Vector{Float64}
end
struct _FVNormBuffers{D,H,L,E}
    dev::D
    host::H
    per::Int
    ncell::Int
    layout::L
    equations::E
    flow_index::Int
end

function _FVNormBuffers(to, ncell::Int; per::Int = 256, nextra::Int = 0,
                         layout::FVFieldLayout = FVFieldLayout{0}(), flow_index::Int = 0)
    nt = cld(ncell + nextra, per)
    nf = _fv_nfields(layout) + (flow_index != 0)
    eq = _FVEquationNormBuffers(to(zeros(Float64, 2nf, nt)), zeros(Float64, 2nf, nt),
                                to(ones(Float64, nf)), ones(Float64, nf))
    return _FVNormBuffers(to(zeros(Float64, 4, nt)), zeros(Float64, 4, nt), per, ncell, layout, eq, flow_index)
end

"""Set fixed positive equation scales for norm reports and Newton stopping only.

Residuals, Jacobians and adjoints are unscaled. No scale is updated from the
iterate. Unit scales retain the legacy momentum/continuity reduction order.
"""
function _fv_equation_scales!(nb::_FVNormBuffers, scales)
    length(scales) == length(nb.equations.host_scales) || throw(DimensionMismatch("one scale per equation is required"))
    values = Float64.(scales)
    all(x -> isfinite(x) && x > 0, values) || throw(ArgumentError("equation scales must be finite and strictly positive"))
    copyto!(nb.equations.host_scales, values)
    copyto!(nb.equations.scales, nb.equations.host_scales)
    return nb
end

# Separate reductions deliberately do not replace m2 += ru*ru + rv*rv above.
@kernel function _fv_equation_norm_kernel!(Pt, @Const(R), @Const(scales), cells, per, flow_index)
    t = @index(Global)
    ncell, layout = _fv_norm_cells(cells)
    nf = _fv_nfields(layout)
    nlocal = nf * ncell
    c0 = (t - 1) * per + 1
    c1 = min(t * per, ncell + length(R) - nlocal)
    nequations = length(scales)
    @inbounds for equation in 1:nequations
        square = 0.0; inf = 0.0
        for cell in c0:c1
            if cell <= ncell && equation <= nf
                value = R[_fv_cell_index(layout, cell, equation)] / scales[equation]
                square += value * value
                inf = max(inf, abs(value))
            elseif cell > ncell && ((nlocal + cell - ncell == flow_index && equation == nf+1) ||
                   (nlocal + cell - ncell != flow_index && equation == _fv_pressure_slot(layout)))
                value = R[nlocal + cell - ncell] / scales[equation]
                square += value * value
                inf = max(inf, abs(value))
            end
        end
        Pt[equation, t] = square
        Pt[nequations + equation, t] = inf
    end
end

function _fv_equation_norms(nb::_FVNormBuffers, R)
    eq = nb.equations
    backend = KernelAbstractions.get_backend(R)
    _fv_equation_norm_kernel!(backend, 64)(eq.dev, R, eq.scales, (nb.ncell, nb.layout), nb.per, nb.flow_index;
                                           ndrange = size(eq.dev, 2))
    KernelAbstractions.synchronize(backend)
    H = eq.dev isa Array ? eq.dev : copyto!(eq.host, eq.dev)
    nf = length(eq.host_scales)
    l2 = zeros(Float64, nf); inf = zeros(Float64, nf)
    @inbounds for t in axes(H, 2), equation in 1:nf
        l2[equation] += H[equation, t]
        inf[equation] = max(inf[equation], H[nf + equation, t])
    end
    return (; l2 = sqrt.(l2), inf)
end

# Hook for reporting and Newton merit/stopping norms. Synthetic scalar equations
# participate in the solve, while unit-scale NS=0 follows the exact legacy path.
function _fv_newton_report_norms(nb::_FVNormBuffers, R)
    m, c, ri = _fv_norms(nb, R)
    equations = _fv_equation_norms(nb, R)
    if nb.flow_index != 0 || !all(isone, nb.equations.host_scales)
        m = hypot(equations.l2[_fv_velocity_slot(nb.layout, 1)],
                  equations.l2[_fv_velocity_slot(nb.layout, 2)])
        c = equations.l2[_fv_pressure_slot(nb.layout)]
        ri = maximum(equations.inf)
    end
    r = hypot(m, c)
    for scalar in 1:_fv_nscalars(nb.layout)
        slot = _fv_scalar_slot(nb.layout, scalar)
        r = hypot(r, equations.l2[slot])
        ri = max(ri, equations.inf[slot])
    end
    nb.flow_index == 0 || (r = hypot(r, equations.l2[end]))
    return m, c, ri, r, equations
end

# (‖R_momentum‖₂, ‖R_continuity‖₂, ‖R‖∞): one kernel + one copy of the
# 4 × ncell/256 partial sums to the host; deterministic, no device allocation.
function _fv_norms(nb::_FVNormBuffers, R)
    ncell = nb.ncell
    backend = KernelAbstractions.get_backend(R)
    _fv_norm_partials_kernel!(backend, 64)(nb.dev, R, (ncell, nb.layout), nb.per; ndrange = size(nb.dev, 2))
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
@kernel function _fv_state_sig_kernel!(Pt, @Const(U), cells, per)
    t = @index(Global)
    _, layout = _fv_norm_cells(cells)
    su = 0.0; sq = 0.0
    k0 = _fv_nfields(layout) * (t - 1) * per + 1
    k1 = min(_fv_nfields(layout) * t * per, length(U))
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
    ncell = nb.ncell
    backend = KernelAbstractions.get_backend(U)
    _fv_state_sig_kernel!(backend, 64)(nb.dev, U, (ncell, nb.layout), nb.per; ndrange = size(nb.dev, 2))
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
`CUDABackendTag()` by `ext/KrakeNNCUDSSExt.jl` (`using CUDA, CUDSS`).
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
    t_coloring = @elapsed colors, ncol = fv_jacobian_coloring(S, prob)
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
                          vec0(), vec0(), vec0(), _FVNormBuffers(to, prob.grid.Nx * prob.grid.Ny;
                              nextra=_fv_npairs(prob.immersed)+(prob.flowrate !== nothing), layout=prob.grid.layout,
                              flow_index=prob.flowrate === nothing ? 0 : _fv_force_index(prob)), 0, 0,
                          t_pattern, t_coloring,
                          (NaN, NaN, NaN), (NaN, NaN, NaN))
end

_fv_sync(U) = KernelAbstractions.synchronize(KernelAbstractions.get_backend(U))

"""Freeze the pass-3 physical scales from case data, never from the iterate.

H is the domain height. nu_ref is the fluid-cell mean viscosity at rest
(prescribed cell values for Newtonian fields, the regularised constitutive
value at zero velocity gradient for GNF/laws). U_c uses the largest wall
speed, body-force magnitude in the plane-Poiseuille estimate, and optional
mean-flow target. DeltaT uses imposed-temperature spread and a one-kelvin
floor. Pure conduction retains positive unused flow scales with U_floor=1e-12.
"""
function _fv_energy_scales(prob)
    p = prob.nu isa Array ? prob : fv_adapt(Array, prob)
    g = p.grid; e = p.energy; H = g.Ny * g.h
    temperatures = [e.boundary[d] for d in 1:4 if e.dirichlet[d]]
    reference_visc = _fv_reference_viscosity(p.visc, minimum(temperatures))
    nu_sum = 0.0; count = 0; force = 0.0
    for j in 1:g.Ny, i in 1:g.Nx
        p.tag[i, j] == FV_CELL_SOLID && continue
        nu_sum += _fv_wall_nu(reference_visc, p.nu[i, j], 0.0, 0.0, 0.0, 0.0, i, j)
        count += 1
        force = max(force, hypot(p.src[fv_index(g,i,j,1)], p.src[fv_index(g,i,j,2)]))
    end
    nu_ref = _fv_reference_nu(p.visc, nu_sum / count)
    isfinite(nu_ref) && nu_ref > 0 || throw(ArgumentError("energy scales require positive finite reference viscosity"))
    wall = maximum(hypot(p.bvel[1,l,d], p.bvel[2,l,d])
                   for l in axes(p.bvel,2), d in 1:4)
    target = p.flowrate === nothing ? 0.0 : abs(p.flowrate.target)
    Uc = max(wall, force * H^2 / (12nu_ref), target, 1e-12)
    spread = maximum(temperatures) - minimum(temperatures)
    deltaT = max(e.sigma_nom * nu_ref * Uc^2 / e.alpha, spread, 1.0)
    momentum = nu_ref * Uc * g.h^2 / H^2
    continuity = Uc * g.h
    energy = e.alpha * deltaT * g.h^2 / H^2
    equations = (momentum,momentum,continuity,energy)
    p.flowrate === nothing || (equations = (equations...,Uc))
    return (; Uc, nu_ref, deltaT, equations)
end

# Temperature increment reduction uses existing, preallocated norm buffers.
# No GPU reduction workspace is allocated in a Newton iteration.
@kernel function _fv_temperature_increment_kernel!(Pt, @Const(dU), layout, ncell, per)
    t = @index(Global)
    value = 0.0
    @inbounds for cell in ((t-1)*per+1):min(t*per,ncell)
        value = max(value, abs(dU[_fv_cell_index(layout,cell,4)]))
    end
    @inbounds Pt[1,t] = value
end
function _fv_temperature_increment(nb, dU)
    backend = KernelAbstractions.get_backend(dU)
    _fv_temperature_increment_kernel!(backend,64)(nb.dev,dU,nb.layout,nb.ncell,nb.per;
                                                   ndrange=size(nb.dev,2))
    KernelAbstractions.synchronize(backend)
    H = nb.dev isa Array ? nb.dev : copyto!(nb.host,nb.dev)
    return maximum(view(H,1,:))
end

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
        _fv_diag_shift_kernel!(backend, 256)(s.nzval, s.asm.diag, Float64(shift), s.prob.grid.layout;
                                             ndrange = s.prob.grid.Nx * s.prob.grid.Ny)
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

With opt-in energy, `atol`/`rtol` do not select the stopping rule: every
equation's scaled maximum residual must be at most 1e-10, and the accepted
temperature increment at most 1e-10*DeltaT_c. The pass-3 scales are computed
once from case data at entry and remain fixed through all line searches.
An initially converged state has zero accepted increment. These scales affect
norms only; the physical residual and its Jacobian remain cell-integrated.

Pseudo-transient continuation (off by default): with `dtau0` set, the momentum
rows get the diagonal shift `h²/Δτ_k`, `Δτ_k = dtau0 ‖R₀‖₂ / ‖R_k‖₂` (switched
evolution relaxation); once `Δτ_k ≥ dtau_max` the shift is dropped for good,
so the last iterations are pure Newton (quadratic convergence).

Returns `(converged, stagnated, iterations, history, mom, cont, inf, alpha,
shift, t_assembly, t_factor, t_solve, t_linesearch, n_factorize, n_refactorize)`:
`history[k]` = `‖R‖₂` before iteration `k` (`history[end]` final), `mom`/`cont`
the separate momentum / continuity 2-norms, `inf` the max norm. Additional
`equation_l2`, `equation_inf` histories use the fixed `equation_scales`, one
entry per cell equation (M3 constraints join pressure/continuity); the optional
mean-flow row has its own final equation and velocity scale U_c. Timings are in
seconds per iteration (`t_factor[1]` includes the analysis). The cache in
`solver.cache` holds the factors of the last assembled Jacobian (shifted if
continuation was still active): call [`fv_newton_finalize!`](@ref) before using
it for an adjoint.
"""
function fv_newton!(s::FVNewtonSolver, U; atol::Real = 1e-12, rtol::Real = 1e-13,
                    maxit::Int = 30, linesearch::Bool = true, armijo::Real = 1e-4,
                    alpha_min::Real = 2.0^-10, dtau0 = nothing, dtau_max::Real = 1e8,
                    verbose::Bool = false, thermal_scales = nothing,
                    abort_small_step::Bool = false)
    p = s.prob
    thermal = p.energy !== nothing
    thermal_scales = thermal ? (thermal_scales === nothing ? _fv_energy_scales(p) : thermal_scales) : nothing
    thermal && _fv_equation_scales!(s.norms, thermal_scales.equations)
    temperature_increment = Float64[]
    h = Float64(p.grid.h)
    R = s.R; dU = s.dU; Ut = s.Ut
    hist = Float64[]; mom = Float64[]; cont = Float64[]; inf = Float64[]
    alphas = Float64[]; shifts = Float64[]
    equation_l2 = Vector{Float64}[]; equation_inf = Vector{Float64}[]
    equation_scales = copy(s.norms.equations.host_scales)
    ta = Float64[]; tf = Float64[]; ts = Float64[]; tl = Float64[]
    fv_residual!(R, U, p)
    m, c, ri, r, equations = _fv_newton_report_norms(s.norms, R)
    push!(equation_l2, equations.l2); push!(equation_inf, equations.inf)
    r0 = r
    push!(hist, r); push!(mom, m); push!(cont, c); push!(inf, ri)
    ptc = dtau0 !== nothing
    converged = thermal ? maximum(equations.inf) <= 1e-10 : ri <= atol || r <= rtol * r0
    stagnated = false
    failure_reason = :none
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
        delta_temperature = thermal ? _fv_temperature_increment(s.norms, dU) : 0.0
        t3 = time_ns()
        alpha = 1.0
        local mt, ct, rit, rt, equation_trial
        while true
            Ut .= U .- alpha .* dU
            fv_residual!(R, Ut, p)
            mt, ct, rit, rt, equation_trial = _fv_newton_report_norms(s.norms, R)
            (!linesearch || rt <= (1 - armijo * alpha) * r || alpha <= alpha_min) && break
            alpha /= 2
        end
        t4 = time_ns()
        if abort_small_step && linesearch && alpha < alpha_min
            failure_reason = :line_search
            stagnated = true
            break
        end
        nfail = (linesearch && rt > (1 - armijo * alpha) * r) ? nfail + 1 : 0
        U .= Ut                         # R already holds R(Ut)
        r = rt
        push!(hist, rt); push!(mom, mt); push!(cont, ct); push!(inf, rit)
        push!(alphas, alpha); push!(shifts, sigma)
        push!(equation_l2, equation_trial.l2); push!(equation_inf, equation_trial.inf)
        push!(ta, (t1 - t0) / 1e9); push!(tf, (t2 - t1) / 1e9)
        push!(ts, (t3 - t2) / 1e9); push!(tl, (t4 - t3) / 1e9)
        verbose && @info "fv_newton" it res = rt inf = rit alpha shift = sigma
        thermal && push!(temperature_increment, alpha * delta_temperature)
        converged = thermal ? (maximum(equation_trial.inf) <= 1e-10 &&
                               alpha * delta_temperature <= 1e-10 * thermal_scales.deltaT) :
                              rit <= atol || rt <= rtol * r0
        if nfail >= 2 && (!thermal || !converged)
            stagnated = true
            break
        end
    end
    result = (; converged, stagnated, iterations = it, history = hist, mom, cont, inf,
            alpha = alphas, shift = shifts, t_assembly = ta, t_factor = tf,
            t_solve = ts, t_linesearch = tl,
            n_factorize = s.n_factorize, n_refactorize = s.n_refactorize,
            equation_l2, equation_inf, equation_scales)
    return thermal ? merge(result, (; temperature_increment, thermal_scales, failure_reason)) : result
end

# ------------------------------------------------------------------ frozen-flow energy solve
"""Workspace for Newton on the temperature block with bitwise frozen u,v,p.

Only the Ncell-by-Ncell temperature matrix is factorized. Coloured duals
still traverse the production residual, with zero seeds in all flow slots.
The full-state workspaces and all reduction/device buffers are allocated once.
"""
mutable struct FVEnergySolver{P,A,M,V,B,N,S}
    prob::P
    asm::A
    J::M
    nzval::V
    linsolve::B
    cache::Union{Nothing,LinearSolveCache}
    U::V
    Rfull::V
    R::V
    dT::V
    trial::V
    norms::N
    scales::S
end

@kernel function _fv_energy_scatter_kernel!(U, @Const(temperature), layout)
    cell = @index(Global)
    @inbounds U[_fv_cell_index(layout,cell,4)] = temperature[cell]
end
@kernel function _fv_energy_gather_kernel!(temperature, @Const(U), layout)
    cell = @index(Global)
    @inbounds temperature[cell] = U[_fv_cell_index(layout,cell,4)]
end
@kernel function _fv_energy_norm_kernel!(Pt, @Const(R), per)
    t = @index(Global)
    square = 0.0; inf = 0.0
    @inbounds for cell in ((t-1)*per+1):min(t*per,length(R))
        square += R[cell]^2
        inf = max(inf,abs(R[cell]))
    end
    @inbounds begin Pt[1,t]=square; Pt[2,t]=inf end
end

"""
    fv_energy_solver(prob, U; to=Array, linsolve=CPUBackendTag(), chunk=4)

Prepare an energy-only solve, freezing the supplied full state U's u,v,p.
U must be a host vector; use the same to/linsolve pair as fv_newton_solver
for GPU execution. Rebuild this workspace when the prescribed flow changes.
"""
function fv_energy_solver(prob::FVSteadyProblem2D, U::AbstractVector;
                          to=Array, linsolve::LinearSolveBackend=CPUBackendTag(), chunk::Int=4)
    prob.energy === nothing && throw(ArgumentError("energy-only solve requires energy=FVEnergy(...)"))
    length(U) == fv_nunknowns(prob) || throw(DimensionMismatch("full frozen state has wrong length"))
    nf = _fv_nfields(prob.grid); nfull = length(U); n = prob.grid.Nx*prob.grid.Ny
    thermal = 4:nf:fv_nunknowns(prob.grid)
    S = fv_jacobian_pattern(prob)[thermal,thermal]
    colors,ncolors = fv_jacobian_coloring(S)
    if linsolve isa CPUBackendTag
        J = SparseMatrixCSC{Float64,Int}(n,n,copy(S.colptr),copy(rowvals(S)),zeros(nnz(S)))
        nzval=nonzeros(J); order=:csc
    else
        _,_,rowptr,colval = _fv_storage_order(S,:csr)
        J,nzval = _fv_device_csr(linsolve,rowptr,colval,n); order=:csr
    end
    rows,cols,_,_ = _fv_storage_order(S,order)
    fullcolors=zeros(Int32,nfull); fullcolors[thermal] .= colors
    D=ForwardDiff.Dual{FVJacobianTag,Float64,chunk}
    Ud=to(Vector{D}(undef,nfull)); Rd=similar(Ud)
    nzrow=to(Int32.(nf .* (rows .- 1) .+ 4))
    nzcol=to(Int32.(nf .* (cols .- 1) .+ 4))
    asm=FVJacobianAssembler{chunk,typeof(nzrow),typeof(Ud)}(
        nfull,length(rows),ncolors,nzrow,nzcol,to(fullcolors),to(zeros(Int32,nfull)),Ud,Rd)
    nb=_FVNormBuffers(to,n)
    return FVEnergySolver(fv_adapt(to,prob),asm,J,nzval,linsolve,nothing,
        to(Float64.(U)),to(zeros(nfull)),to(zeros(n)),to(zeros(n)),to(zeros(n)),
        nb,_fv_energy_scales(prob))
end

function _fv_energy_block_norm(s, R)
    nb=s.norms; backend=KernelAbstractions.get_backend(R)
    _fv_energy_norm_kernel!(backend,64)(nb.dev,R,nb.per;ndrange=size(nb.dev,2))
    KernelAbstractions.synchronize(backend)
    H=nb.dev isa Array ? nb.dev : copyto!(nb.host,nb.dev)
    square=0.0; inf=0.0
    for t in axes(H,2)
        square+=H[1,t]; inf=max(inf,H[2,t])
    end
    return sqrt(square),inf
end
function _fv_energy_block_residual!(s, temperature)
    backend=KernelAbstractions.get_backend(temperature); n=length(temperature)
    _fv_energy_scatter_kernel!(backend,256)(s.U,temperature,s.prob.grid.layout;ndrange=n)
    fv_residual!(s.Rfull,s.U,s.prob;sync=false)
    _fv_energy_gather_kernel!(backend,256)(s.R,s.Rfull,s.prob.grid.layout;ndrange=n)
    KernelAbstractions.synchronize(backend)
    return _fv_energy_block_norm(s,s.R)
end

"""
    fv_energy_solve!(solver, temperature; maxit=30) -> stats

Solve only the T block in place with frozen u,v,p. `temperature` is the raw
state block, i.e. `theta = T - T_ref` of fluid cells (`prob.energy.Tref`; solid
rows are storage identities and start at 0). The residual is affine in T
in T2; Newton normally takes one physical step and one round-off correction.
Stopping requires scaled max residual <=1e-10 AND max increment <=1e-10*DeltaT.
No whole-flow convergence criterion is imposed on the prescribed velocity.
"""
function fv_energy_solve!(s::FVEnergySolver, temperature::AbstractVector; maxit::Int=30)
    length(temperature)==length(s.R) || throw(DimensionMismatch("temperature block has wrong length"))
    _,ri=_fv_energy_block_residual!(s,temperature)
    scale=s.scales.equations[4]; history=[ri/scale]; increments=Float64[]
    converged=ri/scale<=1e-10; iterations=0
    while !converged && iterations<maxit
        iterations+=1
        fv_assemble_jacobian!(s.nzval,s.asm,s.U,s.prob)
        if s.cache===nothing
            s.cache=lin_factorize(s.J;backend=s.linsolve,spd=false)
        else
            lin_refactorize!(s.cache)
        end
        lin_solve!(s.dT,s.cache,s.R)
        _,increment=_fv_energy_block_norm(s,s.dT)
        temperature .-= s.dT
        _,ri=_fv_energy_block_residual!(s,temperature)
        push!(history,ri/scale); push!(increments,increment)
        converged=ri/scale<=1e-10 && increment<=1e-10*s.scales.deltaT
    end
    return (;converged,iterations,history,temperature_increment=increments,
             thermal_scales=s.scales)
end

"""
    fv_energy_solve(prob, U; to=Array, linsolve=CPUBackendTag(), chunk=4, maxit=30)

Return host (u,v,p,T), full device state U and stats after solving energy with
the supplied u,v,p fixed. The caller's U is not changed. Initial T is U's
first scalar. Use fv_energy_solver/fv_energy_solve! to reuse GPU workspaces.
"""
function fv_energy_solve(prob::FVSteadyProblem2D,U::AbstractVector;
                         to=Array,linsolve::LinearSolveBackend=CPUBackendTag(),chunk::Int=4,
                         maxit::Int=30)
    s=fv_energy_solver(prob,U;to,linsolve,chunk)
    temperature=to(Float64.(Array(U)[4:_fv_nfields(prob.grid):fv_nunknowns(prob.grid)]))
    stats=fv_energy_solve!(s,temperature;maxit)
    u,v,p,T=fv_fields(prob,s.U)
    return (;u,v,p,T,U=s.U,stats,solver=s,converged=stats.converged,iterations=stats.iterations)
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
    initial = U0 === nothing ? _fv_initial_force!(zeros(Float64, fv_nunknowns(problem)),problem) : Vector{Float64}(U0)
    if U0 === nothing && problem.energy !== nothing && _fv_has_state_temperature(problem.visc)
        e = problem.energy
        Tw = minimum(e.boundary[d] for d in 1:4 if e.dirichlet[d])
        for cell in eachindex(problem.tag)
            initial[_fv_cell_index(problem.grid,cell,4)] = problem.tag[cell] == FV_CELL_SOLID ? 0.0 : Tw - e.Tref
        end
    end
    U = to(initial)
    stats = fv_newton!(s, U; newton...)
    fields = fv_fields(problem, U)
    u, v, p = fields
    if problem.energy !== nothing
        return (; u, v, p, T=fields[4], U, converged=stats.converged,
                iterations=stats.iterations, residual_history=stats.history, stats, solver=s)
    end
    return (; u, v, p, U, converged = stats.converged, iterations = stats.iterations,
            residual_history = stats.history, stats, solver = s)
end
