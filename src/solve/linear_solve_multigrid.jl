# P3 of the Krylov branch: aggregation multigrid for the blocks of the block
# SIMPLE preconditioner (src/solve/linear_solve_krylov.jl).
#
# `precond = :mg` replaces the ILU(0) of the velocity block A by one multigrid
# cycle, and (`schur = :auto` → `:mg`) the LU of the Schur approximation
# S̃ = C − D diag(A)⁻¹ G by one multigrid cycle as well. Dependency-free and
# generic in the matrix values:
#
#   * aggregates: geometric 2 × 2 × 2 blocks of cells on the structured grid,
#     one aggregate per (block, field component, cell class), the class being
#     "solid" or "not solid" (solid cells carry identity rows; they never share
#     an aggregate with fluid cells; the pinned pressure cell stays with its
#     fluid neighbours, so every coarse Schur matrix inherits the pin and stays
#     non-singular). Coarse nodes keep their block coordinates, so the next
#     level aggregates them the same way (2 × 2 × 2 blocks of blocks);
#   * prolongation P piecewise constant (every row belongs to one aggregate, so
#     P 1 = 1), restriction R = Pᵀ;
#   * coarse operators Galerkin A_c = Pᵀ A P, i.e. A_c[I, J] = Σ A[i, j] over
#     agg(i) = I, agg(j) = J: the coarse pattern and, for each coarse entry,
#     the list of fine positions summed into it are built once on the host;
#     after each refactorization the coarse values are one gather-sum per
#     coarse entry on the device (no atomics, deterministic). Memory: one Int32
#     per fine non-zero per level, plus the coarse matrices (no more non-zeros
#     per row than the fine one), so linear in the number of cells;
#   * smoother: damped Jacobi (`mg_smoother = :jacobi`, ω default 0.7),
#     l1-Jacobi (`:l1jacobi`, default; d_i = sign(a_ii) Σⱼ|a_ij|, ω default 1),
#     or ILU(0) (`:ilu0`, the P1/P2 factorization of each level, ω default 1);
#     `mg_sweeps` pre- and as many post-smoothing sweeps;
#   * cycle: V or W (`mg_cycle`, default W: in 3D a 2 × 2 × 2 coarsening divides
#     the work by 8 per level, so the W-cycle costs about 4/3 of the finest
#     level work and stays a fixed linear operator, as GMRES with a right
#     preconditioner requires; a K-cycle would not be);
#   * coarsest level (≤ `mg_coarse` unknowns, or no further reduction, or
#     `mg_levels` levels): sparse LU through the Schur LU hooks of P2
#     (UMFPACK on CPU, cuDSS on GPU), refactorized with the values.
#
# Transposed solve (adjoint): the transposed SIMPLE is built from the blocks of
# Jᵀ, so its hierarchy is the hierarchy of Aᵀ with the same aggregates:
# Pᵀ Aᵀ P = (Pᵀ A P)ᵀ at every level. With a Jacobi smoother (pre and post
# sweeps alike, coarsest solved exactly) the cycle applied to Aᵀ is exactly the
# transpose of the cycle applied to A; with the ILU(0) smoother it is not
# (ILU(Aᵀ) ≠ ILU(A)ᵀ), GMRES absorbs the difference.
#
# All device work is KernelAbstractions kernels and broadcasts (one code path
# for Array and CuArray, no scalar indexing); the hierarchy pattern is built on
# the host once per geometry.

using KernelAbstractions

mutable struct _MGLevel{VI,VV}
    n::Int
    rowptr::VI
    colval::VI
    diag::VI
    A::VV                        # level matrix values (finest: shared with the block)
    dinv::VV                     # ω / d_i of the Jacobi smoothers (empty for ILU(0))
    x::VV
    b::VV
    r::VV
    t::VV
    smoother::Any                # _KrylovSide (ILU(0) smoother) or nothing
    nc::Int                      # size of the next coarser level (0 on the coarsest)
    agg::VI                      # coarse node of each row (prolongation)
    Rrp::VI                      # rows of each aggregate (restriction)
    Rcv::VI
    Grp::VI                      # fine positions summed into each coarse entry
    Gpos::VI
    coarse::Any                  # coarsest level: sparse LU object (built on first refresh)
end

mutable struct _KrylovMG{L}
    levels::Vector{L}
    cycle::Symbol
    smoother::Symbol
    omega::Float64
    sweeps::Int
    cycles::Int
    scale::Float64
    info::NamedTuple
end

_mg_omega(tag) = tag.mg_omega > 0 ? tag.mg_omega : (tag.mg_smoother === :jacobi ? 0.7 : 1.0)

# Grid nodes of the unknowns of one block: block coordinates (i, j, k), field
# component and cell class (1 = solid). `cells` are 1-based cell indices
# (c = i + (j-1) Nx + (k-1) Nx Ny), `comps` the field slot of each unknown.
function _mg_nodes(cells::AbstractVector{<:Integer}, comps::AbstractVector{<:Integer}, geom)
    Nx, Ny, Nz = geom.dims
    n = length(cells)
    ijk = Matrix{Int32}(undef, 3, n)
    cls = Vector{Int8}(undef, n)
    @inbounds for a in 1:n
        c = Int(cells[a]) - 1
        ijk[1, a] = c % Nx + 1
        ijk[2, a] = (c ÷ Nx) % Ny + 1
        ijk[3, a] = c ÷ (Nx * Ny) + 1
        cls[a] = geom.solid[c+1] ? Int8(1) : Int8(0)
    end
    return (; ijk, comp = Vector{Int32}(comps), cls)
end

# One coarsening step: aggregate = (⌈i/2⌉, ⌈j/2⌉, ⌈k/2⌉, component, class),
# numbered in (k, j, i, component, class) order (cell-interleaved, like the
# fine unknowns). Returns the aggregate of each node and the coarse nodes.
function _mg_aggregate(nodes)
    n = length(nodes.comp)
    n > 0 || throw(ArgumentError("multigrid: empty block"))
    key(a) = (Int32((nodes.ijk[3, a] + 1) ÷ 2), Int32((nodes.ijk[2, a] + 1) ÷ 2),
              Int32((nodes.ijk[1, a] + 1) ÷ 2), nodes.comp[a], Int32(nodes.cls[a]))
    keys = [key(a) for a in 1:n]
    u = sort!(unique(keys))
    id = Dict{NTuple{5,Int32},Int32}(k => Int32(m) for (m, k) in enumerate(u))
    agg = Int32[id[k] for k in keys]
    nc = length(u)
    ijk = Matrix{Int32}(undef, 3, nc)
    comp = Vector{Int32}(undef, nc)
    cls = Vector{Int8}(undef, nc)
    @inbounds for (m, k) in enumerate(u)
        ijk[1, m] = k[3]; ijk[2, m] = k[2]; ijk[3, m] = k[1]
        comp[m] = k[4]; cls[m] = Int8(k[5])
    end
    return agg, (; ijk, comp, cls), nc
end

# Rows of each aggregate (CSR of R = Pᵀ, rows sorted).
function _mg_restriction(agg::Vector{Int32}, nc::Int)
    Rrp = zeros(Int32, nc + 1)
    @inbounds for I in agg
        Rrp[I+1] += 1
    end
    Rrp[1] = 1
    @inbounds for I in 1:nc
        Rrp[I+1] += Rrp[I]
    end
    next = Vector{Int32}(Rrp[1:nc])
    Rcv = Vector{Int32}(undef, length(agg))
    @inbounds for (i, I) in enumerate(agg)
        Rcv[next[I]] = i
        next[I] += 1
    end
    return Rrp, Rcv
end

# Pattern of Pᵀ A P (sorted columns) and, for each coarse entry, the fine
# positions summed into it (every fine entry lands in exactly one).
function _mg_galerkin_pattern(rp::Vector{Int32}, cv::Vector{Int32}, agg::Vector{Int32},
                              Rrp::Vector{Int32}, Rcv::Vector{Int32}, nc::Int)
    crp = Int32[1]; ccv = Int32[]
    mark = zeros(Int32, nc); slot = zeros(Int32, nc); cols = Int32[]
    cpos = Vector{Int32}(undef, length(cv))
    @inbounds for I in 1:nc
        empty!(cols)
        for t in Rrp[I]:(Rrp[I+1]-1), q in rp[Rcv[t]]:(rp[Rcv[t]+1]-1)
            J = agg[cv[q]]
            mark[J] != I && (mark[J] = I; push!(cols, J))
        end
        sort!(cols)
        base = length(ccv)
        for (m, J) in enumerate(cols)
            push!(ccv, J); slot[J] = base + m
        end
        for t in Rrp[I]:(Rrp[I+1]-1), q in rp[Rcv[t]]:(rp[Rcv[t]+1]-1)
            cpos[q] = slot[agg[cv[q]]]
        end
        push!(crp, length(ccv) + 1)
    end
    nzc = length(ccv)
    Grp = zeros(Int32, nzc + 1)
    @inbounds for q in eachindex(cpos)
        Grp[cpos[q]+1] += 1
    end
    Grp[1] = 1
    @inbounds for p in 1:nzc
        Grp[p+1] += Grp[p]
    end
    next = Vector{Int32}(Grp[1:nzc])
    Gpos = Vector{Int32}(undef, length(cv))
    @inbounds for q in eachindex(cpos)
        p = cpos[q]
        Gpos[next[p]] = q
        next[p] += 1
    end
    return crp, ccv, Grp, Gpos
end

"""
    _mg_build(hrp, hcv, A, nodes, tag) -> _KrylovMG

Aggregation hierarchy of the square CSR matrix with host pattern `hrp`/`hcv`
(1-based, sorted, every diagonal stored) and device values `A` (kept by
reference as the finest level values), unknowns located by `nodes`
(`_mg_nodes`). Pattern work only: values are taken by [`_mg_refresh!`](@ref).
"""
function _mg_build(hrp::Vector{Int32}, hcv::Vector{Int32}, A, nodes, tag)
    devi(h) = copyto!(similar(A, Int32, length(h)), h)
    vec(m) = fill!(similar(A, m), 0.0)
    empty_i = devi(Int32[])
    levels = Any[]
    rp, cv, vals, nd = hrp, hcv, A, nodes
    smoother = tag.mg_smoother
    sizes = Int[]; nnzs = Int[]
    while true
        n = length(rp) - 1
        hd = _krylov_diag_positions(rp, cv, n)
        push!(sizes, n); push!(nnzs, length(cv))
        coarsest = n <= tag.mg_coarse || length(levels) + 1 >= tag.mg_levels
        agg = Rrp = Rcv = crp = ccv = Grp = Gpos = nothing
        nc = 0
        if !coarsest
            agg, ndc, nc = _mg_aggregate(nd)
            if nc > 0.9 * n                      # no useful reduction left
                coarsest = true
            else
                Rrp, Rcv = _mg_restriction(agg, nc)
                crp, ccv, Grp, Gpos = _mg_galerkin_pattern(rp, cv, agg, Rrp, Rcv, nc)
            end
        end
        rp_d = devi(rp); cv_d = devi(cv); d_d = devi(hd)
        if coarsest
            L = _MGLevel(n, rp_d, cv_d, d_d, vals, vec(0), vec(n), vec(n), vec(n), vec(0),
                         nothing, 0, empty_i, empty_i, empty_i, empty_i, empty_i, nothing)
            push!(levels, L)
            break
        end
        sm = smoother === :ilu0 ? _KrylovSide(rp_d, cv_d, d_d, vals, true) : nothing
        L = _MGLevel(n, rp_d, cv_d, d_d, vals, smoother === :ilu0 ? vec(0) : vec(n),
                     vec(n), vec(n), vec(n), vec(smoother === :ilu0 ? n : 0), sm, nc,
                     devi(agg), devi(Rrp), devi(Rcv), devi(Grp), devi(Gpos), nothing)
        push!(levels, L)
        rp, cv, vals, nd = crp, ccv, vec(length(ccv)), ndc
    end
    lv = [L for L in levels]                     # concrete element type
    info = (; levels = length(lv), sizes, nnz = nnzs,
            operator_complexity = sum(nnzs) / first(nnzs), grid_complexity = sum(sizes) / first(sizes),
            cycle = tag.mg_cycle, smoother, omega = _mg_omega(tag), sweeps = tag.mg_sweeps,
            cycles = tag.mg_cycles, scale = tag.mg_scale)
    return _KrylovMG(lv, tag.mg_cycle, smoother, _mg_omega(tag), tag.mg_sweeps, tag.mg_cycles,
                     tag.mg_scale, info)
end

# ---- device kernels ----

@kernel function _kmg_galerkin_kernel!(Ac, @Const(Grp), @Const(Gpos), @Const(A))
    q = @index(Global)
    @inbounds begin
        s = 0.0
        for t in Grp[q]:(Grp[q+1]-Int32(1))
            s += A[Gpos[t]]
        end
        Ac[q] = s
    end
end

@kernel function _kmg_dinv_kernel!(dinv, @Const(rowptr), @Const(diag), @Const(A), omega, l1)
    i = @index(Global)
    @inbounds begin
        d = A[diag[i]]
        if l1
            s = 0.0
            for q in rowptr[i]:(rowptr[i+1]-Int32(1))
                s += abs(A[q])
            end
            d = d >= 0 ? s : -s
        end
        dinv[i] = d == 0 ? 0.0 : omega / d
    end
end

# r = b − A x
@kernel function _kmg_resid_kernel!(r, @Const(rowptr), @Const(colval), @Const(A), @Const(x), @Const(b))
    i = @index(Global)
    @inbounds begin
        s = b[i]
        for q in rowptr[i]:(rowptr[i+1]-Int32(1))
            s -= A[q] * x[colval[q]]
        end
        r[i] = s
    end
end

# bc = Pᵀ r
@kernel function _kmg_restrict_kernel!(bc, @Const(Rrp), @Const(Rcv), @Const(r))
    ci = @index(Global)
    @inbounds begin
        s = 0.0
        for t in Rrp[ci]:(Rrp[ci+1]-Int32(1))
            s += r[Rcv[t]]
        end
        bc[ci] = s
    end
end

# x += scale P xc
@kernel function _kmg_prolong_kernel!(x, @Const(agg), @Const(xc), scale)
    i = @index(Global)
    @inbounds x[i] += scale * xc[agg[i]]
end

_mg_backend(mg::_KrylovMG) = KernelAbstractions.get_backend(mg.levels[1].A)
_mg_sync(be) = KernelAbstractions.synchronize(be)

function _mg_resid!(be, L::_MGLevel)
    _kmg_resid_kernel!(be, 256)(L.r, L.rowptr, L.colval, L.A, L.x, L.b; ndrange = L.n)
    _mg_sync(be)
    return L.r
end

"""
    _mg_refresh!(mg, tag) -> mg

New finest values (already in `mg.levels[1].A`): smoother data of every level,
Galerkin values of every coarser level, coarsest LU (built on the first call,
refactorized afterwards).
"""
function _mg_refresh!(mg::_KrylovMG, tag)
    be = _mg_backend(mg)
    nl = length(mg.levels)
    for l in 1:nl
        L = mg.levels[l]
        if l == nl
            if L.coarse === nothing
                L.coarse = _simple_schur_lu(L.rowptr, L.colval, L.A, L.n)
            else
                _simple_schur_refactor!(L.coarse)
            end
            break
        end
        if L.smoother === nothing
            _kmg_dinv_kernel!(be, 256)(L.dinv, L.rowptr, L.diag, L.A, mg.omega,
                                      mg.smoother === :l1jacobi; ndrange = L.n)
        end
        C = mg.levels[l+1]
        _kmg_galerkin_kernel!(be, 256)(C.A, L.Grp, L.Gpos, L.A; ndrange = length(C.A))
        _mg_sync(be)
        if L.smoother !== nothing
            S = L.smoother::_KrylovSide
            copyto!(S.F, S.A)
            _krylov_ilu0!(S)
            _krylov_rebuild!(S, tag)
        end
    end
    return mg
end

# `nsweeps` smoothing sweeps on L.x for L.b; `zero_x`: L.x is taken as zero
# (the first sweep is then x = M⁻¹ b, no product).
function _mg_smooth!(be, mg::_KrylovMG, L::_MGLevel, nsweeps::Int, zero_x::Bool)
    if nsweeps == 0
        zero_x && fill!(L.x, 0.0)
        return L.x
    end
    for sweep in 1:nsweeps
        first = zero_x && sweep == 1
        if L.smoother === nothing
            if first
                L.x .= L.dinv .* L.b
            else
                _mg_resid!(be, L)
                L.x .+= L.dinv .* L.r
            end
        else
            S = L.smoother::_KrylovSide
            if first
                _krylov_precond!(L.t, S, L.b)
                L.x .= mg.omega .* L.t
            else
                _mg_resid!(be, L)
                _krylov_precond!(L.t, S, L.r)
                L.x .+= mg.omega .* L.t
            end
        end
    end
    _mg_sync(be)
    return L.x
end

function _mg_cycle!(be, mg::_KrylovMG, l::Int, zero_x::Bool)
    L = mg.levels[l]
    nl = length(mg.levels)
    if l == nl
        _mg_sync(be)
        _simple_schur_solve!(L.x, L.coarse, L.b)     # exact (overwrites x)
        _mg_sync(be)
        return L.x
    end
    _mg_smooth!(be, mg, L, mg.sweeps, zero_x)
    _mg_resid!(be, L)
    C = mg.levels[l+1]
    _kmg_restrict_kernel!(be, 256)(C.b, L.Rrp, L.Rcv, L.r; ndrange = C.n)
    _mg_sync(be)
    gamma = (mg.cycle === :W && l + 1 < nl) ? 2 : 1
    for c in 1:gamma
        _mg_cycle!(be, mg, l + 1, c == 1)
    end
    _kmg_prolong_kernel!(be, 256)(L.x, L.agg, C.x, mg.scale; ndrange = L.n)
    _mg_sync(be)
    _mg_smooth!(be, mg, L, mg.sweeps, false)
    return L.x
end

"""
    _mg_apply!(x, mg, b) -> x

`mg.cycles` multigrid cycles for `A x = b` from `x = 0` (a fixed linear
operator of `b`).
"""
function _mg_apply!(x, mg::_KrylovMG, b)
    be = _mg_backend(mg)
    L = mg.levels[1]
    L.b .= b
    _mg_sync(be)
    for c in 1:mg.cycles
        _mg_cycle!(be, mg, 1, c == 1)
    end
    x .= L.x
    _mg_sync(be)
    return x
end

# velocity-block hooks of the block SIMPLE preconditioner (linear_solve_krylov.jl)
_simple_avals(b::_KrylovMG) = b.levels[1].A
_simple_ablk_solve!(x, b::_KrylovMG, v) = _mg_apply!(x, b, v)

# ---- host views for tests and reports ----

# Host SparseMatrixCSC of level l (values as last refreshed).
function _mg_level_matrix(mg::_KrylovMG, l::Int)
    L = mg.levels[l]
    rp = Array(L.rowptr); cv = Array(L.colval); v = Array(L.A)
    I = Int[]; J = Int[]
    for i in 1:L.n, q in rp[i]:(rp[i+1]-1)
        push!(I, i); push!(J, cv[q])
    end
    return sparse(I, J, v, L.n, L.n)
end

# Host prolongation matrix from level l + 1 to level l.
function _mg_prolongation(mg::_KrylovMG, l::Int)
    L = mg.levels[l]
    agg = Array(L.agg)
    return sparse(1:L.n, Int.(agg), ones(L.n), L.n, L.nc)
end
