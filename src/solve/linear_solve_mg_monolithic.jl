# P4 of the Krylov branch: monolithic (coupled velocity–pressure) geometric
# multigrid with an additive Vanka smoother, `precond = :vanka`
# (src/solve/linear_solve_krylov.jl). Alternative to the block SIMPLE
# preconditioners P2/P3, whose segregated structure (S̃ = C − D diag(A)⁻¹ G)
# was measured to make the GMRES count grow with N whatever the block solvers
# (P3: 75 / 94 / 146 forward iterations at 16³ / 24³ / 32³).
#
# The hierarchy is built on the LOCAL block of the Jacobian (unknowns
# 1:nlocal, `nfields` fields per cell, pressure in slot `pslot`, velocity
# components in slots 1:3), all unknowns of a cell being coarsened together:
#
#   * coarse nodes: the P3 aggregation of src/solve/linear_solve_multigrid.jl
#     (`_mg_nodes`, `_mg_aggregate`: 2 × 2 × 2 blocks of cells, one coarse
#     node per (block, field slot, cell class), solid cells apart). u, v, w and
#     p of the cells of a block are therefore coarsened consistently; a coarse
#     node is again one field of one coarse cell, so every level has the same
#     block structure and the same construction applies;
#   * transfer operators: restriction R = sum over the block (piecewise
#     constant, the finite-volume restriction: the coarse residual of a block
#     is the sum of its cell residuals), prolongation P trilinear and
#     cell-centred (`vanka_prolong = :linear`, default: along each axis 3/4
#     from the own block and 1/4 from the nearest neighbouring block, in the
#     same class and field slot; a missing neighbour — domain boundary, or no
#     cell of that class in the block — is dropped and the weights
#     renormalised, so P 1 = 1). The orders of the transfers must exceed the
#     order of the equations for a mesh-independent cycle (m_R + m_P > 2):
#     piecewise constant on both sides (1 + 1) is what the P3 aggregation
#     uses and what gives it the measured stationary factor ≈ 0.9 on smooth
#     errors; sum restriction with linear prolongation (1 + 2) is the
#     cell-centred multigrid of Wesseling–Khalil, whose Galerkin operator
#     R A P reproduces the 2h finite-volume operator exactly in 1D (checked
#     by hand on the flux-form Laplacian) and a slightly wider one in 3D.
#     `vanka_prolong = :constant` keeps P = Rᵀ for comparison;
#   * coarse operators Petrov–Galerkin R J P of the full saddle-point block.
#     Galerkin is stable here: summing the continuity rows of a block cancels
#     the mass fluxes through its interior faces, so the coarse continuity
#     row is the net flux of the interpolated velocity through the block
#     boundary; the Rhie–Chow p–p entries (the fourth difference
#     (1, −4, 6, −4, 1)/4 · h d_f per axis) applied to the interpolant of a
#     coarse pressure keep a positive diagonal (3/8 · h d_f per axis per
#     fine cell) and a zero row sum, so the coarse C block is a smaller
#     stabilisation of the same kind; the pinned pressure row (p_P = 0)
#     contributes 3/4 to the diagonal of its block, so no coarse level is
#     singular. Rediscretisation is not needed (the tests check the coarse
#     pressure diagonals). The coarse pattern (union over the block rows of
#     the P footprints of their columns, up to 5 × 5 × 5 cells) is built on
#     the host once; values are one thread per coarse row on the device
#     (binary search in the sorted coarse row, no atomics, deterministic);
#   * smoother: additive Vanka. One patch per fluid pressure unknown t: the
#     dofs are t, the non-pressure unknowns of its cell and the face-normal
#     velocities of its six face neighbours (slot d of the neighbour along
#     axis d; periodic neighbours recognised by the wrap-around offset), kept
#     only when stored in row t of the level matrix and not solid: 10 dofs in
#     the interior, fewer at walls and solids, the same on every level (the
#     coarse rows are wider but the face-normal couplings stay the dominant
#     divergence terms) and on the rows of Jᵀ (row t of Jᵀ is the pressure
#     column of J, which holds the same velocities). The local saddle-point
#     matrix is gathered from the CSR rows of the patch, padded with the
#     identity, and inverted in place by Gauss–Jordan elimination with
#     partial pivoting, one thread per patch, at every refactorisation; a
#     sweep is then
#         x ← x + ω W Σ_t Pᵗ_t M_t⁻¹ P_t (b − J x)  (+ ω D⁻¹ on uncovered rows),
#     W = diag(1 / multiplicity): a velocity unknown lies in up to three
#     patches (its own cell and the two neighbours along its axis), the
#     pressure in one; unknowns in no patch (solid cells, identity rows) take
#     one damped Jacobi step. The local solutions are stored per patch and
#     summed per unknown (gather, no atomics). A single-cell patch (u, v, w,
#     p of one cell) would be useless on this collocated scheme: the
#     continuity row of a cell has a zero coefficient on the velocity of that
#     cell in the interior (ū_f = (u_P + u_N)/2 on opposite faces cancel), so
#     the 4 × 4 block is block-diagonal and the smoother degenerates to point
#     Jacobi. Including the tangential velocities of the neighbours (22 dofs)
#     adds no coupling to the continuity row and 5× the memory. The sweep is
#     a fixed linear operator, so one V- or W-cycle of it is a valid right
#     preconditioner for restarted GMRES (no FGMRES needed);
#   * cycle, coarsest level, knobs: as P3 (`mg_cycle`, `mg_sweeps`,
#     `mg_cycles`, `mg_coarse`, `mg_levels`, `mg_scale`), damping
#     `vanka_omega` (default 0.5: the local Fourier analyses of the literature
#     put the smoothing optimum of additive Vanka with multiplicity weights
#     around 0.7–0.8, but ω = 0.7 fails the D2 gate on the body-force duct
#     while ω = 0.5 with constant prolongation passes it at 32³, 48³ and 64³
#     in both benchmark cases; the
#     overlap of the velocity patches needs the damping),
#     coarsest level by the sparse LU hooks of P2 (UMFPACK / cuDSS);
#   * flow-rate border (and any extra beyond `nlocal`): identity in the
#     preconditioner, like P2/P3 and `border = :krylov`. The border is a
#     rank-one bordering of the preconditioned operator, which costs at most
#     one extra GMRES iteration in exact arithmetic, against one extra cycle
#     per refactorisation for a Sherman–Morrison correction;
#   * transposed solve (adjoint): a second hierarchy is built from the
#     transposed pattern (gathered Jᵀ) with the same rule (R = sum, P
#     trilinear on Jᵀ; the transfer orders are what matter, so it is not the
#     transpose of the forward hierarchy but an equally valid one), same
#     patches with transposed local matrices. No transposed factor solve.
#
# All device work is KernelAbstractions kernels and broadcasts (one code path
# for Array and CuArray, no scalar indexing, no device allocation in a
# solve); patterns, transfers and patches are built on the host once per
# geometry. Memory per level: the level matrix, 12 bytes per unknown for R,
# ≤ 96 bytes per unknown for P, M² + M doubles per patch (M ≤ 10).

using KernelAbstractions

mutable struct _VankaLevel{VI,VV}
    n::Int
    rowptr::VI
    colval::VI
    diag::VI
    A::VV                        # level matrix values (finest: the local block of J)
    np::Int                      # patches (fluid pressure unknowns)
    M::Int                       # largest patch
    pdof::VI                     # dofs of each patch, slot-major: pdof[(e-1) np + t]
    pcnt::VI                     # size of each patch
    Minv::VV                     # inverse local matrices, entry-major: Minv[((e-1) M + f-1) np + t]
    piv::VI                      # pivot record of the inversion (M per patch)
    pr::VV                       # pivot ratio of each patch (−1 = singular)
    dptr::VI                     # patch slots holding each unknown (CSR over unknowns)
    dslot::VI
    wgt::VV                      # ω / multiplicity of each covered unknown, 0 otherwise
    jac::VV                      # ω / a_ii of each uncovered unknown, 0 otherwise
    Y::VV                        # local solutions, slot-major (M np)
    x::VV
    b::VV
    r::VV
    nc::Int                      # size of the next coarser level (0 on the coarsest)
    Rrp::VI                      # restriction, CSR over coarse nodes (fine node, weight)
    Rcv::VI
    Rw::VV
    Prp::VI                      # prolongation, CSR over fine nodes (coarse node, weight)
    Pcv::VI
    Pw::VV
    coarse::Any                  # coarsest level: sparse LU object (built on first refresh)
end

mutable struct _KrylovVanka{VI,VV,L}
    nfields::Int
    pslot::Int
    nlocal::Int
    n::Int
    J::VV                        # operator values (not owned)
    posL::Union{Nothing,VI}      # J position of each local-block entry (nothing: no extras, aliased)
    lidx::VI                     # 1:nlocal on the device
    geom::Any
    levels::Vector{L}
    cycle::Symbol
    omega::Float64
    sweeps::Int
    cycles::Int
    scale::Float64
    pr_host::Vector{Vector{Float64}}
    info::NamedTuple
end

# ---- host construction ----

# Periodic axes of the fine grid, read from the pattern: a fluid continuity
# row holds the face-normal velocity of a cell at the wrap-around offset
# ±(N − 1) along a periodic axis (never otherwise: its stencil stops at ±1).
function _vanka_periodic(rp::Vector{Int32}, cv::Vector{Int32}, nodes, pslot::Int)
    n = length(rp) - 1
    comp = nodes.comp; cls = nodes.cls; ijk = nodes.ijk
    dims = ntuple(d -> n > 0 ? Int(maximum(view(ijk, d, :))) : 1, 3)
    per = fill(false, 3)
    @inbounds for i in 1:n
        (comp[i] == pslot && cls[i] == 0) || continue
        for q in rp[i]:(rp[i+1]-1)
            j = cv[q]
            (1 <= comp[j] <= 3 && cls[j] == 0) || continue
            a = Int(comp[j])
            dims[a] > 2 && abs(ijk[a, j] - ijk[a, i]) == dims[a] - 1 && (per[a] = true)
        end
    end
    return (per[1], per[2], per[3])
end

# Vanka patches of one level from its pattern and nodes: one patch per fluid
# pressure unknown t (field slot `pslot`, class 0), dofs = t, the non-pressure
# unknowns of the same cell and the face-normal velocities (slot d along axis
# d, offset ±1 or the wrap-around of a periodic axis) of the neighbouring
# cells, when stored in row t and fluid; sorted. Returns the slot-major dof
# table, the patch sizes, the largest patch, the CSR (dptr, dslot) of the
# slots holding each unknown, and the multiplicity of each unknown.
function _vanka_patches(rp::Vector{Int32}, cv::Vector{Int32}, nodes, pslot::Int, periodic::NTuple{3,Bool})
    n = length(rp) - 1
    comp = nodes.comp; cls = nodes.cls; ijk = nodes.ijk
    dims = ntuple(d -> n > 0 ? Int(maximum(view(ijk, d, :))) : 1, 3)
    adjacent(δ, N, per) = abs(δ) == 1 || (per && N > 2 && abs(δ) == N - 1)
    centres = Int32[i for i in 1:n if comp[i] == pslot && cls[i] == 0]
    np = length(centres)
    lists = Vector{Vector{Int32}}(undef, np)
    M = 0
    @inbounds for (t, i) in enumerate(centres)
        d = Int32[i]
        for q in rp[i]:(rp[i+1]-1)
            j = cv[q]
            (comp[j] != pslot && cls[j] == 0) || continue
            δ = (ijk[1, j] - ijk[1, i], ijk[2, j] - ijk[2, i], ijk[3, j] - ijk[3, i])
            if δ == (0, 0, 0)
                push!(d, j)
            elseif 1 <= comp[j] <= 3
                a = Int(comp[j])
                if adjacent(δ[a], dims[a], periodic[a]) && all(b -> b == a || δ[b] == 0, 1:3)
                    push!(d, j)
                end
            end
        end
        sort!(d)
        lists[t] = d
        M = max(M, length(d))
    end
    pdof = zeros(Int32, M * np); pcnt = zeros(Int32, np)
    mult = zeros(Int32, n)
    @inbounds for t in 1:np
        d = lists[t]; pcnt[t] = length(d)
        for (e, j) in enumerate(d)
            pdof[(e-1)*np+t] = j
            mult[j] += 1
        end
    end
    dptr = zeros(Int32, n + 1)
    dptr[1] = 1
    @inbounds for i in 1:n
        dptr[i+1] = dptr[i] + mult[i]
    end
    next = Vector{Int32}(dptr[1:n])
    dslot = zeros(Int32, dptr[n+1] - 1)
    @inbounds for t in 1:np, e in 1:pcnt[t]
        j = pdof[(e-1)*np+t]
        dslot[next[j]] = Int32((e - 1) * np + t)
        next[j] += 1
    end
    return pdof, pcnt, M, np, dptr, dslot, mult
end

# Pattern of the local block J[1:nlocal, 1:nlocal] and the J position of each
# of its entries (nothing when the block is the whole matrix).
function _vanka_local_pattern(hrp::Vector{Int32}, hcv::Vector{Int32}, nlocal::Int)
    length(hrp) - 1 == nlocal && return hrp, hcv, nothing
    lrp = Int32[1]; lcv = Int32[]; posL = Int32[]
    @inbounds for i in 1:nlocal
        for q in hrp[i]:(hrp[i+1]-1)
            c = hcv[q]
            c <= nlocal && (push!(lcv, c); push!(posL, Int32(q)))
        end
        push!(lrp, length(lcv) + 1)
    end
    return lrp, lcv, posL
end

# Prolongation from the coarse nodes `ndc` to the fine nodes `nd` (CSR over
# fine nodes with weights): trilinear cell-centred (`:linear`) within the
# class and field slot of the node (the neighbouring block wraps around on a
# periodic axis), or piecewise constant (`:constant`, P = Rᵀ). Every row sums
# to 1.
function _vanka_prolongation(nd, agg::Vector{Int32}, ndc, nc::Int, kind::Symbol, periodic::NTuple{3,Bool})
    n = length(nd.comp)
    if kind === :constant
        return Vector{Int32}(1:n+1), copy(agg), ones(n)
    end
    id = Dict{NTuple{5,Int32},Int32}()
    @inbounds for m in 1:nc
        id[(ndc.ijk[1, m], ndc.ijk[2, m], ndc.ijk[3, m], ndc.comp[m], Int32(ndc.cls[m]))] = Int32(m)
    end
    nbl = ntuple(d -> nc > 0 ? Int(maximum(view(ndc.ijk, d, :))) : 1, 3)
    wrap(i, d) = !periodic[d] ? i : i < 1 ? i + nbl[d] : i > nbl[d] ? i - nbl[d] : i
    Prp = Int32[1]; Pcv = Int32[]; Pw = Float64[]
    cand = Tuple{Int32,Float64}[]
    @inbounds for a in 1:n
        empty!(cand)
        I0 = ntuple(d -> (Int(nd.ijk[d, a]) + 1) ÷ 2, 3)
        sg = ntuple(d -> isodd(nd.ijk[d, a]) ? -1 : 1, 3)
        own = get(id, (Int32(I0[1]), Int32(I0[2]), Int32(I0[3]), nd.comp[a], Int32(nd.cls[a])), Int32(0))
        own == agg[a] || throw(ErrorException("vanka prolongation: node $a is not interpolated from its own block"))
        for sz in 0:1, sy in 0:1, sx in 0:1
            Ib = (wrap(I0[1] + sx * sg[1], 1), wrap(I0[2] + sy * sg[2], 2), wrap(I0[3] + sz * sg[3], 3))
            w = (sx == 0 ? 0.75 : 0.25) * (sy == 0 ? 0.75 : 0.25) * (sz == 0 ? 0.75 : 0.25)
            m = get(id, (Int32(Ib[1]), Int32(Ib[2]), Int32(Ib[3]), nd.comp[a], Int32(nd.cls[a])), Int32(0))
            m > 0 && any(i -> i != 0, (sx, sy, sz)) && m == own && continue   # one block only on the axis
            m > 0 && push!(cand, (m, w))
        end
        isempty(cand) && throw(ErrorException("vanka prolongation: fine node $a has no coarse node"))
        s = sum(last, cand)
        sort!(cand; by = first)
        for (m, w) in cand
            push!(Pcv, m); push!(Pw, w / s)
        end
        push!(Prp, Int32(length(Pcv) + 1))
    end
    return Prp, Pcv, Pw
end

# Pattern (sorted columns) of R A P for CSR transfers R (coarse → fine) and P
# (fine → coarse).
function _vanka_galerkin_pattern(rp::Vector{Int32}, cv::Vector{Int32}, Rrp::Vector{Int32},
                                 Rcv::Vector{Int32}, Prp::Vector{Int32}, Pcv::Vector{Int32}, nc::Int)
    crp = Int32[1]; ccv = Int32[]
    mark = zeros(Int32, nc); cols = Int32[]
    @inbounds for I in 1:nc
        empty!(cols)
        for t in Rrp[I]:(Rrp[I+1]-1)
            i = Rcv[t]
            for q in rp[i]:(rp[i+1]-1)
                j = cv[q]
                for s in Prp[j]:(Prp[j+1]-1)
                    J = Pcv[s]
                    mark[J] != I && (mark[J] = I; push!(cols, J))
                end
            end
        end
        sort!(cols)
        append!(ccv, cols)
        push!(crp, Int32(length(ccv) + 1))
    end
    return crp, ccv
end

"""
    _vanka_build(hrp, hcv, nfields, pslot, nlocal, J, tag, geom) -> _KrylovVanka

Monolithic multigrid hierarchy with Vanka patches of the local block of the
CSR operator with host pattern `hrp`/`hcv` and device values `J` (kept by
reference). Pattern work only: values are taken by `_simple_refresh!`.
"""
function _vanka_build(hrp::Vector{Int32}, hcv::Vector{Int32}, nfields::Int, pslot::Int,
                      nlocal::Int, J, tag::KrylovBackendTag, geom)
    n = length(hrp) - 1
    geom === nothing && throw(ArgumentError("precond = :vanka needs the grid geometry of the matrix"))
    (nfields > 0 && nlocal > 0) || throw(ArgumentError("precond = :vanka needs the block layout of the matrix"))
    lrp, lcv, posL = _vanka_local_pattern(hrp, hcv, nlocal)
    nodes = _mg_nodes(Int32[(i - 1) ÷ nfields + 1 for i in 1:nlocal],
                      Int32[(i - 1) % nfields + 1 for i in 1:nlocal], geom)
    periodic = _vanka_periodic(lrp, lcv, nodes, pslot)
    devi(h) = copyto!(similar(J, Int32, length(h)), h)
    devf(h) = copyto!(similar(J, length(h)), h)
    vec(m) = fill!(similar(J, m), 0.0)
    omega = tag.vanka_omega
    levels = Any[]
    sizes = Int[]; nnzs = Int[]; patches = Int[]; pmax = Int[]
    pr_host = Vector{Float64}[]
    rp, cv, nd = lrp, lcv, nodes
    vals = posL === nothing ? J : vec(length(lcv))
    while true
        nl = length(rp) - 1
        hd = _krylov_diag_positions(rp, cv, nl)
        push!(sizes, nl); push!(nnzs, length(cv))
        coarsest = nl <= tag.mg_coarse || length(levels) + 1 >= tag.mg_levels
        Rrp = Rcv = Prp = Pcv = crp = ccv = ndc = nothing
        Rw = Pw = nothing
        nc = 0
        if !coarsest
            agg, ndc, nc = _mg_aggregate(nd)
            if nc > 0.9 * nl                       # no useful reduction left
                coarsest = true
            else
                Rrp, Rcv = _mg_restriction(agg, nc)
                Rw = ones(length(Rcv))
                Prp, Pcv, Pw = _vanka_prolongation(nd, agg, ndc, nc, tag.vanka_prolong, periodic)
                crp, ccv = _vanka_galerkin_pattern(rp, cv, Rrp, Rcv, Prp, Pcv, nc)
            end
        end
        rp_d = devi(rp); cv_d = devi(cv); d_d = devi(hd)
        empty_i = devi(Int32[])
        if coarsest
            L = _VankaLevel(nl, rp_d, cv_d, d_d, vals, 0, 0, empty_i, empty_i, vec(0), empty_i, vec(0),
                            empty_i, empty_i, vec(0), vec(0), vec(0), vec(nl), vec(nl), vec(0),
                            0, empty_i, empty_i, vec(0), empty_i, empty_i, vec(0), nothing)
            push!(levels, L); push!(patches, 0); push!(pmax, 0)
            break
        end
        pdof, pcnt, M, np, dptr, dslot, mult = _vanka_patches(rp, cv, nd, pslot, periodic)
        np > 0 || throw(ArgumentError("precond = :vanka: level $(length(levels) + 1) has no fluid pressure unknown"))
        wgt = Float64[m > 0 ? omega / m : 0.0 for m in mult]
        L = _VankaLevel(nl, rp_d, cv_d, d_d, vals, np, M, devi(pdof), devi(pcnt), vec(M * M * np),
                        devi(zeros(Int32, M * np)), vec(np), devi(dptr), devi(dslot), devf(wgt),
                        vec(nl), vec(M * np), vec(nl), vec(nl), vec(nl),
                        nc, devi(Rrp), devi(Rcv), devf(Rw), devi(Prp), devi(Pcv), devf(Pw), nothing)
        push!(levels, L); push!(patches, np); push!(pmax, M)
        push!(pr_host, zeros(np))
        rp, cv, vals, nd = crp, ccv, vec(length(ccv)), ndc
    end
    lv = [L for L in levels]                     # concrete element type
    nu = nlocal - nlocal ÷ nfields
    info = (; nu, np = nlocal ÷ nfields, nnz_A = 0, nnz_G = 0, nnz_D = 0, nnz_S = 0, schur = :none,
            mg_A = nothing, mg_S = nothing,
            vanka = (; levels = length(lv), sizes, nnz = nnzs, patches, patch_max = pmax,
                     operator_complexity = sum(nnzs) / first(nnzs),
                     grid_complexity = sum(sizes) / first(sizes), prolong = tag.vanka_prolong,
                     periodic, cycle = tag.mg_cycle, omega, sweeps = tag.mg_sweeps,
                     cycles = tag.mg_cycles, scale = tag.mg_scale, extras = n - nlocal))
    return _KrylovVanka(nfields, pslot, nlocal, n, J, posL === nothing ? nothing : devi(posL),
                        devi(Int32.(1:nlocal)), geom, lv, tag.mg_cycle, omega, tag.mg_sweeps,
                        tag.mg_cycles, tag.mg_scale, pr_host, info)
end

# Block preconditioner of one operator per `tag.precond` (hook of
# linear_solve_krylov.jl: P2/P3 block SIMPLE, or P4 monolithic Vanka).
_krylov_block_build(hrp, hcv, nfields, pslot, nlocal, A, tag::KrylovBackendTag, geom) =
    tag.precond === :vanka ? _vanka_build(hrp, hcv, nfields, pslot, nlocal, A, tag, geom) :
                             _simple_build(hrp, hcv, nfields, pslot, nlocal, A, tag, geom)

# ---- device kernels (one thread per patch, per unknown or per coarse row) ----

# Local matrix of each patch gathered from the CSR rows of its dofs (sorted
# dofs against sorted columns: one merge per row), padded with the identity up
# to M, inverted in place by Gauss–Jordan elimination with partial pivoting
# (row swaps recorded in `piv`, undone as column swaps at the end). `pr[t]` is
# the smallest pivot relative to the largest local entry, −1 on a zero or
# non-finite pivot.
@kernel function _kvk_invert_kernel!(Minv, piv, pr, @Const(rowptr), @Const(colval), @Const(A),
                                     @Const(pdof), @Const(pcnt), np, M)
    t = @index(Global)
    @inbounds begin
        m = pcnt[t]
        amax = 0.0
        for e in 1:M
            for f in 1:M
                Minv[((e-1)*M+(f-1))*np+t] = (e == f && e > m) ? 1.0 : 0.0
            end
            if e <= m
                i = pdof[(e-1)*np+t]
                q = rowptr[i]; qend = rowptr[i+1] - Int32(1)
                f = 1
                while q <= qend && f <= m
                    c = colval[q]; j = pdof[(f-1)*np+t]
                    if c == j
                        v = A[q]
                        Minv[((e-1)*M+(f-1))*np+t] = v
                        amax = max(amax, abs(v))
                        q += Int32(1); f += 1
                    elseif c < j
                        q += Int32(1)
                    else
                        f += 1
                    end
                end
            end
        end
        rmin = Inf
        for k in 1:M
            p = k; best = abs(Minv[((k-1)*M+(k-1))*np+t])
            for r in (k+1):M
                v = abs(Minv[((r-1)*M+(k-1))*np+t])
                v > best && (best = v; p = r)
            end
            piv[(k-1)*np+t] = p
            if p != k
                for f in 1:M
                    a = Minv[((k-1)*M+(f-1))*np+t]; b = Minv[((p-1)*M+(f-1))*np+t]
                    Minv[((k-1)*M+(f-1))*np+t] = b; Minv[((p-1)*M+(f-1))*np+t] = a
                end
            end
            d = Minv[((k-1)*M+(k-1))*np+t]
            if d == 0.0 || !isfinite(d)
                rmin = -1.0
                break
            end
            rmin = min(rmin, amax > 0 ? abs(d) / amax : abs(d))
            dinv = 1.0 / d
            Minv[((k-1)*M+(k-1))*np+t] = 1.0
            for f in 1:M
                Minv[((k-1)*M+(f-1))*np+t] *= dinv
            end
            for r in 1:M
                r == k && continue
                fct = Minv[((r-1)*M+(k-1))*np+t]
                fct == 0.0 && continue
                Minv[((r-1)*M+(k-1))*np+t] = 0.0
                for f in 1:M
                    Minv[((r-1)*M+(f-1))*np+t] -= fct * Minv[((k-1)*M+(f-1))*np+t]
                end
            end
        end
        if rmin >= 0
            for k in M:-1:1
                p = piv[(k-1)*np+t]
                p == k && continue
                for r in 1:M
                    a = Minv[((r-1)*M+(k-1))*np+t]; b = Minv[((r-1)*M+(p-1))*np+t]
                    Minv[((r-1)*M+(k-1))*np+t] = b; Minv[((r-1)*M+(p-1))*np+t] = a
                end
            end
        end
        pr[t] = rmin
    end
end

# ω / a_ii on the unknowns of no patch (wgt = 0), 0 elsewhere
@kernel function _kvk_jac_kernel!(jac, @Const(wgt), @Const(diag), @Const(A), omega)
    i = @index(Global)
    @inbounds begin
        d = A[diag[i]]
        jac[i] = (wgt[i] == 0.0 && d != 0.0) ? omega / d : 0.0
    end
end

# Y_t = M_t⁻¹ r|patch t
@kernel function _kvk_local_kernel!(Y, @Const(Minv), @Const(pdof), @Const(pcnt), @Const(r), np, M)
    t = @index(Global)
    @inbounds begin
        m = pcnt[t]
        for e in 1:m
            s = 0.0
            for f in 1:m
                s += Minv[((e-1)*M+(f-1))*np+t] * r[pdof[(f-1)*np+t]]
            end
            Y[(e-1)*np+t] = s
        end
    end
end

# x_i += ω / mult_i Σ_{slots of i} Y + jac_i r_i
@kernel function _kvk_update_kernel!(x, @Const(dptr), @Const(dslot), @Const(Y), @Const(wgt),
                                     @Const(jac), @Const(r))
    i = @index(Global)
    @inbounds begin
        s = 0.0
        for k in dptr[i]:(dptr[i+1]-Int32(1))
            s += Y[dslot[k]]
        end
        x[i] += wgt[i] * s + jac[i] * r[i]
    end
end

# Ac = R A P, one thread per coarse row (sorted coarse row, binary search of
# each product column; no stored triple products, no atomics).
@kernel function _kvk_galerkin_kernel!(Ac, @Const(crp), @Const(ccv), @Const(Rrp), @Const(Rcv), @Const(Rw),
                                       @Const(rp), @Const(cv), @Const(A), @Const(Prp), @Const(Pcv),
                                       @Const(Pw))
    ic = @index(Global)
    @inbounds begin
        c0 = crp[ic]; c1 = crp[ic+1] - Int32(1)
        for c in c0:c1
            Ac[c] = 0.0
        end
        for t in Rrp[ic]:(Rrp[ic+1]-Int32(1))
            i = Rcv[t]; wi = Rw[t]
            for q in rp[i]:(rp[i+1]-Int32(1))
                a = wi * A[q]
                j = cv[q]
                for s in Prp[j]:(Prp[j+1]-Int32(1))
                    jc = Pcv[s]
                    lo = c0; hi = c1
                    while lo < hi
                        mid = (lo + hi) >> Int32(1)
                        if ccv[mid] < jc
                            lo = mid + Int32(1)
                        else
                            hi = mid
                        end
                    end
                    Ac[lo] += a * Pw[s]
                end
            end
        end
    end
end

# bc = R r
@kernel function _kvk_restrict_kernel!(bc, @Const(Rrp), @Const(Rcv), @Const(Rw), @Const(r))
    ic = @index(Global)
    @inbounds begin
        s = 0.0
        for t in Rrp[ic]:(Rrp[ic+1]-Int32(1))
            s += Rw[t] * r[Rcv[t]]
        end
        bc[ic] = s
    end
end

# x += scale P xc
@kernel function _kvk_prolong_kernel!(x, @Const(Prp), @Const(Pcv), @Const(Pw), @Const(xc), scale)
    i = @index(Global)
    @inbounds begin
        s = 0.0
        for t in Prp[i]:(Prp[i+1]-Int32(1))
            s += Pw[t] * xc[Pcv[t]]
        end
        x[i] += scale * s
    end
end

# ---- refresh, smoother, cycle, application ----

_vanka_backend(s::_KrylovVanka) = KernelAbstractions.get_backend(s.J)

# New values of J: local block gathered (when the matrix has extras), local
# Vanka inverses and Jacobi fallback of every level, Galerkin values of every
# coarser level, coarsest LU (built on the first call, refactorized
# afterwards). Returns the pivot report of the local inversions
# (`min_ratio`, `n_small` below `tag.pivot_tol`, `row_min` = patch of the
# minimum, numbered over the levels); a singular local block raises.
function _simple_refresh!(s::_KrylovVanka, tag::KrylovBackendTag)
    be = _vanka_backend(s)
    nl = length(s.levels)
    L1 = s.levels[1]
    if s.posL !== nothing
        _krylov_take_kernel!(be, 256)(L1.A, s.J, s.posL; ndrange = length(L1.A))
        _mg_sync(be)
    end
    rmin = Inf; nsmall = 0; tmin = 0; base = 0
    for l in 1:nl
        L = s.levels[l]
        if l == nl
            if L.coarse === nothing
                L.coarse = _simple_schur_lu(L.rowptr, L.colval, L.A, L.n)
            else
                _simple_schur_refactor!(L.coarse)
            end
            break
        end
        _kvk_invert_kernel!(be, 256)(L.Minv, L.piv, L.pr, L.rowptr, L.colval, L.A, L.pdof, L.pcnt,
                                     L.np, L.M; ndrange = L.np)
        _kvk_jac_kernel!(be, 256)(L.jac, L.wgt, L.diag, L.A, s.omega; ndrange = L.n)
        C = s.levels[l+1]
        _kvk_galerkin_kernel!(be, 256)(C.A, C.rowptr, C.colval, L.Rrp, L.Rcv, L.Rw, L.rowptr, L.colval,
                                       L.A, L.Prp, L.Pcv, L.Pw; ndrange = C.n)
        _mg_sync(be)
        ph = s.pr_host[l]
        copyto!(ph, L.pr)
        @inbounds for (t, r) in enumerate(ph)
            r < 0 && throw(ErrorException("Vanka multigrid preconditioner failed: singular local block " *
                                          "of patch $t on level $l (size $(L.n)); use another " *
                                          "preconditioner or a direct backend for this case."))
            r < rmin && (rmin = r; tmin = base + t)
            r < tag.pivot_tol && (nsmall += 1)
        end
        base += L.np
    end
    return (; min_ratio = rmin, n_small = nsmall, row_min = tmin)
end

# `nsweeps` additive Vanka sweeps on L.x for L.b; `zero_x`: L.x is taken as
# zero (the first residual is then b, no product).
function _vanka_smooth!(be, L::_VankaLevel, nsweeps::Int, zero_x::Bool)
    if nsweeps == 0
        zero_x && fill!(L.x, 0.0)
        return L.x
    end
    for sweep in 1:nsweeps
        if zero_x && sweep == 1
            fill!(L.x, 0.0)
            L.r .= L.b
            _mg_sync(be)
        else
            _kmg_resid_kernel!(be, 256)(L.r, L.rowptr, L.colval, L.A, L.x, L.b; ndrange = L.n)
            _mg_sync(be)
        end
        _kvk_local_kernel!(be, 256)(L.Y, L.Minv, L.pdof, L.pcnt, L.r, L.np, L.M; ndrange = L.np)
        _mg_sync(be)
        _kvk_update_kernel!(be, 256)(L.x, L.dptr, L.dslot, L.Y, L.wgt, L.jac, L.r; ndrange = L.n)
        _mg_sync(be)
    end
    return L.x
end

function _vanka_cycle!(be, s::_KrylovVanka, l::Int, zero_x::Bool)
    L = s.levels[l]
    nl = length(s.levels)
    if l == nl
        _mg_sync(be)
        _simple_schur_solve!(L.x, L.coarse, L.b)     # exact (overwrites x)
        _mg_sync(be)
        return L.x
    end
    _vanka_smooth!(be, L, s.sweeps, zero_x)
    _kmg_resid_kernel!(be, 256)(L.r, L.rowptr, L.colval, L.A, L.x, L.b; ndrange = L.n)
    _mg_sync(be)
    C = s.levels[l+1]
    _kvk_restrict_kernel!(be, 256)(C.b, L.Rrp, L.Rcv, L.Rw, L.r; ndrange = C.n)
    _mg_sync(be)
    gamma = (s.cycle === :W && l + 1 < nl) ? 2 : 1
    for c in 1:gamma
        _vanka_cycle!(be, s, l + 1, c == 1)
    end
    _kvk_prolong_kernel!(be, 256)(L.x, L.Prp, L.Pcv, L.Pw, C.x, s.scale; ndrange = L.n)
    _mg_sync(be)
    _vanka_smooth!(be, L, s.sweeps, false)
    return L.x
end

# z = M⁻¹ v: `cycles` monolithic cycles on the local block from x = 0 (a
# fixed linear operator of v), identity on the extras.
function _simple_apply!(z, s::_KrylovVanka, v)
    be = _vanka_backend(s)
    L = s.levels[1]
    z .= v                                                   # extras: identity
    _mg_sync(be)
    if s.posL === nothing
        L.b .= z
    else
        _krylov_take_kernel!(be, 256)(L.b, z, s.lidx; ndrange = s.nlocal)
    end
    _mg_sync(be)
    for c in 1:s.cycles
        _vanka_cycle!(be, s, 1, c == 1)
    end
    if s.posL === nothing
        z .= L.x
    else
        _krylov_put_kernel!(be, 256)(z, L.x, s.lidx; ndrange = s.nlocal)
    end
    _mg_sync(be)
    return z
end

# ---- host views for tests and reports ----

# Host SparseMatrixCSC of level l (values as last refreshed).
function _vanka_level_matrix(s::_KrylovVanka, l::Int)
    L = s.levels[l]
    rp = Array(L.rowptr); cv = Array(L.colval); v = Array(L.A)
    I = Int[]; J = Int[]
    for i in 1:L.n, q in rp[i]:(rp[i+1]-1)
        push!(I, i); push!(J, cv[q])
    end
    return sparse(I, J, v, L.n, L.n)
end

# Host restriction (level l → l + 1) and prolongation (l + 1 → l) matrices.
function _vanka_restriction_matrix(s::_KrylovVanka, l::Int)
    L = s.levels[l]
    rp = Array(L.Rrp); cv = Array(L.Rcv); w = Array(L.Rw)
    I = Int[]; J = Int[]
    for i in 1:L.nc, q in rp[i]:(rp[i+1]-1)
        push!(I, i); push!(J, cv[q])
    end
    return sparse(I, J, w, L.nc, L.n)
end

function _vanka_prolongation_matrix(s::_KrylovVanka, l::Int)
    L = s.levels[l]
    rp = Array(L.Prp); cv = Array(L.Pcv); w = Array(L.Pw)
    I = Int[]; J = Int[]
    for i in 1:L.n, q in rp[i]:(rp[i+1]-1)
        push!(I, i); push!(J, cv[q])
    end
    return sparse(I, J, w, L.n, L.nc)
end

# Host dofs of patch t of level l, and the stored inverse of its local matrix.
function _vanka_patch(s::_KrylovVanka, l::Int, t::Int)
    L = s.levels[l]
    np = L.np; M = L.M
    pcnt = Array(L.pcnt); pdof = Array(L.pdof); Minv = Array(L.Minv)
    m = Int(pcnt[t])
    dofs = Int[pdof[(e-1)*np+t] for e in 1:m]
    Mi = [Minv[((e-1)*M+(f-1))*np+t] for e in 1:m, f in 1:m]
    return dofs, Mi
end

"""
    lin_krylov_stationary(cache, b; iters=5, transpose=false, local_block=false) -> NamedTuple

Residual history of the stationary iteration x ← x + M⁻¹ (b − A x) from x = 0
with the preconditioner M of a Krylov cache (one application per step):
`(; norms, ratios, factor)`, `ratios` the residual ratio of each step and
`factor` the geometric mean over the `iters` steps (the convergence factor
of one preconditioner application, e.g. one multigrid cycle).
`local_block = true` iterates on the local block only (the extras of `b`,
e.g. the flow-rate border, are zeroed and their rows ignored), which is the
convergence factor of the block preconditioner itself (the block
preconditioners leave the extras to GMRES; without a block preconditioner the option has no effect).
Uses the GMRES work vectors; no device allocation.
"""
function lin_krylov_stationary(cache::LinearSolveCache, b::AbstractVector; iters::Int = 5,
                               transpose::Bool = false, local_block::Bool = false)
    cache.backend isa KrylovBackendTag ||
        throw(ArgumentError("lin_krylov_stationary needs a Krylov cache (KrylovBackendTag)"))
    f = cache.factor
    side = transpose ? _krylov_tside!(f) : f.fwd
    _krylov_copy!(f.rhs, b)
    x = f.y; r = f.work.r
    n = length(x)
    nloc = local_block && side.simple !== nothing ? side.simple.nlocal : n
    nloc < n && fill!(view(f.rhs, (nloc+1):n), 0.0)
    fill!(x, 0.0)
    r .= f.rhs
    norms = Float64[norm(r)]
    for _ in 1:iters
        _krylov_apply!(side.z, side, r)                      # z = M⁻¹ r
        x .+= side.z
        _krylov_matvec!(side.t3, side, x)
        r .= f.rhs .- side.t3
        nloc < n && fill!(view(r, (nloc+1):n), 0.0)
        push!(norms, norm(r))
    end
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(x))
    ratios = norms[2:end] ./ max.(norms[1:end-1], eps())
    factor = iters > 0 && norms[1] > 0 ? (norms[end] / norms[1])^(1 / iters) : NaN
    return (; norms, ratios, factor)
end
