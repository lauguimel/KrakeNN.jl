# Jacobian pattern of the 3D steady FV residual (fv_steady_residual_3d.jl).
# Plan "solveur FV stationnaire 3D", step 2: the coloured dual assembly, the
# Newton loop and the cuDSS seam of fv_newton.jl are shared unchanged; only the
# structural pattern is dimension specific. It follows the residual's own
# face classification (walls, outlets, periodic wrap, solid cells), so it is a
# superset of the support for every state and for Stokes and Navier-Stokes.

# Neighbour of cell c across direction d through an interior face, or (0, 0, 0).
function _fv3_nb(g, tag, c, d)
    c[1] == 0 && return (0, 0, 0)
    kind, ii, jj, kk = _fv_face_kind(g, tag, c[1], c[2], c[3], d)
    return kind == _FV_F_INT ? (ii, jj, kk) : (0, 0, 0)
end

# The four directions perpendicular to the axis of direction d.
_fv3_tangential_dirs(d) = _fv3_axis(d) == 1 ? (3, 4, 5, 6) :
                          _fv3_axis(d) == 2 ? (1, 2, 5, 6) : (1, 2, 3, 4)

"""
Newtonian dependencies of the rows of a fluid cell P (component => cells):

- momentum row a: component a on the star (P and its 6 face neighbours);
  each other component b on the star plus the four cells
  `nb(nb(P, ±b), ±a)` (transpose stress term `∂u_b/∂x_a` on the b faces);
  pressure on the star plus the second neighbours along each axis
  (Rhie-Chow mean-gradient term);
- continuity row: component a at P and its two neighbours along axis a,
  pressure as in momentum; pinned: `p_P` only; solid: identity.

Interior rows carry 42 (momentum) and 22 (continuity) entries.
"""
function _fv3_newtonian_dependencies(g, tag, i, j, k)
    P = (i, j, k)
    nbs = ntuple(d -> _fv3_nb(g, tag, P, d), 6)
    star = (P, nbs...)
    axis2 = ntuple(d -> _fv3_nb(g, tag, nbs[d], d), 6)
    cross(a, b) = (_fv3_nb(g, tag, nbs[2b - 1], 2a - 1), _fv3_nb(g, tag, nbs[2b - 1], 2a),
                   _fv3_nb(g, tag, nbs[2b], 2a - 1), _fv3_nb(g, tag, nbs[2b], 2a))
    pres = (star..., axis2...)
    mom(a) = ntuple(b -> (b, b == a ? star : (star..., cross(a, b)...)), 3)
    cont = ((1, (P, nbs[1], nbs[2])), (2, (P, nbs[3], nbs[4])), (3, (P, nbs[5], nbs[6])))
    return (; P, nbs, pres, mom, cont)
end

function _fv3_push_newtonian!(I, J, g, tag, i, j, k)
    t = tag[i, j, k]
    if t == FV_CELL_SOLID
        for c in 1:_fv_nfields(g)
            r = fv_index(g, i, j, k, c); push!(I, r); push!(J, r)
        end
        return nothing
    end
    s = _fv3_newtonian_dependencies(g, tag, i, j, k)
    add(row, comp, cells) = for c in cells
        c[1] == 0 && continue
        push!(I, row); push!(J, fv_index(g, c[1], c[2], c[3], comp))
    end
    for a in 1:3
        row = fv_index(g, i, j, k, a)
        for (comp, cells) in s.mom(a)
            add(row, comp, cells)
        end
        add(row, 4, s.pres)
    end
    row = fv_index(g, i, j, k, 4)
    if t == FV_CELL_PINNED
        push!(I, row); push!(J, row)
    else
        for (comp, cells) in s.cont
            add(row, comp, cells)
        end
        add(row, 4, s.pres)
    end
    for scalar in 1:_fv_nscalars(g.layout)
        r = fv_index(g, i, j, k, _fv_scalar_slot(g.layout, scalar)); push!(I, r); push!(J, r)
    end
    return nothing
end

# State-dependent viscosity: the face viscosity of face d of P reads (u, v, w)
# at {P, N, tang(P), tang(N)} (interior) or {P, tang(P)} (wall); momentum rows
# read every non-outlet face, an unpinned continuity row its interior faces.
_fv3_state_viscosity(v) = false
_fv3_state_viscosity(::FVGNFViscosity) = true
_fv3_state_viscosity(::FVLawViscosity) = true

function _fv3_push_viscosity!(I, J, g, tag, i, j, k)
    tag[i, j, k] == FV_CELL_SOLID && return nothing
    P = (i, j, k)
    pinned = tag[i, j, k] == FV_CELL_PINNED
    for d in 1:6
        kind, ii, jj, kk = _fv_face_kind(g, tag, i, j, k, d)
        kind == _FV_F_OUTLET && continue
        td = _fv3_tangential_dirs(d)
        tP = map(e -> _fv3_nb(g, tag, P, e), td)
        cells = if kind == _FV_F_INT
            N = (ii, jj, kk)
            (P, N, tP..., map(e -> _fv3_nb(g, tag, N, e), td)...)
        else
            (P, tP...)
        end
        for equation in 1:4
            (equation <= 3 || (kind == _FV_F_INT && !pinned)) || continue
            row = fv_index(g, i, j, k, equation)
            for c in cells, comp in 1:3
                c[1] == 0 && continue
                push!(I, row); push!(J, fv_index(g, c[1], c[2], c[3], comp))
            end
        end
    end
    return nothing
end

"""
    fv_jacobian_pattern(prob::FVSteadyProblem3D) -> SparseMatrixCSC{Bool,Int}

Structural pattern of `∂R/∂U` of the 3D residual, built from the stencil on
the CPU (host `prob`): the Newtonian dependencies (42 entries per interior
momentum row, 22 per continuity row), widened for a state-dependent viscosity
([`FVGNFViscosity`](@ref), [`FVLawViscosity`](@ref)) to the face-strain
stencil (19 cells, 70 entries per interior momentum row), plus the dense
mean-flow row/column when `u_mean_target` is set.
"""
function fv_jacobian_pattern(prob::FVSteadyProblem3D)
    g = prob.grid
    tag = Array(prob.tag)
    n = fv_nunknowns(prob)
    I = Int[]; J = Int[]
    law = _fv3_state_viscosity(prob.visc)
    sizehint!(I, (law ? 60 : 38) * n); sizehint!(J, (law ? 60 : 38) * n)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        _fv3_push_newtonian!(I, J, g, tag, i, j, k)
        law && _fv3_push_viscosity!(I, J, g, tag, i, j, k)
    end
    _fv_flow_pattern!(I, J, prob)
    return sparse(I, J, trues(length(I)), n, n, |)
end
