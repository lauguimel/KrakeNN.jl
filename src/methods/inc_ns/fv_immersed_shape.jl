# A geometry-specific constructor of the existing parameter-map protocol.

function _fv_shape_same_cells(a, b, name; faces=false)
    size(a) == size(b) || throw(DimensionMismatch("shape map changed $name dimensions"))
    changed = findall(a .!= b)
    isempty(changed) && return
    cells = faces ? unique([(c[2],c[3]) for c in changed]) : [Tuple(c) for c in changed]
    _fv_ib_error("shape map changed frozen $name", cells...)
end

function _fv_shape_topology(reference, candidate)
    reference.grid == candidate.grid ||
        throw(ArgumentError("shape map must keep the fixed Cartesian grid"))
    a, b = reference.immersed, candidate.immersed
    a !== nothing && b !== nothing && a.surface !== nothing && b.surface !== nothing ||
        throw(ArgumentError("FVShapeMap requires checked analytic immersed geometry"))
    a.closure == b.closure || throw(ArgumentError("shape map must preserve the immersed closure"))
    _fv_shape_same_cells(Array(reference.tag), Array(candidate.tag), "fluid/solid/pin tags")
    _fv_shape_same_cells(_fv_ib_primal.(Array(a.qcut)) .> 0,
                         _fv_ib_primal.(Array(b.qcut)) .> 0, "cut links"; faces=true)
    _fv_shape_same_cells(Array(a.surface.wall), Array(b.surface.wall), "wall identities"; faces=true)
    if _fv_mass(a) !== nothing && _fv_mass(b) !== nothing
        _fv_shape_same_cells(Array(a.mass.face_pair), Array(b.mass.face_pair),
                             "body/region pairs"; faces=true)
    end
    _fv_check_surface(a.surface, b.surface)
    _fv_check_mass(_fv_mass(a), _fv_mass(b))
    return candidate
end

"""
    FVShapeMap(builder, base_problem, nparams) -> FVProblemMap

Construct a shape parameter map using the existing [`FVProblemMap`](@ref)
machinery. `builder(theta)` returns a problem with geometry and wall data
generic in the scalar type of `theta`, including ForwardDiff duals. It must
invoke a checked analytic constructor (`fv_immersed_circle`,
`fv_immersed_annulus`, or `fv_immersed_boundary`) on every evaluation, so the
radius, gap, axial-intersection and transversality contracts are rechecked.
Raw cut arrays do not establish that physical resolution contract.

The topology is frozen against a host copy of `base_problem`: grid, cell
classification and pressure pins, cut links, analytic wall identities, and
all M3 body/region connectivity must remain identical. Changed cells are
named in the error. For early rejection, a builder may also pass
`frozen_tag=base_problem.tag` to the analytic geometry constructor.

This is a factory, not an additional parameter-map representation: all
residual, tangent and adjoint derivatives use `FVProblemMap` unchanged.
The builder must return arrays on the backend expected by its solver.
"""
function FVShapeMap(builder, base_problem::FVSteadyProblem2D, nparams::Integer)
    nparams > 0 || throw(ArgumentError("shape map needs at least one parameter"))
    reference = fv_adapt(Array, base_problem)
    _fv_shape_topology(reference, reference)
    return FVProblemMap((prob, theta) -> _fv_shape_topology(reference, builder(theta)), Int(nparams))
end
