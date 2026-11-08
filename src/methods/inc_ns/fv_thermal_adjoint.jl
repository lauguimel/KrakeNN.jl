# Thermal theta derivatives use the same full state transpose as isothermal
# problems. Capability depends on the parameter adapter, never a law whitelist.
_fv_require_adjoint_map(prob::_FVLawProblem, ::FVLawParameters) = nothing

function _fv_temperature_indices(prob::FVSteadyProblem2D)
    prob.energy === nothing && throw(ArgumentError("temperature objectives require an active energy model"))
    g = prob.grid
    cells = findall(!=(FV_CELL_SOLID), vec(Array(prob.tag)))
    isempty(cells) && throw(ArgumentError("temperature objectives require fluid cells"))
    return [_fv_cell_index(g, c, _fv_scalar_slot(g.layout, 1)) for c in cells], cells
end

"""
    fv_mean_temperature(prob; T_wall, target=0) -> FVStateObs

Squared mean fluid-temperature-rise misfit
`0.5 * (mean_fluid(T - T_wall) - target)^2`. Voxel solid cells are excluded.
`T_wall` and `target` are fixed finite scalars; they do not change with law
parameters. The state gradient acts on temperature only, with no explicit
parameter term.
"""
function fv_mean_temperature(prob::FVSteadyProblem2D; T_wall::Real, target::Real=0)
    isfinite(T_wall) && isfinite(target) ||
        throw(ArgumentError("T_wall and target must be finite"))
    index, _ = _fv_temperature_indices(prob)
    return FVStateObs(reshape(index, :, 1), fill(inv(Float64(length(index))), length(index), 1),
                       [Float64(T_wall + target)], 1.0)
end

struct _FVTemperatureL2 <: FVObjective
    index::Vector{Int}
    target::Vector{Float64}
    weight::Vector{Float64}
end

function _fv_temperature_data(value, prob, cells, name)
    if value isa Real
        data = fill(Float64(value), length(cells))
    elseif value isa AbstractMatrix
        size(value) == size(prob.tag) || throw(DimensionMismatch("$name must match the cell grid"))
        all(isfinite, Array(value)) || throw(ArgumentError("$name must be finite"))
        data = Float64.(vec(Array(value))[cells])
    else
        throw(ArgumentError("$name must be a scalar or cell matrix"))
    end
    all(isfinite, data) || throw(ArgumentError("$name must be finite"))
    return data
end

"""
    fv_temperature_l2(prob, target; weights=1) -> FVObjective

Weighted temperature misfit `0.5 * sum_fluid(weights * (T-target)^2)`, without
cell-area or fluid-count normalization. `target` and `weights` accept scalars
or matrices matching the cell grid. Weights must be finite and nonnegative;
targets must be finite. Both are copied at construction and remain fixed
during differentiation. Solid cells contribute neither value nor gradient.
"""
function fv_temperature_l2(prob::FVSteadyProblem2D, target; weights=1)
    index, cells = _fv_temperature_indices(prob)
    target_values = _fv_temperature_data(target, prob, cells, "target")
    weight_values = _fv_temperature_data(weights, prob, cells, "weights")
    all(>=(0), weights isa AbstractMatrix ? Array(weights) : (weights,)) ||
        throw(ArgumentError("temperature weights must be nonnegative"))
    return _FVTemperatureL2(index, target_values, weight_values)
end

function fv_objective(obj::_FVTemperatureL2, U, prob)
    value = zero(eltype(U))
    for k in eachindex(obj.index)
        value += obj.weight[k] * abs2(U[obj.index[k]] - obj.target[k]) / 2
    end
    return value
end
fv_objective_support(obj::_FVTemperatureL2, prob) = copy(obj.index)

function _fv_objective_grad_u(obj::_FVTemperatureL2, U::Vector{Float64}, prob; chunk::Int=8)
    gradient = zeros(Float64, length(U))
    for k in eachindex(obj.index)
        gradient[obj.index[k]] = obj.weight[k] * (U[obj.index[k]] - obj.target[k])
    end
    return fv_objective(obj, U, prob), gradient
end

"""
    fv_flow_force_objective(prob) -> FVLinearFunctional

The solved global x body force `Fx` of a problem with `u_mean_target`.
Its state gradient selects the final global unknown; its explicit parameter
derivative is zero. Fixed-force problems have no such unknown and are rejected.
"""
function fv_flow_force_objective(prob::FVSteadyProblem2D)
    prob.flowrate === nothing && throw(ArgumentError("a force objective requires u_mean_target"))
    return FVLinearFunctional([_fv_force_index(prob)], [1.0])
end
