# Typed state inputs are lowered inside the residual, after device adaptation.
# Matrix inputs follow exactly the original indexing and arithmetic path.
struct _FVTemperatureView{T,U,G} <: AbstractMatrix{T}
    state::U
    grid::G
end
_FVTemperatureView(U, g) = _FVTemperatureView{eltype(U),typeof(U),typeof(g)}(U, g)
Base.size(v::_FVTemperatureView) = (v.grid.Nx, v.grid.Ny)
@inline Base.getindex(v::_FVTemperatureView, i::Int, j::Int) =
    _fv_val(v.state, v.grid, i, j, _fv_scalar_slot(v.grid.layout, 1))

struct _FVUniformTemperature{T} <: AbstractMatrix{T}
    value::T
end
@inline Base.getindex(v::_FVUniformTemperature, i::Int, j::Int) = v.value

# The energy unknown is stored as theta = T - T_ref (see FVEnergy). Laws must
# keep seeing absolute T, so a problem with T_ref != 0 swaps each
# FVStateTemperature selector for one carrying T_ref (see _fv_shift_viscosity).
struct _FVShiftedStateTemperature{T}
    Tref::T
end
struct _FVShiftedTemperatureView{T,U,G,R} <: AbstractMatrix{T}
    state::U
    grid::G
    Tref::R
end
_FVShiftedTemperatureView(U, g, Tref) =
    _FVShiftedTemperatureView{eltype(U),typeof(U),typeof(g),typeof(Tref)}(U, g, Tref)
Base.size(v::_FVShiftedTemperatureView) = (v.grid.Nx, v.grid.Ny)
@inline Base.getindex(v::_FVShiftedTemperatureView, i::Int, j::Int) =
    _fv_val(v.state, v.grid, i, j, _fv_scalar_slot(v.grid.layout, 1)) + v.Tref

_fv_law_input_backend(::_FVShiftedStateTemperature, theta) = true
_fv_law_input_shape(::_FVShiftedStateTemperature, shape) = true
_fv_adapt_law_input(to, f::_FVShiftedStateTemperature) = f
_fv_law_input_eltype(::Type{<:_FVShiftedStateTemperature}) = Bool
const _FVAnyStateTemperature = Union{FVStateTemperature,_FVShiftedStateTemperature}

_fv_shift_law_input(f, Tref) = f
_fv_shift_law_input(::FVStateTemperature, Tref) =
    iszero(Tref) ? FVStateTemperature() : _FVShiftedStateTemperature(Tref)
_fv_shift_law_input(f::_FVShiftedStateTemperature, Tref) = _fv_shift_law_input(FVStateTemperature(), Tref)
_fv_shift_viscosity(v, Tref) = v
_fv_shift_viscosity(v::FVLawViscosity, Tref) =
    FVLawViscosity(v.law, v.theta, map(f -> _fv_shift_law_input(f, Tref), v.fields), v.eps)

# Lowering hook: an energy-active problem built with T_ref != 0 shifts the
# state-temperature inputs of its viscosity. Defined as a more specific
# constructor so that fv_steady_problem, fv_adapt and _fv_with all pass through it.
function FVSteadyProblem2D(grid::FVGrid2D, tag, nu, bvel, src, visc, immersed, energy::_FVEnergyData, flowrate)
    v = _fv_shift_viscosity(visc, energy.Tref)
    return FVSteadyProblem2D{typeof(grid),typeof(tag),typeof(nu),typeof(bvel),typeof(src),
                             typeof(v),typeof(immersed),typeof(energy),typeof(flowrate)}(
        grid, tag, nu, bvel, src, v, immersed, energy, flowrate)
end

@inline _fv_bind_law_input(f, U, g) = f
@inline _fv_bind_law_input(::FVStateTemperature, U, g) = _FVTemperatureView(U, g)
@inline _fv_bind_law_input(f::_FVShiftedStateTemperature, U, g) = _FVShiftedTemperatureView(U, g, f.Tref)
@inline _fv_reference_law_input(f, temperature) = f
@inline _fv_reference_law_input(::_FVAnyStateTemperature, temperature) = _FVUniformTemperature(temperature)
@inline _fv_bind_viscosity(v, U, g) = v
@inline _fv_reference_viscosity(v, temperature) = v
_fv_reference_nu(v, fallback) = fallback
_fv_has_state_temperature(v) = false
_fv_has_state_temperature(v::FVLawViscosity) = any(f -> f isa _FVAnyStateTemperature, v.fields)

@inline _fv_bind_viscosity(v::FVLawViscosity, U, g) =
    FVLawViscosity(v.law, v.theta, map(f -> _fv_bind_law_input(f, U, g), v.fields), v.eps)
@inline _fv_reference_viscosity(v::FVLawViscosity, temperature) =
    FVLawViscosity(v.law, v.theta, map(f -> _fv_reference_law_input(f, temperature), v.fields), v.eps)
_fv_reference_nu(v::FVLawViscosity{<:E5ThermalLaw}, fallback) = v.law.nu_ref
_fv_law_regular_at_rest(::E5ThermalLaw, theta) = true
@inline _fv_law_invariant(law::E5ThermalLaw, theta, I1, fields) =
    (law.nu_ref / law.eta_ref) * _e5_dynamic_viscosity(2 * I1, fields[1], theta, law.shear_scale)

# This opt-in wrapper keeps the original FVGNFViscosity representation and
# isothermal operation order intact. Thermal formulas already belong to the
# rheology layer; only the regular-at-rest invariant forms are supplied here.
struct _FVThermalGNFViscosity{V,F}
    base::V
    temperature::F
end
function _fv_thermal_gnf_viscosity(v::FVGNFViscosity, temperature)
    temperature isa FVStateTemperature ||
        throw(ArgumentError("FVGNFViscosity temperature must be FVStateTemperature()"))
    return _FVThermalGNFViscosity(v, temperature)
end
_fv_has_state_temperature(v::_FVThermalGNFViscosity) = v.temperature isa _FVAnyStateTemperature
_fv_shift_viscosity(v::_FVThermalGNFViscosity, Tref) =
    _FVThermalGNFViscosity(v.base, _fv_shift_law_input(v.temperature, Tref))
function _fv_validate_viscosity(v::_FVThermalGNFViscosity, g, nu)
    _fv_nfields(g) >= 4 || throw(ArgumentError("thermal GNF viscosity requires an active energy model"))
    isbitstype(typeof(v.base.law)) || throw(ArgumentError("thermal GNF law must be isbits"))
    return nothing
end
@inline _fv_bind_viscosity(v::_FVThermalGNFViscosity, U, g) =
    _FVThermalGNFViscosity(v.base, _fv_bind_law_input(v.temperature, U, g))
@inline _fv_reference_viscosity(v::_FVThermalGNFViscosity, temperature) =
    _FVThermalGNFViscosity(v.base, _fv_reference_law_input(v.temperature, temperature))
@inline _fv_visc_eltype(v::_FVThermalGNFViscosity) =
    promote_type(_fv_visc_eltype(v.base), _fv_law_number_type(typeof(v.base.law.thermal)),
                 _fv_law_input_eltype(typeof(v.temperature)))
@inline _fv_visc_zero(v::_FVThermalGNFViscosity) = zero(_fv_visc_eltype(v))

@inline _fv_thermal_gnf_invariant(law, I1, temperature) =
    effective_viscosity_thermal(law, gnf_shear_rate(I1), temperature)
@inline _fv_thermal_gnf_invariant(law::Newtonian, I1, temperature) =
    law.nu * thermal_shift_factor(law.thermal, temperature)
@inline function _fv_thermal_gnf_invariant(law::CarreauYasuda, I1, temperature)
    I1 > zero(I1) && return effective_viscosity_thermal(law, gnf_shear_rate(I1), temperature)
    a = thermal_shift_factor(law.thermal, temperature)
    shifted = CarreauYasuda(law.eta_0 * a, law.eta_inf * a, law.lambda * a, law.a, law.n)
    return gnf_viscosity(shifted, I1)
end
@inline function _fv_thermal_gnf_nu(v::_FVThermalGNFViscosity, ux, uy, vx, vy, temperature)
    sxy = (uy + vx) / 2
    I1 = ux * ux + vy * vy + 2 * sxy * sxy
    return _fv_thermal_gnf_invariant(v.base.law, I1 + v.base.eps * v.base.eps / 2, temperature)
end
@inline function _fv_face_nu(v::_FVThermalGNFViscosity, nuP, nuN, ux, uy, vx, vy, i, j, ii, jj)
    @inbounds temperature = (v.temperature[i,j] + v.temperature[ii,jj]) / 2
    return _fv_thermal_gnf_nu(v, ux, uy, vx, vy, temperature)
end
@inline function _fv_wall_nu(v::_FVThermalGNFViscosity, nuP, ux, uy, vx, vy, i, j)
    @inbounds return _fv_thermal_gnf_nu(v, ux, uy, vx, vy, v.temperature[i,j])
end
const _FVThermalGNFProblem = FVSteadyProblem2D{<:FVGrid2D,<:Any,<:Any,<:Any,<:Any,<:_FVThermalGNFViscosity}
fv_jacobian_pattern(prob::_FVThermalGNFProblem) = _fv_gnf_jacobian_pattern(prob)
