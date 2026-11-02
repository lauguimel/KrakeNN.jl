"""
    _gnf3d_rheology(T, model)

Convert a supported isothermal GNF model's scalar parameters to population
element type `T` before kernel launch. Preserve the model's viscosity limits.
Thermal coupling and unsupported constitutive models raise `ArgumentError`.
"""
function _gnf3d_rheology(::Type{T}, model::Union{Newtonian,PowerLaw,CarreauYasuda,Cross}) where T
    model.thermal isa IsothermalCoupling ||
        throw(ArgumentError("3D GNF collision requires IsothermalCoupling"))
    return _gnf3d_isothermal_model(T, model)
end

function _gnf3d_rheology(::Type, model)
    throw(ArgumentError("3D GNF collision supports Newtonian, PowerLaw, CarreauYasuda, and Cross; got $(typeof(model))"))
end

_gnf3d_isothermal_model(::Type{T}, m::Newtonian) where T =
    Newtonian(T(m.nu), IsothermalCoupling())

_gnf3d_isothermal_model(::Type{T}, m::PowerLaw) where T =
    PowerLaw(T(m.K), T(m.n), T(m.nu_min), T(m.nu_max), IsothermalCoupling())

_gnf3d_isothermal_model(::Type{T}, m::CarreauYasuda) where T =
    CarreauYasuda(T(m.eta_0), T(m.eta_inf), T(m.lambda), T(m.a), T(m.n),
                 IsothermalCoupling())

_gnf3d_isothermal_model(::Type{T}, m::Cross) where T =
    Cross(T(m.eta_0), T(m.eta_inf), T(m.K), T(m.m), IsothermalCoupling())
