"""
    Units

Submodule containing Kraken lattice-unit planning, stability audits, and driver keyword conversion helpers.
Use it through `Kraken.Units` when converting physical nondimensional inputs into lattice-unit simulation plans.

```julia
using Kraken

plan = Kraken.Units.compile(; physics=:newtonian, geometry=(; type=:channel), bc=(;))
```
"""
module Units

import KernelAbstractions

include("audit_trail.jl")

abstract type AbstractPhysicsSpec end

struct NewtonianSpec{T} <: AbstractPhysicsSpec
    Re::T
end

struct ViscoelasticSpec{T} <: AbstractPhysicsSpec
    Re::T
    Wi::T
    beta::T
    bsd_fraction::T
    model::Symbol
    L_max::T
end

struct ThermalBoussinesqSpec{T} <: AbstractPhysicsSpec
    Re::T
    Pr::T
    Ra::T
end

struct EHDSpec{T} <: AbstractPhysicsSpec
    T_ehd::T
    C::T
    M::T
    alpha::T
    Ma_E::T
end

struct PowerLawSpec{T} <: AbstractPhysicsSpec end
struct MultiphaseSpec{T} <: AbstractPhysicsSpec end
struct MHDSpec{T} <: AbstractPhysicsSpec end

struct LBMUnits{T}
    tau_hydro::T
    nu_total_LU::T
    u_LU::T
    R_LU::Int
    Ma::T
    scaling::Symbol
    max_steps::Int
    nu_s_LU::T
    nu_p_LU::T
    lambda_LU::T
    alpha_LU::T
    beta_thermal_LU::T
    dx_real::T
    dt_real::T
    rho_real::T
end

struct GeometryDescriptor
    type::Symbol
    blockage::Float64
    L_up::Float64
    L_down::Float64
    q_wall_dist::Union{Nothing,Vector{Float64}}
    kappa_max::Float64
    stl_hash::UInt64
end

struct BCConfig
    inlet::Symbol
    outlet::Symbol
    north_wall::Symbol
    south_wall::Symbol
    wall_bc::Symbol
end

struct DiscretizationConfig
    advection_scheme::Symbol
    embedded_geometry::Symbol
    embedded_gradient::Bool
end

struct SimulationPlan{T}
    physics_spec::AbstractPhysicsSpec
    units::LBMUnits{T}
    bc::BCConfig
    geometry::GeometryDescriptor
    discretization::DiscretizationConfig
    refinement::Union{Nothing,Vector{LBMUnits{T}}}
    warnings::Vector{Issue}
    notes::Vector{String}
    audit_source::Symbol
end

abstract type AbstractWallBC end
struct HalfwayBB <: AbstractWallBC end
struct BouzidiFL <: AbstractWallBC end
struct LiBBV2 <: AbstractWallBC end
struct MeiBouzidi <: AbstractWallBC end

_has(x::NamedTuple, key::Symbol) = key in keys(x)
_has(x::AbstractDict, key::Symbol) = haskey(x, key)
_has(x, key::Symbol) = hasproperty(x, key)

_get(x::NamedTuple, key::Symbol, default) = key in keys(x) ? getfield(x, key) : default
_get(x::AbstractDict, key::Symbol, default) = get(x, key, default)
_get(x, key::Symbol, default) = hasproperty(x, key) ? getproperty(x, key) : default

_sym(x::Symbol) = x
_sym(x::AbstractString) = Symbol(x)
_sym(x) = Symbol(x)

_bool(x::Bool) = x
_bool(x::Integer) = x != 0
_bool(x::AbstractFloat) = x != 0
_bool(x::Symbol) = x in (:true, :yes, :on, :enabled, Symbol("1"))
_bool(x::AbstractString) = lowercase(strip(x)) in ("true", "yes", "on", "1")

function _as_float_vector(x)
    x === nothing && return nothing
    return Float64[Float64(v) for v in x]
end

function _normalize_geometry(geometry; kwargs...)
    kw = NamedTuple(kwargs)
    L_up = Float64(_get(geometry, :L_up, _get(kw, :L_up, 15.0)))
    L_down = Float64(_get(geometry, :L_down, _get(kw, :L_down, 15.0)))
    return GeometryDescriptor(
        _sym(_get(geometry, :type, :unknown)),
        Float64(_get(geometry, :blockage, 0.0)),
        L_up,
        L_down,
        _as_float_vector(_get(geometry, :q_wall_dist, nothing)),
        Float64(_get(geometry, :kappa_max, 0.0)),
        UInt64(_get(geometry, :stl_hash, 0)),
    )
end

function _normalize_bc(bc)
    return BCConfig(
        _sym(_get(bc, :inlet, :velocity_parabolic)),
        _sym(_get(bc, :outlet, :zou_he_pressure)),
        _sym(_get(bc, :north_wall, :halfwayBB)),
        _sym(_get(bc, :south_wall, :halfwayBB)),
        _sym(_get(bc, :wall_bc, :halfwayBB)),
    )
end

function _normalize_discretization(kw)
    return DiscretizationConfig(
        _sym(_get(kw, :advection_scheme, :muscl_superbee)),
        _sym(_get(kw, :embedded_geometry, :qwall)),
        _bool(_get(kw, :embedded_gradient, false)),
    )
end

function _assemble_plan(spec::AbstractPhysicsSpec, units::LBMUnits{T},
                        bc::BCConfig, geom::GeometryDescriptor,
                        disc::DiscretizationConfig, refinement,
                        issues::Vector{Issue}, notes::Vector{String},
                        source::Symbol, strict::Bool) where {T}
    sorted = sort_issues(issues)
    stored = [i for i in sorted if i.severity !== :info]
    plan_notes = copy(notes)
    append!(plan_notes, [i.message for i in sorted if i.severity === :info])
    emit_warning_logs(stored)
    plan = SimulationPlan{T}(spec, units, bc, geom, disc, refinement, stored,
                             plan_notes, source)
    blockers = blocking_issues(stored)
    strict && !isempty(blockers) && throw(PlanValidationError(plan, blockers))
    return plan
end

function _with_added_issues(plan::SimulationPlan{T}, extra::Vector{Issue};
                            strict::Bool=false) where {T}
    return _assemble_plan(plan.physics_spec, plan.units, plan.bc, plan.geometry,
                          plan.discretization, plan.refinement,
                          vcat(plan.warnings, extra), copy(plan.notes),
                          plan.audit_source, strict)
end

function _shared_validation_issues(units::LBMUnits, spec::AbstractPhysicsSpec,
                                   geom::GeometryDescriptor, bc::BCConfig,
                                   disc::DiscretizationConfig, ::Type{T}) where {T}
    issues = Issue[]
    append!(issues, intrinsic_unit_issues(units, T))
    append!(issues, steady_state_issues(units, geom, spec))
    append!(issues, check_stability(units, geom, bc, spec))
    append!(issues, audit_stl(geom, units).issues)
    append!(issues, check_bc_consistency(bc))
    if spec isa ViscoelasticSpec && disc.embedded_gradient
        push!(issues, warn_issue(:m48_toggle,
            "embedded_gradient=true with production viscoelastic polymer settings matches the M48 toggle hazard"))
    end
    return issues
end

function compile(; physics::Symbol, geometry, bc, refinement=nothing,
                 backend=KernelAbstractions.CPU(), T=Float64,
                 strict::Bool=true, kwargs...)
    haskey(PHYSICS_REGISTRY, physics) ||
        throw(ArgumentError("unknown physics :$physics. Registered: $(sort(collect(keys(PHYSICS_REGISTRY))))"))
    kw = (; kwargs..., T=T, backend=backend)
    preissues = unknown_kw_issues(physics, keys(kwargs))
    spec = _build_spec(PHYSICS_REGISTRY[physics], kw)
    geom = _normalize_geometry(geometry; kwargs...)
    bccfg = _normalize_bc(bc)
    disc = _normalize_discretization(kw)
    return _compile_with_spec(spec, kw, geom, bccfg, disc, refinement,
                              strict, preissues)
end

function audit(driver_kw::NamedTuple; physics::Symbol, geometry, bc,
               backend=KernelAbstractions.CPU(), T=Float64,
               strict::Bool=false, kwargs...)
    haskey(PHYSICS_REGISTRY, physics) ||
        throw(ArgumentError("unknown physics :$physics. Registered: $(sort(collect(keys(PHYSICS_REGISTRY))))"))
    kw = merge(driver_kw, NamedTuple(kwargs), (; T=T, backend=backend))
    preissues = unknown_kw_issues(physics, keys(kwargs); audit_mode=true)
    geom = _normalize_geometry(geometry; kw...)
    bccfg = _normalize_bc(bc)
    disc = _normalize_discretization(kw)
    spec, units = _audit_with_spec_type(PHYSICS_REGISTRY[physics], kw, geom)
    issues = vcat(preissues, _shared_validation_issues(units, spec, geom,
                                                       bccfg, disc, T))
    return _assemble_plan(spec, units, bccfg, geom, disc, nothing, issues,
                          String[], :audit, strict)
end

const _DRIVER_KW_NAMES = (
    :Re, :Wi, :beta, :bsd_fraction, :model, :L_max, :R_LU, :radius,
    :scaling, :tau_hydro, :tau, :nu_total_LU, :nu_total, :nu_s_LU,
    :nu_p_LU, :nu_s, :nu_p, :lambda_LU, :lambda, :u_LU, :u_mean,
    :Ma, :max_steps, :polymer_model, :advection_scheme, :embedded_geometry,
    :embedded_gradient, :wall_bc,
)

const DriverKwargs{T} = NamedTuple{_DRIVER_KW_NAMES,Tuple{
    T,T,T,T,Symbol,T,Int,Int,Symbol,T,T,T,T,T,T,T,T,T,T,T,T,T,Int,
    Symbol,Symbol,Symbol,Bool,Symbol,
}}

function driver_kwargs(plan::SimulationPlan{T})::DriverKwargs{T} where {T}
    spec = plan.physics_spec
    if spec isa ViscoelasticSpec{T}
        Re = spec.Re
        Wi = spec.Wi
        beta = spec.beta
        bsd_fraction = spec.bsd_fraction
        model = spec.model
        L_max = spec.L_max
        polymer_model = _driver_polymer_model(spec.model)
    elseif spec isa NewtonianSpec{T}
        Re = spec.Re
        Wi = T(NaN)
        beta = T(NaN)
        bsd_fraction = T(NaN)
        model = :newtonian
        L_max = T(NaN)
        polymer_model = :none
    else
        throw(phase_stub_error(:driver_kwargs))
    end
    u = plan.units
    return DriverKwargs{T}((
        Re, Wi, beta, bsd_fraction, model, L_max, u.R_LU, u.R_LU,
        u.scaling, u.tau_hydro, u.tau_hydro, u.nu_total_LU,
        u.nu_total_LU, u.nu_s_LU, u.nu_p_LU, u.nu_s_LU, u.nu_p_LU,
        u.lambda_LU, u.lambda_LU, u.u_LU, u.u_LU, u.Ma, u.max_steps,
        polymer_model, plan.discretization.advection_scheme,
        plan.discretization.embedded_geometry,
        plan.discretization.embedded_gradient, plan.bc.wall_bc,
    ))
end

include("physics_registry.jl")
include("lattice_units.jl")
include("steady_state.jl")
include("stability_cone.jl")
include("stl_audit.jl")
include("bc_consistency.jl")
include("report.jl")
include("krk_binding.jl")
include("physics/newtonian.jl")
include("physics/viscoelastic.jl")
include("physics/thermal.jl")
include("physics/non_newt.jl")
include("physics/multiphase.jl")
include("physics/electromagn.jl")

end # module Units
