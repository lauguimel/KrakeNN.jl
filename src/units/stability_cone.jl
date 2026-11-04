const STABILITY_REGISTRY = Dict{Tuple{Type,Type},Vector{Function}}()

const WALL_BC_REGISTRY = Dict{Symbol,Type{<:AbstractWallBC}}(
    :halfwayBB => HalfwayBB,
    :bouzidi_fl => BouzidiFL,
    :li_bb_v2 => LiBBV2,
    :mei_bouzidi => MeiBouzidi,
)

function _camel_to_snake_symbol(T::Type)
    name = replace(string(nameof(T)), r"BC$" => "")
    buf = IOBuffer()
    for (i, c) in enumerate(name)
        if i > 1 && isuppercase(c)
            print(buf, '_')
        end
        print(buf, lowercase(string(c)))
    end
    return Symbol(String(take!(buf)))
end

_wall_symbol(::Type{HalfwayBB}) = :halfwayBB
_wall_symbol(::Type{BouzidiFL}) = :bouzidi_fl
_wall_symbol(::Type{LiBBV2}) = :li_bb_v2
_wall_symbol(::Type{MeiBouzidi}) = :mei_bouzidi
_wall_symbol(T::Type{<:AbstractWallBC}) = _camel_to_snake_symbol(T)

function register_stability!(wall_type::Type{<:AbstractWallBC},
                             spec_type::Type{<:AbstractPhysicsSpec},
                             pred::Function)
    key = (wall_type, spec_type)
    list = get!(STABILITY_REGISTRY, key, Function[])
    pred in list || push!(list, pred)
    WALL_BC_REGISTRY[_wall_symbol(wall_type)] = wall_type
    return pred
end

function _wall_type(sym::Symbol)
    return get(WALL_BC_REGISTRY, sym, nothing)
end

function check_stability(units::LBMUnits, geom::GeometryDescriptor,
                         bc::BCConfig, spec::AbstractPhysicsSpec)
    wall_type = _wall_type(bc.wall_bc)
    wall_type === nothing &&
        return [error_issue(:wall_bc_unknown, "unknown wall_bc :$(bc.wall_bc)")]
    spec_type = Base.typename(typeof(spec)).wrapper
    funcs = get(STABILITY_REGISTRY, (wall_type, spec_type), Function[])
    issues = Issue[]
    for pred in funcs
        append!(issues, pred(units, geom))
    end
    return issues
end

function _halfway_pred(units::LBMUnits, ::GeometryDescriptor)
    issues = Issue[]
    units.tau_hydro < 0.55 &&
        push!(issues, fatal_issue(:halfway_tau_below_floor,
            "HalfwayBB requires tau_hydro >= 0.55"))
    units.tau_hydro > 1.5 &&
        push!(issues, fatal_issue(:halfway_tau_above_ceiling,
            "HalfwayBB requires tau_hydro <= 1.5"))
    units.Ma > 0.05 &&
        push!(issues, fatal_issue(:halfway_mach_above_limit,
            "HalfwayBB requires Ma <= 0.05"))
    return issues
end

function _bouzidi_ve_pred(units::LBMUnits, geom::GeometryDescriptor)
    issues = Issue[]
    units.tau_hydro < 0.6 &&
        push!(issues, fatal_issue(:bouzidi_tau_floor,
            "Bouzidi-FL requires tau_hydro >= 0.6"))
    units.tau_hydro > 1.5 &&
        push!(issues, fatal_issue(:bouzidi_tau_ceiling,
            "Bouzidi-FL requires tau_hydro <= 1.5"))
    if geom.q_wall_dist !== nothing &&
       any(q -> q < 0.05 || q > 0.95, geom.q_wall_dist)
        push!(issues, warn_issue(:q_wall_near_cliff,
            "Bouzidi-FL q_wall distribution has values near 0 or 1"))
    end
    return issues
end

# Viscoelastic parameter-stability predicate.
#
# Mirrors `_bouzidi_ve_pred` (same `(units, geom)` signature, same `Issue`
# ladder) but checks the constitutive/relaxation parameters rather than the
# wall BC. Every quantity is reconstructed from the lattice plan:
#   Wi  = lambda_LU * u_LU / R_LU
#   Re  = u_LU * R_LU / nu_total_LU
#   El  = Wi / Re = lambda_LU * nu_total_LU / R_LU^2   (elasticity number)
#   Wi_cell = lambda_LU * gamma_dot_LU with gamma_dot ~ u/L (polymer CFL proxy)
#
# Graded checks:
#   - lambda_LU vs grid/cell: the low-Wi artifact. When lambda_LU is comparable
#     to or below O(1) cell, the relaxation is sub-grid and the constitutive
#     update is numerically stiff -> the spurious +8-17% drag seen at Wi~1e-3 in
#     the 3D sweep. This is the warning that bit the sweep.
#   - El band: very high El (Wi >> Re) is the hard elastic-instability regime.
#   - polymer CFL / Wi_cell: lambda * gamma_dot >~ O(1) risks conformation
#     blow-up between updates.
#   - SPD / positivity headroom for direct-C: with no log-conformation (the 3D
#     path is Oldroyd-B direct-C only), high Wi (lambda*gamma_dot >~ 1) risks an
#     indefinite conformation tensor -> recommend log-conf.
const _LAMBDA_CELL_FLOOR = 1.0      # lambda_LU below ~1 cell: sub-grid relaxation
const _LAMBDA_CELL_WARN = 4.0       # lambda_LU below a few cells: stiffness band
const _EL_HIGH = 1.0e3              # Wi >> Re: elastic-instability regime
# Polymer-CFL / SPD thresholds on Wi_cell = lambda*gamma_dot. The validated
# RheoTool cylinder baseline sits at Wi_cell ~= 1 (Wi=1), so guardrails are set
# above 1: only Wi well past O(1) (the direct-C stiffness regime, no log-conf in
# 3D) is flagged.
const _WI_CELL_CFL = 2.0           # lambda*gamma_dot polymer-CFL guardrail
const _WI_CELL_SPD = 5.0           # lambda*gamma_dot SPD/positivity headroom

# Local shear-rate proxy gamma_dot ~ u / L over the transverse extent.
function _gamma_dot_proxy(units::LBMUnits, geom::GeometryDescriptor)
    L = _diffusion_length(geom, units.R_LU)
    L <= 0 && return 0.0
    return Float64(units.u_LU) / L
end

function _viscoelastic_param_pred(units::LBMUnits, geom::GeometryDescriptor)
    issues = Issue[]
    lambda = Float64(units.lambda_LU)
    isfinite(lambda) || return issues   # Newtonian / no polymer: nothing to check

    nu = Float64(units.nu_total_LU)
    R = Float64(units.R_LU)

    # --- lambda vs grid/cell : the low-Wi artifact -------------------------
    if lambda <= _LAMBDA_CELL_FLOOR
        push!(issues, warn_issue(:lambda_below_cell,
            "lambda_LU=$(lambda) is at or below O(1) cell: relaxation is sub-grid, " *
            "the constitutive update is stiff and drag is prone to the low-Wi artifact " *
            "(spurious over-prediction)"))
    elseif lambda < _LAMBDA_CELL_WARN
        push!(issues, warn_issue(:lambda_near_cell,
            "lambda_LU=$(lambda) spans only a few cells: low-Wi numerical-stiffness band " *
            "(refine grid or raise lambda_LU to de-risk drag)"))
    end

    # --- elasticity number El = Wi/Re = lambda*nu/R^2 ----------------------
    if R > 0
        El = lambda * nu / (R * R)
        if El >= _EL_HIGH
            push!(issues, warn_issue(:elasticity_number_high,
                "El=Wi/Re=$(El) is in the elastic-instability regime (Wi >> Re)"))
        end
    end

    # --- polymer CFL and SPD/positivity headroom (direct-C) ---------------
    gamma_dot = _gamma_dot_proxy(units, geom)
    Wi_cell = lambda * gamma_dot
    if Wi_cell > _WI_CELL_CFL
        push!(issues, warn_issue(:polymer_cfl_high,
            "lambda*gamma_dot=$(Wi_cell) exceeds the $(_WI_CELL_CFL) polymer-CFL guardrail: " *
            "conformation may blow up between updates"))
    end
    if Wi_cell >= _WI_CELL_SPD
        push!(issues, warn_issue(:direct_c_spd_headroom,
            "lambda*gamma_dot=$(Wi_cell) >= $(_WI_CELL_SPD): direct-C update risks an " *
            "indefinite (non-SPD) conformation tensor at high Wi; prefer log-conformation"))
    end
    return issues
end

register_stability!(HalfwayBB, NewtonianSpec, _halfway_pred)
register_stability!(HalfwayBB, ViscoelasticSpec, _halfway_pred)
register_stability!(HalfwayBB, ViscoelasticSpec, _viscoelastic_param_pred)
register_stability!(BouzidiFL, ViscoelasticSpec, _bouzidi_ve_pred)
register_stability!(BouzidiFL, ViscoelasticSpec, _viscoelastic_param_pred)
register_stability!(LiBBV2, ViscoelasticSpec, _viscoelastic_param_pred)
register_stability!(MeiBouzidi, ViscoelasticSpec, _viscoelastic_param_pred)
