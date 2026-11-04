# Steady-state-time estimation for a compiled lattice-unit plan.
#
# Returns the dominant relaxation time (in lattice steps) needed for a flow to
# settle to steady state, taken as the MAX of three physical timescales:
#   - viscous diffusion : t_diff = L^2 / nu_total
#   - advective flush   : t_adv  = (L_up + L_down) * R / u
#   - polymeric relax   : t_poly = n_poly * lambda      (n_poly in 5..10)
# It also reports whether a steady state is EXPECTED to exist for the configured
# geometry (steady shear / Poiseuille / channel / cylinder: yes; inherently
# transient setups such as Taylor-Green or oscillatory shear: flag).
#
# This extends the existing `_max_steps` heuristic: when the configured
# `max_steps` is below the estimate, a graded `Issue` is emitted so the user is
# warned that the result may be non-converged (the root cause of the suspicious
# low-Wi 3D-sweep numbers).

# Number of relaxation times required for the polymer stress to settle.
# 5..10 lambda is the standard rule; we use the lower bound for the estimate so
# the warning only fires when max_steps is genuinely short.
const _N_POLY_RELAX = 5

# Geometry types known to admit a steady state under steady forcing/BCs.
const _STEADY_GEOMETRY_TYPES = Set{Symbol}((
    :channel, :poiseuille, :couette, :shear, :simple_shear,
    :cylinder, :cylinder_2d, :cylinder_3d, :sphere, :sphere_3d,
    :contraction, :contraction_4to1, :backward_facing_step, :bfs,
    :cavity, :lid_driven_cavity, :square, :square_periodic, :unknown,
))

# Geometry types that are inherently transient (no steady state expected).
const _TRANSIENT_GEOMETRY_TYPES = Set{Symbol}((
    :taylor_green, :decaying_vortex, :oscillatory_shear, :pulsatile,
    :vortex_shedding, :transient,
))

struct SteadyStateEstimate{T}
    t_ss_lu::T          # estimated time to steady state, lattice units (steps)
    n_steps_ss::Int     # ceil(t_ss_lu), the step budget recommendation
    t_diff::T           # viscous diffusion time L^2 / nu_total
    t_adv::T            # advective flush time (L_up+L_down)*R / u
    t_poly::T           # polymeric relaxation time n_poly * lambda
    basis::Symbol       # which timescale dominates: :diffusive | :advective | :polymeric
    exists::Bool        # whether a steady state is expected to exist
end

# Transverse diffusion length in lattice units. For confined flows the relevant
# viscous length is the full transverse extent; with a known blockage ratio the
# domain half-width is R_LU / blockage, otherwise fall back to R_LU itself.
function _diffusion_length(geom::GeometryDescriptor, R_LU::Int)
    R = Float64(R_LU)
    if geom.blockage > 0
        return R / geom.blockage
    end
    return R
end

function _steady_state_exists(geom::GeometryDescriptor)
    geom.type in _TRANSIENT_GEOMETRY_TYPES && return false
    return true
end

"""
    estimate_steady_state(units, geom, spec) -> SteadyStateEstimate

Estimate the time (in lattice steps) for the configured flow to reach steady
state as the maximum of the viscous diffusion, advective flush and polymeric
relaxation timescales. The `exists` field flags geometries that are inherently
transient (no steady state expected).
"""
function estimate_steady_state(units::LBMUnits{T}, geom::GeometryDescriptor,
                               spec::AbstractPhysicsSpec) where {T}
    u = Float64(units.u_LU)
    nu = Float64(units.nu_total_LU)
    L = _diffusion_length(geom, units.R_LU)

    t_diff = nu > 0 ? L^2 / nu : Inf
    t_adv = u > 0 ? (geom.L_up + geom.L_down) * Float64(units.R_LU) / u : Inf

    lambda = Float64(units.lambda_LU)
    t_poly = isfinite(lambda) ? _N_POLY_RELAX * lambda : 0.0

    pairs = ((:diffusive, t_diff), (:advective, t_adv), (:polymeric, t_poly))
    basis = :diffusive
    t_ss = -Inf
    for (name, value) in pairs
        if value > t_ss
            t_ss = value
            basis = name
        end
    end

    n_steps = isfinite(t_ss) ? ceil(Int, t_ss) : typemax(Int)
    return SteadyStateEstimate{T}(T(t_ss), n_steps, T(t_diff), T(t_adv),
                                  T(t_poly), basis, _steady_state_exists(geom))
end

"""
    estimate_steady_state(plan::SimulationPlan) -> SteadyStateEstimate

Accessor on a compiled plan: estimate the steady-state time from its units,
geometry and physics spec.
"""
estimate_steady_state(plan::SimulationPlan) =
    estimate_steady_state(plan.units, plan.geometry, plan.physics_spec)

# Issues emitted from the steady-state estimate, folded into the shared
# validation ladder alongside the existing `max_steps_low` warning.
function steady_state_issues(units::LBMUnits{T}, geom::GeometryDescriptor,
                             spec::AbstractPhysicsSpec) where {T}
    issues = Issue[]
    est = estimate_steady_state(units, geom, spec)
    if !est.exists
        push!(issues, warn_issue(:steady_state_not_expected,
            "geometry :$(geom.type) is inherently transient; no steady state expected " *
            "(steady-state-time estimate is not meaningful)"))
        return issues
    end
    if units.max_steps < est.n_steps_ss
        push!(issues, warn_issue(:max_steps_below_steady_state,
            "max_steps=$(units.max_steps) below steady-state estimate ~$(est.n_steps_ss) " *
            "(dominant timescale :$(est.basis)) -> result may be non-converged"))
    end
    return issues
end
