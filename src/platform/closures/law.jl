"""
    AbstractScalarLaw <: AbstractClosure

A local scalar constitutive law with explicit parameters. Implement
[`law_ninputs`](@ref), [`law_nparams`](@ref), and [`law_value`](@ref).
The input is a fixed-length tuple; parameters are an external flat vector.
Neither contains solver state. Scalar evaluation specializes on concrete law
types, is inline-able, preserves numeric types (including ForwardDiff dual
numbers), and avoids heap allocation. External `AbstractVector` parameters
leave storage ownership with callers, including future device storage; this
interface alone does not establish GPU validation.
"""
abstract type AbstractScalarLaw <: AbstractClosure end

"""Return the compile-time number of scalar inputs of a concrete law."""
function law_ninputs end

"""Return the compile-time length of the flat parameter vector of a law."""
function law_nparams end

"""
    law_value(law, x::Tuple, theta::AbstractVector) -> scalar

Evaluate a local law. Callers supply the lengths declared by `law_ninputs`
and `law_nparams`; the scalar hot path does not validate dimensions or domains.
Viscosity laws require the first input, the shear-rate magnitude, to be > 0.
"""
function law_value end

"""
    law_pullback!(gtheta, law, x::Tuple, theta::AbstractVector, s) -> Tuple

Accumulate `s * ∂law_value/∂theta` into the caller's `gtheta`, without clearing
it, and return `s * ∂law_value/∂x` as a tuple. The default implementation uses
ForwardDiff with static arrays and is intended for small laws. Neural laws
provide a handwritten reverse pass. No parameter/state derivative is detached.
"""
@inline function law_pullback!(gtheta, law::AbstractScalarLaw, x::Tuple,
                              theta::AbstractVector, s)
    p = law_nparams(law)
    ts = SVector(ntuple(i -> theta[i], Val(p)))
    gt = ForwardDiff.gradient(t -> law_value(law, x, t), ts)
    gx = ForwardDiff.gradient(u -> law_value(law, Tuple(u), theta), SVector(x))
    @inbounds for i in 1:p
        gtheta[i] += s * gt[i]
    end
    return ntuple(i -> s * gx[i], Val(law_ninputs(law)))
end

"""
    law_init(law; rng=nothing) -> Vector

Allocate initial parameters on the host. Classical laws return their model
parameters; neural laws use small random weights. An optional `rng` is passed
to `randn(rng)`; `nothing` uses Julia's default generator. Initialization is
not part of the allocation-free scalar evaluation contract.
"""
function law_init end

"""
    ClassicalLaw(model::Union{PowerLaw,CarreauYasuda})

Adapt an existing rheology model to a one-input scalar law, `x = (γ̇,)`.
The flat parameter order is `[K, n, nu_min, nu_max]` for `PowerLaw`, or
`[eta_0, eta_inf, lambda, a, n]` for `CarreauYasuda`. Values delegate to the
existing viscosity implementation, including its power-law clamps. Thermal
metadata stays fixed in `model`; as in `gnf_viscosity`, it is not evaluated.
At nondifferentiable clamp corners, ForwardDiff follows the existing branch
convention; no new smooth approximation is introduced.
"""
struct ClassicalLaw{M<:Union{PowerLaw,CarreauYasuda}} <: AbstractScalarLaw
    model::M
end

@inline law_ninputs(::ClassicalLaw) = 1
@inline law_nparams(::ClassicalLaw{<:PowerLaw}) = 4
@inline law_nparams(::ClassicalLaw{<:CarreauYasuda}) = 5

law_init(law::ClassicalLaw{<:PowerLaw}; rng=nothing) =
    [law.model.K, law.model.n, law.model.nu_min, law.model.nu_max]
law_init(law::ClassicalLaw{<:CarreauYasuda}; rng=nothing) =
    [law.model.eta_0, law.model.eta_inf, law.model.lambda, law.model.a, law.model.n]

@inline function law_value(law::ClassicalLaw{<:PowerLaw}, x::Tuple,
                           theta::AbstractVector)
    @inbounds model = PowerLaw(theta[1], theta[2]; nu_min=theta[3],
                              nu_max=theta[4], thermal=law.model.thermal)
    return effective_viscosity(model, x[1])
end

@inline function law_value(law::ClassicalLaw{<:CarreauYasuda}, x::Tuple,
                           theta::AbstractVector)
    @inbounds model = CarreauYasuda(theta[1], theta[2], theta[3], theta[4],
                                   theta[5]; thermal=law.model.thermal)
    return effective_viscosity(model, x[1])
end

"""
    E5ThermalLaw(theta_truth; T_wall, nu_ref, shear_scale=1)

Eight-parameter E5 viscosity, with inputs `(gamma, temperature)` in solver
shear units and kelvin. Parameters are
`[eta0, lambda, beta, n0, slope, amplitude, Tc, width]` in physical units:
`a = exp(-beta*(T-483.15))`, `n = clamp(n0+slope*(T-463.15),0.2,0.75)`,
`C = 1+amplitude/(1+exp((T-Tc)/width))`, and
`eta = eta0*a*C*(1+(lambda*a*shear_scale*gamma)^2)^((n-1)/2)`.
The returned kinematic viscosity is `nu_ref*eta/eta_ref`.

`eta_ref = eta(0,T_wall;theta_truth)` and `nu_ref` are frozen at construction;
neither follows parameter changes or parameter differentiation. `shear_scale`
converts solver shear rate to inverse seconds. The index clamps retain their
existing piecewise derivative convention. `law_init` returns the reference
parameters; callers own every subsequent parameter vector.
"""
struct E5ThermalLaw{T} <: AbstractScalarLaw
    eta_ref::T
    nu_ref::T
    shear_scale::T
    T_wall::T
    theta_reference::NTuple{8,T}
end

@inline function _e5_dynamic_viscosity(gamma_squared, temperature, theta, shear_scale)
    @inbounds eta0, lambda, beta, n0, slope, amplitude, Tc, width =
        theta[1], theta[2], theta[3], theta[4], theta[5], theta[6], theta[7], theta[8]
    C = typeof(shear_scale)
    a = exp(-beta * (temperature - C(483.15)))
    n = clamp(n0 + slope * (temperature - C(463.15)), C(0.2), C(0.75))
    rise = 1 + amplitude / (1 + exp((temperature - Tc) / width))
    return eta0 * a * rise * (1 + (lambda * a * shear_scale)^2 * gamma_squared)^((n - 1) / 2)
end

function E5ThermalLaw(theta_truth; T_wall::Real, nu_ref::Real, shear_scale::Real=one(nu_ref))
    length(theta_truth) == 8 || throw(DimensionMismatch("E5 requires eight reference parameters"))
    values = promote(float(T_wall), float(nu_ref), float(shear_scale), map(float, Tuple(theta_truth))...)
    Tw, nr, ss = values[1], values[2], values[3]
    theta = ntuple(i -> values[i+3], Val(8))
    all(isfinite, values) && Tw > 0 && nr > 0 && ss > 0 && theta[1] > 0 &&
        theta[2] >= 0 && theta[8] > 0 ||
        throw(ArgumentError("E5 reference data must be finite with positive T_wall, nu_ref, shear_scale, eta0 and width, and nonnegative lambda"))
    eta_ref = _e5_dynamic_viscosity(zero(Tw), Tw, theta, ss)
    isfinite(eta_ref) && eta_ref > 0 || throw(ArgumentError("E5 reference viscosity must be positive and finite"))
    return E5ThermalLaw(eta_ref, nr, ss, Tw, theta)
end
@inline law_ninputs(::E5ThermalLaw) = 2
@inline law_nparams(::E5ThermalLaw) = 8
law_init(law::E5ThermalLaw; rng=nothing) = collect(law.theta_reference)
@inline law_value(law::E5ThermalLaw, x::Tuple, theta::AbstractVector) =
    (law.nu_ref / law.eta_ref) * _e5_dynamic_viscosity(x[1] * x[1], x[2], theta, law.shear_scale)
