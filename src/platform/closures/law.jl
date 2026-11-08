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
