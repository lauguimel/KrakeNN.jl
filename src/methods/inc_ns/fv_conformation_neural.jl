# Learned relaxation law of the steady FV conformation equation (plan
# "discrete viscoelastic adjoints", step F4). Its parameters live in an array
# (host Vector or device array, like FVLawViscosity.theta) instead of the isbits
# tuple of the closed-form laws: `relaxation_theta_array(law) = true` selects
# that storage in FVConformation3D, the lowering, fv_adapt and the parameter map
# FVRelaxationParameters (fv_conformation_adjoint_3d.jl).

"""
    relaxation_theta_array(law::AbstractRelaxationLaw) -> Bool

`true` when the law reads its parameters `theta` from an array (host or
device, indexed in the kernel) instead of an isbits tuple. Default `false`;
[`NeuralRelaxation`](@ref) returns `true`.
"""
relaxation_theta_array(::AbstractRelaxationLaw) = false

"""
    NeuralRelaxation{H}(; alpha0 = 0.0, scale = 1.0) <: AbstractRelaxationLaw

Learned relaxation law: the Giesekus coefficients `(0, 1, alpha0)` plus a
one-hidden-layer network `M` (`H` tanh units, 3 outputs, no output bias) of the
shifted invariants `x = (I1 - 3, I2 - 3, I3 - 1) / scale`,

    (b0, b1, b2) = (0, 1, alpha0) + M(x) - M(0),
    M(x) = V tanh.(W x + b),   W: H × 3, b: H, V: 3 × H.

`M(x) - M(0)` vanishes at rest (`x = 0`), so `b0 = 0` there for any parameters,
and output biases would cancel in it: they are not parameters. The parameter
vector (length `7H`, see [`relaxation_nparams`](@ref)) is

    theta = [W[:, 1]; W[:, 2]; W[:, 3]; b; V[1, :]; V[2, :]; V[3, :]].

Positivity of `c` and dissipation are not guaranteed by this form: they are
checked on each state ([`fv_check_conformation_domain`](@ref)). Start from
[`relaxation_init`](@ref).
"""
struct NeuralRelaxation{H} <: AbstractRelaxationLaw
    alpha0::Float64
    scale::Float64
    function NeuralRelaxation{H}(; alpha0::Real = 0.0, scale::Real = 1.0) where {H}
        H isa Int && H >= 1 || throw(ArgumentError("NeuralRelaxation needs H >= 1 hidden units"))
        isfinite(alpha0) || throw(ArgumentError("alpha0 must be finite"))
        isfinite(scale) && scale > 0 || throw(ArgumentError("scale must be finite and positive"))
        return new{H}(Float64(alpha0), Float64(scale))
    end
end

relaxation_nparams(::NeuralRelaxation{H}) where {H} = 7H
relaxation_theta_array(::NeuralRelaxation) = true

@inline function relaxation_coefficients(law::NeuralRelaxation{H}, theta, I1, I2, I3) where {H}
    s = law.scale
    x1 = (I1 - 3) / s
    x2 = (I2 - 3) / s
    x3 = (I3 - 1) / s
    z = zero(x1) + zero(eltype(theta))
    o0 = z; o1 = z; o2 = z
    @inbounds for k in 1:H
        b = theta[3H + k]
        a = theta[k] * x1 + theta[H + k] * x2 + theta[2H + k] * x3 + b
        hk = tanh(a) - tanh(b)
        o0 += theta[4H + k] * hk
        o1 += theta[5H + k] * hk
        o2 += theta[6H + k] * hk
    end
    return (o0, 1 + o1, law.alpha0 + o2)
end

# index ranges of the parameter blocks
_fv_neural_hidden(::NeuralRelaxation{H}) where {H} = 1:(4H)
_fv_neural_output(::NeuralRelaxation{H}) where {H} = (4H + 1):(7H)

"""
    relaxation_init(law::NeuralRelaxation; rng = nothing, hidden_std = 1.0,
                    output_std = 0.0) -> Vector{Float64}

Parameters of a learned relaxation law: hidden weights and biases drawn
`N(0, hidden_std²)` (nonzero; `rng` as in [`law_init`](@ref)), then output weights `N(0, output_std²)`, drawn
after the hidden block so that the same seed gives the same hidden layer for any
`output_std`. With `output_std = 0` (default) the law starts as Giesekus
`(0, 1, alpha0)` exactly, the output derivatives are nonzero and the hidden
ones vanish (zero output weights); a second point with `output_std = 0.05`
makes the hidden derivatives visible.
"""
function relaxation_init(law::NeuralRelaxation; rng = nothing, hidden_std::Real = 1.0,
                         output_std::Real = 0.0)
    r() = rng === nothing ? randn() : randn(rng)
    θ = zeros(relaxation_nparams(law))
    for q in _fv_neural_hidden(law)
        θ[q] = hidden_std * r()
    end
    for q in _fv_neural_output(law)
        θ[q] = output_std * r()
    end
    return θ
end
