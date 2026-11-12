# Second normal stress difference in the 3D steady FV residual.
#
# Memoryless (Reiner--Rivlin) stress
#     tau = 2 eta(gdot) D + 4 Psi2(gdot) D.D,
# with D the rate-of-strain tensor (sym(grad u)) and gdot = sqrt(2 D:D). In
# simple shear u = gdot y e_x it gives N1 = tau_xx - tau_yy = 0 and
# N2 = tau_yy - tau_zz = Psi2 gdot^2, the term that drives the secondary flow
# of a viscoelastic fluid in a non-circular duct.
#
# The change is local to the traction: the residual calls the identity hooks
# _fv3_face_stress / _fv3_wall_stress (fv_steady_residual_3d.jl) after the
# viscous traction; the methods below add 4 Psi2 sx (D.D) e_n on the same face
# gradient that sets eta. The Rhie--Chow coefficient keeps eta only. Every
# derivative (state Jacobian, parameter duals) comes from ForwardDiff through
# the same kernel.

"""
    FVReinerRivlinViscosity(eta, psi2)

Steady 3D FV stress `tau = 2 eta(gdot) D + 4 psi2(gdot) D.D`. `eta` is an
[`FVGNFViscosity`](@ref) or an [`FVLawViscosity`](@ref) (kinematic viscosity,
as in the GNF path); `psi2` is an [`FVLawViscosity`](@ref) whose law value is
the (kinematic) second normal stress coefficient, `N2 = psi2 gdot^2` in simple
shear (negative for polymer solutions). Both laws read the same face strain
invariant as the GNF path. `psi2 = 0` reduces to the `eta` model exactly up to
the added zero. Isothermal only (no energy model, no state-temperature input).
Pass it as `fv_steady_problem(...; viscosity = model)`; parameters are mapped by
[`FVReinerRivlinParameters`](@ref).
"""
struct FVReinerRivlinViscosity{E,P}
    eta::E
    psi2::P
end

# viscosity hooks: eta sets the viscous traction and the Rhie--Chow coefficient
@inline _fv_face_nu3(v::FVReinerRivlinViscosity, nuP, nuN, G, i, j, k, ii, jj, kk) =
    _fv_face_nu3(v.eta, nuP, nuN, G, i, j, k, ii, jj, kk)
@inline _fv_wall_nu3(v::FVReinerRivlinViscosity, nuP, G, i, j, k) = _fv_wall_nu3(v.eta, nuP, G, i, j, k)

# t + 4 psi sx (D.D)[:, n], with D = sym(G) and G the face velocity gradient
@inline function _fv3_add_dd(t, psi, G, n, sx)
    D11 = G[1]; D22 = G[5]; D33 = G[9]
    D12 = (G[2] + G[4]) / 2
    D13 = (G[3] + G[7]) / 2
    D23 = (G[6] + G[8]) / 2
    c1 = _fv3_pick3(n, D11, D12, D13)
    c2 = _fv3_pick3(n, D12, D22, D23)
    c3 = _fv3_pick3(n, D13, D23, D33)
    q1 = D11 * c1 + D12 * c2 + D13 * c3
    q2 = D12 * c1 + D22 * c2 + D23 * c3
    q3 = D13 * c1 + D23 * c2 + D33 * c3
    f = 4 * psi * sx
    return (t[1] + f * q1, t[2] + f * q2, t[3] + f * q3)
end
@inline _fv3_face_stress(v::FVReinerRivlinViscosity, t, G, n, sx, i, j, k, ii, jj, kk) =
    _fv3_add_dd(t, _fv_face_nu3(v.psi2, zero(eltype(G)), zero(eltype(G)), G, i, j, k, ii, jj, kk),
                G, n, sx)
@inline _fv3_wall_stress(v::FVReinerRivlinViscosity, t, G, n, sx, i, j, k) =
    _fv3_add_dd(t, _fv_wall_nu3(v.psi2, zero(eltype(G)), G, i, j, k), G, n, sx)

@inline _fv_visc_eltype(v::FVReinerRivlinViscosity) =
    promote_type(_fv_visc_eltype(v.eta), _fv_visc_eltype(v.psi2))
@inline _fv_visc_zero(v::FVReinerRivlinViscosity) = zero(_fv_visc_eltype(v))
_fv_adapt_viscosity(to, v::FVReinerRivlinViscosity) =
    FVReinerRivlinViscosity(_fv_adapt_viscosity(to, v.eta), _fv_adapt_viscosity(to, v.psi2))
KernelAbstractions.Adapt.adapt_structure(to, v::FVReinerRivlinViscosity) =
    FVReinerRivlinViscosity(KernelAbstractions.Adapt.adapt(to, v.eta),
                            KernelAbstractions.Adapt.adapt(to, v.psi2))
_fv3_state_viscosity(::FVReinerRivlinViscosity) = true

function _fv3_validate_viscosity(v::FVReinerRivlinViscosity, g, nu)
    v.eta isa Union{FVGNFViscosity,FVLawViscosity} ||
        throw(ArgumentError("Reiner-Rivlin eta must be an FVGNFViscosity or an FVLawViscosity"))
    v.psi2 isa FVLawViscosity ||
        throw(ArgumentError("Reiner-Rivlin psi2 must be an FVLawViscosity"))
    _fv_nfields(g) == 4 ||
        throw(ArgumentError("the Reiner-Rivlin stress is isothermal (no energy model)"))
    for m in (v.eta, v.psi2)
        m isa FVLawViscosity && any(f -> f isa _FVAnyStateTemperature, m.fields) &&
            throw(ArgumentError("the Reiner-Rivlin stress does not read the state temperature"))
        _fv3_validate_viscosity(m, g, nu)
    end
    return nothing
end
_fv_validate_viscosity(v::FVReinerRivlinViscosity, g::FVGrid3D, nu) = _fv3_validate_viscosity(v, g, nu)

# ------------------------------------------------------------------ Giesekus viscometric laws
# Closed-form steady-shear functions of the Giesekus model, written without the
# cancellation of the textbook form at small Wi = lambda gdot:
#   x = 16 a (1 - a) Wi^2, s = sqrt(1 + x) + 1, chi = sqrt(2 / s),
#   g = f / Wi^2 = 16 a (1 - a) / (s^2 (1 + chi) (1 + (1 - 2a) chi))   (g(0) = a),
#   eta_p / eta_p0 = (1 - f)^2 / (1 + (1 - 2a) f),   Psi2 = -eta_p0 lambda g.
@inline function _giesekus_g(wi, a)
    x = 16 * a * (1 - a) * wi * wi
    s = sqrt(1 + x) + 1
    chi = sqrt(2 / s)
    return 16 * a * (1 - a) / (s * s * (1 + chi) * (1 + (1 - 2 * a) * chi))
end
@inline function _giesekus_eta_p(gdot, eta_p0, lambda, a)
    wi = lambda * gdot
    f = _giesekus_g(wi, a) * wi * wi
    return eta_p0 * (1 - f)^2 / (1 + (1 - 2 * a) * f)
end

"""
    GiesekusViscosityLaw()

Shear viscosity of the Giesekus model in closed form, one input `gdot`,
`theta = [eta_s, eta_p0, lambda, alpha]`: `eta_s + eta_p(gdot)`. Use it as the
`eta` law of [`FVReinerRivlinViscosity`](@ref) (with `eps > 0`).
"""
struct GiesekusViscosityLaw <: AbstractScalarLaw end
@inline law_ninputs(::GiesekusViscosityLaw) = 1
@inline law_nparams(::GiesekusViscosityLaw) = 4
law_init(::GiesekusViscosityLaw; rng = nothing) = [1.0, 1.0, 1.0, 0.4]
@inline law_value(::GiesekusViscosityLaw, x::Tuple, theta::AbstractVector) =
    @inbounds theta[1] + _giesekus_eta_p(x[1], theta[2], theta[3], theta[4])

"""
    GiesekusPsi2Law()

Second normal stress coefficient of the Giesekus model in closed form, one input
`gdot`, `theta = [eta_p0, lambda, alpha]`: `Psi2 = -eta_p0 lambda g(lambda gdot)`.
"""
struct GiesekusPsi2Law <: AbstractScalarLaw end
@inline law_ninputs(::GiesekusPsi2Law) = 1
@inline law_nparams(::GiesekusPsi2Law) = 3
law_init(::GiesekusPsi2Law; rng = nothing) = [1.0, 1.0, 0.4]
@inline law_value(::GiesekusPsi2Law, x::Tuple, theta::AbstractVector) =
    @inbounds -theta[1] * theta[2] * _giesekus_g(theta[2] * x[1], theta[3])

"""
    TabulatedGiesekusPsi2Law(nodes)

Giesekus `Psi2` times a piecewise-linear correction in `log10 Wi` (`Wi = lambda
gdot`): `Psi2 = -eta_p0 lambda g(Wi) c(log10 Wi)`, `c` linear between the
`nodes` (increasing `log10 Wi` values) and constant outside them.
`theta = [eta_p0, lambda, alpha, c_1, ..., c_N]`; all are differentiable.
"""
struct TabulatedGiesekusPsi2Law{N} <: AbstractScalarLaw
    nodes::NTuple{N,Float64}
    function TabulatedGiesekusPsi2Law(nodes::NTuple{N,Float64}) where {N}
        N >= 2 || throw(ArgumentError("at least two nodes are required"))
        all(m -> nodes[m + 1] > nodes[m], 1:N-1) || throw(ArgumentError("nodes must increase"))
        return new{N}(nodes)
    end
end
TabulatedGiesekusPsi2Law(nodes::AbstractVector) = TabulatedGiesekusPsi2Law(Tuple(Float64.(nodes)))
@inline law_ninputs(::TabulatedGiesekusPsi2Law) = 1
@inline law_nparams(::TabulatedGiesekusPsi2Law{N}) where {N} = 3 + N
law_init(law::TabulatedGiesekusPsi2Law{N}; rng = nothing) where {N} = [1.0; 1.0; 0.4; ones(N)]

@inline function _fv_hat_interp(nodes::NTuple{N,Float64}, s, theta, off) where {N}
    @inbounds begin
        s <= nodes[1] && return theta[off + 1] + zero(s)
        for m in 1:N-1
            if s <= nodes[m + 1]
                t = (s - nodes[m]) / (nodes[m + 1] - nodes[m])
                return theta[off + m] * (1 - t) + theta[off + m + 1] * t
            end
        end
        return theta[off + N] + zero(s)
    end
end
@inline function law_value(law::TabulatedGiesekusPsi2Law, x::Tuple, theta::AbstractVector)
    @inbounds begin
        wi = theta[2] * x[1]
        c = _fv_hat_interp(law.nodes, log10(max(wi, exp10(law.nodes[1]))), theta, 3)
        return -theta[1] * theta[2] * _giesekus_g(wi, theta[3]) * c
    end
end

# ------------------------------------------------------------------ learned Psi2
"""
    NeuralPsi2Law(mlp::TinyMLP{1}; wi0 = 1e-2)

Learned second normal stress coefficient in the dimensionless form of the
Giesekus closed form, one input `gdot`:

    Psi2 = -eta_p0 lambda g(Wi),   Wi = lambda gdot,   g = mlp(log(Wi + wi0)),

with the network input normalised by `mlp.shift` and `mlp.scale`.
`theta = [eta_p0, lambda, theta_mlp...]` (all differentiable; a calibration
usually holds the first two fixed and learns the network). `wi0 > 0` keeps the
input finite at rest. [`law_init`](@ref) gives `g = 0` (output layer zero) with
unit-variance hidden weights and biases, so `Psi2 = 0` at the start.
[`neural_psi2_g`](@ref) evaluates `g(Wi)`.
"""
struct NeuralPsi2Law{M<:TinyMLP{1}} <: AbstractScalarLaw
    mlp::M
    wi0::Float64
    function NeuralPsi2Law(mlp::M; wi0::Real = 1e-2) where {M<:TinyMLP{1}}
        isfinite(wi0) && wi0 > 0 || throw(ArgumentError("wi0 must be finite and positive"))
        return new{M}(mlp, Float64(wi0))
    end
end
@inline law_ninputs(::NeuralPsi2Law) = 1
@inline law_nparams(law::NeuralPsi2Law) = 2 + law_nparams(law.mlp)
@inline _fv_law_number_type(::Type{NeuralPsi2Law{M}}) where {M} = promote_type(Float64, _fv_law_number_type(M))

"""
    neural_psi2_g(law::NeuralPsi2Law, wi, theta)

The dimensionless `g(Wi) = -Psi2 / (eta_p0 lambda)` of the law at `Wi`.
"""
@inline neural_psi2_g(law::NeuralPsi2Law, wi, theta::AbstractVector) =
    _mlp_value(law.mlp, (log(wi + law.wi0),), theta, Val(false), 2)
@inline law_value(law::NeuralPsi2Law, x::Tuple, theta::AbstractVector) =
    @inbounds -theta[1] * theta[2] * neural_psi2_g(law, theta[2] * x[1], theta)

function law_init(law::NeuralPsi2Law; rng = nothing, eta_p0 = 1.0, lambda = 1.0)
    r() = rng === nothing ? randn() : randn(rng)
    m = law.mlp
    theta = [r() for _ in 1:law_nparams(m)]
    theta[(_mlp_output_offset(m) + 1):end] .= 0      # output weights and bias: g = 0
    return [Float64(eta_p0); Float64(lambda); theta]
end

# ------------------------------------------------------------------ parameters
"""
    FVReinerRivlinParameters(neta, npsi) <: FVParameterMap

Map `theta = [theta_eta; theta_psi2]` to an [`FVReinerRivlinViscosity`](@ref)
problem: `neta` parameters of the `eta` law (0 for an `FVGNFViscosity`, which
stays fixed) and `npsi` of the `psi2` law. The adjoint uses the generic
parameter-dual residual evaluations (`ceil(P / chunk)` residual passes).
"""
struct FVReinerRivlinParameters <: FVParameterMap
    neta::Int
    npsi::Int
    function FVReinerRivlinParameters(neta::Integer, npsi::Integer)
        neta >= 0 && npsi >= 0 || throw(ArgumentError("parameter counts must be nonnegative"))
        return new(Int(neta), Int(npsi))
    end
end
FVReinerRivlinParameters(v::FVReinerRivlinViscosity) =
    FVReinerRivlinParameters(v.eta isa FVLawViscosity ? law_nparams(v.eta.law) : 0,
                             law_nparams(v.psi2.law))
fv_nparams(m::FVReinerRivlinParameters, prob) = m.neta + m.npsi

const _FVRRProblem3D = FVSteadyProblem3D{<:FVGrid3D,<:Any,<:Any,<:Any,<:Any,<:FVReinerRivlinViscosity}

function _fv_param_buffers(m::FVReinerRivlinParameters, p::_FVRRProblem3D, ::Type{T}) where {T}
    v = p.visc
    m.npsi == law_nparams(v.psi2.law) ||
        throw(DimensionMismatch("FVReinerRivlinParameters psi2 count does not match the law"))
    if v.eta isa FVLawViscosity
        m.neta == law_nparams(v.eta.law) ||
            throw(DimensionMismatch("FVReinerRivlinParameters eta count does not match the law"))
        return (similar(v.eta.theta, T), similar(v.psi2.theta, T))
    end
    m.neta == 0 || throw(ArgumentError("an FVGNFViscosity eta has no mapped parameters (neta = 0)"))
    return (nothing, similar(v.psi2.theta, T))
end
_fv_with_theta(v::FVLawViscosity, theta) = FVLawViscosity(v.law, theta, v.fields, v.eps)
function _fv_param_problem!(B, m::FVReinerRivlinParameters, p::_FVRRProblem3D, values)
    v = p.visc
    eta = v.eta
    if B[1] !== nothing
        copyto!(B[1], 1, values, 1, m.neta)
        eta = _fv_with_theta(v.eta, B[1])
    end
    copyto!(B[2], 1, values, m.neta + 1, m.npsi)
    return _fv_with(p; visc = FVReinerRivlinViscosity(eta, _fv_with_theta(v.psi2, B[2])))
end
_fv_param_problem!(B, m::FVReinerRivlinParameters, p::FVSteadyProblem, values) =
    throw(ArgumentError("FVReinerRivlinParameters requires a 3D FVReinerRivlinViscosity problem"))
