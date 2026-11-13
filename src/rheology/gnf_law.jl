# --- Scalar constitutive laws (ClassicalLaw, DenseNet, ...) in the 2D LBM GNF path ---
#
# The GNF step (`gnf_step!`, `run_gnf_periodic_steady`) evaluates its viscosity
# through `gnf_viscosity(closure, I1)` with an untyped isbits `closure`. The
# wrapper below gives any one-input `AbstractScalarLaw` with its flat parameter
# vector such a method, so that the law slot used by `FVLawViscosity` also
# drives the lattice Boltzmann solver. The value is the FV face hook
# `_fv_law_invariant` evaluated on the physical invariant: the same expression,
# including the regular-at-rest Carreau--Yasuda adapter.
#
# The steady adjoint (`gnf_linearise`, `gnf_steady_adjoint`) is closure-generic;
# only the parameter contraction Σ_c s_c ∂ν_c/∂θ needs the law, done here with
# `law_pullback!` (handwritten reverse pass for DenseNet), as in the FV adjoint.

"""
    LBMLawViscosity(law::AbstractScalarLaw, theta; eps=1e-3, nu_scale=1, gamma_scale=1)

Closure of the 2D LBM GNF step (`run_gnf_periodic_steady`, `gnf_step!`,
`gnf_linearise`) built from a one-input scalar law (`law_ninputs(law) == 1`,
input = shear rate), for instance [`ClassicalLaw`](@ref) or [`DenseNet`](@ref).
The lattice kinematic viscosity at the strain-rate invariant `I1 = S:S`
(lattice units) is

    ν_lat = nu_scale * law(γ̇),   γ̇ = √(2 (gamma_scale² I1 + eps²/2)),

i.e. the law is written in its own (physical) units: `gamma_scale = 1/dt`
converts the lattice shear rate, `nu_scale = dt/dx²` converts the viscosity,
and `eps` is the regularisation of [`FVLawViscosity`](@ref) in the law units.
With the same `law`, `theta` and `eps`, the FV and LBM solvers therefore
evaluate the same function of the physical invariant.

`eps` must be positive except for a regular-at-rest `ClassicalLaw`
(Carreau--Yasuda with `a ≥ 2`). With `eps = 0` and unit scales, a
`ClassicalLaw(CarreauYasuda(...))` gives bit for bit the viscosity of the
`CarreauYasuda` closure itself. `theta` may live on any backend (it is adapted
at kernel launch); host evaluations (rest relaxation time, adjoint blocks) use
a host copy. Parameter gradients: [`gnf_law_gradient`](@ref),
[`gnf_steady_law_gradient`](@ref).
"""
struct LBMLawViscosity{L<:AbstractScalarLaw,A,T}
    law::L
    theta::A
    eps::T
    nu_scale::T
    gamma_scale::T
end

function LBMLawViscosity(law::AbstractScalarLaw, theta::AbstractVector; eps::Real=1e-3,
                         nu_scale::Real=1.0, gamma_scale::Real=1.0)
    isbitstype(typeof(law)) || throw(ArgumentError("LBM scalar law must be isbits"))
    law_ninputs(law) == 1 ||
        throw(ArgumentError("the LBM law slot takes the shear rate only (law_ninputs == 1)"))
    length(theta) == law_nparams(law) ||
        throw(DimensionMismatch("theta must have $(law_nparams(law)) parameters"))
    e, ns, gs = promote(float(eps), float(nu_scale), float(gamma_scale))
    isfinite(e) && e >= 0 || throw(ArgumentError("eps must be finite and nonnegative"))
    isfinite(ns) && ns > 0 && isfinite(gs) && gs > 0 ||
        throw(ArgumentError("nu_scale and gamma_scale must be finite and positive"))
    e > 0 || _fv_law_regular_at_rest(law, theta) ||
        throw(ArgumentError("this scalar law requires eps > 0 to be differentiable at rest"))
    return LBMLawViscosity(law, theta, e, ns, gs)
end

@inline _gnf_law_invariant(c::LBMLawViscosity, I1) =
    c.gamma_scale * c.gamma_scale * I1 + c.eps * c.eps / 2

@inline gnf_viscosity(c::LBMLawViscosity, I1) =
    c.nu_scale * _fv_law_invariant(c.law, c.theta, _gnf_law_invariant(c, I1), ())

function gnf_convert_closure(::Type{T}, c::LBMLawViscosity) where {T<:AbstractFloat}
    theta = eltype(c.theta) === T ? c.theta : T.(c.theta)
    return LBMLawViscosity(c.law, theta, T(c.eps), T(c.nu_scale), T(c.gamma_scale))
end

# Host evaluations of a closure (rest relaxation time, adjoint blocks). Identity
# for every closure without device storage.
_gnf_host_closure(c) = c
_gnf_host_closure(c::LBMLawViscosity) =
    c.theta isa Array ? c : LBMLawViscosity(c.law, Array(c.theta), c.eps, c.nu_scale, c.gamma_scale)

function KernelAbstractions.Adapt.adapt_structure(to, c::LBMLawViscosity)
    return LBMLawViscosity(c.law, KernelAbstractions.Adapt.adapt(to, c.theta),
                           c.eps, c.nu_scale, c.gamma_scale)
end

"""
    gnf_law_gradient(lin::GNFLinearisation, s, closure::LBMLawViscosity) -> Vector{Float64}

Parameter gradient `Σ_c s_c ∂ν_lat,c/∂θ` for a law closure, with `s` the
per-cell cotangent of the lattice viscosity (`gnf_nu_sensitivity`,
`gnf_steady_adjoint`) and the strain invariants of the linearisation `lin`.
Each fluid cell calls `law_pullback!(g, law, (γ̇_c,), θ, nu_scale * s_c)` at the
regularised shear rate of [`LBMLawViscosity`](@ref), the state held fixed (the
state dependence of ν is in the adjoint blocks). Host computation, cost
O(cells × parameters).
"""
function gnf_law_gradient(lin::GNFLinearisation, s::AbstractMatrix, closure::LBMLawViscosity)
    c = _gnf_host_closure(gnf_convert_closure(Float64, closure))
    size(s) == (lin.Nx, lin.Ny) || throw(DimensionMismatch("s must be $(lin.Nx)×$(lin.Ny)"))
    g = zeros(law_nparams(c.law))
    @inbounds for j in 1:lin.Ny, i in 1:lin.Nx
        lin.is_solid[i, j] && continue
        sc = s[i, j]
        iszero(sc) && continue
        x = (gnf_shear_rate(_gnf_law_invariant(c, lin.I1[i, j])),)
        law_pullback!(g, c.law, x, c.theta, c.nu_scale * sc)
    end
    return g
end

"""
    gnf_steady_law_gradient(f, tau, gu; is_solid, closure::LBMLawViscosity, Fx, Fy=0,
                            backend=CPU(), T=Float64, kwargs...) -> NamedTuple

Gradient with respect to every law parameter θ of an objective L of the
post-collision lattice velocity at the fixed point (f, tau) of the GNF step
with `closure`: `gu = ∂L/∂u_lat` in any form accepted by `gnf_steady_adjoint`
(Nx × Ny × 2 array, tuple or `(; ux, uy)`). Linearises (`gnf_linearise`),
solves the steady adjoint (`gnf_steady_adjoint`, `kwargs` passed on, default
solver `:kstep`), then contracts with [`gnf_law_gradient`](@ref). Returns
`(; grad, s, lin, op, stats, t_lin, t_adjoint, t_grad)` (times in seconds).
"""
function gnf_steady_law_gradient(f, tau, gu; is_solid, closure::LBMLawViscosity, Fx, Fy=0,
                                 backend=KernelAbstractions.CPU(), T::Type=Float64, kwargs...)
    t0 = time()
    lin = gnf_linearise(f, tau, is_solid, closure, Fx, Fy)
    t1 = time()
    op = GNFStepOperator(lin; backend, T)
    s, stats = gnf_steady_adjoint(lin, op, gu; t_lin=t1 - t0, kwargs...)
    t2 = time()
    grad = gnf_law_gradient(lin, s, closure)
    t3 = time()
    return (; grad, s, lin, op, stats, t_lin=t1 - t0, t_adjoint=t2 - t1, t_grad=t3 - t2)
end
