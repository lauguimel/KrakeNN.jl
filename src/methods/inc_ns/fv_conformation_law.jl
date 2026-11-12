# Relaxation laws and pointwise conformation-tensor algebra of the steady FV
# conformation equation (plan "CONF", rungs CONF-1/CONF-1b; see fv_conformation_3d.jl).
#
# The conformation c is a symmetric 3 x 3 tensor stored as the 6-tuple
#   (c_xx, c_xy, c_xz, c_yy, c_yz, c_zz).
# The steady upper-convected equation of one fluid particle reads
#   u . grad c - L c - c L^T + (1/lambda) Rlx(c) = 0 ,   L_ij = d u_i / d x_j,
# with the general isotropic relaxation term (Cayley-Hamilton: every isotropic
# tensor function of a symmetric c is of this form)
#   Rlx(c) = b0 I + b1 (c - I) + b2 (c - I)^2 ,   b_k = b_k(I1, I2, I3; theta),
# frame-indifferent for any b_k; zero at rest (c = I) iff b0(3, 3, 1) = 0, which
# every law below satisfies. Oldroyd-B: (0, 1, 0). Giesekus: (0, 1, alpha).
# PTT: (0, f(tr c), 0). FENE-P: (f - 1, f, 0), i.e. f c - I. A learned law only
# has to supply the scalar coefficients from the invariants and theta; the
# polynomial in c is shared. The polymer stress is tau_p = G S(c),
# S = `relaxation_stress(law, theta, c)`, c - I unless the law says otherwise
# (FENE-P: f c - I). Every function here is generic in the number type
# (Float64, Float32, BigFloat, ForwardDiff.Dual) and branch-free in c, so dual
# derivatives are exact wherever the coefficients are smooth (polynomial laws
# everywhere, including at c = I; exponential PTT everywhere; FENE-P for
# tr c < L^2).

"""
    AbstractRelaxationLaw

Relaxation law of the steady conformation equation
`Rlx(c) = b0 I + b1 (c - I) + b2 (c - I)^2`. A subtype `R` implements

- `relaxation_nparams(::R) -> Int`, the length of its parameter tuple `theta`;
- `relaxation_coefficients(::R, theta, I1, I2, I3) -> (b0, b1, b2)`, the scalar
  coefficients from the invariants of `c` (`I1 = tr c`,
  `I2 = (tr(c)^2 - tr(c^2)) / 2`, `I3 = det c`) and the parameters; `b0` must
  vanish at rest (`I1 = I2 = 3`, `I3 = 1`). A 2-tuple `(b1, b2)` is accepted
  and means `b0 = 0` (the CONF-1 interface);

and optionally

- `relaxation_stress(::R, theta, c) -> NTuple{6}`, the dimensionless polymer
  stress `S(c)` of `τ_p = G S(c)` (default `c - I`), with
  `relaxation_stress_linear(::R) = false` when `S` is not linear in the
  components `(a, b)` of row `a` (the Jacobian pattern then couples each
  momentum row to all six components);
- `relaxation_check(::R, theta)`, throwing `ArgumentError` on invalid
  parameters (default: none).

Instances are isbits (passed by value to the residual kernel); `theta` is an
isbits tuple. Provided: [`PolynomialRelaxation`](@ref) (Oldroyd-B, Giesekus),
[`LinearPTTRelaxation`](@ref), [`ExponentialPTTRelaxation`](@ref),
[`FENEPRelaxation`](@ref).
"""
abstract type AbstractRelaxationLaw end

relaxation_check(::AbstractRelaxationLaw, theta) = nothing
@inline relaxation_stress_linear(::AbstractRelaxationLaw) = true

"""
    PolynomialRelaxation()

Constant-coefficient relaxation `Rlx(c) = theta[1] (c - I) + theta[2] (c - I)^2`
(`b0 = 0`): Oldroyd-B with `theta = (1, 0)`, Giesekus with `theta = (1, alpha)`.
"""
struct PolynomialRelaxation <: AbstractRelaxationLaw end
relaxation_nparams(::PolynomialRelaxation) = 2
@inline relaxation_coefficients(::PolynomialRelaxation, theta, I1, I2, I3) = (theta[1], theta[2])

"""
    LinearPTTRelaxation()

Linear Phan-Thien–Tanner law with the upper-convected derivative (`ξ = 0`),
`theta = (epsilon,)`, `epsilon >= 0`:

```math
\\mathrm{Rlx}(\\mathbf{c}) = f(\\mathrm{tr}\\,\\mathbf{c})\\,(\\mathbf{c}-\\mathbf{I}),
\\qquad f = 1 + \\varepsilon\\,(\\mathrm{tr}\\,\\mathbf{c} - 3),
\\qquad \\boldsymbol{\\tau}_p = G(\\mathbf{c}-\\mathbf{I}),
```

i.e. `(b0, b1, b2) = (0, f, 0)`. With `τ_p = G (c - I)`, `η_p = G λ`, this is
`f(τ) τ_p + λ τ_p^∇ = 2 η_p D`, `f = 1 + (ελ/η_p) tr τ_p`, the linear form of
Phan-Thien & Tanner (1977), J. Non-Newtonian Fluid Mech. 2, 353–365,
doi:10.1016/0377-0257(77)80021-9, as written by Oliveira & Pinho (1999),
J. Fluid Mech. 387, 271–280, doi:10.1017/S002211209900453X (`ξ = 0`).

The Gordon–Schowalter derivative of the full PTT model (`ξ ≠ 0`) adds
`ξ (D τ + τ D)`, which depends on the rate of deformation `D` and not on `c`
alone: it cannot be written as an isotropic `Rlx(c)` and needs a modified
convected derivative in the residual. It is not implemented.

Simple shear (`Wi = λ γ̇`): `η_p(γ̇)/η_p = 1/f`, `Ψ1/(η_p λ) = 2/f²`,
`Ψ2 = 0`, with `f` the real root of `f³ - f² - 2 ε Wi² = 0`.
"""
struct LinearPTTRelaxation <: AbstractRelaxationLaw end
relaxation_nparams(::LinearPTTRelaxation) = 1
@inline function relaxation_coefficients(::LinearPTTRelaxation, theta, I1, I2, I3)
    f = 1 + theta[1] * (I1 - 3)
    return (zero(f), f, zero(f))
end
relaxation_check(::LinearPTTRelaxation, theta) =
    theta[1] >= 0 || throw(ArgumentError("PTT epsilon must be nonnegative"))

"""
    ExponentialPTTRelaxation()

Exponential Phan-Thien–Tanner law with the upper-convected derivative
(`ξ = 0`), `theta = (epsilon,)`, `epsilon >= 0`:

```math
\\mathrm{Rlx}(\\mathbf{c}) = f(\\mathrm{tr}\\,\\mathbf{c})\\,(\\mathbf{c}-\\mathbf{I}),
\\qquad f = \\exp\\!\\left[\\varepsilon\\,(\\mathrm{tr}\\,\\mathbf{c} - 3)\\right],
\\qquad \\boldsymbol{\\tau}_p = G(\\mathbf{c}-\\mathbf{I}),
```

i.e. `f(τ) = exp((ελ/η_p) tr τ_p)` of Phan-Thien (1978), J. Rheol. 22,
259–283, doi:10.1122/1.549481. `ξ ≠ 0` is left out for the reason given in
[`LinearPTTRelaxation`](@ref).

Simple shear: `η_p(γ̇)/η_p = 1/f`, `Ψ1/(η_p λ) = 2/f²`, `Ψ2 = 0`, with
`f² ln f² = 4 ε Wi²`, i.e. `f = exp(W(4 ε Wi²)/2)` (`W` the Lambert function).
"""
struct ExponentialPTTRelaxation <: AbstractRelaxationLaw end
relaxation_nparams(::ExponentialPTTRelaxation) = 1
@inline function relaxation_coefficients(::ExponentialPTTRelaxation, theta, I1, I2, I3)
    f = exp(theta[1] * (I1 - 3))
    return (zero(f), f, zero(f))
end
relaxation_check(::ExponentialPTTRelaxation, theta) =
    theta[1] >= 0 || throw(ArgumentError("PTT epsilon must be nonnegative"))

"""
    FENEPRelaxation()

FENE-P (Peterlin closure of the FENE dumbbell), `theta = (L2,)` with `L2 = L²`
the squared maximum extensibility, `L2 > 3`:

```math
\\mathrm{Rlx}(\\mathbf{c}) = f\\,\\mathbf{c} - \\mathbf{I},
\\qquad \\boldsymbol{\\tau}_p = G\\,(f\\,\\mathbf{c} - \\mathbf{I}),
\\qquad f = \\frac{L^2 - 3}{L^2 - \\mathrm{tr}\\,\\mathbf{c}},
```

i.e. `(b0, b1, b2) = (f - 1, f, 0)` and the stress hook
[`relaxation_stress`](@ref) `= f c - I`. The Peterlin closure is that of
Bird, Dotson & Johnson (1980), J. Non-Newtonian Fluid Mech. 7, 213–235,
doi:10.1016/0377-0257(80)85007-5; the factor is normalised so that `c = I` at
rest (`f(I) = 1`), the same factor as the log-conformation FENE-P kernel
(`logconf_source_with_divergence_fenep_3d`). Defined for `tr c < L²`: a Newton
iterate beyond that bound makes `f` negative or infinite (no clipping here, so
that the residual stays smooth).

Simple shear: `c_yy = c_zz = 1/f`, `c_xy = Wi/f²`, `c_xx = 1/f + 2 Wi²/f³`,
`η_p(γ̇)/η_p = 1/f`, `Ψ1/(η_p λ) = 2/f²`, `Ψ2 = 0`, with `f` the real root of
`f³ - f² - 2 Wi²/L² = 0` (the linear-PTT cubic with `ε = 1/L²`).
"""
struct FENEPRelaxation <: AbstractRelaxationLaw end
relaxation_nparams(::FENEPRelaxation) = 1
@inline _fvc_fenep_factor(theta, I1) = (theta[1] - 3) / (theta[1] - I1)
@inline function relaxation_coefficients(::FENEPRelaxation, theta, I1, I2, I3)
    f = _fvc_fenep_factor(theta, I1)
    return (f - 1, f, zero(f))
end
relaxation_check(::FENEPRelaxation, theta) =
    theta[1] > 3 || throw(ArgumentError("FENE-P L2 must exceed 3"))
@inline relaxation_stress_linear(::FENEPRelaxation) = false

# ------------------------------------------------------------------ symmetric 3 x 3 algebra
# (i, j) of the six stored components, and the storage index of (a, b).
const _FVC_I = (1, 1, 1, 2, 2, 3)
const _FVC_J = (1, 2, 3, 2, 3, 3)
@inline function _fvc_sym(a, b)
    lo = min(a, b); hi = max(a, b)
    return lo == 1 ? hi : (lo == 2 ? hi + 2 : 6)
end
@inline _fvc_get(c, a, b) = c[_fvc_sym(a, b)]
# L as the 9-tuple (ux, uy, uz, vx, vy, vz, wx, wy, wz): entry 3(a-1)+b = d u_a / d x_b
@inline _fvc_L(L, a, b) = L[3 * (a - 1) + b]

"""Invariants `(tr c, (tr(c)^2 - tr(c^2)) / 2, det c)` of a symmetric 6-tuple."""
@inline function _fvc_invariants(c)
    xx, xy, xz, yy, yz, zz = c
    I1 = xx + yy + zz
    I2 = xx * yy + yy * zz + xx * zz - xy * xy - xz * xz - yz * yz
    I3 = xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + xz * (xy * yz - yy * xz)
    return I1, I2, I3
end

"""`d * d` of a symmetric 6-tuple `d`, as a symmetric 6-tuple."""
@inline function _fvc_square(d)
    xx, xy, xz, yy, yz, zz = d
    return (xx * xx + xy * xy + xz * xz,
            xx * xy + xy * yy + xz * yz,
            xx * xz + xy * yz + xz * zz,
            xy * xy + yy * yy + yz * yz,
            xy * xz + yy * yz + yz * zz,
            xz * xz + yz * yz + zz * zz)
end

"""`c - I` of a symmetric 6-tuple."""
@inline _fvc_minus_identity(c) =
    (c[1] - one(c[1]), c[2], c[3], c[4] - one(c[4]), c[5], c[6] - one(c[6]))

"""
    fv_relaxation(law, theta, c) -> NTuple{6}

Relaxation term `b0 I + b1 (c - I) + b2 (c - I)^2` of the 6-tuple `c`
`(c_xx, c_xy, c_xz, c_yy, c_yz, c_zz)`, with the coefficients from
`relaxation_coefficients(law, theta, I1, I2, I3)` (a 2-tuple means `b0 = 0`
and gives the CONF-1 arithmetic unchanged).
"""
@inline function fv_relaxation(law::AbstractRelaxationLaw, theta, c)
    I1, I2, I3 = _fvc_invariants(c)
    b = relaxation_coefficients(law, theta, I1, I2, I3)
    d = _fvc_minus_identity(c)
    d2 = _fvc_square(d)
    return _fvc_relax(b, d, d2)
end
@inline _fvc_relax(b::Tuple{Any,Any}, d, d2) = ntuple(m -> b[1] * d[m] + b[2] * d2[m], Val(6))
@inline function _fvc_relax(b::Tuple{Any,Any,Any}, d, d2)
    b0, b1, b2 = b
    z = zero(b0)
    return ntuple(m -> b1 * d[m] + b2 * d2[m] + (_FVC_I[m] == _FVC_J[m] ? b0 : z), Val(6))
end

"""
    relaxation_stress(law, theta, c) -> NTuple{6}

Dimensionless polymer stress `S(c)` of `τ_p = G S(c)`, `G = ν_p/λ`: `c - I`
by default, `f c - I` for [`FENEPRelaxation`](@ref).
"""
@inline relaxation_stress(::AbstractRelaxationLaw, theta, c) = _fvc_minus_identity(c)
@inline function relaxation_stress(::FENEPRelaxation, theta, c)
    xx, xy, xz, yy, yz, zz = c
    f = _fvc_fenep_factor(theta, xx + yy + zz)
    return (f * xx - 1, f * xy, f * xz, f * yy - 1, f * yz, f * zz - 1)
end

"""
    fv_upper_convected(L, c) -> NTuple{6}

Stretching term `L c + c L^T` (symmetric) of the 6-tuple `c`, with `L` the
velocity gradient 9-tuple `(ux, uy, uz, vx, vy, vz, wx, wy, wz)`,
`L_ab = d u_a / d x_b`.
"""
@inline function fv_upper_convected(L, c)
    return ntuple(Val(6)) do m
        a = _FVC_I[m]; b = _FVC_J[m]
        (_fvc_L(L, a, 1) * _fvc_get(c, 1, b) + _fvc_L(L, a, 2) * _fvc_get(c, 2, b) +
         _fvc_L(L, a, 3) * _fvc_get(c, 3, b)) +
        (_fvc_get(c, a, 1) * _fvc_L(L, b, 1) + _fvc_get(c, a, 2) * _fvc_L(L, b, 2) +
         _fvc_get(c, a, 3) * _fvc_L(L, b, 3))
    end
end

"""
    fv_conformation_source(law, theta, lambda, L, c) -> NTuple{6}

Homogeneous steady residual `-(L c + c L^T) + Rlx(c) / lambda` (the conformation
equation without its advection term): zero at the steady state of a
homogeneous flow of velocity gradient `L`.
"""
@inline function fv_conformation_source(law::AbstractRelaxationLaw, theta, lambda, L, c)
    S = fv_upper_convected(L, c)
    r = fv_relaxation(law, theta, c)
    il = inv(lambda)
    return ntuple(m -> r[m] * il - S[m], Val(6))
end
