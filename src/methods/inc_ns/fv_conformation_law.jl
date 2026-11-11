# Relaxation laws and pointwise conformation-tensor algebra of the steady FV
# conformation equation (plan "CONF", rung CONF-1; see fv_conformation_3d.jl).
#
# The conformation c is a symmetric 3 x 3 tensor stored as the 6-tuple
#   (c_xx, c_xy, c_xz, c_yy, c_yz, c_zz).
# The steady upper-convected equation of one fluid particle reads
#   u . grad c - L c - c L^T + (1/lambda) Rlx(c) = 0 ,   L_ij = d u_i / d x_j,
# with the isotropic relaxation term
#   Rlx(c) = b1(I1, I2, I3; theta) (c - I) + b2(I1, I2, I3; theta) (c - I)^2 ,
# frame-indifferent and zero at rest (c = I) by construction, for any b1, b2.
# Oldroyd-B: b1 = 1, b2 = 0. Giesekus: b1 = 1, b2 = alpha. A learned law only
# has to supply the two scalar coefficients from the invariants and theta; the
# polynomial in c is shared. Every function here is generic in the number type
# (Float64, Float32, BigFloat, ForwardDiff.Dual): it is polynomial in c and in L,
# so dual derivatives are exact everywhere, including at c = I.

"""
    AbstractRelaxationLaw

Relaxation law of the steady conformation equation
`Rlx(c) = b1 (c - I) + b2 (c - I)^2`. A subtype `R` implements

- `relaxation_nparams(::R) -> Int`, the length of its parameter tuple `theta`;
- `relaxation_coefficients(::R, theta, I1, I2, I3) -> (b1, b2)`, the two scalar
  coefficients from the invariants of `c` (`I1 = tr c`,
  `I2 = (tr(c)^2 - tr(c^2)) / 2`, `I3 = det c`) and the parameters.

Instances are isbits (passed by value to the residual kernel); `theta` is an
isbits tuple. [`PolynomialRelaxation`](@ref) covers Oldroyd-B and Giesekus.
"""
abstract type AbstractRelaxationLaw end

"""
    PolynomialRelaxation()

Constant-coefficient relaxation `Rlx(c) = theta[1] (c - I) + theta[2] (c - I)^2`:
Oldroyd-B with `theta = (1, 0)`, Giesekus with `theta = (1, alpha)`.
"""
struct PolynomialRelaxation <: AbstractRelaxationLaw end
relaxation_nparams(::PolynomialRelaxation) = 2
@inline relaxation_coefficients(::PolynomialRelaxation, theta, I1, I2, I3) = (theta[1], theta[2])

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

Relaxation term `b1 (c - I) + b2 (c - I)^2` of the 6-tuple `c`
`(c_xx, c_xy, c_xz, c_yy, c_yz, c_zz)`, with `(b1, b2)` from
`relaxation_coefficients(law, theta, I1, I2, I3)`.
"""
@inline function fv_relaxation(law::AbstractRelaxationLaw, theta, c)
    I1, I2, I3 = _fvc_invariants(c)
    b1, b2 = relaxation_coefficients(law, theta, I1, I2, I3)
    d = _fvc_minus_identity(c)
    d2 = _fvc_square(d)
    return ntuple(m -> b1 * d[m] + b2 * d2[m], Val(6))
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
