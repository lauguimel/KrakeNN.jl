# --- Strain rate from non-equilibrium distributions ---
#
# The strain rate tensor is computed from the non-equilibrium stress tensor:
#   Π_αβ^neq = Σ_q (f_q - f_eq_q) · c_qα · c_qβ
#   S_αβ = -Π_αβ^neq / (2ρcs²τ)     [cs² = 1/3 for D2Q9/D3Q19]
#
# The second invariant (shear rate magnitude):
#   γ̇ = √(2 · S_αβ · S_αβ)  =  √(2(S_xx² + S_yy² + 2·S_xy²))  in 2D
#
# This is a PURELY LOCAL operation (no neighbor reads) → ideal for GPU.

"""
    strain_rate_magnitude_2d(f1,...,f9, feq1,...,feq9, rho, tau) → γ̇

Compute the shear rate magnitude from non-equilibrium distributions (D2Q9).

Uses the non-equilibrium stress tensor Π^neq = Σ (f - f_eq) · c ⊗ c,
with the relation S_αβ = -Π^neq_αβ / (2ρcs²τ).

# D2Q9 velocity convention (1-based):
    q=1: (0,0), q=2: (1,0), q=3: (0,1), q=4: (-1,0), q=5: (0,-1)
    q=6: (1,1), q=7: (-1,1), q=8: (-1,-1), q=9: (1,-1)
"""
@inline function strain_rate_magnitude_2d(
    f1, f2, f3, f4, f5, f6, f7, f8, f9,
    feq1, feq2, feq3, feq4, feq5, feq6, feq7, feq8, feq9,
    rho, tau
)
    T = typeof(rho)

    # Non-equilibrium parts
    fneq1 = f1 - feq1; fneq2 = f2 - feq2; fneq3 = f3 - feq3
    fneq4 = f4 - feq4; fneq5 = f5 - feq5; fneq6 = f6 - feq6
    fneq7 = f7 - feq7; fneq8 = f8 - feq8; fneq9 = f9 - feq9

    # Π_xx = Σ fneq_q · cx_q²
    # Only q with cx ≠ 0 contribute: q=2(+1), q=4(-1), q=6(+1), q=7(-1), q=8(-1), q=9(+1)
    # cx² = 1 for all of these
    Pi_xx = fneq2 + fneq4 + fneq6 + fneq7 + fneq8 + fneq9

    # Π_yy = Σ fneq_q · cy_q²
    # q with cy ≠ 0: q=3(+1), q=5(-1), q=6(+1), q=7(+1), q=8(-1), q=9(-1)
    Pi_yy = fneq3 + fneq5 + fneq6 + fneq7 + fneq8 + fneq9

    # Π_xy = Σ fneq_q · cx_q · cy_q
    # q=6: cx=+1, cy=+1 → +1
    # q=7: cx=-1, cy=+1 → -1
    # q=8: cx=-1, cy=-1 → +1
    # q=9: cx=+1, cy=-1 → -1
    Pi_xy = fneq6 - fneq7 + fneq8 - fneq9

    # S_αβ = -Π_αβ / (2 · ρ · cs² · τ)  with cs² = 1/3
    inv_denom = -one(T) / (T(2) * rho * T(1.0/3.0) * tau)
    # Simplify: inv_denom = -3 / (2ρτ)
    S_xx = Pi_xx * inv_denom
    S_yy = Pi_yy * inv_denom
    S_xy = Pi_xy * inv_denom

    # γ̇ = √(2 · S_αβ · S_αβ) = √(2(S_xx² + S_yy² + 2·S_xy²))
    return sqrt(T(2) * (S_xx * S_xx + S_yy * S_yy + T(2) * S_xy * S_xy))
end

"""
    strain_rate_invariant_2d(f1,...,f9, feq1,...,feq9, rho, tau) → I1 = S:S

Invariant `S:S = S_xx² + S_yy² + 2·S_xy²` of the strain-rate tensor (D2Q9),
computed with the same operations in the same order as
`strain_rate_magnitude_2d`, so that `sqrt(2 * I1)` equals its result bit for
bit. Unlike `γ̇ = √(2 S:S)`, `I1` is a polynomial in the populations: its
derivative is finite where `S = 0`, which a closure written in `I1` keeps
(the square root has an infinite derivative at 0, and ForwardDiff then
returns NaN).
"""
@inline function strain_rate_invariant_2d(
    f1, f2, f3, f4, f5, f6, f7, f8, f9,
    feq1, feq2, feq3, feq4, feq5, feq6, feq7, feq8, feq9,
    rho, tau
)
    T = typeof(rho)

    fneq1 = f1 - feq1; fneq2 = f2 - feq2; fneq3 = f3 - feq3
    fneq4 = f4 - feq4; fneq5 = f5 - feq5; fneq6 = f6 - feq6
    fneq7 = f7 - feq7; fneq8 = f8 - feq8; fneq9 = f9 - feq9

    Pi_xx = fneq2 + fneq4 + fneq6 + fneq7 + fneq8 + fneq9
    Pi_yy = fneq3 + fneq5 + fneq6 + fneq7 + fneq8 + fneq9
    Pi_xy = fneq6 - fneq7 + fneq8 - fneq9

    inv_denom = -one(T) / (T(2) * rho * T(1.0/3.0) * tau)
    S_xx = Pi_xx * inv_denom
    S_yy = Pi_yy * inv_denom
    S_xy = Pi_xy * inv_denom

    return S_xx * S_xx + S_yy * S_yy + T(2) * S_xy * S_xy
end

"""
    strain_rate_magnitude_3d(f1,...,f19, feq1,...,feq19, rho, tau) → γ̇

Compute the shear rate magnitude from non-equilibrium distributions (D3Q19).

# D3Q19 velocity convention (1-based):
    q=1:  (0,0,0)   rest
    q=2:  (1,0,0)   q=3:  (-1,0,0)
    q=4:  (0,1,0)   q=5:  (0,-1,0)
    q=6:  (0,0,1)   q=7:  (0,0,-1)
    q=8:  (1,1,0)   q=9:  (-1,-1,0)
    q=10: (1,-1,0)  q=11: (-1,1,0)
    q=12: (1,0,1)   q=13: (-1,0,-1)
    q=14: (1,0,-1)  q=15: (-1,0,1)
    q=16: (0,1,1)   q=17: (0,-1,-1)
    q=18: (0,1,-1)  q=19: (0,-1,1)
"""
@inline function strain_rate_magnitude_3d(
    f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
    f11, f12, f13, f14, f15, f16, f17, f18, f19,
    feq1, feq2, feq3, feq4, feq5, feq6, feq7, feq8, feq9, feq10,
    feq11, feq12, feq13, feq14, feq15, feq16, feq17, feq18, feq19,
    rho, tau
)
    T = typeof(rho)

    fneq2 = f2 - feq2; fneq3 = f3 - feq3
    fneq4 = f4 - feq4; fneq5 = f5 - feq5
    fneq6 = f6 - feq6; fneq7 = f7 - feq7
    fneq8 = f8 - feq8; fneq9 = f9 - feq9
    fneq10 = f10 - feq10; fneq11 = f11 - feq11
    fneq12 = f12 - feq12; fneq13 = f13 - feq13
    fneq14 = f14 - feq14; fneq15 = f15 - feq15
    fneq16 = f16 - feq16; fneq17 = f17 - feq17
    fneq18 = f18 - feq18; fneq19 = f19 - feq19

    # Π_xx = Σ fneq · cx² (q with cx≠0: 2,3,8,9,10,11,12,13,14,15)
    Pi_xx = fneq2 + fneq3 + fneq8 + fneq9 + fneq10 + fneq11 + fneq12 + fneq13 + fneq14 + fneq15

    # Π_yy = Σ fneq · cy² (q with cy≠0: 4,5,8,9,10,11,16,17,18,19)
    Pi_yy = fneq4 + fneq5 + fneq8 + fneq9 + fneq10 + fneq11 + fneq16 + fneq17 + fneq18 + fneq19

    # Π_zz = Σ fneq · cz² (q with cz≠0: 6,7,12,13,14,15,16,17,18,19)
    Pi_zz = fneq6 + fneq7 + fneq12 + fneq13 + fneq14 + fneq15 + fneq16 + fneq17 + fneq18 + fneq19

    # Π_xy = Σ fneq · cx · cy
    # q=8: (+1)(+1)=+1, q=9: (-1)(-1)=+1, q=10: (+1)(-1)=-1, q=11: (-1)(+1)=-1
    Pi_xy = fneq8 + fneq9 - fneq10 - fneq11

    # Π_xz = Σ fneq · cx · cz
    # q=12: (+1)(+1)=+1, q=13: (-1)(-1)=+1, q=14: (+1)(-1)=-1, q=15: (-1)(+1)=-1
    Pi_xz = fneq12 + fneq13 - fneq14 - fneq15

    # Π_yz = Σ fneq · cy · cz
    # q=16: (+1)(+1)=+1, q=17: (-1)(-1)=+1, q=18: (+1)(-1)=-1, q=19: (-1)(+1)=-1
    Pi_yz = fneq16 + fneq17 - fneq18 - fneq19

    inv_denom = -T(3) / (T(2) * rho * tau)
    S_xx = Pi_xx * inv_denom
    S_yy = Pi_yy * inv_denom
    S_zz = Pi_zz * inv_denom
    S_xy = Pi_xy * inv_denom
    S_xz = Pi_xz * inv_denom
    S_yz = Pi_yz * inv_denom

    # γ̇ = √(2 · S_αβ · S_αβ) = √(2(S_xx² + S_yy² + S_zz² + 2(S_xy² + S_xz² + S_yz²)))
    return sqrt(T(2) * (S_xx*S_xx + S_yy*S_yy + S_zz*S_zz +
                        T(2) * (S_xy*S_xy + S_xz*S_xz + S_yz*S_yz)))
end
