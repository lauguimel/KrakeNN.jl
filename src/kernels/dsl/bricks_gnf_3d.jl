"""
    CollideTRTGNFGuoField_3D()

D3Q19 TRT collision with isothermal generalized Newtonian viscosity and a
per-cell Guo force. Equilibrium uses `u_star = u_bare + F/(2rho)`; the strain
moment includes `(F*u_star' + u_star*F')/2`. The strain denominator uses the
previous even relaxation time `1/sp_field`, then both relaxation fields are
updated from the model viscosity and TRT magic parameter. Only the rheology
model's own viscosity limits apply. Guo even and odd source parts use their
respective new TRT rates, adding exactly F to the collision momentum.
"""
struct CollideTRTGNFGuoField_3D <: LBMBrick end
required_args(::CollideTRTGNFGuoField_3D) =
    (:f_out, :Fx_field, :Fy_field, :Fz_field, :rheology, :sp_field, :sm_field, :trt_lambda)
phase(::CollideTRTGNFGuoField_3D) = :fluid
emit_code(::CollideTRTGNFGuoField_3D) = quote
    fx = Fx_field[i, j, k]
    fy = Fy_field[i, j, k]
    fz = Fz_field[i, j, k]
    if fx != zero(T) || fy != zero(T) || fz != zero(T)
        inv_ρ = one(T) / ρ
        ux = (ρ * ux + fx / T(2)) * inv_ρ
        uy = (ρ * uy + fy / T(2)) * inv_ρ
        uz = (ρ * uz + fz / T(2)) * inv_ρ
        usq = ux * ux + uy * uy + uz * uz
    end

    feq1  = feq_3d(Val(1),  ρ, ux, uy, uz, usq)
    feq2  = feq_3d(Val(2),  ρ, ux, uy, uz, usq)
    feq3  = feq_3d(Val(3),  ρ, ux, uy, uz, usq)
    feq4  = feq_3d(Val(4),  ρ, ux, uy, uz, usq)
    feq5  = feq_3d(Val(5),  ρ, ux, uy, uz, usq)
    feq6  = feq_3d(Val(6),  ρ, ux, uy, uz, usq)
    feq7  = feq_3d(Val(7),  ρ, ux, uy, uz, usq)
    feq8  = feq_3d(Val(8),  ρ, ux, uy, uz, usq)
    feq9  = feq_3d(Val(9),  ρ, ux, uy, uz, usq)
    feq10 = feq_3d(Val(10), ρ, ux, uy, uz, usq)
    feq11 = feq_3d(Val(11), ρ, ux, uy, uz, usq)
    feq12 = feq_3d(Val(12), ρ, ux, uy, uz, usq)
    feq13 = feq_3d(Val(13), ρ, ux, uy, uz, usq)
    feq14 = feq_3d(Val(14), ρ, ux, uy, uz, usq)
    feq15 = feq_3d(Val(15), ρ, ux, uy, uz, usq)
    feq16 = feq_3d(Val(16), ρ, ux, uy, uz, usq)
    feq17 = feq_3d(Val(17), ρ, ux, uy, uz, usq)
    feq18 = feq_3d(Val(18), ρ, ux, uy, uz, usq)
    feq19 = feq_3d(Val(19), ρ, ux, uy, uz, usq)

    # Non-equilibrium moments use the half-force-shifted equilibrium above.
    fneq2 = fp2 - feq2; fneq3 = fp3 - feq3
    fneq4 = fp4 - feq4; fneq5 = fp5 - feq5
    fneq6 = fp6 - feq6; fneq7 = fp7 - feq7
    fneq8 = fp8 - feq8; fneq9 = fp9 - feq9
    fneq10 = fp10 - feq10; fneq11 = fp11 - feq11
    fneq12 = fp12 - feq12; fneq13 = fp13 - feq13
    fneq14 = fp14 - feq14; fneq15 = fp15 - feq15
    fneq16 = fp16 - feq16; fneq17 = fp17 - feq17
    fneq18 = fp18 - feq18; fneq19 = fp19 - feq19
    Pi_xx = fneq2 + fneq3 + fneq8 + fneq9 + fneq10 + fneq11 + fneq12 + fneq13 + fneq14 + fneq15
    Pi_yy = fneq4 + fneq5 + fneq8 + fneq9 + fneq10 + fneq11 + fneq16 + fneq17 + fneq18 + fneq19
    Pi_zz = fneq6 + fneq7 + fneq12 + fneq13 + fneq14 + fneq15 + fneq16 + fneq17 + fneq18 + fneq19
    Pi_xy = fneq8 - fneq9 - fneq10 + fneq11
    Pi_xz = fneq12 - fneq13 - fneq14 + fneq15
    Pi_yz = fneq16 - fneq17 - fneq18 + fneq19

    # Guo stress correction: Pi += (F*u_star' + u_star*F')/2.
    Pi_xx += fx * ux
    Pi_yy += fy * uy
    Pi_zz += fz * uz
    Pi_xy += (fx * uy + ux * fy) * T(0.5)
    Pi_xz += (fx * uz + ux * fz) * T(0.5)
    Pi_yz += (fy * uz + uy * fz) * T(0.5)
    tau_previous = one(T) / sp_field[i, j, k]
    gamma_dot = strain_rate_from_pi_3d(
        Pi_xx, Pi_yy, Pi_zz, Pi_xy, Pi_xz, Pi_yz, ρ, tau_previous)
    nu_local = effective_viscosity(rheology, gamma_dot)
    s_plus = one(T) / (T(3) * nu_local + T(0.5))
    s_minus = one(T) / (trt_lambda / (T(3) * nu_local) + T(0.5))
    sp_field[i, j, k] = s_plus
    sm_field[i, j, k] = s_minus

    a = (s_plus + s_minus) * T(0.5)
    b = (s_plus - s_minus) * T(0.5)

    wr = T(1.0/3.0); wa = T(1.0/18.0); we = T(1.0/36.0)
    t3 = T(3); t9 = T(9)
    # Guo source S_q = w_q·[3·(c_q-u)·F + 9·(c_q·u)(c_q·F)]  (D3Q19), split into
    # its even part S⁺_q = w_q·[9·(c_q·u)(c_q·F) - 3·u·F] (second moment uF + Fu)
    # and its odd part S⁻_q = 3·w_q·c_q·F (first moment F). Each part takes the
    # Crank–Nicolson factor of the rate that relaxes it:
    #   (1 - s_plus/2)·S⁺_q + (1 - s_minus/2)·S⁻_q
    # (TRT form of the MRT source (I - D/2)·M·S, D the diagonal matrix of
    # relaxation rates; Silva 2020, Comput. Fluids 203, 104537,
    # doi:10.1016/j.compfluid.2020.104537). With the F/2 equilibrium
    # shift above, the first moment gains exactly F per collision for any
    # (s_plus, s_minus); one (1 - s_plus/2) factor on the whole S_q injected
    # (1 + (s_minus - s_plus)/2)·F. Pairs (q, q̄) share S⁺ and flip S⁻.
    guo_even = one(T) - s_plus / T(2)
    guo_odd  = one(T) - s_minus / T(2)
    uF = ux*fx + uy*fy + uz*fz

    Sp1     = wr * (-uF) * t3
    Sp_2_3  = wa * (ux * fx * t9 - uF * t3);                Sm_2_3  = wa * fx * t3          # c2  = ( 1, 0, 0)
    Sp_4_5  = wa * (uy * fy * t9 - uF * t3);                Sm_4_5  = wa * fy * t3          # c4  = ( 0, 1, 0)
    Sp_6_7  = wa * (uz * fz * t9 - uF * t3);                Sm_6_7  = wa * fz * t3          # c6  = ( 0, 0, 1)
    Sp_8_11 = we * ((ux+uy) * (fx+fy) * t9 - uF * t3);      Sm_8_11 = we * (fx+fy) * t3     # c8  = ( 1, 1, 0)
    Sp_9_10 = we * ((-ux+uy) * (-fx+fy) * t9 - uF * t3);    Sm_9_10 = we * (-fx+fy) * t3    # c9  = (-1, 1, 0)
    Sp_12_15 = we * ((ux+uz) * (fx+fz) * t9 - uF * t3);     Sm_12_15 = we * (fx+fz) * t3    # c12 = ( 1, 0, 1)
    Sp_13_14 = we * ((-ux+uz) * (-fx+fz) * t9 - uF * t3);   Sm_13_14 = we * (-fx+fz) * t3   # c13 = (-1, 0, 1)
    Sp_16_19 = we * ((uy+uz) * (fy+fz) * t9 - uF * t3);     Sm_16_19 = we * (fy+fz) * t3    # c16 = ( 0, 1, 1)
    Sp_17_18 = we * ((-uy+uz) * (-fy+fz) * t9 - uF * t3);   Sm_17_18 = we * (-fy+fz) * t3   # c17 = ( 0,-1, 1)

    f_out[i, j, k, 1]  = fp1  - s_plus * (fp1 - feq1)                     + guo_even * Sp1
    f_out[i, j, k, 2]  = fp2  - a * (fp2  - feq2)  - b * (fp3  - feq3)  + guo_even * Sp_2_3   + guo_odd * Sm_2_3
    f_out[i, j, k, 3]  = fp3  - a * (fp3  - feq3)  - b * (fp2  - feq2)  + guo_even * Sp_2_3   - guo_odd * Sm_2_3
    f_out[i, j, k, 4]  = fp4  - a * (fp4  - feq4)  - b * (fp5  - feq5)  + guo_even * Sp_4_5   + guo_odd * Sm_4_5
    f_out[i, j, k, 5]  = fp5  - a * (fp5  - feq5)  - b * (fp4  - feq4)  + guo_even * Sp_4_5   - guo_odd * Sm_4_5
    f_out[i, j, k, 6]  = fp6  - a * (fp6  - feq6)  - b * (fp7  - feq7)  + guo_even * Sp_6_7   + guo_odd * Sm_6_7
    f_out[i, j, k, 7]  = fp7  - a * (fp7  - feq7)  - b * (fp6  - feq6)  + guo_even * Sp_6_7   - guo_odd * Sm_6_7
    f_out[i, j, k, 8]  = fp8  - a * (fp8  - feq8)  - b * (fp11 - feq11) + guo_even * Sp_8_11  + guo_odd * Sm_8_11
    f_out[i, j, k, 11] = fp11 - a * (fp11 - feq11) - b * (fp8  - feq8)  + guo_even * Sp_8_11  - guo_odd * Sm_8_11
    f_out[i, j, k, 9]  = fp9  - a * (fp9  - feq9)  - b * (fp10 - feq10) + guo_even * Sp_9_10  + guo_odd * Sm_9_10
    f_out[i, j, k, 10] = fp10 - a * (fp10 - feq10) - b * (fp9  - feq9)  + guo_even * Sp_9_10  - guo_odd * Sm_9_10
    f_out[i, j, k, 12] = fp12 - a * (fp12 - feq12) - b * (fp15 - feq15) + guo_even * Sp_12_15 + guo_odd * Sm_12_15
    f_out[i, j, k, 15] = fp15 - a * (fp15 - feq15) - b * (fp12 - feq12) + guo_even * Sp_12_15 - guo_odd * Sm_12_15
    f_out[i, j, k, 13] = fp13 - a * (fp13 - feq13) - b * (fp14 - feq14) + guo_even * Sp_13_14 + guo_odd * Sm_13_14
    f_out[i, j, k, 14] = fp14 - a * (fp14 - feq14) - b * (fp13 - feq13) + guo_even * Sp_13_14 - guo_odd * Sm_13_14
    f_out[i, j, k, 16] = fp16 - a * (fp16 - feq16) - b * (fp19 - feq19) + guo_even * Sp_16_19 + guo_odd * Sm_16_19
    f_out[i, j, k, 19] = fp19 - a * (fp19 - feq19) - b * (fp16 - feq16) + guo_even * Sp_16_19 - guo_odd * Sm_16_19
    f_out[i, j, k, 17] = fp17 - a * (fp17 - feq17) - b * (fp18 - feq18) + guo_even * Sp_17_18 + guo_odd * Sm_17_18
    f_out[i, j, k, 18] = fp18 - a * (fp18 - feq18) - b * (fp17 - feq17) + guo_even * Sp_17_18 - guo_odd * Sm_17_18
end
