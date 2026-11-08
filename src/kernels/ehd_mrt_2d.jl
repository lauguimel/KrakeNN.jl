using KernelAbstractions

# EHD-local D2Q9 MRT Navier-Stokes collision with spatial Guo forcing.
# Moment order follows Jiachen's Lattice.m setMRT and KrakeNN's collide_mrt_2d:
# rho, e, eps, jx, qx, jy, qy, pxx, pxy.

@kernel function ehd_collide_mrt_2d_kernel!(f, @Const(Fx), @Const(Fy), @Const(is_solid),
                                            s_e, s_eps, s_q, s_nu)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            bounce_back_2d!(f, i, j)
        else
            T = eltype(f)
            f1 = f[i,j,1]; f2 = f[i,j,2]; f3 = f[i,j,3]
            f4 = f[i,j,4]; f5 = f[i,j,5]; f6 = f[i,j,6]
            f7 = f[i,j,7]; f8 = f[i,j,8]; f9 = f[i,j,9]
            fx = Fx[i, j]
            fy = Fy[i, j]

            rho = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
            e   = -T(4)*f1 - f2 - f3 - f4 - f5 + T(2)*(f6 + f7 + f8 + f9)
            eps =  T(4)*f1 - T(2)*(f2 + f3 + f4 + f5) + f6 + f7 + f8 + f9
            jx  = f2 - f4 + f6 - f7 - f8 + f9
            qx  = -T(2)*f2 + T(2)*f4 + f6 - f7 - f8 + f9
            jy  = f3 - f5 + f6 + f7 - f8 - f9
            qy  = -T(2)*f3 + T(2)*f5 + f6 + f7 - f8 - f9
            pxx = f2 - f3 + f4 - f5
            pxy = f6 - f7 + f8 - f9

            invrho = one(T) / rho
            ux = (jx + fx / T(2)) * invrho
            uy = (jy + fy / T(2)) * invrho
            usq = ux * ux + uy * uy

            e_eq   = -T(2) * rho + T(3) * rho * usq
            eps_eq = rho - T(3) * rho * usq
            qx_eq  = -rho * ux
            qy_eq  = -rho * uy
            pxx_eq = rho * (ux * ux - uy * uy)
            pxy_eq = rho * ux * uy

            w1 = T(4) / T(9)
            w2 = T(1) / T(9)
            w6 = T(1) / T(36)

            g1 = w1 * T(3) * ((-ux) * fx + (-uy) * fy)
            g2 = w2 * (T(3) * ((one(T) - ux) * fx + (-uy) * fy) + T(9) * ux * fx)
            g3 = w2 * (T(3) * ((-ux) * fx + (one(T) - uy) * fy) + T(9) * uy * fy)
            g4 = w2 * (T(3) * ((-one(T) - ux) * fx + (-uy) * fy) + T(9) * ux * fx)
            g5 = w2 * (T(3) * ((-ux) * fx + (-one(T) - uy) * fy) + T(9) * uy * fy)
            g6 = w6 * (T(3) * ((one(T) - ux) * fx + (one(T) - uy) * fy) +
                       T(9) * (ux + uy) * (fx + fy))
            g7 = w6 * (T(3) * ((-one(T) - ux) * fx + (one(T) - uy) * fy) +
                       T(9) * (-ux + uy) * (-fx + fy))
            g8 = w6 * (T(3) * ((-one(T) - ux) * fx + (-one(T) - uy) * fy) +
                       T(9) * (-ux - uy) * (-fx - fy))
            g9 = w6 * (T(3) * ((one(T) - ux) * fx + (-one(T) - uy) * fy) +
                       T(9) * (ux - uy) * (fx - fy))

            ge   = -T(4)*g1 - g2 - g3 - g4 - g5 + T(2)*(g6 + g7 + g8 + g9)
            geps =  T(4)*g1 - T(2)*(g2 + g3 + g4 + g5) + g6 + g7 + g8 + g9
            gjx  = g2 - g4 + g6 - g7 - g8 + g9
            gqx  = -T(2)*g2 + T(2)*g4 + g6 - g7 - g8 + g9
            gjy  = g3 - g5 + g6 + g7 - g8 - g9
            gqy  = -T(2)*g3 + T(2)*g5 + g6 + g7 - g8 - g9
            gpxx = g2 - g3 + g4 - g5
            gpxy = g6 - g7 + g8 - g9

            e_star   = e   - s_e   * (e   - e_eq)   + (one(T) - s_e   / T(2)) * ge
            eps_star = eps - s_eps * (eps - eps_eq) + (one(T) - s_eps / T(2)) * geps
            # Momentum moments: j_eq = j + F/2 (half-force velocity shift), so the
            # relaxation term -s*(j - j_eq) contributes +s*F/2 and the source term
            # (1 - s/2)*F completes to the full Guo momentum input j* = j + F,
            # independent of the momentum relaxation rate.
            jx_star  = jx + gjx
            qx_star  = qx  - s_q   * (qx  - qx_eq)  + (one(T) - s_q   / T(2)) * gqx
            jy_star  = jy + gjy
            qy_star  = qy  - s_q   * (qy  - qy_eq)  + (one(T) - s_q   / T(2)) * gqy
            pxx_star = pxx - s_nu  * (pxx - pxx_eq) + (one(T) - s_nu  / T(2)) * gpxx
            pxy_star = pxy - s_nu  * (pxy - pxy_eq) + (one(T) - s_nu  / T(2)) * gpxy

            r = rho; es = e_star; ep = eps_star
            jxs = jx_star; qxs = qx_star; jys = jy_star; qys = qy_star
            ps = pxx_star; pxys = pxy_star

            f[i,j,1] = T(1/9)*r - T(1/9)*es + T(1/9)*ep
            f[i,j,2] = T(1/9)*r - T(1/36)*es - T(1/18)*ep + T(1/6)*jxs - T(1/6)*qxs + T(1/4)*ps
            f[i,j,3] = T(1/9)*r - T(1/36)*es - T(1/18)*ep + T(1/6)*jys - T(1/6)*qys - T(1/4)*ps
            f[i,j,4] = T(1/9)*r - T(1/36)*es - T(1/18)*ep - T(1/6)*jxs + T(1/6)*qxs + T(1/4)*ps
            f[i,j,5] = T(1/9)*r - T(1/36)*es - T(1/18)*ep - T(1/6)*jys + T(1/6)*qys - T(1/4)*ps
            f[i,j,6] = T(1/9)*r + T(1/18)*es + T(1/36)*ep + T(1/6)*jxs + T(1/12)*qxs + T(1/6)*jys + T(1/12)*qys + T(1/4)*pxys
            f[i,j,7] = T(1/9)*r + T(1/18)*es + T(1/36)*ep - T(1/6)*jxs - T(1/12)*qxs + T(1/6)*jys + T(1/12)*qys - T(1/4)*pxys
            f[i,j,8] = T(1/9)*r + T(1/18)*es + T(1/36)*ep - T(1/6)*jxs - T(1/12)*qxs - T(1/6)*jys - T(1/12)*qys + T(1/4)*pxys
            f[i,j,9] = T(1/9)*r + T(1/18)*es + T(1/36)*ep + T(1/6)*jxs + T(1/12)*qxs - T(1/6)*jys - T(1/12)*qys - T(1/4)*pxys
        end
    end
end

"""
    ehd_collide_mrt_2d!(f, Fx, Fy, is_solid, ν; s_e=1.64, s_eps=1.54, s_q=1.0)

EHD-local D2Q9 MRT Navier-Stokes collision with spatial Guo forcing. The
relaxation vector follows Jiachen's `setMRT`: conserved rates `s_c=1`,
`s_e=1.64`, `s_eps=1.54`, `s_q=1`, and `s_nu=1/(3ν+0.5)`.
"""
function ehd_collide_mrt_2d!(f, Fx, Fy, is_solid, ν; s_e=1.64, s_eps=1.54, s_q=1.0)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    T = eltype(f)
    nu_t = T(ν)
    s_nu = one(T) / (T(3) * nu_t + T(0.5))
    kernel! = ehd_collide_mrt_2d_kernel!(backend)
    kernel!(f, Fx, Fy, is_solid, T(s_e), T(s_eps), T(s_q), s_nu; ndrange=(Nx, Ny))
end
