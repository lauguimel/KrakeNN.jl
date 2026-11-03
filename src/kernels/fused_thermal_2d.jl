using KernelAbstractions

# --- Inline per-node building blocks for fused LBM step ---
#
# Each function operates on local data (no global array writes).
# Julia JIT inlines them into a single GPU kernel → 1 launch per step.

# --- Stream: pull 9 populations from neighbors with bounce-back ---

@inline function stream_pull_node(f_in, i, j, Nx, Ny)
    im = max(i - 1, 1); ip = min(i + 1, Nx)
    jm = max(j - 1, 1); jp = min(j + 1, Ny)

    fp1 = f_in[i, j, 1]
    fp2 = ifelse(i > 1,             f_in[im, j,  2], f_in[i, j, 4])
    fp3 = ifelse(j > 1,             f_in[i,  jm, 3], f_in[i, j, 5])
    fp4 = ifelse(i < Nx,            f_in[ip, j,  4], f_in[i, j, 2])
    fp5 = ifelse(j < Ny,            f_in[i,  jp, 5], f_in[i, j, 3])
    fp6 = ifelse(i > 1  && j > 1,   f_in[im, jm, 6], f_in[i, j, 8])
    fp7 = ifelse(i < Nx && j > 1,   f_in[ip, jm, 7], f_in[i, j, 9])
    fp8 = ifelse(i < Nx && j < Ny,  f_in[ip, jp, 8], f_in[i, j, 6])
    fp9 = ifelse(i > 1  && j < Ny,  f_in[im, jp, 9], f_in[i, j, 7])

    return (fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9)
end

# --- Thermal BCs: anti-bounce-back for Dirichlet (west/east), bounce-back elsewhere ---
#
# Half-way anti-bounce-back: the Dirichlet walls sit at x = 0.5 and x = Nx + 0.5,
# the same place as the half-way no-slip walls of the flow, so the thermal and
# fluid cavities share the width H = Nx. `stream_pull_node` has already put
# the bounced post-collision value g*_opp(x_b) into each wall-crossing slot;
# anti-bounce-back flips its sign and adds the wall term:
#   g_q(x_b, t+1) = -g*_opp(x_b, t) + 2 w_q T_wall.
# Diagonals crossing a corner take the west/east Dirichlet rule.

@inline function apply_thermal_bc_cavity(gp, i, j, Nx, Ny, T_hot, T_cold)
    T = typeof(T_hot)
    g1, g2, g3, g4, g5, g6, g7, g8, g9 = gp

    # West wall (x = 0.5): fixed T_hot — unknown populations pointing east: 2, 6, 9
    if i == 1
        g2 = -g2 + T(2) * T(1.0/9.0)  * T_hot
        g6 = -g6 + T(2) * T(1.0/36.0) * T_hot
        g9 = -g9 + T(2) * T(1.0/36.0) * T_hot
    end

    # East wall (x = Nx + 0.5): fixed T_cold — unknown populations pointing west: 4, 7, 8
    if i == Nx
        g4 = -g4 + T(2) * T(1.0/9.0)  * T_cold
        g7 = -g7 + T(2) * T(1.0/36.0) * T_cold
        g8 = -g8 + T(2) * T(1.0/36.0) * T_cold
    end

    # Top/bottom: adiabatic (bounce-back already handled by stream_pull_node)
    return (g1, g2, g3, g4, g5, g6, g7, g8, g9)
end

# --- Macroscopic: recover ρ, ux, uy from flow populations with Guo force correction ---

@inline function macroscopic_boussinesq(fp, T_local, β_g, T_ref_buoy)
    f1, f2, f3, f4, f5, f6, f7, f8, f9 = fp
    T = typeof(T_local)

    fy = β_g * (T_local - T_ref_buoy)
    ρ = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
    inv_ρ = one(T) / ρ
    ux = (f2 - f4 + f6 - f7 - f8 + f9) * inv_ρ
    uy = ((f3 - f5 + f6 + f7 - f8 - f9) + fy / T(2)) * inv_ρ

    return ρ, ux, uy, fy
end

# --- Collide thermal: BGK on g populations ---

@inline function collide_thermal_node(gp, ux, uy, ω_T)
    g1, g2, g3, g4, g5, g6, g7, g8, g9 = gp
    T = typeof(ω_T)

    Temp = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9
    t3 = T(3)

    geq = T(4.0/9.0) * Temp
    g1 = g1 - ω_T * (g1 - geq)

    geq = T(1.0/9.0) * Temp * (one(T) + t3 * ux)
    g2 = g2 - ω_T * (g2 - geq)

    geq = T(1.0/9.0) * Temp * (one(T) + t3 * uy)
    g3 = g3 - ω_T * (g3 - geq)

    geq = T(1.0/9.0) * Temp * (one(T) - t3 * ux)
    g4 = g4 - ω_T * (g4 - geq)

    geq = T(1.0/9.0) * Temp * (one(T) - t3 * uy)
    g5 = g5 - ω_T * (g5 - geq)

    geq = T(1.0/36.0) * Temp * (one(T) + t3 * (ux + uy))
    g6 = g6 - ω_T * (g6 - geq)

    geq = T(1.0/36.0) * Temp * (one(T) + t3 * (-ux + uy))
    g7 = g7 - ω_T * (g7 - geq)

    geq = T(1.0/36.0) * Temp * (one(T) + t3 * (-ux - uy))
    g8 = g8 - ω_T * (g8 - geq)

    geq = T(1.0/36.0) * Temp * (one(T) + t3 * (ux - uy))
    g9 = g9 - ω_T * (g9 - geq)

    return (g1, g2, g3, g4, g5, g6, g7, g8, g9), Temp
end

# --- Collide flow: BGK + Guo Boussinesq forcing ---

@inline function collide_boussinesq_node(fp, ρ, ux, uy, fy, ω)
    f1, f2, f3, f4, f5, f6, f7, f8, f9 = fp
    T = typeof(ω)
    fx = zero(T)
    usq = ux * ux + uy * uy
    guo_pref = one(T) - ω / T(2)
    t3 = T(3)

    Sq = T(4.0/9.0)*((-ux)*fx+(-uy)*fy)*t3
    f1 = f1 - ω*(f1-feq_2d(Val(1), ρ, ux, uy, usq)) + guo_pref*Sq

    Sq=T(1.0/9.0)*((one(T)-ux)*fx+(-uy)*fy)*t3+T(1.0/9.0)*ux*fx*T(9)
    f2 = f2 - ω*(f2-feq_2d(Val(2), ρ, ux, uy, usq)) + guo_pref*Sq

    Sq=T(1.0/9.0)*((-ux)*fx+(one(T)-uy)*fy)*t3+T(1.0/9.0)*uy*fy*T(9)
    f3 = f3 - ω*(f3-feq_2d(Val(3), ρ, ux, uy, usq)) + guo_pref*Sq

    Sq=T(1.0/9.0)*((-one(T)-ux)*fx+(-uy)*fy)*t3+T(1.0/9.0)*ux*fx*T(9)
    f4 = f4 - ω*(f4-feq_2d(Val(4), ρ, ux, uy, usq)) + guo_pref*Sq

    Sq=T(1.0/9.0)*((-ux)*fx+(-one(T)-uy)*fy)*t3+T(1.0/9.0)*uy*fy*T(9)
    f5 = f5 - ω*(f5-feq_2d(Val(5), ρ, ux, uy, usq)) + guo_pref*Sq

    Sq=T(1.0/36.0)*((one(T)-ux)*fx+(one(T)-uy)*fy)*t3+T(1.0/36.0)*(ux+uy)*(fx+fy)*T(9)
    f6 = f6 - ω*(f6-feq_2d(Val(6), ρ, ux, uy, usq)) + guo_pref*Sq

    Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(one(T)-uy)*fy)*t3+T(1.0/36.0)*(-ux+uy)*(-fx+fy)*T(9)
    f7 = f7 - ω*(f7-feq_2d(Val(7), ρ, ux, uy, usq)) + guo_pref*Sq

    Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(-one(T)-uy)*fy)*t3+T(1.0/36.0)*(-ux-uy)*(-fx-fy)*T(9)
    f8 = f8 - ω*(f8-feq_2d(Val(8), ρ, ux, uy, usq)) + guo_pref*Sq

    Sq=T(1.0/36.0)*((one(T)-ux)*fx+(-one(T)-uy)*fy)*t3+T(1.0/36.0)*(ux-uy)*(fx-fy)*T(9)
    f9 = f9 - ω*(f9-feq_2d(Val(9), ρ, ux, uy, usq)) + guo_pref*Sq

    return (f1, f2, f3, f4, f5, f6, f7, f8, f9)
end

# --- Variable viscosity variant ---

@inline function collide_boussinesq_vt_node(fp, ρ, ux, uy, fy, T_local,
                                             ν_ref, T0_visc, α_visc)
    T = typeof(ν_ref)
    ν_local = ν_ref * exp(α_visc * (T_local - T0_visc))
    ω = one(T) / (T(3) * ν_local + T(0.5))
    return collide_boussinesq_node(fp, ρ, ux, uy, fy, ω)
end

# --- Write 9 populations to output array ---

@inline function write_populations!(f_out, i, j, fp)
    f_out[i,j,1] = fp[1]
    f_out[i,j,2] = fp[2]; f_out[i,j,3] = fp[3]
    f_out[i,j,4] = fp[4]; f_out[i,j,5] = fp[5]
    f_out[i,j,6] = fp[6]; f_out[i,j,7] = fp[7]
    f_out[i,j,8] = fp[8]; f_out[i,j,9] = fp[9]
    return nothing
end

# =====================================================================
# Fused kernel: 1 launch per timestep (constant viscosity)
# =====================================================================

@kernel function fused_natconv_step_kernel!(f_out, @Const(f_in), g_out, @Const(g_in),
                                             Temp, Nx, Ny,
                                             ω_f, ω_T, β_g, T_ref_buoy,
                                             T_hot, T_cold)
    i, j = @index(Global, NTuple)

    @inbounds begin
        # 1. Stream both populations
        fp = stream_pull_node(f_in, i, j, Nx, Ny)
        gp = stream_pull_node(g_in, i, j, Nx, Ny)

        # 2. Thermal BCs (Dirichlet west/east, adiabatic top/bottom)
        gp = apply_thermal_bc_cavity(gp, i, j, Nx, Ny, T_hot, T_cold)

        # 3. Macroscopic temperature
        T_local = gp[1]+gp[2]+gp[3]+gp[4]+gp[5]+gp[6]+gp[7]+gp[8]+gp[9]
        Temp[i,j] = T_local

        # 4. Macroscopic flow (with Boussinesq force correction)
        ρ, ux, uy, fy = macroscopic_boussinesq(fp, T_local, β_g, T_ref_buoy)

        # 5. Collide thermal
        gp_out, _ = collide_thermal_node(gp, ux, uy, ω_T)
        write_populations!(g_out, i, j, gp_out)

        # 6. Collide flow
        fp_out = collide_boussinesq_node(fp, ρ, ux, uy, fy, ω_f)
        write_populations!(f_out, i, j, fp_out)
    end
end

# =====================================================================
# Fused kernel: 1 launch per timestep (variable viscosity)
# =====================================================================

@kernel function fused_natconv_vt_step_kernel!(f_out, @Const(f_in), g_out, @Const(g_in),
                                                Temp, Nx, Ny,
                                                ν_ref, T0_visc, α_visc,
                                                ω_T, β_g, T_ref_buoy,
                                                T_hot, T_cold)
    i, j = @index(Global, NTuple)

    @inbounds begin
        fp = stream_pull_node(f_in, i, j, Nx, Ny)
        gp = stream_pull_node(g_in, i, j, Nx, Ny)

        gp = apply_thermal_bc_cavity(gp, i, j, Nx, Ny, T_hot, T_cold)

        T_local = gp[1]+gp[2]+gp[3]+gp[4]+gp[5]+gp[6]+gp[7]+gp[8]+gp[9]
        Temp[i,j] = T_local

        ρ, ux, uy, fy = macroscopic_boussinesq(fp, T_local, β_g, T_ref_buoy)

        gp_out, _ = collide_thermal_node(gp, ux, uy, ω_T)
        write_populations!(g_out, i, j, gp_out)

        fp_out = collide_boussinesq_vt_node(fp, ρ, ux, uy, fy, T_local,
                                             ν_ref, T0_visc, α_visc)
        write_populations!(f_out, i, j, fp_out)
    end
end

# =====================================================================
# Public API
# =====================================================================

"""
    fused_natconv_step!(f_out, f_in, g_out, g_in, Temp, Nx, Ny,
                        ω_f, ω_T, β_g, T_ref_buoy, T_hot, T_cold)

Single fused kernel for natural convection with constant viscosity.
Performs stream + BC + macroscopic + collide for both f and g in one launch.
"""
function fused_natconv_step!(f_out, f_in, g_out, g_in, Temp, Nx, Ny,
                              ω_f, ω_T, β_g, T_ref_buoy, T_hot, T_cold)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    kernel! = fused_natconv_step_kernel!(backend)
    kernel!(f_out, f_in, g_out, g_in, Temp, Nx, Ny,
            ET(ω_f), ET(ω_T), ET(β_g), ET(T_ref_buoy),
            ET(T_hot), ET(T_cold); ndrange=(Nx, Ny))
end

"""
    fused_natconv_vt_step!(f_out, f_in, g_out, g_in, Temp, Nx, Ny,
                            ν_ref, T0_visc, α_visc, ω_T, β_g, T_ref_buoy,
                            T_hot, T_cold)

Single fused kernel for natural convection with variable viscosity.
"""
function fused_natconv_vt_step!(f_out, f_in, g_out, g_in, Temp, Nx, Ny,
                                 ν_ref, T0_visc, α_visc, ω_T, β_g, T_ref_buoy,
                                 T_hot, T_cold)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    kernel! = fused_natconv_vt_step_kernel!(backend)
    kernel!(f_out, f_in, g_out, g_in, Temp, Nx, Ny,
            ET(ν_ref), ET(T0_visc), ET(α_visc),
            ET(ω_T), ET(β_g), ET(T_ref_buoy),
            ET(T_hot), ET(T_cold); ndrange=(Nx, Ny))
end
