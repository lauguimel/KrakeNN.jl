using KernelAbstractions

# --- Thermal D2Q9: Double Distribution Function for temperature ---
#
# Temperature field T is tracked by a separate population g[i,j,q] on D2Q9.
# Equilibrium: g_eq_q = w_q * T * (1 + c_q·u / cs²)
# Collision:   g_q = g_q - ω_T * (g_q - g_eq_q)
# where ω_T = 1 / (3α + 0.5), α = thermal diffusivity
# Macroscopic: T = Σ g_q

# --- Stream thermal populations (reuses same stream kernels as flow) ---
# For walls: adiabatic (bounce-back) or fixed T (anti-bounce-back)

# --- Collide thermal (BGK on temperature populations) ---

@kernel function collide_thermal_2d_kernel!(g, @Const(ux), @Const(uy), ω_T)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(g)
        g1=g[i,j,1]; g2=g[i,j,2]; g3=g[i,j,3]; g4=g[i,j,4]
        g5=g[i,j,5]; g6=g[i,j,6]; g7=g[i,j,7]; g8=g[i,j,8]; g9=g[i,j,9]

        # Macroscopic temperature
        Temp = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9

        # Local velocity (from flow solver)
        u_x = ux[i,j]
        u_y = uy[i,j]

        # Thermal equilibrium: g_eq = w_q * T * (1 + 3 * c_q · u)
        # (simplified: no u² terms for passive scalar transport)
        t3 = T(3)

        geq = T(4.0/9.0) * Temp
        g[i,j,1] = g1 - ω_T * (g1 - geq)

        geq = T(1.0/9.0) * Temp * (one(T) + t3 * u_x)
        g[i,j,2] = g2 - ω_T * (g2 - geq)

        geq = T(1.0/9.0) * Temp * (one(T) + t3 * u_y)
        g[i,j,3] = g3 - ω_T * (g3 - geq)

        geq = T(1.0/9.0) * Temp * (one(T) - t3 * u_x)
        g[i,j,4] = g4 - ω_T * (g4 - geq)

        geq = T(1.0/9.0) * Temp * (one(T) - t3 * u_y)
        g[i,j,5] = g5 - ω_T * (g5 - geq)

        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (u_x + u_y))
        g[i,j,6] = g6 - ω_T * (g6 - geq)

        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (-u_x + u_y))
        g[i,j,7] = g7 - ω_T * (g7 - geq)

        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (-u_x - u_y))
        g[i,j,8] = g8 - ω_T * (g8 - geq)

        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (u_x - u_y))
        g[i,j,9] = g9 - ω_T * (g9 - geq)
    end
end

"""
    collide_thermal_2d!(g, ux, uy, ω_T)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using Kraken

methods(Kraken.collide_thermal_2d!)
```
"""
function collide_thermal_2d!(g, ux, uy, ω_T)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(g, 1), size(g, 2)
    kernel! = collide_thermal_2d_kernel!(backend)
    kernel!(g, ux, uy, ω_T; ndrange=(Nx, Ny))
end

@kernel function collide_thermal_masked_2d_kernel!(g, @Const(ux), @Const(uy), ω_T, @Const(is_skip))
    i, j = @index(Global, NTuple)
    @inbounds if !is_skip[i, j]
        T = eltype(g)
        u_x = T(ux[i, j]); u_y = T(uy[i, j])
        Temp = g[i,j,1] + g[i,j,2] + g[i,j,3] + g[i,j,4] +
               g[i,j,5] + g[i,j,6] + g[i,j,7] + g[i,j,8] + g[i,j,9]
        t3 = T(3)

        for q in 1:9
            w_q = ifelse(q == 1, T(4)/T(9), ifelse(q <= 5, T(1)/T(9), T(1)/T(36)))
            cx = ifelse(q==1,T(0), ifelse(q==2,T(1), ifelse(q==3,T(0), ifelse(q==4,T(-1),
                 ifelse(q==5,T(0), ifelse(q==6,T(1), ifelse(q==7,T(-1), ifelse(q==8,T(-1), T(1)))))))))
            cy = ifelse(q==1,T(0), ifelse(q==2,T(0), ifelse(q==3,T(1), ifelse(q==4,T(0),
                 ifelse(q==5,T(-1), ifelse(q==6,T(1), ifelse(q==7,T(1), ifelse(q==8,T(-1), T(-1)))))))))
            geq = w_q * Temp * (one(T) + t3 * (cx * u_x + cy * u_y))
            g[i, j, q] -= ω_T * (g[i, j, q] - geq)
        end
    end
end

function collide_thermal_masked_2d!(g, ux, uy, ω_T, is_skip)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(g, 1), size(g, 2)
    kernel! = collide_thermal_masked_2d_kernel!(backend)
    kernel!(g, ux, uy, ω_T, is_skip; ndrange=(Nx, Ny))
end

# --- Compute temperature from thermal populations ---

@kernel function compute_temperature_2d_kernel!(Temp, @Const(g))
    i, j = @index(Global, NTuple)
    @inbounds begin
        Temp[i,j] = g[i,j,1] + g[i,j,2] + g[i,j,3] + g[i,j,4] +
                    g[i,j,5] + g[i,j,6] + g[i,j,7] + g[i,j,8] + g[i,j,9]
    end
end

"""
    compute_temperature_2d!(Temp, g)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using Kraken

methods(Kraken.compute_temperature_2d!)
```
"""
function compute_temperature_2d!(Temp, g)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(Temp)
    kernel! = compute_temperature_2d_kernel!(backend)
    kernel!(Temp, g; ndrange=(Nx, Ny))
end

# --- Fixed temperature BC (half-way anti-bounce-back for Dirichlet) ---
#
# The wall sits half a lattice spacing outside the boundary node (y = 0.5 for
# the south face, y = Ny + 0.5 for the north face, x = 0.5 and x = Nx + 0.5 for
# west/east), where the half-way no-slip walls of the flow sit. These kernels
# act on the post-stream array of a bounce-back stream kernel (`stream_2d!`,
# `stream_periodic_x_wall_y_2d!`): each wall-crossing slot already holds the
# bounced post-collision value g*_opp(x_b, t), and anti-bounce-back sets
#   g_q(x_b, t+1) = -g*_opp(x_b, t) + 2 w_q T_wall,
# i.e. a sign flip of that slot plus the wall term. The update is not
# idempotent: call it exactly once per time step, after streaming and before
# collision, and do not call two Dirichlet faces that share a corner (the
# diagonal slot at the corner node would be flipped twice).

@kernel function apply_fixed_temp_south_2d_kernel!(g, T_wall)
    i = @index(Global)
    j = 1
    @inbounds begin
        TT = eltype(g)
        # South wall (y = 0.5), unknown populations pointing north: 3, 6, 7
        g[i,j,3] = -g[i,j,3] + TT(2) * TT(1.0/9.0)  * T_wall
        g[i,j,6] = -g[i,j,6] + TT(2) * TT(1.0/36.0) * T_wall
        g[i,j,7] = -g[i,j,7] + TT(2) * TT(1.0/36.0) * T_wall
    end
end

@kernel function apply_fixed_temp_north_2d_kernel!(g, T_wall, Ny)
    i = @index(Global)
    j = Ny
    @inbounds begin
        TT = eltype(g)
        # North wall (y = Ny + 0.5), unknown populations pointing south: 5, 8, 9
        g[i,j,5] = -g[i,j,5] + TT(2) * TT(1.0/9.0)  * T_wall
        g[i,j,8] = -g[i,j,8] + TT(2) * TT(1.0/36.0) * T_wall
        g[i,j,9] = -g[i,j,9] + TT(2) * TT(1.0/36.0) * T_wall
    end
end

"""
    apply_fixed_temp_south_2d!(g, T_wall, Nx)

Fixed temperature `T_wall` on the south face by half-way anti-bounce-back: the
wall sits at y = 0.5, half a lattice spacing outside the boundary nodes, where
the half-way no-slip wall of the flow sits. `g` must be the post-stream array of
a bounce-back stream kernel; call once per step, after streaming and before
collision (the update flips the sign of the bounced slots, so a second call
undoes the first).

```julia
using Kraken

methods(Kraken.apply_fixed_temp_south_2d!)
```
"""
function apply_fixed_temp_south_2d!(g, T_wall, Nx)
    backend = KernelAbstractions.get_backend(g)
    kernel! = apply_fixed_temp_south_2d_kernel!(backend)
    kernel!(g, eltype(g)(T_wall); ndrange=(Nx,))
end

"""
    apply_fixed_temp_north_2d!(g, T_wall, Nx, Ny)

Fixed temperature `T_wall` on the north face by half-way anti-bounce-back: the
wall sits at y = Ny + 0.5, half a lattice spacing outside the boundary nodes, where
the half-way no-slip wall of the flow sits. `g` must be the post-stream array of
a bounce-back stream kernel; call once per step, after streaming and before
collision (the update flips the sign of the bounced slots, so a second call
undoes the first).

```julia
using Kraken

methods(Kraken.apply_fixed_temp_north_2d!)
```
"""
function apply_fixed_temp_north_2d!(g, T_wall, Nx, Ny)
    backend = KernelAbstractions.get_backend(g)
    kernel! = apply_fixed_temp_north_2d_kernel!(backend)
    kernel!(g, eltype(g)(T_wall), Ny; ndrange=(Nx,))
end

# --- Fixed temperature BC for west/east walls (natural convection cavity) ---

@kernel function apply_fixed_temp_west_2d_kernel!(g, T_wall)
    j = @index(Global)
    i = 1
    @inbounds begin
        TT = eltype(g)
        # West wall (x = 0.5), unknown populations pointing east:
        # 2 (+1,0), 6 (+1,+1), 9 (+1,-1)
        g[i,j,2] = -g[i,j,2] + TT(2) * TT(1.0/9.0)  * T_wall
        g[i,j,6] = -g[i,j,6] + TT(2) * TT(1.0/36.0) * T_wall
        g[i,j,9] = -g[i,j,9] + TT(2) * TT(1.0/36.0) * T_wall
    end
end

@kernel function apply_fixed_temp_east_2d_kernel!(g, T_wall, Nx)
    j = @index(Global)
    i = Nx
    @inbounds begin
        TT = eltype(g)
        # East wall (x = Nx + 0.5), unknown populations pointing west:
        # 4 (-1,0), 7 (-1,+1), 8 (-1,-1)
        g[i,j,4] = -g[i,j,4] + TT(2) * TT(1.0/9.0)  * T_wall
        g[i,j,7] = -g[i,j,7] + TT(2) * TT(1.0/36.0) * T_wall
        g[i,j,8] = -g[i,j,8] + TT(2) * TT(1.0/36.0) * T_wall
    end
end

"""
    apply_fixed_temp_west_2d!(g, T_wall, Ny)

Fixed temperature `T_wall` on the west face by half-way anti-bounce-back: the
wall sits at x = 0.5, half a lattice spacing outside the boundary nodes, where
the half-way no-slip wall of the flow sits. `g` must be the post-stream array of
a bounce-back stream kernel; call once per step, after streaming and before
collision (the update flips the sign of the bounced slots, so a second call
undoes the first).

```julia
using Kraken

methods(Kraken.apply_fixed_temp_west_2d!)
```
"""
function apply_fixed_temp_west_2d!(g, T_wall, Ny)
    backend = KernelAbstractions.get_backend(g)
    kernel! = apply_fixed_temp_west_2d_kernel!(backend)
    kernel!(g, eltype(g)(T_wall); ndrange=(Ny,))
end

"""
    apply_fixed_temp_east_2d!(g, T_wall, Nx, Ny)

Fixed temperature `T_wall` on the east face by half-way anti-bounce-back: the
wall sits at x = Nx + 0.5, half a lattice spacing outside the boundary nodes, where
the half-way no-slip wall of the flow sits. `g` must be the post-stream array of
a bounce-back stream kernel; call once per step, after streaming and before
collision (the update flips the sign of the bounced slots, so a second call
undoes the first).

```julia
using Kraken

methods(Kraken.apply_fixed_temp_east_2d!)
```
"""
function apply_fixed_temp_east_2d!(g, T_wall, Nx, Ny)
    backend = KernelAbstractions.get_backend(g)
    kernel! = apply_fixed_temp_east_2d_kernel!(backend)
    kernel!(g, eltype(g)(T_wall), Nx; ndrange=(Ny,))
end

# --- BGK collision with per-node Boussinesq body force (Guo scheme) ---

@kernel function collide_boussinesq_2d_kernel!(f, @Const(Temp), @Const(is_solid),
                                                ω, β_g, T_ref)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            bounce_back_2d!(f, i, j)
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            # Per-node buoyancy force: Fy = β·g·(T - T_ref)
            fy = β_g * (Temp[i,j] - T_ref)
            fx = zero(T)

            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ
            ux = ((f2-f4+f6-f7-f8+f9) + fx/T(2)) * inv_ρ
            uy = ((f3-f5+f6+f7-f8-f9) + fy/T(2)) * inv_ρ
            usq = ux*ux + uy*uy

            guo_pref = one(T) - ω / T(2)

            Sq = T(4.0/9.0) * ((-ux)*fx + (-uy)*fy)*T(3)
            f[i,j,1] = f1 - ω*(f1-feq_2d(Val(1), ρ, ux, uy, usq)) + guo_pref*Sq

            Sq = T(1.0/9.0)*((one(T)-ux)*fx+(-uy)*fy)*T(3) + T(1.0/9.0)*ux*fx*T(9)
            f[i,j,2] = f2 - ω*(f2-feq_2d(Val(2), ρ, ux, uy, usq)) + guo_pref*Sq

            Sq = T(1.0/9.0)*((-ux)*fx+(one(T)-uy)*fy)*T(3) + T(1.0/9.0)*uy*fy*T(9)
            f[i,j,3] = f3 - ω*(f3-feq_2d(Val(3), ρ, ux, uy, usq)) + guo_pref*Sq

            Sq = T(1.0/9.0)*((-one(T)-ux)*fx+(-uy)*fy)*T(3) + T(1.0/9.0)*ux*fx*T(9)
            f[i,j,4] = f4 - ω*(f4-feq_2d(Val(4), ρ, ux, uy, usq)) + guo_pref*Sq

            Sq = T(1.0/9.0)*((-ux)*fx+(-one(T)-uy)*fy)*T(3) + T(1.0/9.0)*uy*fy*T(9)
            f[i,j,5] = f5 - ω*(f5-feq_2d(Val(5), ρ, ux, uy, usq)) + guo_pref*Sq

            Sq = T(1.0/36.0)*((one(T)-ux)*fx+(one(T)-uy)*fy)*T(3) + T(1.0/36.0)*(ux+uy)*(fx+fy)*T(9)
            f[i,j,6] = f6 - ω*(f6-feq_2d(Val(6), ρ, ux, uy, usq)) + guo_pref*Sq

            Sq = T(1.0/36.0)*((-one(T)-ux)*fx+(one(T)-uy)*fy)*T(3) + T(1.0/36.0)*(-ux+uy)*(-fx+fy)*T(9)
            f[i,j,7] = f7 - ω*(f7-feq_2d(Val(7), ρ, ux, uy, usq)) + guo_pref*Sq

            Sq = T(1.0/36.0)*((-one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3) + T(1.0/36.0)*(-ux-uy)*(-fx-fy)*T(9)
            f[i,j,8] = f8 - ω*(f8-feq_2d(Val(8), ρ, ux, uy, usq)) + guo_pref*Sq

            Sq = T(1.0/36.0)*((one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3) + T(1.0/36.0)*(ux-uy)*(fx-fy)*T(9)
            f[i,j,9] = f9 - ω*(f9-feq_2d(Val(9), ρ, ux, uy, usq)) + guo_pref*Sq
        end
    end
end

"""
    collide_boussinesq_vt_2d!(f, Temp, is_solid, ν_ref, T_ref, A_arr, β_g)

BGK collision with per-node Boussinesq force AND temperature-dependent viscosity.
ν(T) = ν_ref · exp(A_arr · (1/T - 1/T_ref))  (Arrhenius law)
ω(T) = 1 / (3·ν(T) + 0.5)
"""
@kernel function collide_boussinesq_vt_2d_kernel!(f, @Const(Temp), @Const(is_solid),
                                                    ν_ref, T_ref, A_arr, β_g)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            bounce_back_2d!(f, i, j)
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            # Temperature-dependent viscosity (Arrhenius)
            T_local = Temp[i,j]
            ν_local = ν_ref * exp(A_arr * (one(T)/T_local - one(T)/T_ref))
            ω_local = one(T) / (T(3) * ν_local + T(0.5))

            # Per-node buoyancy
            fy = β_g * (T_local - T_ref)
            fx = zero(T)

            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ
            ux = ((f2-f4+f6-f7-f8+f9) + fx/T(2)) * inv_ρ
            uy = ((f3-f5+f6+f7-f8-f9) + fy/T(2)) * inv_ρ
            usq = ux*ux + uy*uy

            guo_pref = one(T) - ω_local / T(2)

            Sq=T(4.0/9.0)*((-ux)*fx+(-uy)*fy)*T(3)
            f[i,j,1]=f1-ω_local*(f1-feq_2d(Val(1), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((one(T)-ux)*fx+(-uy)*fy)*T(3)+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,2]=f2-ω_local*(f2-feq_2d(Val(2), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,3]=f3-ω_local*(f3-feq_2d(Val(3), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-one(T)-ux)*fx+(-uy)*fy)*T(3)+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,4]=f4-ω_local*(f4-feq_2d(Val(4), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,5]=f5-ω_local*(f5-feq_2d(Val(5), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(ux+uy)*(fx+fy)*T(9)
            f[i,j,6]=f6-ω_local*(f6-feq_2d(Val(6), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(-ux+uy)*(-fx+fy)*T(9)
            f[i,j,7]=f7-ω_local*(f7-feq_2d(Val(7), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(-ux-uy)*(-fx-fy)*T(9)
            f[i,j,8]=f8-ω_local*(f8-feq_2d(Val(8), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(ux-uy)*(fx-fy)*T(9)
            f[i,j,9]=f9-ω_local*(f9-feq_2d(Val(9), ρ, ux, uy, usq))+guo_pref*Sq
        end
    end
end

"""
    collide_boussinesq_vt_2d!(f, Temp, is_solid, ν_ref, T_ref, A_arr, β_g)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using Kraken

methods(Kraken.collide_boussinesq_vt_2d!)
```
"""
function collide_boussinesq_vt_2d!(f, Temp, is_solid, ν_ref, T_ref, A_arr, β_g)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    ET = eltype(f)
    kernel! = collide_boussinesq_vt_2d_kernel!(backend)
    kernel!(f, Temp, is_solid, ET(ν_ref), ET(T_ref), ET(A_arr), ET(β_g); ndrange=(Nx, Ny))
end

"""
    collide_boussinesq_2d!(f, Temp, is_solid, ω, β_g, T_ref)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using Kraken

methods(Kraken.collide_boussinesq_2d!)
```
"""
function collide_boussinesq_2d!(f, Temp, is_solid, ω, β_g, T_ref)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    T = eltype(f)
    kernel! = collide_boussinesq_2d_kernel!(backend)
    kernel!(f, Temp, is_solid, ω, T(β_g), T(T_ref); ndrange=(Nx, Ny))
end

# --- BGK collision with modified Arrhenius viscosity (linear exponent) ---

"""
    collide_boussinesq_vt_modified_2d!(f, Temp, is_solid, ν_ref, T0_visc, α_visc, β_g, T_ref_buoy)

BGK collision with per-node Boussinesq force AND temperature-dependent viscosity.
ν(T) = ν_ref · exp(α_visc · (T - T0_visc))  (modified Arrhenius / Frank-Kamenetskii)
ω(T) = 1 / (3·ν(T) + 0.5)
Buoyancy: Fy = β_g · (T - T_ref_buoy)

Used for rheological contrast studies: α_visc = ln(Rc), Rc = η_max/η_min.
"""
@kernel function collide_boussinesq_vt_modified_2d_kernel!(f, @Const(Temp), @Const(is_solid),
                                                            ν_ref, T0_visc, α_visc, β_g, T_ref_buoy)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            bounce_back_2d!(f, i, j)
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            # Modified Arrhenius: ν(T) = ν_ref * exp(α * (T - T0_visc))
            T_local = Temp[i,j]
            ν_local = ν_ref * exp(α_visc * (T_local - T0_visc))
            ω_local = one(T) / (T(3) * ν_local + T(0.5))

            # Per-node buoyancy
            fy = β_g * (T_local - T_ref_buoy)
            fx = zero(T)

            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ
            ux = ((f2-f4+f6-f7-f8+f9) + fx/T(2)) * inv_ρ
            uy = ((f3-f5+f6+f7-f8-f9) + fy/T(2)) * inv_ρ
            usq = ux*ux + uy*uy

            guo_pref = one(T) - ω_local / T(2)

            Sq=T(4.0/9.0)*((-ux)*fx+(-uy)*fy)*T(3)
            f[i,j,1]=f1-ω_local*(f1-feq_2d(Val(1), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((one(T)-ux)*fx+(-uy)*fy)*T(3)+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,2]=f2-ω_local*(f2-feq_2d(Val(2), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,3]=f3-ω_local*(f3-feq_2d(Val(3), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-one(T)-ux)*fx+(-uy)*fy)*T(3)+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,4]=f4-ω_local*(f4-feq_2d(Val(4), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,5]=f5-ω_local*(f5-feq_2d(Val(5), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(ux+uy)*(fx+fy)*T(9)
            f[i,j,6]=f6-ω_local*(f6-feq_2d(Val(6), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(-ux+uy)*(-fx+fy)*T(9)
            f[i,j,7]=f7-ω_local*(f7-feq_2d(Val(7), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(-ux-uy)*(-fx-fy)*T(9)
            f[i,j,8]=f8-ω_local*(f8-feq_2d(Val(8), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(ux-uy)*(fx-fy)*T(9)
            f[i,j,9]=f9-ω_local*(f9-feq_2d(Val(9), ρ, ux, uy, usq))+guo_pref*Sq
        end
    end
end

"""
    collide_boussinesq_vt_modified_2d!(f, Temp, is_solid, ν_ref, T0_visc, α_visc, β_g, T_ref_buoy)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using Kraken

methods(Kraken.collide_boussinesq_vt_modified_2d!)
```
"""
function collide_boussinesq_vt_modified_2d!(f, Temp, is_solid, ν_ref, T0_visc, α_visc, β_g, T_ref_buoy)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    ET = eltype(f)
    kernel! = collide_boussinesq_vt_modified_2d_kernel!(backend)
    kernel!(f, Temp, is_solid, ET(ν_ref), ET(T0_visc), ET(α_visc), ET(β_g), ET(T_ref_buoy); ndrange=(Nx, Ny))
end
