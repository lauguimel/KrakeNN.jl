using KernelAbstractions

# --- Shan-Chen pseudo-potential multiphase model ---
#
# Inter-particle force: F(x) = -G·ψ(x)·Σ_α w_α·ψ(x+e_α)·e_α
# where ψ(ρ) = ρ₀·(1 - exp(-ρ/ρ₀)) is the pseudo-potential
# and G controls interaction strength (G < 0 for attraction → phase separation)
#
# The force is added to the collision via Guo forcing scheme.
# Equation of state: p = ρ·cs² + G·cs²/2·ψ²

# --- Compute pseudo-potential ψ(ρ) ---

@kernel function compute_psi_2d_kernel!(ψ, @Const(ρ), ρ0)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(ψ)
        ψ[i,j] = ρ0 * (one(T) - exp(-ρ[i,j] / ρ0))
    end
end

"""
    compute_psi_2d!(ψ, ρ, ρ0)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.compute_psi_2d!)
```
"""
function compute_psi_2d!(ψ, ρ, ρ0)
    backend = KernelAbstractions.get_backend(ρ)
    Nx, Ny = size(ρ)
    T = eltype(ψ)
    kernel! = compute_psi_2d_kernel!(backend)
    kernel!(ψ, ρ, T(ρ0); ndrange=(Nx, Ny))
end

# --- Compute Shan-Chen interaction force ---

@kernel function compute_sc_force_2d_kernel!(Fx, Fy, @Const(ψ), G, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(Fx)
        ψ_local = ψ[i,j]

        # D2Q9 neighbor contributions (periodic wrapping)
        ip = ifelse(i > 1, i-1, Nx); im = ifelse(i < Nx, i+1, 1)
        jp = ifelse(j > 1, j-1, Ny); jm = ifelse(j < Ny, j+1, 1)

        # Axis weights: w=1/9, diagonal weights: w=1/36
        # F = -G·ψ(x)·Σ w_α·ψ(x+e_α)·e_α
        fx = -G * ψ_local * (
            T(1.0/9.0)  * (ψ[ip,j] - ψ[im,j]) +                              # E-W
            T(1.0/36.0) * (ψ[ip,jp] - ψ[im,jp] - ψ[im,jm] + ψ[ip,jm])       # diagonals
        )
        fy = -G * ψ_local * (
            T(1.0/9.0)  * (ψ[i,jp] - ψ[i,jm]) +                              # N-S
            T(1.0/36.0) * (ψ[ip,jp] + ψ[im,jp] - ψ[im,jm] - ψ[ip,jm])       # diagonals
        )

        Fx[i,j] = fx
        Fy[i,j] = fy
    end
end

"""
    compute_sc_force_2d!(Fx, Fy, ψ, G, Nx, Ny)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.compute_sc_force_2d!)
```
"""
function compute_sc_force_2d!(Fx, Fy, ψ, G, Nx, Ny)
    backend = KernelAbstractions.get_backend(ψ)
    T = eltype(Fx)
    kernel! = compute_sc_force_2d_kernel!(backend)
    kernel!(Fx, Fy, ψ, T(G), Nx, Ny; ndrange=(Nx, Ny))
end

# --- BGK collision with per-node Shan-Chen force (Guo scheme) ---

@kernel function collide_sc_2d_kernel!(f, @Const(Fx_sc), @Const(Fy_sc), @Const(is_solid), ω)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            bounce_back_2d!(f, i, j)
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            fx = Fx_sc[i,j]
            fy = Fy_sc[i,j]

            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ
            # Force-corrected velocity (Guo)
            ux = ((f2-f4+f6-f7-f8+f9) + fx/T(2)) * inv_ρ
            uy = ((f3-f5+f6+f7-f8-f9) + fy/T(2)) * inv_ρ
            usq = ux*ux + uy*uy

            guo_pref = one(T) - ω / T(2)

            Sq=T(4.0/9.0)*((-ux)*fx+(-uy)*fy)*T(3)
            f[i,j,1]=f1-ω*(f1-feq_2d(Val(1), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((one(T)-ux)*fx+(-uy)*fy)*T(3)+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,2]=f2-ω*(f2-feq_2d(Val(2), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,3]=f3-ω*(f3-feq_2d(Val(3), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-one(T)-ux)*fx+(-uy)*fy)*T(3)+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,4]=f4-ω*(f4-feq_2d(Val(4), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,5]=f5-ω*(f5-feq_2d(Val(5), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(ux+uy)*(fx+fy)*T(9)
            f[i,j,6]=f6-ω*(f6-feq_2d(Val(6), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(-ux+uy)*(-fx+fy)*T(9)
            f[i,j,7]=f7-ω*(f7-feq_2d(Val(7), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(-ux-uy)*(-fx-fy)*T(9)
            f[i,j,8]=f8-ω*(f8-feq_2d(Val(8), ρ, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(ux-uy)*(fx-fy)*T(9)
            f[i,j,9]=f9-ω*(f9-feq_2d(Val(9), ρ, ux, uy, usq))+guo_pref*Sq
        end
    end
end

"""
    collide_sc_2d!(f, Fx_sc, Fy_sc, is_solid, ω)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_sc_2d!)
```
"""
function collide_sc_2d!(f, Fx_sc, Fy_sc, is_solid, ω)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_sc_2d_kernel!(backend)
    kernel!(f, Fx_sc, Fy_sc, is_solid, ω; ndrange=(Nx, Ny))
end
