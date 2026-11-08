using KernelAbstractions

# --- BGK collision with Guo body-force term ---

@kernel function collide_guo_2d_kernel!(f, @Const(is_solid), ω, Fx, Fy)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            # Solid obstacle: full bounce-back (swap opposite directions)
            bounce_back_2d!(f, i, j)
        else
            T = eltype(f)
            f1 = f[i,j,1]; f2 = f[i,j,2]; f3 = f[i,j,3]
            f4 = f[i,j,4]; f5 = f[i,j,5]; f6 = f[i,j,6]
            f7 = f[i,j,7]; f8 = f[i,j,8]; f9 = f[i,j,9]

            # Force components cast to element type
            fx = T(Fx)
            fy = T(Fy)

            ρ = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
            inv_ρ = one(T) / ρ
            # Force-corrected velocity: u = (Σf·c + F/2) / ρ  (Guo et al. 2002)
            ux = ((f2 - f4 + f6 - f7 - f8 + f9) + fx / T(2)) * inv_ρ
            uy = ((f3 - f5 + f6 + f7 - f8 - f9) + fy / T(2)) * inv_ρ
            usq = ux * ux + uy * uy

            # Guo forcing prefactor
            guo_pref = one(T) - ω / T(2)

            Sq = T(4.0/9.0) * ((-ux) * fx + (-uy) * fy) * T(3)
            f[i,j,1] = f1 - ω * (f1 - feq_2d(Val(1), ρ, ux, uy, usq)) + guo_pref * Sq

            Sq = T(1.0/9.0) * ((one(T) - ux) * fx + (-uy) * fy) * T(3) +
                 T(1.0/9.0) * (ux) * (fx) * T(9)
            f[i,j,2] = f2 - ω * (f2 - feq_2d(Val(2), ρ, ux, uy, usq)) + guo_pref * Sq

            Sq = T(1.0/9.0) * ((-ux) * fx + (one(T) - uy) * fy) * T(3) +
                 T(1.0/9.0) * (uy) * (fy) * T(9)
            f[i,j,3] = f3 - ω * (f3 - feq_2d(Val(3), ρ, ux, uy, usq)) + guo_pref * Sq

            Sq = T(1.0/9.0) * ((-one(T) - ux) * fx + (-uy) * fy) * T(3) +
                 T(1.0/9.0) * (-ux) * (-fx) * T(9)
            f[i,j,4] = f4 - ω * (f4 - feq_2d(Val(4), ρ, ux, uy, usq)) + guo_pref * Sq

            Sq = T(1.0/9.0) * ((-ux) * fx + (-one(T) - uy) * fy) * T(3) +
                 T(1.0/9.0) * (-uy) * (-fy) * T(9)
            f[i,j,5] = f5 - ω * (f5 - feq_2d(Val(5), ρ, ux, uy, usq)) + guo_pref * Sq

            Sq = T(1.0/36.0) * ((one(T) - ux) * fx + (one(T) - uy) * fy) * T(3) +
                 T(1.0/36.0) * (ux + uy) * (fx + fy) * T(9)
            f[i,j,6] = f6 - ω * (f6 - feq_2d(Val(6), ρ, ux, uy, usq)) + guo_pref * Sq

            Sq = T(1.0/36.0) * ((-one(T) - ux) * fx + (one(T) - uy) * fy) * T(3) +
                 T(1.0/36.0) * (-ux + uy) * (-fx + fy) * T(9)
            f[i,j,7] = f7 - ω * (f7 - feq_2d(Val(7), ρ, ux, uy, usq)) + guo_pref * Sq

            Sq = T(1.0/36.0) * ((-one(T) - ux) * fx + (-one(T) - uy) * fy) * T(3) +
                 T(1.0/36.0) * (-ux - uy) * (-fx - fy) * T(9)
            f[i,j,8] = f8 - ω * (f8 - feq_2d(Val(8), ρ, ux, uy, usq)) + guo_pref * Sq

            Sq = T(1.0/36.0) * ((one(T) - ux) * fx + (-one(T) - uy) * fy) * T(3) +
                 T(1.0/36.0) * (ux - uy) * (fx - fy) * T(9)
            f[i,j,9] = f9 - ω * (f9 - feq_2d(Val(9), ρ, ux, uy, usq)) + guo_pref * Sq
        end
    end
end

# --- Public API ---

"""
    collide_guo_2d!(f, is_solid, ω, Fx, Fy)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_guo_2d!)
```
"""
function collide_guo_2d!(f, is_solid, ω, Fx, Fy)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_guo_2d_kernel!(backend)
    kernel!(f, is_solid, ω, Fx, Fy; ndrange=(Nx, Ny))
end

# --- Per-node force field variant ---

@kernel function collide_guo_field_2d_kernel!(f, @Const(is_solid), @Const(Fx_field), @Const(Fy_field), ω)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            bounce_back_2d!(f, i, j)
        else
            T = eltype(f)
            fx = Fx_field[i,j]
            fy = Fy_field[i,j]

            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ
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
    collide_guo_field_2d!(f, is_solid, Fx_field, Fy_field, ω)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_guo_field_2d!)
```
"""
function collide_guo_field_2d!(f, is_solid, Fx_field, Fy_field, ω)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_guo_field_2d_kernel!(backend)
    kernel!(f, is_solid, Fx_field, Fy_field, ω; ndrange=(Nx, Ny))
end
