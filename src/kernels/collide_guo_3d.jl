using KernelAbstractions

# --- BGK collision with Guo body-force term (3D, D3Q19) ---

@kernel function collide_guo_3d_kernel!(f, @Const(is_solid), ω, Fx, Fy, Fz)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j, k]
            # Solid obstacle: full bounce-back (swap opposite directions)
            tmp = f[i,j,k,2]; f[i,j,k,2] = f[i,j,k,3]; f[i,j,k,3] = tmp
            tmp = f[i,j,k,4]; f[i,j,k,4] = f[i,j,k,5]; f[i,j,k,5] = tmp
            tmp = f[i,j,k,6]; f[i,j,k,6] = f[i,j,k,7]; f[i,j,k,7] = tmp
            tmp = f[i,j,k,8]; f[i,j,k,8] = f[i,j,k,11]; f[i,j,k,11] = tmp
            tmp = f[i,j,k,9]; f[i,j,k,9] = f[i,j,k,10]; f[i,j,k,10] = tmp
            tmp = f[i,j,k,12]; f[i,j,k,12] = f[i,j,k,15]; f[i,j,k,15] = tmp
            tmp = f[i,j,k,13]; f[i,j,k,13] = f[i,j,k,14]; f[i,j,k,14] = tmp
            tmp = f[i,j,k,16]; f[i,j,k,16] = f[i,j,k,19]; f[i,j,k,19] = tmp
            tmp = f[i,j,k,17]; f[i,j,k,17] = f[i,j,k,18]; f[i,j,k,18] = tmp
        else
            T = eltype(f)
            f1=f[i,j,k,1]; f2=f[i,j,k,2]; f3=f[i,j,k,3]; f4=f[i,j,k,4]
            f5=f[i,j,k,5]; f6=f[i,j,k,6]; f7=f[i,j,k,7]; f8=f[i,j,k,8]
            f9=f[i,j,k,9]; f10=f[i,j,k,10]; f11=f[i,j,k,11]; f12=f[i,j,k,12]
            f13=f[i,j,k,13]; f14=f[i,j,k,14]; f15=f[i,j,k,15]; f16=f[i,j,k,16]
            f17=f[i,j,k,17]; f18=f[i,j,k,18]; f19=f[i,j,k,19]

            # Force components cast to element type
            fx = T(Fx)
            fy = T(Fy)
            fz = T(Fz)

            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+f18+f19
            inv_ρ = one(T) / ρ
            # Force-corrected velocity: u = (Σf·c + F/2) / ρ  (Guo et al. 2002)
            ux = ((f2-f3+f8-f9+f10-f11+f12-f13+f14-f15) + fx/T(2)) * inv_ρ
            uy = ((f4-f5+f8+f9-f10-f11+f16-f17+f18-f19) + fy/T(2)) * inv_ρ
            uz = ((f6-f7+f12+f13-f14-f15+f16+f17-f18-f19) + fz/T(2)) * inv_ρ
            usq = ux*ux + uy*uy + uz*uz

            wr = T(1.0/3.0); wa = T(1.0/18.0); we = T(1.0/36.0)
            t3 = T(3); t45 = T(4.5); t15 = T(1.5); t9 = T(9)

            # Guo forcing prefactor
            guo_pref = one(T) - ω / T(2)

            # Guo source term: S_q = w_q * [(c_q - u)·F * 3 + (c_q·u)(c_q·F) * 9]
            # q=1: rest (0,0,0), w=1/3
            feq = wr * ρ * (one(T) - t15*usq)
            Sq = wr * ((-ux)*fx + (-uy)*fy + (-uz)*fz) * t3
            f[i,j,k,1] = f1 - ω*(f1-feq) + guo_pref*Sq

            # q=2: (+1,0,0), w=1/18
            cu = ux
            feq = wa * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = wa * ((one(T)-ux)*fx + (-uy)*fy + (-uz)*fz) * t3 + wa * cu * fx * t9
            f[i,j,k,2] = f2 - ω*(f2-feq) + guo_pref*Sq

            # q=3: (-1,0,0), w=1/18
            cu = -ux
            feq = wa * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = wa * ((-one(T)-ux)*fx + (-uy)*fy + (-uz)*fz) * t3 + wa * cu * (-fx) * t9
            f[i,j,k,3] = f3 - ω*(f3-feq) + guo_pref*Sq

            # q=4: (0,+1,0), w=1/18
            cu = uy
            feq = wa * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = wa * ((-ux)*fx + (one(T)-uy)*fy + (-uz)*fz) * t3 + wa * cu * fy * t9
            f[i,j,k,4] = f4 - ω*(f4-feq) + guo_pref*Sq

            # q=5: (0,-1,0), w=1/18
            cu = -uy
            feq = wa * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = wa * ((-ux)*fx + (-one(T)-uy)*fy + (-uz)*fz) * t3 + wa * cu * (-fy) * t9
            f[i,j,k,5] = f5 - ω*(f5-feq) + guo_pref*Sq

            # q=6: (0,0,+1), w=1/18
            cu = uz
            feq = wa * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = wa * ((-ux)*fx + (-uy)*fy + (one(T)-uz)*fz) * t3 + wa * cu * fz * t9
            f[i,j,k,6] = f6 - ω*(f6-feq) + guo_pref*Sq

            # q=7: (0,0,-1), w=1/18
            cu = -uz
            feq = wa * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = wa * ((-ux)*fx + (-uy)*fy + (-one(T)-uz)*fz) * t3 + wa * cu * (-fz) * t9
            f[i,j,k,7] = f7 - ω*(f7-feq) + guo_pref*Sq

            # q=8: (+1,+1,0), w=1/36
            cu = ux+uy
            cdotf = fx+fy
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((one(T)-ux)*fx + (one(T)-uy)*fy + (-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,8] = f8 - ω*(f8-feq) + guo_pref*Sq

            # q=9: (-1,+1,0), w=1/36
            cu = -ux+uy
            cdotf = -fx+fy
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((-one(T)-ux)*fx + (one(T)-uy)*fy + (-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,9] = f9 - ω*(f9-feq) + guo_pref*Sq

            # q=10: (+1,-1,0), w=1/36
            cu = ux-uy
            cdotf = fx-fy
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((one(T)-ux)*fx + (-one(T)-uy)*fy + (-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,10] = f10 - ω*(f10-feq) + guo_pref*Sq

            # q=11: (-1,-1,0), w=1/36
            cu = -ux-uy
            cdotf = -fx-fy
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((-one(T)-ux)*fx + (-one(T)-uy)*fy + (-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,11] = f11 - ω*(f11-feq) + guo_pref*Sq

            # q=12: (+1,0,+1), w=1/36
            cu = ux+uz
            cdotf = fx+fz
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((one(T)-ux)*fx + (-uy)*fy + (one(T)-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,12] = f12 - ω*(f12-feq) + guo_pref*Sq

            # q=13: (-1,0,+1), w=1/36
            cu = -ux+uz
            cdotf = -fx+fz
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((-one(T)-ux)*fx + (-uy)*fy + (one(T)-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,13] = f13 - ω*(f13-feq) + guo_pref*Sq

            # q=14: (+1,0,-1), w=1/36
            cu = ux-uz
            cdotf = fx-fz
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((one(T)-ux)*fx + (-uy)*fy + (-one(T)-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,14] = f14 - ω*(f14-feq) + guo_pref*Sq

            # q=15: (-1,0,-1), w=1/36
            cu = -ux-uz
            cdotf = -fx-fz
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((-one(T)-ux)*fx + (-uy)*fy + (-one(T)-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,15] = f15 - ω*(f15-feq) + guo_pref*Sq

            # q=16: (0,+1,+1), w=1/36
            cu = uy+uz
            cdotf = fy+fz
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((-ux)*fx + (one(T)-uy)*fy + (one(T)-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,16] = f16 - ω*(f16-feq) + guo_pref*Sq

            # q=17: (0,-1,+1), w=1/36
            cu = -uy+uz
            cdotf = -fy+fz
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((-ux)*fx + (-one(T)-uy)*fy + (one(T)-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,17] = f17 - ω*(f17-feq) + guo_pref*Sq

            # q=18: (0,+1,-1), w=1/36
            cu = uy-uz
            cdotf = fy-fz
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((-ux)*fx + (one(T)-uy)*fy + (-one(T)-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,18] = f18 - ω*(f18-feq) + guo_pref*Sq

            # q=19: (0,-1,-1), w=1/36
            cu = -uy-uz
            cdotf = -fy-fz
            feq = we * ρ * (one(T) + t3*cu + t45*cu*cu - t15*usq)
            Sq = we * ((-ux)*fx + (-one(T)-uy)*fy + (-one(T)-uz)*fz) * t3 + we * cu * cdotf * t9
            f[i,j,k,19] = f19 - ω*(f19-feq) + guo_pref*Sq
        end
    end
end

# --- Public API ---

"""
    collide_guo_3d!(f, is_solid, ω, Fx, Fy, Fz)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_guo_3d!)
```
"""
function collide_guo_3d!(f, is_solid, ω, Fx, Fy, Fz)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny, Nz = size(f,1), size(f,2), size(f,3)
    kernel! = collide_guo_3d_kernel!(backend)
    kernel!(f, is_solid, ω, Fx, Fy, Fz; ndrange=(Nx, Ny, Nz))
end

# --- Per-node force field variant ---

@kernel function collide_guo_field_3d_kernel!(f, @Const(is_solid), @Const(Fx_field), @Const(Fy_field), @Const(Fz_field), ω)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j, k]
            tmp = f[i,j,k,2]; f[i,j,k,2] = f[i,j,k,3]; f[i,j,k,3] = tmp
            tmp = f[i,j,k,4]; f[i,j,k,4] = f[i,j,k,5]; f[i,j,k,5] = tmp
            tmp = f[i,j,k,6]; f[i,j,k,6] = f[i,j,k,7]; f[i,j,k,7] = tmp
            tmp = f[i,j,k,8]; f[i,j,k,8] = f[i,j,k,11]; f[i,j,k,11] = tmp
            tmp = f[i,j,k,9]; f[i,j,k,9] = f[i,j,k,10]; f[i,j,k,10] = tmp
            tmp = f[i,j,k,12]; f[i,j,k,12] = f[i,j,k,15]; f[i,j,k,15] = tmp
            tmp = f[i,j,k,13]; f[i,j,k,13] = f[i,j,k,14]; f[i,j,k,14] = tmp
            tmp = f[i,j,k,16]; f[i,j,k,16] = f[i,j,k,19]; f[i,j,k,19] = tmp
            tmp = f[i,j,k,17]; f[i,j,k,17] = f[i,j,k,18]; f[i,j,k,18] = tmp
        else
            T = eltype(f)
            fx = Fx_field[i,j,k]
            fy = Fy_field[i,j,k]
            fz = Fz_field[i,j,k]

            f1=f[i,j,k,1]; f2=f[i,j,k,2]; f3=f[i,j,k,3]; f4=f[i,j,k,4]
            f5=f[i,j,k,5]; f6=f[i,j,k,6]; f7=f[i,j,k,7]; f8=f[i,j,k,8]
            f9=f[i,j,k,9]; f10=f[i,j,k,10]; f11=f[i,j,k,11]; f12=f[i,j,k,12]
            f13=f[i,j,k,13]; f14=f[i,j,k,14]; f15=f[i,j,k,15]; f16=f[i,j,k,16]
            f17=f[i,j,k,17]; f18=f[i,j,k,18]; f19=f[i,j,k,19]

            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+f18+f19
            inv_ρ = one(T) / ρ
            ux = ((f2-f3+f8-f9+f10-f11+f12-f13+f14-f15) + fx/T(2)) * inv_ρ
            uy = ((f4-f5+f8+f9-f10-f11+f16-f17+f18-f19) + fy/T(2)) * inv_ρ
            uz = ((f6-f7+f12+f13-f14-f15+f16+f17-f18-f19) + fz/T(2)) * inv_ρ
            usq = ux*ux + uy*uy + uz*uz

            wr = T(1.0/3.0); wa = T(1.0/18.0); we = T(1.0/36.0)
            t3 = T(3); t45 = T(4.5); t15 = T(1.5); t9 = T(9)
            guo_pref = one(T) - ω / T(2)

            feq = wr * ρ * (one(T) - t15*usq)
            Sq = wr * ((-ux)*fx + (-uy)*fy + (-uz)*fz) * t3
            f[i,j,k,1] = f1 - ω*(f1-feq) + guo_pref*Sq

            cu=ux; feq=wa*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=wa*((one(T)-ux)*fx+(-uy)*fy+(-uz)*fz)*t3+wa*cu*fx*t9
            f[i,j,k,2]=f2-ω*(f2-feq)+guo_pref*Sq

            cu=-ux; feq=wa*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=wa*((-one(T)-ux)*fx+(-uy)*fy+(-uz)*fz)*t3+wa*cu*(-fx)*t9
            f[i,j,k,3]=f3-ω*(f3-feq)+guo_pref*Sq

            cu=uy; feq=wa*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=wa*((-ux)*fx+(one(T)-uy)*fy+(-uz)*fz)*t3+wa*cu*fy*t9
            f[i,j,k,4]=f4-ω*(f4-feq)+guo_pref*Sq

            cu=-uy; feq=wa*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=wa*((-ux)*fx+(-one(T)-uy)*fy+(-uz)*fz)*t3+wa*cu*(-fy)*t9
            f[i,j,k,5]=f5-ω*(f5-feq)+guo_pref*Sq

            cu=uz; feq=wa*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=wa*((-ux)*fx+(-uy)*fy+(one(T)-uz)*fz)*t3+wa*cu*fz*t9
            f[i,j,k,6]=f6-ω*(f6-feq)+guo_pref*Sq

            cu=-uz; feq=wa*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=wa*((-ux)*fx+(-uy)*fy+(-one(T)-uz)*fz)*t3+wa*cu*(-fz)*t9
            f[i,j,k,7]=f7-ω*(f7-feq)+guo_pref*Sq

            cu=ux+uy; cdotf=fx+fy; feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=we*((one(T)-ux)*fx+(one(T)-uy)*fy+(-uz)*fz)*t3+we*cu*cdotf*t9
            f[i,j,k,8]=f8-ω*(f8-feq)+guo_pref*Sq

            cu=-ux+uy; cdotf=-fx+fy; feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=we*((-one(T)-ux)*fx+(one(T)-uy)*fy+(-uz)*fz)*t3+we*cu*cdotf*t9
            f[i,j,k,9]=f9-ω*(f9-feq)+guo_pref*Sq

            cu=ux-uy; cdotf=fx-fy; feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=we*((one(T)-ux)*fx+(-one(T)-uy)*fy+(-uz)*fz)*t3+we*cu*cdotf*t9
            f[i,j,k,10]=f10-ω*(f10-feq)+guo_pref*Sq

            cu=-ux-uy; cdotf=-fx-fy; feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=we*((-one(T)-ux)*fx+(-one(T)-uy)*fy+(-uz)*fz)*t3+we*cu*cdotf*t9
            f[i,j,k,11]=f11-ω*(f11-feq)+guo_pref*Sq

            cu=ux+uz; cdotf=fx+fz; feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=we*((one(T)-ux)*fx+(-uy)*fy+(one(T)-uz)*fz)*t3+we*cu*cdotf*t9
            f[i,j,k,12]=f12-ω*(f12-feq)+guo_pref*Sq

            cu=-ux+uz; cdotf=-fx+fz; feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=we*((-one(T)-ux)*fx+(-uy)*fy+(one(T)-uz)*fz)*t3+we*cu*cdotf*t9
            f[i,j,k,13]=f13-ω*(f13-feq)+guo_pref*Sq

            cu=ux-uz; cdotf=fx-fz; feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=we*((one(T)-ux)*fx+(-uy)*fy+(-one(T)-uz)*fz)*t3+we*cu*cdotf*t9
            f[i,j,k,14]=f14-ω*(f14-feq)+guo_pref*Sq

            cu=-ux-uz; cdotf=-fx-fz; feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=we*((-one(T)-ux)*fx+(-uy)*fy+(-one(T)-uz)*fz)*t3+we*cu*cdotf*t9
            f[i,j,k,15]=f15-ω*(f15-feq)+guo_pref*Sq

            cu=uy+uz; cdotf=fy+fz; feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=we*((-ux)*fx+(one(T)-uy)*fy+(one(T)-uz)*fz)*t3+we*cu*cdotf*t9
            f[i,j,k,16]=f16-ω*(f16-feq)+guo_pref*Sq

            cu=-uy+uz; cdotf=-fy+fz; feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=we*((-ux)*fx+(-one(T)-uy)*fy+(one(T)-uz)*fz)*t3+we*cu*cdotf*t9
            f[i,j,k,17]=f17-ω*(f17-feq)+guo_pref*Sq

            cu=uy-uz; cdotf=fy-fz; feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=we*((-ux)*fx+(one(T)-uy)*fy+(-one(T)-uz)*fz)*t3+we*cu*cdotf*t9
            f[i,j,k,18]=f18-ω*(f18-feq)+guo_pref*Sq

            cu=-uy-uz; cdotf=-fy-fz; feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=we*((-ux)*fx+(-one(T)-uy)*fy+(-one(T)-uz)*fz)*t3+we*cu*cdotf*t9
            f[i,j,k,19]=f19-ω*(f19-feq)+guo_pref*Sq
        end
    end
end

"""
    collide_guo_field_3d!(f, is_solid, Fx_field, Fy_field, Fz_field, ω)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_guo_field_3d!)
```
"""
function collide_guo_field_3d!(f, is_solid, Fx_field, Fy_field, Fz_field, ω)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny, Nz = size(f,1), size(f,2), size(f,3)
    kernel! = collide_guo_field_3d_kernel!(backend)
    kernel!(f, is_solid, Fx_field, Fy_field, Fz_field, ω; ndrange=(Nx, Ny, Nz))
end
