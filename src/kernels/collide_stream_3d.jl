using KernelAbstractions

# --- Stream kernel (pull scheme with bounce-back at boundaries) ---

@kernel function stream_3d_kernel!(f_out, @Const(f_in), Nx, Ny, Nz)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        # Clamped neighbor indices — safe for CPU where ifelse evaluates both branches
        im = max(i-1, 1); ip = min(i+1, Nx)
        jm = max(j-1, 1); jp = min(j+1, Ny)
        km = max(k-1, 1); kp = min(k+1, Nz)

        f_out[i,j,k,1] = f_in[i, j, k, 1]  # rest

        # Axis-aligned
        f_out[i,j,k,2]  = ifelse(i > 1,             f_in[im, j,  k,  2],  f_in[i,j,k, 3])
        f_out[i,j,k,3]  = ifelse(i < Nx,            f_in[ip, j,  k,  3],  f_in[i,j,k, 2])
        f_out[i,j,k,4]  = ifelse(j > 1,             f_in[i,  jm, k,  4],  f_in[i,j,k, 5])
        f_out[i,j,k,5]  = ifelse(j < Ny,            f_in[i,  jp, k,  5],  f_in[i,j,k, 4])
        f_out[i,j,k,6]  = ifelse(k > 1,             f_in[i,  j,  km, 6],  f_in[i,j,k, 7])
        f_out[i,j,k,7]  = ifelse(k < Nz,            f_in[i,  j,  kp, 7],  f_in[i,j,k, 6])

        # Edge xy
        f_out[i,j,k,8]  = ifelse(i > 1  && j > 1,   f_in[im,jm,k, 8],  f_in[i,j,k,11])
        f_out[i,j,k,9]  = ifelse(i < Nx && j > 1,   f_in[ip,jm,k, 9],  f_in[i,j,k,10])
        f_out[i,j,k,10] = ifelse(i > 1  && j < Ny,  f_in[im,jp,k,10],  f_in[i,j,k, 9])
        f_out[i,j,k,11] = ifelse(i < Nx && j < Ny,  f_in[ip,jp,k,11],  f_in[i,j,k, 8])

        # Edge xz
        f_out[i,j,k,12] = ifelse(i > 1  && k > 1,   f_in[im,j,km,12],  f_in[i,j,k,15])
        f_out[i,j,k,13] = ifelse(i < Nx && k > 1,   f_in[ip,j,km,13],  f_in[i,j,k,14])
        f_out[i,j,k,14] = ifelse(i > 1  && k < Nz,  f_in[im,j,kp,14],  f_in[i,j,k,13])
        f_out[i,j,k,15] = ifelse(i < Nx && k < Nz,  f_in[ip,j,kp,15],  f_in[i,j,k,12])

        # Edge yz
        f_out[i,j,k,16] = ifelse(j > 1  && k > 1,   f_in[i,jm,km,16],  f_in[i,j,k,19])
        f_out[i,j,k,17] = ifelse(j < Ny && k > 1,   f_in[i,jp,km,17],  f_in[i,j,k,18])
        f_out[i,j,k,18] = ifelse(j > 1  && k < Nz,  f_in[i,jm,kp,18],  f_in[i,j,k,17])
        f_out[i,j,k,19] = ifelse(j < Ny && k < Nz,  f_in[i,jp,kp,19],  f_in[i,j,k,16])
    end
end

# --- BGK collision kernel ---

@kernel function collide_3d_kernel!(f, @Const(is_solid), ω)
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
        f1=f[i,j,k,1]; f2=f[i,j,k,2]; f3=f[i,j,k,3]; f4=f[i,j,k,4]
        f5=f[i,j,k,5]; f6=f[i,j,k,6]; f7=f[i,j,k,7]; f8=f[i,j,k,8]
        f9=f[i,j,k,9]; f10=f[i,j,k,10]; f11=f[i,j,k,11]; f12=f[i,j,k,12]
        f13=f[i,j,k,13]; f14=f[i,j,k,14]; f15=f[i,j,k,15]; f16=f[i,j,k,16]
        f17=f[i,j,k,17]; f18=f[i,j,k,18]; f19=f[i,j,k,19]

        ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9+f10+f11+f12+f13+f14+f15+f16+f17+f18+f19
        inv_ρ = one(T) / ρ
        ux = (f2-f3+f8-f9+f10-f11+f12-f13+f14-f15) * inv_ρ
        uy = (f4-f5+f8+f9-f10-f11+f16-f17+f18-f19) * inv_ρ
        uz = (f6-f7+f12+f13-f14-f15+f16+f17-f18-f19) * inv_ρ
        usq = ux*ux + uy*uy + uz*uz

        wr = T(1.0/3.0); wa = T(1.0/18.0); we = T(1.0/36.0)
        t3 = T(3); t45 = T(4.5); t15 = T(1.5)

        feq = wr * ρ * (one(T) - t15*usq)
        f[i,j,k,1] = f1 - ω*(f1-feq)

        cu=ux;   feq=wa*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,2]=f2-ω*(f2-feq)
        cu=-ux;  feq=wa*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,3]=f3-ω*(f3-feq)
        cu=uy;   feq=wa*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,4]=f4-ω*(f4-feq)
        cu=-uy;  feq=wa*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,5]=f5-ω*(f5-feq)
        cu=uz;   feq=wa*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,6]=f6-ω*(f6-feq)
        cu=-uz;  feq=wa*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,7]=f7-ω*(f7-feq)

        cu=ux+uy;   feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,8]=f8-ω*(f8-feq)
        cu=-ux+uy;  feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,9]=f9-ω*(f9-feq)
        cu=ux-uy;   feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,10]=f10-ω*(f10-feq)
        cu=-ux-uy;  feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,11]=f11-ω*(f11-feq)
        cu=ux+uz;   feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,12]=f12-ω*(f12-feq)
        cu=-ux+uz;  feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,13]=f13-ω*(f13-feq)
        cu=ux-uz;   feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,14]=f14-ω*(f14-feq)
        cu=-ux-uz;  feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,15]=f15-ω*(f15-feq)
        cu=uy+uz;   feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,16]=f16-ω*(f16-feq)
        cu=-uy+uz;  feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,17]=f17-ω*(f17-feq)
        cu=uy-uz;   feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,18]=f18-ω*(f18-feq)
        cu=-uy-uz;  feq=we*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq); f[i,j,k,19]=f19-ω*(f19-feq)
        end  # else (not solid)
    end
end

# --- Public API ---

"""
    stream_3d!(f_out, f_in, Nx, Ny, Nz)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.stream_3d!)
```
"""
function stream_3d!(f_out, f_in, Nx, Ny, Nz)
    backend = KernelAbstractions.get_backend(f_in)
    kernel! = stream_3d_kernel!(backend)
    kernel!(f_out, f_in, Nx, Ny, Nz; ndrange=(Nx, Ny, Nz))
end

"""
    collide_3d!(f, is_solid, ω)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_3d!)
```
"""
function collide_3d!(f, is_solid, ω)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny, Nz = size(f,1), size(f,2), size(f,3)
    kernel! = collide_3d_kernel!(backend)
    kernel!(f, is_solid, ω; ndrange=(Nx, Ny, Nz))
end
