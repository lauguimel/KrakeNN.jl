using KernelAbstractions

# --- Stream kernel (pull scheme with bounce-back at boundaries) ---

@kernel function stream_2d_kernel!(f_out, @Const(f_in), Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        # Pull from neighbors; if source is out of bounds → bounce-back
        # Clamped indices ensure valid memory access on both CPU and GPU
        # (ifelse evaluates both branches, so the "dead" read must be in-bounds)
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

        f_out[i,j,1] = fp1
        f_out[i,j,2] = fp2; f_out[i,j,3] = fp3
        f_out[i,j,4] = fp4; f_out[i,j,5] = fp5
        f_out[i,j,6] = fp6; f_out[i,j,7] = fp7
        f_out[i,j,8] = fp8; f_out[i,j,9] = fp9
    end
end

# --- BGK collision kernel ---

@kernel function collide_2d_kernel!(f, @Const(is_solid), ω)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            bounce_back_2d!(f, i, j)
        else
            T = eltype(f)
            f1 = f[i,j,1]; f2 = f[i,j,2]; f3 = f[i,j,3]
            f4 = f[i,j,4]; f5 = f[i,j,5]; f6 = f[i,j,6]
            f7 = f[i,j,7]; f8 = f[i,j,8]; f9 = f[i,j,9]

            ρ, ux, uy = moments_2d(f1, f2, f3, f4, f5, f6, f7, f8, f9)
            usq = ux * ux + uy * uy

            f[i,j,1] = f1 - ω * (f1 - feq_2d(Val(1), ρ, ux, uy, usq))
            f[i,j,2] = f2 - ω * (f2 - feq_2d(Val(2), ρ, ux, uy, usq))
            f[i,j,3] = f3 - ω * (f3 - feq_2d(Val(3), ρ, ux, uy, usq))
            f[i,j,4] = f4 - ω * (f4 - feq_2d(Val(4), ρ, ux, uy, usq))
            f[i,j,5] = f5 - ω * (f5 - feq_2d(Val(5), ρ, ux, uy, usq))
            f[i,j,6] = f6 - ω * (f6 - feq_2d(Val(6), ρ, ux, uy, usq))
            f[i,j,7] = f7 - ω * (f7 - feq_2d(Val(7), ρ, ux, uy, usq))
            f[i,j,8] = f8 - ω * (f8 - feq_2d(Val(8), ρ, ux, uy, usq))
            f[i,j,9] = f9 - ω * (f9 - feq_2d(Val(9), ρ, ux, uy, usq))
        end
    end
end

# --- Public API ---

"""
    stream_2d!(f_out, f_in, Nx, Ny; sync=false)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.stream_2d!)
```
"""
function stream_2d!(f_out, f_in, Nx, Ny; sync=false)
    backend = KernelAbstractions.get_backend(f_in)
    kernel! = stream_2d_kernel!(backend)
    kernel!(f_out, f_in, Nx, Ny; ndrange=(Nx, Ny))
    sync && KernelAbstractions.synchronize(backend)
end

"""
    collide_2d!(f, is_solid, ω; sync=false)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_2d!)
```
"""
function collide_2d!(f, is_solid, ω; sync=false)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_2d_kernel!(backend)
    kernel!(f, is_solid, ω; ndrange=(Nx, Ny))
    sync && KernelAbstractions.synchronize(backend)
end
