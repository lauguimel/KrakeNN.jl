using KernelAbstractions

# --- 2D macroscopic computation ---

@kernel function compute_macroscopic_2d_kernel!(ρ, ux, uy, @Const(f))
    i, j = @index(Global, NTuple)
    @inbounds begin
        f1 = f[i,j,1]; f2 = f[i,j,2]; f3 = f[i,j,3]
        f4 = f[i,j,4]; f5 = f[i,j,5]; f6 = f[i,j,6]
        f7 = f[i,j,7]; f8 = f[i,j,8]; f9 = f[i,j,9]

        ρ_local = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
        inv_ρ = one(ρ_local) / ρ_local
        ρ[i,j] = ρ_local
        ux[i,j] = (f2 - f4 + f6 - f7 - f8 + f9) * inv_ρ
        uy[i,j] = (f3 - f5 + f6 + f7 - f8 - f9) * inv_ρ
    end
end

"""
    compute_macroscopic_2d!(ρ, ux, uy, f; sync=false)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using Kraken

methods(Kraken.compute_macroscopic_2d!)
```
"""
function compute_macroscopic_2d!(ρ, ux, uy, f; sync=false)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(ρ)
    kernel! = compute_macroscopic_2d_kernel!(backend)
    kernel!(ρ, ux, uy, f; ndrange=(Nx, Ny))
    sync && KernelAbstractions.synchronize(backend)
end

# --- 3D macroscopic computation ---

@kernel function compute_macroscopic_3d_kernel!(ρ, ux, uy, uz, @Const(f))
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        f1  = f[i,j,k,1];  f2  = f[i,j,k,2];  f3  = f[i,j,k,3]
        f4  = f[i,j,k,4];  f5  = f[i,j,k,5];  f6  = f[i,j,k,6]
        f7  = f[i,j,k,7];  f8  = f[i,j,k,8];  f9  = f[i,j,k,9]
        f10 = f[i,j,k,10]; f11 = f[i,j,k,11]; f12 = f[i,j,k,12]
        f13 = f[i,j,k,13]; f14 = f[i,j,k,14]; f15 = f[i,j,k,15]
        f16 = f[i,j,k,16]; f17 = f[i,j,k,17]; f18 = f[i,j,k,18]
        f19 = f[i,j,k,19]

        ρ_local = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 +
                  f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f19
        inv_ρ = one(ρ_local) / ρ_local
        ρ[i,j,k] = ρ_local
        ux[i,j,k] = (f2 - f3 + f8 - f9 + f10 - f11 + f12 - f13 + f14 - f15) * inv_ρ
        uy[i,j,k] = (f4 - f5 + f8 + f9 - f10 - f11 + f16 - f17 + f18 - f19) * inv_ρ
        uz[i,j,k] = (f6 - f7 + f12 + f13 - f14 - f15 + f16 + f17 - f18 - f19) * inv_ρ
    end
end

"""
    compute_macroscopic_3d!(ρ, ux, uy, uz, f)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using Kraken

methods(Kraken.compute_macroscopic_3d!)
```
"""
function compute_macroscopic_3d!(ρ, ux, uy, uz, f)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny, Nz = size(ρ)
    kernel! = compute_macroscopic_3d_kernel!(backend)
    kernel!(ρ, ux, uy, uz, f; ndrange=(Nx, Ny, Nz))
end

# --- 2D force-corrected macroscopic computation ---

@kernel function compute_macroscopic_forced_2d_kernel!(ρ, ux, uy, @Const(f), Fx, Fy)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)
        f1 = f[i,j,1]; f2 = f[i,j,2]; f3 = f[i,j,3]
        f4 = f[i,j,4]; f5 = f[i,j,5]; f6 = f[i,j,6]
        f7 = f[i,j,7]; f8 = f[i,j,8]; f9 = f[i,j,9]

        ρ_local = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
        inv_ρ = one(ρ_local) / ρ_local
        ρ[i,j] = ρ_local
        ux[i,j] = ((f2 - f4 + f6 - f7 - f8 + f9) + T(Fx) / T(2)) * inv_ρ
        uy[i,j] = ((f3 - f5 + f6 + f7 - f8 - f9) + T(Fy) / T(2)) * inv_ρ
    end
end

"""
    compute_macroscopic_forced_2d!(ρ, ux, uy, f, Fx, Fy)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using Kraken

methods(Kraken.compute_macroscopic_forced_2d!)
```
"""
function compute_macroscopic_forced_2d!(ρ, ux, uy, f, Fx, Fy)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(ρ)
    kernel! = compute_macroscopic_forced_2d_kernel!(backend)
    kernel!(ρ, ux, uy, f, Fx, Fy; ndrange=(Nx, Ny))
end

# --- 3D force-corrected macroscopic computation ---

@kernel function compute_macroscopic_forced_3d_kernel!(ρ, ux, uy, uz, @Const(f), Fx, Fy, Fz)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)
        f1  = f[i,j,k,1];  f2  = f[i,j,k,2];  f3  = f[i,j,k,3]
        f4  = f[i,j,k,4];  f5  = f[i,j,k,5];  f6  = f[i,j,k,6]
        f7  = f[i,j,k,7];  f8  = f[i,j,k,8];  f9  = f[i,j,k,9]
        f10 = f[i,j,k,10]; f11 = f[i,j,k,11]; f12 = f[i,j,k,12]
        f13 = f[i,j,k,13]; f14 = f[i,j,k,14]; f15 = f[i,j,k,15]
        f16 = f[i,j,k,16]; f17 = f[i,j,k,17]; f18 = f[i,j,k,18]
        f19 = f[i,j,k,19]

        ρ_local = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 +
                  f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f19
        inv_ρ = one(ρ_local) / ρ_local
        ρ[i,j,k] = ρ_local
        ux[i,j,k] = ((f2 - f3 + f8 - f9 + f10 - f11 + f12 - f13 + f14 - f15) + T(Fx) / T(2)) * inv_ρ
        uy[i,j,k] = ((f4 - f5 + f8 + f9 - f10 - f11 + f16 - f17 + f18 - f19) + T(Fy) / T(2)) * inv_ρ
        uz[i,j,k] = ((f6 - f7 + f12 + f13 - f14 - f15 + f16 + f17 - f18 - f19) + T(Fz) / T(2)) * inv_ρ
    end
end

"""
    compute_macroscopic_forced_3d!(ρ, ux, uy, uz, f, Fx, Fy, Fz)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using Kraken

methods(Kraken.compute_macroscopic_forced_3d!)
```
"""
function compute_macroscopic_forced_3d!(ρ, ux, uy, uz, f, Fx, Fy, Fz)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny, Nz = size(ρ)
    kernel! = compute_macroscopic_forced_3d_kernel!(backend)
    kernel!(ρ, ux, uy, uz, f, Fx, Fy, Fz; ndrange=(Nx, Ny, Nz))
end

# --- 3D force-corrected macroscopic with a per-cell force FIELD ---

@kernel function compute_macroscopic_forced_field_3d_kernel!(ρ, ux, uy, uz, @Const(f),
                                                              @Const(Fx), @Const(Fy), @Const(Fz))
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)
        f1  = f[i,j,k,1];  f2  = f[i,j,k,2];  f3  = f[i,j,k,3]
        f4  = f[i,j,k,4];  f5  = f[i,j,k,5];  f6  = f[i,j,k,6]
        f7  = f[i,j,k,7];  f8  = f[i,j,k,8];  f9  = f[i,j,k,9]
        f10 = f[i,j,k,10]; f11 = f[i,j,k,11]; f12 = f[i,j,k,12]
        f13 = f[i,j,k,13]; f14 = f[i,j,k,14]; f15 = f[i,j,k,15]
        f16 = f[i,j,k,16]; f17 = f[i,j,k,17]; f18 = f[i,j,k,18]
        f19 = f[i,j,k,19]

        ρ_local = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 +
                  f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f19
        inv_ρ = one(ρ_local) / ρ_local
        ρ[i,j,k] = ρ_local
        fx = Fx[i,j,k]; fy = Fy[i,j,k]; fz = Fz[i,j,k]
        ux[i,j,k] = ((f2 - f3 + f8 - f9 + f10 - f11 + f12 - f13 + f14 - f15) + fx / T(2)) * inv_ρ
        uy[i,j,k] = ((f4 - f5 + f8 + f9 - f10 - f11 + f16 - f17 + f18 - f19) + fy / T(2)) * inv_ρ
        uz[i,j,k] = ((f6 - f7 + f12 + f13 - f14 - f15 + f16 + f17 - f18 - f19) + fz / T(2)) * inv_ρ
    end
end

"""
    compute_macroscopic_forced_field_3d!(ρ, ux, uy, uz, f, Fx, Fy, Fz)

3D force-corrected macroscopic moments with PER-CELL force fields
`Fx, Fy, Fz` (standard Guo `u = (Σf·c + F/2)/ρ` readout). Field analogue of
`compute_macroscopic_forced_3d!`, consistent with `collide_guo_field_3d!`.
"""
function compute_macroscopic_forced_field_3d!(ρ, ux, uy, uz, f, Fx, Fy, Fz)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny, Nz = size(ρ)
    kernel! = compute_macroscopic_forced_field_3d_kernel!(backend)
    kernel!(ρ, ux, uy, uz, f, Fx, Fy, Fz; ndrange=(Nx, Ny, Nz))
end

# --- Pressure-based macroscopic (He-Chen-Zhang model) ---
#
# In the pressure-based formulation for two-phase flows:
#   p = cs² · Σf_q      (pressure, NOT density)
#   u = (Σf_q·e_q + F/2) / ρ(C)   (velocity from physical density)
#
# The physical density ρ(C) comes from the VOF field, not from distributions.
# This allows density ratios > 1000 while keeping distributions O(1).

@kernel function compute_macroscopic_pressure_2d_kernel!(p, ux, uy,
                                                          @Const(f), @Const(C),
                                                          @Const(Fx), @Const(Fy),
                                                          ρ_l, ρ_g)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)
        f1 = f[i,j,1]; f2 = f[i,j,2]; f3 = f[i,j,3]
        f4 = f[i,j,4]; f5 = f[i,j,5]; f6 = f[i,j,6]
        f7 = f[i,j,7]; f8 = f[i,j,8]; f9 = f[i,j,9]

        # Pressure: p = cs² · Σf = (1/3) · Σf
        p_local = (f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9) / T(3)
        p[i,j] = p_local

        # Physical density from VOF (floor at ρ_l/100 for high density ratio stability)
        c = C[i,j]
        ρ_raw = c * ρ_l + (one(T) - c) * ρ_g
        ρ_local = max(ρ_raw, ρ_l * T(0.01))

        # Velocity: u = (j + F/2) / ρ(C)
        inv_ρ = one(T) / ρ_local
        jx = f2 - f4 + f6 - f7 - f8 + f9
        jy = f3 - f5 + f6 + f7 - f8 - f9
        ux[i,j] = (jx + Fx[i,j] / T(2)) * inv_ρ
        uy[i,j] = (jy + Fy[i,j] / T(2)) * inv_ρ
    end
end

"""
    compute_macroscopic_pressure_2d!(p, ux, uy, f, C, Fx, Fy; ρ_l=1.0, ρ_g=0.001)

Pressure-based macroscopic computation for two-phase flows (He-Chen-Zhang model).

- `p`:  pressure field (p = cs²·Σf)
- `ux, uy`: velocity (momentum / ρ(C), with half-force correction)
- `C`:  VOF field (liquid fraction)
- `Fx, Fy`: total body force (surface tension + axisym correction + gravity)
- `ρ_l, ρ_g`: physical densities of liquid and gas
"""
function compute_macroscopic_pressure_2d!(p, ux, uy, f, C, Fx, Fy;
                                          ρ_l=1.0, ρ_g=0.001)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(p)
    T = eltype(f)
    kernel! = compute_macroscopic_pressure_2d_kernel!(backend)
    kernel!(p, ux, uy, f, C, Fx, Fy, T(ρ_l), T(ρ_g); ndrange=(Nx, Ny))
end
