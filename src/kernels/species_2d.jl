using KernelAbstractions

# --- Species transport D2Q9: passive scalar via DDF ---
#
# Concentration C tracked by population h[i,j,q] on D2Q9.
# Equilibrium: h_eq_q = w_q * C * (1 + c_q·u / cs²)
# Collision:   h_q = h_q - ω_D * (h_q - h_eq_q)
# where ω_D = 1 / (3D + 0.5), D = mass diffusivity
# Macroscopic: C = Σ h_q

@kernel function collide_species_2d_kernel!(h, @Const(ux), @Const(uy), ω_D)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(h)
        h1=h[i,j,1]; h2=h[i,j,2]; h3=h[i,j,3]; h4=h[i,j,4]
        h5=h[i,j,5]; h6=h[i,j,6]; h7=h[i,j,7]; h8=h[i,j,8]; h9=h[i,j,9]

        C = h1 + h2 + h3 + h4 + h5 + h6 + h7 + h8 + h9
        u_x = ux[i,j]
        u_y = uy[i,j]
        t3 = T(3)

        heq = T(4.0/9.0) * C
        h[i,j,1] = h1 - ω_D * (h1 - heq)

        heq = T(1.0/9.0) * C * (one(T) + t3 * u_x)
        h[i,j,2] = h2 - ω_D * (h2 - heq)

        heq = T(1.0/9.0) * C * (one(T) + t3 * u_y)
        h[i,j,3] = h3 - ω_D * (h3 - heq)

        heq = T(1.0/9.0) * C * (one(T) - t3 * u_x)
        h[i,j,4] = h4 - ω_D * (h4 - heq)

        heq = T(1.0/9.0) * C * (one(T) - t3 * u_y)
        h[i,j,5] = h5 - ω_D * (h5 - heq)

        heq = T(1.0/36.0) * C * (one(T) + t3 * (u_x + u_y))
        h[i,j,6] = h6 - ω_D * (h6 - heq)

        heq = T(1.0/36.0) * C * (one(T) + t3 * (-u_x + u_y))
        h[i,j,7] = h7 - ω_D * (h7 - heq)

        heq = T(1.0/36.0) * C * (one(T) + t3 * (-u_x - u_y))
        h[i,j,8] = h8 - ω_D * (h8 - heq)

        heq = T(1.0/36.0) * C * (one(T) + t3 * (u_x - u_y))
        h[i,j,9] = h9 - ω_D * (h9 - heq)
    end
end

"""
    collide_species_2d!(h, ux, uy, ω_D)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_species_2d!)
```
"""
function collide_species_2d!(h, ux, uy, ω_D)
    backend = KernelAbstractions.get_backend(h)
    Nx, Ny = size(h, 1), size(h, 2)
    kernel! = collide_species_2d_kernel!(backend)
    kernel!(h, ux, uy, ω_D; ndrange=(Nx, Ny))
end

@kernel function compute_concentration_2d_kernel!(C, @Const(h))
    i, j = @index(Global, NTuple)
    @inbounds begin
        C[i,j] = h[i,j,1]+h[i,j,2]+h[i,j,3]+h[i,j,4]+h[i,j,5]+h[i,j,6]+h[i,j,7]+h[i,j,8]+h[i,j,9]
    end
end

"""
    compute_concentration_2d!(C, h)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.compute_concentration_2d!)
```
"""
function compute_concentration_2d!(C, h)
    backend = KernelAbstractions.get_backend(h)
    Nx, Ny = size(C)
    kernel! = compute_concentration_2d_kernel!(backend)
    kernel!(C, h; ndrange=(Nx, Ny))
end

# --- Fixed concentration BCs (anti-bounce-back Dirichlet) ---

@kernel function apply_fixed_conc_south_2d_kernel!(h, C_wall)
    i = @index(Global)
    @inbounds begin
        T = eltype(h)
        h[i,1,3] = -h[i,1,5] + T(2)*T(1.0/9.0)*C_wall
        h[i,1,6] = -h[i,1,8] + T(2)*T(1.0/36.0)*C_wall
        h[i,1,7] = -h[i,1,9] + T(2)*T(1.0/36.0)*C_wall
    end
end

@kernel function apply_fixed_conc_north_2d_kernel!(h, C_wall, Ny)
    i = @index(Global)
    @inbounds begin
        T = eltype(h)
        h[i,Ny,5] = -h[i,Ny,3] + T(2)*T(1.0/9.0)*C_wall
        h[i,Ny,8] = -h[i,Ny,6] + T(2)*T(1.0/36.0)*C_wall
        h[i,Ny,9] = -h[i,Ny,7] + T(2)*T(1.0/36.0)*C_wall
    end
end

"""
    apply_fixed_conc_south_2d!(h, C_wall, Nx)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.apply_fixed_conc_south_2d!)
```
"""
function apply_fixed_conc_south_2d!(h, C_wall, Nx)
    backend = KernelAbstractions.get_backend(h)
    kernel! = apply_fixed_conc_south_2d_kernel!(backend)
    kernel!(h, eltype(h)(C_wall); ndrange=(Nx,))
end

"""
    apply_fixed_conc_north_2d!(h, C_wall, Nx, Ny)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.apply_fixed_conc_north_2d!)
```
"""
function apply_fixed_conc_north_2d!(h, C_wall, Nx, Ny)
    backend = KernelAbstractions.get_backend(h)
    kernel! = apply_fixed_conc_north_2d_kernel!(backend)
    kernel!(h, eltype(h)(C_wall), Ny; ndrange=(Nx,))
end
