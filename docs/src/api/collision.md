# Collision operators

Collision kernels relax the distribution `f` toward its local
equilibrium. KrakeNN exports a plain BGK operator, a Guo forced
variant (for body forces without spurious terms), MRT for
higher-Reynolds cavities, and thermal/Boussinesq operators used by
the natural-convection drivers. Every kernel is written to run on
CPU via `@threads` and on GPU via KernelAbstractions with the same
source.


## Quick reference

| Symbol | Purpose |
|--------|---------|
| `collide_2d!` | Plain BGK collision — 2D |
| `collide_3d!` | Plain BGK collision — 3D |
| `collide_guo_2d!` | Guo body-force collision — 2D |
| `collide_guo_field_2d!` | Guo forcing collision, per-cell force field — 2D |
| `collide_guo_3d!` | Guo body-force collision — 3D |
| `collide_guo_field_3d!` | Guo forcing collision, per-cell force field — 3D |
| `collide_thermal_2d!` | Advection-diffusion collision for temperature DDF — 2D |
| `collide_boussinesq_2d!` | Boussinesq thermal collision — 2D |
| `collide_boussinesq_vt_2d!` | Boussinesq collision (velocity/temperature) — 2D |
| `collide_boussinesq_vt_modified_2d!` | Boussinesq collision (velocity/temperature, modified) — 2D |
| `collide_axisymmetric_2d!` | Axisymmetric BGK collision — 2D |
| `collide_li_axisym_2d!` | Li et al. axisymmetric BGK — 2D |
| `collide_mrt_2d!` | Multiple-Relaxation-Time collision — 2D |

## Details

### `collide_2d!`

**Source:** `src/kernels/collide_stream_2d.jl`

```julia
function collide_2d!(f, is_solid, ω; sync=true)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_2d_kernel!(backend)
    kernel!(f, is_solid, ω; ndrange=(Nx, Ny))
    sync && KernelAbstractions.synchronize(backend)
end
```


### `collide_guo_2d!`

**Source:** `src/kernels/collide_guo_2d.jl`

```julia
function collide_guo_2d!(f, is_solid, ω, Fx, Fy)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_guo_2d_kernel!(backend)
    kernel!(f, is_solid, ω, Fx, Fy; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
```


### `collide_mrt_2d!`

**Source:** `src/kernels/collide_mrt_2d.jl`

```julia
"""
    collide_mrt_2d!(f, is_solid, ν; s_e=1.4, s_eps=1.4, s_q=1.2)

MRT collision for D2Q9 (Lallemand & Luo, 2000).
The stress relaxation rate s_ν = 1/(3ν + 0.5) is computed from viscosity.
Other rates (s_e, s_eps, s_q) can be tuned for stability (default values from literature).
"""
function collide_mrt_2d!(f, is_solid, ν; s_e=1.4, s_eps=1.4, s_q=1.2)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    T = eltype(f)
    s_nu = T(1.0 / (3.0 * ν + 0.5))
    kernel! = collide_mrt_2d_kernel!(backend)
    kernel!(f, is_solid, T(s_e), T(s_eps), T(s_q), s_nu; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
```


### `collide_thermal_2d!`

**Source:** `src/kernels/thermal_2d.jl`

```julia
function collide_thermal_2d!(g, ux, uy, ω_T)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(g, 1), size(g, 2)
    kernel! = collide_thermal_2d_kernel!(backend)
    kernel!(g, ux, uy, ω_T; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
```


### `collide_boussinesq_2d!`

**Source:** `src/kernels/thermal_2d.jl`

```julia
function collide_boussinesq_2d!(f, Temp, is_solid, ω, β_g, T_ref)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    T = eltype(f)
    kernel! = collide_boussinesq_2d_kernel!(backend)
    kernel!(f, Temp, is_solid, ω, T(β_g), T(T_ref); ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
```


