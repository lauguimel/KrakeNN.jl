# Streaming

Streaming propagates post-collision distributions to their
neighbours along the discrete velocities. KrakeNN provides standard
streaming for 2D/3D, as well as periodic and axisymmetric variants
used by channel, Taylor-Green, and pipe drivers. GPU-safe index
clamping is handled inside the kernels (see the LBM patterns memo
on the `ifelse` fix for `stream_2d!`).


## Quick reference

| Symbol | Purpose |
|--------|---------|
| `stream_2d!` | Standard streaming step — 2D |
| `stream_3d!` | Standard streaming step — 3D |
| `stream_periodic_x_wall_y_2d!` | Streaming with periodic-x / wall-y — 2D |
| `stream_fully_periodic_2d!` | Streaming with doubly periodic BCs — 2D |
| `stream_periodic_x_axisym_2d!` | Streaming for axisymmetric runs with periodic x |
| `stream_axisym_inlet_2d!` | Streaming for axisymmetric inlet problems |

## Details

### `stream_2d!`

**Source:** `src/kernels/collide_stream_2d.jl`

```julia
function stream_2d!(f_out, f_in, Nx, Ny; sync=true)
    backend = KernelAbstractions.get_backend(f_in)
    kernel! = stream_2d_kernel!(backend)
    kernel!(f_out, f_in, Nx, Ny; ndrange=(Nx, Ny))
    sync && KernelAbstractions.synchronize(backend)
end
```


### `stream_3d!`

**Source:** `src/kernels/collide_stream_3d.jl`

```julia
function stream_3d!(f_out, f_in, Nx, Ny, Nz)
    backend = KernelAbstractions.get_backend(f_in)
    kernel! = stream_3d_kernel!(backend)
    kernel!(f_out, f_in, Nx, Ny, Nz; ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end
```


### `stream_periodic_x_wall_y_2d!`

**Source:** `src/kernels/stream_periodic_2d.jl`

```julia
function stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
    backend = KernelAbstractions.get_backend(f_in)
    kernel! = stream_periodic_x_wall_y_2d_kernel!(backend)
    kernel!(f_out, f_in, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
```


### `stream_fully_periodic_2d!`

**Source:** `src/kernels/stream_periodic_2d.jl`

```julia
function stream_fully_periodic_2d!(f_out, f_in, Nx, Ny)
    backend = KernelAbstractions.get_backend(f_in)
    kernel! = stream_fully_periodic_2d_kernel!(backend)
    kernel!(f_out, f_in, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
```


