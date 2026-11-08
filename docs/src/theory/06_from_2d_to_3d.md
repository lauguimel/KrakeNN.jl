```@meta
EditURL = "06_from_2d_to_3d.jl"
```

# From 2D to 3D: the D3Q19 Lattice

Moving from two to three dimensions in LBM is conceptually straightforward:
the algorithm (collide, stream, apply BCs) stays exactly the same, only the
lattice changes. The most common choice is **D3Q19**, which balances accuracy
and memory cost [Qian, d'Humieres & Lallemand (1992)](@cite qian1992lattice).

## Why D3Q19?

Several 3D lattices exist:

| Lattice | Velocities | Memory per node | Isotropy |
|:--------|:----------:|:---------------:|:--------:|
| D3Q15   |     15     |    Low          | 4th order |
| D3Q19   |     19     |    Medium       | 4th order |
| D3Q27   |     27     |    High         | 6th order (full cube) |

D3Q19 is the sweet spot: it has enough velocity vectors to achieve fourth-order
isotropy (same as D3Q15) while being more numerically stable, and it uses 30%
less memory than D3Q27.

## The D3Q19 velocity set

The 19 velocities consist of:
- 1 rest vector: ``(0, 0, 0)``
- 6 face neighbours: ``(\pm 1, 0, 0)``, ``(0, \pm 1, 0)``, ``(0, 0, \pm 1)``
- 12 edge neighbours: ``(\pm 1, \pm 1, 0)``, ``(\pm 1, 0, \pm 1)``,
  ``(0, \pm 1, \pm 1)``

The associated weights are:

```math
w_q = \begin{cases}
  1/3  & \text{rest} \quad (1 \text{ vector}) \\
  1/18 & \text{face} \quad (6 \text{ vectors}) \\
  1/36 & \text{edge} \quad (12 \text{ vectors})
\end{cases}
```

The lattice speed of sound remains ``c_s^2 = 1/3``, identical to D2Q9.

## Equilibrium distribution

The equilibrium formula generalises naturally to 3D:

```math
f_q^{\mathrm{eq}} = w_q \, \rho \left[
    1
    + \frac{\mathbf{e}_q \cdot \mathbf{u}}{c_s^2}
    + \frac{(\mathbf{e}_q \cdot \mathbf{u})^2}{2 c_s^4}
    - \frac{|\mathbf{u}|^2}{2 c_s^2}
\right]
```

The only difference is that ``\mathbf{u} = (u_x, u_y, u_z)`` is now
three-dimensional, and the dot products involve all three components.

## Algorithm changes

Compared to the 2D algorithm, the 3D version:

- Loops over 19 directions instead of 9.
- Uses 3D arrays `f[Nx, Ny, Nz, 19]` instead of `f[Nx, Ny, 9]`.
- Streaming accesses 18 neighbours (6 face + 12 edge) instead of 8.

The collision operator remains purely local and unchanged in structure.
The viscosity--relaxation relation ``\nu = c_s^2(\tau - 1/2)`` is identical.

## Memory considerations

For a ``128^3`` grid, a D3Q19 simulation stores:

```math
128^3 \times 19 \times 8 \;\text{bytes (Float64)} \approx 300 \;\text{MB}
```

This must fit in GPU memory along with macroscopic fields (``\rho, u_x, u_y, u_z``).
For large 3D problems, Float32 can halve the memory cost with minimal
accuracy loss (LBM is already a second-order method).

## KrakeNN D3Q19 interface

```julia
using KrakeNN

lattice3d = D3Q19()

@show lattice_dim(lattice3d)  # 3
@show lattice_q(lattice3d)    # 19
@show cs2(lattice3d)          # 1/3

# Weights: 1 rest + 6 face + 12 edge
w = weights(lattice3d)
@show w[1]   # 1/3  (rest)
@show w[2]   # 1/18 (face)
@show w[8]   # 1/36 (edge)

# Velocity components
cx = velocities_x(lattice3d)
cy = velocities_y(lattice3d)
cz = velocities_z(lattice3d)

# Print all 19 velocity vectors
for q in 1:19
    println("q=$q: e = ($(cx[q]), $(cy[q]), $(cz[q]))  w = $(w[q])")
end
```

The 3D kernels (`stream_3d!`, `collide_3d!`, `compute_macroscopic_3d!`)
follow the same conventions as their 2D counterparts, with an additional
spatial dimension in the `ndrange` parameter:

```julia
collide_3d!(f_out, f_in, ρ, ux, uy, uz, ω, lattice; ndrange=(Nx, Ny, Nz))
stream_3d!(f_out, f_in, lattice; ndrange=(Nx, Ny, Nz))
```

## See in action

- [Lid-driven cavity (2D & 3D)](../examples/04_cavity_2d.md) — D3Q19 reference
  case on GPU.

