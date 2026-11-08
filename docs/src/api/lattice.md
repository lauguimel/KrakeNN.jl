# Lattice

The `lattice` module defines the discrete-velocity sets used throughout
KrakeNN — `D2Q9` for 2D flows and `D3Q19` for 3D flows — together with
their weights, opposite-direction tables, and the equilibrium
distribution `f_i^eq(ρ, u)`. Everything downstream (collision,
streaming, boundary conditions) is written in terms of these small
traits, so swapping lattices is a single type-parameter change.


## Quick reference

| Symbol | Purpose |
|--------|---------|
| `AbstractLattice` | Abstract parent type for lattice structures |
| `D2Q9` | 2D nine-velocity lattice singleton |
| `D3Q19` | 3D nineteen-velocity lattice singleton |
| `lattice_dim` | Spatial dimension (2 or 3) |
| `lattice_q` | Number of discrete velocities (9 or 19) |
| `weights` | Lattice weights wᵢ |
| `velocities_x` | Discrete velocity x-components |
| `velocities_y` | Discrete velocity y-components |
| `velocities_z` | Discrete velocity z-components |
| `opposite` | Opposite-direction table (for bounce-back) |
| `cs2` | Speed of sound squared (1/3 in lattice units) |
| `equilibrium` | Equilibrium distribution f_i^eq(ρ, u) |

## Details

### `D2Q9`

**Source:** `src/lattice/d2q9.jl`

```julia
"""
    D2Q9 <: AbstractLattice{2, 9}

Standard 2D lattice with 9 discrete velocities.

Velocity ordering:
    0: ( 0, 0)  — rest
    1: ( 1, 0)  — east
    2: ( 0, 1)  — north
    3: (-1, 0)  — west
    4: ( 0,-1)  — south
    5: ( 1, 1)  — NE
    6: (-1, 1)  — NW
    7: (-1,-1)  — SW
    8: ( 1,-1)  — SE
"""
struct D2Q9 <: AbstractLattice{2, 9} end
```


### `D3Q19`

**Source:** `src/lattice/d3q19.jl`

```julia
"""
    D3Q19 <: AbstractLattice{3, 19}

Standard 3D lattice with 19 discrete velocities.

Velocity ordering:
     0: ( 0, 0, 0)  — rest
    1–6: axis-aligned (±x, ±y, ±z)
   7–18: edge-aligned (±x±y, ±x±z, ±y±z)
"""
struct D3Q19 <: AbstractLattice{3, 19} end
```


### `equilibrium`

**Source:** `src/lattice/lattice.jl`

```julia
"""
    equilibrium(lattice, ρ, u, q) -> Float64

Compute equilibrium distribution for direction `q` given density `ρ`
and velocity vector `u`.

    f_eq = w_q · ρ · (1 + (c_q · u)/cs² + (c_q · u)²/(2·cs⁴) - u·u/(2·cs²))
"""
function equilibrium end
```


### `weights`

**Source:** `src/lattice/d2q9.jl`

```julia
weights(::D2Q9) = _D2Q9_W
```


### `opposite`

**Source:** `src/lattice/d2q9.jl`

```julia
opposite(::D2Q9) = _D2Q9_OPP
```


