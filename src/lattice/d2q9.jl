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

const _D2Q9_W = SVector{9, Float64}(
    4.0/9.0,                                     # rest
    1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,         # axis-aligned
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0      # diagonals
)

const _D2Q9_CX = SVector{9, Int32}(0, 1, 0, -1,  0, 1, -1, -1,  1)
const _D2Q9_CY = SVector{9, Int32}(0, 0, 1,  0, -1, 1,  1, -1, -1)

# opposite(q) gives the index of the velocity pointing in the opposite direction
const _D2Q9_OPP = SVector{9, Int32}(1, 4, 5, 2, 3, 8, 9, 6, 7)

weights(::D2Q9) = _D2Q9_W
"""
    velocities_x(::D2Q9)

Public function in the lattice trait API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.velocities_x)
```
"""
velocities_x(::D2Q9) = _D2Q9_CX
"""
    velocities_y(::D2Q9)

Public function in the lattice trait API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.velocities_y)
```
"""
velocities_y(::D2Q9) = _D2Q9_CY
opposite(::D2Q9) = _D2Q9_OPP

@inline function equilibrium(::D2Q9, ρ::T, ux::T, uy::T, q::Int) where T
    w = T(_D2Q9_W[q])
    cx = T(_D2Q9_CX[q])
    cy = T(_D2Q9_CY[q])
    cu = cx * ux + cy * uy
    usq = ux * ux + uy * uy
    return w * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
