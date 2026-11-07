"""
    D3Q19 <: AbstractLattice{3, 19}

Standard 3D lattice with 19 discrete velocities.

Velocity ordering:
     0: ( 0, 0, 0)  — rest
    1–6: axis-aligned (±x, ±y, ±z)
   7–18: edge-aligned (±x±y, ±x±z, ±y±z)
"""
struct D3Q19 <: AbstractLattice{3, 19} end

const _D3Q19_W = SVector{19, Float64}(
    1.0/3.0,                                                             # rest
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,       # axis
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,                            # edges xy
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,                            # edges xz
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0                             # edges yz
)

#                          0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18
const _D3Q19_CX = SVector{19, Int32}(0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0)
const _D3Q19_CY = SVector{19, Int32}(0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1)
const _D3Q19_CZ = SVector{19, Int32}(0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1)

# opposite directions: q -> opp(q)
const _D3Q19_OPP = SVector{19, Int32}(
    1,   # 0 -> 0
    3,   # 1 (+x) -> 2 (-x) (index 3)
    2,   # 2 (-x) -> 1 (+x) (index 2)
    5,   # 3 (+y) -> 4 (-y) (index 5)
    4,   # 4 (-y) -> 3 (+y) (index 4)
    7,   # 5 (+z) -> 6 (-z) (index 7)
    6,   # 6 (-z) -> 5 (+z) (index 6)
    11,  # 7 (+x,+y) -> 10 (-x,-y) (index 11)
    10,  # 8 (-x,+y) -> 9 (+x,-y) (index 10)
    9,   # 9 (+x,-y) -> 8 (-x,+y) (index 9)
    8,   # 10 (-x,-y) -> 7 (+x,+y) (index 8)
    15,  # 11 (+x,+z) -> 14 (-x,-z) (index 15)
    14,  # 12 (-x,+z) -> 13 (+x,-z) (index 14)
    13,  # 13 (+x,-z) -> 12 (-x,+z) (index 13)
    12,  # 14 (-x,-z) -> 11 (+x,+z) (index 12)
    19,  # 15 (+y,+z) -> 18 (-y,-z) (index 19)
    18,  # 16 (-y,+z) -> 17 (+y,-z) (index 18)
    17,  # 17 (+y,-z) -> 16 (-y,+z) (index 17)
    16   # 18 (-y,-z) -> 15 (+y,+z) (index 16)
)

weights(::D3Q19) = _D3Q19_W
velocities_x(::D3Q19) = _D3Q19_CX
velocities_y(::D3Q19) = _D3Q19_CY
"""
    velocities_z(::D3Q19)

Public function in the lattice trait API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.velocities_z)
```
"""
velocities_z(::D3Q19) = _D3Q19_CZ
opposite(::D3Q19) = _D3Q19_OPP

@inline function equilibrium(::D3Q19, ρ::T, ux::T, uy::T, uz::T, q::Int) where T
    w = T(_D3Q19_W[q])
    cx = T(_D3Q19_CX[q])
    cy = T(_D3Q19_CY[q])
    cz = T(_D3Q19_CZ[q])
    cu = cx * ux + cy * uy + cz * uz
    usq = ux * ux + uy * uy + uz * uz
    return w * ρ * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end
