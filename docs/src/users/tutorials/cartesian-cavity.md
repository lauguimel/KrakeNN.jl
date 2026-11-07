# Cartesian Cavity Tutorial

This tutorial walks through the **lid-driven cavity** — the canonical
incompressible-flow test case — end to end: the physics, the `.krk` file, how to
run it, and what result to expect. The quantitative validation against the Ghia,
Ghia & Shin (1982) reference lives on the
[Cartesian cavity benchmark page](../benchmarks/cartesian-cavity.md); this page is
the hands-on companion to it.

```@contents
Pages = ["users/tutorials/cartesian-cavity.md"]
Depth = 2
```

## The problem

A square box of fluid, closed on all four sides. The **top wall (lid) slides** at a
constant tangential speed; the other three walls are stationary no-slip walls. The
moving lid drags fluid along, a primary recirculating vortex fills the cavity, and
weaker secondary vortices form in the bottom corners. The flow is steady and
laminar at the Reynolds numbers used here.

The single governing parameter is the Reynolds number `Re = U·L/ν`, with `U` the
lid speed and `L` the cavity side. We run `Re = 100`, `400`, `1000`. In lattice
units we fix the lid speed at `u_lid = 0.1` (diffusive scaling) and choose the
viscosity from `ν = u_lid · N / Re`.

![Lid-driven cavity boundary conditions.  The north wall is a moving lid imposed via a Zou–He velocity BC (grey dashed baseline with red velocity arrows); the south, east and west walls are no-slip half-way bounce-back walls (hatched).](cartesian-cavity-bc.svg)

## The `.krk` file

The `Re = 100` case (`examples/cavity.krk`) is as compact as it gets:

```
# Lid-driven cavity flow at Re = 100
# Validation: Ghia et al. (1982)

Simulation cavity D2Q9
Domain  L = 1.0 x 1.0  N = 128 x 128
Physics nu = 0.128

Boundary north velocity(ux = 0.1, uy = 0)
Boundary south wall
Boundary east  wall
Boundary west  wall

Run 60000 steps
Output vtk every 10000 [rho, ux, uy]
```

Reading it block by block:

- **`Simulation cavity D2Q9`** — a 2D case on the D2Q9 stencil.
- **`Domain L = 1.0 x 1.0  N = 128 x 128`** — a unit square at `128²` resolution.
- **`Physics nu = 0.128`** — the LU viscosity. With `u_lid = 0.1` and `N = 128`,
  `ν = 0.1 · 128 / 100 = 0.128`, i.e. `Re = 100`.
- **`Boundary north velocity(ux = 0.1, uy = 0)`** — the moving lid, imposed via a
  Zou–He velocity BC on the top node row.
- **`Boundary {south,east,west} wall`** — no-slip half-way bounce-back on the other
  three walls.
- **`Run 60000 steps`** — iterate to (near) steady state.
- **`Output vtk every 10000 [rho, ux, uy]`** — dump density and velocity every
  10 000 steps into `output/`.

### Changing the Reynolds number

To run `Re = 400` or `Re = 1000`, raise the resolution to `256²` and recompute the
viscosity so that `ν = u_lid · N / Re`:

| Re   | N      | ν (LU)  |
|------|--------|---------|
| 100  | 128²   | 0.128   |
| 400  | 256²   | 0.064   |
| 1000 | 256²   | 0.0256  |

These are exactly the runs tabulated on the
[benchmark page](../benchmarks/cartesian-cavity.md), which also lists the number of
steps each needs to reach the steady-state residual (`25 500`, `120 000`,
`500 000`).

## Running it

```julia
using KrakeNN
run_simulation("examples/cavity.krk")
```

The runner parses the file, builds a D2Q9 lattice with the BGK collision, applies
the BCs, iterates 60 000 steps, and writes `output/cavity_*.vtr`. By default it
uses the available accelerated backend (Metal Float32 on an Apple M-series; CUDA
on NVIDIA). Open the `.vtr` snapshots in ParaView to see the primary vortex.

## Expected result

At steady state the velocity field shows one large primary vortex whose centre sits
above and to the right of the cavity centre at `Re = 100`, migrating toward the
centre as `Re` grows, with secondary corner vortices appearing at `Re = 1000`.

Quantitatively, the standard check is the **u-velocity along the vertical
centreline (x = 0.5)** and the **v-velocity along the horizontal centreline
(y = 0.5)**, compared to Ghia 1982. As reported on the benchmark page, KrakeNN (BGK)
lands at a relative-L2 error on the u-centreline of **0.47 % / 0.41 % / 1.05 %** at
`Re = 100 / 400 / 1000`, clearing the strict < 1 % gate at `Re = 100` and `400`
and effectively at `Re = 1000`. OpenFOAM `icoFoam` independently validates the same
Ghia reference (0.49 % / 0.23 % / 0.46 %).

A subtlety worth knowing (and the reason the benchmark page is so explicit about
it): the moving lid is imposed *on* the node row (Zou–He) while the other walls sit
half a cell *outside* the last node (bounce-back), so the physical cavity height
along the lid axis is `H = (N − 0.5)·Δ`. Mapping the u-centreline with the correct
wall-aware coordinate is what brings the error under 1 %; a naive `(j − 0.5)/N`
mapping injects a spurious first-order error. The benchmark page documents this in
full.

## Where to go next

- The [Cartesian cavity benchmark](../benchmarks/cartesian-cavity.md) — full error
  tables, the wall-aware-coordinate discussion, the OpenFOAM cross-check, and the
  3D cubic-cavity extension.
- The [KRK reference](../krk-reference.md) — every block and keyword used above.
- The [thermal natural-convection tutorial](thermal-natural-convection.md) — the
  same cavity geometry with a temperature field and buoyancy.
