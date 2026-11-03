# Thermal Natural Convection Tutorial

This tutorial adds heat to the cavity: a **differentially-heated square cavity**,
where a hot wall and a cold wall drive a buoyant circulation. It is the hands-on
companion to the
[thermal natural-convection benchmark page](../benchmarks/thermal-natural-convection.md),
which carries the quantitative validation against the de Vahl Davis (1983)
reference.

```@contents
Pages = ["users/tutorials/thermal-natural-convection.md"]
Depth = 2
```

## The problem

A square cavity with a **hot left wall** (`T = 1`) and a **cold right wall**
(`T = 0`); the top and bottom walls are **adiabatic** (insulated, zero
temperature-gradient). All four walls are no-slip. Gravity acts downward. Fluid
near the hot wall expands, rises, crosses the top, cools at the cold wall, and
sinks — a single convection cell. The dimensionless driver is the Rayleigh number
`Ra` (with Prandtl number `Pr = 0.71` for air); we run `Ra = 10³, 10⁴, 10⁵`.

Kraken solves this with a **double-distribution thermal LBM**: the usual D2Q9 BGK
distribution carries momentum, a second D2Q9 distribution advects temperature, and
buoyancy is fed back into the flow through a Boussinesq body force
`F = ρ β g (T − T₀)`. The driver is `run_natural_convection_2d`, selected
automatically when the `thermal` module is active.

![Differentially-heated cavity boundary conditions.  The west wall is a hot Dirichlet wall (T = 1, red hatched no-slip) and the east wall a cold Dirichlet wall (T = 0, blue hatched no-slip); the north and south walls are adiabatic no-slip walls (∂ₙT = 0, grey hatched).  Gravity acts downward (grey arrow).](thermal-natural-convection-bc.svg)

## The `.krk` file

The thermal module is activated with `Module thermal`; the hot/cold walls are
plain `wall` boundaries carrying a `T = ...` Dirichlet value, and the insulated
walls are walls with no `T` (zero-gradient by default). A self-contained
differentially-heated cavity reads:

```
# Differentially-heated square cavity (natural convection)
# Validation: de Vahl Davis (1983), Pr = 0.71

Simulation natural_convection D2Q9
Domain L = 1.0 x 1.0  N = 192 x 192
Physics nu = 0.02 Pr = 0.71 Ra = 1e3

Module thermal

Boundary west  wall T = 1.0     # hot
Boundary east  wall T = 0.0     # cold
Boundary south wall             # adiabatic (no T → zero-gradient)
Boundary north wall             # adiabatic

Run 270000 steps
Output vtk every 30000 [rho, ux, uy, T]
```

The `T = ...` syntax on a `wall` is exactly the pattern used by the bundled
`examples/heat_conduction.krk` case (hot west wall, cold east wall) — see the
[KRK reference](../krk-reference.md) for the boundary grammar. The
thermal coefficients live in `Physics`: `Pr` and `Ra` set the regime; the driver
derives the thermal diffusivity and the Boussinesq force from them.

### Choosing resolution and precision per Rayleigh number

The benchmark page documents a real finding: the thermal boundary layer thins as
`Ra^(1/4)`, so the mesh needed to clear the 1 % Nusselt gate grows with `Ra`, and
**`Ra = 10³` must be run in Float64** (its buoyancy force `β g ∝ 1/N³ ≈ 10⁻⁹` LU
underflows Float32 for `N ≥ 320`). The validated settings are:

| Ra   | Mesh  | Backend     | Steps to steady |
|------|-------|-------------|-----------------|
| 10³  | 192²  | CPU F64     | 270 000         |
| 10⁴  | 320²  | Metal F32   | 500 000         |
| 10⁵  | 384²  | Metal F32   | 2 160 000       |

Match the `Domain N`, the backend, and the `Run` length to the row you want to
reproduce.

!!! note "Settings measured before the wall change"
    This table was measured when the hot and cold walls sat on the first and last
    nodes, one lattice spacing closer together than the no-slip walls (issue #97).
    They now sit half a spacing outside those nodes, like the no-slip walls, and the
    first-order error this caused is gone: on CPU Float64 at `N = 64`, `Nu` is
    within 0.13 % (`Ra = 10³`) and 0.69 % (`Ra = 10⁴`) of de Vahl Davis. The
    meshes above have not been re-measured.

## Running it

```julia
using Kraken
run_simulation("natural_convection.krk")
```

The runner sees `Module thermal`, dispatches to `run_natural_convection_2d`,
iterates the coupled flow+temperature solve, and writes `output/*.vtr` snapshots
carrying `T` alongside the velocity. Open them in ParaView and colour by `T` to see
the thermal plume and the convection cell.

!!! note "Reproducing exact benchmark numbers"
    The benchmark page reproduces its table with the driver settings above (CPU
    Float64 at `Ra = 10³`; Metal Float32 at `Ra = 10⁴/10⁵`). Run the longer step
    counts and the matching backend to land on the published Nusselt values.

## A qualitative variant: Rayleigh–Bénard

For a hot-**bottom** / cold-**top** cell with periodic sides — Rayleigh–Bénard
convection — Kraken ships a ready preset and example, `examples/rayleigh_benard.krk`:

```
Preset rayleigh_benard_2d
Run 30000 steps
Output vtk every 5000 [rho, ux, uy, T]
```

```julia
run_simulation("examples/rayleigh_benard.krk")
```

This develops the expected convection rolls above the critical `Ra_c ≈ 1708`. As
the benchmark page states, the Rayleigh–Bénard demo is **qualitative only** — no
quantitative roll-cell reference is asserted.

## Expected result

At steady state the differentially-heated cavity shows a single clockwise (hot-left)
convection cell, with thin thermal boundary layers hugging the hot and cold walls
that sharpen as `Ra` increases. The headline metric is the average **Nusselt
number** `Nu` on the hot wall (the dimensionless wall heat flux). As reported on the
[benchmark page](../benchmarks/thermal-natural-convection.md), Kraken lands at a
Nu error of **+0.79 % / +0.93 % / +0.79 %** at `Ra = 10³ / 10⁴ / 10⁵` (measured
with the former on-node thermal walls; with the current half-way walls, `N = 64`
gives +0.13 % and +0.69 % at `Ra = 10³` and `10⁴`) versus de
Vahl Davis 1983, with both mid-plane velocity extrema under 1 %. OpenFOAM
`buoyantBoussinesqSimpleFoam` independently corroborates the reference at the
flanking `Ra = 10³` (−0.56 %) and `Ra = 10⁵` (−0.60 %).

The same double-distribution method extends to a **3D cubic cavity** (`D3Q19` flow
+ `D3Q19` temperature, driver `run_natural_convection_3d`); the benchmark page
tabulates its monotone mesh convergence toward the Tric et al. (2000) spectral
reference.

## Where to go next

- The [thermal natural-convection benchmark](../benchmarks/thermal-natural-convection.md)
  — full Nu/velocity error tables, the per-`Ra` precision recipe, the OpenFOAM
  cross-check, and the 3D cubic-cavity extension.
- The [Cartesian cavity tutorial](cartesian-cavity.md) — the same geometry without
  heat.
- The [KRK reference](../krk-reference.md) — the `Module`, `Physics`, and thermal
  `Boundary` grammar.
