# Sphere Drag 3D Tutorial

This tutorial is the hands-on companion to the
[3D STL sphere-drag benchmark page](../benchmarks/sphere-drag-3d.md). It walks
through the **3D STL-obstacle workflow**: how to put an arbitrary triangulated
surface into a KrakeNN simulation, set up an external flow around it, and measure
the drag — using flow past a sphere at `Re = 20` as the worked example.

```@contents
Pages = ["users/tutorials/sphere-drag-3d.md"]
Depth = 2
```

## The problem

A sphere held fixed in a uniform stream. We measure its **drag coefficient**
`C_d = 2 F_x / (u_in² A)`, with `A` the projected frontal area (the `y`–`z`
silhouette of the solid), and compare to the classic Clift, Grace & Weber (1978)
free-stream correlation `C_d = (24/Re)(1 + 0.15 Re^0.687)`, which gives
`C_d ≈ 2.61` at `Re = 20`.

The key new ingredient versus the cavity tutorials is **geometry from an STL file**.
Instead of an implicit inequality, the obstacle is a triangle mesh, voxelised onto
the lattice, with an **interpolated bounce-back** (`wall=libb`) wall treatment so
the curved surface and the cut-link momentum exchange (which gives the drag) are
resolved sub-cell.

![Sphere-drag boundary conditions, shown on the 3D duct mid-plane.  The west boundary is a uniform velocity inlet (cool arrows in) and the east boundary a constant-pressure outlet (ρ = 1, cool arrows out); the north and south duct walls are no-slip (hatched); the sphere is an immersed STL obstacle with interpolated bounce-back (libb).](sphere-drag-3d-bc.svg)

## The `.krk` file

The 3D STL drag case (`examples/geometry_stl/sphere_stl_3d_drag.krk`) is:

```
# 3D sphere drag (raw LU, Re = 20)
Simulation mgeo7_sphere_drag D3Q19

Domain L = 120 x 60 x 60  N = 120 x 60 x 60
Physics nu = 0.032

Obstacle sph wall=libb stl(file = "examples/geometry_stl/sphere_drag.stl")

Boundary west  velocity(ux = 0.04, uy = 0)
Boundary east  pressure(rho = 1.0)
Boundary south wall
Boundary north wall

Run 10000 steps
```

Block by block:

- **`Simulation ... D3Q19`** — a 3D case on the D3Q19 stencil.
- **`Domain L = 120 x 60 x 60  N = 120 x 60 x 60`** — a `120×60×60` channel (here
  `L = N`, so coordinates are lattice units).
- **`Physics nu = 0.032`** — chosen so `ν = u_in · (2R) / Re`; with `u_in = 0.04`,
  sphere radius `R = 8`, this is `Re = 20`.
- **`Obstacle sph wall=libb stl(file = "...sphere_drag.stl")`** — the STL geometry.
  The `wall=libb` selector turns on interpolated bounce-back, which is what makes
  the cut-link drag integrator on the curved surface accurate.
- **`Boundary west velocity(...)`** — uniform inflow; **`east pressure(rho = 1.0)`**
  — a constant-pressure outlet; the `north`/`south` walls bound the duct.

The STL obstacle has optional placement parameters — `scale`, `translate=[x,y,z]`,
`z_slice` — documented in the [KRK reference](../krk-reference.md).
For example, `sphere_stl_3d_lu.krk` uses `scale = 20.0` to enlarge a small bundled
mesh.

### Physical units instead of raw LU

If you prefer to specify the geometry in millimetres and a Reynolds number rather
than hand-tuning `nu` and the inflow speed, add a `Units { ... }` block and let the
bridge compute them. The bundled `sphere_stl_3d_mm.krk` does exactly this:

```
Units { length = mm  L_ref = 0.2  R_LU = 4  Re = 1.0  scaling = acoustic }
Physics nu = auto

Obstacle sph wall=libb stl(file = "examples/geometry_stl/sphere.stl")
Boundary west velocity(ux = u_LU, uy = 0)
```

The bridge resolves the `0.2 mm` reference length to `R_LU = 4` cells, solves for
the LU viscosity (injected as `Physics nu`) and inflow speed (the variable
`u_LU`), and applies the matching STL scale. See the
[KRK reference `Units` block](../krk-reference.md).

## Running it

```julia
using KrakeNN
run_simulation("examples/geometry_stl/sphere_stl_3d_drag.krk")
```

3D drag runs are GPU work. The benchmark uses CUDA Float64 on Aqua for the
production sweep; a local CPU run of the `.krk` is a parse + voxelise smoke check
rather than a converged drag measurement.

There is also a **self-consistency twin** that runs locally on a GPU and checks the
STL drag path against the validated analytic-sphere driver:

```bash
julia --project=. test/test_sphere_stl_drag_krk.jl
```

It reproduces the analytic-sphere driver `run_sphere_libb_3d` (itself validated
against Clift) to **0.4 %** at matched lattice registration — the STL voxeliser and
the analytic q-wall are placed on the *same* lattice obstacle for the comparison.

## Expected result

Confined drag is sensitive to the lateral blockage `D/W` (sphere diameter over duct
width): the walls add drag, so `C_d` is higher than the free-stream value and falls
as the walls recede. The benchmark page runs a blockage sweep at fixed resolution
`R = 16` and extrapolates to the unbounded limit:

| Blockage `D/W` | `C_d`  |
|---------------:|-------:|
| 20 %           | 5.438  |
| 14.3 %         | 4.470  |
| 10 %           | 3.830  |
| 8 %            | 3.582  |
| 6 %            | 3.381  |
| **`D/W → 0` (quadratic LSQ)** | **2.84** |
| **Clift 1978 free-stream**    | **2.61** |

The extrapolated free-stream `C_d ≈ 2.84` is **+8.9 % vs Clift's 2.61**, with the
residual understood as finite lattice resolution (refining `R = 8 → 16` already
moved `C_d` −2.5 % toward the reference). The benchmark page details the
interpretation and the reproduction commands.

## Where to go next

- The [3D STL sphere-drag benchmark](../benchmarks/sphere-drag-3d.md) — the full
  blockage-convergence table, the residual interpretation, and the Aqua/PBS
  reproduction recipe.
- The [KRK reference `Obstacle` block and `Units` block](../krk-reference.md) — the
  STL and physical-units grammar.
- The other `examples/geometry_stl/*.krk` cases — 2D cylinder cross-sections,
  raw-LU and physical-units twins.
