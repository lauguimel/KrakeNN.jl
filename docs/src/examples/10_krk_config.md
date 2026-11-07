```@meta
EditURL = "10_krk_config.jl"
```

# Configuration Files (`.krk`)

```@raw html
<DownloadMenu :files="[{label:'cavity.krk',href:'/downloads/krk_config/cavity.krk'},{label:'poiseuille.krk',href:'/downloads/krk_config/poiseuille.krk'},{label:'cylinder.krk',href:'/downloads/krk_config/cylinder.krk'},{label:'couette.krk',href:'/downloads/krk_config/couette.krk'}]" />
```

**Concepts:** [Spatial boundary conditions](../theory/19_spatial_bcs.md) ·
`.krk` DSL parser, presets, helpers

**Validates against:** tutorial example — no quantitative validation target.
Each .krk in this tutorial is a working case covered by other example pages.

**Download:** [`cavity.krk`](../assets/krk/cavity.krk) (and the other .krk files
linked from their respective example pages)

**Hardware:** n/a (tutorial, cases run from other pages)

---

## Overview

KrakeNN can run simulations from plain-text **`.krk`** configuration files,
inspired by the [Gerris Flow Solver](http://gfs.sourceforge.net/).  No Julia
code is required: geometry, boundary conditions, physics, and output are all
defined in a readable, line-based format.

```
result = run_simulation("cavity.krk")
```

## Syntax reference

A `.krk` file is a sequence of **statements**, one per line (multi-line blocks
use `{ }`).  Comments start with `#`.

| Statement | Purpose | Example |
|-----------|---------|---------|
| `Simulation` | Name and lattice | `Simulation cavity D2Q9` |
| `Domain` | Physical size and resolution | `Domain L = 1 x 1 N = 128 x 128` |
| `Physics` | Material properties | `Physics nu = 0.1 Pr = 1.0` |
| `Define` | User variables | `Define U = 0.05` |
| `Obstacle` | Solid region (wall) | `Obstacle cyl { (x-2)^2+(y-1)^2 <= 0.5^2 }` |
| `Fluid` | Fluid region (rest is solid) | `Fluid pipe { y > 0.2 && y < 0.8 }` |
| `Boundary` | Boundary condition per face | `Boundary north velocity(ux = 0.1)` |
| `Initial` | Initial condition expressions | `Initial { ux = 0.01*sin(x) }` |
| `Run` | Number of timesteps | `Run 60000 steps` |
| `Output` | VTK output | `Output vtk every 1000 [rho, ux, uy]` |
| `Diagnostics` | CSV diagnostics | `Diagnostics every 100 [step, drag]` |

### Expressions

Any value can be a math expression using the variables `x`, `y`, `z` (spatial),
`t` (time), `Lx`, `Ly`, `Nx`, `Ny`, `dx`, `dy`, and any user-defined
variable from `Define`.  Available functions: `sin`, `cos`, `tan`, `exp`,
`log`, `sqrt`, `abs`, `tanh`, `min`, `max`, `pi`, etc.

### Geometry: `Obstacle` vs `Fluid`

| Directives | Default | Logic |
|------------|---------|-------|
| `Obstacle` only | all fluid | solid where condition = true |
| `Fluid` only | all solid | fluid where condition = true |
| `Fluid` + `Obstacle` | all solid | fluid region minus obstacles |

### Boundary conditions

- **`wall`** — half-way bounce-back (no-slip)
- **`velocity(ux = ..., uy = ...)`** — Zou--He velocity (scalar or spatial expression)
- **`pressure(rho = ...)`** — Zou--He pressure outlet
- **`periodic`** — `Boundary x periodic` makes both west/east periodic

Spatial BCs like `ux = 4*U*y*(H-y)/H^2` are evaluated per node.
Time-dependent BCs like `ux = 0.1*sin(2*pi*t/5000)` are re-evaluated each step.

---

## Example 1 — Lid-driven cavity

Download: [`cavity.krk`](../assets/krk/cavity.krk)

```krk
Simulation cavity D2Q9
Domain  L = 1 x 1   N = 128 x 128
Physics nu = 0.128

Boundary north velocity(ux = 0.1, uy = 0)
Boundary south wall
Boundary east  wall
Boundary west  wall

Run 60000 steps
Output vtk every 10000 [rho, ux, uy]
```

```julia
using KrakeNN

result = run_simulation(joinpath(@__DIR__, "..", "..", "..", "examples", "cavity.krk"))
nothing #hide
```

The returned `NamedTuple` contains `ρ`, `ux`, `uy` on CPU, ready for
post-processing.

---

## Example 2 — Poiseuille flow (body force)

Download: [`poiseuille.krk`](../assets/krk/poiseuille.krk)

```krk
Simulation poiseuille D2Q9
Domain  L = 0.125 x 1.0  N = 4 x 32
Physics nu = 0.1  Fx = 1e-5

Boundary x periodic
Boundary south wall
Boundary north wall

Run 10000 steps
Output vtk every 2000 [rho, ux, uy]
```

Body forces (`Fx`, `Fy`) on the `Physics` line automatically select the Guo
forcing scheme.

---

## Example 3 — Cylinder with parabolic inlet

Download: [`cylinder.krk`](../assets/krk/cylinder.krk)

```krk
Simulation cylinder D2Q9
Domain  L = 10 x 2.5  N = 200 x 50

Define U  = 0.05
Define H  = 2.5
Define cx = 2.5
Define cy = 1.25
Define R  = 0.5

Physics nu = 0.05

Obstacle cylinder { (x - cx)^2 + (y - cy)^2 <= R^2 }

Boundary west  velocity(ux = 4*U*y*(H - y)/H^2, uy = 0)
Boundary east  pressure(rho = 1.0)
Boundary south wall
Boundary north wall

Run 20000 steps
Output vtk every 1000 [rho, ux, uy]
Diagnostics every 100 [step, drag, lift]
```

User variables (`Define`) are substituted into all expressions.  The `Obstacle`
directive marks solid nodes using a condition function ``f(x,y)``.
The west boundary uses a **spatial** Zou--He profile: the expression
`4*U*y*(H-y)/H^2` is evaluated per node on the inlet face.

---

## Example 4 — Couette flow

Download: [`couette.krk`](../assets/krk/couette.krk)

```krk
Simulation couette D2Q9
Domain  L = 0.125 x 1.0  N = 4 x 32

Define u_wall = 0.05

Physics nu = 0.1

Boundary x periodic
Boundary south wall
Boundary north velocity(ux = u_wall, uy = 0)

Run 10000 steps
Output vtk every 2000 [rho, ux, uy]
```

---

## Internal flows with `Fluid`

For internal flows (channels, contractions), define the fluid region explicitly.
Everything outside is solid.

```
Fluid channel { y > 0.2*Ly && y < 0.8*Ly }
```

Combine with `Obstacle` to add obstructions inside the fluid region:

```
Fluid channel { y > 0.2*Ly && y < 0.8*Ly }
Obstacle cylinder { (x - 5)^2 + (y - Ly/2)^2 <= 0.3^2 }
```

---

## STL geometry import

Complex geometries can be imported from STL files (binary or ASCII).
The mesh is voxelized onto the LBM grid using ray casting.

### Syntax

```
Obstacle body stl(file = "geometry.stl")
Obstacle body stl(file = "geometry.stl", scale = 0.001, translate = [1.0, 0.5, 0.0])
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `file`    | (required) | Path to STL file (relative to `.krk` file) |
| `scale`   | `1.0`   | Uniform scaling factor (applied first) |
| `translate` | `[0,0,0]` | Translation vector ``(t_x, t_y, t_z)`` (applied after scale) |
| `z_slice` | `0.0`   | z-plane for 2D cross-section (only for D2Q9) |

For **2D simulations** (D2Q9), the STL is sliced at `z = z_slice` to extract
a 2D contour, then voxelized via 2D ray casting.
For **3D simulations** (D3Q19), full volumetric ray casting along the z-axis
is used.

### Example: airfoil from STL

```krk
Simulation airfoil D2Q9
Domain  L = 4 x 2  N = 400 x 200
Physics nu = 0.01

Obstacle wing stl(file = "naca0012.stl", scale = 0.5, translate = [1.0, 1.0, 0.0])

Boundary west  velocity(ux = 0.05, uy = 0)
Boundary east  pressure(rho = 1.0)
Boundary south wall
Boundary north wall

Run 50000 steps
Output vtk every 1000 [rho, ux, uy]
```

!!! note "Watertight meshes"
    The STL surface must be closed (watertight) for the ray-casting voxelization
    to produce correct inside/outside classification.

---

## Programmatic usage

The `.krk` file is parsed into a `SimulationSetup` struct that can also be
built from a string:

```julia
using KrakeNN

setup = parse_kraken("""
    Simulation test D2Q9
    Domain L = 1 x 1  N = 32 x 32
    Physics nu = 0.1

    Boundary north velocity(ux = 0.05)
    Boundary south wall
    Boundary east  wall
    Boundary west  wall

    Run 1000 steps
""")

setup.domain.Nx  # 32
```

The struct can be inspected, modified, and passed to `run_simulation(setup)`.

