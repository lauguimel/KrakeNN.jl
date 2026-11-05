# KRK Reference

The `.krk` file is Kraken's plain-text case format. One file fully describes a
simulation — geometry, physics, boundary conditions, run length and output — in a
declarative, block-per-line syntax inspired by Gerris. You never write Julia to
run a standard case: you write a `.krk` file and hand it to `run_simulation`.

```@contents
Pages = ["users/krk-reference.md"]
Depth = 2
```

## Run your first `.krk` in 5 minutes

Create a file `mycase.krk`:

```
# Lid-driven cavity at Re = 100
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

Run it from Julia:

```julia
using Kraken
run_simulation("mycase.krk")
```

That is the whole loop: the runner parses the file, builds the lattice, picks the
right backend driver from the lattice and the active modules, iterates, and writes
VTK snapshots into `output/`. Open the `.vtr` files in ParaView to view the field.

A few conventions hold throughout:

- **Lines are blocks.** Each non-blank, non-comment line is one block. Comments
  start with `#`. Brace blocks (`{ ... }`) may span multiple lines.
- **One `Simulation` line, one `Domain`, one `Physics`.** Most other blocks
  (`Boundary`, `Obstacle`, `Output`, `Refine`) may appear several times.
- **Lattice units.** Unless a `Units { ... }` block is present, every number is in
  lattice units (LU). The diffusive/acoustic scaling that maps LU to physical units
  is your responsibility — or you delegate it to a `Units` block (see below).
- **Expressions are allowed** anywhere a number is. `4*U*y*(H - y)/H^2`,
  `0.05*sin(2*pi*x)`, and references to `Define`d variables all evaluate at parse
  time (boundary-profile expressions also see the spatial coordinates `x`, `y`, `z`).

## Block catalogue

The parser recognises the following block keywords (see
`src/io/kraken_parser.jl`). Each is documented below with its keywords and a
minimal worked example. Two directives — `Preset` and `Sweep` — are pre-expanded
before the main parse and are documented under [Presets](@ref) and
[Sweeps](@ref).

| Block         | Multiplicity | Purpose                                            |
|---------------|--------------|----------------------------------------------------|
| `Simulation`  | exactly 1    | case name + lattice (`D2Q9` / `D3Q19`)             |
| `Domain`      | exactly 1    | physical size `L` and grid resolution `N`          |
| `Physics`     | 1            | transport coefficients (`nu`, `alpha`, `Pr`, …)    |
| `Define`      | many         | named scalar constants for reuse in expressions    |
| `Units`       | ≤ 1          | physical-units → LU bridge (auto `nu`, `u_LU`)     |
| `Module`      | many         | activate a solver module (`thermal`, `axisymmetric`, …) |
| `Obstacle`    | many         | solid region (implicit inequality or STL)          |
| `Fluid`       | many         | fluid-tagged region (two-phase / init by region)   |
| `Boundary`    | many         | a domain-face boundary condition                   |
| `Initial`     | ≤ 1          | initial field expressions (`ux`, `uy`, `C`, …)     |
| `Velocity`    | ≤ 1          | frozen velocity field (advection-only module)      |
| `Refine`      | many         | under development, not validated (grid refinement)  |
| `Rheology`    | many         | non-Newtonian / viscoelastic constitutive model    |
| `Setup`       | 1            | extra driver helper key/values                     |
| `Run`         | exactly 1    | number of time steps                               |
| `Output`      | many         | field snapshots (`vtk` / `png` / `gif`)            |
| `Diagnostics` | ≤ 1          | scalar time-series columns (`drag`, `lift`, …)     |

### `Simulation`

`Simulation <name> <lattice>` — the case name (used for output filenames) and the
lattice stencil. Valid lattices: **`D2Q9`** (2D) and **`D3Q19`** (3D). Exactly one
per file.

```
Simulation cavity D2Q9
```

### `Domain`

`Domain L = <Lx> x <Ly> [x <Lz>]  N = <Nx> x <Ny> [x <Nz>]` — physical extent and
grid resolution. The third (`Lz`/`Nz`) factor is read only for `D3Q19`. Values may
be numbers or `Define`d variables.

```
Domain L = 10.0 x 2.5  N = 200 x 50          # 2D
Domain L = 1.0 x 1.0 x 1.0  N = 64 x 64 x 64 # 3D
```

When a `Units` block is present, `L` must equal `N` in the active dimensions so
that the geometry coordinates are themselves lattice units.

### `Physics`

`Physics key = value ...` — transport coefficients, all in lattice units. Any
`key = value` pair is accepted; the keys consumed by the drivers include:

| Key    | Meaning                                                      |
|--------|--------------------------------------------------------------|
| `nu`   | kinematic viscosity (LU). `nu = auto` defers to a `Units` block |
| `alpha`| thermal diffusivity (thermal module)                         |
| `Pr`   | Prandtl number (thermal module)                              |
| `Ra`   | Rayleigh number (thermal module)                             |
| `Fx`, `Fy` | constant body force per direction (e.g. Poiseuille drive) |

```
Physics nu = 0.1 alpha = 0.01           # forced-conduction
Physics nu = 0.02 Pr = 0.71 Ra = 1e5    # natural convection
Physics nu = 0.1  Fx = 1e-5             # body-force-driven Poiseuille
```

### `Define`

`Define <NAME> = <number>` — a named constant. Defines are visible to every
expression parsed afterwards (boundary profiles, obstacle conditions, `Domain`
sizes). They are also the override points for parametric runs: a keyword passed to
`run_simulation`/`parse_kraken` overrides the matching `Define`.

```
Define U = 0.05
Define R = 0.5
Boundary west velocity(ux = 4*U*y*(H - y)/H^2, uy = 0)
```

### `Units`

`Units { length = <unit>  L_ref = <val>  R_LU = <int>  Re = <val>  [scaling = <acoustic|diffusive|auto>]  [L_up = ...] [L_down = ...] }`
— the physical-units bridge. Given a reference length `L_ref` in physical units
(resolved to `R_LU` lattice cells) and a target Reynolds number, the bridge solves
for the LU viscosity and inflow speed and injects them as the variable `u_LU` and
as `Physics nu`. Required keys: `length`, `L_ref`, `R_LU`, `Re`. The block
currently requires at least one STL geometry region and `Domain L == N`.

```
Units { length = mm  L_ref = 0.2  R_LU = 4  Re = 1.0  scaling = acoustic }
Physics nu = auto
Boundary west velocity(ux = u_LU, uy = 0)
```

### `Module`

`Module <name>` — activate a solver module; may appear several times. The runner
dispatches on the active set. Documented module names include **`thermal`**
(double-distribution temperature + Boussinesq buoyancy), **`axisymmetric`**
(`(z, r)` mesh with `z`/`wall`/`axis` face aliases), **`advection_only`** (frozen-
velocity VOF advection, needs a `Velocity` and `Initial { C = ... }`), and
**`twophase_vof`** (two-phase with surface tension).

```
Module thermal
```

### `Obstacle` / `Fluid`

`Obstacle <name> [wall=<wall|libb>] { <inequality> }` or
`Obstacle <name> [wall=libb] stl(file = "...", [scale=...], [translate=[x,y,z]], [z_slice=...])`
— a solid region, defined either by an implicit inequality (true = solid) or by an
STL mesh. The `wall=` selector chooses the wall treatment: `wall` (default,
half-way bounce-back) or `libb` (interpolated/linear bounce-back, needed for
curved STL surfaces and cut-link drag). `Fluid` uses the same syntax to tag a
fluid sub-region.

```
Obstacle cylinder { (x - cx)^2 + (y - cy)^2 <= R^2 }
Obstacle sph wall=libb stl(file = "examples/geometry_stl/sphere.stl", scale = 20.0)
```

### `Boundary`

`Boundary <face> <type>(<params>)` or `Boundary <face> <type> [extra = ...]`, plus
the axis shorthand `Boundary <x|y|z> periodic`. Faces are **`west` / `east` /
`south` / `north`** (2D) plus **`top` / `bottom`** (3D); `front` / `back` are
legacy 3D aliases. Boundary types: **`wall`** (no-slip bounce-back), **`velocity`**
(Zou–He imposed velocity), **`pressure`** (imposed density/pressure outlet),
**`periodic`**, **`outflow`**, **`neumann`**, **`symmetry`**. A trailing `T = ...`
sets a thermal Dirichlet/condition on the face. Velocity-profile parameters may be
spatial expressions in `x`, `y`, `z`.

```
Boundary north velocity(ux = 0.1, uy = 0)
Boundary east  pressure(rho = 1.0)
Boundary west  wall T = 1.0          # no-slip + hot Dirichlet
Boundary x periodic                  # west+east periodic in one line
```

### `Initial` / `Velocity`

`Initial { <field> = <expr> ... }` seeds the initial fields; `Velocity { ux = ...
uy = ... }` supplies a frozen velocity for the advection-only module. Both use the
same brace syntax, and expressions see `x`, `y`, `z`.

```
Initial { ux = 0.05*sin(2*pi*x)*cos(2*pi*y) uy = -0.05*cos(2*pi*x)*sin(2*pi*y) }
```

### `Refine`

`Refine <name> { region = [x0, y0, x1, y1] (or 6 coords in 3D), ratio = 2, parent = base, [criterion options] }`
— a patch-based, 2:1 nested refinement region. Grid refinement (patch and conservative AMR) is under development and not validated; it is not part of the supported feature set. The keyword is
parsed, but results obtained with it must not be relied on; see
[Limitations](../theory/10_limitations.md).

```
Refine corner_patch {
    region = [0.5, 0.5, 1.0, 1.0],
    ratio  = 2,
    parent = base
}
```

### `Rheology`

`Rheology [phase] <model> { key = value ... }` — a non-Newtonian or viscoelastic
constitutive law. Optional `phase` is `liquid` / `gas` / `default`. Known models:
`newtonian`, `power_law`, `carreau`, `cross`, `bingham`, `herschel_bulkley`,
`oldroyd_b`, `fene_p`, `saramito`. Each model reads its own parameters (e.g.
`power_law` reads `K`, `n`; `oldroyd_b` reads `nu_s`, `nu_p`, `lambda`). A thermal
coupling is inferred from `E_a` (Arrhenius, absolute temperatures only: `T_ref > 0`) or `C1`/`C2` (WLF, base-10 form with the published constants). The coupling acts only in thermal cases; in an isothermal run these keys are ignored and a warning names them.

```
Rheology power_law { K = 0.1  n = 0.5 }
Rheology oldroyd_b { nu_s = 0.0885  nu_p = 0.0615  lambda = 6000 }
```

### `Run`

`Run <N> steps` — the number of LBM time steps. Exactly one per file.

```
Run 60000 steps
```

### `Output`

`Output <format> every <N> [field1, field2, ...] [fps = <N>]` — snapshot writer.
Formats: **`vtk`** (`.vtr`, open in ParaView), **`png`**, **`gif`** (the `fps`
parameter applies to `gif`). Fields are the lattice fields to dump (`rho`, `ux`,
`uy`, `uz`, `T`, …). Several `Output` lines may coexist. Output lands in `output/`.

```
Output vtk every 10000 [rho, ux, uy]
Output vtk every 5000 [rho, ux, uy, uz, T]
```

### `Diagnostics`

`Diagnostics every <N> [col1, col2, ...]` — a scalar time-series log. Columns seen
in the examples include `step`, `drag`, `lift`. Useful for tracking the drag/lift
on an obstacle as the run converges.

```
Diagnostics every 100 [step, drag, lift]
```

### `Setup`

`Setup { key = value ... }` — extra driver helper key/values merged into the
driver call. This is an escape hatch for driver-specific options not covered by a
dedicated block; document the keys against the target driver before relying on
them.

## Presets

A `Preset <name>` line expands, before parsing, into a full set of `.krk` lines for
a canonical case. You can then override any expanded block by re-stating it after
the `Preset` line — the later block wins. The presets shipped in
`kraken_parser.jl` (`_expand_preset`) are:

| Preset              | Case                                                |
|---------------------|-----------------------------------------------------|
| `cavity_2d`         | lid-driven cavity, `128²`, lid `ux = 0.1`           |
| `poiseuille_2d`     | body-force Poiseuille channel, periodic in `x`      |
| `couette_2d`        | Couette shear, moving top wall                      |
| `taylor_green_2d`   | doubly-periodic Taylor–Green vortex                 |
| `rayleigh_benard_2d`| hot-bottom / cold-top Rayleigh–Bénard, `Module thermal` |

```
Preset rayleigh_benard_2d
Run 30000 steps
Output vtk every 5000 [rho, ux, uy, T]
```

## Sweeps

A `Sweep param = [a, b, c]` directive turns one file into a family of cases. Use
`parse_kraken_sweep` (or `run_simulation` over the sweep) to materialise one
`SimulationSetup` per value; the swept name overrides the matching `Define`.

```
Define Re = 100
Sweep Re = [100, 400, 1000]
```

## Things to know / current gaps

- **`natural_convection` is not a preset.** The thermal-cavity benchmark is built
  from explicit blocks (`Module thermal` + hot/cold `Boundary ... T = ...`), not a
  preset — `_expand_preset` knows only the five names above. The
  [thermal natural-convection benchmark](benchmarks/thermal-natural-convection.md)
  writes the case out explicitly. Adding `natural_convection_2d`/`_3d` presets is a natural
  follow-up.
- **`Setup` is intentionally open-ended.** Its keys are driver-specific and are
  not enumerated by the parser; treat it as an advanced escape hatch and validate
  the keys against the driver you target.
- **`Units` requires an STL region** and `Domain L == N` today; the implicit-
  geometry path runs in pure lattice units.
