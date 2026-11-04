# Directives

A .krk file is a sequence of line-oriented directives. The parser recognizes
the following top-level keywords (see `_parse_kraken_internal_single` in
`src/io/kraken_parser.jl`):

```
Simulation, Domain, Physics, Define, Obstacle, Fluid, Boundary,
Refine, Initial, Velocity, Module, Run, Output, Diagnostics,
Rheology, Setup, Preset, Sweep, Backend
```

Unknown keywords trigger a Levenshtein-based *did you mean?* suggestion.

## Simulation

**Syntax:** `Simulation <name> <lattice>`

- `name` — identifier used in output directory names and logs.
- `lattice` — `D2Q9` (2D) or `D3Q19` (3D). Nothing else is accepted.

**Example**

```krk
Simulation cavity_2d D2Q9
```

**Parser:** `_parse_simulation`

```julia
function _parse_simulation(line::String)
    tokens = split(line)
    length(tokens) < 3 && throw(ArgumentError("Simulation needs name and lattice: $line"))
    name = String(tokens[2])
    lattice = Symbol(tokens[3])
    lattice in (:D2Q9, :D3Q19) || throw(ArgumentError("Unknown lattice '$lattice'. Use D2Q9 or D3Q19"))
    return name, lattice
end
```

## Domain

**Syntax:** `Domain L = <Lx> x <Ly> [x <Lz>]  N = <Nx> x <Ny> [x <Nz>]`

Physical extents (`L`) and grid resolution (`N`), separated by a literal `x`.
Values can be numeric literals or identifiers previously declared with
`Define`. The 3D component is optional in 2D.

**Example**

```krk
Define N = 128
Domain L = 1.0 x 1.0   N = N x N
Domain L = 2.0 x 1.0 x 1.0  N = 128 x 64 x 64
```

**Parser:** `_parse_domain` — accepts bare numbers or user variables via
`_eval_domain_value`. Kwargs passed to `load_kraken(...; Nx=256)` override the
file defaults.

## Physics

**Syntax:** `Physics key1 = val1 key2 = val2 ...`

Physical parameters stored in a `Dict{Symbol,Float64}`. Special keys `Fx`,
`Fy`, `Fz` become body-force expressions. Values can be numeric literals or
simple expressions referencing `Define` variables.

Common keys: `nu` (kinematic viscosity, lattice units), `Pr` (Prandtl), `Ra`
(Rayleigh), `alpha` (thermal diffusivity), `gbeta_DT` (buoyancy),
`Fx`/`Fy`/`Fz` (body force).

**Example**

```krk
Define Re = 1000
Physics nu = 0.01 Pr = 0.71 Ra = 1e5
Physics nu = 0.1 Fx = 1e-5
```

**Parser:** `_parse_physics`.

## Define

**Syntax:** `Define <NAME> = <literal>`

Declare a user variable used in other directives. Overridable via kwargs to
`load_kraken`. First-pass scanned before anything else; kwargs win over the
file value.

**Example**

```krk
Define Re  = 1000
Define N   = 128
Define Lid = 0.1
```

**Parser:** `_parse_define`.

## Boundary

**Syntax:**

```
Boundary <face> <type>(key = val, ...)
Boundary <face> wall                       # simple form, no parentheses
Boundary <face> wall T = 1.0               # simple form with trailing kwargs
Boundary <axis> periodic                   # shorthand: axis ∈ {x, y, z}
```

`<face>` is one of `north`, `south`, `east`, `west` (2D) or additionally
`top`, `bottom` (3D only). Axisymmetric runs can use `z`, `wall`, `axis` —
see [Aliases](aliases.md). `<type>` is one of `wall`, `velocity`, `pressure`,
`periodic`, `symmetry` — see [BC types](bc_types.md).

The `Boundary x periodic` shorthand expands to a pair of `:periodic`
BoundarySetup entries for the east/west (or north/south) faces.

**Example**

```krk
Boundary north velocity(ux = 0.1, uy = 0)
Boundary south wall
Boundary x     periodic
Boundary north wall T = 0.0
```

**Parser:** `_parse_boundary` (see the [BC types](bc_types.md) page for the
full excerpt).

## Refine

**Syntax:**

```
Refine <name> { region = [x0, y0, x1, y1], ratio = 2, parent = <name> }
```

Declare a nested refinement patch over an axis-aligned box. `ratio` defaults
to `2`; `parent` defaults to `""` (the base grid).

!!! warning "Under development"
    Grid refinement (patch and conservative AMR) is under development and not validated; it is not part of the supported feature set. This keyword is kept for reference only.

**Example**

```krk
Refine nearwall { region = [0.0, 0.0, 1.0, 0.1], ratio = 2 }
Refine tip     { region = [0.4, 0.0, 0.6, 0.1], ratio = 2, parent = nearwall }
```

**Parser:** `_parse_refine`.

## Initial / Velocity

**Syntax:** `Initial { field = expr field = expr ... }`

Initial fields are parsed into a `Dict{Symbol,KrakenExpr}` with keys like
`ux`, `uy`, `uz`, `T`, `C`, `rho`. `Velocity { ... }` uses the exact same
syntax and is an alias specifically for prescribed velocity fields (used by
some runners to impose a frozen background flow).

**Example**

```krk
Initial  { ux = 0.05*sin(2*pi*x) uy = -0.05*cos(2*pi*y) }
Velocity { ux = 0.1*(1 - (y/Ly)^2) uy = 0 }
```

**Parser:** `_parse_initial`.

## Module

**Syntax:** `Module <name>`

Activate an optional physics module. Known modules: `thermal`,
`axisymmetric`. See the [Modules](modules.md) page for what each one enables
(thermal BC kwargs, face aliases, etc.).

**Example**

```krk
Module thermal
Module axisymmetric
```

**Parser:** `_parse_module`.

## Run

**Syntax:** `Run <N> steps`

Total number of LBM time steps. The word `steps` is cosmetic — any non-digit
tail is ignored.

**Example**

```krk
Run 10000 steps
```

**Parser:** `_parse_run`.

## Output

**Syntax:** `Output <format> every <N> [field1, field2, ...]`

Periodic output of fields. `format` is typically `vtk`; `N` is the interval
in time steps; the bracket list enumerates fields (`ux`, `uy`, `rho`, `T`,
`C`, …). Output directory defaults to `output/`.

**Example**

```krk
Output vtk every 500 [ux, uy, rho, T]
```

**Parser:** `_parse_output`.

## Diagnostics

**Syntax:** `Diagnostics every <N> [col1, col2, ...]`

Periodic scalar diagnostics (kinetic energy, Nusselt number, mass, …) logged
to stdout and/or CSV.

**Example**

```krk
Diagnostics every 100 [step, time, KE, uMax, Nu]
```

**Parser:** `_parse_diagnostics`.

## Obstacle / Fluid

**Syntax:**

```
Obstacle <name> [wall(T = ..., ...)] { <condition> }
Obstacle <name> stl(file = "path.stl", scale = ..., translate = [x, y, z])
Fluid    <name> { <condition> }
```

Declare a solid obstacle or a fluid region by a condition expression
`(x, y [, z]) -> Bool`, or by reference to an STL file. The optional
`wall(...)` block attaches boundary-condition values to the obstacle (e.g.
temperature). See `_parse_geometry_region` / `_parse_stl_params`.

**Example**

```krk
Obstacle disk wall(T = 1.0) { (x - 0.5)^2 + (y - 0.5)^2 < 0.01 }
Obstacle cyl  stl(file = "cylinder.stl", scale = 1e-3, translate = [0, 0, 0])
Fluid    jet  { y < 0.2 }
```

## Setup

**Syntax:** `Setup key = val key = val ...`

Non-dimensional helper directive — see [Helpers](helpers.md) for the
back-computation formulas. Known keys: `reynolds`, `rayleigh`, `prandtl`,
`L_ref`, `U_ref`.

**Example**

```krk
Setup reynolds = 1000
Setup rayleigh = 1e5 prandtl = 0.71
```

**Parser:** `_parse_setup`.

## Preset

**Syntax:** `Preset <name>`

Expand a canonical case into its constituent directives before any other
parsing happens. See [Presets](presets.md) for the 5 presets and their exact
expansions.

**Example**

```krk
Preset cavity_2d
```

**Parser:** `_expand_preset` + `_preset_lines`.

## Sweep

**Syntax:** `Sweep <param> = [v1, v2, v3, ...]`

Cartesian-product parameter sweep. Each combination produces one
`SimulationSetup`. Parse with `parse_kraken_sweep` / `load_kraken_sweep`
(which always return a `Vector{SimulationSetup}`).

**Example**

```krk
Sweep Re = [100, 400, 1000]
Sweep N  = [64, 128, 256]
```

**Parser:** `_parse_sweep`, driver in `_parse_kraken_internal`.

## Backend

**Syntax:** `Backend <cpu|cuda|metal>`

Selects the compute backend for `run_simulation`, so the `.krk` file alone
decides how a case runs — no separate Julia driver script required. At most
one `Backend` directive is allowed per file.

**Resolution order** (see `_resolve_run_backend` in `src/simulation_runner.jl`):

1. An explicit `backend=...` keyword argument to `run_simulation` — this
   ALWAYS wins over the file, for backward compatibility with existing
   Julia-driven scripts.
2. Otherwise, this `Backend` directive, if present.
3. Otherwise, `KernelAbstractions.CPU()` (unchanged historical default).

Only the backend *name* is checked at parse time (fail fast on a typo — same
`ArgumentError` style as an unknown lattice). Whether `cuda`/`metal` is
actually usable (package loads, `functional()` returns `true`) is checked
lazily at run time in `_backend_from_krk_symbol`: a non-functional or
unloadable GPU backend raises a clear `ArgumentError` naming the backend and
suggesting `Backend cpu` — it is **never** silently downgraded to CPU.

**Example**

```krk
Backend cpu
Backend cuda
Backend metal
```

**Parser:** `_parse_backend`.

## Rheology

**Syntax:** `Rheology [phase] <model> { key = value ... }`

Per-phase non-Newtonian constitutive model. `phase` is `liquid`, `gas`, or
`default`; `<model>` is one of `newtonian`, `power_law`, `carreau`, `cross`,
`bingham`, `herschel_bulkley`, `oldroyd_b`, `fene_p`, `saramito`.

**Example**

```krk
Rheology power_law { K = 0.1 n = 0.5 }
Rheology liquid carreau { eta_0 = 1.0 eta_inf = 0.01 lambda = 1.0 a = 2.0 n = 0.5 }
```

**Parser:** `_parse_rheology` + `build_rheology_model`.
