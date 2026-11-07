# The .krk DSL — overview

KrakeNN exposes two equivalent entry points to the LBM solver:

1. The **Julia API** (`run_cavity_2d(...)`, `run_simulation("file.krk")`, and friends) —
   documented in the [Julia API reference](../api/drivers.md). Use it when you want full
   programmatic control, scripting, or plotting inside a notebook.
2. The **.krk configuration format** — a small declarative DSL parsed by
   `src/io/kraken_parser.jl`. Use it when you want a self-contained,
   human-readable description of a run that can be version-controlled, shared,
   diffed, and swept with kwarg overrides.

## Philosophy

- **Declarative, not imperative.** A .krk file describes *what* the simulation
  is (geometry, physics, boundary conditions), not *how* to step it. The solver
  picks the runner.
- **Gerris-flavored syntax.** Line-oriented, one directive per line, curly
  braces for blocks, `#` for comments. No TOML / YAML / JSON.
- **One source of truth.** Every directive maps to a function in
  `src/io/kraken_parser.jl`; presets expand into plain .krk lines so that
  `Preset cavity_2d` is exactly equivalent to writing the expanded block
  by hand.
- **Fail loudly at parse time.** The parser runs sanity checks (tau, Mach),
  validates face names against the lattice dimension, and offers Levenshtein
  "did you mean?" suggestions on typos.

## Minimal example

```krk
# 2D lid-driven cavity at Re = 100
Simulation cavity D2Q9
Domain     L = 1.0 x 1.0   N = 128 x 128
Physics    nu = 0.01
Boundary   north velocity(ux = 0.1, uy = 0)
Boundary   south wall
Boundary   east  wall
Boundary   west  wall
Run        10000 steps
```

Run it from Julia:

```julia
using KrakeNN
setup = load_kraken("cavity.krk")
run_krk(setup)                          # or the file path directly
```

Override parameters without editing the file (parametric studies):

```julia
setup = load_kraken("cavity.krk"; Re=400, N=256)
```

## When to use what

| Use .krk if you... | Use the Julia API if you... |
|--------------------|------------------------------|
| want a reproducible run description | want to script / loop / plot inline |
| are sharing with a collaborator | need custom post-processing hooks |
| are sweeping parameters | need non-standard runners |
| prefer declarative configs | prefer direct control |

## Reference structure

The rest of this section documents every piece of the DSL:

- [Directives](directives.md) — all top-level keywords
- [Boundary condition types](bc_types.md) — `wall`, `velocity`, `pressure`, …
- [Modules](modules.md) — `thermal`, `axisymmetric`
- [Presets](presets.md) — 5 built-in canonical cases
- [Setup helpers](helpers.md) — `reynolds`, `rayleigh`, `prandtl`
- [Expressions](expressions.md) — the KrakenExpr grammar
- [Sanity checks](sanity.md) — tau and Mach bounds at parse time
- [Error messages](errors.md) — common errors and fixes
- [Face aliases](aliases.md) — dimension- and module-dependent face names
