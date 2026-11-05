# Modules

A `Module <name>` directive toggles an optional physics add-on. Kraken's
v0.1.0 DSL recognizes two modules:

- `thermal` — couple the LBM to a thermal DDF (double-distribution function).
- `axisymmetric` — rewrite the 2D D2Q9 kernel on an `(r, z)` mesh with an
  axis-of-symmetry source term.

The list of active modules is collected in `_parse_kraken_internal_single`
before any `Boundary` line is parsed, so module-dependent aliases can kick in
during boundary parsing:

```julia
for line in lines
    _first_word(line) == "Module" || continue
    push!(modules, _parse_module(line))
end
is_axisym = :axisymmetric in modules
```

## `thermal`

**Syntax**

```krk
Module thermal
```

**What it activates**

- Allocation of a second DDF for temperature transport.
- Acceptance of the `T = <expr>` kwarg on `Boundary <face> wall` directives (used by
  `heat_conduction` cases only; the natural-convection and Rayleigh-Bénard cases ignore the values).
  This is *not* a separate BC type — the parser re-uses the trailing-kwarg
  form of the `wall` simple parser. Example: `Boundary south wall T = 1.0`.
- Forwarding of `Pr`, `Ra`, `alpha`, `gbeta_DT` from `Physics` to the thermal
  runner.
- In `Obstacle <name> wall(T = ...) { ... }`, the temperature value is
  attached to the obstacle's bounce-back surface.

**Why is this a module and not always-on?**

Thermal LBM doubles memory (two DDFs) and adds ~30 % runtime. Gating it
behind an explicit `Module thermal` keeps cold isothermal runs cheap and
makes the intent of a .krk file visible at a glance.

**Example**

```krk
Module thermal
Physics nu = 0.01 Pr = 0.71 Ra = 1e5
Boundary south wall T = 1.0
Boundary north wall T = 0.0
Boundary x     periodic
```

## `axisymmetric`

**Syntax**

```krk
Module axisymmetric
```

**What it activates**

- The axisymmetric D2Q9 kernel: the 2D grid is interpreted as an `(r, z)`
  plane with `z` along the x-axis and `r` along the y-axis, with `r = 0`
  on the south face.
- A radial source term in the collision step that re-expresses the full
  3D axisymmetric Navier–Stokes operator.
- User-facing face aliases `z` → `x`, `wall` → `north`, `axis` → `south`,
  implemented in `_resolve_axisym_face`:

```julia
_resolve_axisym_face(face::AbstractString) =
    face == "z"    ? "x"     :
    face == "wall" ? "north" :
    face == "axis" ? "south" : String(face)
```

**Why aliases?**

In an axisymmetric run the user thinks in `(r, z)`, not `(x, y)`. Writing
`Boundary wall velocity(uz = 0.1, ur = 0)` reads naturally; writing
`Boundary north velocity(ux = 0.1, uy = 0)` does not. The parser therefore
accepts both and maps the geometric names onto the D2Q9 faces before
validation runs.

The aliases are only active when `is_axisym` is `true`; in a plain 2D case,
`Boundary wall ...` is an error (`wall` is not a valid face name).

**Example**

```krk
Simulation jet_axi D2Q9
Domain L = 10.0 x 1.0  N = 512 x 64
Module axisymmetric
Physics nu = 0.01
Boundary z    periodic
Boundary wall wall
Boundary axis symmetry
Run 5000 steps
```

## Adding a module

The list of recognized modules lives implicitly in the runners (no
whitelist in the parser itself — `_parse_module` just returns a `Symbol`).
Unknown modules are silently ignored by the parser and surface later as
"no runner found for modules [...]" errors in `run_krk`.

Source: `src/io/kraken_parser.jl`.
