# Boundary condition types

The `Boundary` directive accepts the following BC types (see
`known_bc_types` in `_parse_boundary`, `src/io/kraken_parser.jl`):

```julia
known_bc_types = (:wall, :velocity, :pressure, :periodic, :outflow,
                  :neumann, :symmetry)
```

The v0.1.0 documented set is `wall`, `velocity`, `pressure`, `periodic`,
`symmetry`. The parser also accepts `outflow` / `neumann` but those are
runner-specific and not covered here.

## `wall`

No-slip wall (bounce-back). In thermal mode, a wall can carry a fixed
temperature via a trailing `T = ...` kwarg.

**Syntax**

```
Boundary <face> wall
Boundary <face> wall T = 1.0
```

**Example**

```krk
Boundary south wall
Boundary north wall T = 0.0
```

**Notes**

- Standard half-way bounce-back. No kwargs needed for isothermal cases.
- With `Module thermal`, the `T` kwarg is read for `heat_conduction` cases only (one opposing
  pair of faces). The `natural_convection` and `rayleigh_benard` cases ignore it and use the
  driver's own wall temperatures (hot = 1, cold = 0).

## `velocity`

Zou–He velocity inlet/moving wall. `ux`, `uy` (and `uz` in 3D) are
expressions in `(x, y, z, t)` — constant, parabolic, time-pulsed, whatever
the expression grammar allows.

**Syntax**

```
Boundary <face> velocity(ux = <expr>, uy = <expr> [, uz = <expr>])
```

**Example**

```krk
Boundary north velocity(ux = 0.1, uy = 0)
Boundary west  velocity(ux = 0.1*(1 - ((y - Ly/2)/(Ly/2))^2), uy = 0)
Boundary west  velocity(ux = 0.1*(1 + 0.05*sin(2*pi*t/100)), uy = 0)
```

**Notes**

- The magnitude of the velocity at the inlet is what `_probe_U_ref` picks up
  for the Mach-bound sanity check. Keep `|u| ≤ 0.1` in lattice units.
- In 2D (D2Q9), only `ux` and `uy` are meaningful.

## `pressure`

Zou–He pressure (density) boundary. Sets `rho` — Kraken LBM uses `rho`
directly rather than `p` because pressure in LBM is just `p = rho/3` in
lattice units.

**Syntax**

```
Boundary <face> pressure(rho = <expr>)
```

**Example**

```krk
Boundary east pressure(rho = 1.0)
Boundary west pressure(rho = 1.001)      # slight drop drives a Poiseuille flow
```

**Notes**

- Use a small `Δrho` to avoid compressibility artefacts.
- Pair with `velocity` on the opposite face, or use `Fx` body force, not both.

## `periodic`

Periodic BC. Has two forms: per-face or axis shorthand.

**Syntax**

```
Boundary <face> periodic                # face-by-face
Boundary x periodic                     # shorthand: east + west
Boundary y periodic                     # shorthand: north + south
```

**Example**

```krk
Boundary x periodic
Boundary y periodic
```

**Notes**

- The shorthand form expands to two `BoundarySetup` records. Implementation:

```julia
if axis == "x"
    return [BoundarySetup(:west, :periodic, ...),
            BoundarySetup(:east, :periodic, ...)]
end
```

- Periodic BCs must be declared in pairs; declaring only one face is an error.

## `symmetry`

Mirror-symmetry BC. Accepted by the parser mainly for axisymmetric cases,
where the kernel enforces the axis condition internally. Non-axisymmetric
runners may treat it as a no-op.

**Syntax**

```
Boundary <face> symmetry
```

**Example**

```krk
Module axisymmetric
Boundary axis symmetry        # resolved to 'south' via the axisym alias
```

**Notes**

- See [Aliases](aliases.md) for the `wall`/`axis`/`z` axisymmetric face
  aliases.

## BC parser excerpt

Relevant portion of `_parse_boundary` showing the type dispatch:

```julia
known_bc_types = (:wall, :velocity, :pressure, :periodic, :outflow,
                  :neumann, :symmetry)

type_m = match(r"^(\w+)\(", after_face)
if type_m !== nothing
    bc_type = Symbol(type_m.captures[1])
    if bc_type ∉ known_bc_types
        sug = _suggest_name(String(type_m.captures[1]), known_bc_types)
        msg = "Unknown boundary type '$bc_type'"
        sug !== nothing && (msg *= " (did you mean: $sug?)")
        throw(ArgumentError(msg))
    end
    ...
end
```

Source: `src/io/kraken_parser.jl`.
