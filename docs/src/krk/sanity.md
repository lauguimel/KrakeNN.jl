# Sanity checks

`parse_kraken` runs `sanity_check(setup)` at the end of parsing
(`src/io/kraken_parser.jl`, ~line 1204). Failures fall into two categories:

- **Errors** (throw): parameter combinations that will definitely blow up
  the solver, e.g. tau below the lower stability bound.
- **Warnings** (`@warn`): parameter combinations that are physically
  suspect — marginal stability, compressibility, or CFL risk.

## Tau bound

The BGK relaxation time is `τ = 3ν + 0.5`. Bounds:

| Condition | Severity | Action |
|-----------|----------|--------|
| `τ < 0.50` | **error** | increase `ν` or reduce `Re` |
| `τ < 0.51` | **warn**  | still runs but marginally stable |
| `τ ≥ 0.51` | OK | |

Source excerpt:

```julia
ν = get(setup.physics.params, :nu, NaN)
if !isnan(ν)
    τ = 3ν + 0.5
    if τ < 0.5
        error("Sanity check failed: tau = $τ < 0.5 (unstable). " *
              "To fix: increase ν (current $ν), or use Setup reynolds with " *
              "a larger L_ref / smaller Re.")
    elseif τ < 0.51
        @warn "tau = $τ is very close to 0.5 (marginally stable). " *
              "To fix: increase ν from $ν, or reduce Re / increase grid N " *
              "from $(setup.domain.Nx)."
    end
end
```

**How to fix**: either raise `Physics nu = ...`, or keep the same physical
Reynolds number and refine the grid (higher `N` → larger `L_ref` → larger
`ν` for the same `Re`).

## Mach / U_ref bound

The textbook LBM compressibility bound is `U_ref ≤ 0.1` in lattice units
(Krüger et al. 2017). In Mach-number space this is
`Ma = U_ref / cs ≤ 0.1·√3 ≈ 0.173`, **not** `Ma ≤ 0.1`. KrakeNN compares
`U_ref` against `0.1` directly to avoid a false-positive that would fire on
every default case where `U_ref` falls back to exactly `0.1`.

This bound was reviewed and corrected in Phase 4.3 prereq A after an
off-by-`√3` bug.

| Condition | Severity | Action |
|-----------|----------|--------|
| `U_ref > 0.1` | **warn** | compressibility errors likely |
| `U_ref > 0.3` | **warn** | CFL risk on top of compressibility |

Source excerpt:

```julia
U_ref = _probe_U_ref(setup.boundaries)
if U_ref > 0.1
    cs = 1.0 / sqrt(3.0)
    Ma = U_ref / cs
    @warn "Lattice velocity U_ref = $U_ref exceeds 0.1 " *
          "(Mach number Ma = $Ma, compressibility errors likely). " *
          "To fix: decrease U_ref to ≤ 0.1, " *
          "or increase grid N to reduce lattice velocity."
end
if U_ref > 0.3
    @warn "Lattice velocity U_ref = $U_ref > 0.3 (CFL risk). " *
          "To fix: decrease U_ref, or increase N from $(setup.domain.Nx)."
end
```

**How it picks `U_ref`**: `_probe_U_ref` scans boundary conditions for the
first `velocity` BC and returns `sqrt(ux² + uy²)`. If no velocity BC is
present (pure body-force or pressure-driven), it defaults to `0.1`, which
silently passes the bound — so for those cases you should monitor the
diagnostics yourself.

```julia
function _probe_U_ref(boundaries::Vector{BoundarySetup})
    for b in boundaries
        if b.type == :velocity
            ux = 0.0; uy = 0.0
            haskey(b.values, :ux) && (ux = Float64(evaluate(b.values[:ux])))
            haskey(b.values, :uy) && (uy = Float64(evaluate(b.values[:uy])))
            mag = sqrt(ux^2 + uy^2)
            mag > 0 && return mag
        end
    end
    return 0.1
end
```

## Face / lattice validation

Separate from `sanity_check`, `_validate_faces_vs_lattice` runs *before* the
sanity check and rejects 3D-only face names on a D2Q9 lattice:

```julia
d2_faces = (:north, :south, :east, :west, :front, :back)
if setup.lattice == :D2Q9
    for b in setup.boundaries
        if !(b.face in d2_faces)
            throw(ArgumentError(
                "Boundary face ':$(b.face)' is not valid for D2Q9. " *
                "2D face names are: north/south/east/west."))
        end
    end
end
```

See [Aliases](aliases.md) for the complete face-name table.

## Silencing warnings

All warnings use `@warn` and can be suppressed via Julia's standard logger
(`Logging.disable_logging(Logging.Warn)`). Errors cannot be silenced —
that's the point.
