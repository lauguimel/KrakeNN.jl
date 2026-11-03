# Error messages

The parser prefers loud, actionable errors at parse time over silent
misbehaviour at run time. This page collects the common error messages,
their causes, and how to fix them.

Whenever a typo is detected, a Levenshtein "did you mean?" suggestion is
appended via `_suggest_name`.

## Parser errors

| Error message | Cause | Fix |
|---------------|-------|-----|
| `Unknown keyword 'xxx' in .krk file (did you mean: yyy?)` | Typo in a top-level directive name | Fix the spelling — the suggestion is usually right |
| `Unknown lattice 'Dxx'. Use D2Q9 or D3Q19` | `Simulation foo Dxx` with a bad lattice | Use `D2Q9` or `D3Q19` |
| `Simulation needs name and lattice: ...` | Missing tokens after `Simulation` | `Simulation <name> D2Q9` |
| `Cannot parse Domain L = ... : ...` | Missing `L = Lx x Ly` block | `Domain L = 1.0 x 1.0  N = 128 x 128` |
| `Cannot parse Domain N = ... : ...` | Missing `N = Nx x Ny` block | same as above |
| `Unknown variable 'xxx' in Domain` | `Domain N = N x N` without a preceding `Define N = ...` | Add `Define N = 128`, or pass `N` as kwarg to `load_kraken` |
| `Cannot parse Define: ...` | `Define X = <expr>` with a non-literal RHS | `Define` accepts only bare `Float64` literals, not expressions |
| `Missing 'Domain' in .krk file` | No `Domain` directive at all | Add one |
| `Missing 'Simulation' in .krk file` | No `Simulation` directive | Add one |
| `Missing 'Run' in .krk file` | No `Run N steps` directive | Add one |

## Boundary errors

| Error message | Cause | Fix |
|---------------|-------|-----|
| `Unknown boundary face 'xxx'` | Face name not in the allowed set | Use `north`/`south`/`east`/`west` (+ `top`/`bottom` for D3Q19) |
| `Boundary face ':top' is not valid for D2Q9` | Used `top`/`bottom` on a 2D lattice | Switch to `D3Q19` or use `north`/`south` |
| `Unknown boundary type 'velocty' (did you mean: velocity?)` | BC type typo | Fix the spelling |
| `Cannot parse Boundary face: ...` | Malformed `Boundary` line | Check the syntax in [Directives](directives.md) |

## Refine errors

Grid refinement is under development and not validated; these messages are listed for reference only.

| Error message | Cause | Fix |
|---------------|-------|-----|
| `Missing name in Refine: ...` | `Refine { ... }` with no name | `Refine patch1 { region = [...] }` |
| `Missing { ... } block in Refine: ...` | No brace block | Add `{ region = [x0,y0,x1,y1] }` |
| `Missing 'region = [x0, y0, x1, y1]' in Refine: ...` | Region key missing | Add it |
| `Refine region must have 4 values: ...` | Wrong number of coordinates | Always 4 values: `[x0, y0, x1, y1]` |

## Setup errors

| Error message | Cause | Fix |
|---------------|-------|-----|
| `Unknown Setup key 'Reynolds' (did you mean: reynolds?)` | Case-sensitive key typo | Use the lowercase form |
| `Setup reynolds conflicts with Physics nu (both specified)` | Both dimensional and non-dim ν specified | Remove one: keep `Setup reynolds = ...` OR `Physics nu = ...` |
| `Setup rayleigh conflicts with Physics nu (both specified)` | Same as above, for Ra | Remove one |

## Sanity-check errors / warnings

| Message | Severity | Cause | Fix |
|---------|----------|-------|-----|
| `Sanity check failed: tau = 0.49 < 0.5 (unstable)` | error | `ν` too small → `τ < 0.5` | Increase `ν`, or reduce `Re`, or refine grid |
| `tau = 0.505 is very close to 0.5 (marginally stable)` | warn | `τ < 0.51` | Same as above — still runs |
| `Lattice velocity U_ref = 0.15 exceeds 0.1` | warn | Inlet velocity too high | Lower `ux` in the velocity BC, or refine grid |
| `Lattice velocity U_ref = 0.35 > 0.3 (CFL risk)` | warn | Even higher | Same fix, more urgent |

See the [Sanity](sanity.md) page for the full context.

## Preset errors

```
ArgumentError: Unknown Preset 'cavity2d' (did you mean: cavity_2d?)
```

Cause: typo in `Preset <name>`. The 5 known names are
`cavity_2d`, `poiseuille_2d`, `couette_2d`, `taylor_green_2d`,
`rayleigh_benard_2d`. See [Presets](presets.md).

## Sweep errors

```
ArgumentError: parse_kraken: got 12 setups (Sweep directive present?).
Use parse_kraken_sweep for sweeps.
```

Cause: calling the scalar `parse_kraken` / `load_kraken` on a file that
contains a `Sweep` directive. Use `parse_kraken_sweep` / `load_kraken_sweep`
— they always return a `Vector{SimulationSetup}`.

## Expression errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Function 'rand' is not allowed in expressions.` | Non-whitelisted call | Use a literal or a whitelisted function — see [Expressions](expressions.md) |
| `Unknown symbol 'U' in expression.` | Undeclared variable | `Define U = ...` or use a built-in (`x`, `y`, `t`, `Lx`, …) |
| `Disallowed expression type ':import' in: ...` | Tried to use Julia imports in a .krk expression | Not allowed — expressions are sandboxed |
| `ccall is not allowed in expressions` | Self-explanatory | — |

## Levenshtein suggestions in action

The heuristic lives in `_suggest_name`:

```julia
function _suggest_name(name::AbstractString, candidates)
    best = nothing
    best_d = typemax(Int)
    for c in candidates
        d = _levenshtein(lowercase(String(name)), lowercase(String(c)))
        if d < best_d
            best_d = d
            best = String(c)
        end
    end
    threshold = max(2, length(name) ÷ 3)
    return best_d <= threshold ? best : nothing
end
```

The threshold is `max(2, length(name) ÷ 3)`, so short words get distance 2
and longer words tolerate proportionally more. This means `Reynolds` →
`reynolds` (distance 1), `cavty_2d` → `cavity_2d` (distance 1), but a
completely unrelated token like `blorp` gets no suggestion.
