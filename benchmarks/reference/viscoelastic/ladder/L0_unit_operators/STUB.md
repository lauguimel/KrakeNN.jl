# L0 — unit-operator tests (STUB)

Status: **STUB**. Not implemented in mission M38; will be promoted in a
follow-up mission.

## What L0 will test

Algebraic / 1-cell / no-pipeline unit checks of the basic operators on
which the viscoelastic stack depends. Pass criterion is exact identity
(atol 1e-12) since there is no advection or BC involved.

| Operator                          | Reference                  | Already-existing test(s)                                |
|-----------------------------------|----------------------------|---------------------------------------------------------|
| `exp` / `log` of SPD 2×2 sym2     | analytic eigen             | `test/test_logconformation.jl`, `test/test_viscoelastic_logfv_patch_ladder.jl` |
| Ψ ↔ C round-trip                  | identity                   | `test/test_logconformation.jl`, `test/test_logconformation_3d.jl`              |
| Oldroyd-B stress closure τ=G(C-I) | algebraic                  | `test/test_viscoelastic_equations.jl`                                          |
| Loewner direct-C vs log-conf source | self-consistency         | `test/test_viscoelastic_equation_patch_ladder.jl`                              |
| FVFD div / grad / stress accumulator | analytic discrete identity | `test/test_fvfd_operators_2d.jl`                                            |
| Hermite source 2nd-moment closure | algebraic                  | `test/test_viscoelastic_force_accounting.jl` "standalone Hermite source has exact bulk moment closure" |
| Hermite CE-correction factor       | analytic 1/(1-s/2)         | `test/test_viscoelastic_force_accounting.jl` "standalone source is larger than in-collision Liu source by CE factor" |

## Promotion plan

These tests **already exist** under `test/` and pass under the standard
`Pkg.test()` invocation. The L0 promotion task is to:

1. Mirror them as `bench/viscoelastic_validation/L0_unit_operators/run.jl`
   that *invokes* the existing test files (via `include`) and reports a
   single dashboard line per family, rather than duplicating the test
   code.
2. Cross-link from the L0 dashboard back to the test files so a failure
   surfaces "FAIL: exp/log SPD round-trip; see test/test_logconformation.jl".

No new `src/` API needed. No new analytic reference needed (everything is
algebraic identity).

## Cost target

< 1 s wall-clock for the dashboard.
