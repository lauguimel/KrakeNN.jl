# L1 — planar Poiseuille Oldroyd-B (full pipeline, analytic reference)

## Purpose

End-to-end (LBM + log-FV + Hermite source + BSD coupling) test of the
viscoelastic stack on the cheapest case with a closed-form analytic
reference: pressure-driven planar Poiseuille of an Oldroyd-B fluid at
steady state.

This is the canonical regression sentinel for the full pipeline. Any
constitutive bug, body-force discretisation bug, wall-BC bug, or Hermite
CE-correction sign error must show up here — and only here — before being
allowed to manifest in confined-cylinder Cd.

## Geometry & discretisation

- Domain: periodic in x, walls at j = 1 and j = Ny.
- Grid: **Nx = 8, Ny = 32**. Periodic-in-x means a narrow slice suffices;
  Ny = 32 resolves the analytic parabola + the τ_xx ∝ y² curve without
  artefacts. Wall walltime budget on a single CPU core ~ 1-2 min for
  10000 LBM steps with default polymer subcycling.
- Lattice: D2Q9.
- BC: HWBB at top/bottom walls; periodic in x; body force `Fx` drives the
  flow.

**No symmetry plane**: we explicitly avoid halving the domain by symmetry
because the polymer-stress symmetry trap (τ_xy = 0 at the plane, but
mirrored ghost cells do not enforce this for the polymer kernel) is a known
false-positive class.

## Parameters (lattice units)

| Parameter            | Symbol | Value      | Notes |
|----------------------|--------|------------|-------|
| Solvent viscosity    | ν_s    | 0.04       | LBM ν                       |
| Polymer viscosity    | ν_p    | 0.06       | sets β = ν_s/(ν_s+ν_p) = 0.4 |
| Body force           | Fx     | 5e-6       | sets γ̇ ~ 1.5e-3 ⇒ Wi small  |
| Polymer relaxation   | λ      | 5.0        | gives Wi = λ γ̇_max ≈ 7e-3   |
| BSD fraction         | bsd    | 1.0        | full BSD splitting           |
| Polymer substeps     | n_sub  | `:auto`    | driver picks ≥ 8             |
| Max steps            | nsteps | 10000      | converges to steady state    |
| Backend              |        | `CPU()`    | F64                          |

Re ≈ 1, low Wi to ensure the steady analytic limit is reachable with a
modest number of LBM steps (the Wi = 1 transient is what L2 will cover).

## Driver

Uses the public API `Kraken.run_viscoelastic_logfv_poiseuille_coupled_2d`
(`src/drivers/viscoelastic_logfv_2d.jl` line 2287). This is the
production code path for body-force Poiseuille with full pipeline; the run
returns `ux`, `uy`, `rho`, `ψxx/ψxy/ψyy` (log-conformation), and
`reference_ux` (analytic Newtonian-equivalent parabola from the body
force).

The L1 wrapper additionally re-derives the analytic τ_xx, τ_xy, τ_yy
from the body force using the Bird-Armstrong-Hassager steady solution
(see `REFERENCES.md` §1) and reports relative errors per quantity.

## Expected wall-clock

CPU F64, 8 × 32 cells, 10000 LBM steps with ~ 8 polymer substeps per LBM
step: ~ 30-60 s on a single core. The `< 2 min` budget gives a safety
margin.

## Assertions (codified in `compare.jl`)

| Quantity                  | Reference                    | Threshold      |
|---------------------------|------------------------------|----------------|
| `u_centerline`            | Fx · h² / (2 · ν_total)      | relL2 < 5e-3   |
| `tau_xy_wall`             | ν_p · γ̇_wall                 | relL2 < 5e-2   |
| `tau_xx_wall`             | 2 · λ · ν_p · γ̇_wall²        | relL2 < 5e-2   |
| `max abs(rho - 1)`        | n/a                          | abs < 1e-3     |
| `max abs(uy_interior)`    | 0 (1D symmetry of Poiseuille)| abs < 5e-6     |
| `min eig(C)`              | (SPD; > 0.99 for low Wi)     | > 0.8          |
| `no NaN / no Inf`         | sentinel                     | hard fail      |

Thresholds reflect the LBM half-cell wall offset (the dominant error
floor for a steady-state body-force test) plus the second-order BSD
discretisation. They are tight enough to catch the historical failure
modes flagged in `README.md` §4.

## Pitfalls / debugging hints

- If `u_centerline` is off by a multiplicative factor: check ν_total =
  ν_s + ν_p in the analytic reference vs the LBM ν_lbm = ν_s + bsd·ν_p.
- If `tau_xy_wall` is off but `u_centerline` is OK: the constitutive
  equation works but the Hermite source has the wrong CE correction
  factor (see `test_viscoelastic_force_accounting.jl` "standalone source
  is larger than in-collision Liu source by CE factor" testset).
- If `tau_xx_wall` is off but `tau_xy_wall` is OK: integration in time
  has not reached steady state — increase `max_steps`, or reduce λ.
- If `min eig(C) < 0.8`: polymer is going singular; reduce Fx, or check
  BSD splitting.

## Out of scope

- Transient validation (L2's job).
- Wi > 1 (HWNP regime — out of scope for an L1 sentinel; covered by
  bench/viscoelastic_audit for diagnostics).
- Grid-convergence order verification (a Phase B follow-up could add a
  16² → 32² → 64² ladder; L1 keeps a single point for CI speed).
