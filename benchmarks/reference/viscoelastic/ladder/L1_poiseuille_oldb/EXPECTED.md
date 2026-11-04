# L1 — expected behaviour & failure-mode diagnosis

`compare.jl` reports per-quantity PASS / FAIL. This document explains what
each verdict means and what to inspect when a check fails.

## PASS state

All ten checks green. Wall-clock should be < 2 min on a single CPU core.
Each `tau_*_profile_interior_relL2` should typically land an order of
magnitude under its threshold; the threshold leaves room for grid coarseness
without going into the noise floor.

A clean PASS confirms:
1. constitutive law correct (`tau_xy`, `tau_xx`),
2. body-force discretisation + BSD coupling correct (`u_centerline`,
   `u_profile`),
3. wall BC correct (wall-pair quantities),
4. continuity / density drift acceptable (`rho`, `uy_interior`),
5. log-conformation SPD invariant maintained (`min_eig_C`).

## FAIL diagnosis matrix

### `u_centerline_relative` FAIL but `tau_xy_*` PASS

The constitutive equation is fine but the momentum coupling is wrong.

Likely causes:
- `nu_total = nu_s + nu_p` ≠ `nu_lbm = nu_s + bsd · nu_p` mismatch in the
  analytic reference (check `_logfv_lbm_poiseuille_reference` vs the
  Bird-AH formula).
- BSD fraction parsed as `1 - bsd` somewhere.
- Wrong sign in the polymer-force divergence.

### `tau_xy_*` FAIL but `tau_xx_*` PASS

Wrong polymer viscosity scaling somewhere downstream of ψ. Likely:
- `prefactor = nu_p / lambda` vs `etaP / lambda` mismatch in lattice units
  (ρ = 1 in LBM, so these coincide, but a unit-confusion can creep in).
- BSD splitting taking the wrong fraction of `nu_p`.

### `tau_xx_*` FAIL but `tau_xy_*` PASS

The constitutive law is fine in the lower (linear) moment but wrong in the
quadratic. Likely:
- Hermite CE-correction factor `1/(1 - s/2)` missing or doubled (see
  `test_viscoelastic_force_accounting.jl` "standalone source is larger
  than in-collision Liu source by CE factor").
- λ used inconsistently (e.g. λ_lu vs λ_physical), giving wrong
  τ_xx = 2 λ ν_p γ̇² scaling.
- Polymer substepping insufficient (Wi · n_sub too small to resolve the
  conformation eigenstructure). Increase `polymer_substeps`.

### `min_eig_C < 0.8`

The conformation tensor is going singular. The Oldroyd-B steady solution has
C_xx = 1 + 2(λ γ̇)² > 1, C_yy = 1, C_xy = λ γ̇; the minimum eigenvalue is
always between 1 (γ̇ → 0) and a smaller positive number that decreases with
Wi. At Wi ≈ 7e-3 (our L1 setpoint), min eig should be > 0.99.

- If << 0.99 but > 0.8: probable polymer substepping issue or BSD
  miscoupling driving the conformation off the SPD manifold transiently.
- If between 0.8 and 1: borderline; tighten substepping.

### `rho_max_abs_deviation > 1e-3`

LBM density drift. Causes:
- BC mismatch (wall ghost cells leaking).
- Body force not zero-mean over a full period (here the geometry is
  periodic; F_body is a uniform constant, integral momentum drift is
  compensated by HWBB walls, but only if BSD doesn't add a residual).
- BSD correction term not divergence-free at the discrete level.

### `uy_interior_max_abs > 5e-6`

Spurious transverse flow. Causes:
- Polymer force has a y-component (it should be zero for this 1D problem).
- BC asymmetry between top and bottom walls.
- Hermite source incorrectly transposed.

### `no_nan_no_inf` FAIL

Hard fail; trace back through the polymer subcycle. Likely cause: λ too
large for the chosen substep count, conformation goes singular, ψ → -∞ in
some component. Reduce λ or increase substeps.

## How to diagnose live (≤ 5 min)

```julia
using JSON3
payload = JSON3.read(read("results/L1_run_latest.json", String))
diag = payload.diagnostics
# Look at profiles for the failing quantity
prof = payload.profiles
# Plot diagnostics — never form a hypothesis without 6 plots:
#   mesh    quiver    u(y)    rho(y)    tau_xy(y)    tau_xx(y)
# (feedback_always_plot rule)
```

## Linked references

- `bench/viscoelastic_validation/REFERENCES.md` §1 (Bird-Armstrong-Hassager,
  Waters & King)
- `test/test_viscoelastic_force_accounting.jl` (Hermite CE factor)
- `test/test_logfv_frozen_channel_cde.jl` (frozen-velocity sanity)
