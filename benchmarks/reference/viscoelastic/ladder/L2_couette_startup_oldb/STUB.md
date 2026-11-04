# L2 — Couette start-up Oldroyd-B (STUB)

Status: **STUB**. Not implemented in mission M38; will be promoted in a
follow-up mission.

## Why this level

L1 only tests the steady solution. Steady-state passes do NOT prove the
time integrator. A transient case with damped oscillations (the signature
of viscoelasticity) catches sign errors and time-discretisation bugs that
L1 misses.

Couette start-up is the cheapest such case: 1D shear, single wall jump,
analytic transient series available, no body force needed.

## Reference

Waters & King (1970), *Rheologica Acta* 9 (3), 345-355. See
`ref/waters_king_1970_couette.json` for the formula structure (the
Poiseuille variant; the Couette variant shares the (α_n, β_n, γ_n)
eigenvalue structure and differs in the inhomogeneous part).

Basilisk implementation reference at
`/Users/guillaume/Documents/Recherche/Codes CFD/basilisk/src/test/poiseuille-oldroydb.c`
lines 106-119 (Poiseuille — adapt for Couette by changing the
inhomogeneous coefficient).

## Design sketch

- Geometry: `Nx = 4`, `Ny = 32`, periodic in x, walls top/bottom.
- BC: bottom wall fixed at u = 0; top wall jumps from u = 0 to u = U_wall
  at t = 0 (Heaviside).
- Parameters: β ∈ {0.1, 0.5, 0.9}, Wi = λ · U_wall / H ∈ {0.5, 1.0}.
- Driver hook needed: Kraken already exposes
  `run_viscoelastic_logfv_poiseuille_coupled_2d` for body-force flow.
  For Couette we need an analogous driver with a top-wall velocity BC
  but no body force. Likely cheap to add (small variation on the
  Poiseuille coupled driver).
- Reference values: compute Waters-King series to KF = 8 modes; sample at
  centre-channel u_x(y=H/2, t) over t ∈ [0, 5 λ] in 50 points.

## Assertions sketch

- `u(y=H/2, t)` time series relL2 vs Waters-King < 5e-3 at Ny = 32.
- Damping envelope of the oscillation matches α_n decay rate (one extra
  diagnostic check).
- `min eig(C)` stays > 0.6 (lower than L1 because the start-up transient
  can dip).
- No NaN / Inf.

## Cost target

< 30 s on a single CPU core for one (β, Wi) point. The full sweep
(3 × 2 = 6 points) target is < 2 min.

## Out of scope until L2 promotion

- Multi-mode comparisons (FENE-P, PTT).
- High Wi (HWNP regime).
