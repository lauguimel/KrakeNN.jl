# Reproducibility bundle — FVFD-3D viscoelastic Poiseuille convergence

Self-contained bundle for the **viscoelastic Poiseuille** benchmark page
(`docs/src/users/benchmarks/ve3d-poiseuille-convergence.md`). Regenerates the
two-panel convergence figure in the locked dark docs style (`krakendark`,
`#1b1b1f`, LaTeX-if-available).

## Case

Planar Poiseuille flow of an Oldroyd-B fluid (`β = 0.5`, `Wi_wall = 0.5`),
periodic in `x`/`z`, no-slip walls in `y`. The shear conformation
`C_xy = λ·γ̇(y)` is sharply curved near the wall — the cleanest demonstration of
the **#2B cure**: Kraken's **FVFD log-conformation** polymer transport reproduces
the near-wall conformation **machine-exactly**, where the diffusive **LBM-CDE**
path on the *same case* is **25.9 % off near-wall**.

| Series | Method | Notes |
|--------|--------|-------|
| **N_y = 32 / 64 / 128 sweep** | Kraken FVFD log-conf, CUDA F64 | the mesh-convergence series, Aqua H100. |
| **LBM-CDE reference** | diffusive lattice-Boltzmann conformation transport | flat 25.9 % near-wall / +13 % u-overshoot from the `N_y = 32` payoff canary. |

## Files

- `ve3d_poiseuille_sweep.csv` — the Aqua H100 / CUDA F64 sweep:
  `ny, near_wall_Cxy_err_abs, near_wall_Cxy_err_metric, u_ratio, nan_free,
  lambda, Wi_wall, steps, time_s`.
- `plot_ve3d_poiseuille_sweep.py` — self-contained reproducer (csv + matplotlib
  + `krakendark`). Reads the CSV next to it and writes the docs page PNG
  `docs/src/users/benchmarks/ve3d-poiseuille-convergence.png`.
  Run `conda run -n kraken-v0-3-figures python plot_ve3d_poiseuille_sweep.py`.

The GPU sweep itself is **not** CI-reproducible (requires a GPU run); the FVFD-vs-LBM
near-wall cure is asserted in CI by the payoff canary
`test/test_fvfd_poiseuille_payoff_3d.jl` (coarse `N_y = 32`, CPU Float64).

## Reference

R. Fattal, R. Kupferman (2004), *Constitutive laws for the matrix-logarithm of the
conformation tensor*, J. Non-Newtonian Fluid Mech. **123**, 281–285.
