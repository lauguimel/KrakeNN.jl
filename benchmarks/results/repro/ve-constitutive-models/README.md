# Reproducibility bundle — viscoelastic constitutive models

Self-contained bundle for the **constitutive-models** page
(`docs/src/users/benchmarks/ve-constitutive-models.md`). Drives each of
Kraken's four 3D log-conformation constitutive models single-cell through two
homogeneous flows with the **real** Kraken constitutive solver (no hardcoded
model formulas) and renders the two-panel comparison figure in the locked dark
docs style (`krakendark`, `#1b1b1f`, LaTeX-if-available).

## Case

A single conformation cell `(1,1,1)` driven by an imposed, spatially-uniform
velocity gradient, integrated to its constitutive fixed point — the same
single-cell pattern as the constitutive canaries
`test/test_fvfd_{logconf,fenep,giesekus,ptt}_3d.jl`. Each step calls the actual
3D log-conformation kernel
(`logfv_constitutive_step_log_{,_fenep,_giesekus,_ptt}_3d!`) the coupled drivers
use; `C = exp(Ψ)` is reconstructed via `mat_exp_sym3x3`. With `G = 1`, `λ = 1`
the control parameters are read directly: `Wi = λγ̇` (shear) and `λε̇`
(extension).

| Flow | Imposed `gradU` | Control | Reported |
|------|-----------------|---------|----------|
| **Steady simple shear** | `γ̇·(e_x⊗e_y)` | `Wi = λγ̇ ∈ [0, 10]` | `N1 = G(C_xx−C_yy)`, `η_p,app = G·C_xy/γ̇`, `tr C` |
| **Steady planar extension** | `ε̇·diag(1,−1,0)` | `λε̇ ∈ [0, 0.49]` | `C_xx`, `tr C` |

The four models at representative non-Newtonian parameters:

| Series | Model | Parameter | Signature |
|--------|-------|-----------|-----------|
| `oldroydb` | `LogConfOldroydB` | — | constant viscosity; `N1 = 2 Wi²`; extension pole at `λε̇=0.5` (`C_xx=1/(1−2λε̇)`). |
| `fenep` | `LogConfFENEP` | `L²=50` | finite extensibility (`tr C < L²`); shear-thinning. |
| `giesekus` | `LogConfGiesekus` | `α=0.2` | quadratic mobility; shear-thinning; bounded extension. |
| `ptt` | `LogConfPTT` (linear) | `ε=0.25` | trace multiplier `Y=1+ε(tr C−3)`; shear-thinning; bounded extension. |

Each model is independently validated against its closed-form steady simple
shear (Giesekus/PTT residual ≤ 1e-6, FENE-P/OB-limit ≤ 1e-3) and the Oldroyd-B
limit (`α,ε→0` or `L²→∞`) is recovered **bit-identically** — see the canary
files above.

## Files

- `make_csv.jl` — drives all four models through both flows with the real
  Kraken solver and writes `ve_constitutive_models.csv`. Run (from repo root):
  `julia --project=. benchmarks/results/repro/ve-constitutive-models/make_csv.jl`.
  CPU Float64, ~1 min; CI-reproducible (no GPU).
- `ve_constitutive_models.csv` — the sweep:
  `flow, model, control, Cxx, Cxy, Cyy, Czz, trC, N1, eta_p, steps`.
- `plot.py` — self-contained figure reproducer (csv + matplotlib +
  `krakendark`). Reads the CSV next to it and writes the docs page PNG
  `docs/src/users/benchmarks/ve-constitutive-models.png`.
  Run `conda run -n kraken-v0-3-figures python plot.py`.

## In-flow (coupled Poiseuille) — sibling pair

Where the single-cell scripts above impose the velocity gradient, this pair
solves the **coupled** problem: the FVFD Poiseuille driver
`run_viscoelastic_fvfd_poiseuille_3d` runs the D3Q19 solvent and the
log-conformation polymer two-way coupled (`F_poly = ∇·τ_p`), so the flow sets
its own shear rate `γ̇(y)`. Same channel and operating point as the
constitutive-coupling canaries `test/test_fvfd_{fenep,giesekus,ptt}_coupled_3d.jl`
(periodic `x`/`z`, half-way bounce-back `y` walls, `β = ν_s/ν_total = 0.5`,
`λ` set so `Wi_wall ≈ 1`). Only the `polymer_model` spec changes between runs.

| Channel | `Nx×Ny×Nz` | `β` | `Wi_wall` | `Fx` | steps |
|---------|-----------|-----|-----------|------|-------|
| planar Poiseuille | `6×32×6` | 0.5 | ≈ 1 | `1.5e-5` | 10 000 |

- `make_poiseuille_csv.jl` — runs the 4 coupled drivers and writes the
  wall-normal `y`-profiles to `ve_poiseuille_profiles.csv`
  (`model, j, u, gamma, Cxx, Cyy, Czz, Cxy, trC, N1`). Run (from repo root):
  `julia --project=. benchmarks/results/repro/ve-constitutive-models/make_poiseuille_csv.jl`.
  CPU Float64, ~1–2 min; CI-reproducible (no GPU).
- `plot_poiseuille.py` — dark two-panel figure: `tr C(y)` (wall→centre) and the
  coupled `u(y)`. Writes the docs page PNG
  `docs/src/users/benchmarks/ve-constitutive-models-poiseuille.png`.
  Run `conda run -n kraken-v0-3-figures python plot_poiseuille.py`.

Near-wall stretch ranking (`tr C` at the wall plane, `Wi_wall ≈ 1`):
Oldroyd-B `5.00` > FENE-P `4.74` > PTT `4.44` > Giesekus `4.27`; all relax to
`tr C ≈ 3` at the low-shear core.

## Cross-slot strand (planar stagnation point) — third pair

The cross-slot's central birefringent **strand** is a planar stagnation-point
flow effect. With the imposed straining field `u = (ε̇·x, −ε̇·y, 0)` — outflow
along `±x`, inflow along `±y`, stagnation point at the centre — conformation
**advection** builds a high-stretch strand along the outflow axis while the
inflow axis relaxes toward `tr C = 3`, even though the imposed strain rate is
spatially uniform. This pair drives the real FVFD planar-extension solver
`run_viscoelastic_fvfd_extensional_3d` (`velocity_mode=:imposed`: the
conformation is advected + relaxed by the actual solver while the velocity is
held at the analytical stagnation field) for each model and writes the `z`-mid
`tr C(x, y)` field. Operating point matched to the validated extensional canary:
small `ε̇` in lattice units (so the coupled D3Q19 solvent step stays stable) with
a large `λ` giving `λε̇ = 0.35` — a strong Oldroyd-B stretch below the
coil-stretch pole `λε̇ = 0.5`.

| Field | `Nx×Ny×Nz` | `β` | `λε̇` | `ε̇` | `λ` | steps |
|-------|-----------|-----|-------|------|-----|-------|
| imposed planar stagnation | `32×32×4` | 0.5 | 0.35 | `0.005` | 70 | 1500 |

- `make_crossslot_csv.jl` — runs the 4 imposed-stagnation drivers and writes the
  `z`-mid `tr C(x, y)` field to `ve_crossslot_fields.csv`
  (`model, i, j, x, y, trC`; `x, y` are LU offsets from the stagnation point).
  Run (from repo root):
  `julia --project=. benchmarks/results/repro/ve-constitutive-models/make_crossslot_csv.jl`.
  CPU Float64, no GPU; CI-reproducible (~a few minutes).
- `plot_crossslot.py` — dark figure: a 2×2 grid of `tr C(x, y)` for the four
  models on a shared colour scale + a strand profile along the outflow axis.
  Writes the docs page PNG
  `docs/src/users/benchmarks/ve-constitutive-models-crossslot.png`.
  Run `conda run -n kraken-v0-3-figures python plot_crossslot.py`.

Strand-intensity ranking (peak `tr C` on the outflow axis, `λε̇ = 0.35`):
Oldroyd-B `4.71` > FENE-P `4.45` > PTT `3.94` > Giesekus `3.83`; outflow/inflow
anisotropy `1.30× / 1.23× / 1.11× / 1.10×`. Same ordering as the homogeneous
extension `C_xx` plateaus and the coupled Poiseuille near-wall stretch.

## Reference

R. Fattal, R. Kupferman (2004), *Constitutive laws for the matrix-logarithm of
the conformation tensor*, J. Non-Newtonian Fluid Mech. **123**, 281–285 — the
log-conformation representation shared by all four models.
