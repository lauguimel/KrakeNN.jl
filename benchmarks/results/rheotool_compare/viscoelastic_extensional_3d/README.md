# RheoTool reference — Oldroyd-B planar extension (mandate §3.2 extensional module)

Third-party numerical reference for Kraken's 3D FVFD **extensional** module
(`run_viscoelastic_fvfd_extensional_3d`), analogous to the §3.2 Poiseuille bench.

## Case

A **uniform planar-extension box**: `U = (ε̇·x, −ε̇·y, 0)`, advancing the
Oldroyd-B conformation tensor to its analytic fixed point. This is realised in
RheoTool with **`rheoTestFoam`** — a single-cell material tester that imposes a
*homogeneous* deformation `gradU = ε̇·diag(1,−1,0)`. This is the cleanest
possible analogue of Kraken's uniform-strain box: no geometry mismatch, no
stagnation point, no residence-time / coil-stretch gradient (unlike a cross-slot,
which RheoTool also ships but which has spatially-varying strain).

Operating point (matched to Kraken's `extensional_oldroyd_b.krk`):

| param | value |
|-------|-------|
| `λ` (relaxation time) | 50 |
| `ε̇` (extension rate) | 0.005 |
| `λ·ε̇` (Wi) | 0.25 |
| `β = η_s/(η_s+η_p)` | 0.5 (`η_s=η_p=0.05`) |

## Analytic Oldroyd-B planar-extension fixed point (2·λ·ε̇ < 1)

`C_xx = 1/(1−2λε̇) = 2`,  `C_yy = 1/(1+2λε̇) = 2/3`,  `C_zz = 1`.

## Result — RheoTool ≡ Kraken ≡ analytic

RheoTool's `rheoTestFoam` returns the *total* extra-stress
`τ_total = τ_p + η_s·(L+Lᵀ−⅔ tr(L) I)`; with `L = ε̇·diag(1,−1,0)` the conformation
is recovered as `C = I + (λ/η_p)·(τ_total − η_s·2ε̇·diag(1,−1,0))`.

| component | analytic | RheoTool (steady, t=40λ) | rel. err | Kraken FVFD (1000 steps) | rel. err |
|-----------|----------|--------------------------|----------|--------------------------|----------|
| `C_xx` | 2.0000000 | 2.0000000 | 0.0 % | 1.9999546 | 0.0023 % |
| `C_yy` | 0.6666667 | 0.6666667 | <1e-5 % | 0.6666667 | 5e-12 % |
| `C_zz` | 1.0000000 | 1.0000000 | 0.0 % | 1.0000000 | 0.0 % |

**RheoTool reaches the analytic fixed point to machine precision.** Kraken's
1000-step canary is 0.0023 % from it on `C_xx`, and exact to round-off on
`C_yy` and `C_zz` (CPU, Float64, measured 2026-09-24 with the fix of #54).

Before #54 this table read `C_xx = 1.9922628` (0.39 %) and `C_yy = 0.6666768`
(0.0015 %), and the gap was attributed to a slow coil-stretch relaxation at a
finite horizon. That explanation was wrong: the MUSCL-Superbee advection fell
back to first-order upwind within two cells of the z faces although z is
periodic, i.e. in 4 of the 6 z layers of this box. With the periodic z stencil
wrapped instead, the same 1000-step run gives the values above.

## Files

- `rheotool_extensional_transient.csv` — RheoTool `rheoTestFoam` transient
  `C(t)` (downsampled), reconstructed from the total extra-stress.
- `viscoelastic_extensional_3d_error_norms.csv` — the headline comparison table.
- `plot.py` — self-contained reproducer (reads the transient CSV, regenerates the
  PNG in the dark Documenter style). Run with
  `conda run -n kraken-v0-3-figures python plot.py`.
- `viscoelastic_extensional_3d_compare.png` — the figure.

The RheoTool case template lives at `bench/rheotool/extensional_ve_planar/`
(clean: `0/`, `constant/`, `system/`, `Allrun` — no solved time dirs or logs).
Run it with the local Docker image `openfoam9-rheotool:v1.2`:

```sh
docker run --rm --platform linux/amd64 --entrypoint /bin/bash \
  -v "$PWD:/case" openfoam9-rheotool:v1.2 -c \
  'source /opt/openfoam9/etc/bashrc; \
   export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH; \
   cd /case && blockMesh && rheoTestFoam'
```

The conformation appears in the `Report` file (columns
`t extStressXX extStressXY extStressXZ extStressYY extStressYZ extStressZZ`).

## Reference

F. Pimenta, M.A. Alves (2017), *Stabilization of an open-source finite-volume
solver for viscoelastic fluid flows*, J. Non-Newtonian Fluid Mech. **239**, 85–104.
