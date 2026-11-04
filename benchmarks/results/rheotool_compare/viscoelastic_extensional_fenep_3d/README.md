# RheoTool reference — FENE-P planar extension (mandate §3.2 extensional module)

Third-party numerical reference for Kraken's 3D FVFD **FENE-P** extensional
module (`run_viscoelastic_fvfd_extensional_3d` with `LogConfFENEP`), the
finite-extensibility analogue of the Oldroyd-B planar-extension bench in
`../viscoelastic_extensional_3d/`.

## Case

A **uniform planar-extension box**: `U = (ε̇·x, −ε̇·y, 0)`, advancing the FENE-P
conformation tensor to its steady state. As for Oldroyd-B this is realised with
RheoTool's **`rheoTestFoam`** — a single-cell material tester imposing the
*homogeneous* deformation `gradU = ε̇·diag(1,−1,0)`, the cleanest analogue of
Kraken's uniform-strain box (no geometry mismatch, no stagnation point, no
coil-stretch residence-time gradient).

Operating point (matched to Kraken's FENE-P extensional canary,
`test_fvfd_fenep_extensional_3d.jl` gate G2):

| param | value |
|-------|-------|
| `λ` (relaxation time) | 50 |
| `ε̇` (extension rate) | 0.005 |
| `2·λ·ε̇` (planar coil-stretch number) | 0.5 |
| `β = η_s/(η_s+η_p)` | 0.5 (`η_s=η_p=0.05`) |
| `L²` (FENE extensibility) | 50 |

## No closed form — a genuine cross-validation

FENE-P planar extension has **no closed-form fixed point** (transcendental in
`tr C`), so this is a RheoTool-vs-Kraken **cross-validation** at matched
parameters — both codes numerical. Exactly what §3.2 asks for.

## The two FENE-P closures are NOT identical

A key, honestly-reported finding: RheoTool and Kraken implement **different
FENE-P Peterlin closures**.

- **RheoTool** (`FENE_P.C`, `solveInTau=false`): transports a conformation `A`
  with equilibrium `A = a·I`, `a = L²/(L²−3)`, and Peterlin factor
  `varf = 1/(1 − tr A / L²)`; `τ_p = (η_p/λ)(varf·A − a·I)`.
- **Kraken** (`logconformation_lbm_3d.jl`): transports `C` with equilibrium
  `C = I` and Peterlin factor `f = (L²−3)/(L²−tr C)`; `τ_p = (η_p/λ)(f·C − I)`.

These are two standard FENE-P variants that differ in the Peterlin *argument*
(`tr A` vs `tr C`, with the `a`-rescaling `A = a·C`). They **coincide only as
`L²→∞`** (both reduce to Oldroyd-B). The conformation below is reported in
Kraken's convention (`C = A/a`).

## Result — RheoTool FENE-P vs Kraken FENE-P (Kraken-convention `C`)

RheoTool's `rheoTestFoam` returns the *total* extra-stress
`τ_total = τ_p + η_s·(L+Lᵀ−⅔ tr(L) I)`; with `L = ε̇·diag(1,−1,0)` the polymer
stress is `τ_p = τ_total − η_s·2ε̇·diag(1,−1,0)`, from which `A` (and `C=A/a`)
is recovered by inverting RheoTool's Peterlin relation. The reconstruction is
validated below by the OB-limit run.

| quantity | Kraken FENE-P | RheoTool FENE-P | difference |
|----------|---------------|-----------------|------------|
| `C_xx` | 1.9497 (canary) / 1.9497 (transcendental) | 1.7374 | **10.9 %** |
| `C_yy` | 0.6610 (transcendental) | 0.6347 | 4.0 % |
| `C_zz` | 0.9873 (transcendental) | 0.9297 | 5.8 % |
| `tr C`  | 3.5980 (canary) / 3.5980 (transcendental) | 3.3019 | **8.3 %** |

Kraken's own code is faithful: its 1000-step canary (`C_xx=1.949729`,
`tr C=3.597994`) matches its own steady-state transcendental solution
(`C_xx=1.949745`, `tr C=3.598009`) to **1e-5** relative (measured 2026-09-24
with the periodic-z advection fix of #54; before it the canary read
`C_xx=1.944`, `tr C=3.60`, 0.3 % off, because the advection fell back to first
order in 4 of the 6 z layers). The ~11 % `C_xx` gap to RheoTool is therefore a
**genuine constitutive-closure difference**, not a bug or a discretisation
error in either solver — both bound the stretch below `L²=50` (`tr C < L²`).

## OB-limit sanity — validates the reconstruction

Running the *same* RheoTool case with `L²=10⁵` (Hookean spring) returns
`C_xx = 1.99985`, i.e. **0.007 %** from the analytic Oldroyd-B planar fixed
point `C_xx = 1/(1−2λε̇) = 2`. This confirms (i) the solvent-subtraction +
Peterlin-inversion reconstruction pipeline is correct, and (ii) both codes'
FENE-P models share the same Oldroyd-B limit; the finite-`L²` gap is purely the
closure-variant difference.

## Files

- `rheotool_fenep_extensional_transient.csv` — RheoTool `rheoTestFoam` transient
  `C(t)` (downsampled), reconstructed in Kraken's convention.
- `viscoelastic_extensional_fenep_3d_error_norms.csv` — headline cross-validation
  table.
- `make_csv.jl` — reproducer: reads the `Report`, reconstructs `C`, writes both
  CSVs. Run with
  `julia make_csv.jl ../../../../bench/rheotool/extensional_fenep_ve_planar/Report`.
- `plot.py` — self-contained figure reproducer (dark Documenter style). Run with
  `conda run -n kraken-v0-3-figures python plot.py`.
- `viscoelastic_extensional_fenep_3d_compare.png` — the figure.

The RheoTool case template lives at
`bench/rheotool/extensional_fenep_ve_planar/` (clean: `0/`, `constant/`,
`system/`, `Allrun` — no solved time dirs or logs). Run it with the local
Docker image `openfoam9-rheotool:v1.2`:

```sh
docker run --rm --platform linux/amd64 --entrypoint /bin/bash \
  -v "$PWD:/case" openfoam9-rheotool:v1.2 -c \
  'source /opt/openfoam9/etc/bashrc; \
   export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH; \
   cd /case && blockMesh && rheoTestFoam'
```

The conformation appears in the `Report` file (columns
`t extStressXX extStressXY extStressXZ extStressYY extStressYZ extStressZZ`),
which is the *total* extra-stress; subtract the solvent contribution and invert
RheoTool's Peterlin relation (see `make_csv.jl`) to recover the conformation.

## Reference

F. Pimenta, M.A. Alves (2017), *Stabilization of an open-source finite-volume
solver for viscoelastic fluid flows*, J. Non-Newtonian Fluid Mech. **239**, 85–104.
