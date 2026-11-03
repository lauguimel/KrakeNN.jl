# Validation matrix

The single summary of where each public Kraken module stands against an external
reference. Every numeric cell is the headline metric of the linked benchmark page —
no value is reproduced here that is not on one of those pages. Each row gives the
**published/analytical reference**, an **independent other-solver comparison** where one
exists, and the **tolerance met** against the mandate §3 acceptance bar.

Where RheoTool covers a case it is the literal reference solver; where it does not, §3's
escape clause anchors the row to the canonical published data set (Ghia 1982 for the
cavity, de Vahl Davis 1983 for thermal, Clift et al. 1978 for sphere drag), each with an
independent OpenFOAM cross-check where available.

## Matrix

| Module | Analytical / published reference | Other-solver comparison | Tolerance target (§3) | Result actually met | Artifact |
|---|---|---|---|---|---|
| **Newtonian — lid-driven cavity** (D2Q9 BGK) | Ghia, Ghia & Shin (1982), centreline `u`/`v` | OpenFOAM `icoFoam` (FVM); independently clears < 1 % vs Ghia (0.49 / 0.23 / 0.46 %) | rel-L2 < 1 % on u-centreline (strict); ≤ 5 % local field | **PASS** — rel-L2 **0.47 % / 0.41 % / 1.05 %** at Re = 100 / 400 / 1000 (Re = 1000 marginal, abs-RMS 0.43 %) | [Cartesian cavity](cartesian-cavity.md); literal both-solver bundle `benchmarks/results/rheotool_compare/newtonian/` (`.krk` + CSVs + `plot.py` + figure + README) |
| **Thermal — natural convection** (double-distribution D2Q9) | de Vahl Davis (1983), `Nu` at Pr = 0.71 | OpenFOAM `buoyantBoussinesqSimpleFoam` (FVM SIMPLE); corroborates dVD at Ra = 10³ (−0.56 %) and 10⁵ (−0.60 %) | < 1 % on Nu; < 2 % on velocity extrema | **PASS** — Nu err **+0.79 % / +0.93 % / +0.79 %** at Ra = 10³ / 10⁴ / 10⁵; velocity extrema all < 1 % | [Thermal natural convection](thermal-natural-convection.md); reproducible bundle (`.krk` + CSV + plot script) forthcoming |
| **Geometry / STL — 3D sphere drag** (D3Q19 + LI-BB cut-link) | Clift, Grace & Weber (1978), free-stream `C_d ≈ 2.61` at Re = 20 | none (no RheoTool unbounded-sphere case); blockage extrapolation `D/W → 0` is the internal cross-check | ≤ 5 % local; resolution-limited geometry gate | **+8.9 %** — extrapolated free-stream `C_d ≈ 2.84` vs Clift 2.61 (quadratic LSQ, R² = 0.9998); residual is finite lattice resolution | [Sphere drag 3D](sphere-drag-3d.md); CSV `bench/geometry_stl/sphere_drag_conv*.csv` |
| **Grid refinement (patch and conservative AMR)** | not applicable | not applicable | not applicable | **UNDER DEVELOPMENT, NOT VALIDATED** — not part of the supported feature set. See [Limitations](../../theory/10_limitations.md). | — |
| **Non-Newtonian — 3D generalised-Newtonian channel** (D3Q19 TRT + LI-BB + Guo) | exact planar-channel solution from the stress balance, independent of the solver (Carreau, power law n = 0.7 and 1.5) | none | observed order ≥ 1.8 on each resolution pair | **PASS** — CPU Float64, N = 16 → 32 → 64: order **2.000 / 2.000** (Carreau, wall fraction 0.5), **1.990 / 1.995** (Carreau, 0.3), **2.000 / 2.000** (power law n = 0.7), **1.932 / 1.950** (n = 1.5). One A100 Float64 sweep, N = 32 → 256: orders **1.950 to 2.001** (power law n = 0.7 only to N = 128); Float32 run-through only. Isothermal, Julia API only (no `.krk` route) | [3D GNF channel](gnf3d-channel-convergence.md); `test/analytical/gnf3d_channel_order.jl` |
| **Viscoelastic — Oldroyd-B cylinder** (TRT-LBM, halfway BB) | none published at this confinement (Oldroyd-B `C_d` is solver-dependent) | RheoTool `rheoFoam` fine-mesh (`Cylinder/Oldroyd-BLog`), log-conformation | < 1 % on `C_d` at Wi ≤ 1, R = 50, Re = 1 | **PASS** — `C_d` within **−0.37 % / −0.66 % / −0.75 %** of rheoFoam at Wi = 0.1 / 0.5 / 1.0, β = 0.59 (diffusive scaling τ = 0.95, halfway BB; re-measured 2026-10-14 after the #67 forcing fix); validated only for Wi ≤ 1 | [Viscoelastic cylinder](../tutorials/viscoelastic-cylinder.md) |
| **GPU — single-GPU efficiency** (BGK D2Q9, CUDA F64) | memory-bandwidth roofline (A100-40GB, 1.555 TB/s, 144 B per D2Q9 F64 update ⇒ 10 799 MLUPS ceiling) | Palabos single-GPU F64 D3Q19 TRT (4656 MLUPS, published) | ≥ 0.5 of roofline | **BELOW GATE** — **3461 MLUPS** at N = 2048 ⇒ **0.32** of the A100-40GB roofline (0.24 of the 80 GB ceiling); 0.74 of the Palabos number, which is D3Q19 and not like-for-like | [GPU certification](gpu-certification.md); CSV `benchmarks/results/certification_a100.csv` |
| **GNF steady adjoint — Carreau–Yasuda gradient** (D2Q9, 32×16 periodic obstacle channel, discrete adjoint without Enzyme) | none (the check is internal) | central finite differences of the objective; on CPU also the transposed product against Enzyme reverse mode | adjoint-vs-FD gate 2.1e-2 to 6.2e-2 (absolute, loose); CPU/GPU operator parity ≤ 1e-14 relative | **PASS** — one A100 (Float64): gradient vs FD **9.4e-10 / 3.7e-11 / 7.1e-08 / 2.0e-09** (absolute, four parameters); device vs CPU products 1.2e-16 / 8.7e-17 relative; CPU vs Enzyme 3.6e-16 | [GNF adjoint gradient](gnf-adjoint-gradient.md); CSV and log `benchmarks/validation/gnf_adjoint/results/` |

## Notes

- **Follow the artifact link** for the full error-norm tables, methodology, mesh/backend
  details and caveats — the matrix carries only the headline metric.
- **3D rows carry larger residuals** and are documented as resolution-limited on their
  pages (sphere drag +8.9 %). 3D natural convection is wired (`natural_convection_3d`
  preset) but only smoke-validated; a quantitative 3D thermal benchmark is a future item.
  The 2D rows are the strict acceptance gates.
- **Grid refinement is not part of the validated surface.** Grid refinement (patch and conservative AMR) is under development and not validated; it is not part of the supported feature set. The conservative-tree AMR drag coefficient is off by a factor 1.06 to 5.6 against the reference, and patch-refined thermal walls give Nu 1.26 to 1.30 in pure conduction instead of 1.0 (issue #98). See [Limitations](../../theory/10_limitations.md).
- **Literal reference-solver backfill.** The Newtonian cavity row ships a complete
  `benchmarks/results/rheotool_compare/newtonian/` bundle (both-solver CSVs, per-Re error
  norms, the `.krk`, and a self-contained `plot.py`). The thermal and viscoelastic
  literal bundles are the next legs of the same backfill.
