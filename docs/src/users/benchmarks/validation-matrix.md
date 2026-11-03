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
| **AMR — conservative-tree (route-native D)** | analytic mass conservation (zero net flux ⇒ roundoff drift) | dense-leaf "oracle" Cartesian reference at finest leaf-equivalent resolution | mass drift at roundoff (< 1e-12); route-native obstacle Cd vs oracle | **PASS (conservation)** — relative mass drift **~1.7e-13** (square/cylinder), all obstacle rows < 1e-12 (worst 2.08e-12). Cylinder Cd vs the dense-leaf oracle depends on patch placement: with an **interface-buffered** patch (interfaces moved off the boundary layer) route-native Cd is **1.06× / 1.05× the oracle** at scale 1 / 2; with a **compact patch** cutting through the near-wall region the gap is large (route-native / oracle Cd ≈ **3.2×–5.6×**). Accuracy on integrated quantities is an **OPEN v0.2.0 item**, not a passing gate — see showcase | [Refinement showcase](../../benchmarks/refinement_showcase.md); compact-patch CSVs `benchmarks/results/repro/refinement/amr_obstacle_convergence_2d_aqua_conv_20757949.csv` (+`…20761330.csv`); interface-buffered CSV `benchmarks/results/repro/refinement/amr_obstacle_convergence_2d_aqua_interface_buffered_20260505.csv`; design note `docs/design/amr_d_publication_validation.md` |
| **Non-Newtonian — 3D generalised-Newtonian channel** (D3Q19 TRT + LI-BB + Guo) | exact planar-channel solution from the stress balance, independent of the solver (Carreau, power law n = 0.7 and 1.5) | none | observed order ≥ 1.8 on each resolution pair | **PASS** — CPU Float64, N = 16 → 32 → 64: order **2.000 / 2.000** (Carreau, wall fraction 0.5), **1.990 / 1.995** (Carreau, 0.3), **2.000 / 2.000** (power law n = 0.7), **1.932 / 1.950** (n = 1.5). One A100 Float64 sweep, N = 32 → 256: orders **1.950 to 2.001** (power law n = 0.7 only to N = 128); Float32 run-through only. Isothermal, Julia API only (no `.krk` route) | [3D GNF channel](gnf3d-channel-convergence.md); `test/analytical/gnf3d_channel_order.jl` |
| **Viscoelastic — Oldroyd-B cylinder** (TRT-LBM, halfway BB) | none published at this confinement (Oldroyd-B `C_d` is solver-dependent) | RheoTool `rheoFoam` fine-mesh (`Cylinder/Oldroyd-BLog`), log-conformation | < 1 % on `C_d` at Wi ≤ 1, R = 50, Re = 1 | **PASS** — `C_d` within **−0.37 % / −0.66 % / −0.75 %** of rheoFoam at Wi = 0.1 / 0.5 / 1.0, β = 0.59 (diffusive scaling τ = 0.95, halfway BB; re-measured 2026-10-14 after the #67 forcing fix); validated only for Wi ≤ 1 | [Viscoelastic cylinder](../tutorials/viscoelastic-cylinder.md) |
| **GPU — single-GPU efficiency** (BGK D2Q9, CUDA F64) | memory-bandwidth roofline (A100-40GB, 1.555 TB/s, 144 B per D2Q9 F64 update ⇒ 10 799 MLUPS ceiling) | Palabos single-GPU F64 D3Q19 TRT (4656 MLUPS, published) | ≥ 0.5 of roofline | **BELOW GATE** — **3461 MLUPS** at N = 2048 ⇒ **0.32** of the A100-40GB roofline (0.24 of the 80 GB ceiling); 0.74 of the Palabos number, which is D3Q19 and not like-for-like | [GPU certification](gpu-certification.md); CSV `benchmarks/results/certification_a100.csv` |

## Notes

- **Follow the artifact link** for the full error-norm tables, methodology, mesh/backend
  details and caveats — the matrix carries only the headline metric.
- **3D rows carry larger residuals** and are documented as resolution-limited on their
  pages (sphere drag +8.9 %). 3D natural convection is wired (`natural_convection_3d`
  preset) but only smoke-validated; a quantitative 3D thermal benchmark is a future item.
  The 2D rows are the strict acceptance gates.
- **AMR conservation ≠ AMR accuracy.** Mass conservation is exact to roundoff across
  all shipped CSVs. Integrated-quantity accuracy is *not* yet a passing gate: the
  cylinder `C_d` gap vs the dense-leaf oracle is small only when the coarse/fine
  interface is buffered off the boundary layer (~1.06×, interface-buffered CSV), and is
  large for a compact patch cutting the near-wall region (≈3.2–5.6×, the shipped
  `…_conv_*` CSVs). This near-interface transport gap is the open item motivating the
  subcycling integration tracked for the next release. Separately, a bit-exact
  full-domain refined-patch ↔ dense-leaf parity (`L2 = L∞ < 1e-14`) is demonstrated for
  the **3D Poiseuille** route-native canary (`test/test_conservative_tree_streaming_3d.jl`),
  establishing that the refined path *reproduces* the leaf solution when the whole domain
  is refined; it is not a 2D-cylinder accuracy result.
- **Literal reference-solver backfill.** The Newtonian cavity row ships a complete
  `benchmarks/results/rheotool_compare/newtonian/` bundle (both-solver CSVs, per-Re error
  norms, the `.krk`, and a self-contained `plot.py`). The thermal and viscoelastic
  literal bundles are the next legs of the same backfill.
