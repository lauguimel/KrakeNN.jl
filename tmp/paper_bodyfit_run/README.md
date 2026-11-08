# Paper Body-Fitted Gmsh Run

This folder contains the isolated artifacts for the current paper-oriented
body-fitted Gmsh workflow. It is intentionally separated from the many older
`tmp/diag_*` experiments.

## Files

- `paper_gmsh_bodyfit_workflow.jl`
  Generates a two-block Gmsh mesh with a 29.055 degree interface, imports it
  with Kraken, checks block orientation, mesh metrics, analytic feq moments,
  shared-node exchange, and complete pre-step ghost fill.

- `paper_gmsh_slbm_no_collision.jl`
  Uses the same Gmsh workflow, then measures SLBM departure + interpolation
  errors without collision, forcing, or post-step boundary conditions. It
  compares the current production local-Jacobian departure with the optional
  production Q1-cell Newton departure.

- `paper_gmsh_slbm_one_step.jl`
  Runs the real `slbm_bgk_step!` kernel once on the same Gmsh case. This
  checks that the departure fix survives the fused `stream -> BGK` path before
  adding forcing, inlet/outlet rebuilds, or moving-wall corrections.

- `paper_gmsh_forced_poiseuille.jl`
  Adds a diagnostic Guo-force BGK step on the same Gmsh case. Physical
  boundaries are analytically clamped after each step on purpose: this checks
  the force + departure + interpolation core before claiming any production
  inlet/outlet or no-slip boundary condition.

- `paper_gmsh_forced_poiseuille_physical_bc.jl`
  Reuses the same forced BGK diagnostic but replaces the analytic post-step
  clamp by the script-local physical-normal BC reconstruction. It compares
  three pre-step ghost policies: current standard ghosts, BC-consistent
  equilibrium ghosts from tags/profiles, and analytic/manufactured ghosts.

- `paper_gmsh_forced_brick_ladder.jl`
  Step-by-step diagnostic for the forced case. It separates population pull
  error, exact analytic-pull BGK+force error, interpolated-pull BGK+force
  error, and the actual diagnostic kernel. This is the script that identified
  the discrete force scaling and interface/ghost-stencil boundary layer.

- `paper_convergent_singleblock.jl`
  Generates a one-block body-fitted convergent channel (`H=1` at `x=0`,
  `H=0.5` at `x=2`), writes the matching `.msh` and `.krk`, runs a local
  SLBM smoke/diagnostic case, and saves `bodyfit_convergent_singleblock.png`
  for visual inspection.

- `paper_convergent_mesh_sweep.jl`
  Mesh/time-stability sweep for the one-block convergent. The default run uses
  `91x46,121x61,181x91,241x121` for `20000` steps because shorter `2000` to
  `8000` step runs were stable but not time-converged in global outlet flux.

- `paper_convergent_divergent_mesh_sweep.jl`
  Convergent-divergent follow-up. The default `KRK_CD_LAYOUT=single_spline`
  uses one smooth body-fitted block and is stable for the 20000-step sweep.
  `KRK_CD_LAYOUT=two_block` keeps the straight convergent/divergent pieces
  split at the throat. The two-block divergence was traced to block
  reorientation that put the physical inlet/outlet on logical north/south
  faces; `autoreorient_blocks` now preserves the physical tag semantics
  `inlet -> west`, `interface -> east/west`, `outlet -> east`, and the
  four-grid 20000-step two-block sweep is stable.
  `KRK_CD_WALL_GHOST=project_normal_corners` is the current wall diagnostic
  closure: ghost layers still use copy+halfway data, then the post-step wall
  populations are projected so the physical normal velocity vanishes; wall
  edge endpoints are set to no-slip to remove throat/corner leakage. Older
  `KRK_CD_WALL_GHOST=reflect_walls` remains available as a diagnostic mode:
  it reduces wall-normal leakage but worsens outlet-flux convergence.

- `paper_cd_investigate.jl`
  Focused diagnostics for the convergent-divergent cases. It writes signed
  boundary-flux CSVs and two-block debug traces with density extrema,
  interface jumps, wall flux, and max wall-normal velocity.

Current convergent-divergent `u_n` / `Q` debug:

| layout | mesh | steps | wall mode | Q rel. error | max wall `u.n` | wall flux | interface jump |
|---|---:|---:|---|---:|---:|---:|---:|
| single spline | `91x46` | 20000 | `copy_halfway` | `1.939e-03` | `5.63e-04` | `5.7e-05` | n/a |
| single spline | `91x46` | 20000 | `project_normal_corners` | `4.671e-04` | `5.51e-17` | `-2.46e-08` | n/a |
| two block | `91x46` | 20000 | `project_normal_corners` | `1.527e-04` | `4.58e-17` | `3.4e-18` | `0.0` |
| single spline | `241x121` | 20000 | `project_normal_corners` | `2.723e-02` | `4.68e-17` | n/a | n/a |
| single spline | `241x121` | 60000 | `project_normal_corners` | `7.660e-03` | `6.74e-17` | `-2.92e-09` | n/a |
| two block | `241x121` | 20000 | `project_normal_corners` | `2.220e-02` | `7.23e-17` | n/a | n/a |
| two block | `241x121` | 60000 | `project_normal_corners` | `7.225e-03` | `4.21e-17` | `-4.0e-19` | `0.0` |

Interpretation: the high wall-normal velocity was a wall/corner closure issue,
not a two-block orientation issue. `project_normal_corners` removes the wall
leakage to roundoff in the state metric and leaves only tiny segment-integral
corner residuals in the signed boundary-flux diagnostic. The remaining `Q`
gap at fine resolution decreases substantially between 20000 and 60000 steps
while wall flux and interface jump stay near zero, so the next target is
time/outlet convergence of the open-boundary closure, not another wall fix.
The mesh-sweep CSVs therefore report both instantaneous `Q_rel_err` and
tail-window metrics computed over the last quarter of the sampled history:
`Q_rel_tail_mean_abs`, `Q_rel_tail_rms`, and `Q_rel_tail_max_abs`. Use the
tail metrics for convergence/stability decisions; the instantaneous final
`Q_rel_err` can be dominated by the phase of the weakly-compressible outlet
oscillation.

- `paper_gmsh_interpolation_exactness.jl`
  Exactness diagnostic for the interpolation space itself. It fills the
  extended grid analytically and proves that `Q1 Newton + ng=3 + biquadratic`
  is at machine precision for fields representable by biquadratic
  interpolation, while full-equilibrium Poiseuille is not.

- `paper_gmsh_interpolation_bench.jl`
  CPU/GPU micro-benchmark for the real `slbm_bgk_step!` BGK kernel on the
  same two-block Gmsh case. It compares `bilinear`, `biquadratic`, and
  `quartic` interpolation cost with the same Q1-Newton departure field.
  Set `KRK_PAPER_BACKEND=metal` on Apple GPUs; Metal runs in `Float32`.

- `paper_compute_tradeoff.jl`
  Measures a Cartesian `fused_bgk_step!` baseline and compares the per-node
  cost to body-fitted SLBM. It prints the break-even node reduction needed for
  body-fitted SLBM to beat a Cartesian LBM run at equal timestep count.

- `paper_scaling_sweep.jl`
  Size sweep for the paper performance section. It runs Cartesian
  `fused_bgk_step!` on increasing square grids and SLBM on increasing Gmsh
  two-block grids, so H100 launch/occupancy effects are not confused with
  asymptotic GLUP/s.

- `paper_make_figures.jl`
  Post-processes the Aqua H100 logs into clean CSV/Markdown tables and PNG
  figures. It does not rerun any solver.

- `paper_sphere3d_roi_model.jl`
  Computes the speedup envelope for a future 3D sphere drag convergence
  benchmark: `total speedup = node reduction / per-node cost ratio`. This is
  a cost model, not a CFD validation.

- `paper_cylinder2d_focused_roi_model.jl`
  Computes a 2D focused-cylinder pre-screen using `cylinder_focused_mesh`.
  It combines finest local spacing, equivalent Cartesian node count, mesh
  quality, and measured H100 per-node cost. This is a ROI filter before a
  drag-convergence benchmark, not a CFD validation.

- `paper_cylinder2d_drag_probe.jl`
  Runs a short Cartesian LI-BB versus focused-SLBM 2D drag plumbing probe on
  the same Schaefer-Turek-style channel. It writes tagged CSV/Markdown history
  files and is meant to catch stability/normalisation issues before a long
  convergence campaign.

- `paper_cylinder2d_convergence_compare.jl`
  Runs the H100 comparison needed for the paper question:
  Cartesian LI-BB on uniform grids versus the production
  `.krk -> Mesh gmsh(.msh) -> Module slbm_drag` O-grid body-fitted path.
  It writes `Cd/Cl` errors against Schaefer-Turek 2D-1 references and records
  unstable body-fitted meshes instead of aborting the whole sweep. When plots
  are enabled, it writes both convergence-error and force-history PNGs. Set
  `KRK_CYL_CONV_PLOT_ONLY=1` with the same `KRK_CYL_CONV_TAG` to regenerate
  those figures locally from H100 CSVs without rerunning the solver.

- `paper_make_ogrid_msh.jl`
  Generates `meshes/cylinder_ogrid_8block.geo` and
  `meshes/cylinder_ogrid_8block.msh`, the paper O-grid cylinder mesh used by
  the `.krk` driven drag path.

- `krk/cylinder_ogrid_msh_drag.krk`
  Minimal paper case for the new production path:
  `.krk -> Mesh gmsh(file = "...msh") -> Module slbm_drag -> Cd/Cl`.
  The file references the generated `.msh` relative to the `.krk` location.

- `paper_krk_msh_drag.jl`
  Thin paper runner around the production `.krk` path. It selects
  CPU/CUDA/Metal from environment variables, optionally regenerates the
  default O-grid `.msh`, runs `run_simulation("...krk")`, and writes
  `paper_tables/krk_msh_drag*.csv/.md`.

- `SPHERE_3D_CONVERGENCE_PLAN.md`
  Audit/plan for the H100 sphere benchmark. It separates what Kraken can run
  now, a 3D embedded sphere/cut-link SLBM path, from the missing true Gmsh
  hexa multiblock body-fitted sphere path.

- `pbs/paper_bodyfit_h100.pbs`
  Aqua H100 batch script for the paper numbers. It runs the exactness gate,
  one-step CUDA diagnostic, interpolation benchmark, and compute tradeoff in
  both CUDA `Float64` and CUDA `Float32`.

- `pbs/paper_bodyfit_h100_scaling.pbs`
  Aqua H100 batch script for the grid-size scaling sweep.

- `pbs/paper_cylinder2d_drag_h100.pbs`
  Aqua H100 batch script for the 2D cylinder drag probes. Default cases:
  CUDA `Float64`/`Float32`, focused mesh `111x31`, `strength=1.5,2.0`,
  `10000` steps.

- `pbs/paper_krk_msh_drag_h100.pbs`
  Aqua H100 smoke script for the new `.krk -> .msh -> SLBM drag` route. It
  regenerates the O-grid `.msh` and runs the `.krk` case with CUDA `Float64`.

- `paper_boundary_rebuild_unit.jl`
  Checks the local D2Q9 boundary-condition algebra on a minimal reference
  patch: Zou-He velocity, Zou-He pressure, and Ladd moving-wall reflection.
  This is intentionally geometry-free so BC bugs are not hidden by mesh or
  interpolation errors.

- `paper_cartesian_poiseuille_bc.jl`
  Runs a small Cartesian Poiseuille channel with the production
  `fused_trt_step! + BCSpec2D/apply_bc_rebuild_2d!` path. It checks both
  west/east and south/north inlet-outlet orientations before returning to
  Gmsh/SLBM geometry.

- `paper_gmsh_bc_semantics.jl`
  Audits the same Gmsh two-block case after `autoreorient_blocks` and checks
  whether the physical boundary normals still match the logical face normals
  assumed by the Cartesian Zou-He/halfway-BB kernels.

- `paper_physical_normal_bc_unit.jl`
  Prototype CPU/local reconstruction where the physical normal is independent
  from the logical edge. It checks Zou-He velocity, Zou-He pressure, and Ladd
  wall reflection on all `logical edge x physical normal` combinations, then
  replays the actual Gmsh tags after reorientation.

- `paper_gmsh_physical_normal_bc_tags.jl`
  Applies the same physical-normal prototype to the actual Gmsh physical tags
  with the analytic Poiseuille profile: inlet `u_x(y)`, outlet pressure, and
  no-slip walls.

- `test_multiblock_gmsh_bodyfit_workflow.jl`
  Test version of the workflow. The file `test/test_multiblock_gmsh_bodyfit_workflow.jl`
  is only a small include wrapper so the code lives here.

- `test_departure_geometry.jl`
  Unit-style checks for the Q1 Jacobian, Q1 Newton inversion, and the Gmsh
  no-collision departure comparison. The file
  `test/test_multiblock_gmsh_departure_geometry.jl` is only an include wrapper.

- `test_interpolation_exactness.jl`
  Non-plot test for the interpolation exactness diagnostic. The file
  `test/test_multiblock_gmsh_interpolation_exactness.jl` is only an include
  wrapper.

- `test_slbm_one_step.jl`
  Non-plot test for the one-step BGK diagnostic. The file
  `test/test_multiblock_gmsh_slbm_one_step.jl` is only an include wrapper.

- `test_forced_poiseuille.jl`
  Non-plot test for the forced one-step bulk diagnostic. The file
  `test/test_multiblock_gmsh_forced_poiseuille.jl` is only an include wrapper.

- `test_gmsh_forced_poiseuille_physical_bc.jl`
  Non-plot one-step test for the physical-normal post-step BC path. The file
  `test/test_multiblock_gmsh_forced_poiseuille_physical_bc.jl` is only an
  include wrapper.

- `test_boundary_rebuild_unit.jl`
  Non-plot test for the local BC algebra. The file
  `test/test_multiblock_gmsh_boundary_rebuild_unit.jl` is only an include
  wrapper.

- `test_cartesian_poiseuille_bc.jl`
  Non-plot smoke/integration test for the Cartesian Poiseuille BC path in both
  orientations. The file
  `test/test_multiblock_gmsh_cartesian_poiseuille_bc.jl` is only an include
  wrapper.

- `test_gmsh_bc_semantics.jl`
  Non-plot test for the Gmsh BC semantic audit. The file
  `test/test_multiblock_gmsh_bc_semantics.jl` is only an include wrapper.

- `test_physical_normal_bc_unit.jl`
  Non-plot test for the physical-normal BC prototype. The file
  `test/test_multiblock_gmsh_physical_normal_bc_unit.jl` is only an include
  wrapper.

- `test_gmsh_physical_normal_bc_tags.jl`
  Non-plot test for the Gmsh physical-normal tag/profile diagnostic. The file
  `test/test_multiblock_gmsh_physical_normal_bc_tags.jl` is only an include
  wrapper.

- `test_krk_msh_drag.jl`
  Non-plot smoke test for the new `.krk -> .msh -> SLBM drag` route. The file
  `test/test_krk_gmsh_drag.jl` is only an include wrapper.

- `plots/`
  Human-readable figures generated by the scripts: mesh, velocity, density,
  profiles, and no-collision error maps.
  - `bodyfit_workflow_details.png`: mesh, `ux`, `uy`, `rho`, `abs(ux-analytic)`,
    and centerline profile.
  - `slbm_no_collision_local_jacobian_bilinear.png`: production departure,
    bilinear population interpolation.
  - `slbm_no_collision_local_jacobian_biquadratic.png`: production departure,
    biquadratic population interpolation.
  - `slbm_no_collision_q1_newton_bilinear.png`: Q1-cell Newton departure,
    bilinear population interpolation.
  - `slbm_no_collision_q1_newton_biquadratic.png`: Q1-cell Newton departure,
    biquadratic population interpolation.
  - `slbm_one_step_*`: one-step BGK maps/profiles. The best case currently
    writes `slbm_one_step_q1_newton_biquadratic_summary.png` because the
    detailed heatmap can have constant color fields.
  - `forced_poiseuille_*`: forced Poiseuille maps/profiles/history. These are
    diagnostic plots with analytic boundary clamps, not final open-channel
    validation plots.
  - `paper_h100_scaling.png`: H100 throughput scaling from Aqua logs.
  - `paper_h100_break_even.png`: matched-node H100 break-even fractions.
  - `paper_sphere3d_roi_model.png`: speedup envelope for a future 3D sphere
    body-fitted convergence benchmark.
  - `paper_cylinder2d_focused_roi.png`: 2D focused-cylinder ROI pre-screen
    for selecting drag-convergence candidates.
  - `bodyfit_workflow_basic.png`: earlier two-panel mesh/field snapshot kept for comparison.
  - `legacy/`: superseded no-collision plots from before departure modes were
    printed in the file names.

- `paper_tables/`
  Clean tables generated from the H100 logs and ROI scripts:
  - `h100_scaling.csv`, `h100_exactness.csv`, `h100_tradeoff.csv`,
    `h100_summary.md`.
  - `sphere3d_measured_cost_proxy.csv`, `sphere3d_roi_model.csv`,
    `sphere3d_roi_model.md`.
  - `cylinder2d_focused_mesh_candidates.csv`, `cylinder2d_focused_roi.csv`,
    `cylinder2d_focused_roi.md`.
  - `cylinder2d_drag_probe*.csv`, `cylinder2d_drag_probe*.md`: short drag
    probe outputs, including H100-tagged files after Aqua retrieval.
  - H100 drag probe note: job `20423906.aqua` finished with `Exit_status=1`.
    The Cartesian subcase reached `Cd=5.5041`, then `focused_slbm` diverged at
    step `7000` with NaN density. See `aqua_logs/kraken_cyl2d_drag.o20423906`.
  - `cylinder2d_convergence_compare_cuda_float64_conv_surface.*`: H100
    comparison job `20430719.aqua`, `Exit_status=0`. Cartesian LI-BB is close
    to the 2D-1 references; current body-fitted O-grid `slbm_drag` is not yet
    quantitatively valid (`Cd≈-0.3` and one fine O-grid diverges).
  - `cylinder2d_convergence_compare_cuda_float64_conv_fine.*`: H100 comparison
    job `20453570.aqua`, `Exit_status=0`, after changing the density guard to
    inspect physical cells only. Cartesian LI-BB remains close (`Cd=5.5573` at
    `D_eff=50`). Body-fitted is still not publishable: `20x16` fails at step
    `8000`, `36x24` fails at `6500`, and the only completed O-grid (`28x20`)
    gives `Cd=1.1117` with physical `rho=0.9156-3.3438`.
  - `cylinder2d_convergence_compare_cuda_float64_conv_reflect_ghost.*`: H100
    comparison job `20453900.aqua`, `Exit_status=0`, opt-in full reflection
    ghost on `:cylinder` edges. This worsened stability: all O-grids failed
    before producing averaged `Cd/Cl`. The option is kept for diagnostics but
    is disabled by default.
  - `krk_msh_drag_h100_20428703.txt`: H100 smoke for
    `.krk -> .msh -> SLBM drag`, job `20428703.aqua`, `Exit_status=0`.
    Output tables:
    `krk_msh_drag_cuda_float64_smoke.csv`,
    `krk_msh_drag_history_cuda_float64_smoke.csv`, and
    `krk_msh_drag_cuda_float64_smoke.md`. This is a 20-step pipeline gate,
    not a converged drag result.

## Commands

```bash
julia --project=. tmp/paper_bodyfit_run/paper_gmsh_bodyfit_workflow.jl
julia --project=. tmp/paper_bodyfit_run/paper_gmsh_slbm_no_collision.jl
julia --project=. tmp/paper_bodyfit_run/paper_gmsh_slbm_one_step.jl
julia --project=. tmp/paper_bodyfit_run/paper_gmsh_forced_poiseuille.jl
julia --project=. tmp/paper_bodyfit_run/paper_gmsh_forced_poiseuille_physical_bc.jl
julia --project=. tmp/paper_bodyfit_run/paper_gmsh_forced_brick_ladder.jl
julia --project=. tmp/paper_bodyfit_run/paper_gmsh_interpolation_exactness.jl
julia --project=. tmp/paper_bodyfit_run/paper_gmsh_interpolation_bench.jl
julia --project=. tmp/paper_bodyfit_run/paper_compute_tradeoff.jl
julia --project=. tmp/paper_bodyfit_run/paper_scaling_sweep.jl
julia --project=. tmp/paper_bodyfit_run/paper_make_figures.jl
julia --project=. tmp/paper_bodyfit_run/paper_sphere3d_roi_model.jl
julia --project=. tmp/paper_bodyfit_run/paper_cylinder2d_focused_roi_model.jl
julia --project=. tmp/paper_bodyfit_run/paper_cylinder2d_drag_probe.jl
julia --project=. tmp/paper_bodyfit_run/paper_cylinder2d_convergence_compare.jl
julia --project=. tmp/paper_bodyfit_run/paper_make_ogrid_msh.jl
julia --project=. -e 'using KrakeNN; res = run_simulation("tmp/paper_bodyfit_run/krk/cylinder_ogrid_msh_drag.krk"); @show res.Cd res.Cl res.D_eff'
KRK_KRK_MSH_STEPS=20 KRK_KRK_MSH_TAG=local_smoke julia --project=. tmp/paper_bodyfit_run/paper_krk_msh_drag.jl
ssh aqua 'cd /home/maitreje/Kraken.jl && qsub tmp/paper_bodyfit_run/pbs/paper_krk_msh_drag_h100.pbs'
ssh aqua 'cd /home/maitreje/Kraken.jl && qsub tmp/paper_bodyfit_run/pbs/paper_cylinder2d_convergence_h100.pbs'
julia --project=. -e 'ENV["KRK_PAPER_BACKEND"]="metal"; ENV["KRK_PAPER_FT"]="Float32"; include("tmp/paper_bodyfit_run/paper_gmsh_interpolation_bench.jl"); main()'
julia --project=. -e 'ENV["KRK_PAPER_BACKEND"]="metal"; ENV["KRK_PAPER_FT"]="Float32"; include("tmp/paper_bodyfit_run/paper_compute_tradeoff.jl"); main()'
julia --project=. -e 'ENV["KRK_PAPER_BACKEND"]="metal"; ENV["KRK_PAPER_FT"]="Float32"; ENV["KRK_PAPER_INTERP"]="bilinear,biquadratic,quartic"; ENV["KRK_PAPER_DEPARTURE"]="q1_newton"; ENV["KRK_PAPER_SKIP_PLOTS"]="1"; include("tmp/paper_bodyfit_run/paper_gmsh_slbm_one_step.jl"); main()'
julia --project=. tmp/paper_bodyfit_run/paper_boundary_rebuild_unit.jl
julia --project=. tmp/paper_bodyfit_run/paper_cartesian_poiseuille_bc.jl
julia --project=. tmp/paper_bodyfit_run/paper_gmsh_bc_semantics.jl
julia --project=. tmp/paper_bodyfit_run/paper_physical_normal_bc_unit.jl
julia --project=. tmp/paper_bodyfit_run/paper_gmsh_physical_normal_bc_tags.jl
julia --project=. test/test_multiblock_gmsh_bodyfit_workflow.jl
julia --project=. test/test_multiblock_gmsh_departure_geometry.jl
julia --project=. test/test_multiblock_gmsh_interpolation_exactness.jl
julia --project=. test/test_multiblock_gmsh_slbm_one_step.jl
julia --project=. test/test_multiblock_gmsh_forced_poiseuille.jl
julia --project=. test/test_multiblock_gmsh_forced_poiseuille_physical_bc.jl
julia --project=. test/test_multiblock_gmsh_boundary_rebuild_unit.jl
julia --project=. test/test_multiblock_gmsh_cartesian_poiseuille_bc.jl
julia --project=. test/test_multiblock_gmsh_bc_semantics.jl
julia --project=. test/test_multiblock_gmsh_physical_normal_bc_unit.jl
julia --project=. test/test_multiblock_gmsh_physical_normal_bc_tags.jl
julia --project=. test/test_krk_gmsh_drag.jl
```

Targeted multiblock subset:

```bash
julia --project=. -e 'using Test; using KrakeNN; @testset "multiblock paper subset" begin include("test/test_gmsh_loader.jl"); include("test/test_multiblock_topology.jl"); include("test/test_multiblock_exchange.jl"); include("test/test_multiblock_exchange_shared.jl"); include("test/test_multiblock_reorient.jl"); include("test/test_multiblock_mesh_extend.jl"); include("test/test_multiblock_gmsh.jl"); include("test/test_multiblock_gmsh_bodyfit_workflow.jl"); include("test/test_multiblock_gmsh_departure_geometry.jl"); include("test/test_multiblock_gmsh_slbm_one_step.jl"); include("test/test_multiblock_gmsh_forced_poiseuille.jl"); include("test/test_multiblock_gmsh_forced_poiseuille_physical_bc.jl"); include("test/test_multiblock_gmsh_boundary_rebuild_unit.jl"); include("test/test_multiblock_gmsh_cartesian_poiseuille_bc.jl"); include("test/test_multiblock_gmsh_bc_semantics.jl"); include("test/test_multiblock_gmsh_physical_normal_bc_unit.jl"); include("test/test_multiblock_gmsh_physical_normal_bc_tags.jl"); include("test/test_multiblock_canal.jl"); end'
```

Result: `1461 / 1461` tests pass.

## Current Results

Workflow:

- sanity: 0 errors, 1 expected shared-node warning;
- interface angle: 29.055 degrees;
- interface coordinate error: about `3e-14`;
- feq moment errors: machine precision;
- shared-node ghost exchange: exact on the analytic feq field;
- complete pre-step ghost fill: no non-finite populations.

No-collision SLBM:

| interpolation | departure | geometry error | population error | moment error |
|---|---|---:|---:|---:|
| bilinear | local Jacobian | `2.258e-02` | `7.812e-06` | `2.510e-05` |
| biquadratic | local Jacobian | `2.258e-02` | `2.403e-06` | `8.300e-06` |
| bilinear | Q1-cell Newton | `4.263e-14` | `8.131e-06` | `2.609e-05` |
| biquadratic | Q1-cell Newton | `4.263e-14` | `4.948e-08` | `4.404e-08` |

Interpretation: Gmsh import, orientation, metrics, tags, analytic init, and
ghost exchange are clean. The local-Jacobian SLBM departure geometry is the
dominant error on the non-affine block. Inverting the actual Q1 Gmsh cell
removes that geometric error to roundoff; the remaining error is then the
population interpolation error.

Interpolation exactness on the same Gmsh mesh, with the extended grid filled
analytically and no ghost exchange/BC involved:

| profile | equilibrium | geometry error | population error | moment error |
|---|---|---:|---:|---:|
| constant | quadratic | `4.263e-14` | `4.163e-17` | `1.110e-16` |
| Couette linear shear | quadratic | `4.263e-14` | `6.106e-16` | `1.193e-15` |
| Poiseuille | linearized | `4.263e-14` | `1.804e-15` | `3.452e-15` |
| Poiseuille | full quadratic, biquadratic interp | `4.263e-14` | `4.948e-08` | `4.404e-08` |
| Poiseuille | full quadratic, quartic interp | `4.263e-14` | `1.624e-15` | `3.567e-15` |

This is the key machine-precision gate. `Q1 Newton + ng=3 + biquadratic` is
machine-accurate when the population field belongs to the biquadratic
interpolation space. Full D2Q9 equilibrium for parabolic Poiseuille contains
`ux^2`, hence quartic-in-`y` population terms; biquadratic interpolation cannot
represent those terms exactly. A diagnostic tensor-product quartic stencil does
recover machine precision on this manufactured Q1 Gmsh case and is now exposed
in production BGK SLBM as `interp=:quartic`. The
full-Poiseuille biquadratic population error scales as `u_max^2`: `4.948e-08`,
`4.948e-10`, `4.948e-12` for `u_max = 0.04, 0.004, 0.0004`.

Interpolation benchmark, real `slbm_bgk_step!` kernel, same two-block Gmsh
mesh, `departure=:q1_newton`, `ng=3`, `omega=1`, `257 x 129` transfinite
points per block direction input, `71010` extended kernel nodes.

CPU `Float64`:

| interpolation | stencil samples per node | best time | ns/node | MLUP/s | relative |
|---|---:|---:|---:|---:|---:|
| bilinear | 36 | `5.512e-01 s` | `7.762e+01` | `1.288e+01` | `1.00x` |
| biquadratic | 81 | `1.406e+00 s` | `1.981e+02` | `5.049e+00` | `2.55x` |
| quartic | 225 | `2.963e+00 s` | `4.172e+02` | `2.397e+00` | `5.37x` |

Metal `Float32` on the local Apple GPU:

| interpolation | stencil samples per node | best time | ns/node | MLUP/s | relative | speedup vs CPU |
|---|---:|---:|---:|---:|---:|---:|
| bilinear | 36 | `1.969e-02 s` | `2.773e+00` | `3.606e+02` | `1.00x` | `28.0x` |
| biquadratic | 81 | `2.574e-02 s` | `3.624e+00` | `2.759e+02` | `1.31x` | `54.6x` |
| quartic | 225 | `6.084e-02 s` | `8.567e+00` | `1.167e+02` | `3.09x` | `48.7x` |

Interpretation: the measured CPU cost follows the stencil size but is not as
bad as the raw sample count ratio (`2.25x` and `6.25x`). On the GPU, the wider
stencils amortize better: `quartic` is about `3.1x` the bilinear cost, not
`6.25x`. For a paper/MVP solver, `biquadratic` remains the sensible default;
`quartic` is a validation or high-accuracy option when the manufactured
population field actually requires quartic reproduction.

Metal one-step BGK diagnostic, `Float32`, `departure=:q1_newton`, no plots:

| interpolation | geometry error | pull moment error | macro consistency | feq rebuild error | nodal ux error |
|---|---:|---:|---:|---:|---:|
| bilinear | `4.263e-14` | `2.609e-05` | `9.499e-08` | `4.293e-08` | `3.066e-05` |
| biquadratic | `4.263e-14` | `1.788e-07` | `7.451e-08` | `3.615e-08` | `4.645e-06` |
| quartic | `4.263e-14` | `2.384e-07` | `6.892e-08` | `3.455e-08` | `4.620e-06` |

This run validates the local GPU path. It is intentionally not the
machine-precision gate: Metal uses `Float32`, so the strict `1e-16` proof
remains the CPU `Float64` exactness diagnostic.

Compute tradeoff, local Metal `Float32`, Cartesian `fused_bgk_step!` baseline
on `267 x 266 = 71022` nodes versus SLBM on `71010` extended Gmsh nodes:

| kernel | ns/node | MLUP/s | per-node cost vs Cartesian |
|---|---:|---:|---:|
| Cartesian BGK | `1.226e+00` | `8.153e+02` | `1.00x` |
| SLBM bilinear | `2.768e+00` | `3.613e+02` | `2.26x` |
| SLBM biquadratic | `3.821e+00` | `2.617e+02` | `3.12x` |
| SLBM quartic | `8.649e+00` | `1.156e+02` | `7.05x` |

Break-even body-fitted node fraction at equal timestep count:

| interpolation | max body-fitted nodes vs Cartesian | required node reduction |
|---|---:|---:|
| bilinear | `44.3%` | `55.7%` |
| biquadratic | `32.1%` | `67.9%` |
| quartic | `14.2%` | `85.8%` |

This is the honest compute statement: body-fitted SLBM is not automatically
cheaper per node than Cartesian LBM. It becomes a runtime win only when the
body-fitted mesh removes enough Cartesian/immersed-boundary nodes or permits a
coarser mesh at the same engineering accuracy. Its immediate paper value is
the Gmsh-to-SLBM workflow and boundary-aligned accuracy; raw speedup requires
a substantial node-count reduction.

H100 CUDA `Float64`, same small matched-node diagnostic size (`~71k` nodes):

| kernel | ns/node | MLUP/s | per-node cost vs Cartesian |
|---|---:|---:|---:|
| Cartesian BGK | `1.076e-01` | `9.292e+03` | `1.00x` |
| SLBM bilinear | `4.245e-01` | `2.355e+03` | `3.94x` |
| SLBM biquadratic | `7.090e-01` | `1.410e+03` | `6.59x` |
| SLBM quartic | `1.743e+00` | `5.737e+02` | `16.20x` |

H100 CUDA `Float32`, same small matched-node diagnostic size:

| kernel | ns/node | MLUP/s | per-node cost vs Cartesian |
|---|---:|---:|---:|
| Cartesian BGK | `1.077e-01` | `9.282e+03` | `1.00x` |
| SLBM bilinear | `2.505e-01` | `3.993e+03` | `2.32x` |
| SLBM biquadratic | `6.964e-01` | `1.436e+03` | `6.46x` |
| SLBM quartic | `1.743e+00` | `5.737e+02` | `16.18x` |

These H100 numbers are a matched-node diagnostic, not the asymptotic H100
throughput claim. H100 Cartesian LBM is known to keep scaling with grid size
into the `10-30 GLUP/s` range depending on kernel/precision. The separate
`paper_scaling_sweep.jl` run is the source for final large-grid claims.

H100 exactness/one-step CUDA `Float64`:

| interpolation | pull moment error | macro consistency | feq rebuild error |
|---|---:|---:|---:|
| bilinear | `2.609e-05` | `4.441e-16` | `2.776e-17` |
| biquadratic | `4.404e-08` | `4.441e-16` | `2.776e-17` |
| quartic | `1.235e-15` | `4.441e-16` | `5.551e-17` |

This confirms that the production CUDA kernel preserves the CPU `Float64`
machine-precision conclusion for the quartic manufactured Poiseuille pull.

H100 scaling sweep, CUDA `Float64`:

| case | nodes | bilinear/cartesian MLUP/s | biquadratic MLUP/s | quartic MLUP/s |
|---|---:|---:|---:|---:|
| Cartesian BGK 128² | `16384` | `1.701e+03` | - | - |
| Cartesian BGK 256² | `65536` | `6.981e+03` | - | - |
| Cartesian BGK 512² | `262144` | `1.773e+04` | - | - |
| Cartesian BGK 1024² | `1048576` | `1.518e+04` | - | - |
| Cartesian BGK 2048² | `4194304` | `1.588e+04` | - | - |
| Cartesian BGK 4096² | `16777216` | `1.613e+04` | - | - |
| SLBM 65x33 | `5538` | `2.589e+02` | `1.273e+02` | `7.366e+01` |
| SLBM 129x65 | `19170` | `9.098e+02` | `4.013e+02` | `1.817e+02` |
| SLBM 257x129 | `71010` | `2.392e+03` | `1.435e+03` | `5.722e+02` |
| SLBM 513x257 | `272994` | `1.934e+03` | `1.186e+03` | `5.940e+02` |
| SLBM 1025x513 | `1070178` | `2.565e+03` | `1.210e+03` | `6.189e+02` |

H100 scaling sweep, CUDA `Float32`:

| case | nodes | bilinear/cartesian MLUP/s | biquadratic MLUP/s | quartic MLUP/s |
|---|---:|---:|---:|---:|
| Cartesian BGK 128² | `16384` | `1.716e+03` | - | - |
| Cartesian BGK 256² | `65536` | `7.061e+03` | - | - |
| Cartesian BGK 512² | `262144` | `2.788e+04` | - | - |
| Cartesian BGK 1024² | `1048576` | `2.547e+04` | - | - |
| Cartesian BGK 2048² | `4194304` | `2.939e+04` | - | - |
| Cartesian BGK 4096² | `16777216` | `2.992e+04` | - | - |
| SLBM 65x33 | `5538` | `2.606e+02` | `1.311e+02` | `7.651e+01` |
| SLBM 129x65 | `19170` | `9.081e+02` | `3.945e+02` | `1.865e+02` |
| SLBM 257x129 | `71010` | `3.285e+03` | `1.438e+03` | `6.114e+02` |
| SLBM 513x257 | `272994` | `3.809e+03` | `2.425e+03` | `6.370e+02` |
| SLBM 1025x513 | `1070178` | `3.915e+03` | `2.375e+03` | `6.627e+02` |

Final performance interpretation: H100 Cartesian BGK reaches the expected
large-grid range (`16.1 GLUP/s` Float64, `29.9 GLUP/s` Float32). Current SLBM
BGK is far more memory/arithmetic intensive and tops out, on this Gmsh
diagnostic family, around `2.6 GLUP/s` Float64 bilinear and `3.9 GLUP/s`
Float32 bilinear; biquadratic and quartic are lower. Therefore the body-fitted
case must reduce nodes substantially or improve error-per-node enough to
justify the higher cost. This is a feature/workflow paper argument, not a raw
throughput win over optimized Cartesian LBM.

Sphere 3D ROI model:

| reduction per direction | node reduction | assumed SLBM cost | total speedup |
|---|---:|---:|---:|
| `10 x 10 x 10` | `1000x` | `20x` | `50x` |
| `10 x 20 x 10` | `2000x` | `20x` | `100x` |
| `20 x 20 x 20` | `8000x` | `20x` | `400x` |

This is the compute argument to test with a future H100 drag convergence:
body-fitted SLBM does not need to win per node if it reaches the same drag
error with orders-of-magnitude fewer nodes. The current code can run 3D
embedded sphere/cut-link diagnostics, but a true Gmsh hexa multiblock
body-fitted sphere still needs dedicated 3D multiblock geometry/BC work.

One-step BGK, real `slbm_bgk_step!` kernel with `omega=1`:

| interpolation | departure | geometry error | pull moment error | macro consistency | feq rebuild error | nodal ux error |
|---|---|---:|---:|---:|---:|---:|
| bilinear | local Jacobian | `2.258e-02` | `2.510e-05` | `3.331e-16` | `0.000e+00` | `2.967e-05` |
| biquadratic | local Jacobian | `2.258e-02` | `8.300e-06` | `4.441e-16` | `0.000e+00` | `1.139e-05` |
| bilinear | Q1-cell Newton | `4.263e-14` | `2.609e-05` | `3.331e-16` | `0.000e+00` | `3.066e-05` |
| biquadratic | Q1-cell Newton | `4.263e-14` | `4.404e-08` | `4.441e-16` | `0.000e+00` | `4.609e-06` |

Interpretation: the fused BGK kernel is consistent with the no-collision
diagnostic. The remaining best-case one-step error is the interpolation
limit, while BGK rebuild and macro storage are at roundoff. The nodal
Poiseuille error is intentionally not a steady-state proof because this run
has no body force and no post-step inlet/outlet rebuild.

Forced Poiseuille diagnostic, script-local Guo-force BGK with analytic
physical-boundary clamps:

| interpolation | departure | step | max ux error | bulk max ux error | bulk max uy | bulk rho dev |
|---|---|---:|---:|---:|---:|---:|
| biquadratic | local Jacobian | 1 | `5.579e-04` | `8.173e-06` | `8.993e-06` | `6.062e-07` |
| biquadratic | Q1-cell Newton | 1 | `5.774e-04` | `6.687e-07` | `4.404e-08` | `2.205e-08` |

Interpretation: after one forced step, the edge/interface layer is dominated
by the deliberately crude analytic boundary clamp, so the global max is not a
geometry criterion. In the bulk, Q1-cell Newton plus biquadratic interpolation
reduces the forced-step error by more than one order of magnitude and keeps
macro/population consistency at roundoff. Long-time runs with this clamp are
not a production Poiseuille validation; real inlet/outlet and wall BCs still
need their own local tests.

Forced Poiseuille with physical-normal post-step BC:

| ghost prefill | step | max ux error | L2 ux error | max uy | rho dev | bulk max ux | bulk rho |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard wall/copy ghosts | 20 | `8.350e-03` | `1.293e-03` | `7.877e-03` | `1.358e-02` | `3.634e-03` | `6.397e-03` |
| BC-equilibrium ghosts | 20 | `1.623e-03` | `1.835e-04` | `7.931e-04` | `3.839e-03` | `4.388e-04` | `9.703e-04` |
| analytic manufactured ghosts | 20 | `3.284e-03` | `2.093e-04` | `7.931e-04` | `3.839e-03` | `4.388e-04` | `9.703e-04` |

Interpretation: the physical-normal post-step BC is finite and works as a
one-step smoke path, but it is not sufficient for long-time accuracy if the
pre-step ghost layer is still filled by the generic wall/copy policy. Replacing
that ghost layer by BC-consistent equilibrium data from tags/profiles improves
the bulk error by almost one order of magnitude at 20 steps and even beats the
manufactured analytic ghost in global max error for this diagnostic. A naive
mirrored reflected-departure variant was also tried; it did not improve the
long run because the mirror interpolation stencil can still touch ghost-layer
data near the wall. The production-relevant path is therefore a BC-aware
pre-step ghost/reflected-departure policy with controlled interpolation
stencils.

Forced Poiseuille brick ladder, latest diagnostic:

| mode | force scale | step | max ux error | L2 ux error | bulk max ux | bulk rho |
|---|---:|---:|---:|---:|---:|---:|
| regular + biquadratic + physical-normal BC | `1.0` | 20 | `1.219e-03` | `1.685e-04` | `2.274e-04` | `4.153e-04` |
| regular + biquadratic + analytic clamp | `dx_ref^2/(6ν)` | 20 | `1.046e-03` | `1.422e-04` | `2.507e-04` | `4.313e-04` |
| global physical departure + biquadratic + analytic clamp | `dx_ref^2/(6ν)` | 20 | `1.033e-03` | `1.337e-04` | `1.291e-04` | `2.717e-04` |
| global physical departure + adaptive biquadratic + analytic clamp | `dx_ref^2/(6ν)` | 20 | `2.661e-04` | `7.084e-05` | `9.127e-05` | `1.408e-04` |

Interpretation: the previously reported "roundoff" layer was the local algebra
and the Q1 departure geometry, not the full forced multistep solve. The brick
ladder shows:

- with continuum `Fx`, even an analytic-pull oracle has a one-step bulk error
  `6.219e-07`;
- with `Fx` scaled by `dx_ref^2/(6ν)`, that oracle drops to `4.718e-16`, so the
  interior forced parabolic state is a discrete invariant;
- the remaining one-step global error comes from interface/physical boundary
  stencils: margin 0 is `5.781e-04`, margin 1 is `6.081e-05`, and margin 2 is
  back to `6.608e-08`;
- direct physical multi-block departure plus adaptive interpolation reduces the
  20-step manufactured case by roughly 4x globally and 2-3x in the bulk, but it
  is still not a roundoff proof because the open boundary/interface layer remains
  a real numerical boundary condition.

Local BC unit diagnostic:

| BC family | faces | population error | moment error |
|---|---|---:|---:|
| Zou-He velocity | west/east/south/north | `5.551e-17` | `2.220e-16` |
| Zou-He pressure | west/east/south/north | `4.163e-17` | `2.220e-16` |
| Ladd moving wall | west/east/south/north | `3.469e-18` | n/a |

This diagnostic exposed and fixed a production sign bug in
`src/kernels/boundary_rebuild.jl`: south/north pressure reconstruction used
the opposite sign for `u_y`. It also led to a long-time Cartesian channel
check that exposed wrong south/north diagonal pair/sign formulas. The local
reference-patch and Cartesian orientation tests now catch those cases directly.

Cartesian Poiseuille BC integration, production TRT + BC rebuild:

| orientation | steps | max profile error | L2 profile error | cross velocity | rho dev |
|---|---:|---:|---:|---:|---:|
| west/east | 1500 | `1.176e-03` | `6.986e-04` | `7.924e-04` | `8.204e-03` |
| south/north | 1500 | `4.962e-04` | `1.200e-04` | `3.351e-04` | `3.591e-04` |

The south/north channel initially produced NaNs before the y-face diagonal
Zou-He formulas were fixed. This test is a smoke/integration guard, not a final
accuracy claim.

Gmsh BC semantics audit:

| block | logical edge | tag | physical normal | Cartesian kernel normal | angle |
|---|---|---|---|---|---:|
| block_A | west | wall_bot | `(0,-1)` | `(-1,0)` | 90 deg |
| block_A | south | outlet | `(1,0)` | `(0,-1)` | 90 deg |
| block_A | north | inlet | `(-1,0)` | `(0,1)` | 90 deg |
| block_B | east | wall_top | `(0,1)` | `(1,0)` | 90 deg |
| block_B | south | outlet | `(1,0)` | `(0,-1)` | 90 deg |
| block_B | north | inlet | `(-1,0)` | `(0,1)` | 90 deg |

Interpretation: after `autoreorient_blocks`, the interface is correctly mapped
to a west/east pair, but every physical boundary is rotated relative to the
logical face normal expected by the existing Cartesian BC kernels. Therefore
the corrected `BCSpec2D` path is valid for Cartesian/local tests, but must not
be applied directly to this body-fitted Gmsh case. The next BC step is a
physical-normal boundary operator, not another analytic clamp.

Physical-normal BC unit prototype:

| group | cases | max population error | max moment error |
|---|---:|---:|---:|
| velocity | 16 | `0.000e+00` | `1.110e-16` |
| pressure | 16 | `1.110e-16` | `2.220e-16` |
| wall Ladd | 16 | `0.000e+00` | `1.110e-16` |
| actual Gmsh tags | 6 | `5.551e-17` | `1.110e-16` |

Interpretation: a local operator that separates `logical_edge` from
`physical_normal` fixes the semantic gap exposed above. This is still a
CPU/local diagnostic, not a production kernel, but it proves the algebra needed
for the body-fitted Gmsh boundaries.

Gmsh physical-normal tag/profile diagnostic:

| block | logical edge | tag | physical normal | population error | moment error |
|---|---|---|---|---:|---:|
| block_A | west | wall_bot | south | `0.000e+00` | `2.220e-16` |
| block_A | south | outlet | east | `2.082e-16` | `4.441e-16` |
| block_A | north | inlet | west | `1.388e-17` | `2.220e-16` |
| block_B | east | wall_top | north | `0.000e+00` | `2.220e-16` |
| block_B | south | outlet | east | `2.082e-16` | `2.949e-16` |
| block_B | north | inlet | west | `4.163e-17` | `2.220e-16` |

Interpretation: with the actual Gmsh tags and the analytic Poiseuille profile
`u_x(y)`, the physical-normal reconstruction is exact to roundoff. This closes
the local BC/tag/profile layer; the remaining work is coupling this operator to
the SLBM streaming state and ghost policy.

## Production/Test Hooks

The run also required two repository-level hooks:

- `src/multiblock/exchange.jl`: exchange now rejects non-opposite interface
  edge pairs and shared-node exchange rejects flipped interfaces before
  copying.
- `src/curvilinear/slbm.jl` and `src/multiblock/mesh_extend.jl`:
  `build_slbm_geometry(...; departure=:q1_newton)` and
  `build_block_slbm_geometry_extended(...; departure=:q1_newton)` now
  precompute Q1-cell Newton departures without changing the default
  local-Jacobian mode.
- `src/kernels/boundary_rebuild.jl`: south/north Zou-He pressure rebuild now
  reconstructs `u_y` with the correct sign, and south/north velocity/pressure
  diagonals use the correct opposite populations and tangential signs.
- `test/runtests.jl`: the multiblock tests used by this workflow are now
  included in the main test suite.
