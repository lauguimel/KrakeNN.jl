# 3D steady FV solver — GPU receipts of increment I1

Scripts for increment I1 of the 3D steady finite-volume plan
(`src/methods/inc_ns/fv_steady_residual_3d.jl`, `fv_newton_3d.jl`). The CPU
part of the validation (V0: reduction to 2D, coloured = dense Jacobian, MMS
smoke, simple-shear law invariant, kernel inference) is the test file
`test/analytical/incns_fv3d_test.jl`.

Every script:

- loads CUDA/CUDSS at top level when `FV3D_BACKEND=cuda` (default), and runs
  on the CPU with UMFPACK when `FV3D_BACKEND=cpu` (tiny smoke sizes only);
- takes its grid sizes from environment variables (below);
- prints one `SUMMARY` line per case and a final `PASS <script>` or
  `FAIL <script>: <reasons>` line against the plan's gates, and exits 0 / 1;
- writes its numbers to `FV3D_OUT` (default
  `benchmarks/fv3d/results/<script>_<backend>.toml`).

| Script | What | Gate | GPU command (defaults) |
|---|---|---|---|
| `v0p_mms.jl` | V0': 3D MMS (variable ν), 4 side sets (walls, periodic x, xy, xyz) × Stokes/NS (NS skipped for xyz: singular Jacobian), L2 and L∞ of velocity and pressure, orders, location of the largest pressure error (wall distance, wall-layer share) | finest pair, L2 velocity: Stokes ≥ 1.8, NS ≥ 0.9; Stokes pressure ≥ 1.27 (2D MMS 1.47 − 0.2; target 1.5 reported) | `FV3D_NS=16,24,32,48` (`FV3D_SIDES=` filters the sets) |
| `v1_duct_cavity.jl` | V1: square duct vs Fourier series; cubic cavity Re 100/400 centrelines | duct L2 velocity order ≥ 1.8 (sections `FV3D_DUCT_NS`, ≥ 3 grids); cavity reported only | `FV3D_DUCT_NS=32,64,128 FV3D_CAVITY_N=32,48 FV3D_RE=100,400` |
| `v2_small_lbm.jl` | V2-small: cavity vs `run_cavity_3d`; GNF plane channel vs `run_gnf_channel_3d` | relative L2 velocity deviation ≤ 3 % per case | `FV3D_LBM_N=24 FV3D_RE=100 FV3D_LBM_STEPS=60000 FV3D_GNF_N=32 FV3D_GNF_STEPS=120000` |
| `step2_measure.jl` | step 2: CPU/GPU parity + zero device allocation; colours, nnz, memory, first factorisation, time per Newton iteration (cavity Re 100, Newtonian and GNF) | parity 1e-12, 0 device bytes, Newton converged | `FV3D_NS=32,48,64 FV3D_CHECK_N=8` |
| `i2_adjoint_measure.jl` | I2: CPU/GPU parity of adjoint gradients (uniform ν, ν field, side velocity, x source, NN law parameters) for the composite PIV objective (3 planes + flow-rate/Fx + wall-force terms); device bytes of the objective and of the gradient; gradient cost vs median Newton iteration (finalize / factors reused) | parity 1e-12 relative (∞-norm), ≤ 1024 device bytes for the all-linear objective and gradient, the same at `FV3D_CHECK_N` and at the largest `FV3D_NS`, Newton converged | `FV3D_CHECK_N=8 FV3D_NS=32` (`FV3D_NS=32,48` for the cost curve) |
| `i4_krylov_measure.jl` | I4 (input of D2): branch K (GMRES, `KrylovBackendTag`) with P1 = ILU(0), P2 = block SIMPLE and P3 = block SIMPLE with multigrid blocks (`mg`) and P3L = block upper triangular with LSC (`lsc`; other `KrylovBackendTag` options through `FV3D_TAG_EXTRA`) side by side on a Carreau duct with obstacle at Re 50, imposed flow rate and body force; GMRES iterations per Newton step, forward / transposed iterations, time, device bytes per solve (forward η = 1e-6, transposed 1e-10), pivot report, adjoint gradient, device memory of each branch measured alone; branch D (cuDSS) compared where it fits | on the gated preconditioners (`FV3D_GATE`, default P2): Newton and transposed solves converge; forward iterations ≤ 200 at the largest N and growing at most linearly in N; memory ≤ `FV3D_MEM_GB` (9); K vs D state and gradient 1e-6 | `FV3D_NS=24,32,48 FV3D_DIRECT_MAX=32 FV3D_PRECONDS=ilu0,simple` |
| `i5_mg_diagnose.jl` | I5 (input of D2): diagnostic of the block preconditioners at the converged duct state, per `KrylovBackendTag` configuration (`FV3D_TAGS`): stationary multigrid reduction per level (a), GMRES on A (b), on S̃ (c), on the true Schur complement with exact A (e), on J (d), each also with exact inner solves | none (measurements only) | `FV3D_NS=16,24,32` |
| `v0p_energy_mms.jl` | I3, V0' energy rows: manufactured temperature on N³, conduction (u = 0) and convection (V0' flow, viscous heating), side sets walls / periodic x, L2 and L∞ of T, orders | finest pair, L2 and L∞ of T: conduction ≥ 1.8, convection ≥ 0.9; with `FV3D_LINSOLVE=krylov`: every GMRES solve converges, K vs D state 1e-6 for N ≤ `FV3D_DIRECT_MAX` (24) | `FV3D_NS=12,16,24` (10 GB), `16,24,32,48` (80 GB); `FV3D_LINSOLVE=krylov FV3D_NS=16,24,32,48,64` (run on an 80 GB H100, peak 4.7 GB); `FV3D_CASES=` filters |
| `v4_heated_duct.jl` | I3, V4: (a) heated periodic square duct vs a separated-variable reference (no LBM, see below); (b) 2D channel with obstacle, thermal CY law and heating vs its Nz = 3 extrusion on the device | (a) relative L2 and L∞ of θ ≤ 3 % at the finest N; (b) residual h × 2D to 1e-14, solves to 1e-7; with `FV3D_LINSOLVE=krylov` ((a) only): every GMRES solve converges, K vs D state 1e-6 for N ≤ `FV3D_DIRECT_MAX` (24) | `FV3D_NS=16,24 FV3D_EXT_N=32` (10 GB), `FV3D_NS=16,24,32 FV3D_EXT_N=64` (80 GB); `FV3D_LINSOLVE=krylov FV3D_NS=16,24,32,48 FV3D_EXT_N=64` (run on an 80 GB H100, peak 4.2 GB) |

Run on the cluster (after `setup_fv3d.sh` on the login node, with
`~/kraken_fv3d/src_fv3d` a git archive of `feat/fv-3d`), each as
`FV3D_BACKEND=cuda julia --project=env src_fv3d/benchmarks/fv3d/<script>.jl`
with `JULIA_DEPOT_PATH=~/kraken_fv3d/depot` and `JULIA_CUDA_USE_COMPAT=false`.

CPU smoke runs (checked 2026-11-09, each under 15 s on a laptop):

```
FV3D_BACKEND=cpu FV3D_NS=4,6 julia --project benchmarks/fv3d/v0p_mms.jl        # PASS
FV3D_BACKEND=cpu julia --project benchmarks/fv3d/v1_duct_cavity.jl             # PASS (duct 6,8,12; cavity 6³)
FV3D_BACKEND=cpu julia --project benchmarks/fv3d/v2_small_lbm.jl               # FAIL expected: N = 5 LBM is too coarse
FV3D_BACKEND=cpu FV3D_NS=6,8 julia --project benchmarks/fv3d/step2_measure.jl  # PASS (device checks skipped)
FV3D_BACKEND=cpu FV3D_CHECK_N=6 FV3D_NS=8 julia --project benchmarks/fv3d/i2_adjoint_measure.jl  # not run yet
FV3D_BACKEND=cpu FV3D_NS=6,8 julia --project benchmarks/fv3d/i4_krylov_measure.jl   # not run yet
FV3D_BACKEND=cpu FV3D_NS=4,6 julia --project benchmarks/fv3d/v0p_energy_mms.jl     # not run yet
FV3D_BACKEND=cpu FV3D_NS=4,6 FV3D_EXT_N=8 julia --project benchmarks/fv3d/v4_heated_duct.jl  # not run yet; (a) above 3 % expected at N = 6
```

Notes.

- Memory: the direct factorisation grows like cells^(4/3) (about 25 GB at 48³,
  more than 80 GB at 64³). Grids that run out of GPU memory are reported as
  `SKIPPED` (v0p, v1 duct/cavity) and listed under `skipped_oom` in the TOML; v2
  counts it as a failure. In v2 the FV cavity grid is (2N)³ (h = ½), so
  `FV3D_LBM_N=32` meant 64³ cells and ran out of memory; N = 24 gives 48³.
- v0p `periodic_xyz` (no wall): the mean velocity is a free mode and the
  momentum source is made zero-mean; Stokes only (the NS Jacobian is singular
  without a mean-flow row, Newton stalls).
- v1 duct sections come from `FV3D_DUCT_NS` only (not `FV3D_NS`, which other
  scripts share); fewer than 3 grids is a failure with that message.

- Cubic cavity: lid at z = 1 moving in +x; `u` on the vertical centreline
  (x = y = ½), `w` on the horizontal one (y = z = ½). No Ku–Hirsh–Taylor
  (1987) or Albensoeder–Kuhlmann (2005) numbers are shipped; pass a TOML file
  as `FV3D_CAVITY_REF` (tables `re100`, `re400` with any of `u_min`, `z_umin`,
  `w_max`, `x_wmax`, `w_min`, `x_wmin`) to print deviations, otherwise the
  line says `NOREF`.
- V2 cavity geometry: the LBM driver puts halfway bounce-back on five walls and
  the Zou–He lid on the top node layer, so its domain is N × N × (N − ½) lattice
  units; the FV grid is 2N × 2N × (2N − 1) cells of h = ½ and each LBM node is
  compared with the mean of the 8 FV cells around it.
- The Rhie–Chow coefficient keeps the 2D form d_f = h² / (4ν_f + |ū·n| h), so
  that a z-uniform 3D state reproduces the 2D residual exactly (V0a).

## 3D adjoint (increment I2)

`src/methods/inc_ns/fv_adjoint_3d.jl` adds the 3D methods of the generic
adjoint layers (transposed solve with the Newton factors, parameter duals
through the residual kernel): maps `FVUniformViscosity`, `FVViscosityField`,
`FVSideVelocity` (six sides), `FVUniformSource` (comp 1..4), `FVLawParameters`
(law capture with six face slots); objectives `fv_flow_rate`,
`fv_flow_force_objective` (Fx) and `FVWallForce` (legacy kind, explicit term).
`src/methods/inc_ns/fv_piv_objective.jl` adds `FVPIVObjective`: plane-PIV
misfit (`piv_misfit / σ_u²` per plane) plus weighted terms, with a device path
(normal-contraction and scatter kernels, four matrix products per measured
component, dense weight vectors for linear terms) and host duals for nonlinear
terms only. CPU validation (V3): `test/analytical/incns_fv3d_adjoint_test.jl`
(adjoint vs centred FD for every map × objective, gate 2e-7 + 1e-6 relative,
best of steps 1e-4 / 2e-5; law contraction vs parameter duals 1e-12; `abs`
convention at zero-flux faces).

`abs` convention: ForwardDiff takes `abs'(x)` from the sign bit of `x`, so at
an exactly zero face flux the Jacobian (and the adjoint) holds one of the two
one-sided derivatives (the one picked by the signed zero), not their mean. The
plan assumed `abs'(0) = 0`; the test records the measured behaviour.

Device bytes of the all-linear path: the I2 claim was "0 bytes". The gate is
now ≤ 1024 bytes for `fv_objective_gradient!` and `fv_steady_gradient`
(prebuilt workspace and objective), and the bytes must be the SAME at the
parity size and at the largest cost size: a small fixed amount (scalars moved
through device memory) is tolerated, anything that grows with the grid is not.

## 3D energy equation (increment I3)

`src/methods/inc_ns/fv_energy_3d.jl` adds `FVEnergy3D`, the opt-in
temperature equation div(uT) − α∇²T = 2σν S:S + q of the 3D problem
(`fv_steady_problem(Nx, Ny, Nz; energy = FVEnergy3D(...))`), as additive
methods on `FVSteadyProblem3D`; the 2D files are not changed and
`energy = nothing` keeps the I1 path. The row is the 2D energy row face by
face (face area h², cell volume h³, six sides, linear-upwind or upwind
convection with the Rhie–Chow flux), θ = T − T_ref is the fifth cell unknown
(`U[5:5:end]`, T_ref = mean of the prescribed wall temperatures), solid rows
are storage identities. Also 3D: `FVStateTemperature` in GNF laws (η(γ̇, T)),
`fv_thermal_continuation` (σ steps), the round-off floor and the scaled
gates of the thermal Newton (`:tolerance` or `:roundoff_floor`, as in 2D;
`thermal_scales` overrides the case-data scales), the law adjoint through T
(`FVLawParameters`), `fv_mean_temperature` and `fv_temperature_l2`. Only wall
and periodic flow sides take an energy equation; each fluid component must
touch a prescribed temperature (checked at construction).

Branch K takes the temperature without code change: the block SIMPLE puts
every non-pressure field in the velocity block, so with energy U has
4 × ncell unknowns and the Schur part stays the pressure. ILU(0) works on the
whole matrix as before. Both are tested against branch D on the CPU.

V4 reference. The plan asks for θ within 3 % of the 3D thermal LBM on a heated
duct. The 3D thermal LBM driver covers closed cavities only (no periodic
body-force duct, no volumetric heat source); a duct driver is estimated at
half a day of development plus its GPU runs and was not written. `v4` (a)
uses instead a separated-variable reference of the same duct (unidirectional
flow, source q0 + q1 cos kx): θ0 and the complex x-mode Θ1 solved on a fine
section grid (≥ 300 cells per side) by sparse LU on the CPU. This checks the
convection, diffusion, source and wall terms in 3D against an independent
solution, but not against another discretisation of the coupled flow.

CPU validation: `test/analytical/incns_fv3d_thermal_test.jl` (in
`runtests.jl` after the Krylov file): (a) z-uniform 3D residual = h × 2D
residual with pins cleared, with and without heating and with a thermal law;
(b) coloured = dense Jacobian on 4 × 5 × 6; (c) law adjoint through T vs
centred FD (2e-7 + 1e-6 relative) and contraction vs duals (1e-12); (d)
thermal Newton, 3D = extruded 2D, branch K, heating continuation; (e)
argument checks. 159/159 pass (Aqua CPU, 2026-11-11).

GPU results (H100, direct solver through CUDSS, 2026-11-11/12). V0' energy
at N = 16, 24, 32, 48: L2 order of T on the last three pairs 2.00–2.01 for
conduction (walls and periodic x) and 1.66 → 1.57 for convection (it
decreases slowly with N; gate 0.9). V4 (a) duct at N = 16, 24, 32: θ
relative L2 2.5e-3 at N = 32, order 1.99–2.00. V4 (b) 128 × 64 channel and
its extrusion: residual and solve equal the 2D ones to 2e-15. Device memory
measured (the earlier estimates, scaled from I1 step 2, were low): 48–58 GB
for the 48³ cube, 27.9 GB for the N = 32 duct, 8 GB for the N = 24 duct.
Branch K with temperature is tested on the CPU only. The scripts print the
device memory in use after each solve.

Branch K in the I3 scripts (not run yet). The direct factorisation limits V0'
energy to 48³ and V4 (a) to N = 32 on one 80 GB GPU. `FV3D_LINSOLVE=krylov`
solves the Newton steps of `v0p_energy_mms.jl` and of `v4_heated_duct.jl` (a)
with GMRES + P3L, the V5 tag (`fv3d_p3_tag` in `fv3d_common.jl`, also used by
`fv5_linsolve`: block upper triangular, LSC Schur, 4 W-cycles of multigrid per
block, l1-Jacobi, forward η = 1e-6, restart 50, at most 2000 iterations);
`FV3D_LINSOLVE=direct` (default) keeps cuDSS. The tag goes to
`fv_newton_solver(p; linsolve)`; the thermal Newton and its gates are the
same for both solvers (inexact Newton steps with forcing 1e-6), so the order
and θ gates are unchanged. Per grid the scripts print the GMRES iterations of
every Newton step and their maximum, the number of unconverged GMRES solves
(any is a failure: the tag is non-strict, so the script counts them) and, for
N ≤ `FV3D_DIRECT_MAX` (default 24), the relative max difference between the
Krylov and the direct state (gate 1e-6). V4 (b) stays on the direct solver:
its size is set by `FV3D_EXT_N` and its 1e-7 gate compares two direct
solves. Memory of the Krylov branch: linear in cells (the earlier estimate
of 20–35 GB at 64³ was high).

Krylov branch on the GPU (H100, P3L tag): every gate passes. V4 (a) to
N = 48: θ relative L2 1.09e-3, L∞ 1.15e-3, orders L2 1.99/2.00/2.02, peak
device memory 4.2 GB (27.9 GB with cuDSS at N = 32), largest Newton 44 s.
V0' energy to 64³: conduction orders 2.00 (both side sets), convection
1.54 (walls) and 1.52 (periodic x) on the finest pair, peak device memory
4.7 GB. The same rows on a 10 GB MIG slice up to N = 32 pass with 1.2 GB.

## 3D branch K: GMRES + ILU(0) or block SIMPLE (increment I4)

`src/solve/linear_solve_krylov.jl` adds `KrylovBackendTag` to the linear-solve
seam: restarted GMRES (`gnf_gmres!`, restart 50, classical Gram–Schmidt twice)
right-preconditioned by ILU(0) of the assembled Jacobian (`KrylovCSRMatrix`,
CSR on the device of the assembler). The Newton and adjoint API is unchanged:
`fv_newton_solver(p; to, linsolve = KrylovBackendTag())`. GPU methods
(cuSPARSE `ilu02!`, triangular solves with the analysis kept between
applications, cuDSS LU of the P2 Schur matrix) are in `ext/KrakeNNCUDSSExt.jl`,
so `using CUDA, CUDSS` is needed on GPU as for branch D; SpMV and the P2 block
kernels are KernelAbstractions kernels in the core file.

- Tolerances: forward `rtol = 1e-6` (fixed forcing term of an inexact Newton,
  no Eisenstat–Walker adaptation), transposed `rtol_transpose = 1e-10`, both
  on the true residual ‖b − A x‖/‖b‖.
- Preconditioner P1 = ILU(0) of the whole saddle-point matrix (u, v, w, p
  interleaved per cell). Rhie–Chow puts a non-zero pressure diagonal in every
  continuity row, so ILU(0) does not break down structurally; every
  factorisation checks |Uᵢᵢ| / maxⱼ|Aᵢⱼ| per row, raises on a zero or
  non-finite pivot and reports small ones (`lin_krylov_pivot`). First Aqua
  measurement (2026-11-09): correct (K vs D 2e-11 on the CPU test grids) but
  weak — forward iterations 43 → 61 (N = 6 → 8) with imposed flow rate, GMRES
  stalls at the 2000 cap at N = 32 on GPU. D2 fails for P1.
- Preconditioner P2 (`precond = :simple`, plan Étape 3 / R5) = block SIMPLE:
  unknowns split into velocity-like U (every field but pressure), pressure P
  and extras X (the flow-rate border); J = [A G; D C] on U ∪ P,
  S̃ = C − D diag(A)⁻¹ G; one application is û = ILU(0)(A)⁻¹ v_U,
  p = S̃⁻¹ (v_P − D û), u = û − diag(A)⁻¹ G p, z_X = v_X (the border is left to
  GMRES, as with `border = :krylov`). S̃ is factorised by LU (`schur = :lu`,
  plan choice: cuDSS on GPU, UMFPACK on CPU) or by ILU(0) (`schur = :ilu0`,
  cheaper in memory, weaker). Block and S̃ patterns are built once on the
  host; values are refreshed on device (S̃ one thread per pressure row, no
  stored products). The transposed solve builds a second SIMPLE from the blocks
  of Jᵀ (its Schur matrix is S̃ᵀ, factorised on its own), so no transposed
  cuDSS solve is used. `lin_krylov_blocks(cache)` gives the block sizes.
- Flow-rate border (R11): the full augmented system goes to GMRES; with
  `border = :krylov` (default) the ILU(0) is taken on the local block only
  (border row and column removed, unit diagonal), `border = :ilu` keeps the
  border inside the ILU(0). Prototype, judged by D2.
- Transposed solve: the transposed pattern and a permutation are built once
  on the host at the first adjoint; after each refactorisation Jᵀ and the
  factor values (UᵀLᵀ) are gathered on device. Extra memory: about 24 bytes
  per non-zero.
- Memory of branch K ≈ 52 bytes per non-zero + 8 (restart + 1) bytes per
  unknown (`krylov_estimate` in the script prints it per N), against cuDSS
  factors that grow like cells^(4/3).
- Device allocations per solve: the GMRES work vectors (`r`, Hessenberg column,
  cycle coefficients) are preallocated with the basis, and SpMV is a
  KernelAbstractions kernel; the former cuSPARSE `mul!` allocated its workspace
  at every call, i.e. at every GMRES iteration (the 21 MB per solve seen at
  N = 32, inferred, not measured separately).
- The script releases each solver before the next build (device arrays freed,
  cuDSS objects finalized, GC, `CUDA.reclaim`), so each branch's memory is
  measured alone.
- CPU validation: `test/analytical/incns_fv3d_krylov_test.jl` (K vs D solves
  forward/transposed, Newton state, adjoint gradients with and without imposed
  flow rate, one FD check; for P2 with Schur LU and Schur ILU(0): S̃ against a
  dense C − D diag(A)⁻¹ G, solves, Newton state, adjoint gradients).

### P3: block SIMPLE with multigrid blocks (not run yet)

`precond = :mg` (code in `src/solve/linear_solve_multigrid.jl`) keeps the
block SIMPLE of P2 and replaces both block solves by one multigrid cycle: the
velocity block A and S̃ each get their own hierarchy (S̃ by default,
`schur = :auto`; `precond = :mg, schur = :lu` keeps the LU of S̃ = "P3s",
`precond = :simple, schur = :mg` puts multigrid on S̃ only).

- Coarsening: unsmoothed aggregation on the cell grid, aggregates of 2 × 2 × 2
  cells per field component (u, v, w never mixed), solid and fluid cells never
  in the same aggregate (the pinned pressure cell stays a fluid cell, so the
  coarse S̃ keeps the pin). P is piecewise constant (P 1 = 1), R = Pᵀ,
  coarse operators Pᵀ A P (Galerkin). Patterns and the list of fine
  positions summed into each coarse entry are built once on the host; values
  are a gather-sum on the device at each refactorisation (no atomics). The
  matrix needs its grid (`krylov_csr_matrix(...; geometry)`, passed by
  `fv_newton_solver`); without it P3 raises.
- Cycle: W (default, `mg_cycle = :V` possible), `mg_sweeps` (2) pre- and
  post-smoothing sweeps of l1-Jacobi (default), damped Jacobi (ω = 0.7) or
  ILU(0) (`mg_smoother`), coarsening stops at `mg_coarse` (2000) unknowns,
  coarsest level by LU (cuDSS on GPU, UMFPACK on CPU, same hooks as the P2
  Schur LU). `mg_cycles` cycles per application, `mg_scale` scales the
  coarse correction (unsmoothed aggregation tends to under-correct; 1 by
  default). The cycle is a fixed linear operator, which restarted GMRES
  needs; a K-cycle (inner Krylov) would not be.
- Transposed solve: built from the blocks of Jᵀ as in P2, so its hierarchy is
  that of Aᵀ, i.e. the transposed coarse operators. With (damped) Jacobi the
  cycle on Aᵀ is exactly the transpose of the cycle on A (tested); l1-Jacobi
  (row sums) and ILU(0) are not, which only changes the GMRES count of the
  adjoint, not its answer.
- Memory: linear in cells (operator complexity printed per run, expected
  ≈ 1.15–1.3 in 3D), plus one LU of at most `mg_coarse` unknowns per
  hierarchy.
- Benchmark: `FV3D_PRECONDS=mg[,mg_lu,simple]`, knobs `FV3D_MG_*` (header of
  the script). The D2 count for P3 is the largest GMRES count of the
  forward solve and of every Newton-step solve; one line
  `D2 P3 <case>: PASS|FAIL` per case.
- Case "source": the PTC shift of `fv_newton!` was h²/Δτ while the 3D residual
  is integrated over h³ cells, so `dtau0 = 1` was a physical step of h = 1/N and
  the body-force case needed about 2.5 N iterations (N = 8: 47 observed). Fixed
  in the library: the shift is now V/Δτ with V = h³ in 3D (h² in 2D, unchanged),
  so `dtau0` means the same in both. The `× N` workaround is removed; the source
  case uses `dtau0 = FV3D_SOURCE_DTAU` (default 1), full force first, force
  continuation only as a fallback (`FV3D_SOURCE_RAMP`). Other 3D runs with
  `dtau0 = 1.0` (v1, v2, step2) now have a shift N times weaker than before
  (not re-measured). Regression test: last testset of `incns_fv3d_test.jl`.
- Risk, stated before measuring: the SIMPLE approximation S̃ of the true
  Schur complement may itself make GMRES counts grow with N, whatever the
  block solvers; `FV3D_PRECONDS=mg,mg_lu,simple` separates the two effects.
- CPU validation: testset (5) of `test/analytical/incns_fv3d_krylov_test.jl`.
- Measured (flow-rate case, Re 50, H100, reported 2026-11-10): forward GMRES 75 / 94 / 146 at
  16³ / 24³ / 32³; the stationary aggregation cycle has an asymptotic factor
  ≈ 0.9 on the velocity block and up to 0.97 on S̃ (smooth errors poorly
  corrected by piecewise-constant transfers). A block-triangular LSC
  preconditioner with exact inner solves takes 26 / 26 iterations, 134 /
  173 / 450 with the same aggregation cycles inside.

### P4: monolithic multigrid with an additive Vanka smoother (not run yet)

`precond = :vanka` (code in `src/solve/linear_solve_mg_monolithic.jl`, hooks
`_krylov_block_build` and the tag fields `vanka_omega`, `vanka_prolong` in
`linear_solve_krylov.jl`) drops the SIMPLE splitting: one multigrid cycle on
the coupled velocity–pressure block of J is the preconditioner, the flow-rate
border is left to GMRES as in P2/P3 (a rank-one bordering of the
preconditioned operator: at most one extra GMRES iteration in exact
arithmetic). Target set by the P3 measurement: about 30 iterations flat in N,
at less memory than the direct solve.

- Hierarchy: the P3 aggregation of the cell grid (2 × 2 × 2 blocks, solid and
  fluid apart), all four unknowns of a cell coarsened together so every
  level keeps the (u, v, w, p) block structure. Transfers: restriction by
  sums (the finite-volume restriction), prolongation trilinear and
  cell-centred (3/4 own block, 1/4 nearest block per axis, same field and
  class; dropped and renormalised at walls, wrapped on periodic axes), so
  the transfer orders add up to 3 > 2 (Wesseling–Khalil cell-centred
  multigrid; the aggregation of P3 adds up to 2, hence its 0.9 factor).
  `vanka_prolong = :constant` keeps P = Rᵀ for the A/B. Coarse operators
  R J P (Petrov–Galerkin) of the full saddle-point block: summing continuity
  rows cancels the interior mass fluxes of a block, the Rhie–Chow
  (1, −4, 6, −4, 1)/4 pressure stencil on the trilinear interpolant keeps a
  positive diagonal (3/8 · h d_f per axis per block in 1D) and a zero row
  sum, the pinned row contributes to its block diagonal: the coarse C block
  stays a stabilisation of the same kind, no rediscretisation. Patterns on
  the host once (coarse stencil up to 5 × 5 × 5 cells, operator complexity
  expected ≈ 2.1–2.3, grid complexity ≈ 1.14), values one thread per coarse
  row on the device (binary search in the sorted row, no atomics).
- Smoother: additive Vanka, one patch per fluid pressure unknown = the
  pressure, the three velocities of its cell and the face-normal velocities
  of its six neighbours (10 dofs; a single-cell 4 × 4 patch would be point
  Jacobi on this collocated scheme, since the continuity row of a cell has
  a zero coefficient on its own velocity in the interior). Local matrices
  gathered from the CSR rows, inverted in place (Gauss–Jordan, partial
  pivoting, one thread per patch) at each refactorisation; a sweep adds
  ω / multiplicity times the local solutions (a velocity lies in up to three
  patches), ω = `vanka_omega` (0.7), damped Jacobi on uncovered rows (solid
  cells). Cycle and knobs as P3 (`mg_cycle` W, `mg_sweeps` 2, `mg_coarse`,
  `mg_levels`, `mg_cycles`, `mg_scale`), coarsest level by the P2 LU hooks
  (cuDSS / UMFPACK). The cycle is a fixed linear operator: restarted GMRES,
  no FGMRES.
- Transposed solve: a second hierarchy with the same rule on the gathered
  Jᵀ (not the transpose of the forward hierarchy: the transfer orders are
  what matters), same patches with the local matrices of Jᵀ.
- Diagnostics: `lin_krylov_stationary(cache, b; iters, transpose,
  local_block)` gives the residual ratios of the stationary iteration with
  one application per step (the convergence factor of one cycle, first and
  tail); the benchmark prints them per N for P3 and P4 ("SUMMARY ...
  stationary factor"), with the hierarchy line (levels, nnz, patches,
  operator complexity).
- Memory per side (64³, estimated, not measured): fine values gathered
  0.45 GB (flow-rate case only), coarse levels ≈ 0.8 GB at operator
  complexity 2.2, transfers ≈ 0.1 GB, patch inverses 0.2 GB, work vectors
  0.1 GB; with J, Jᵀ, both sides, the GMRES basis and the Newton/adjoint
  vectors about 6–7 GB in all, against cuDSS factors growing like
  cells^(4/3).
- Benchmark: `FV3D_PRECONDS=vanka[,mg]`, knobs `FV3D_VANKA_OMEGA`,
  `FV3D_VANKA_PROLONG` and the `FV3D_MG_*` of P3; the D2 count and the
  `D2 P4 <case>` line follow the P3 rule.
- CPU validation: testset (6) of `test/analytical/incns_fv3d_krylov_test.jl`
  (hierarchy, patches, solves, Newton, adjoint, FD check).
- Source-case diagnostic (written, not run): with constant prolongation P4
  passes the flow-rate case but not the body-force case (GMRES at the 2000
  cap at N = 32, transposed also at N = 48). `i4_vanka_diag.jl` takes the
  converged state of the direct branch and prints, per case and N, the
  per-level smoother and cycle factors, the singular values and eigenvalues
  of the coarsest level, the dominant mode of the stationary iteration, the
  GMRES(50) / GMRES-DR / GMRES(200) residual histories with the harmonic Ritz
  values, the flow-rate matrix solved without its border, and variants
  (ω 0.5, coarse scale 1.5, one level fewer, 4 sweeps). Hypotheses and the
  lines that decide them are in its header. Not a gate.

### I5: diagnostic of the block preconditioners, block structure and Schur approximations

Why: P3 needs 143 / 312 / 631 forward GMRES iterations at 32³ / 48³ / 64³
(gate D2: ≤ 200 at 64³). Every change that brings P3 closer to exact SIMPLE
made it worse (2 MG cycles, coarse scale > 1, exact LU of S̃ = P3s, ILU(0)
on A = P2). The SIMPLE Schur approximation S̃ = C − D diag(A)⁻¹ G is a
Laplacian. The true S = C − D A⁻¹ G is not a Laplacian on smooth pressure
modes: it is mass-like in Stokes flow and dominated by convection at Re 50.
So S̃⁻¹S spreads more and more as N grows. The inexact MG cycle on S̃ hides
part of that error, which is why making the cycle more exact made P3 worse.
Measurement (e) below confirms the diagnosis.

`benchmarks/fv3d/i5_mg_diagnose.jl` is the shared measuring tool. A
configuration is any set of `KrylovBackendTag` options
(`FV3D_TAGS="label:opt=val,…;…"`, common options in `FV3D_TAG_BASE`). At
the converged state of the i4 flow-rate duct, for each configuration:
(a) every multigrid hierarchy as a stationary iteration, reduction factor
per cycle from each starting level; (b) GMRES on A alone with the
velocity-block solver; (c) GMRES on S̃ alone with the Schur solver; (e) with
an exact velocity block, GMRES on the true S preconditioned by Ŝ⁻¹, which
measures the Schur approximation alone; (d) GMRES on J with the full
preconditioner, forward and transposed. For N ≤ `FV3D_EXACT_MAX`, each
configuration also runs with exact inner solves (`+exact`: `ablock=lu,
schur=lu`), which measures the block approximation itself. There is no gate:
it prints SUMMARY lines and writes a TOML file.

New options (code in `src/solve/linear_solve_block_schur.jl`, CPU and GPU,
transposed path included; the defaults keep P2 / P3 unchanged):

- `block`: `:simple` (default), `:lower` or `:upper` block triangular.
- `schur_approx`: `:simple`, `:simplec` (Q = l1 row sum of A), `:lsc`
  (Ŝ⁻¹ = S̃⁻¹ (α C − D Q⁻¹ A Q⁻¹ G) S̃⁻¹, Q = diag(A), i.e. scaled BFBt;
  S̃ is the stabilised Poisson matrix of the collocated grid; `lsc_alpha` = α),
  `:pcd` (algebraic: F_p = mean of the u-u, v-v, w-w blocks of A, which
  works because pressure and velocity share the cells; Ŝ⁻¹ = F_p S̃⁻¹ Q_p⁻¹;
  F_p keeps the velocity wall conditions, not the Neumann/Robin conditions
  of the PCD literature).
- `schur_q` (`:auto`, `:diag`, `:rowsum`, `:scalar` = `schur_qscale` · I) and
  `schur_shift` σ: S̃ = C − D Q⁻¹ G − σ diag(D Q⁻¹ G). With `:scalar` and q
  = cell volume, Q is the velocity mass matrix of the uniform grid (the LSC
  weight of Elman et al.). The benchmarks accept `schur_qscale=volume` (h³)
  or `=stokes` (6 ν0 h), resolved per N. Not measured yet. S̃ still depends
  on the state through C (Rhie–Chow), so it is still refactorised at every
  Newton step.
- `ablock`: `:auto` (ILU(0) for P2, MG for P3), `:ilu0`, `:mg`, `:lu`
  (exact sparse LU of A: UMFPACK / cuDSS).
- CPU validation: testset (6) of `test/analytical/incns_fv3d_krylov_test.jl`
  (346 assertions, all passing on the cluster at commit cfcbd15c4) checks the
  assembled S̃ and one application, forward and transposed, against a dense
  reference with exact inner solves. It also checks solves with ILU and
  multigrid inner solves, and Newton states and adjoint gradients against
  branch D.

Measured (H100, i5 at commit cfcbd15c4, flow-rate duct, converged state;
GMRES(50) iterations, forward to 1e-6, transposed to 1e-10; "MG" = one W-cycle
per inner solve unless stated):

| configuration | 16³ | 24³ | 32³ | transposed 16³ / 24³ / 32³ |
|---|---|---|---|---|
| P3 (SIMPLE, MG on A and S̃) | 75 | 94 | 146 | |
| LSC upper, exact A and S̃ | 25 | 26 | 25 | 39 / 42 / 42 |
| LSC upper, MG, `mg_cycles=1` | 134 | 173 | 450 | |
| LSC upper, MG, `mg_cycles=2` | 55 | 50 | 64 | |
| LSC upper, MG, `mg_cycles=4` | 29 | 28 | 34 | 45 / 43 / 49 |
| LSC upper, MG, `mg_cycles=1, mg_sweeps=4` | 66 | 63 | 97 | |
| LSC upper, MG on A, exact S̃ | 53 | 53 | 47 | |
| LSC upper, exact A, MG on S̃ | 133 | 144 | 196 | |
| PCD upper, MG | 993 | > 2000 | > 2000 | |
| PCD upper, exact A and S̃ | 189 | 198 | | |
| SIMPLE + shift σ = 4, MG | 1205 | > 2000 | > 2000 | |

On the true S at 24³ (measurement (e), exact A): LSC needs 24 iterations,
PCD 158, SIMPLE 146. At 32³, LSC with `mg_cycles=4` takes 1.3 s per forward
solve and 0.39 s per factorisation.

What this shows:

- LSC is the right Schur approximation: with exact inner solves its count does
  not grow with N (25 / 26 / 25).
- The inexact pressure solves are now the weak part. With MG on S̃ the count
  falls from 450 to 34 at 32³ when going from 1 to 4 cycles. Exact S̃ with MG
  on A gives 47–53, while exact A with MG on S̃ gives 133–196. LSC uses two S̃
  solves per application, so an under-converged S̃ cycle counts twice. The
  stationary S̃ cycle measured in (a) reduces the slowest error by a factor
  that gets worse with N (piecewise-constant aggregation). A better pressure
  hierarchy would remove the need for 4 cycles. That work belongs to the
  multigrid workstream.
- Adjoint: the transposed preconditioner is built by the same code from the
  blocks of Jᵀ (Aᵀ, G ← Dᵀ, D ← Gᵀ, C ← Cᵀ). Since diag(Aᵀ) = diag(A), its LSC
  is exactly the transpose of the forward LSC. Testset (6) checks one
  transposed application against the dense LSC of Jᵀ. The multigrid
  hierarchies are built from Aᵀ and S̃ᵀ. The block structure is upper
  triangular in the Jᵀ blocks, not the transpose of the forward one. Both give
  2 iterations with exact blocks and exact S. The transposed counts are about
  1.6× the forward ones because their tolerance is 1e-10 instead of 1e-6. Per
  decade of residual they are the same: exact LSC needs 25/6 ≈ 4.2 iterations
  per decade forward and 42/10 = 4.2 transposed.
- Measured failures, not fixed:
  - Shift σ: it adds a mass term, which is the Schur complement of the Stokes
    limit. At Re 50 the smooth-mode S is much smaller and dominated by
    convection, so the shift moves Ŝ away from S (1599 iterations on the true
    S at 24³).
  - PCD: with exact inner solves its count does not grow with N, but it is
    about 7× the LSC count. The likely cause is the algebraic F_p without the
    pressure boundary conditions of PCD (not checked).
  - Neither is worth pursuing while LSC works.

D2 gate with LSC: `i4_krylov_measure.jl` takes `FV3D_PRECONDS=lsc` (P3L:
`precond=mg, block=upper, schur_approx=lsc`, MG on A and S̃) or `lsc_lu` (P3Ls,
exact S̃). Any other `KrylovBackendTag` option goes through `FV3D_TAG_EXTRA`
(e.g. `mg_cycles=4`), applied last.

Which count the D2 gate uses (unchanged): for P3/P3s/P3L/P3Ls it is the
largest of the step-(2) forward solve and every Newton-step solve. The step-(2)
solve uses a synthetic right-hand side that is O(1) in every row, including the
continuity rows, so it is harder than a Newton step. With P3 at 64³ it needs
631 iterations against 325 for the worst Newton step, so it usually sets the
gated count. Both counts are printed (`gmres per newton step […] (max m)`,
`fwd it (synthetic rhs)`), with one line per case listing both against N. A
NOTE line flags any solve longer than the restart length (50).

## V5 design: 3D duct with obstacle, law recovery from plane PIV

- Geometry: periodic-x square duct, `Nx × Ny × Nz = 96 × 48 × 48` FV cells
  (h = 1/48), one off-centre voxel obstacle (a bar of 8 × 16 × 12 cells
  touching the bottom wall, so no symmetry plane), driven by a uniform body
  force (the LBM driver imposes the force only, so the FV problem uses the
  same uniform x source, no `u_mean_target`).
- Truth: `run_gnf_duct_obstacle_3d` (src/drivers/gnf_duct_3d.jl, D3Q19 TRT,
  Guo force, LI-BB, same voxel mask, halfway bounce-back on the obstacle) with
  a Carreau–Yasuda law (n = 0.5, λ chosen so that λ γ̇ spans 0.1–10 in the
  wake); one LBM node per FV cell (`q_w = 0.5`, so walls sit on FV faces),
  velocities rescaled from lattice units; the LBM mean flow rate gives a flow
  rate target term.
- Observations: three PIV planes from the LBM field through `piv_operator` on
  the FV grid — a stereo z-plane through the obstacle mid-height
  (thickness 2h, window 2h × 2h), an x-plane one obstacle length downstream,
  a y-plane through the wake — with 2 % velocity noise (`σ_u`), plus the
  flow rate through one x slab as a target term (the pressure-drop observable
  of a force-driven duct).
- Law to recover: `FVLawParameters` of (i) the Carreau–Yasuda law from a wrong
  start (η0 × 2, n = 0.8), (ii) a `TinyMLP` `LearnedViscosity` on (γ̇); gate =
  recovered η(γ̇) within the 95 % band of the identifiability estimate
  `(ŜᵀŜ)⁻¹` over the γ̇ range seen by the planes, plus prediction of a held-out
  plane.
- Cost: one forward solve (~10 Newton iterations) + one gradient per
  optimiser step; extrapolating the step-2 memory (25 GB at 48³, growth
  cells^(4/3)) gives about 60 GB at 96 × 48 × 48, so one 80 GB GPU (estimate,
  not measured).

## V5 run: scripts, outputs, cost

Five stages, each a separate script that reads the files of the previous
stages from `FV5_OUT` (default `benchmarks/fv3d/results/v5_<backend>`), so a
stage can be rerun or moved to another job alone. Shared code and every knob:
`v5_common.jl` (header). `v5_all.jl` runs `FV5_STAGES` (default `abcde`) in one
process, for the smoke run.

```
# CPU smoke run, 24 × 12 × 12, all stages in one process
FV3D_BACKEND=cpu FV5_N=12 FV5_OUT=<dir> julia --project=<env> benchmarks/fv3d/v5_all.jl
# full size 96 × 48 × 48, one GPU, stage by stage (same FV5_OUT throughout)
FV3D_BACKEND=cuda FV5_N=48 FV5_OUT=<dir> julia --project=<env> benchmarks/fv3d/v5a_truth_lbm.jl
FV3D_BACKEND=cpu               FV5_OUT=<dir> julia --project=<env> benchmarks/fv3d/v5b_piv.jl
FV3D_BACKEND=cuda              FV5_OUT=<dir> julia --project=<env> benchmarks/fv3d/v5c_model_floor.jl
FV3D_BACKEND=cuda              FV5_OUT=<dir> julia --project=<env> benchmarks/fv3d/v5d_fit_carreau.jl
FV3D_BACKEND=cuda              FV5_OUT=<dir> julia --project=<env> benchmarks/fv3d/v5e_fit_mlp.jl
```

Stage b only needs the files of stage a; `FV3D_BACKEND=cpu` keeps it off the
GPU. FV5_N is read by stage a only; later stages read the case from
`truth/truth.toml`.

| stage | does | writes | GPU memory (est.) | wall time at full size (est.) |
|---|---|---|---|---|
| a `v5a_truth_lbm` | LBM truth, true Carreau law, to steady state (`FV5_LBM_TOL`) | `truth/{u,v,w,nu,is_solid}.bin`, `truth/truth.toml` | < 1 GB | 2–10 min (steps and MLUPS not measured) |
| b `v5b_piv` | 4 planes, 2 % noise, seed `FV5_SEED`; flow rate ± 1 % | `piv/*.bin`, `piv/piv.toml` | none | seconds |
| c `v5c_model_floor` | FV with the true law from rest (PTC) + 1 adjoint | `c_floor.toml`, `piv/*_twin_c*.bin` | ≤ 18 GB | 3–6 min |
| d `v5d_fit_carreau` | BFGS on (ln ν0, ln λ, n) + identifiability (6 solves) + gates with model error | `d_carreau_<data>.toml`, `d_carreau_progress_<data>.toml`, `figure/d_*` | ≤ 18 GB | 25–45 min (measured: 21 min, 24 iterations) |
| e `v5e_fit_mlp` | BFGS on all TinyMLP parameters | `e_mlp_<data>.toml`, `e_mlp_progress_<data>.toml`, `figure/e_*` | ≤ 18 GB | 60–120 min |

Every stage prints `SUMMARY` lines (objective, parameters, η(γ̇) relative
error over the 5–95 % shear-rate range seen by the fitted planes, wall time,
GPU memory, Newton and GMRES counts) and ends with `PASS`/`FAIL` (exit code)
for the run itself; stage d also prints `GATE` lines and a `CLAIM` line
(next section), which do not set the exit code. The optimiser writes its last
point after every iteration to `<stage>_progress_<data>.toml` (one file per
data variant, so a twin run no longer overwrites the noisy one; for `noisy`
the older name without the variant is read when the new one is missing):
`FV5_RESUME=1` restarts d or e from it (a 2 h fit can be split over two 1 h
jobs; the BFGS curvature is lost at the restart). `FV5_RESUME=1 FV5_MAXITER=0`
does no optimisation: one Newton solve from rest and one gradient at the last
point, then the identifiability solves, the gates and the figure files; the
counters of the earlier fit are kept under `previous_run` in
`d_carreau_<data>.toml`.

What the stages compare. Stage c gives the model-error floor: the objective
of the FV solution with the exact law against clean LBM planes (LBM and FV
discretisations differ at h = 1/48, no fit can remove that). The expected
objective from noise alone is about N_obs / 2. `FV5_DATA=twin` (or
`twin_noisy`) fits the FV planes at the true law written by stage c instead
of the LBM ones: an inverse crime that isolates the optimiser and the
gradients from the model error.

### V5 identifiability gate with model error (stage d)

First full-size run (96 × 48 × 48, GPU): FV with the true law against the LBM
truth differs by 0.8 % (field rel L2), 0.147 σ_u rms on the planes, J(true,
clean) = 78.3 for 7242 observations + flow rate; the Carreau fit converged
(2J/N = 0.977, η(γ̇) within 2.4 % max over γ̇ = 0.909–5.22, ν0 −17 %, λ −29 %,
n +1.8 %, cond(ŜᵀŜ) = 1.4e7) but the old gate failed (10/64 points): its band
was noise only, and the noise band of ln η at the best-constrained shear rate
(1.8e-3) is far below the model error.

Why the model error needs its own term. The LBM−FV discrepancy r_model is
systematic, not white: adding it as extra variance (σ_eff² = σ_u² +
σ_model², option a) widens the band by sqrt(1 + 0.147²) = 1.011, nothing,
while its projection on the three parameters does not average out over 7242
observations. Stage d therefore predicts the bias it causes:

- r_model = (FV planes at the true law − clean LBM planes)/σ_u, plus
  (Q_fv − Q_lbm)/σ_Q, built from files stage c already writes
  (`piv/<plane>_twin_c<k>.bin`, `c_floor.toml`); check printed:
  ½‖r_model‖² must equal stage c's J(true, clean).
- Expected bias of the fit: the Gauss–Newton shift δx_model = −(ŜᵀŜ)⁻¹ Ŝᵀ
  r_model (linearised, Ŝ taken at the fit); along the law, b(γ̇) = gᵀδx_model
  and σ(γ̇) = sqrt(gᵀ(ŜᵀŜ)⁻¹g), g = ∂ln η/∂x.
- Bands are simultaneous: k = sqrt(χ²₃(0.95)) = 2.80 (every point of the
  64-point grid jointly; the pointwise 1.96 count is also printed).

Printed (`SUMMARY`, `GATE`, `CLAIM`) and saved under `ident` in
`d_carreau_<data>.toml`, with err = ln η_fit − ln η_true on the observed range:

| gate | condition | what a PASS means |
|---|---|---|
| noise-only | \|err\| ≤ k σ | no model error visible (expected FAIL against LBM data) |
| bias-corrected | \|err − b\| ≤ k σ | the error left after the predicted model bias is noise |
| combined | \|err\| ≤ \|b\| + k σ | the law is inside its noise + model-error band |
| twin (`FV5_DATA=twin` / `twin_noisy`) | \|err\| ≤ k σ | no model error by construction; the noise-only gate must pass |

Also printed: the parameter-space χ² of x̂ − x_true without and with δx_model
(3 dof, threshold 7.81), the predicted δx_model next to the observed x̂ −
x_true, the correlation of (ln ν0, ln λ), the weakest eigenvector of ŜᵀŜ, and
per parameter the 95 % marginal half-width (factor for ν0 and λ, relative for
n); a parameter counts as identifiable when it is ≤ `FV5_PARAM_TOL` (10 %).
`CLAIM` = "η(γ̇) recovered within its noise + model-error uncertainty over the
observed range, ν0 and λ not individually identifiable": SUPPORTED when the
bias-corrected and combined gates pass (twin: the twin gate) and neither ν0
nor λ is identifiable. `FV5_DATA=clean` tests the linearised bias directly
(no noise: x̂ should land on x_true + δx_model).

Rerun stage d alone on the existing fit, then the twin (same FV5_OUT):

```
FV3D_BACKEND=cuda FV5_OUT=<dir> FV5_RESUME=1 FV5_MAXITER=0 julia --project=<env> benchmarks/fv3d/v5d_fit_carreau.jl
FV3D_BACKEND=cuda FV5_OUT=<dir> FV5_DATA=twin_noisy        julia --project=<env> benchmarks/fv3d/v5d_fit_carreau.jl
```

### V5 figure data (stages d and e) and the figure script

Written to `<FV5_OUT>/figure/` (paths in the indexes are relative to FV5_OUT;
`.bin` = Float64 little-endian, column-major, masked vectors set to 0):

- `d_<data>.toml`, `e_<data>.toml`: index (case scales ν0, λ, Ly, σ_u,
  U_bulk; γ̇ range; k; gates; per plane: normal, position, shape, vector
  centres, component names, file names).
- `d_law_<data>.csv`: on 121 log points from a decade below to a decade above
  the observed range, γ̇, λγ̇, η_true, η_fit, η/η0 of both, σ and b of ln η,
  the noise band, the total band (|b| + kσ) and the bias-corrected band of the
  true law (all over η0), and an `in_observed_range` flag.
- `d_gamma_hist_<data>.csv`: 48 log bins over the same range, γ̇ counts in
  the fitted-plane footprints (FV state at the fit and LBM truth) and in all
  fluid cells (FV).
- `e_law_<data>.csv`: γ̇, λγ̇, η_true, η_MLP, η_MLP at the start, η/η0.
- `<d|e>_<data>_<plane>_{model,data,residual,clean}_c<k>.bin`: the four
  planes (held-out one included) at the fit; masks stay in `piv/`.

`v5_figure.py` (numpy, matplotlib, Python ≥ 3.11) draws (a) η/η0 against λγ̇
— truth, Carreau fit, MLP fit, both 95 % bands on the observed range, the
footprint γ̇ histogram behind — and (b) one plane (`--plane`, default `z_mid`):
measured and fitted |u|/U_b and the residual of one component over σ_u:

```
python3 benchmarks/fv3d/v5_figure.py <FV5_OUT> --data noisy [--plane z_mid] [--fit d|e]
```

Cost reasoning (none of it measured). Memory: the P3 multigrid hierarchy and
the GMRES basis grow linearly with the number of cells (221k here, twice
48³); the i4 gate asks P3 to stay under 9 GB at 48³, so ≤ 18 GB is that
bound scaled, not a measurement (P3 has not run on a GPU yet). Run stage c
first: its `SUMMARY v5c cost` line gives the peak memory and the time of one
solve and one gradient, from which d and e follow. Time: one evaluation is a
warm Newton solve (3–6 iterations near the optimum) plus one adjoint
(re-assembly, refactorisation of the hierarchy, one transposed GMRES at
`rtol_transpose = 1e-10`); taken as 20–40 s, Carreau needs about 30
evaluations plus 6 identifiability solves, the MLP (H = 8: 26 parameters)
about 120–200. The LBM time depends on the number of steps to reach
`FV5_LBM_TOL = 1e-9`, at viscous time Ly²/ν_lat ≈ 11 500 steps for N = 48
(ν_lat = 0.2 at rest, lower where the fluid thins).

Deviations from the design above: the x-plane is stereo (the 2C x-plane
would only see v and w, the secondary flow); ν∞ = 0.1 ν0, λ = 5, mean
velocity 0.2 (see below); sheet σ = h/2
(`FV5_SHEET`); a fourth plane (z = 0.6 Lz, 2C) is held out; the linear
solver is GMRES + P3L (block upper triangular, LSC Schur, 4 multigrid
cycles per block; `FV5_PRECOND=p3` restores P3, with which stages a–e were
run; `FV5_LINSOLVE=direct` restores cuDSS, about 60 GB by the estimate above).
Stage d on the noisy twin, rerun under P3L on an H100 (2026-11-12): same fit
as under P3 (23 BFGS iterations, η max error 0.21 % over the observed range,
no unconverged solve), wall 402 s instead of 1201 s, GPU peak 6.94 GiB
instead of 6.31 GiB.

LBM stability (first smoke run, stage a diverged with ν∞ = 0). The GNF
kernel sets τ = 1/2 + 3 ν(γ̇) with no lower bound, so with ν∞ = 0 the
relaxation time falls towards 1/2 wherever the shear rate is high (the
corners of the voxel obstacle) and the TRT scheme diverges. The true law now
has a second Newtonian plateau ν∞ = 0.1 ν0 (shear-thinning polymer solutions
and blood have one; a = 2 keeps it regular at rest, so FV `eps = 0` still
holds for the Carreau stages, and the fits keep ν∞ and a at their true
values). The lattice viscosity at rest follows from the floor:
ν_lat = (τ_min − 1/2)/(3 ν∞/ν0) = 0.2 for τ_min = 0.56 (`FV5_TAU_MIN`), so
τ ∈ [0.56, 1.1] everywhere. A larger ν_lat means a larger dt and a larger
lattice velocity, so the mean velocity drops to 0.2 and λ rises to 5 to keep
λ U/Ly = 1 as before (Re_bulk ≈ 2 for the Newtonian flow, higher once the
fluid thins). Stage a stops with a clear message when the LBM velocity is
not finite, and the LBM driver stops at the first non-finite convergence
check instead of running to `max_steps`.
