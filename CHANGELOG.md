# Changelog

All notable changes to Kraken.jl will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Opt-in steady stop rule for `run_cylinder_libb_2d`,
  `run_viscoelastic_fvfd_poiseuille_3d` and the 2D log-FV coupled viscoelastic
  drivers (including `run_viscoelastic_logfv_cylinder_coupled_2d`).
  `stop = :cd_change` samples the drag coefficient every `check_every` steps
  (default: one flow-through) and stops when two consecutive samples differ by
  at most `tol` in relative value (defaults `1e-6` for the LI-BB cylinder,
  `2e-4` for the log-FV drivers); `stop = :change` of the 3D channel stops when
  the relative change of the velocity and of the polymer stress between two
  checks are both at most `tol` (default `1e-12`). The results gain
  `converged`, `stop_step` and the sample history (`cd_history`,
  `change_history`), and the log-FV drivers also `stop_reason` and `Cd_last`.
  With the default `stop = :none` the drivers run exactly `max_steps` steps and
  return the same values as before. In the log-FV drivers a `step_callback`
  that returns exactly `true` now ends the run (#123).
- Linear-solve seam, opt-in additions (the existing `lin_factorize` /
  `lin_solve!(cache, b)` calls are unchanged): `lin_refactorize!(cache[, A_new])`
  re-runs the numeric factorization with new values on the same sparsity pattern,
  reusing the symbolic analysis (CPU CHOLMOD/UMFPACK and cuDSS); in-place
  `lin_solve!(x, cache, b)`; multiple right-hand sides `lin_solve!(cache, B)`;
  transposed solves `lin_solve!(...; transpose = true)` and
  `lin_supports_transpose(cache)`. On cuDSS the in-place solve and the device-side
  refactorization allocate nothing on the Kraken side (preallocated staging
  buffers). The cuDSS transposed solve requires CUDSS.jl >= 1.0: older versions
  silently ignore the transpose mode, so Kraken raises an explicit error there
  instead of returning `A \ b`. A different sparsity pattern is refused (#104).
- Newton solver for the steady colocated FV residual (step C2 of the FV Newton
  roadmap), `solve_incns_fv_newton(; problem, to, linsolve)` and
  `solve(params, IncNS(:fv_newton))`. The Jacobian is the exact derivative of
  the residual kernel, assembled by coloured forward-mode duals (pattern built
  from the stencil, colouring by SparseMatrixColorings, new dependency) through
  the same KernelAbstractions kernel on CPU and GPU, written directly into the
  CSC (UMFPACK) or CSR (cuDSS) matrix values. Linear solves go through the
  linear-solve seam: one `lin_factorize` (symbolic analysis), then
  `lin_refactorize!` and in-place `lin_solve!` at every iteration. Armijo line
  search; optional pseudo-transient continuation (`dtau0`).
  `fv_newton_finalize!` leaves the factors of the unshifted Jacobian at the
  solution, for a later adjoint solve. Checked on CPU: the assembled Jacobian
  equals the dense Jacobian on four configurations (cavity, masked channel,
  periodic channel, doubly periodic); quadratic convergence to ‖R‖₂ ≈ 1e-16
  (cavity Re = 100, 32², 5 iterations; Re = 1000, 64², 11 iterations);
  agreement with the SIMPLE cavity within 5e-3 and with Ghia et al. (1982) at
  128², Re = 100 (max centreline error 0.0072 of the lid speed). On an A100
  (Aqua job 27632281) the GPU Jacobian equals the CPU one (1e-12 bound), Newton
  on cuDSS follows the CPU residual history at 64² (one symbolic analysis, then
  refactorizations; direct and transposed solves of the finalized factors), and
  assembly, refactorization, solve and whole Newton iterations allocate 0 device
  bytes. Lid-driven cavity from rest on an A100 (Aqua job 27631719): 512²,
  Re = 100 in 5 iterations and 12.6 s, Re = 1000 in 12 iterations and 15.3 s
  (symbolic analysis 10 s once, refactorization 0.43 s); 1024², Re = 100 in 5
  iterations and 56 s, Re = 1000 in 11 iterations and 69 s (analysis 44 s once,
  refactorization 2.3 s, 22 GiB with the factors resident).
- `benchmarks/validation/gnf_calibration/`: a calibration script that fits the
  three Carreau parameters (`a = 2` fixed) to synthetic steady velocity fields
  on the periodic obstacle channel using `gnf_steady_gradient` and a small BFGS
  minimiser (five fixed starts, ten fixed noise seeds at 0.1, 1 and 5 %), and a
  cost bench that times one forward solve, one adjoint gradient and the `2p`
  forward solves of a finite-difference gradient on four grids.
  `run_gnf_adjoint_gpu.jl` takes the test point, grid and finite-difference
  step sweep as options. CPU smoke runs pass; GPU results are not yet
  produced (#124).
- Generalised-Newtonian viscosity in the steady FV residual and its Newton
  solver (step C3 of the FV Newton roadmap):
  `fv_steady_problem(...; rheology = law, gnf_eps = ε)` with the existing
  `PowerLaw`, `CarreauYasuda`, `Cross`, `Bingham`, `HerschelBulkley` and
  `Newtonian` laws (kinematic units, the same law objects and the same
  `gnf_viscosity` evaluation as the LBM GNF drivers). The face viscosity is the
  law at the face strain rate `γ̇_ε = √(2 S:S + ε²)` (full strain-rate tensor;
  `ε > 0` is a change of model, required except for Carreau–Yasuda with
  `a ≥ 2`). The Jacobian includes `∂ν/∂U` because the residual kernel computes
  ν itself; its sparsity pattern is derived from the exact stencil of the face
  strain rate (3 × 3 cell block, 27 entries per interior row instead of 23).
  `fv_gnf_parameters` / `fv_with_gnf_parameters` expose the law parameters as
  a vector that may hold duals (for the coming adjoint). A `PowerLaw` with finite
  clamp bounds (zero derivative where active, Newton may stall) triggers one `@warn`. Checked on CPU: the
  assembled Jacobian equals the dense ForwardDiff Jacobian bit for bit for five
  laws on four configurations; power-law Poiseuille (n = 0.5 and 1.5) within
  1.3 % of the regularised 1D reference at 16 cells across and second order
  under refinement, tending to the analytical power-law profile as ε → 0;
  quadratic Newton convergence from rest (5–7 iterations); n = 1 gives the
  Newtonian residual bit for bit; on the Carreau–Yasuda obstacle channel of
  the LBM GNF adjoint tests, FV and LBM velocities differ by 16.7 %, 5.0 % and
  1.6 % (relative L2) at 4, 8 and 16 cells per unit length. The PowerLaw
  `nu_min`/`nu_max` clamp is applied as written and stalls Newton where
  active. FV viscosity-type promotion now uses the law's numeric field types
  (excluding `thermal`), keeping the residual accumulators concrete for scalar
  and ForwardDiff dual states. CPU inference guards cover all six law types,
  mixed law/regularisation precisions and dual law parameters. On an A100
  (Aqua job 27701869) the GPU tests pass (CPU/GPU parity of the GNF residual,
  Jacobian and dual-law residual; Newton on cuDSS against the CPU history;
  0 device bytes in the hot loop). Power-law cavity n = 0.5 from rest: 256² in
  13 iterations and 3.3 s, 512² in 32 iterations and 20.3 s, 1024² in 22
  iterations and 83 s (first factorization 31.6 s, refactorization 2.4 s,
  23 GiB with the factors resident).
- Steady discrete adjoint on the FV Newton solver (step C4 of the FV Newton
  roadmap), `src/methods/inc_ns/fv_adjoint.jl`. `fv_steady_gradient(solver, U,
  θ, objective; params)` re-assembles the unshifted Jacobian at the converged
  state, refactorizes it (no new symbolic analysis), solves Jᵀλ = ∂J/∂U with
  the Newton factors (UMFPACK on CPU, cuDSS with CUDSS.jl ≥ 1.0 on GPU) and
  returns dJ/dθ = ∂J/∂θ − λᵀ∂R/∂θ, with ∂R/∂θ from parameter duals through
  the same residual kernel. Parameters: uniform viscosity, viscosity field
  (coloured, one cotangent per cell), side velocity (e.g. lid speed or inlet
  amplitude), uniform source, or any `FVProblemMap(f, n)` building the problem
  from θ. Objectives: state misfit (`FVStateObs`, also from a `LineProfile`),
  linear functionals (`fv_flow_rate`, `fv_pressure_drop`) and wall force
  (`FVWallForce`, drag/lift on the solid mask or a side). Also
  `fv_steady_adjoint` (any ∂J/∂U), `fv_steady_tangent` (forward sensitivity)
  and `fv_adjoint_workspace` (preallocated buffers). New capability value
  `AssembledJacobian`; `IncNS(:fv_newton)` now also declares `SteadyAdjoint`.
  Checked on CPU against dense forward-mode Jacobians (parameter cotangent to
  2e-15), central finite differences (worst relative gap 9e-9 over the tested
  directions) and the forward tangent (6e-16); dQ/dν = −Q/ν in Stokes flow.
  Generalised-Newtonian problems: the parameter maps keep the viscosity law,
  the law parameters are adjoint parameters through
  `FVProblemMap(fv_with_gnf_parameters, n)`, and `FVWallForce` uses the law at
  the wall strain rate, as the residual does (adjoint vs central differences
  within 3e-9 for Carreau–Yasuda and power law). On an A100 (Aqua job
  27713496) the GPU tests pass, including CPU/GPU parity of a GNF
  law-parameter gradient; lid-driven cavity at 1024² (3.1 M unknowns, Newton
  62 s): one gradient costs 0.06–1.0 s once the factors exist (lid drag or
  velocity profile; uniform ν, viscosity field or lid speed), with 0 device
  bytes allocated and no new symbolic analysis.
  Wall-closure study in `benchmarks/fv_c4/`: drag, pressure drop and their
  gradients converge at order ≈ 1.4 whether ν is constant or variable;
  `FVWallForce(...; pressure = :linear)` is the recommended drag objective
  (the default stays `:cell`, the residual's own wall pressure).
- Immersed curved walls in the steady FV residual, Newton and adjoint (step C5
  of the FV Newton roadmap), `src/methods/inc_ns/fv_immersed*.jl`. Analytic cut
  geometry (`fv_immersed_circle`, `fv_immersed_annulus`,
  `fv_immersed_boundary`) with a ghost-cell wall closure (`FVImmersedClosure`,
  linear or quadratic) that enters the same residual kernel, so the coloured
  Jacobian, cuDSS factors and discrete adjoint apply unchanged. Mass
  compatibility `FVImmersedMass`: M3 (default) adds one unknown per connected
  body / fluid region pair, so closed domains conserve mass to roundoff and the
  solution does not depend on the pressure pin; M1 stays as an explicit
  comparison mode. New objectives `FVWallTorque` and immersed `FVWallForce`;
  `FVShapeMap` gives shape derivatives (radius, centre) at fixed cell topology.
  Checked on CPU: Jacobian vs dense duals 8.9e-16; force and torque gradients
  vs finite differences within 6.6e-8 (relative); Taylor–Couette and a
  non-rigid manufactured solution converge at velocity order ≈ 2.0 and 2.6.
  On an A100 (Aqua job 27741796) the C3, C4 and C5 GPU tests pass; immersed
  Stokes flow at 512² (0.79 M unknowns): M3 costs the same as M1 (first
  factorization 9.4 s, refactorization 0.36 s, 4.3 GB of factors), one force
  gradient 0.04–0.06 s once the factors exist, 0 device bytes allocated in
  the Newton and gradient loops.
  Known limits, documented in `benchmarks/fv_c5/README.md`: the pressure drop
  across a cylinder in a channel is not monotone in h on the tested grids, and
  the surface-integral torque derivative converges more slowly than the
  residual-flux one.
- Scalar laws and a learned viscosity in the steady FV Newton solver (branch
  `feat/learned-law`): `AbstractScalarLaw` (`law_value`, hand-written
  `law_pullback!`, allocation-free) with `ClassicalLaw` for the existing GNF
  laws, `TinyMLP` and `LearnedViscosity` (shear stress strictly increasing in
  shear rate by default, optional prescribed field inputs such as temperature
  or concentration); `FVLawViscosity` evaluates any such law at every FV
  viscosity point (interior, wall, inlet, immersed) with prescribed cell
  fields; `FVLawParameters` gives `dJ/dtheta` through `fv_steady_gradient`,
  by default with one face-cotangent contraction instead of parameter duals
  (CPU 64x64, 248 parameters: 1.46 s to 67 ms; both paths agree to 1e-13,
  central differences to 4e-7). `ClassicalLaw` reproduces `FVGNFViscosity`.
  H100 run recorded in `benchmarks/fv_law/`.
- Steady thermal coupling on the colocated FV Newton solver (branch
  `feat/fv-thermal`): `FVEnergy` adds a temperature equation (linear-upwind
  convection on the Rhie–Chow flux, diffusion, viscous heating `2 sigma nu S:S`,
  Dirichlet or adiabatic walls, adiabatic voxel solids) stored as
  `theta = T - T_ref`; temperature-dependent viscosity through
  `FVStateTemperature` and `E5ThermalLaw`; imposed mean velocity
  (`u_mean_target`, `fv_flow_force`); `fv_thermal_continuation` in the heating
  coefficient with fold diagnostics; a round-off floor stopping gate for the
  energy rows (`C = 64`, result field `gate`); the discrete adjoint through the
  temperature (`fv_mean_temperature`, `fv_temperature_l2`,
  `fv_flow_force_objective`); the FV-vs-LBM comparison driver
  `benchmarks/fv_thermal/e5_compare.jl` with acceptance thresholds. Validation:
  independent Nahme channel reference (second order), and on the E5 six-cylinder
  cell at 600x200 (H100, 2026-11-08) mean temperature rise +1.4 %, flow rate
  +1.4 %, `dm/dtheta` norm difference 2.4 % (worst component 4.5 %, 8 of 8
  signs), adjoint vs finite differences 2.7e-6; at 300x100 +3.1 %, +3.3 %,
  6.0 % (worst 10.7 %). Two grids only, Float64 only, no transient. See
  `docs/src/users/fv-thermal.md`.
- Plane PIV observation operator for 3D fields: `PIVPlane`, `piv_operator`,
  `piv_apply`, its exact adjoint `piv_apply_adjoint` and `piv_misfit` (in-plane
  or stereo samples, Gaussian light-sheet thickness, interrogation window),
  tested in `test/analytical/plane_piv_3d.jl`.
- 3D steady colocated FV solver `FVGrid3D`, `FVSteadyProblem3D` through
  `fv_steady_problem(Nx, Ny, Nz; ...)`, the 3D residual (same KernelAbstractions
  kernel on CPU and GPU, generic element type, six sides that are walls with a
  prescribed velocity or periodic, voxel solids, imposed mean flow rate, GNF and
  learned laws), its coloured Jacobian pattern, and the 2D Newton solver,
  assembly and discrete adjoint extended to 3D (maps `FVUniformViscosity`,
  `FVViscosityField`, `FVSideVelocity`, `FVUniformSource`, `FVLawParameters`;
  objectives flow rate, Fx, wall force). `FVPIVObjective` and
  `fv_objective_gradient!` give a composite plane-PIV misfit with weighted terms
  and its gradient on the device. The pseudo-transient shift is now `V/dtau`
  with `V` the cell measure (`h^3` in 3D, `h^2` in 2D, unchanged). Validation:
  z-uniform 3D residual equal to the 2D one, coloured equal to dense Jacobian,
  MMS, adjoint against centred finite differences
  (`test/analytical/incns_fv3d_test.jl`, `incns_fv3d_adjoint_test.jl`); GPU
  scripts in `benchmarks/fv3d/`.
- Krylov branch of the linear-solve seam, `KrylovBackendTag` (restarted GMRES on
  an assembled `KrylovCSRMatrix`, forward and transposed solves, CPU and CUDA
  through `ext/KrakeNNCUDSSExt.jl`), selected by `fv_newton_solver(p; linsolve =
  KrylovBackendTag(...))` with the Newton and adjoint API unchanged.
  Preconditioners: ILU(0) of the whole Jacobian (`precond = :ilu0`); block
  SIMPLE, ILU(0) of the velocity block and LU of the SIMPLE Schur
  approximation (`:simple`); block SIMPLE with one aggregation-multigrid cycle
  per block (`:mg`); a block-triangular preconditioner with an LSC
  Schur-complement approximation ("block LSC": velocity block by aggregation
  multigrid, pressure through the least-squares commutator approximation of
  the Schur complement; `precond = :mg, block = :upper, schur_approx = :lsc`,
  also `:simplec`, `:pcd`, `schur_q`, `schur_shift`, `ablock` options); and a
  coupled (monolithic) multigrid with a Vanka smoother ("Vanka multigrid":
  one multigrid on the whole velocity–pressure system, whose smoother solves
  cell by cell the small local system of the cell's velocities and its
  pressure; `precond = :vanka`, `vanka_omega`, `vanka_prolong`). The default
  `precond = :auto` resolves to the measured setting of each case: Vanka
  multigrid with constant prolongation (`vanka_prolong = :constant`) and
  damping `vanka_omega = 0.5` on 3D FV Jacobians without energy equation;
  block LSC with 4 multigrid cycles per block on 3D ones with energy equation
  (the setting of the I3 runs: V4 duct to N = 48, V0' energy to 64³, peak
  4.7 GB); ILU(0) elsewhere (2D problems, matrices without block layout or
  grid geometry), so 2D behaviour is unchanged. An explicit
  `precond = :vanka` keeps trilinear prolongation by default. Diagnostics
  `lin_krylov_log`, `lin_krylov_pivot`, `lin_krylov_blocks`,
  `lin_krylov_stationary`. Memory is linear in the cells instead of the
  cells^(4/3) of the direct factors. Block LSC with 4 multigrid cycles per
  block takes 29 / 28 / 34 forward GMRES iterations at 16³ / 24³ / 32³ on the
  flow-rate duct (H100). Vanka multigrid with `vanka_omega = 0.5` and
  constant prolongation passes the D2 gate of `i4_krylov_measure.jl` at 32³,
  48³ and 64³ in both benchmark cases (with 0.7 it fails on the body-force
  case). At N = 32 with first-order upwind convection the Vanka multigrid
  default is reliable up to Re ≈ 250 (208 GMRES iterations); beyond, Newton
  or GMRES fail (measured at Re ≈ 415); block LSC at Re ≈ 250 was
  not measured to the end. Tested against the direct branch in
  `test/analytical/incns_fv3d_krylov_test.jl`.
- 3D energy equation `FVEnergy3D` (`fv_steady_problem(Nx, Ny, Nz; energy)`):
  temperature as the fifth cell unknown, viscous heating, temperature-dependent
  laws, heating continuation, round-off floor gate and the law adjoint through
  the temperature, as in 2D. GPU (H100): V0' energy MMS second order for
  conduction, V4 heated duct θ relative L2 1.09e-3 at N = 48 with the Krylov
  branch (peak 4.2 GB), V0' energy to 64³ (peak 4.7 GB). Tested in
  `test/analytical/incns_fv3d_thermal_test.jl`.
- V5 benchmark, law recovery from plane PIV in 3D (`benchmarks/fv3d/v5*`): LBM
  truth on a 96 × 48 × 48 duct with an off-centre obstacle and a Carreau–Yasuda
  law, three noisy PIV planes plus flow rate, Carreau and TinyMLP fits through
  the 3D adjoint, identifiability gate with a predicted model-error bias.
  Against the LBM data the Carreau fit recovers η(γ̇) within 2.4 % over the
  observed shear-rate range. On the noisy twin (stage d, H100) the same fit
  takes 402 s with block LSC instead of 1201 s with block SIMPLE and
  multigrid, and 143 s with Vanka multigrid (peak 5.07 GiB instead of
  6.94 GiB); Vanka multigrid is the V5 default (`FV5_PRECOND=p4`, the value
  name is kept for compatibility).

- Network and parametric viscosity laws in the 2D and 3D generalised-Newtonian
  LBM step, with a steady adjoint in 3D. `DenseNet` is an isbits feedforward
  network law with per-layer widths and activations (it reproduces `TinyMLP` on
  equivalent weights, with a hand-written pullback checked against ForwardDiff)
  and works in `FVLawViscosity` as well. `load_onnx_law` (package extension
  `KrakeNNONNXExt`, weak dependency ProtoBuf) reads a linear chain of
  `Gemm`/`MatMul`+`Add` layers with `Tanh`/`Relu`/`Sigmoid`/`Softplus`
  activations, input normalisation and an `exp` output into a `DenseNet`.
  `LBMLawViscosity` wraps any scalar law (`ClassicalLaw`, `DenseNet`) as a GNF
  closure in the D2Q9 and D3Q19 steps (3D invariant `gamma_dot^2/2`); with a
  `ClassicalLaw` Carreau-Yasuda it is bitwise identical to the direct closure.
  `gnf_law_gradient` (2D) and `gnf3d_law_gradient` / `gnf3d_steady_law_gradient`
  (3D) contract the steady adjoint sensitivity through the law pullback. The 3D
  adjoint is built on the new `AbstractGNFStepOperator`, shared with the 2D
  operator by the bordered-GMRES and K-step solvers, with the matrix-free
  `GNF3DStepOperator` (JVP/VJP) and `gnf3d_adjoint_solve`. Tests check the
  adjoint against finite differences and a `DenseNet` plane channel against the
  2D FV solution at second order; GPU parity and cost scripts are in
  `benchmarks/lbm3d_law` and `test/analytical` (#138).

### Changed
- The package is renamed `KrakeNN` (module, `using KrakeNN`, extension modules `KrakeNN*Ext`); the UUID is unchanged, so existing environments only need `Kraken` replaced by `KrakeNN` in their code. The repository moved to `github.com/lauguimel/KrakeNN.jl` and the documentation to `lauguimel.github.io/KrakeNN.jl`.
- `capabilities(::IncNS)` is now driver-dependent: only `IncNS(:fv_newton)`
  declares `SteadyResidual`; the other drivers declare `ForwardSolve` only, and
  `residual(prob, IncNS(:simple), U)` raises an `ArgumentError`.
- Compat widened to `CUDA = "5, 6"` and `CUDSS = "0.3, 0.4, 0.5, 0.6, 1"`
  (CUDSS.jl 1.0 requires CUDA.jl 6). Existing CUDA.jl 5 environments keep
  resolving; a fresh environment now resolves to CUDA.jl 6. LinearSolve.jl does
  not yet allow CUDSS.jl 1.0, so an environment with both stays on CUDSS.jl <= 0.8
  (no cuDSS transposed solve) (#104).
- The EHD direct Poisson GPU solve writes into a preallocated device buffer
  (`EhdPoissonSetupGPU.sol`) instead of allocating a vector each step (#104).
- The GNF adjoint documentation now states that the GPU check ran the
  bordered-GMRES solver (`:gmres`), not the `:kstep` default of
  `gnf_steady_gradient` (#124).
- New `NeqExtrapolationPressure(ρ_out)` east outlet for
  `apply_bc_rebuild_2d!`: after the bulk step, the selected rows of column `Nx`
  are set by non-equilibrium extrapolation from the completed column `Nx-1`,
  with imposed density and no further collision. End rows next to `HalfwayBB`
  or `InterfaceBC` are included, using the same row selection as Zou-He
  x-faces; corners next to a south/north Zou-He face are excluded.
  It is now used by `run_cylinder_libb_2d`, the log-FV viscoelastic
  drivers sharing `default_step_bcspec_2d`, the 2D STL `wall = libb` path and
  the Schäfer–Turek 2D-2 benchmark. With the corrected weights, the Zou-He
  pressure outlet of this path develops a sign-flipping mode that the old
  weights damped (#118, #126).
- `ZouHePressure` is unchanged and its docstring describes that mode.
  Explicit callers, west/south/north pressure faces of the 2D STL path, the
  direct Zou-He drivers, the Gmsh physical-normal route and all 3D routes
  keep their current behaviour. The extrapolation outlet copies all nine
  populations from the neighbour after its wall treatment and collision,
  including wall-adjacent end rows. Place the new outlet several diameters
  downstream of obstacles; backflow through it is not supported (#126).

### Fixed
- `run_gnf_duct_obstacle_3d` stops with `converged = false` at the first
  convergence check where the velocity change is not finite, instead of
  running to `max_steps` after a divergence.
- Parabolic inlets now deliver the nominal flow rate between half-way walls. The
  profile of `run_cylinder_libb_2d`, `parabolic_face_profile_2d`,
  `run_sphere_libb_3d`, `run_conformation_sphere_libb_3d` and the AD cylinder and
  viscoelastic models is sampled with an exact discrete mean, and the Zou-He inlet
  rebuild (2D, 3D and the AD copies) now imposes the profile on every row,
  including the rows next to the walls; in 3D, `apply_bc_rebuild_3d!` takes an
  opt-in `halfway_walls` argument that covers edges and corners. The delivered
  flow-rate error of `run_cylinder_libb_2d` at Ny = 24/40/80 drops from
  -0.89/-0.32/-0.081 % to +0.032/+0.012/+0.0029 % (parabolic) and from
  -7.9/-4.7/-2.4 % to +0.43/+0.26/+0.13 % (uniform); the 3D doubly parabolic inlet
  at N = 16 from -12.45 % to +0.12 %. Drag coefficients of cylinder and sphere
  runs change accordingly (R = 3, Ny = 40: parabolic Cd 24.056 to 24.083, uniform
  16.08 to 17.02); published values that used the old inlet are marked in the
  documentation as measured before v0.7 (#111, #115, #119).
- Adiabatic walls of the thermal solver reflect the thermal populations
  specularly (zero normal heat flux, tangential flux kept) instead of bouncing the
  diagonals back, in the fused 2D kernels, the GNF natural-convection transport,
  the CPU step used for gradients, the 3D thermal kernels and the Rayleigh-Benard
  horizontal walls. Pure conduction is now exact on every row and the Ra = 1e3
  Nusselt numbers converge at second order. Thermal results change; GPU numbers
  in the documentation measured before this fix are marked as such (#113, #119).
- The hot-wall Nusselt number of `run_natural_convection_2d`,
  `run_natural_convection_gnf_2d` and the 3D cavity is averaged over all wall rows
  instead of rows 2 to N-1. At N = 64, Ra = 1e3 the 2D cavity gives Nu = 1.1175
  (De Vahl Davis hot-wall Nu₀ = 1.117) (#119).
- `WLFCoupling` now evaluates the shift in base 10,
  `a_T = 10^(-C1 dT/(C2 + dT))`, so the published Williams-Landel-Ferry
  constants (the `.krk` defaults `C1 = 8.86`, `C2 = 101.6`) are used as written;
  the previous natural-exponential form made the default shift too weak, and
  results that use `WLFCoupling` change. `ArrheniusCoupling` requires absolute
  temperatures: `run_natural_convection_gnf_2d` and the `.krk` rheology loader
  now throw an `ArgumentError` for `T <= 0` or a non-finite `T_ref` or `E_a` (the
  default walls `T_hot = 1`, `T_cold = 0` are refused with an Arrhenius
  coupling). An isothermal `.krk` run that sets `E_a`, `C1`, `C2` or `T_ref` now
  warns that they are ignored. The FENE-P docstring states which Peterlin
  function each kernel set uses (#108, #120).
- Viscoelastic cylinder runs with a small solvent fraction (`beta <= 0.1`) no
  longer diverge within a few hundred steps. The x second derivative of the
  both-sides-diffusion correction is now zero at the first or last column of an
  open x end; the one-sided stencil used there had the wrong sign on the
  period-2 mode of the outlet corner cells and anti-damped it. Wall-x and
  periodic-x stencils are unchanged and results at `beta = 0.59` are unchanged
  to within 1e-5. Runs at `beta <= 0.1` now complete (cylinder R = 50, Wi = 1,
  300 000 steps, A100), but no grid study exists for them and their `Cd` values
  are not validated (#121).
- The 3D FVFD Oldroyd-B channel (`run_viscoelastic_fvfd_poiseuille_3d`) no
  longer develops a growing wall-normal checkerboard mode that made `tau_yy`
  grow from zero until the run crashed after a few thousand steps.
  `fvfd_velocity_gradient_3d!` has a new keyword `wall_normal_noslip` (default
  `false`, unchanged); the channel driver sets it, which imposes `u_y = 0` at
  the half-way wall in the wall-row `du_y/dy`. In a 4000-step test run
  `max|tau_yy|/max|tau_xy|` falls from 1.4e2 to 1.3e-12 and the velocity errors
  are unchanged (#116, #122).
- The D2Q9 normal-stress weights in the regularized TRT rebuild at 2D Zou-He
  nodes and in the generated local collision brick, including the AD copies,
  are corrected. The rebuild now preserves mass and momentum and reproduces
  the prescribed stress moments; it previously created `(1 − s₊)·S/18` of
  mass per rebuilt node and step (#118, #126).

- The immersed-boundary adjoint gates of the steady FV solver
  (`incns_fv_immersed_m3_adjoint`, `_objectives`, `_shape`) fail intermittently
  on Julia 1.12 with code coverage enabled: the first adjoint call returns NaN or
  an inaccurate value. Only in that configuration a failing gate is now recorded
  as `@test_broken` (with a warning) instead of an error; on other Julia versions
  and without coverage the gates are unchanged plain `@test`s. Test-only change,
  the cause is still open (#136, #137).

## [0.6.0] — 2026-11-04

This release corrects a forcing defect that changed viscoelastic results, and repairs
the repository front (quickstart, capability table, benchmark pages, test wiring). See
the erratum below before reusing any viscoelastic number from 0.2.0 to 0.5.1.

### Erratum — viscoelastic results before 0.6.0 (#67, #70)
The TRT collision with a per-cell Guo force injected `g·F` per step instead of `F`,
with `g = 1 + (s₋ − s₊)/2` (`g = 0.52` at `ν = 0.05`, `0.44` at `ν = 0.04`, `0.82` at
`ν = 0.10`, `1.02` at `ν = 0.15`; source: #67 and PR #70). The defect was present from
v0.2.0 to v0.5.1 on the 2D log-FV viscoelastic drivers, the 3D VE sphere and coupled
planar-extension drivers, and the AD viscoelastic shape-sensitivity mirror. The BGK
Guo kernels, the imposed-velocity planar extension and the Poiseuille drivers were not
affected. Measured effect of the fix: Oldroyd-B confined cylinder vs RheoTool,
`Cd` 129.92 / 118.68 / 119.24 → 129.95 / 118.92 / 119.49 at `Wi = 0.1 / 0.5 / 1`
(`benchmarks/results/rheotool_compare/viscoelastic/kraken_cd_vs_wi.csv`, pre-fix values
in its header), so the error against RheoTool goes from −0.40 / −0.86 / −0.96 % to
−0.37 / −0.66 / −0.75 % (`error_norms.csv`): small shifts, still under 1 %. The `β = 0.3`
case is no longer claimed within 1 %. The 3D VE sphere canary `Cd` at `Wi = 0.01` moves
16.212 → 17.331 (+6.9 %), and the AD shape gradient `d(Cd_polymer)/dR` moves −1.6 %
(#70); sensitivities with respect to `ν_s` and `Fx` move more (`ν_s` changes sign), see
#67. No sphere number and none of those sensitivities was published. Results of your
own that went through these paths with a nonzero force and `ν ≠ √3/12` should be rerun.

### Added
- Steady discrete adjoint of the generalised-Newtonian (GNF) lattice Boltzmann step (#100):
  `gnf_steady_gradient` returns the objective `J = ½ Σ (value − target)²` of the
  converged steady velocity (`GNFVelocityObs`) and its gradient with respect to the
  four Carreau–Yasuda viscosity-law parameters `(ln η∞*, ln k*, n, a)`;
  `gnf_steady_adjoint` accepts any cotangent on the lattice velocity. The fixed point
  comes from the new `run_gnf_periodic_steady` driver. The adjoint is the transposed
  linearisation of Kraken's own step at the fixed point (local blocks by ForwardDiff
  on the host, hand-written KernelAbstractions products, mass mode removed by
  bordering); it does not use Enzyme. Default solver `:kstep` (GMRES on the K-step
  system, K = 1001); `:gmres` and `:fixed_point` are selectable. Scope: 2D D2Q9,
  periodic in x, body-force driven, Carreau–Yasuda law only. Verified on CPU and on one A100 (Float64), on 32×16 and 16×8 grids (GPU: 32×16
  only). On CPU the gradient agrees with central finite differences of J to 7.1e-08
  or better in absolute value (the gate is far looser), the Jacobian-vector product
  agrees with central differences of the real step, and the transposed product
  agrees with Enzyme to 3.6e-16 relative (the Enzyme comparison runs in its own CI
  step, `test/ad/gnf_adjoint_enzyme_test.jl`). On the A100 the device products match
  the CPU ones to 1.2e-16 relative and the gradient matches finite differences to
  7.1e-08 or better; the Float32 operator is informational only. The in-suite
  CPU/GPU parity test is skipped without CUDA. See
  `users/benchmarks/gnf-adjoint-gradient.md`.
- `Backend <cpu|cuda|metal>` directive in `.krk` files, resolved by the simulation runner (#93).
- `examples/quickstart.jl`, the exact script the README shows, run on CPU by a
  `quickstart` CI job (#95).
- Opt-in test tier `KRAKEN_TEST_EXPERIMENTAL=1` for 23 test files that were never wired
  into the suite; the scalar advection order test now runs in the default suite (#94).
- Electroconvection checkpoint and resume (#60): snapshot and restore of the completed-cycle
  populations, force history and convergence scalars, with controlled updates of the
  temperature parameter and rejection of incompatible or corrupt state. The opt-in moment
  injector (above) is extended to the transient electroconvection driver and to the
  checkpoint format, which moves to schema 2 (#79); a schema-1 file is migrated on read.
- `test/analytical/ve3d_ext_velocity_readback.jl` and `ve3d_ext_zouhe_zplanes.jl` record two
  defects of the coupled 3D extensional driver as `@test_broken` (#71; issues #68, #69, open):
  it overwrites the collision velocity with a read-back of the post-collision populations,
  and its Zou-He face rebuilds skip the first and last z planes although z is periodic.
  Treat the coupled 3D extensional mode as not validated until they are fixed.
- RheoTool reference data for the viscoelastic cavity (#57) and a regression test for the
  field-stopping repair of #23 (#58).
- Experimental hydrostatic `injector_closure=:static_moment` Julia keyword and
  `.krk` `Physics injector_closure = static_moment` selector (#73, #78).
  The opt-in normal-moment reconstruction preserves the outgoing charge aggregate
  and enforces the flat injector charge. It is restricted to the periodic-x,
  zero-fluid-velocity hydrostatic driver with regularized charge, direct Poisson,
  Float64 and `delta_U=1`. The default `:nee`, collector and field operators are
  unchanged; the option is rejected by the coupled EC driver and `ECState`.
  Opt-in analytic diagnostics use wall-node coordinates (`y_mapping=:wall_nodes`),
  not the legacy interior half-link mapping; their errors are not directly
  comparable across closures. This does not establish complete EHD validation.
- 3D generalised-Newtonian (GNF) flow on D3Q19 with TRT collision, linearly
  interpolated bounce-back walls and Guo forcing (#101): `run_gnf_channel_3d` and the fused
  step `fused_trt_gnf_libb_guo_field_step_3d!`. Supports `PowerLaw`, `CarreauYasuda`
  and `Cross` (and Newtonian), isothermal only, Julia API only (no `.krk` keyword yet).
  On planar channels against an independent analytical reference the observed order is
  1.93 to 2.00 on CPU Float64 (N = 16, 32, 64) and 1.95 to 2.001 on an A100 (CUDA
  Float64, N = 32 to 256, archived in `benchmarks/validation/gnf3d/results/`; the code
  that ran is kept reachable by the tag `archive/gnf3d-gpu-sweep`). Float32 was only
  run through on the GPU, not validated. See
  `docs/src/users/benchmarks/gnf3d-channel-convergence.md`.
- `run_natural_convection_gnf_2d`, `gnf_reference_scales` and `cavity_powerlaw` (#105):
  differentially heated D2Q9 cavity for generalised Newtonian fluids, driving
  `collide_rheology_thermal_2d!` with the thermal double-distribution populations.
  Ra and Pr are defined with the reference kinematic viscosity `nu0`; with
  `cavity_powerlaw` they are the power-law definitions of Turan et al. (2011).
  A `.krk` `natural_convection` case with a non-Newtonian `Rheology` block now
  routes to this driver. Thermal walls half-way, as in the Newtonian driver
  (H = N). 2D only, Float64. n = 1 reproduces the Newtonian driver to round-off
  (Nu = 1.1185 at N = 64, Ra = 1e3; de Vahl Davis hot-wall value 1.117, +0.13 %). Against Turan et al.
  (Pr = 10, figure readings) on CPU: Nu = 5.393 vs 5.40 at Ra = 1e4, n = 0.6
  (N = 32/48 extrapolated). On one A100 (CUDA Float64, N = 64/128/256): Ra = 1e4,
  n = 0.6 and 1.0 within 0.6 % at every N; Ra = 1e3, n = 1.8 within 0.13 %;
  Ra = 1e3, n = 1.0 gives 1.12 vs 1.14 (-1.7 % to -1.9 %, paper value read to ±0.02);
  Ra = 1e3, n = 0.6 gives 1.99 vs 2.07 (-3.8 %), grid-converged and the same on CPU,
  gap documented and not explained; Ra = 1e4, n = 1.8 did not converge and is not a
  validated point. GPU and CPU agree at N = 64 to 2e-5 in Nu. Results in
  `benchmarks/validation/natconv_gnf/results/`; the code that ran is kept reachable
  by the tag `archive/gnf-thermal-turan-gpu`. No 3D, no grid refinement. See
  `docs/src/users/benchmarks/gnf-natural-convection.md`.
- Steady 2D colocated finite-volume residual of the incompressible Navier–Stokes
  (or Stokes) equations, `fv_residual!(R, U, prob)`, written as one
  KernelAbstractions kernel that runs unchanged on CPU and GPU and is generic in
  the element type (Float64, Float32, ForwardDiff duals in the state or in the
  parameter arrays). It is step C1 of the FV Newton roadmap: the residual that
  the coming Newton solve (Jacobian by coloured duals) and discrete adjoint will
  differentiate. Discretisation: first-order upwind convection on Rhie–Chow face
  fluxes with `d_f = h²/(4ν_f + |ū_f|h)`, full viscous stress tensor with a cell
  viscosity field, body/mass sources, wall / inlet / outlet / periodic sides,
  solid mask, one pressure pin per closed fluid pocket (`fv_pressure_pins!`).
  Problems are built with `fv_steady_problem` and moved to a device with
  `fv_adapt`. `SteadyResidual` is declared by `IncNS(:fv_newton)`
  (`residual(prob::FVSteadyProblem2D, IncNS(:fv_newton), U)`); this is the residual of the
  coupled FV discretisation, not of the SIMPLE drivers. Checked on CPU against the
  J0 prototype (identical residuals on 11 configurations), exact discrete
  solutions (R = 0), manufactured solutions with variable viscosity (velocity
  order 2 for Stokes; the pressure is first order in the wall cells), duals
  against finite differences, and the SIMPLE cavity (centreline difference
  1.2e-3 at 32², Re = 100). On an A100 (Aqua job 27617685) the GPU residual
  matches the CPU one (1.8e-17 relative at 1024² in Float64, within the 1e-14
  test bound in Dual{4} on three cases), allocates 0
  bytes per call, and takes 0.43 ms at 1024² in Float64 (2.4 ms in Dual{4}).
  No Newton solver yet: nothing user-facing solves with it.

### Changed
- Docs: grid refinement (patch and conservative AMR) is now documented as under development and is no longer listed as a supported feature (#103).
- CI builds the documentation on pull requests into `dev/platform` with a pinned Vitepress npm tree (#81); the contributor guide states the full-suite gate as CI actually runs it (#59).
- Capability table (`docs/src/capabilities.md`) (#96) now lists MRT, TRT, the `bin/kraken` CLI,
  Saramito, thermal VE/GNF, 3D outlet, refined 3D thermal and the Metal/CUDA backends.
  The H100 throughput figures, which no script or log backs, are removed. The GPU
  certification page said 0.68 of the bandwidth ceiling (PASS) using the 304 bytes of
  D3Q19; a D2Q9 Float64 update moves 144 bytes, so 3461 MLUPS on an A100-40GB is 0.32 of
  the ceiling, below the 0.5 gate. The viscoelastic cylinder page now states what the
  archived rheoFoam logs show (serial runs, end time 6 or 10) instead of 2-core MPI to
  t = 20.
- Documentation statements corrected against the code: the VE cylinder wall is a linearly
  interpolated (Bouzidi) bounce-back with the analytic cut-link distance, only the channel
  walls are half-way; the VE cylinder page no longer says the residual gap to RheoTool is
  discretisation (the grid extrapolation of the R = 30 and 50 values sits +0.7 % or -0.2 %
  from the single RheoTool value, for a first- or second-order error); the FENE-P Oldroyd-B
  limit is checked at `L² = 1e8` to 1e-3, not byte-identical; coupled 3D planar extension is
  not validated (#68, #69 open); the buoyancy force has no density factor; GNF thermal
  coupling has a driver; the De Vahl Davis comparison uses the hot-wall values (Nu₀ = 1.117,
  2.238, 4.509; the cavity average is 1.118, 2.243, 4.519) and the old "1.4 %" is +0.13 %;
  in `.krk` files natural-convection and Rayleigh-Bénard cases do not read the wall `T`
  values and do not switch buoyancy on from them; the GNF adjoint Enzyme check has its own
  CI step; the GNF thermal page lists the non-zero fractions of nodes at a viscosity bound
  in the A100 sweep; the GPU certification README uses 144 bytes per D2Q9 update; the
  Guo (2002) Boussinesq, Schäfer-Turek and Krüger entries of `docs/refs.bib`.
- Electroconvection `.krk` cases (#37) now apply paired `Boundary west wall` / `Boundary east wall` as stationary no-slip sidewalls instead of ignoring them; omitted sides retain free slip. Explicit non-lateral `Boundary` declarations are rejected because electrode/plate conditions are built into this driver.
- Documentation numbers re-measured after the TRT Guo forcing fix (#67), same settings
  as before (Aqua A100, CUDA Float64; post-fix source `edbc20916`, jobs 26763246/47/48,
  26766115, 26766116/17; pre-fix control `249146d4d`, job 26763249, reproduces the
  published 119.2410). Oldroyd-B confined cylinder vs RheoTool (`R = 50`, `β = 0.59`):
  `Cd` 129.92 / 118.68 / 119.24 → 129.95 / 118.92 / 119.49 and error −0.40 / −0.86 /
  −0.96 % → −0.37 / −0.66 / −0.75 % at `Wi = 0.1 / 0.5 / 1`; still under 1 %. `Wi = 1`
  convergence −6.2 / −1.9 / −0.96 % → −6.1 / −1.7 / −0.75 % at `R = 10 / 30 / 50`.
  `β = 0.3` is no longer described as within 1 %: −1.33 % at `Wi = 0.5` (−1.77 % before
  the fix) and +1.24 % at `Wi = 1`, where the RheoTool reference is not steady at
  `t = 20`. High-`Wi` `Cd` at `Wi = 3`: 141.8 / 116.3 → 131.6 / 118.4 (`R = 50 / 30`),
  still NaN-free to `Wi = 10`. Wake-`N1` ratios 0.770 / 0.704 / 0.890 / 1.437 → 0.771 /
  0.705 / 0.894 / 1.448. AD viscoelastic shape sensitivity: adjoint vs finite difference
  0.42 % → 0.63 %, the `fwd_tol = 1e-11` counter-example 22.6 % → 53.3 %, and the "~20×"
  cancellation ratio, which the runs do not measure, is removed. Pages:
  `viscoelastic-cylinder` (benchmark and tutorial), `validation-matrix`,
  `ad-shape-sensitivity-viscoelastic`, `docs/agent/ad-ve-implication.md`, and the
  `benchmarks/results/rheotool_compare/viscoelastic/` README, `kraken_cd_vs_wi.csv` and
  `error_norms.csv`, and `comparison.png` (and its copy `viscoelastic-cylinder.png`),
  regenerated from the post-fix `kraken_cd_vs_wi.csv`. `m8_refs/*.csv` keep the
  pre-fix values as the dated record of the M8 study.

### Fixed
- **Thermal Dirichlet walls are half-way, like the no-slip walls** (#97, PR #102). The
  fixed-temperature kernels (`apply_fixed_temp_*_2d!`, `apply_fixed_temp_*_3d!`, the
  fused natural-convection step) put the wall on the first and last nodes, so the
  thermal cavity was `N − 1` wide while the flow cavity was `N` wide. They now apply
  the half-way anti-bounce-back `g_q = −g_q + 2 w_q T_w` to the bounced slot, once per
  step after streaming (calling it twice undoes it). The drivers start from the
  half-way linear profile `T_i = T_hot − ΔT (i − ½)/N`, and `Nu` uses the second-order
  wall gradient `(−8 T_w + 9 T_1 − T_2)/3` with `H = N` (AD mirror and AD Nusselt QoI
  updated alike). CPU Float64, `N = 64`: `Ra = 10³` Nu 1.1423 → 1.1185 (+2.26 % →
  +0.13 % vs de Vahl Davis 1.117), `Ra = 10⁴` 2.3585 → 2.2535 (+5.39 % → +0.69 %);
  pure conduction Nu 1.072 → 1.0003 at `N = 16`. With periodic side walls the linear
  profile and `Nu = 1` now hold to round-off; in a closed cavity the adiabatic
  bounce-back walls leave a second-order departure (`Nu − 1 = 3.1·10⁻⁴` at `N = 16`).
  Refined thermal drivers re-apply the kernel after restriction only on the faces a
  patch touches; their pure-conduction Nu moves from 1.29–1.31 to 1.26–1.30, and that
  separate refined-wall defect stays open. The published GPU benchmark rows were
  measured with the former walls and are flagged as such.
- **Adjacent thermal Dirichlet faces flip each corner slot once** (#97). With two
  Dirichlet faces sharing a corner (an edge in 3D), which the refined `.krk` drivers
  allow, the per-face kernels flipped the diagonal slot that crosses both walls once per
  face. New exported `apply_fixed_temp_2d!` and `apply_fixed_temp_3d!` apply several
  faces in one call and flip it once, with the mean of the two wall temperatures
  (x-faces apply it, then y-faces; z-faces never). The refined 2D/3D drivers and the
  patch builders (`build_patch_thermal_bcs_3d`) use them; one face alone is
  bit-identical to before. A conduction square with hot west and south and cold east
  and north is now symmetric about the diagonal to 4·10⁻¹⁶ (8·10⁻² before).
- **AD Nusselt QoI uses the actual wall distance and cavity width** (#97). `nu_pure`
  reads the hot-wall fraction `δ = q_hot` and `L = Nx − 1 + q_hot + q_cold` from the
  cut-link fractions and uses the one-sided gradient for a wall at distance `δ`;
  `nu_driver` takes `q_hot` and `L`. At the shape-sensitivity case (`N = 16`,
  `Ra = 10³`, `L = 16.2`) `dNu/dL` goes from −0.067 (width held at `N`) to
  +1.44·10⁻³, adjoint and finite difference within 1.0·10⁻³. The cut-link thermal wall
  is exact only at `q = 0.5`: in pure conduction `Nu − 1` falls by 8·10⁻³ per unit of
  cold-wall fraction (`q` from 0.3 to 0.9), while moving that wall by a whole cell at
  fixed `q` changes `Nu` by +1.74·10⁻² at `Ra = 10³`. That cut-link error stays open.
- `strain_rate_magnitude_3d` used a D3Q19 velocity ordering that is not the one of the
  lattice tables, so the off-diagonal sums `Π_xy`, `Π_xz`, `Π_yz` had wrong signs and a
  pure shear in any plane returned a zero strain rate. The sums now follow the lattice;
  the final contraction moved to the new `strain_rate_from_pi_3d`. Any 3D result computed
  with the old function (exported, so user code can call it) was wrong when
  off-diagonal stress was present. Inside this repository, no `src/` driver, test or
  benchmark called it before the 3D GNF path, so no previously published Kraken result
  changes; it affects only external callers.
- **TRT Guo forcing injects exactly `F` per step** (#67). The TRT collision
  bricks with a per-cell Guo force (`CollideTRTDirectGuoField`,
  `CollideTRTDirectGuoField_3D`) scaled the whole Guo source by `1 − s₊/2`. The odd
  part of the source, which carries the momentum, relaxes at `s₋`, so each step added
  `g·F` with `g = 1 + (s₋ − s₊)/2` instead of `F`. At the default `Λ = 3/16`, `g = 1`
  only at `ν = √3/12 ≈ 0.144`; `g = 0.52` at `ν = 0.05`, `0.44` at `ν = 0.04`, `0.82`
  at `ν = 0.10`, `1.02` at `ν = 0.15`. The source is now split: even part
  `w[9(c·u)(c·F) − 3u·F]` at `1 − s₊/2`, odd part `3w c·F` at `1 − s₋/2` (TRT form of
  the MRT forcing, Silva 2020, doi:10.1016/j.compfluid.2020.104537). Affected paths:
  `fused_trt_libb_v2_guo_field_step!` (2D log-FV coupled step behind the cylinder,
  BFS, contraction and square-channel coupled drivers, and the passive BFS precursor;
  all three `wall_bc` variants), `fused_trt_libb_v2_guo_field_step_3d!` (VE sphere and
  coupled planar-extension drivers), and the AD mirror `ad_ve_coupled_step!`. The BGK
  Guo kernels were correct and are unchanged. New analytic tests in
  `test/analytical/trt_guo_field_momentum.jl` check the even and odd parts separately
  for five `(s₊, s₋)` pairs in 2D and 3D, Float64 and Float32, the SRT limit against the
  BGK Guo kernels, and the AD mirror against the 2D brick. The coupled planar-extension
  test moves from 0.83 % to 0.18 % on `C_xx` against the nominal strain rate (the
  coupled flow now runs 0.72 % above it, down from 1.41 %). Published numbers
  affected: see the documentation entry under Changed above.
- **Corrected the 0.5.0 notes** (#52). The four log-conformation closures run on the
  FVFD path only; the LBM-CDE drivers (sphere, Couette, Poiseuille) accept Oldroyd-B
  only, in direct conformation form. Couette and LBM-CDE Poiseuille are not reachable
  from `.krk`, the planar extension is reachable in imposed-velocity mode only, and the
  Oldroyd-B sphere driver is not validated against a viscoelastic reference. Several
  figures were misstated: the FVFD near-wall and velocity errors per resolution, the
  planar-extension `C_yy` error and the cause of its 0.39 % `C_xx` gap, the number of
  `@test_broken` (13, not 22) and of files over 700 lines (13, not 8), and `L² → ∞`
  "byte-for-byte". The LBM-CDE Poiseuille error is now listed as a known issue of this
  release rather than as a changed result. `docs/src/capabilities.md` §9 now lists the
  3D viscoelastic models instead of reporting rheology as 2D-only, and
  `docs/src/users/benchmarks/ve3d-poiseuille-convergence.md` states the velocity error
  per resolution.
- **3D FVFD advection is second order next to a periodic z face** (#54). The
  MUSCL-Superbee operator fell back to first-order upwind within two cells of the z
  faces even when z is periodic, i.e. everywhere in a thin quasi-2D box. The 0.5.0
  planar-extension canary moves from 0.39 % to 0.0023 % on `C_xx` (the 0.5.0 notes
  attributed the 0.39 % to slow relaxation; it was this fallback), and the FENE-P
  canary now matches its transcendental fixed point to 1e-5. The coupled
  planar-extension test moves from 0.06 % to 0.83 % on `C_xx` against the nominal
  strain rate (gate 1 %): the coupled flow runs 1.4 % above the nominal rate, which
  the old fallback happened to offset; against the measured rate it is -0.59 %.
  *These coupled values were measured before the TRT Guo forcing fix above, which
  brings them to 0.18 %, 0.72 % and -0.54 %.*
- `compute_polymeric_force_3d!` takes `wall_x` / `wall_z`, giving x and z walls the
  second-order one-sided difference the y walls use (a clamped stencil returned half
  the derivative there). Faces that are neither periodic nor walls keep the clamp.
- `logfv_max_grad_norm_3d` reduces on the arrays' device instead of copying nine
  fields to the host every step.
- **The adaptive DDF potential solve also stops on the electric field** (#23). With
  `phi_scheme = :lbm` and no `phi_substeps`, the solve stopped once `phi` changed by at
  most `phi_tol` over one iteration, but `E` is rebuilt from the first moment of the
  populations, which relaxes on its own: a cold start could return an unconverged field
  (`E*` errors of 1.5e-5 to 3.9e-3 on the ES-002 capacitors at the electroconvection
  check cadence, 0.25 to 0.5 at the hydrostatic one; relative error 0.29 on the first
  potential solve of the default hydrostatic run). A check now also requires the field
  change to be at most the new keyword `field_tol` (default `1e-4`, independent of
  `phi_tol`) of `run_electroconvection_2d`, `init_state(ECState; ...)` and
  `run_ehd_hydrostatic_2d`: the change of `E` since the previous check, relative to
  `max(max|E|, E_ref)`, with `E_ref = |phi_bottom − phi_top|/(Ny − 1)` the applied field
  (`1/(Ny − 1)` in all public drivers). The applied field is a minimum scale: it applies
  while `E` is below it everywhere, as at the first checks of a cold start, where `E`
  starts at 0. A `floatmin` guard avoids a division by zero when the plates are at the
  same potential and the field is zero. Measured at every accepting check: `max|E| / E_ref`
  is 1.484 to 1.495 in the default hydrostatic run and in 60x96 electroconvection over
  20 cycles (Float64 and Float32), 0.99999999994 to 1.0043 on the ES-002 capacitors; at
  each of these acceptances the change relative to `max|E|` alone also met `field_tol`.
  It is a Julia keyword only, not a `.krk` one. Checks run every 8 iterations in the
  electroconvection solver and after every iteration in the hydrostatic driver
  (`src/drivers/ehd_phi_ddf.jl`). `field_tol = Inf` restores the previous rule bit for
  bit and is meant for non-regression comparisons only.
  Cost on CPU, Float64: the default 60x96 electroconvection run takes 80, 64, 24, 40, 24,
  32, 24, 24, 16, 16, 8, 8 potential iterations over its first 12 cycles (8 each before;
  0.028 s instead of 0.010 s for those cycles), then 8 per cycle (measured up to cycle
  18). On small grids the extra iterations persist: 112 per cycle at cycles 7 to 11 on
  8x12, 48 to 144 at the last cycle of 23- to 30-cycle runs on 16x24 (8 before). The
  default hydrostatic run takes 1.00 s instead of 0.88 s, with the same `err_E` (0.76 %).
  Consequences: results on this path change (the worked example of
  `docs/src/users/simulation-state-checkpoints.md` was regenerated); a negative or `NaN`
  `field_tol`, and on the adaptive path a `phi_max_iter` that is not an integral value of
  at least 1, are now an `ArgumentError` at entry (`phi_max_iter < 1` used to skip the
  solve silently); when `phi` converged but `E` did not, the non-convergence error also
  gives the last relative field change. `field_tol` is part of the EC checkpoint
  identity (`src/drivers/ehd_ec_checkpoint.jl`): a restore with a different or missing
  `field_tol` is refused, and `Inf` round-trips.
- *Known limits of the #23 repair.* `field_tol` bounds a change between checks, not the
  error of `E`. Once the slow diffusive mode of the pseudo-time iteration dominates, the
  relative error is about `κ * field_tol`, `κ ≈ (Ny - 1)^2 / (m * gamma * π^2)` with `m`
  the check cadence: about 5, 22 and 380 on 8x12, 16x24 and 60x96 at `gamma = 0.3`. The
  ES-002 public-driver case keeps an `E*` error of 3.5e-5 at default settings and stays
  `@test_broken` (#63); with `field_tol = 1e-8` it meets the 1e-6 gate. In Float32,
  on grids with `Ny - 1 ≳ 100`, the iteration freezes at bit level before `E` converges
  and the check accepts whatever `field_tol` is: use `phi_scheme = :direct` or Float64
  for an accurate `E`.

## [0.5.1] — 2026-10-23

### Changed
- Jiachen Zhao (Queensland University of Technology) joins the authors, second position,
  in `CITATION.cff`, `Project.toml` and the BibTeX entries of the README and the
  documentation home page. `CITATION.cff` now carries the ORCIDs of J. Zhao and
  E. Sauret. The `version` field of those two BibTeX entries now reads 0.5.1 (it read
  0.2.0 in the README and 0.4.0 on the home page).

### Fixed
- **CUDSS weak-dependency UUID corrected** (#61). Since v0.3.0 the `KrakenCUDSSExt`
  extension could not be activated by `using CUDA, CUDSS, Kraken`, because `Project.toml`
  declared a UUID that differs from the registered one in one character. Whether the
  July 2026 GPU validation exercised the extension is not recorded. The corrected
  extension was run on an H100 on the fix branch (Julia 1.11.9, CUDA 5.11.3,
  CUDSS 0.6.5, Float64): it loads through plain `using`, and 41/41 GPU assertions pass
  with no CPU fallback. New offline test `test/platform/weakdeps_metadata_test.jl`
  checks every `[weakdeps]` UUID against the registry; the GPU test now asserts the
  extension is active when CUDSS loads.
- The 0.3.0 entry "GPU-validated (parity ≤1.6e-12)" for the cuDSS solve and the matching
  line of `docs/agent/solve-linear-implication.md` now carry a correction note (#61): the
  loading path of that July run is not recorded.

## [0.5.0] — 2026-09-24

### Added
- **3D viscoelastic flow, four constitutive models** (`src/kernels/logconformation_fv_3d.jl`,
  `src/kernels/logconformation_lbm_3d.jl`, `src/fvfd/operators_3d*.jl`,
  `src/drivers/viscoelastic_*_3d.jl`, `src/rheology/linalg_3d.jl`): log-conformation
  constitutive step in 3D with Oldroyd-B, FENE-P, Giesekus and PTT (linear and
  exponential), on the FVFD finite-volume grid (`run_viscoelastic_fvfd_poiseuille_3d`,
  `run_viscoelastic_fvfd_extensional_3d`). The LBM-CDE drivers (Oldroyd-B sphere, planar
  Couette, planar Poiseuille) evolve the conformation directly and accept Oldroyd-B only.
  Reachable from `.krk`: the sphere, FVFD Poiseuille and FVFD planar extension (imposed
  velocity only; the coupled mode is Julia-only); Couette and LBM-CDE Poiseuille are
  Julia-only. The sphere driver is not validated against a viscoelastic reference: its
  tests check the Newtonian limit against Kraken's Newtonian sphere driver and a NaN-free
  run at `Wi = 0.01`. Each closure matches its closed-form steady simple-shear fixed point
  (Giesekus, PTT residual ≤ 1e-6; FENE-P and the Oldroyd-B limit ≤ 1e-3); setting `α = 0`
  or `ε = 0` recovers the Oldroyd-B trajectory byte for byte, and FENE-P approaches it as
  `L²` grows (1.3e-8 on `C_xx` at `L² = 1e8` in planar extension).
  *Corrected 2026-09-24 (#52). The entry as tagged said the four closures ran "on either
  the LBM lattice or the FVFD finite-volume grid", that all five drivers were "reachable
  from `.krk`", and that `L² → ∞` recovered Oldroyd-B "byte-for-byte"; none was true.*
- **FVFD transport cures the near-wall conformation error of the LBM-CDE path.** On
  identical `N_y = 32` viscoelastic Poiseuille, the near-wall `C_xy` error is 3.9e-15
  against an anti-tautological reference built from the measured shear (at most 1.9e-7
  absolute, 3.9e-7 relative, over `N_y = 32–128`), where the diffusive LBM-CDE path
  returns 25.9 % error and a peak velocity ratio of ≈ 1.13. The FVFD peak velocity is
  within 0.086 % of the analytic parabola at `N_y = 32` and 0.0095 % at `N_y = 128` (H100,
  CUDA Float64). Both paths ship in this release; see Known issues for LBM-CDE.
  *Corrected 2026-09-24 (#52). The entry as tagged said "machine-exact (≤ 1.9e-7, i.e.
  ≤ 2e-5 %)" and "to 0.01 %" at `N_y = 32`: 1.9e-7 is the absolute `N_y = 128` value, and
  the velocity is within 0.01 % only at `N_y = 128`.*
- **Resumable simulation state and HDF5 checkpoints** (#26) (`src/platform/state.jl`,
  `src/io/checkpoint_hdf5.jl`, `src/drivers/ehd_ec_state.jl`): the `init_state` /
  `advance!` / `solution` contract, checkpoints written as HDF5, and the electroconvection
  driver running on the contract bit-identically to the frozen legacy driver. An
  interrupted run resumes instead of restarting. New direct dependency: `HDF5`.
  Documented in `docs/src/users/simulation-state-checkpoints.md` and `docs/src/api/platform.md`.
- **RheoTool cross-validation** of FVFD-3D Poiseuille, Oldroyd-B planar extension and
  FENE-P planar extension. Runs performed in a local `openfoam9-rheotool:v1.2` container;
  OpenFOAM case templates under `bench/rheotool/`, raw output and comparison tables under
  `benchmarks/results/rheotool_compare/`. On Poiseuille, Kraken's relative L2 error against
  the analytic profile is 1.2e-3 on velocity and 4.1e-9 on `N1`, where RheoTool gives 1.3e-3
  and 4.6e-3. On Oldroyd-B planar extension RheoTool reaches the analytic fixed point to
  machine precision and Kraken's 1000-step canary sits 0.39 % below on `C_xx` and 0.0015 %
  off on `C_yy`; these gaps come from a first-order advection fallback in the periodic z
  direction (#54), not from a slow relaxation as the tagged entry said. The 10.9 % FENE-P
  `C_xx` gap is a closure-variant difference (Peterlin argument `tr C` vs `tr A`), not a
  defect. These are benchmarks, not automated gates: no test reads the reference data.
- **Steady-state time estimate and viscoelastic parameter-stability check**
  (`src/units/steady_state.jl`).
- **Finite-difference cross-check of the viscoelastic polymer-drag adjoint** (#41)
  (`test/ad/test_ad_ve_fd_check.jl`): four design directions against central finite
  differences of the reconverged primal, with a three-point Richardson error estimate. The
  19 pre-existing adjoint assertions all pass under a known miscompile; this one does not.
- **References page** in the documentation (#46). `DocumenterCitations` was loaded and
  `docs/refs.bib` held 39 entries, but no page carried a `@bibliography` block, so all 37
  cited keys rendered as unlinked text.

### Changed
- The test suite is a two-tier merge gate (#24, #27, #30, #31): a fast pull-request tier, a
  full tier on the integration branch and nightly, superseded runs cancelled, and any
  single test file runnable on its own locally or on CI.
- The four Enzyme-driven test files run with production bounds checking
  (`--check-bounds=auto`) (#39, #41, #43). Under the `Pkg.test` default Enzyme's reverse
  mode miscompiles them; reported upstream as EnzymeAD/Enzyme.jl#3614.
- Known failures are carried as `@test_broken` rather than kept out of the suite (#29);
  the full CI run of this release carries 13 (12 in the LBM tier, 1 in the IncNS tier;
  the tagged entry said 22).
- The contribution model replaces the `src/` edit ban with directory ownership and review
  (#25); `AGENTS.md` and `.github/CODEOWNERS` carry the rules.
- `CITATION.cff`: G. Maitrejean's affiliation is now Grenoble INP, Université Grenoble Alpes.

### Fixed
- Type inference no longer expands the Enzyme shape-sensitivity thunks when
  `run_simulation` is called without automatic differentiation (#39, #40), which crashed
  the full test tier on Julia 1.12 / x86_64 Linux.
- Two IncNS tests asserted round-off residues against macOS literals; they now assert
  bounds (#36).
- `docs/node_modules` is no longer tracked (#45). The committed copy held the macOS
  `rollup` binary, so the documentation build failed on Linux with
  `Cannot find module @rollup/rollup-linux-x64-gnu`.
- The citation block on the documentation home page declared version 0.2.0 with no DOI and
  announced a DOI that had already been minted (#48).

### Known issues
- Multiblock ghost exchange reads the wrong source block at `n_ghost = 2`, and a topology
  guard is missing (#28).
- `zou_he_pressure_3d_kernel!` still omits the wall-parallel diagonal populations that the
  velocity BC includes (#20).
- DDF potential stopping can accept an unconverged electric-field moment (#23).
- Enzyme reverse mode miscompiles a loop branching on a `Const` Bool mask at neighbour
  indices under `--check-bounds=yes`: segfault on x86_64 Linux, wrong gradient on aarch64
  macOS (#41, EnzymeAD/Enzyme.jl#3614). Production mode (`--check-bounds=auto`) is exact
  and is what CI and the documented workflow use.
- 13 files exceed the 700-LOC budget (#15; the tagged entry said 8).
- The LBM-CDE 3D Poiseuille driver (`run_conformation_poiseuille_libb_3d`) over-smooths the
  conformation near walls: 25.9 % near-wall `C_xy` error and a peak velocity ratio ≈ 1.13
  at `N_y = 32` (4 `@test_broken`). Use `run_viscoelastic_fvfd_poiseuille_3d`.

## [0.4.0] — 2026-09-14

### Added
- **Electrohydrodynamics / electroconvection** (`src/kernels/ehd_*`, `src/drivers/`):
  charge-transport and electric-potential LBM sub-solvers, Coulomb body force,
  free-slip sidewalls, MRT collision, GPU-ready kernels, a `.krk` user surface
  (`benchmarks/krk/ehd/`) and a critical-threshold sweep script
  (`benchmarks/ehd/tc_sweep.jl`). Hydrostatic base state matches the analytical
  charge-density and field profiles below 1%; onset threshold `T_c ~ 166.5`
  against 163.5 from Luo, Wu, Yi & Tan, Phys. Rev. E 93, 023309 (2016), a +1.8%
  deviation — mesh-convergence study for `T_c` still pending, so read it as
  consistent-with rather than converged.
- **West pressure boundary in 2D** (`apply_zou_he_pressure_west_2d!`), completing
  the east/west pair. Validated against analytic Poiseuille: max relative error
  2.03e-3 at Ny=16 and 5.15e-4 at Ny=32 (2nd order).
- **Thermal conduction driver** accepts `nu`, `alpha` and an `orientation`
  (`:vertical` default, `:horizontal` for west/east).
- Regression tests exercising the public `.krk` runner rather than the drivers
  directly: `test/analytical/H2-004-route.jl`, `test/analytical/TH-002-route.jl`.
- `AGENTS.md` and `.github/CODEOWNERS`: contributor working rules, ownership per
  directory, bug-report-as-failing-test convention.

### Fixed
- **MRT Guo forcing injected only half the requested body force.** Any MRT run
  with a body force before this release was forced at half the intended value.
- `_D2Q9_CX` was defined twice in the module with different types, which made the
  package fail to load on Julia 1.11 — the declared minimum (#17).
- The generic `.krk` runner silently ignored west pressure boundaries; unsupported
  faces now raise an explicit error instead of a no-op (#18).
- The conduction `.krk` fallback dropped `nu`, `alpha` and the thermal face
  settings, silently solving a different problem. `examples/heat_conduction.krk`
  is itself a west/east case and had been running south/north defaults at nu=0.05
  instead of its documented nu=0.1, alpha=0.01 (#19).
- 3D cavity test ran at omega = 1.8868, above the measured BGK stability ceiling
  of ~1.81, and produced NaN at step 194. Reconfigured at the same Re = 32 with
  tightened assertions.

### Changed
- **Breaking:** driver spec `OldroydB` renamed `OldroydBSpec` (name clash with the
  rheology model of the same name). Scripts naming the old type must be updated.
- `main` reconciled with the development line; the two now carry the same content.

### Known issues
- 3 failures in multi-block ghost exchange (`test_multiblock_exchange.jl`).
- `zou_he_pressure_3d_kernel!` still omits the wall-parallel diagonal populations
  that the velocity BC now includes (#20).

## [0.3.0] — 2026-07-22

### Added
- **Platform contract layer** (`src/platform/`): `AbstractProblem/Method/Solution/
  Observable/Closure`, `Capability` enum, and the verbs `solve`/`sample`/`observe`/
  `predict` — LBM wrapped bit-for-bit; `IncNS` runs under the same contract.
- **Residual/adjoint seam**: `residual` + `adjoint_vjp` over four parameter natures
  (geometry, scalar ν, ν(x) field, thermal), delegating to the validated AD paths.
- **Calibration stack**: `ParameterSpace`, `loss`, `fit` (projected Barzilai–Borwein
  + Armijo, zero new deps) and `fit(method=:lbfgs)` with Tikhonov regularisation via
  the `KrakenOptimExt` weak dependency. Twin experiments: scalar ν recovered to 4% in
  5 iterations; sine ν(y) field to 7.5% rel-L2.
- **Steady incompressible Navier–Stokes co-solver** (FVFD): SIMPLE/SIMPLEC with
  2nd-order scalar and momentum convection (implicit upwind + deferred correction),
  matrix-free multigrid Poisson, steady scalar transport with dedicated BCs.
- **GPU ablation ladder** (lid-driven cavity): A100 32.9×, H100 44.1× (C3, F64),
  RTX A6000 36.3× (C4 mixed precision — the winning rung on consumer silicon);
  1024² converges in 100.4k iterations, Ghia error 2.05%.
- **LinearSolve.jl front-end + cuDSS direct solve** behind `[weakdeps]`
  (`solve_poisson_direct`): assembled sparse alternative to the MG path on the same
  discretization; CPU MMS order 2.00, GPU-validated (parity ≤1.6e-12).
  *Correction (0.5.1, see #61):* the GPU claim is not supported as written. The declared
  CUDSS UUID was wrong, so `using CUDSS` could not activate `KrakenCUDSSExt`, and the
  loading path of the July run (extension or manual include) was not recorded. The
  extension path was first validated with the UUID fix of 0.5.1 (Aqua job 26559033,
  reported 2026-10-10: H100, 41/41 GPU assertions, no CPU fallback).

### Changed
- Documentation reframed method-agnostic (architecture page, maturity table) — the
  DocumenterVitepress site from the v0.2 line is the canonical doc toolchain.

### Fixed
- CUDSS extension trigger: CUDA removed from `[weakdeps]` (it is a strong dep;
  extensions use parent strong deps since Julia 1.11). Known Julia 1.12 caveat
  documented: sibling declared weakdeps must be co-installed for the CUDSS
  extension to precompile.

## v0.2.1

Documentation patch — no functional or source changes.

- Landing page: reworked navigation and the "What Kraken can do" capability grid.
- Examples: per-example `.krk` download dropdowns; boundary-condition schematics
  redrawn with a shared toolkit.
- DocumenterVitepress theme polish (custom CSS); `.krk` syntax highlighting;
  velocity-field lead plots; axisymmetric reference; theory-page cleanups.

## v0.2.0

Multiphysics release.

- **Units module** (LU ↔ physical): explicit conversion between lattice
  units and physical SI quantities for setup and post-processing.
- **Geometry / STL immersed boundary**: arbitrary solid geometries via STL
  import with cut-link (interpolated bounce-back) boundary treatment.
- **Viscoelastic Oldroyd-B cylinder**: validated to within <1% of RheoTool
  reference drag.
- **Thermal natural convection**: validated against the de Vahl Davis
  differentially heated cavity benchmark.
- **GPU certification**: reference benchmarks certified on GPU backends.

## [0.1.0] — 2026-04-14

### Added (since audit, 2026-04-13/14)
- **Unified .krk dispatch for refinement** (2D + 3D, isothermal + thermal)
  via `_run_refined` / `_run_refined_3d` — no dedicated driver needed.
- **Fine-grid sanity checks**: `τ_T_fine` (thermal refinement) and `N/Re`
  on refined patches, with `[2D]/[3D]` tag in the parameter summary.
- **Thermal BL resolution check**: warns when `N_eff < 3·Ra^(1/4)`,
  accounting for refinement ratios near thermal walls.
- **Capabilities matrix page** (`docs/src/capabilities.md`) listing every
  feature with status, links to theory/examples/API.
- **CLI wrapper** (`bin/krk`), VSCode `.krk` syntax highlighting, ASCII
  kwargs aliases (nu/rho/sigma/tau/…).

### Fixed (since audit)
- **Metal GPU refinement crash**: `trunc(Int,...)` replaced by
  `unsafe_trunc(Int,...)` in all 2D refinement, thermal-refinement, and
  dual-grid kernels (previously allocated on GPU → InvalidIRError).
- **3D FH kernels**: `stencil_clamped` guard removed — it forced α=0 at
  domain boundaries, which inflated Nu ~70% for 3D natconv refined.
  Root cause (prev buffer size) was already fixed in 534bb62.
- **test/Project.toml**: declares `KernelAbstractions` (was missing,
  causing `Pkg.test()` to error on Poiseuille 3D / thermal / species).
- **CI test suite**: `test_rheology.jl` and `test_viscoelastic.jl` added
  to `runtests.jl` (were present but not wired in).

## [0.1.0-dev] — 2026-04-10

### Added
- **Core LBM solver**: D2Q9 and D3Q19 lattices, BGK and MRT collision,
  Guo forcing, streaming (periodic, wall, axisymmetric)
- **Boundary conditions**: Zou-He velocity/pressure (2D+3D), bounce-back,
  spatially varying BCs via KrakenExpr expressions
- **Thermal LBM**: double distribution function with Boussinesq coupling,
  Rayleigh-Benard and natural convection drivers
- **Axisymmetric LBM**: Li et al. (2010) collision kernel,
  Hagen-Poiseuille pipe flow driver
- **Grid refinement**: patch-based with Filippova-Hanel rescaling,
  temporal interpolation, bilinear prolongation/restriction
- **.krk configuration DSL**: declarative simulation setup, presets
  (cavity_2d, poiseuille_2d, couette_2d, taylor_green_2d, rayleigh_benard_2d),
  Reynolds/Rayleigh helpers, sanity checks, spell-correction, parameter sweeps
- **GPU backends**: CPU, CUDA (H100/A100 tested), Metal (Apple Silicon)
  via KernelAbstractions.jl — single-source GPU portability
- **I/O**: VTK output (.vti/.pvd), STL import + voxelizer, diagnostics logger
- **Post-processing**: extract_line, probe, field_error, domain_stats
- **Documentation**: 13 theory pages, 11 validated examples-tutorials,
  dual API reference (Julia + .krk DSL), getting-started guide, cookbook
- **Benchmarks**: convergence studies (Poiseuille order 2, Taylor-Green order 2,
  cavity vs Ghia 1982), MLUPS performance (7675 MLUPS on H100)

### Known limitations
- Multiphase, rheology, viscoelastic, and species transport are implemented
  but not included in the v0.1.0 scope (available on dev branch)
- .krk runner does not dispatch on non-thermal refined cases
- Grid refinement cavity benchmark requires the Julia API directly
