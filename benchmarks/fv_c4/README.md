# C4 — steady discrete adjoint on the FV Newton solver: receipts

Scripts for step C4 of the FV Newton roadmap
(`src/methods/inc_ns/fv_adjoint.jl`).

- `wall_closure.jl` — does the C1 wall closure (half-cell, p_f = p_P, ν_P;
  first-order wall pressure when ν varies) degrade a wall objective and its
  adjoint gradient? Stokes channel past a cell-aligned square, ν = 1 + θφ(x, y)
  at θ0 = 0 and 1, N = 8…128; objectives: drag with the residual's wall
  pressure, drag with an objective-level linear wall-pressure extrapolation
  (`FVWallForce(...; pressure = :linear)`), and an interior Δp. Prints J,
  dJ/dθ, observed orders, Richardson estimates, and an adjoint-vs-FD check on
  the coarsest grid. Run: `julia --project benchmarks/fv_c4/wall_closure.jl [Nmax]`.
  Receipt: `results/wall_closure_cpu.txt`. Finding: all three objectives and
  their gradients converge at the same observed order (≈ 1.4; 1.57 with the
  linear wall pressure), identically for constant and variable ν; the gradient
  carries the same relative discretization error as J (≈ 0.7 % at N = 128).
  The variable-ν wall pressure is not the leading error here (the obstacle
  corners and the first-order pressure part of the drag are).
- `c4_gpu.jl` + `c4_gpu.pbs` + `setup_c4.sh` — GPU job: the C4 test files with
  `KRAKEN_REQUIRE_GPU=1` (transposed cuDSS solve with the Newton factors,
  CPU/GPU gradient parity, zero device allocation with a workspace, free
  memory over 10 gradients), then the gradient cost relative to one Newton
  solve and one Newton iteration on the cavity at 512² and 1024² (Re = 100)
  for the uniform-ν, lid-speed and ν-field parameters and two objectives.
  CPU smoke run (32², 64², UMFPACK): `C4_BENCH_CPU=1 julia --project
  benchmarks/fv_c4/c4_gpu.jl`, receipt `results/c4_bench_cpu_smoke.txt`.
  Prepare the job folder on the Aqua login node with `setup_c4.sh` (after
  copying a git archive of feat/fv-c4 to `~/kraken_c4/src_c4`), then
  `qsub -P <RPID> c4_gpu.pbs`. Receipts: `results/aqua_<jobid>/`.
