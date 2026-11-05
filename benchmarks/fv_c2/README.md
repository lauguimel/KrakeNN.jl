# C2 — coloured dual Jacobian + Newton: receipts

Scripts for step C2 of the FV Newton roadmap
(`src/methods/inc_ns/fv_newton.jl`).

- `pin_scaling.jl` — effect of the pressure-pin row scale on the Newton linear
  solve through the CPU seam (UMFPACK): pinned row scaled by 1, by the median
  continuity diagonal, by 1e-6 and 1e6, at a mid-Newton state of the cavity.
  Run: `julia --project benchmarks/fv_c2/pin_scaling.jl`.
  Receipt: `results/pin_scaling_cpu.txt` (identical solutions, backward error
  ≤ 1.3e-13; no rescale needed on UMFPACK).
- `c2_gpu.jl` + `c2_gpu.pbs` + `setup_c2.sh` — GPU job: the C2 test files with
  `KRAKEN_REQUIRE_GPU=1` (CPU/GPU Jacobian parity, Newton on cuDSS against the
  CPU history, one analysis then refactorizations, zero device allocation in
  the hot path, free memory over 10 iterations), then the cavity Newton
  benchmark at 512² and 1024² (Re = 100 and 1000, from rest; per-iteration
  assembly / refactorize / solve split, first factorization, iteration count,
  Ghia errors, cuDSS memory estimates, nvidia-smi peak memory, hot-loop
  `CUDA.@allocated` and free-memory drift) and the pin-scale check on cuDSS.
  Prepare the job folder on the Aqua login node with `setup_c2.sh` (after
  copying a git archive of the branch to `~/kraken_c2/src_c2`), then
  `qsub -P <RPID> c2_gpu.pbs`. Receipts: `results/aqua_<jobid>/`.
