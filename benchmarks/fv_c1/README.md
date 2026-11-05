# C1 — steady FV residual: receipts

Scripts for step C1 of the FV Newton roadmap
(`src/methods/inc_ns/fv_steady_residual.jl`).

- `j0_parity.jl` — residual of C1 vs the J0 prototype
  (`benchmarks/fv_newton_j0/FVJ0.jl` on `feat/fv-newton-j0`) on random states,
  11 configurations (cavity NS/Stokes, inlet/outlet channel, periodic channel
  with body force, masked manifold). Run:
  `FVJ0_PATH=<path>/FVJ0.jl julia --project benchmarks/fv_c1/j0_parity.jl`.
  Receipt: `results/j0_parity_cpu.txt` (identical residuals).
- `c1_gpu.jl` + `c1_gpu.pbs` — GPU job: the C1 test files with
  `KRAKEN_REQUIRE_GPU=1` (GPU/CPU parity in Float64 and Dual{4}, zero device
  allocation, free memory over 10 calls) and the residual timing at 512² and
  1024². Submit from the job folder prepared on the Aqua login node (see the
  header of `c1_gpu.pbs`): `qsub -P <RPID> c1_gpu.pbs`. Receipts:
  `results/aqua_<jobid>/`.
