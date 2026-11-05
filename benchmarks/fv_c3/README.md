# C3 — GNF viscosity in the steady FV residual + Newton: receipts

GPU job for step C3 of the FV Newton roadmap
(`src/methods/inc_ns/fv_gnf.jl`).

- `c3_gpu.jl` + `c3_gpu.pbs` + `setup_c3.sh` — the C3 test files with
  `KRAKEN_REQUIRE_GPU=1` (`incns_fv_gnf.jl` on the CPU, `incns_fv_gnf_gpu.jl`:
  CPU/GPU parity of the GNF residual, Jacobian and dual-law residual, Newton on
  cuDSS against the CPU history, zero device allocation in the hot path), then
  a GNF Newton benchmark from rest: power-law cavity n = 0.5, K = 0.01,
  ε = 1e-2 at 256², 512², 1024², and the Carreau-Yasuda obstacle channel of
  the LBM comparison at m = 32 and 64. Per case: nnz/row, colours, set-up,
  per-iteration assembly / refactorize / solve / line search and residual
  history, first factorization, cuDSS memory estimates and fill, hot-loop
  `CUDA.@allocated` and free-memory drift, GPU vs CPU Jacobian at 256².
- Run: copy a git archive of `feat/fv-c3` to `~/kraken_c3/src_c3`, run
  `setup_c3.sh` on the Aqua login node, then `qsub -P <RPID> c3_gpu.pbs`.
  Receipts: `results/aqua_<jobid>/`.
- Local smoke run of the bench path without a GPU (CPU backend, 32² cavity and
  m = 4 channel, CUDA-only measurements skipped):
  `C3_CPU_SMOKE=1 julia --project=<env> benchmarks/fv_c3/c3_gpu.jl`.
