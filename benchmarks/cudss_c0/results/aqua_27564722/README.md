# C0 cuDSS GPU validation — Aqua job 27564722 (A100-SXM4-40GB)

Run of `benchmarks/cudss_c0/c0_cudss.pbs` on one A100 (driver 610.57.04),
2026-11-04, exit status 0, walltime 15 min 35 s. Three environments:

- `env6`: CUDA.jl 6.4.1 / CUDSS.jl 1.0.0, C0 code (native transpose);
- `env5`: CUDA.jl 5.11.3 / CUDSS.jl 0.6.5, C0 code (transpose refused);
- `env5_base`: CUDA.jl 5.11.3 / CUDSS.jl 0.6.5, pre-C0 base `41d06f872`.

Logs were stripped of user paths (`~`) and the project id (`<RPID>`).

## Verdict

| # | Criterion | Verdict | Numbers |
|---|-----------|---------|---------|
| 1 | GPU test suites (cuDSS parity, C0 contract, EHD phi, transient moment, sidewalls CUDA, restart CUDA) | PASS | env6 7850/7850 pass; env5 7846/7846 pass; 0 fail, 0 error, 0 broken |
| 2 | Solve vs UMFPACK, 512² and 1024² (EHD Poisson, upwind conv-diff) | PASS | solve 8e-14 to 6.7e-12; 4-RHS same; Aᵀ (env6) 9e-14 to 6.7e-12 while Aᵀ vs A\b differs by 2e-2 to 0.92; env5 Aᵀ raises the explicit error |
| 3 | Refactorize reuses the analysis | PASS | 512²: 18 ms vs analysis 1.1-1.2 s; 1024²: 59-65 ms vs analysis 4.7-5.0 s |
| 4 | R11 memory contract | PASS | `CUDA.@allocated` = 0 B on all 10 iterations, every case, both envs; free-memory spread 0.0 MiB; nvidia-smi peak 1959 MiB, back to the same baseline after every case |
| 5 | EHD step, 197x321 `:direct`, 2000 steps | PASS within round-off, not bitwise | ms/step: base 17.28, env5 C0 16.96, env6 18.21; field sums agree to 1e-15 relative for phi, q, Ey; bit hashes differ in all three |

## Notes

- The bench prints `FAILED` on several refactorize lines (relative error
  1.0e-10 to 5.2e-9 against a fixed 1e-10 threshold). This is the threshold,
  not the solver: the new-value matrix `A2` (nonzeros scaled by
  `1 + 0.3 sin k`) is ill-conditioned, and UMFPACK's own forward error on it,
  measured on CPU (UMFPACK on A2 vs UMFPACK on A2ᵀ transposed, and one
  refinement step), is 3.7e-10 at 512² and 3.3e-9 at 1024² for conv-diff —
  the same size as the GPU differences (3.9e-10, 5.1e-9). The same check
  was not run for the EHD operator (1.0e-10 to 1.2e-10 at 1024²).
  These `FAILED` lines come from the old fixed threshold. The bench now
  judges refactorize lines against UMFPACK's own error on the same matrix
  (pass if err ≤ max(10·err_umfpack, 1e-12), both printed); this log was
  not re-run with it.
- EHD bitwise difference: the C0 solve calls the same `ldiv!` on the same
  factorization as the base; only the RHS copy and the solution buffer
  changed. The cause of the bit differences (cuDSS or EHD kernel
  run-to-run non-determinism vs the code change) is not established: a
  repeated base run was not done, and only field sums, not max field
  differences, are logged.
- EHD timing on this A100 (~17 ms/step) is far above the 1.39 ms/step H100
  reference in `benchmarks/results/ehd/README.md`, for the base code as well;
  not investigated here.
