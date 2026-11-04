# PR79 short restart acceptance — prospective protocol, 2026-10-23

Authority: Guillaume's PR79 comment 6360360842 and Jiachen's execution request.
This prospectively replaces CUDA bit-exact acceptance, not the historical six
failures of job26985556. No physical-accuracy gate changes. No solver edits.

## Scope

Source 41f5209d19f8cbb798a3027166d6022867675f6c; Float64; Nx=10, Ny=16;
C=M=10, T=190, Ma_E=.01, alpha=1e-4; regularized/direct/MRT; history_interval=1;
closures :nee and :moment; all other defaults identical to the old coupled.jl
fixture. Record full identity and mapping. One Julia/BLAS thread and one H100;
require actual EhdPoissonSetupGPU, CUDSS extension and no CPU fallback.
This is short restart verification, not physical validation or long-time identity.

## Frozen sequence and gates

1. Calibration process: eight independently initialised continuous 20-cycle GPU
   trajectories per closure, without restart. Retain carried fields and umax
   history at cycles0..20. At each matched cycle, compute the maximum elementwise
   pairwise absolute discrepancy (elementwise max minus min across repeats).
2. Each closure/quantity's tolerance is TWO times its largest calibration
   discrepancy across cycles. Factor two is an engineering finite-sample guard,
   not a statistical confidence interval. Fixed before data collection. Zero
   envelope gives zero tolerance; no epsilon floor or retrospective enlargement.
   Also require tolerance <= 1e-12*max(1,peak absolute signal). Write and hash
   tolerances.toml before launching the independent verification process.
3. Verification process: three fresh continuous/restart pairs per closure,
   split7+13; three fresh continuous cold controls. Compare f_in,q_f_in,Fx_prev,
   Fy_prev,phi and full umax history at every matched cycle8..20 for restart and
   0..20 for cold controls. Both must fall within frozen per-quantity tolerances.
   Record absolute/scaled relative errors and first differing cycle. Retain raw
   traces. No retry with larger tolerances if any check fails.
4. CPU: one fresh7+13 pair per closure, bit-exact carried fields and histories
   at every post-restore cycle. Exact cycle, parameter histories and configuration.
5. Disk restoration before advancement: all carried fields, series, scalar
   diagnostics, cycle, identity and parameters remain bit-exact on both backends.
6. Diagnostic clones only: three fresh7-cycle GPU states/restores per closure.
   Compare carried inputs and qfield before Poisson, then call the unchanged
   production Poisson solve and field reconstruction in actual order. Capture
   stage differences; discard these partially advanced states afterwards.
   Differences before Poisson cannot be attributed solely to cuDSS. If none is
   observed, report NOT_OBSERVED. Poisson-first differences support localisation
   on this fixture, not exclusion of every possible shared restart bug.

All acceptance gates must pass; stage localisation is descriptive. Report any
unobserved seam divergence as a limitation, not as evidence of determinism.
No Float32, multithread CPU, parameter-change or general-mesh qualification.

## Resources and exit decision

One H100, one CPU, 3GB host memory,20min walltime. Calibration cap480s and
verification cap600s including imports/JIT. Runtime dominates tiny-array memory.
Private frozen source/harness/environment; pre/post hashes and loaded-path check.
Only metadata/syntax preflight on login; compute exclusively in PBS allocation.
No repeated static ladder or long H256/H512 T190 simulation.

If PASS, report measured envelope and stage evidence on PR79. Guillaume merges
PR78 first; then update PR79 base without rewriting shared history, inspect its
incremental diff and dispatch full CI on the final head. Reuse Aqua evidence only
if executed code paths are unchanged. Historical H256 charge-error failure,
maintainer review/merge and physical qualification remain separate matters.
