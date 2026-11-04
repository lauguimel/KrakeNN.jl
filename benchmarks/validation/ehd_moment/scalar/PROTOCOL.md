# EC 2D integrated boundary, restart and T190 readiness

Prospective protocol, 2026-10-20; source follows agreed #73 comments 6262900965
and 6263245891. This is PR2 development, not a merged or qualified solver.
Parent PR1 is unmerged head 8c52503c2dbc6b12871ec6562063f6f7c6645854;
the new topic branch depends on it and will target dev/platform after base reconciliation.

## Decision and sequence

Job 26985482 admitted the external boundary (754 assertions); this run ports the
same checks to the actual source implementation, then checks ECState persistence
and performs one bounded coupled pilot. No static ladder or long T190 rerun.

1. Actual source kernels: identical exact transients, grids, times and criteria
   from ../ec2d-pr2-20261020/PROTOCOL.md (copied as ADMISSION_PROTOCOL.md when frozen).
   CPU1/CPU2/CUDA repeated corners at 1025x17; periodic H16/32/64; box controls
   tau=.8 H16/32/64 and tau=.5004434050067377/.5017736200269506 H8/16/32.
   Periodic finest L2<=2%, Linf/corners<=3%, L2 order>=1.7. Box finest L2<=5%,
   Linf/corners<=8%, order>=.8; new order>=legacy-.15, each-grid error<=1.10*legacy+1e-5.
   Normalise by the analytic perturbation q_exact-1, not background. No new gate.
2. CPU closure invariants and schema2 identity validation; schema1 migration to
   nee only, with mandatory old identity keys. Mismatches/unsupported options refused.
   Continuous-versus-restored CPU states must be exactly equal.
3. Real ECState CPU/CUDA Float64 regularized/direct/MRT at T190, 10x16:
   HDF5 disk roundtrip and 7+13 versus 20-cycle runs for nee and moment.
   Require exact carried-field restart and repeated-run equality; report absolute
   differences before any failure. No arbitrary GPU tolerance introduced. Preserve
   a failure as an integration/reproducibility limit, not change scientific gates.
4. Only after 1-3 pass: one H256,158x257,T190,C=M=10,alpha=1e-4,Ma_E=.01,
   free-slip,force_projection=none,perturbation1e-4,mode1,regularized/direct/MRT
   CUDA Float64 pilot:10 warmup+1000 timed cycles. Finite positive q, finite u,
   injector charge relative error<1e-11, genuine GPU Poisson and no fallback.
   H256 keeps tau_q within the admitted scalar range. Save a checkpoint and
   mapping/runtime metadata. This does NOT test nonlinear
   saturation, literature Vmax, physical current budgets or mesh convergence.

## Frozen evidence and resources

Freeze source commit, harness, project/manifest, reference arrays and all inputs;
check pathof(Kraken) and SHA. Julia1.11.9/CUDA5.11.3/CUDSS0.6.5, H100,
2CPU/3GB/20min. Previous admission used ~1.01GiB/2m05s; pilot D2Q9 and sparse
factorisation are small relative to 3GB. Phase caps:300s/600s/220s below wall limit.
No dependency updates during job. CPU checks bounded, long loops GPU only.
Write original logs, checkpoints, metrics and manifests. Review/merge stays with
Guillaume; final-head full CI is still required before review-ready PR.

## Exit to complete benchmark

If these gates pass, the next scientific job is the full coupled T190 branch
on the integrated source, with existing literature QoIs and conservation gates,
not another static scan. Use pilot runtime to bound resources; do not use a
1000-step pilot as saturated-flow evidence. If a gate fails, repair the named
integration issue only and rerun the affected check, retaining all other evidence.
