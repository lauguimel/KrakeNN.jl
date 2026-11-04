# PR 2 admission: ordered two-dimensional transient boundary checks

2026-10-20, prospective supplement to the external 2026-10-19 screen. Implements
Guillaume's #73 comment 6263245891. Old frozen protocols/results are not changed.
This run uses production collision/stream operators from source
8c52503c2dbc6b12871ec6562063f6f7c6645854 with external two-moment and race-free
legacy boundary controls. It is NOT integrated PR 2, Poisson-coupled EC, restart,
or complete T190 qualification. Those remain downstream; no static rerun.

## Reference and dimensionless mapping

Retain q_t+U*q_y=D0*Laplacian(q), D0=.05, U=.1, q=1+.05*exp(y)*sin(k*y)*
cos(kx*x)*exp(-lambda*t), lambda=D0*(kx^2+k^2+1). Periodic-x Dirichlet isolation:
kx=2*pi,k=pi; EC box: kx=pi, k*cos(k)+sin(k)=0, with insulating sides and a
zero-normal-gradient collector. Bottom q=1. Same analytic initialization and
leading nonequilibrium gradient as the earlier screen; no forcing or fitted BC.
dt=(tau-.5)/(3*H^2*D0); compare at actual rounded final time, not nominal time.
All errors include wall nodes and normalize by q_exact-1, not total background.

1. Retain six tau=.8 H16/32/64, t=.2 cases for new closure. Compare GPU to
   existing independent NumPy arrays <=1e-11; exact reference arrays <=1e-13.
   CPU/GPU H16 q and populations <=1e-11. These Julia cases were NOT_RUN in
   the prior loading failures; this is missing evidence, not repeated success.
2. Add legacy injector under IDENTICAL ordered schedule for the box ladder.
   Do not call the concurrent production box kernel as a clean control (#64).
3. Add exact box ladders H8/16/32 at tau=.5004434050067377 and .5017736200269506,
   t=.02, both closures. These sample the low-diffusion parameter range; the
   shorter fixed physical duration limits cost, and does not prove long-time
   nonlinear stability. Retain 17x17/256-step prescribed-shear CPU/GPU controls.
4. CPU 1-thread, CPU 2-thread and three repeats per backend, plus three GPU
   repeats: a 1025x17 deterministic non-equilibrium boundary fixture (Nx+Ny>1024).
   Check all q/f differences and all four corner values <=1e-11 across backends;
   same-backend repeated populations must be identical. Injector q=1 and scalar
   corner reconstruction <=2e-14. This fixture probes boundary scheduling, not
   full-flow convergence; analytic ladders test time propagation separately.

## Frozen gates and rationale

- Periodic isolation: successive L2 orders >=1.7; finest L2<=2%, Linf<=3%,
  corner-sample Linf<=3%. Retain prior screen gates for near-second-order flat
  Dirichlet verification without claiming exact order from three grids.
- Every new box ladder: successive L2 orders >=.8, finest L2<=5%, Linf<=8%,
  four-corner Linf<=8%. The retained node-copy Neumann boundaries are only
  expected to be first-order; .8 allows pre-asymptotic effects but rejects a
  flat/increasing error. 5%/8% cap bulk/worst local error relative to the small
  perturbation (not background); these are bounded scalar-compatibility gates,
  not the separate static 1% gate or a full EC accuracy promise.
- Same-schedule comparison: new L2 order >= legacy order-.15 at each refinement;
  new L2/Linf/corner error <=1.10*legacy+1e-5 on each grid. The .15-order allowance
  corresponds to about 11% difference per factor-two refinement; 10% error
  allowance bounds practical regression and 1e-5 avoids comparing roundoff at
  exact injector corners. Both independent exact-solution gates must still pass.
- All analytic runs must remain finite with q>0. The short shear control also
  requires q<2. Invalid fields or runtime failures stop admission; retain failure.
- Report injector (excluding corners), sidewalls, collector, corners and interior
  Linf separately. Their location is not causal proof: if a gate fails or attribution
  is needed, isolate side/collector operators before proposing their repair.

## Race-free dependency argument

Electrode kernel writes only rows 1/Ny, columns 2:Nx-1 in a box; each thread reads
its own outgoing populations or untouched row 2/Ny-1. It never writes sidewalls.
Synchronize; recompute q. Side kernel writes only columns 1/Nx; each j reads
completed columns 2/Nx-1, including their completed electrode values at corners.
These neighbour columns are not written by side threads (Nx>=4). Synchronize and
recompute q before next collision. No same-launch read/write conflict exists.
The new and legacy closures use this same schedule; legacy production default
is NOT changed. A passing fixture is supporting evidence, not the dependency proof.

## Execution and exit decision

Reuse the now-qualified Julia1.11.9/CUDA5.11.3 environment and compiled -O0 mode.
Freeze source, scripts, references, environment and hashes. H100,2 CPU,3 GB,
25-minute allocation; single-thread fixture process<=300s, main process<=1000s,
individual small CPU time loops<=30s. No dependency installation, automatic retry,
or waiting for the queued job in chat. Record every executed/unrun requirement.

Success permits implementing the agreed PR 2 source/checkpoint option and repeating
the required checks on those real integrated kernels. It does not permit claiming
T190 PASS. Failure stops that integration admission; diagnose the measured failing
gate rather than repeating static cases or weakening thresholds.
