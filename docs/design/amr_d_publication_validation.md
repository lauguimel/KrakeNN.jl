# AMR D Publication Validation Plan

Date: 2026-05-05

## Scope

The publication-facing D stream is a static fixed-patch AMR feature:

- 2D D2Q9;
- 3D D3Q19 smoke coverage;
- ratio-2 refinement;
- one fixed patch known before the run;
- no dynamic regrid during the time loop;
- no multi-level claim;
- no GPU claim;
- no AD claim.

This keeps the feature honest and publishable. The pool/topology is fixed for
the whole simulation, so validation can focus on transport, collision,
boundaries and obstacle accuracy instead of runtime allocation or adaptation.

## Strategy

The current route-native implementation is conservative and preserves rest
states, but obstacle accuracy is sensitive to a coarse/fine interface too close
to the body. For publication D, the obstacle validation therefore uses an
interface-buffered patch:

```text
base obstacle domain: Nx=24, Ny=14
default patch:         8:17 x 4:11
publication patch:     3:22 x 1:14
```

The publication patch makes the obstacle and near wake live in a fine,
wall-to-wall band. The coarse/fine interfaces are moved upstream/downstream
instead of cutting through the obstacle boundary layer, while retaining two
active coarse columns on each periodic-x side in the base case.

This is not a trick: it is the expected static-AMR usage pattern for body
flows. A body-fitted or locally refined solver should not place its strongest
resolution transition directly on the quantity being measured.

## Validation Ladder

Required local validation sequence:

1. subcycling ledger canaries remain green;
2. route-native topology/streaming canaries remain green;
3. open-boundary/BFS surgical canaries remain green;
4. Couette route-native profile gates remain green;
5. Poiseuille route-native profile gates remain green;
6. square obstacle conserves fluid mass to roundoff;
7. cylinder interface-buffered route-native Cd is within 10% of the leaf
   oracle on short local canaries;
8. aqua ladder for square/cylinder scales 1 and 2 with
   `patch_strategy=:interface_buffered`.
9. publication table with `cartesian_coarse`, `leaf_oracle` and
   `amr_route_native` rows for square/cylinder, including `u/v`, `Cd`,
   error versus leaf oracle, elapsed time and MLUPS.

Required 3D validation sequence:

1. D3Q19 coalesce/explode/moment/collision primitives remain green;
2. 3D route topology keeps face/edge split/coalesce weights conservative and
   explicitly rejects corner transfers;
3. 3D route-native streaming preserves active population sums for zeroed
   boundary packets;
4. periodic-x plus stationary y/z wall routing wraps and bounces both coarse
   and fine packets;
5. fixed-patch D3Q19 forced-channel smoke remains finite, accelerates only in
   x, and conserves active mass to roundoff;
6. 3D solid-mask route canary conserves mass for a solid cube fully contained
   in the refined patch.

Local baseline recorded in
`benchmarks/results/amr_obstacle_convergence_2d_local_interface_buffered_20260505.csv`:

- cylinder scale 1: route-native Cd/oracle Cd = `1.060x`;
- cylinder scale 2: route-native Cd/oracle Cd = `1.052x`;
- square scales 1 and 2: finite positive streamwise velocity and roundoff mass
  drift;
- all route-native obstacle rows: relative mass drift below `1e-12`.

Aqua baseline recorded in
`benchmarks/results/amr_obstacle_convergence_2d_aqua_interface_buffered_20260505.csv`
matches the local values for Cd and mass drift. PBS job: `20808208.aqua`.

Publication-table local canary recorded in
`benchmarks/results/amr_d_publication_summary_2d_local_D_pub_canary_20260505.csv`.
It verifies the reporting path for:

- `cartesian_coarse`;
- `leaf_oracle`;
- `amr_route_native`;
- square `u/v` accuracy and mass conservation;
- cylinder `u`, `Cd`, `Fx/Fy` and mass conservation;
- elapsed-time, speedup and MLUPS columns.

Publication-table aqua closure recorded in
`benchmarks/results/amr_d_publication_summary_2d_aqua_D_pub_20260505_long.csv`
from PBS job `20812821.aqua`. The detailed verdict is in
`docs/design/amr_d_closure_report_20260505.md`.

The preserved artifact manifest is
`benchmarks/results/AMR_D_PUBLICATION_RESULTS.md`. It indexes the primary aqua
CSV files, the PBS log, local canary CSVs, `.krk` reproducibility sources and
stored figure outputs.

Accuracy passes for D: AMR is much closer than coarse Cartesian for square
`u/v` and cylinder `Cd` on scales `{1,2,4}`.

Runtime speedup does not pass on CPU: AMR uses fewer active cells but is slower
than the dense leaf oracle in this route-native prototype. Do not claim CPU
speedup for D.

3D local fixed-patch channel smoke:

- runner: `run_conservative_tree_poiseuille_route_native_3d`;
- default domain: `Nx=8, Ny=8, Nz=6`;
- compact stress patch: `3:6 x 3:6 x 2:5`;
- publication-style buffered patch:
  `patch_strategy=:cross_section_buffered`, giving `2:7 x 1:8 x 1:6` for
  the default domain;
- boundary policy: periodic x, stationary bounce-back y/z;
- forcing: Guo D3Q19, `Fx=2e-5`;
- compact `steps=80`: `ux_mean=2.5897914250811675e-4`, transverse means below
  `4e-17`, relative mass drift `1.08e-13`;
- local-patch profile diagnostic versus dense leaf oracle:
  `linf=6.879e-4`, relative `linf=0.583`, mean-velocity ratio `0.389`.
  This is a diagnostic, not a publication accuracy gate;
- buffered `steps=80`: relative dense-oracle `linf=0.231`,
  mean-velocity ratio `0.768`, relative mass drift `1.41e-14`;
- full-domain refined patch parity canary:
  `Nx=4, Ny=4, Nz=3`, patch `1:4 x 1:4 x 1:3`, `steps=20`,
  dense-leaf oracle errors `l2=0`, `linf=0`.

## Limits

The default compact patch is still useful as a stress test. Its cylinder Cd
gap documents the remaining near-interface transport error. That error is the
motivation for future subcycling integration.

Do not claim that D has solved arbitrary coarse/fine obstacle placement until
the subcycling time integrator is wired and the compact-patch ladder also
passes.

Do not claim 3D obstacle or sphere AMR yet. The current 3D gate is a
fixed-patch D3Q19 channel smoke plus exact full-patch dense-oracle parity. The
local-patch profile gap must be reduced by subcycling or a matched
coarse/fine-time oracle before a 3D profile-level accuracy claim. The current
3D solid-mask canary is a mass-conservation gate only, not a drag or sphere
accuracy claim.

## Commands

Local surgical checks:

```bash
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_subcycling_2d.jl"); include("test/test_conservative_tree_obstacle_interface_2d.jl"); include("test/test_conservative_tree_open_boundary_2d.jl"); include("test/test_conservative_tree_streaming_2d.jl")'
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_3d.jl"); include("test/test_conservative_tree_topology_3d.jl"); include("test/test_conservative_tree_streaming_3d.jl")'
```

Publication obstacle ladder:

```bash
KRK_AMR_CONV_PATCH_STRATEGY=interface_buffered \
KRK_AMR_CONV_BASE_STEPS=1200 \
KRK_AMR_CONV_SCALES=1,2 \
julia --project=. benchmarks/amr_obstacle_convergence_2d.jl
```

Aqua:

```bash
KRK_AMR_CONV_PATCH_STRATEGY=interface_buffered \
qsub hpc/amr_obstacle_convergence_2d_aqua.pbs
```

Publication accuracy/efficiency table:

```bash
KRK_AMR_D_PATCH_STRATEGY=interface_buffered \
KRK_AMR_D_SCALES=1,2,4 \
KRK_AMR_D_BASE_STEPS=2400 \
KRK_AMR_D_AVG_WINDOW=600 \
julia --project=. benchmarks/amr_d_publication_table_2d.jl
```

Aqua:

```bash
KRK_AMR_D_PATCH_STRATEGY=interface_buffered \
qsub hpc/amr_d_publication_table_2d_aqua.pbs
```

Reproduce from `.krk` case files:

```bash
KRK_AMR_TAG=from_krk_20260505 \
julia --project=. benchmarks/amr_d_publication_from_krk_2d.jl
```

Quick `.krk` canary:

```bash
KRK_AMR_D_CASES=square_scale1.krk,cylinder_scale1.krk \
KRK_AMR_D_STEPS_OVERRIDE=40 \
KRK_AMR_D_AVG_WINDOW_OVERRIDE=20 \
KRK_AMR_TAG=from_krk_smoke \
julia --project=. benchmarks/amr_d_publication_from_krk_2d.jl
```

Publication figure:

```bash
julia --project=. benchmarks/plot_amr_d_publication_2d.jl
```
