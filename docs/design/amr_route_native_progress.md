# AMR Route-Native Progress Ledger

Date: 2026-05-04

This document tracks the autonomous AMR work against the first eight milestones
from `docs/design/amr_complete_project_plan.md`. It is intentionally strict:
completed means backed by surgical patch tests and, where relevant, a short
macro-flow.

## Current Validation Commands

```bash
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_streaming_2d.jl")'
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_open_boundary_2d.jl")'
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_gpu_pack_2d.jl")'
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_subcycling_2d.jl")'
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_adaptation_2d.jl"); include("test/test_conservative_tree_multipatch_2d.jl")'
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_topology_2d.jl"); include("test/test_conservative_tree_2d.jl")'
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_3d.jl"); include("test/test_conservative_tree_topology_3d.jl"); include("test/test_conservative_tree_streaming_3d.jl")'
```

## 1. Streaming Composite Natif 2D

Status: done for one ratio-2 patch.

Implemented:

- same-level active-cell routes;
- coarse-to-fine face and corner routes using leaf-equivalent routed fractions
  with explicit coarse residuals (`1/4` per child trajectory);
- fine-to-coarse face and corner coalesces;
- route topology and packed route representation;
- periodic-x, wall-y, moving-wall-y and solid-mask route variants.

Surgical tests:

- packet canaries for direct, split and coalesce routes;
- leaf-equivalent coarse-to-fine face/corner fraction and residual canaries;
- orientation-wise conservation;
- route-native streaming against leaf-grid oracle canaries;
- periodic and wall boundary conservation.

Remaining before "complete AMR 2D":

- multi-patch routes;
- open inlet/outlet routes.

## 2. Collision Locale Active 2D

Status: done for BGK and Guo on active cells.

Implemented:

- coarse active cells collide with volume `1`;
- fine active cells collide with volume `0.25`;
- coarse cells covered by the patch are excluded;
- Guo forcing works in route-native Poiseuille.

Validation:

- route-native Couette runner;
- route-native Poiseuille runner;
- `validate_conservative_tree_route_native_phase_p_2d` compares route-native
  Couette and Poiseuille to the Phase-P oracle runners.

## 3. Boundaries Natives 2D

Status: partially done.

Done:

- periodic x;
- static wall y;
- moving wall y for Couette;
- integrated cell-level Zou-He west velocity and east pressure closures;
- composite active-cell Zou-He west/east application for coarse and fine
  boundary cells;
- one-step open-x route smoke with wall-y and composite Zou-He closures;
- surgical open-boundary canaries for fine inlet/outlet faces when a patch
  touches west/east boundaries;
- short inlet-spanning open-channel smoke with bounded finite drift;
- short route-native open-channel smoke with bounded mass drift;
- open-channel mass ledger canary comparing a local patch with a full-domain
  patch; the local-patch target `mass_rel_drift < 0.02` now passes when the
  open runners use leaf-equivalent coarse same-level routes;
- zero-inlet open-channel rest-state canary; the full-domain patch remains at
  rest to roundoff, and the local patch now also remains at rest with
  leaf-equivalent coarse same-level routes;
- bounce-back solid mask;
- square obstacle route-native smoke;
- vertical facing step route-native smoke.
- solid-aware composite Zou-He skips solid inlet/outlet cells and rejects
  partially solid active coarse cells;
- nominal route-native BFS smoke at 240 steps stays finite and has positive
  mean streamwise velocity;
- short local-patch BFS route-native canary tracks the leaf open-solid oracle
  for mass drift and mean velocity after the leaf-equivalent coarse route fix.

Not done yet:

- BFS convergence and publication-quality comparison against the Cartesian
  oracle;
- long-horizon open-channel/BFS stability envelope. The route-native BFS
  short canary now tracks the oracle, but convergence studies still need the
  nominal/longer canaries, not early transients.

Next surgical patch:

- add BFS route-native versus Cartesian benchmark tables after one more
  boundary packet audit, then run the same ladder for square and cylinder
  obstacles.

Benchmark hook:

- `benchmark_conservative_tree_cartesian_vs_amr_2d` now emits comparable
  `leaf_oracle` and `amr_route_native` rows for BFS, square and cylinder;
- `benchmarks/amr_cartesian_vs_route_native_2d.jl` writes those rows to CSV;
- `hpc/amr_cartesian_vs_route_native_2d_aqua.pbs` is the aqua submission
  wrapper (`KRK_AMR_STEPS`, `KRK_AMR_FLOWS`, `KRK_AMR_TAG`).
- `convergence_conservative_tree_obstacles_2d` emits a scaled square/cylinder
  ladder with Cartesian leaf-oracle and AMR route-native rows;
- `benchmarks/amr_obstacle_convergence_2d.jl` writes the convergence ladder to
  CSV, including cylinder drag/Cd columns;
- `hpc/amr_obstacle_convergence_2d_aqua.pbs` is the aqua submission wrapper
  (`KRK_AMR_CONV_FLOWS`, `KRK_AMR_CONV_SCALES`,
  `KRK_AMR_CONV_BASE_STEPS`, `KRK_AMR_CONV_STEP_EXPONENT`,
  `KRK_AMR_CONV_AVG_WINDOW`, `KRK_AMR_TAG`).

Aqua runs:

- `20757938.aqua` produced
  `benchmarks/results/amr_cartesian_vs_route_native_2d_aqua_20757938.csv`
  at 1200 steps for BFS, square and cylinder. Square and cylinder conserve
  mass to roundoff; BFS remains finite but has a much larger open-flow mass
  drift than the leaf oracle. A follow-up full-patch BFS surgical test shows
  the route-native open-solid path matches the leaf oracle when no coarse/fine
  interface cuts the open channel; the remaining BFS issue is therefore the
  localized AMR coarse/fine open-flow coupling.
- `20757949.aqua` produced
  `benchmarks/results/amr_obstacle_convergence_2d_aqua_conv_20757949.csv`
  for square/cylinder scales 1 and 2. This is a numerical ladder canary, not
  yet a publication convergence claim; route-native obstacle metrics still
  differ materially from the leaf oracle while the transport mass ledger stays
  conservative.
- `20761329.aqua` reran the benchmark after full-packet coarse-to-fine routes.
  BFS streamwise velocity is much closer to the leaf oracle, but open-flow
  mass drift remains high (`1.91e-1` versus `4.93e-3` for the oracle).
- `20761330.aqua` reran the square/cylinder convergence ladder after
  full-packet coarse-to-fine routes. Obstacle mass conservation remains at
  roundoff; cylinder Cd moved closer to the leaf oracle but is still not a
  convergence claim.
- `20763397.aqua` reran BFS only after enabling leaf-equivalent coarse
  same-level routes for open runners. At 1200 steps the AMR route-native row
  tracks the leaf oracle closely (`ux=2.395e-2` versus `2.375e-2`,
  `uy=-1.735e-3` versus `-1.732e-3`, mass drift `9.35e-3` versus `4.93e-3`).
- `20763404.aqua` reran square/cylinder scales 1 and 2 after restoring the
  non-full-packet topology. Mass conservation remains at roundoff, but
  square/cylinder velocities and cylinder Cd are still materially far from the
  leaf oracle. The next obstacle patch must audit periodic/solid interface
  transport before claiming convergence.
- `benchmarks/results/amr_obstacle_convergence_2d_leafeq_obstacles_20260504.csv`
  records the local square/cylinder ladder after enabling leaf-equivalent
  coarse routes for obstacle runners. Metrics improve materially while mass
  remains at roundoff: cylinder Cd moves from `8.52e2 -> 2.65e2` at scale 1
  and from `9.86e1 -> 3.26e1` at scale 2, versus leaf-oracle `1.54e2` and
  `1.75e1`.
- `20763538.aqua` completed the same leaf-equivalent obstacle ladder on aqua and
  produced
  `benchmarks/results/amr_obstacle_convergence_2d_aqua_leafeq_obstacles_20260504.csv`
  with matching metrics and roundoff-level mass conservation.
- 2026-05-05 publication-D patch strategy: the obstacle convergence runner now
  accepts `patch_strategy=:interface_buffered`, which uses a fixed wall-to-wall
  fine band around the cylinder/square and moves coarse/fine interfaces away
  from the body. A local short cylinder canary is within 10% of the leaf oracle
  while conserving mass to roundoff
  (`benchmarks/results/amr_obstacle_convergence_2d_local_interface_buffered_smoke.csv`:
  Cd ratios `1.029x` at scale 1 and `1.016x` at scale 2). A local 1200/2400
  step ladder with the widened publication patch gives Cd ratios around
  `1.060x` at scale 1 and `1.052x` at scale 2. This is the
  publication path for static fixed-patch D; the compact default patch remains
  the stress test for future subcycling.
- `20808208.aqua` completed the same 1200/2400-step interface-buffered ladder
  on aqua and produced
  `benchmarks/results/amr_obstacle_convergence_2d_aqua_interface_buffered_20260505.csv`.
  The CPU aqua values match local: cylinder Cd ratios are `1.060x` at scale 1
  and `1.052x` at scale 2, with route-native relative mass drift below
  `1e-12`.
- 2026-05-05 D publication table: `amr_d_publication_table_2d.jl` now emits
  raw and summary CSVs with `cartesian_coarse`, `leaf_oracle` and
  `amr_route_native` rows. The summary reports `u/v`, `Cd`, errors against the
  dense leaf oracle, elapsed time, speedup versus leaf and MLUPS. Local canary:
  `benchmarks/results/amr_d_publication_summary_2d_local_D_pub_canary_20260505.csv`.
  Long scale `{1,2,4}` runs belong on aqua via
  `hpc/amr_d_publication_table_2d_aqua.pbs`.
- `20812821.aqua` completed the D publication table at scales `{1,2,4}` and
  produced
  `benchmarks/results/amr_d_publication_summary_2d_aqua_D_pub_20260505_long.csv`.
  Accuracy closes: AMR cylinder `Cd` relative error is `5.98%`, `5.63%`,
  `3.63%`, versus coarse Cartesian errors above `500%`. Runtime speedup does
  not close on CPU: AMR uses `87.5%` of the dense leaf cells but runs at
  roughly `0.67x-0.75x` leaf speed on the larger cases. See
  `docs/design/amr_d_closure_report_20260505.md`.
- The publication-D result bundle is now reproducible from
  `benchmarks/krk/amr_d_publication_2d/*.krk` via
  `benchmarks/amr_d_publication_from_krk_2d.jl`. Retained artifacts are
  indexed in `benchmarks/results/AMR_D_PUBLICATION_RESULTS.md`, and stored
  figure outputs live under `benchmarks/results/figures/`.
- 2026-05-05 D convergence gallery: `amr_d_convergence_gallery_2d.jl` now
  generates Cartesian-vs-D plots for Couette, Poiseuille X/Y bands, BFS,
  square and cylinder. Source templates are in
  `benchmarks/krk/amr_d_convergence_2d/`; retained local smoke/ramp outputs
  are indexed in `benchmarks/results/AMR_D_CONVERGENCE_GALLERY_RESULTS.md`.
  The four-level cylinder nesting probe parses but is intentionally rejected
  before conservative-tree runtime because nested `Refine parent` routing is
  not implemented yet.

Notes:

- `20761329.aqua` and `20761330.aqua` are retained as exploratory full-packet
  canaries. A one-step leaf-oracle audit showed that full-packet coarse-to-fine
  injection doubles interface packets in the current non-subcycled route-native
  path. The non-subcycled topology therefore uses leaf-equivalent residual
  routes again; full-packet transfer belongs with the subcycling milestone where
  the packet is split in time instead of injected in one coarse step.
- A zero-inlet open-channel audit showed the deeper blocker: the current
  non-subcycled local interface conserves packet sums but does not preserve the
  rest distribution locally. Zou-He open boundaries then convert that interface
  perturbation into mass and velocity drift. Open-channel and BFS route-native
  runners now opt into `coarse_route_mode=:leaf_equivalent`, which splits
  coarse same-level packets into leaf-equivalent residuals and restores the
  rest-state canary. Square/cylinder route-native runners also use this mode
  after a local audit showed improved obstacle metrics with roundoff-level mass
  conservation.
- 2026-05-04 Phase A diagnostic (obstacle + interface coarse/fine).
  - `test/test_conservative_tree_obstacle_interface_2d.jl` adds 4 rest-state
    canaries (obstacle inside fine, in coarse, straddling interface, adjacent
    to interface), a `diff_route_vs_oracle_one_step` localiser at
    1/2/5/20/100 steps, and a MEA drag isolation test. All 22 assertions pass.
  - Rest-state result: route-native preserves mass and velocity to roundoff
    in every obstacle + interface configuration. There is no separate solid
    transport bug. The constant `max|ΔF| ≈ 1/9` between paths is a harmless
    storage difference inside solid leaf cells (route-native zeros solid
    output cells, oracle reconstructs them from the coarse equilibrium); the
    fluid-cell velocities are bit-identical (`max|Δu| = 0`).
  - Driven-flow result: under the cylinder regime (Fx=2e-5, Nx=24, Ny=14,
    cylinder of radius 3 in patch 8:17 × 4:11), the route-native vs oracle
    leaf transport diff grows roughly linearly: `max|Δu|` is 1.77e-6 at step
    1, 9.08e-6 at step 5, 3.40e-5 at step 20, 3.66e-4 at step 100. Linear
    growth is consistent with the documented "non-subcycled coarse-to-fine
    injection doubles interface packets" issue. Extrapolated to the 1200-step
    cylinder runner, the accumulated transport mismatch is the primary
    candidate cause of the ~1.86x Cd ratio (route-native 3.26e+01 vs leaf
    oracle 1.75e+01 at scale 2).
  - MEA drag is bit-identical when fed identical Fpre/Fpost arrays, so the
    drag computation itself is not the source of the discrepancy.
  - Verdict: B-1 (priority #1 obstacle gap is blocked by priority #2
    subcycling). No surgical transport fix in src/ is justified at this
    stage. Next step is to wire `ConservativeTreeSubcycleLedger2D` into the
    route-native time integrator, then re-evaluate the obstacle ladder with
    one coarse step plus two fine half-steps.

## 4. Multi-Patch Statique 2D

Status: ownership tables started; route topology still pending.

Implemented:

- `ConservativeTreePatchSet2D` groups several disjoint ratio-2 patches;
- parent-cell and leaf-cell owner tables;
- active coarse mask and active volume accounting;
- overlap and domain rejection;
- parser-adjacent `.krk` helpers converting base-grid `Refine` blocks to
  conservative-tree patch ranges.

Validated by:

- disjoint patch ownership canaries;
- parent/leaf owner lookup canaries;
- active volume and active coarse mask canaries;
- `.krk` `Refine` helper canaries for valid, unsupported ratio, nested parent
  and 3D-refine rejection.

Required next sequence:

1. Add route tests for coarse-to-fine and fine-to-coarse near two patches.
2. Add patch-patch adjacency tests.
3. Add route-native streaming over disjoint patch sets.
4. Only then run macro-flows with two refined bands.

Exit gate:

- no double ownership;
- active mass/volume matches oracle;
- all route weights sum to one by orientation.
- route topology covers patch-patch adjacency without packet loss.

## 5. Adaptation Dynamique CPU 2D

Status: first conservative prescribed regrid path done.

Implemented:

- direct conservative patch regrid;
- mask-driven patch adaptation for solid proximity;
- pure solid-mask patch indicator;
- pure scalar-threshold patch indicator;
- pure gradient-magnitude indicator field;
- composite AMR leaf velocity field for indicator input;
- velocity-gradient patch decision with direct conservative regrid;
- short route-native Poiseuille macro-flow driven by velocity-gradient regrid;
- mask-adaptive VFS route-native macro-flow around the step;
- range-level hysteresis primitive for grow/shrink decisions;
- prescribed adaptive Poiseuille route-native canary;
- production-facing adaptation policy/proposal/plan layer;
- parent-grid indicator plans with padding, min-size, growth limits and
  hysteresis;
- `.krk` `Refine` blocks converted to named adaptation proposals;
- `.krk` adaptive `Refine` criteria parse as
  `criterion = gradient(ux) > threshold` with `update_every`, `pad`,
  `max_growth`, `shrink_margin` and `balance = 1`;
- `.krk` criterion helpers convert parsed criteria to adaptation policies and
  indicator-driven plans;
- plan application helper that calls the direct conservative regrid path only
  when the selected patch range changes.

Validated by:

- direct regrid equals leaf-oracle regrid for grow, shrink and shift;
- population sums are conserved through regrid;
- indicator and hysteresis functions are tested without mutating transport
  state;
- composite velocity extraction feeds the gradient selector on a surgical
  local-speed patch;
- velocity-gradient adaptation regrids that patch while conserving population
  sums;
- gradient-driven adaptive Poiseuille regrids conservatively over a short
  route-native run;
- mask-driven VFS regrids around the step while keeping fluid-mass drift
  bounded;
- adaptive Poiseuille keeps mass drift bounded;
- policy-level tests cover domain clamp, min-size expansion, max-growth
  limiting, near-shrink hysteresis, indicator-driven plans, `.krk` proposals
  and conservative regrid through a plan.
- parser-level tests cover adaptive `Refine` criterion fields and reject
  unsupported `balance != 1`.

Not done yet:

- error indicators;
- integration of physical error indicators with hysteresis;
- multi-level enforcement of 2:1 balancing beyond the parsed DSL contract;
- adaptive square obstacle/VFS macro-flow.

Next surgical patch:

- add physical error indicators on top of the policy/proposal/plan layer, then
  connect the parsed `.krk` criteria to production macro-flow runners.

## Multi-Level Rule

Current executable route-native topology is still level 0 to level 1. The DSL
contract is already fixed for the complete AMR path:

- level 0 is the Cartesian base grid;
- level `k + 1` may only refine inside a level `k` parent patch;
- all active adjacent leaves must satisfy `abs(level_a - level_b) <= 1`;
- `.krk` blocks must keep `balance = 1`; other values are rejected until a
  different balancing rule is implemented and tested;
- before activating a new patch, the proposal must be padded, clamped, checked
  for ownership conflicts, then 2:1-balanced against neighbours.

## 6. Sous-Cycling Temporel 2D

Status: conservative packet ledger started; no time integrator yet.

Implemented:

- `ConservativeTreeSubcycleLedger2D` for one coarse step and two fine
  half-steps;
- coarse-to-fine face and corner packet deposits split in time;
- fine-to-coarse face and corner packet accumulation by half-step;
- orientation and total packet sums for cycle-level canaries;
- reset helper for repeated canary cycles.

Validated by:

- a face packet is consumed once across two half-steps;
- a corner packet is split only in time, not spatially duplicated;
- fine-to-coarse half-step packets accumulate to the expected orientation sum;
- a symmetric full-cycle ledger preserves expected orientation and total sums;
- rest equilibrium axis face cycle balances per orientation: a coarse cell at
  rest deposits exactly `w_q * V_c` into the ledger across both sub-steps and
  the fine equilibrium reflux in the opposite direction matches it to
  roundoff for q in 2:5;
- rest equilibrium corner cycle exposes the known 2x geometric imbalance:
  `coarse_to_fine[q] = w_q * V_c` while `fine_to_coarse[opp(q)] = w_q * V_c / 2`
  for q in 6:9 (one fine corner cell vs four coarse children, two sub-steps).
  This is intrinsic to the subcycling corner geometry, not a ledger bug.
  The future time integrator must compensate it via Filippova-Hanel rescaling
  or an explicit residual-on-coarse contribution.
- unsupported ratio, invalid face/corner direction, wrong block shape and bad
  substep are rejected.

Still required before implementation (next focused session, larger scope):

1. design choice: how the ledger composes with the existing route-native
   coarse same-level path under `coarse_route_mode=:leaf_equivalent`
   (the current production setting). The ledger split-in-time replaces the
   one-shot non-subcycled coarse-to-fine packet doubling identified in the
   2026-05-04 Phase A diagnostic; the residual-on-coarse arithmetic must be
   reconciled with the ledger's split-in-time arithmetic so they do not
   double-count;
2. wire the ledger into route-native coarse/fine streaming with one coarse
   stream + collision and two fine half-step stream + collision passes;
3. define and document the collision ordering (coarse first vs fine first,
   and where the reflux lands in time);
4. choose a Filippova-Hanel-style omega rescaling rule for fine cells when
   `omega_coarse != omega_fine` is required to keep stress consistent across
   the interface (current runners use `omega_coarse == omega_fine`);
5. integration canaries to add before any macro-flow:
   - subcycled rest-state on a 1-patch composite reproduces non-subcycled
     rest-state to roundoff;
   - subcycled Couette and Poiseuille short runs match the non-subcycled
     route-native path to better than the current `:leaf_equivalent` accuracy;
   - subcycled obstacle (cylinder) ladder local first, then aqua, with the
     target Cd ratio versus oracle dropping below 1.10x.

The scope of step 1-5 above is genuinely a multi-day design effort. It must
not be rushed in the same session as the diagnostic that motivated it.

Exit gate:

- cycle-level mass conservation per orientation;
- Couette and Poiseuille short runs remain close to the non-subcycled path.

## 7. GPU 2D

Status: GPU-ready route packing and CPU parity canaries started; no device
kernel yet.

Implemented:

- `ConservativeTreeGPURoutePack2D` structure-of-arrays route pack;
- primitive route arrays for source/destination packed cell, `q`, route kind
  and weight;
- primitive block and cell metadata arrays;
- packed route weight-sum canary helper;
- CPU replay path driven only by the packed arrays for interior routes.

Validated by:

- primitive array types are `Int32`, `UInt8` and typed route weights;
- packed route source/destination ids match logical topology routes;
- route categories match direct/interface/boundary topology categories;
- every packed source/orientation route sum remains one;
- packed CPU replay matches logical route-native interior streaming exactly.

Still required before implementation:

- transfer this pack to CUDA/Metal arrays;
- write the no-allocation device kernel for interior routes;
- add boundary route kernels after open-boundary hardening;
- compare CPU/GPU route-native Poiseuille on a small fixed patch.

First target:

- device parity for one small packed route packet canary, then one small
  route-native Poiseuille run.

## 8. Topologie Et Primitives 3D

Status: fixed-patch D3Q19 route-native smoke gate active.

Existing 3D refinement utilities are not yet the complete conservative-tree
D3Q19 route topology. The first accepted 3D slice is deliberately small:

- D3Q19 directions and opposites;
- coarse volume `1`, fine volume `1/8`;
- 8-child coalesce and uniform explode primitives;
- split/coalesce weights for face and edge crossings;
- explicit empty corner route set, because D3Q19 has no body-diagonal
  populations;
- topology canaries first, then one fixed-patch channel macro-flow before any
  sphere or open-boundary claim.

Validated canaries:

- `test/test_conservative_tree_3d.jl`
- `test/test_conservative_tree_topology_3d.jl`
- `test/test_conservative_tree_streaming_3d.jl`
- D3Q19 integer accessors match the lattice constants;
- 8-child coalesce preserves every oriented population, mass and momentum;
- uniform explode followed by coalesce conserves the parent;
- integrated D3Q19 equilibrium preserves volume-weighted mass and momentum;
- fixed ratio-2 3D patch allocation uses active fine state and parent shadow
  ledger arrays with D3Q19 layout;
- active 3D topology stores inactive coarse ledger cells under the patch and
  active fine children with volume `1/8`;
- route tables classify D3Q19 links as direct, coarse-to-fine, fine-to-coarse
  or boundary;
- face packets split over four boundary-adjacent children and coalesce from
  the same child set;
- edge packets split over two edge-adjacent children and coalesce from the
  same child set;
- every logical 3D link has route weights summing to one;
- route-native 3D interior streaming moves direct packets, splits
  coarse-to-fine face/edge packets, coalesces fine-to-coarse face/edge packets
  by accumulation, and conserves active D3Q19 population sums when boundary
  sources are zeroed;
- periodic-x 3D transport wraps coarse boundary packets, wraps coarse packets
  into a fine patch touching the periodic seam, and wraps fine boundary packets
  back to active coarse cells;
- periodic-x plus stationary wall-y/z 3D transport bounces coarse and fine wall
  packets, keeps periodic x wrapping, and gives y/z walls priority at corner
  exits;
- integrated D3Q19 BGK collision conserves mass and momentum, `omega=1`
  projects to equilibrium at conserved moments, and integrated D3Q19 Guo
  collision conserves mass while driving momentum in the force direction;
- a tiny fixed-patch 3D transport+BGK loop with periodic x and stationary
  wall-y/z conserves active mass;
- `run_conservative_tree_poiseuille_route_native_3d` runs a small fixed-patch
  D3Q19 channel with periodic x, stationary y/z walls and Guo forcing. At
  `steps=80`, the current canary gives mean velocity
  `ux=2.5897914250811675e-4`, transverse means below `4e-17`, and relative
  mass drift `1.08e-13`;
- the same runner now builds a dense D3Q19 leaf oracle. A full-domain refined
  patch matches that oracle exactly (`l2=0`, `linf=0` on the
  `4 x 4 x 3`, 20-step canary), proving the 3D route tables are leaf-equivalent
  when no coarse/fine interface remains;
- the default local patch is conservative but not profile-accurate against the
  dense leaf oracle yet: at 80 steps its profile relative `linf` error is
  `0.583` and the mean-velocity ratio is `0.389`. This is the 3D analogue of
  the non-subcycled interface limitation, so it is not a publication profile
  claim;
- `patch_strategy=:cross_section_buffered` refines the full y/z cross-section
  and leaves only x-direction coarse/fine interfaces. On the default domain it
  uses `2:7 x 1:8 x 1:6`, reduces the 80-step relative `linf` profile gap to
  `0.231`, and keeps mass drift at `1.41e-14`;
- a 3D solid-mask route canary now covers a solid cube fully contained in the
  refined patch under periodic-x/wall-yz routing. This gate currently asserts
  mass conservation and finite momentum only; it is not a 3D drag or sphere
  accuracy claim;
- the same runner has a short Float32 compile/run canary;
- all corner transfer calls reject, documenting the empty D3Q19 corner route
  set.

Exit gate:

- orientation-wise conservation for every D3Q19 route primitive;
- active mass and momentum agree before and after projection/restriction;
- fixed-patch 3D route-native channel smoke remains finite, accelerates in x
  only, conserves active mass to roundoff, and matches the dense leaf oracle
  exactly when the refined patch covers the full domain.

## Publication-P Milestone Gate

Currently publishable inside the D stream only as:

```text
2D, single fixed patch, route-native conservative AMR:
Couette + Poiseuille + square obstacle + VFS
interface-buffered cylinder ladder as static fixed-patch validation
BFS/open-channel short canaries, not long-horizon production BCs
3D, single fixed patch, route-native conservative AMR:
D3Q19 topology/streaming/collision + forced channel smoke
full-patch dense-leaf parity, but no local-patch 3D profile claim yet
```

The wording must not claim:

- arbitrary compact-patch obstacle convergence, because the compact default
  patch still exposes the non-subcycled interface error;
- robust long-horizon open boundaries, because only one-step/short smokes and
  the 1200-step BFS canary are covered;
- multi-patch route-native transport or adaptation, because only ownership and
  `.krk` setup helpers are started;
- subcycling;
- GPU AMR;
- 3D obstacle/sphere AMR.

Next commits should continue in this order:

1. reduce the local-patch 3D channel profile gap with subcycling or a matched
   coarse/fine-time oracle;
2. add paper-facing tables for coarse Cartesian vs AMR vs leaf oracle;
3. keep compact-patch cylinder as the subcycling stress test;
4. route tests over multi-patch ownership tables;
5. wire the subcycling ledger into the time integrator;
6. GPU packing parity canaries only after the CPU D publication gate.
