# AMR D Multilevel Audit

Date: 2026-05-05

Scope: conservative-tree route-native AMR D, with fixed refinement decided from
`.krk` files. This audit covers the modifications required to move from the
current 0/1-level path to true nested 2D multilevel, then the corresponding 3D
extension.

## Verdict

Multilevel D is not a small extension of the current runners. The projection
operators are already useful, and the route-table idea is the right direction,
but the runtime data model is still specialized to:

- one base array, `coarse_F`;
- one ratio-2 patch, `ConservativeTreePatch2D`;
- active leaves only at level 0 or level 1;
- masks and boundary closures expressed on a uniform `2*Nx x 2*Ny` leaf grid.

Therefore the correct next step is a new multilevel conservative-tree core
around active leaf cell ids, not a recursive extension of
`ConservativeTreePatch2D`.

The existing rejection of nested `.krk` `Refine parent` blocks must remain in
place until the multilevel topology, routing, and rest-state tests are green.

## Current Code Facts

### 2D topology is single-patch level 0/1

Relevant code:

- `src/refinement/conservative_tree_topology_2d.jl:101` defines
  `_inside_fine_patch(i, j, patch)` from one `ConservativeTreePatch2D`.
- `src/refinement/conservative_tree_topology_2d.jl:109` defines
  `_inside_leaf_domain(i, j, Nx, Ny)` as `2*Nx x 2*Ny`.
- `src/refinement/conservative_tree_topology_2d.jl:257` builds base cells and
  one set of level-1 children.
- `src/refinement/conservative_tree_topology_2d.jl:291` builds links by
  branching on `cell.level == 0` versus level 1.
- `src/refinement/conservative_tree_topology_2d.jl:411` exposes
  `create_conservative_tree_topology_2d(Nx, Ny, patch)`, with a single patch
  argument.

The route categories are already useful:

- `SAME_LEVEL`, `COARSE_TO_FINE`, `FINE_TO_COARSE`, `BOUNDARY`;
- `DIRECT`, `SPLIT_FACE`, `SPLIT_CORNER`, `COALESCE_FACE`,
  `COALESCE_CORNER`, `ROUTE_BOUNDARY`.

But they are currently generated only for adjacent levels 0 and 1 under one
patch.

### 2D state access is single-patch

Relevant code:

- `src/refinement/conservative_tree_2d.jl:1046` defines
  `ConservativeTreePatch2D` as `parent_i_range`, `parent_j_range`, `fine_F`,
  `coarse_shadow_F`.
- `src/refinement/conservative_tree_2d.jl:1064` rejects `ratio != 2`.
- `src/refinement/conservative_tree_2d.jl:1078` checks that `fine_F` is exactly
  `(2*nx_parent, 2*ny_parent, 9)`.
- `src/refinement/conservative_tree_streaming_2d.jl:8` reads populations from
  either `coarse_F` or one `patch.fine_F`.
- `src/refinement/conservative_tree_streaming_2d.jl:21` writes populations to
  either `coarse_F` or one `patch.fine_F`.
- `src/refinement/conservative_tree_streaming_2d.jl:48` rejects any active cell
  with `level > 1`.

This means the storage API cannot address:

- two disjoint refined patches in one stream;
- a child patch inside a level-1 patch;
- a level-2 or level-3 active leaf;
- a packed sparse pool of active leaves.

### Multi-patch ownership exists but is not runtime routing

Relevant code:

- `src/refinement/conservative_tree_multipatch_2d.jl:1` states that the layer is
  ownership-only.
- `src/refinement/conservative_tree_multipatch_2d.jl:36` builds parent and leaf
  owners for disjoint base-level patches.
- `src/refinement/conservative_tree_multipatch_2d.jl:46` rejects overlapping
  parent cells.
- `src/refinement/conservative_tree_multipatch_2d.jl:180` rejects nested
  `.krk` `Refine parent` blocks.

This is good as a first DSL canary, but it is not enough for multilevel. Nested
refinement needs parent-child ownership, not just disjoint base ownership.

### 3D has the same structural limitation

Relevant code:

- `src/refinement/conservative_tree_3d.jl:186` defines
  `ConservativeTreePatch3D`.
- `src/refinement/conservative_tree_3d.jl:206` rejects `ratio != 2`.
- `src/refinement/conservative_tree_3d.jl:228` checks `fine_F` as
  `(2*nx_parent, 2*ny_parent, 2*nz_parent, 19)`.
- `src/refinement/conservative_tree_streaming_3d.jl:58` checks topology layout.
- `src/refinement/conservative_tree_streaming_3d.jl:84` rejects levels above 1.

The 3D code is valuable as a one-patch prototype, but it should follow the 2D
multilevel design after the 2D gates are green.

## Required Changes By Layer

### 1. Tree specification

Add a pure static tree representation, separate from arrays:

```julia
struct ConservativeTreeCellKey2D
    level::Int
    i::Int
    j::Int
end

struct ConservativeTreeSpec2D
    Nx::Int
    Ny::Int
    max_level::Int
    cells::Vector{ConservativeTreeCell2D}
    key_to_id::Dict{Tuple{Int,Int,Int},Int}
    parent::Vector{Int}
    children::Vector{NTuple{4,Int}}
    active_leaf::Vector{UInt8}
end
```

Exact field names can change, but the core requirement is stable:

- every cell has a key `(level, i, j)`;
- coordinates are expressed in that level's own lattice;
- each non-leaf parent can have exactly 4 children in 2D;
- active cells are leaves only;
- inactive parent cells remain in the tree as conservative ledgers;
- adjacent active leaves may differ by at most one level.

The same pattern extends to 3D with 8 children and `D3Q19`.

### 2. `.krk` nested refine builder

Current DSL parsing can already read nested refine blocks, but the
conservative-tree helper rejects them. Keep that rejection until the new builder
is ready.

The new builder must validate:

- every `Refine` has `ratio = 2`;
- a root refine covers base-level cells;
- a child refine has a valid `parent`;
- child region is inside its parent region after snapping to cell boundaries;
- same-level siblings do not overlap;
- overlap is allowed only as child-inside-parent;
- all active leaf adjacencies satisfy 2:1 balance;
- invalid level jumps are rejected before topology construction.

Required tests:

- valid two-level child inside parent;
- valid four-level cylinder nesting canary;
- invalid sibling overlap;
- invalid child outside parent;
- invalid missing parent name;
- invalid level jump or unbalanced adjacency.

### 3. Active-leaf state storage

Replace runtime access through `coarse_F + patch.fine_F` with a storage backend
that addresses cells by cell id.

Recommended CPU reference first:

```julia
struct ConservativeTreeState2D{T}
    F_in::Matrix{T}      # n_cells x 9, or n_active_plus_ledgers x 9
    F_out::Matrix{T}
    rho::Vector{T}
    ux::Vector{T}
    uy::Vector{T}
end
```

Rules:

- stream/collide loops address `cell_id`, not `(coarse_F, patch)`;
- active leaves are the only cells streamed and collided;
- parent ledger cells can store restricted populations for diagnostics,
  regridding, or future adaptation;
- layout remains CPU-simple first, then can be packed into blocks later;
- no GPU-specific layout should be chosen before correctness gates pass.

The old `ConservativeTreePatch2D` should become a compatibility adapter for
single-level tests, not the main multilevel state model.

### 4. Generalized route table

Build routes for every active leaf and every D2Q9 population.

For source cell `(level, i, j)` and direction `q`, the neighbor can be:

- same level: direct route;
- one level coarser: coalesce route;
- one level finer: split route;
- domain boundary: boundary route.

Routes across level jumps larger than one must not exist. If such a route would
be needed, the tree is invalid because 2:1 balance failed.

Route table requirements:

- `src::Int`, `dst::Int`, `q::Int`, `weight::T`, `kind::RouteKind`;
- boundary routes keep `dst = 0` or a boundary id;
- non-boundary outgoing weights for one source packet sum to 1;
- route building is allocation-tolerant;
- `stream!` itself should be allocation-free after the table exists.

Existing split/coalesce concepts can be reused, but the one-patch coordinate
helpers cannot.

### 5. Projection and restriction

The local conservative operators are reusable:

- `coalesce_F_2d!`;
- `explode_uniform_F_2d!`;
- the 3D equivalents.

What must change is orchestration:

- recursive coalesce from deepest leaves to parent ledgers;
- recursive explode from parent ledgers to newly created children;
- tree-to-tree regrid for adaptation or changed static layouts;
- tree-to-uniform conversion only for tests and plotting, not production.

Surgical tests:

- one parent to 4 children preserves all 9 population sums;
- 4 children to parent preserves all 9 population sums;
- parent to level 2 and back preserves sums to roundoff;
- level 4 nested block roundtrip preserves sums to roundoff;
- random populations, not only equilibrium.

### 6. Collision and forcing

Collision must loop on active leaf ids:

```julia
for cid in topology.active_cells
    collide_cell!(state, cid, omega_by_level[level(cid)])
end
```

For the D publication path, it is acceptable to keep same omega across levels
for the first frozen-tree validation if that matches the existing D behavior.
If level-dependent physical `dx` and subcycling are enabled, omega and forcing
must be level-aware and tested explicitly.

Required gates:

- rest state remains exact on nested trees;
- uniform velocity remains uniform on nested trees;
- body force Poiseuille is consistent between level 0, level 1, and nested
  level 2/3 patches;
- mass drift remains at roundoff for closed periodic cases.

### 7. Boundaries

Boundary closures must be expressed on active leaves, not on `coarse_F` or a
single fine patch.

Required boundary work:

- periodic x routes for all levels;
- wall closures for Couette/Poiseuille at any leaf level touching the wall;
- Zou-He inlet and pressure outlet only on level-0 boundaries at first;
- explicit rejection if an open boundary intersects refined leaves before the
  dedicated open-boundary tests are green.

This matches the safest strategy for BFS: keep the inlet/outlet on level 0
first, harden open-channel tests, then run the BFS macro-flow.

### 8. Solid masks and drag

Current solid masks are uniform level-1 rasters:

- 2D: `2*Nx x 2*Ny`;
- 3D: `2*Nx x 2*Ny x 2*Nz`.

Multilevel needs geometry queries by active leaf cell:

```julia
is_solid(cell_key, geometry)
cell_center(cell_key, domain)
cell_volume(level)
```

For the first publication-grade D path:

- square obstacle can use center-based solid classification;
- cylinder should force refinement around the body so no coarse active cell cuts
  the obstacle boundary;
- active coarse cells must still reject partial-solid classification;
- drag must be computed on active fluid-solid links, not on a uniform leaf
  raster.

Required drag tests:

- one packet bounce-back force on a known solid link;
- square obstacle with all boundary cells refined;
- cylinder mask reproducibility from `.krk`;
- drag on active-link MEA agrees with uniform Cartesian leaf reference on a
  small case before long runs.

### 9. Subcycling

Subcycling is the clean route for level-dependent time steps, but it should not
be introduced before static multilevel streaming and collision are correct.

Required order:

1. frozen tree, same global step, same omega;
2. level-aware omega/forcing canaries;
3. subcycling scheduler;
4. macro-flow reruns.

Subcycling constraints:

- adaptation or tree rebuild only between complete coarse steps;
- no topology rebuild between fine half steps;
- parent ledgers updated at synchronization points;
- open boundaries stay level 0 until tested.

### 10. 3D extension

Do not fork the design. After 2D gates are green, port the same abstractions:

- `ConservativeTreeSpec3D`;
- active-leaf state keyed by cell id;
- generalized D3Q19 route builder;
- 8-child coalesce/explode recursion;
- level-aware wall/periodic closures;
- 3D masks from geometry queries;
- sphere/cylinder-like canaries only after box and channel tests.

The current 3D one-patch path is useful for testing local 8-child
coalesce/explode and route categories, but not as the final storage model.

## Implementation Phases

### ML0 - static nested tree only

Goal: build a nested tree from programmatic specs and `.krk`, without running
LBM.

Deliverables:

- `ConservativeTreeSpec2D`;
- tree builder from named refinements;
- active volume exactness;
- parent/child tables;
- 2:1 balance validator;
- nested `.krk` canaries.

Gate:

- four-level cylinder nesting builds a valid tree;
- invalid nesting cases fail with precise errors;
- no stream/collide API is changed yet.

### ML1 - conservative projection recursion

Goal: prove that the multilevel tree can carry conservative populations.

Deliverables:

- recursive coalesce;
- recursive explode;
- tree-to-uniform oracle for tests and plots;
- random-population roundtrip tests.

Gate:

- population sums are preserved to roundoff on 2, 3, and 4 levels;
- tests cover every D2Q9 population independently.

### ML2 - route table without physics

Goal: route packets on active leaves only.

Deliverables:

- multilevel route builder;
- same/coarse/fine/boundary route classification;
- route-weight conservation tests.

Gate:

- single-packet tests pass for same-level, fine-to-coarse, coarse-to-fine,
  face, corner, and boundary cases;
- no generated route crosses more than one level.

### ML3 - stream rest-state

Goal: prove transport correctness before collision complexity.

Deliverables:

- `stream_multilevel_routes_F_2d!`;
- active-leaf state accessors;
- periodic rest-state tests;
- uniform velocity tests.

Gate:

- rest state unchanged to roundoff for nested levels 1, 2, 3, 4;
- closed periodic mass drift at roundoff.

### ML4 - collide and simple channels

Goal: recover channel physics.

Deliverables:

- BGK collide on active leaves;
- Guo forcing on active leaves;
- Couette and Poiseuille runners with nested static trees;
- `.krk` examples.

Gate:

- Couette profile matches Cartesian leaf reference;
- Poiseuille profile matches Cartesian leaf reference;
- efficiency report compares active leaves vs uniform leaf grid.

### ML5 - solid geometry and drag

Goal: make obstacle flows meaningful.

Deliverables:

- geometry-backed solid query;
- active-link bounce-back;
- active-link MEA drag;
- square and cylinder `.krk` cases;
- plot-friendly output.

Gate:

- square obstacle reproduces uniform Cartesian reference within documented
  tolerance;
- cylinder short run has finite, stable `Cd`;
- active coarse partial-solid cells are rejected unless fully classified.

### ML6 - BFS after open boundaries are hard

Goal: avoid debugging BFS on top of unvalidated outlet behavior.

Deliverables:

- open-channel surgical tests at level 0 boundaries;
- BFS with refinement away from inlet/outlet first;
- later optional refined internal step region.

Gate:

- open-channel mass drift and outlet stability documented;
- BFS runs without NaN and with reproducible macro diagnostics.

### ML7 - four-level cylinder publication canary

Goal: the first real multilevel D publication target.

Deliverables:

- four-level nested cylinder `.krk`;
- Cartesian leaf-equivalent reference;
- AMR D run;
- convergence and efficiency plots;
- result manifest.

Gate:

- reproducible local short run;
- Aqua long run for production data;
- `Cd`, `u/v` field errors, mass drift, active-cell count, and runtime recorded;
- no speedup claim unless measured against a fair Cartesian leaf-equivalent
  baseline.

### ML8 - 3D port

Goal: repeat the same ladder in 3D.

Deliverables:

- static nested tree 3D;
- projection recursion 3D;
- route streaming 3D;
- channel flow 3D;
- solid sphere/cylinder canaries.

Gate:

- 3D rest and channel tests pass before obstacle macro-flow tests.

## Files To Add Or Refactor

Likely new files:

- `src/refinement/conservative_tree_spec_2d.jl`;
- `src/refinement/conservative_tree_multilevel_state_2d.jl`;
- `src/refinement/conservative_tree_multilevel_topology_2d.jl`;
- `src/refinement/conservative_tree_multilevel_streaming_2d.jl`;
- `src/refinement/conservative_tree_geometry_2d.jl`;
- matching 3D files after 2D validation.

Likely modified files:

- `src/refinement/conservative_tree_multipatch_2d.jl` for `.krk` nested
  builder handoff;
- `src/refinement/conservative_tree_2d.jl` to reuse projection kernels and keep
  single-patch adapters;
- `src/refinement/conservative_tree_streaming_2d.jl` only for compatibility
  wrappers, not as the multilevel core;
- `src/KrakeNN.jl` exports;
- tests under `test/test_conservative_tree_*`.

## What Not To Do

- Do not recursively nest `ConservativeTreePatch2D`; it keeps the wrong storage
  abstraction.
- Do not enable nested `.krk` conservative-tree runtime dispatch before ML0-ML3
  are green.
- Do not use the older `RefinedDomain.parent_of` metadata as proof of a correct
  nested D runtime; its stepping path is patch/base oriented, not
  conservative-tree active-leaf routing.
- Do not let neighboring active leaves differ by more than one level.
- Do not debug BFS before open-channel boundaries are validated with surgical
  tests.
- Do not claim efficiency until active-cell AMR and Cartesian leaf-equivalent
  runs are measured with the same physical setup.

## Immediate Next Patch Set

The next implementation sequence should be:

1. Add `ConservativeTreeSpec2D` and programmatic nested tree builder.
2. Add `.krk` nested refine validation but keep runtime dispatch disabled.
3. Add ML0 tests for volume, parent/children, 2:1 balance, and invalid nesting.
4. Add recursive coalesce/explode tests on random populations.
5. Only then start the multilevel route table.

The first multilevel "done" milestone is not a cylinder `Cd`. It is a
four-level nested tree that can preserve rest-state transport to roundoff.
