---
module: geometry
path: src/geometry/
owner_concern: geometry
status: implemented
last_verified: 2026-05-31
depends_on:
  - lbm
  - io-krk
---

# geometry — module implication map

`src/geometry/` is the **obstacle-realization layer**: it turns a surface
description (an STL file, or a `.krk` `Obstacle`/`Fluid` region with an STL or a
condition expression) into the discrete fields the LBM core consumes — a boolean
`is_solid` mask and a per-link cut-fraction array `q_wall` for LI-BB walls — plus
a small read-only `GeometryDescriptor` summary for the units audit. It is **not a
Julia submodule**: `geometry/Geometry.jl` is a flat `include` chain
(`stl_reader → voxelizer → stl_cut_fraction → mask_apply → libb_precompute →
descriptor`) compiled directly into the `Kraken` top-level by
`src/KrakeNN.jl`. Everything here runs once at setup time, never inside the time
loop.

## Public surface

The module is included flat into `Kraken`; the descriptor names are `export`ed,
the rest are de-facto public `KrakeNN.<name>` helpers called by drivers / the
runner.

- `read_stl(filename; T=Float64) -> STLMesh{T}` — zero-dependency STL loader
  (auto-detects binary vs ASCII via `_is_ascii_stl`). Builds triangles + bbox.
- `transform_mesh(mesh; scale=1.0, translate=(0,0,0)) -> STLMesh{T}` — affine
  scale-then-translate of a mesh and its bbox (used to place STL in LU coords).
- `STLMesh{T}` / `STLTriangle{T}` — the mesh container (triangles + `bbox_min` /
  `bbox_max`) and a single normal+3-vertex facet.
- `voxelize_2d(mesh, Nx, Ny, dx, dy; z_slice=0.0) -> Matrix{Bool}` — z-slice the
  mesh to a contour, then 2D point-in-polygon (+x ray casting) per cell center.
- `voxelize_3d(mesh, Nx, Ny, Nz, dx, dy, dz) -> Array{Bool,3}` — Möller–Trumbore
  +z ray casting per `(i,j)` column; requires a closed (watertight) surface.
- `precompute_q_wall_from_stl_2d(mesh, Nx, Ny, dx, dy; z_slice, FT, sub_cell=true)
  -> (q_wall::(Nx,Ny,9), is_solid)` — LI-BB cut-fraction field; `sub_cell=true`
  gives Bouzidi sub-cell `q_w` via ray-segment intersection, else `q_w=0.5`.
- `precompute_q_wall_from_stl_3d(mesh, Nx, Ny, Nz, dx, dy, dz; FT, sub_cell=false)
  -> (q_wall::(Nx,Ny,Nz,19), is_solid)` — 3D version; sub-cell `q_w` via
  Möller–Trumbore against every triangle (default halfway, `q_w=0.5`).
- `GeometryDescriptor` (exported) — immutable summary `(type::Symbol, blockage,
  q_wall_dist, stl_hash::UInt64, is_solid)` consumed read-only by the units
  audit. NOTE: a *separate* duck-typed `GeometryDescriptor` also lives in
  `src/units/Units.jl`; this one is the geometry-module realization.
- `build_geometry_descriptor(type, is_solid; q_wall_dist, stl_hash, blockage)`
  (and a kwargs-only method) — constructs the descriptor; auto-computes blockage
  as `count(is_solid)/length(is_solid)` when not given.
- `stl_kappa_max(mesh_lu) -> Float64` — curvature proxy `1 / (0.5·min bbox span)`
  on the LU-scaled mesh; feeds the dimensionless `R_LU·kappa_max` audit gate.
- `obstacle_extents_in_R(mask, R_LU; flow_axis=1) -> (L_up, L_down)` — solid
  centroid up/down-stream extents in units of `R_LU` along `flow_axis`.
- `halfway_wall_distances(mask) -> Vector{Float64}` — flat list of `0.5` entries,
  one per fluid→solid cut link (the halfway-BB `q_wall` histogram).
- Internal-but-load-bearing helpers driven by the runner: `_apply_geometry!`,
  `_apply_geometry_3d!`, `_apply_patch_geometry!`, `_apply_patch_geometry_3d!`
  (`mask_apply.jl`); `_has_stl_libb_obstacle`, `_precompute_stl_libb_q_wall_2d`,
  `_precompute_stl_libb_q_wall_3d`, `_halfway_q_wall_from_mask_2d`
  (`libb_precompute.jl`).

## Reads from

- `lbm` (`src/lattice/`) — the D2Q9 / D3Q19 lattice topology
  (`velocities_x`/`velocities_y`/`velocities_z`, `D2Q9()`, `D3Q19()`). Read
  read-only in `stl_cut_fraction.jl` and `libb_precompute.jl` to walk the `q`
  link stencil (`q in 2:9` / `q in 2:19`) when classifying cut links.
- `io-krk` (`src/io/`) — the parsed setup: `SimulationSetup` and its `domain` /
  `regions` (each region's `kind`, `stl::STLSource`, `condition`, `bc_type`), and
  the expression evaluator `evaluate(condition; x,y,z,Lx,…)`. `mask_apply.jl` and
  `libb_precompute.jl` read these to realize masks from `.krk` regions.
- `refinement` (`src/refinement/`) — `RefinementPatch{T}` / `RefinementPatch3D{T}`
  (their `Nx/Ny/Nz`, `n_ghost`, `dx`, `x_min/y_min/z_min`, `is_solid`). Read
  read-only by `_apply_patch_geometry!` / `_apply_patch_geometry_3d!` to evaluate
  obstacle conditions at a patch's native resolution. (Not in `depends_on` as a
  ship-plan §2.2 slug; recorded here because the patch helpers consume it.)
- STL files on disk are read via `Base.read` / `readlines` (an external input,
  not a sibling module).

## Writes to

- **Returns / produces new arrays**: `read_stl`/`transform_mesh` return fresh
  `STLMesh`; `voxelize_2d/3d` return new `Bool` masks; `precompute_q_wall_from_stl_2d/3d`
  and the `*_libb_q_wall_*` helpers allocate and return new `q_wall` (and a mask).
  `build_geometry_descriptor` returns an immutable `GeometryDescriptor`.
- **Mutates caller-owned destination arrays in place**: `_apply_geometry!` /
  `_apply_geometry_3d!` `copyto!` into the caller's `is_solid` (after building a
  host `solid_cpu` scratch); `_apply_patch_geometry!` / `_apply_patch_geometry_3d!`
  `copyto!` into `patch.is_solid`. These are the blast-radius surfaces — the
  mask they overwrite drives every wall BC downstream.
- **Throws, never silently degrades** (see Failure modes): `ArgumentError` on a
  truncated STL, zero-radius mesh, non-obstacle LI-BB region, mask/domain size
  mismatch, or zero cut links found.
- **Mutates no module-global registry** and **writes no files**. The
  `STL_AUDIT_CACHE` memo is owned by the `units` module, not here.
- **Mutates NOTHING in the time loop**: every function is a setup-time builder;
  the produced `is_solid`/`q_wall` are read (not re-written) by the kernels.

## Backend constraints

- **Setup-time / host-only.** All voxelization and cut-fraction precompute is
  scalar host code (`Array{Bool}`, plain loops). It runs once before the time
  loop; nothing here is a KernelAbstractions kernel and nothing enters a GPU
  kernel. The produced fields are then `copyto!`'d / uploaded by the caller to
  the chosen backend.
- **No per-step cost**, but **non-trivial one-time cost**: 2D voxelize is
  `O(N_cells · N_triangles)` (ray vs every facet), 3D sub-cell `q_w` is
  `O(N_fluid_boundary · N_triangles)` with **no BVH/acceleration** — fine for
  thousands of triangles, a setup bottleneck for large meshes (flagged as future
  work in `stl_cut_fraction.jl`).
- **Float-type is parametric and respected**: `read_stl`/voxelizers/precompute
  carry `T`/`FT` so a Float32 run gets Float32 masks/`q_wall`. CAVEAT: binary
  STL payloads are physically `Float32` on disk (`reinterpret(Float32, …)`),
  widened to `T` on read — a Float64 run does not recover precision the file
  never had.
- **`@inbounds` hot loops**: the precompute loops use `@inbounds` on
  hand-checked bounds; an off-by-one in the stencil walk would read out of
  bounds silently.

## Failure modes

The module is young (M1-era net-new dir, `status: implemented`) so most receipts
are structural footguns rather than postmortem'd missions; cite them before
trusting a mask.

- **Closed-surface assumption (silent wrong mask).** `voxelize_3d` parity
  counting and `voxelize_2d` odd-crossing both REQUIRE a watertight surface; an
  open/leaky STL produces a plausible-but-wrong `is_solid` with no error. The
  z-ray dedup (`_deduplicate_hits!`, `tol=1e-10`) only patches shared
  edges/vertices, not genuine holes.
- **`q_w = 0.5` collapses LI-BB to halfway-BB and hides cut-link bugs** —
  directly the [smoke-must-exercise-cut-links] receipt. `sub_cell=false` (3D
  default) and any closed-box smoke where `q_wall=0.5` algebraically reduce
  Bouzidi-FL to halfway-BB; validate cut-fraction code on a cylinder/sphere with
  genuine sub-0.5 `q_w`, not a box.
- **2D vs 3D `q_wall` merge rule differs.** `_precompute_stl_libb_q_wall_2d`
  overwrites the halfway seed with the STL `q_w` only where *both* are >0;
  `_precompute_stl_libb_q_wall_3d` starts from zero and keeps the **minimum**
  positive `q_w` across overlapping regions. Mixing conventions, or assuming the
  2D rule in 3D, will mis-set boundary links.
- **LI-BB only on STL Obstacle regions.** `wall=libb` on a `:fluid` region or a
  condition-expression region throws `ArgumentError("wall=libb is only supported
  on STL Obstacle regions")`; a `.krk` with `wall=libb` and no cut links throws
  `"no fluid-solid cut links were found"`.
- **STL voxelization is NOT done on refinement patches.** `_apply_patch_geometry!`
  / `_apply_patch_geometry_3d!` skip any region whose `stl !== nothing`
  (`# TODO: STL voxelization on patches`). An STL obstacle inside a refined patch
  is silently absent at the fine level — a real footgun for AMR + STL geometry.
- **Cell-center convention is load-bearing and must match everywhere.** All
  classifiers use cell centers `((i-0.5)·dx, …)`; the patch helpers offset by
  `n_ghost` (`(if_ - ng - 0.5)·dx`). A frame mismatch (1-LU offset) silently
  shifts the mask — the geometry analogue of the [wall-ring-idx-frame] +0.5-LU
  bias that biased Cd by +24%.
- **`stl_kappa_max` uses the smallest *positive* bbox span**: a degenerate /
  near-planar slice (one span ≈ 0) is filtered out, but a genuinely zero-volume
  mesh throws `ArgumentError("STL mesh has zero effective radius")`. The proxy is
  exact only for spheres/cylinders, conservative otherwise.
- **Binary STL bounds**: a truncated file (`< 84` bytes, or shorter than
  `84 + 50·ntri`) throws rather than reading garbage — but the `ntri` count is
  trusted from the header, so a corrupt count over-allocates.

## Touch order

For a suspected geometry bug (wrong obstacle shape, missing/extra solid cells,
bad `q_wall`, an STL that won't load), inspect in this order:

1. `src/geometry/mask_apply.jl` — `_apply_geometry!` / `_apply_geometry_3d!` /
   the `_apply_patch_*` helpers. 80% of "obstacle is wrong / missing" bugs are
   the region→mask realization here (fluid-vs-obstacle `kind`, the patch STL
   `TODO` skip, the cell-center/ghost offset).
2. `src/geometry/voxelizer.jl` — if the mask itself is wrong shape: the 2D/3D
   ray-casting (`_ray_triangle_z`, `_ray_crosses_segment_x`, `_slice_mesh_z`,
   `_deduplicate_hits!`). Suspect the watertight assumption first.
3. `src/geometry/stl_cut_fraction.jl` — if `is_solid` is right but `q_wall` /
   LI-BB is wrong: `precompute_q_wall_from_stl_2d/3d` and the ray-intersection
   `t`-parameter (`_ray_seg_intersect_t`, `_ray_tri_intersect_t`); check the
   `q_w = t` link-length assumption and the `q in 2:9 / 2:19` stencil.
4. `src/geometry/libb_precompute.jl` — if multiple regions / merge is wrong:
   `_precompute_stl_libb_q_wall_2d` vs `_3d` (the differing overwrite-vs-minimum
   merge rule), `_halfway_q_wall_from_mask_2d`, and the no-cut-links guard.
5. `src/geometry/stl_reader.jl` — only for load/parse problems: binary vs ASCII
   detection (`_is_ascii_stl`), the `Float32` reinterpret, `transform_mesh`
   scale/translate placement, and the bbox accumulation.
6. `src/geometry/descriptor.jl` — for a wrong units-audit summary: `blockage`
   auto-count, `stl_kappa_max`, `obstacle_extents_in_R`, `halfway_wall_distances`
   (and confirm you are looking at this `GeometryDescriptor`, not the duck-typed
   one in `src/units/Units.jl`).
