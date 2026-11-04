# Multi-block structured meshes (v0.3)

Kraken's v0.3 milestone is a **multi-block structured LBM solver**: a
collection of logically-rectangular blocks stitched together at
interfaces, each block running its own curvilinear LBM step on the
GPU. The design target is to handle CFD-grade complex geometries —
body-fitted O-grids around bluff bodies, C-grids around airfoils —
that cannot be mapped onto a single transfinite patch.

This page describes the Phase A MVP: topology declaration, sanity
checking, ghost-layer state, and pre-step ghost exchange. Phase B
will add the physical-wall ghost-fill helper needed for bit-exact
equivalence to single-block over many steps, and Phase C will wire up
LI-BB per block for curved-body closures on multi-block meshes.

## Data model

Three types live in [src/multiblock/topology.jl](https://github.com/lauguimel/Kraken.jl/blob/main/src/multiblock/topology.jl):

- **`Block`** — an `id::Symbol`, a self-contained `CurvilinearMesh`,
  and a `NamedTuple` of four edge tags `(west, east, south, north)`.
  Each tag is either a user-chosen physical name (`:inlet`,
  `:cylinder`, `:wall`, …) or the reserved `INTERFACE_TAG =
  :interface`.

- **`Interface`** — two 2-tuples `(block_id, edge_symbol)` declaring
  that edge X of block A talks to edge Y of block B.

- **`MultiBlockMesh2D`** — a flat vector of blocks + a flat vector of
  interfaces, plus a precomputed `block_by_id::Dict` for O(1) name
  lookup.

Runtime state for one block lives in
[src/multiblock/state.jl](https://github.com/lauguimel/Kraken.jl/blob/main/src/multiblock/state.jl):

- **`BlockState2D`** — populations `f` + macroscopic fields
  (`ρ, ux, uy`) on an **extended grid** of size `(Nξ + 2·Ng, Nη + 2·Ng)`
  where `Ng` is the ghost-layer width (default 1 for D2Q9, 2 for
  D3Q19 equivalents). The physical interior occupies indices
  `(Ng + 1 .. Ng + Nξ, Ng + 1 .. Ng + Nη)`.
- **`allocate_block_state_2d(block; n_ghost=1, backend)`** — allocates
  the extended arrays on any Kraken-supported backend (CPU / CUDA /
  Metal) and initialises `ρ ≡ 1`, `ux = uy ≡ 0`, `f = 0`.
- **`interior_f(state)`** — `SubArray` view of the physical interior
  populations for user-facing initialisation, BC application, and
  diagnostic extraction.
- **`ext_dims(state)`** — extended `(Nξ_ext, Nη_ext)` for calling
  single-block step kernels with the right `Nx, Ny`.

The design is deliberately **imperative and modular**: no macros, no
hidden dispatch, everything is a plain struct + function a user can
trace in a debugger. The fused-kernel DSL lives at the per-block
inner-loop level ([src/kernels/dsl/](https://github.com/lauguimel/Kraken.jl/tree/main/src/kernels/dsl)); this
layer is purely for assembling blocks and coordinating their
interfaces.

```julia
mesh_A = cartesian_mesh(; x_min=0.0, x_max=0.5, y_min=0.0, y_max=1.0, Nx=11, Ny=11)
mesh_B = cartesian_mesh(; x_min=0.5, x_max=1.0, y_min=0.0, y_max=1.0, Nx=11, Ny=11)

blk_A = Block(:left,  mesh_A; west=:inlet,     east=:interface,
                               south=:wall,    north=:wall)
blk_B = Block(:right, mesh_B; west=:interface, east=:outlet,
                               south=:wall,    north=:wall)

mbm = MultiBlockMesh2D([blk_A, blk_B];
                        interfaces=[Interface(; from=(:left, :east),
                                                 to=(:right, :west))])

issues = sanity_check_multiblock(mbm)
any(iss -> iss.severity === :error, issues) && error(...)

states = [allocate_block_state_2d(b; n_ghost=1) for b in mbm.blocks]
# … user initialises interior_f(states[k]) with feq …
```

## Sanity invariants

[src/multiblock/sanity.jl](https://github.com/lauguimel/Kraken.jl/blob/main/src/multiblock/sanity.jl) implements
nine invariant families. They fire as `MultiBlockSanityIssue` records
with a severity (`:error` / `:warning`), a short `code` symbol, a
locating `(block_id, edge)` pair, and a human-readable message:

| Code | Catches |
|---|---|
| `NoDuplicateBlockIDs` | two blocks with the same `id` symbol |
| `AllEdgeTagsKnown` | a non-`Symbol` edge tag |
| `UniformElementType` | mixed-precision meshes (FP32 + FP64) |
| `InterfaceRefsExist` | an `Interface` points at an unknown block or edge |
| `InterfaceBothEdgesMarked` | one of the interface edges is tagged as a physical BC instead of `:interface` |
| `InterfaceEdgesSameLength` | `Nξ`/`Nη` mismatch on the two edges |
| `InterfaceEdgesColocated` | physical coordinates of the two edges differ by more than `tol` |
| `InterfaceOrientationTrivial` | edges run in opposite directions (flip not yet supported) |
| `InterfaceEveryMarkedEdgeUsed` | an edge tagged `:interface` is not referenced by any `Interface` |

The sanity check is called explicitly (not at construction) so the
user can inspect the full diagnostic list before deciding to fix or
proceed.

## Ghost-layer exchange (Phase A.5b)

Each block's populations live on an extended grid that includes `Ng`
ghost cells on every side. Before each timestep, `exchange_ghost_2d!`
walks the interface list and **pre-fills the ghost rows** from the
neighbour block's physical interior, one cell deep from the shared
interface.

For a west-east interface (A on left, `:east`; B on right, `:west`),
with `Ng = 1` and physical sizes `Nξ_A, Nξ_B`:

    A[Nξ_A + 2, :, :]  ←  B[2, :, :]    (A's east ghost  ← B's first interior col)
    B[1, :, :]         ←  A[Nξ_A + 1, :, :]   (B's west ghost ← A's last interior col)

Analogous for south-north with `j` instead of `i`. After this fill,
the step kernel reads from the ghost rows when its pull would cross
the interface — no halfway-BB fallback fires on a physical
interior-boundary cell, and the step at the interface column produces
**bit-exact** the same populations as a single-block step would. The
test [test_multiblock_canal.jl](https://github.com/lauguimel/Kraken.jl/blob/main/test/test_multiblock_canal.jl)
verifies this: after a one-step BGK on uniform-flow init, A's
interior-east column `f_A_out[Nξ_A + Ng, :, :]` and B's
interior-west column `f_B_out[Ng + 1, :, :]` are identical.

The exchange is `view(...) .= view(...)` broadcasts, so it works
transparently on CPU, CUDA, and Metal backends without a dedicated
KernelAbstractions kernel. Post-step, ghost-row contents in `f_out`
are garbage (the step kernel's halfway-BB fired at the extended
array's outer ghosts) but are irrelevant — the next iteration's
`exchange_ghost_2d!` overwrites them before the step reads them.

## Known limitation: physical-wall ghost fill (Phase A.5c, deferred)

The ghost-layer exchange handles the INTERFACE edges correctly, but
blocks with a **physical-wall edge** (e.g., `:wall`, `:inlet`,
`:outlet`) leave the ghost on that side filled with halfway-BB
garbage from the previous step. At step 2 and beyond, the interior
cell adjacent to a physical wall pulls from that garbage ghost, and
the resulting pollution walks one cell inward per step. After roughly
`N_phys / 2` steps the contamination reaches the far side of the
block.

The fix is a helper `fill_physical_wall_ghost_2d!(block, state)` that
pre-fills the physical-wall ghost rows with the halfway-BB reflection
of the adjacent interior cell (`ghost[1, j, q_in] ← interior[2, j, q_opp]`
for each incoming `q`). With that helper in place, multi-block
simulations with mixed interface + physical-wall BCs reproduce
single-block results bit-exactly over any number of steps. Phase A.5c.

## Importing multi-block meshes from gmsh (Phase B.1)

A multi-surface `.geo` or pre-meshed `.msh` file is loaded with

```julia
mbm, groups = load_gmsh_multiblock_2d("ogrid.geo";
                                         FT=Float64,
                                         layout=:topological)
```

implemented in
[src/multiblock/mesh_gmsh_multiblock.jl](https://github.com/lauguimel/Kraken.jl/blob/main/src/multiblock/mesh_gmsh_multiblock.jl).
The loader walks every 2D physical surface, builds one `Block` per
surface using the existing per-surface layout helpers
(`:axis_aligned` for rectangles, `:topological` for O-grids with arc
edges), and stitches them together:

- **Block id** — `Symbol(physical_surface_name)` when the surface has
  a named Physical group, else `Symbol("block_<gmsh_tag>")`.
- **Edge tags** — for each of the four logical edges `(:west, :east,
  :south, :north)`, the loader finds the bounding gmsh curve by
  matching curve endpoints to the block's corner nodes. The curve's
  Physical-group name (e.g. `Physical Curve("inlet")`) becomes the
  edge tag. Curves that are geometrically shared between two surfaces
  and carry no physical name default to `:interface`; curves whose
  Physical name appears in the loader's `interface_names` tuple
  (default `("interface",)`) are forced to `:interface`.
- **Interfaces** — every edge tagged `:interface` is paired with
  exactly one mate across a different block by spatial colocation of
  the edge-coordinate sequence (aligned OR reversed). Both shared-node
  (zero offset) and non-overlap (one-cell offset) topologies are
  accepted.

gmsh's natural output is **shared-node** (adjacent surfaces literally
share the same boundary curve, so the two blocks' edge nodes
coincide). The current `exchange_ghost_2d!` kernel expects
**non-overlap** (edge offset = one cell); on shared-node input the
colocation invariant reports a `:warning` and the caller must either
wire a shared-node exchange variant (Phase B.2) or shift one block's
coordinates to create the offset. Blocks loaded via the
`:topological` layout can also end up with flipped ξ/η orientation
between neighbours — in that case the sanity check
`InterfaceOrientationTrivial` flags which edges need to be reoriented
before a simulation can run.

## MVP validation

- [test/test_multiblock_topology.jl](https://github.com/lauguimel/Kraken.jl/blob/main/test/test_multiblock_topology.jl)
  (31 tests) — data-model invariants + sanity check happy/failure paths.
- [test/test_multiblock_exchange.jl](https://github.com/lauguimel/Kraken.jl/blob/main/test/test_multiblock_exchange.jl)
  (163 tests) — `BlockState2D` allocator and `interior_f` view;
  ghost-fill correctness on W-E and S-N interfaces for `Ng = 1, 2`;
  away-from-edge interior untouched; swapped `from/to`; error paths
  (length mismatch, mixed `n_ghost`, unsupported same-normal pair).
- [test/test_multiblock_canal.jl](https://github.com/lauguimel/Kraken.jl/blob/main/test/test_multiblock_canal.jl)
  (10 tests) — 2-block BGK canal with ghost-layer pipeline: `u = 0`
  uniform preserved over 10 steps, `u = 0.05` step-1 interface
  columns bit-exact, 1000-step smoke run stays finite, and 20-step
  bit-exact match vs single-block once the physical-wall ghost fill
  is layered in.
- [test/test_multiblock_gmsh.jl](https://github.com/lauguimel/Kraken.jl/blob/main/test/test_multiblock_gmsh.jl)
  (44 tests) — gmsh loader on an on-the-fly 2-block canal (`.geo`
  with shared interior curve) and on a 4-block O-grid around a
  cylinder (`:topological` layout, four 90° quadrants between an
  inner arc and an outer arc). Both exercises assert correct block
  count, edge tags, interface count, and (structural) sanity.
