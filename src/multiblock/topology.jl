# =====================================================================
# Multi-block structured topology (v0.3).
#
# Philosophy: imperative modular. No clever macros or dispatch trees at
# the user-facing level; everything is plain structs + functions a user
# can trace in a debugger. The fused-kernel DSL (src/kernels/dsl/) stays
# for the inner loop; THIS layer is for assembling blocks and wiring
# their interfaces.
#
# A MultiBlockMesh2D is a flat list of `Block`s + a flat list of
# `Interface`s between them. Each `Block` carries:
#   - a self-contained `CurvilinearMesh` (already curvilinear, or
#     Cartesian-equivalent via `cartesian_mesh`),
#   - a 4-tuple of edge tags (:west, :east, :south, :north) that are
#     either a user-chosen physical name (:inlet, :cylinder, ...) or
#     the reserved symbol :interface — meaning "this edge is stitched
#     to another block via an `Interface`".
#
# An `Interface` says: block A edge X talks to block B edge Y. MVP
# restricts orientation to "same direction, opposite normals" so the
# copy is a straight memcpy along the shared edge. Non-trivial
# orientations (rotated, flipped) are a follow-up.
#
# Exchange (streaming across the interface) is done by a dedicated
# ghost-copy kernel (see exchange.jl); this file just declares the
# topology. Sanity checks live in sanity.jl.
# =====================================================================

"""
    EDGE_SYMBOLS_2D

The four canonical edge names on a 2D logical block. They correspond
to the standard ξ/η directions of a `CurvilinearMesh`:

| symbol  | lattice index          | outward normal (logical) |
|---------|------------------------|--------------------------|
| :west   | i = 1                  | −ξ                       |
| :east   | i = Nξ                 | +ξ                       |
| :south  | j = 1                  | −η                       |
| :north  | j = Nη                 | +η                       |

These are the only values allowed in a `Block`'s `boundary_tags` keys
and in an `Interface` endpoint's edge slot.
"""
const EDGE_SYMBOLS_2D = (:west, :east, :south, :north)

"""
    INTERFACE_TAG

Reserved tag inside a `Block.boundary_tags` NamedTuple. An edge whose
tag equals `INTERFACE_TAG` is expected to appear exactly once in the
`interfaces` list of the enclosing `MultiBlockMesh2D` (caught by the
sanity check). Any other `Symbol` is treated as a user-chosen physical
boundary name (e.g. `:inlet`, `:cylinder`, `:wall`).
"""
const INTERFACE_TAG = :interface

"""
    Block{T, AT}

One structured block in a multi-block mesh. Wraps:

- `id::Symbol`       — unique name used to reference this block in
                       interface declarations and diagnostics. The
                       sanity check rejects duplicates.
- `mesh::CurvilinearMesh{T, AT}` — the logical `Nξ × Nη` grid and its
                       metric. Any KrakeNN-compatible curvilinear mesh
                       works (uniform Cartesian, polar, gmsh-imported).
- `boundary_tags::NamedTuple` — four keys `(:west, :east, :south, :north)`,
                       each mapped to a `Symbol` that is either a
                       physical BC name or `INTERFACE_TAG`. A block
                       that touches nothing has no `:interface` tag;
                       a block with all four edges interior has all
                       four tags equal to `INTERFACE_TAG`.

Construct with `Block(id, mesh; west=..., east=..., south=..., north=...)`;
the kwarg form enforces that the four edges are always named explicitly
so there is no silent default.
"""
struct Block{T<:AbstractFloat, AT}
    id::Symbol
    mesh::CurvilinearMesh{T, AT}
    boundary_tags::NamedTuple{(:west, :east, :south, :north), NTuple{4, Symbol}}
end

function Block(id::Symbol, mesh::CurvilinearMesh{T, AT};
               west::Symbol, east::Symbol,
               south::Symbol, north::Symbol) where {T, AT}
    tags = (; west=west, east=east, south=south, north=north)
    return Block{T, AT}(id, mesh, tags)
end

"""
    Interface(from=(:block_a, :east), to=(:block_b, :west))

Declares that edge `east` of block `:block_a` is stitched to edge
`west` of block `:block_b`. MVP enforces:

- both edges have the same length in lattice units (Nξ or Nη match);
- both tags in the corresponding `boundary_tags` are `INTERFACE_TAG`;
- the local cell metric on the two edges is "consistent" — same dx
  within a tolerance, same physical coordinates at the shared edge.

Orientation: MVP supports only the trivial case where the two edges
run in the same local direction so the copy is index-aligned. Non-
trivial orientations (flip, 90° rotation) are tracked as a follow-up.

Both `from` and `to` endpoints are stored as 2-tuples `(block_id, edge_symbol)`.
"""
struct Interface
    from::Tuple{Symbol, Symbol}
    to::Tuple{Symbol, Symbol}
end

function Interface(; from::Tuple{Symbol, Symbol}, to::Tuple{Symbol, Symbol})
    return Interface(from, to)
end

"""
    MultiBlockMesh2D{T, AT}

A flat collection of `Block`s + a flat list of `Interface`s. Carries a
precomputed `block_by_id::Dict{Symbol, Int}` so the rest of the solver
can look up blocks by name in O(1) without scanning the vector on
every kernel call.

Constructor: `MultiBlockMesh2D(blocks; interfaces=Interface[])`. Does
NOT run sanity checks at construction — call `sanity_check_multiblock`
explicitly before launching a simulation (matches the v0.2 pattern in
`src/sanity_check.jl`).

The type parameters `T, AT` are inherited from the first block; the
sanity check verifies that all blocks share the same element type so
the kernels can run in one homogeneous compiled path.
"""
struct MultiBlockMesh2D{T<:AbstractFloat, AT}
    blocks::Vector{Block{T, AT}}
    interfaces::Vector{Interface}
    block_by_id::Dict{Symbol, Int}
end

function MultiBlockMesh2D(blocks::Vector{<:Block};
                           interfaces::Vector{Interface}=Interface[])
    isempty(blocks) && error("MultiBlockMesh2D: need at least one block")
    T  = typeof(blocks[1]).parameters[1]
    AT = typeof(blocks[1]).parameters[2]
    typed_blocks = Block{T, AT}[b for b in blocks]
    ix = Dict{Symbol, Int}()
    for (k, b) in enumerate(typed_blocks)
        ix[b.id] = k     # duplicates are flagged by the sanity check, not here
    end
    return MultiBlockMesh2D{T, AT}(typed_blocks, copy(interfaces), ix)
end

"""
    getblock(mbm::MultiBlockMesh2D, id::Symbol) -> Block

Look up a block by its `id`. Throws a `KeyError` with a list of valid
ids if the id is not found (so a typo in an `Interface` declaration is
caught with a useful message instead of a silent hash miss).
"""
function getblock(mbm::MultiBlockMesh2D, id::Symbol)
    idx = get(mbm.block_by_id, id, 0)
    idx == 0 && throw(KeyError("block id $id not found; known ids = $(keys(mbm.block_by_id))"))
    return mbm.blocks[idx]
end

"""
    edge_length(block::Block, edge::Symbol) -> Int

Number of nodes along `edge`. Returns `block.mesh.Nη` for west/east
(the edge runs along η) and `block.mesh.Nξ` for south/north (the edge
runs along ξ). Throws on an unknown edge symbol.
"""
function edge_length(block::Block, edge::Symbol)
    if edge === :west || edge === :east
        return block.mesh.Nη
    elseif edge === :south || edge === :north
        return block.mesh.Nξ
    else
        error("edge_length: unknown edge $edge; valid = $EDGE_SYMBOLS_2D")
    end
end

"""
    edge_coords(block::Block, edge::Symbol) -> (xs, ys)

Physical `(X, Y)` coordinates of the nodes lying on `edge`, in the
direction the edge is naturally parameterised (west/east along j,
south/north along i). Used by the sanity check to verify that two
interface edges are physically colocated.
"""
function edge_coords(block::Block, edge::Symbol)
    mesh = block.mesh
    if edge === :west
        xs = [mesh.X[1, j] for j in 1:mesh.Nη]
        ys = [mesh.Y[1, j] for j in 1:mesh.Nη]
    elseif edge === :east
        xs = [mesh.X[mesh.Nξ, j] for j in 1:mesh.Nη]
        ys = [mesh.Y[mesh.Nξ, j] for j in 1:mesh.Nη]
    elseif edge === :south
        xs = [mesh.X[i, 1] for i in 1:mesh.Nξ]
        ys = [mesh.Y[i, 1] for i in 1:mesh.Nξ]
    elseif edge === :north
        xs = [mesh.X[i, mesh.Nη] for i in 1:mesh.Nξ]
        ys = [mesh.Y[i, mesh.Nη] for i in 1:mesh.Nξ]
    else
        error("edge_coords: unknown edge $edge")
    end
    return xs, ys
end
