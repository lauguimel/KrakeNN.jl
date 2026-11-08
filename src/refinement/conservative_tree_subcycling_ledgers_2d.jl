# Conservative subcycling ledgers for route-native D2Q9 AMR.
#
# This is not yet the full time integrator. It records the packet accounting
# needed for one coarse step and two fine half-steps so interface transfers can
# be tested before they are put in the hot loop.

"""
    ConservativeTreeSubcycleLedger2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeSubcycleLedger2D
```
"""
struct ConservativeTreeSubcycleLedger2D{T}
    ratio::Int
    coarse_to_fine::Array{T,4}
    fine_to_coarse::Matrix{T}
end

struct ConservativeTreeSubcycleEvent2D
    tick::Int
    phase::Symbol
    src_level::Int
    dst_level::Int
end

struct ConservativeTreeSubcycleSchedule2D
    max_level::Int
    ratio::Int
    finest_ticks::Int
    level_step_ticks::Vector{Int}
    events::Vector{ConservativeTreeSubcycleEvent2D}
end

struct ConservativeTreeSubcycleLedgerBank2D{T}
    schedule::ConservativeTreeSubcycleSchedule2D
    pair_ledgers::Vector{ConservativeTreeSubcycleLedger2D{T}}
end

struct ConservativeTreeSubcyclePackedLedgerPair2D{T}
    parent_ids::Vector{Int}
    coarse_to_fine::Array{T,5}
    fine_to_coarse::Array{T,3}
end

struct ConservativeTreeSubcycleRoutePacketCache2D{T}
    key_to_slot::Dict{Tuple{Int,Int},Int}
    dst_ids::Vector{Int}
    qs::Vector{Int}
    packets::Vector{T}
end

struct ConservativeTreeSubcycleSpatialLedgerBank2D{T}
    spec::ConservativeTreeSpec2D
    schedule::ConservativeTreeSubcycleSchedule2D
    ledger_pairs::Vector{ConservativeTreeSubcyclePackedLedgerPair2D{T}}
    parent_ledger_slot::Vector{Int}
    route_packet_caches::Vector{ConservativeTreeSubcycleRoutePacketCache2D{T}}
    route_packet_slot_by_route::Vector{Int}
    inactive_route_packet_slots_by_level::Vector{Matrix{Int}}
    refined_parent_ids_by_level::Vector{Vector{Int}}
    inactive_refined_ids_by_level::Vector{Vector{Int}}
end

"""
    create_conservative_tree_subcycle_ledger_2d(;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.create_conservative_tree_subcycle_ledger_2d)
```
"""
function create_conservative_tree_subcycle_ledger_2d(;
        T::Type{<:Real}=Float64,
        ratio::Integer=2)
    r = Int(ratio)
    r == 2 || throw(ArgumentError("conservative-tree subcycling currently requires ratio = 2"))
    return ConservativeTreeSubcycleLedger2D{T}(
        r, zeros(T, 2, 2, 9, r), zeros(T, 9, r))
end

function _check_conservative_tree_schedule_ratio_2d(ratio::Integer)
    r = Int(ratio)
    r >= 2 || throw(ArgumentError("subcycling ratio must be >= 2"))
    return r
end

function _conservative_tree_level_step_ticks_2d(max_level::Int, ratio::Int)
    return [ratio^(max_level - level) for level in 0:max_level]
end

function _push_conservative_tree_schedule_interval_2d!(
        events::Vector{ConservativeTreeSubcycleEvent2D},
        level::Int,
        max_level::Int,
        ratio::Int,
        tick_start::Int,
        tick_end::Int)
    if level == max_level
        push!(events, ConservativeTreeSubcycleEvent2D(
            tick_end, :advance, level, level))
        return events
    end

    child = level + 1
    push!(events, ConservativeTreeSubcycleEvent2D(
        tick_start, :sync_down, level, child))

    interval_ticks = tick_end - tick_start
    interval_ticks % ratio == 0 ||
        throw(ArgumentError("subcycle interval is not divisible by ratio"))
    child_ticks = div(interval_ticks, ratio)
    for substep in 1:ratio
        child_start = tick_start + (substep - 1) * child_ticks
        child_end = child_start + child_ticks
        _push_conservative_tree_schedule_interval_2d!(
            events, child, max_level, ratio, child_start, child_end)
    end

    push!(events, ConservativeTreeSubcycleEvent2D(
        tick_end, :sync_up, child, level))
    push!(events, ConservativeTreeSubcycleEvent2D(
        tick_end, :advance, level, level))
    return events
end

"""
    create_conservative_tree_subcycle_schedule_2d(max_level; ratio=2)

Build a level-agnostic recursive subcycling calendar for one level-0 coarse
step. Time is expressed in integer ticks of the finest level. For `ratio = 2`,
level `l` advances every `2^(max_level-l)` finest ticks.

The event order is recursive and deterministic:

1. `:sync_down` from a parent level to its child at the beginning of that
   parent interval;
2. all child sub-intervals;
3. `:sync_up` from child to parent at the synchronization point;
4. `:advance` of the parent level.

This object owns no populations and performs no physics. It is the dispatch
contract that the future route/reflux kernels must follow for any number of
levels.
"""
function create_conservative_tree_subcycle_schedule_2d(max_level::Integer;
                                                       ratio::Integer=2)
    ml = Int(max_level)
    ml >= 0 || throw(ArgumentError("max_level must be nonnegative"))
    r = _check_conservative_tree_schedule_ratio_2d(ratio)
    finest_ticks = r^ml
    level_step_ticks = _conservative_tree_level_step_ticks_2d(ml, r)
    events = ConservativeTreeSubcycleEvent2D[]
    _push_conservative_tree_schedule_interval_2d!(
        events, 0, ml, r, 0, finest_ticks)
    return ConservativeTreeSubcycleSchedule2D(
        ml, r, finest_ticks, level_step_ticks, events)
end

include("conservative_tree_subcycle_buffers_2d.jl")

function create_conservative_tree_subcycle_ledger_bank_2d(
        schedule::ConservativeTreeSubcycleSchedule2D;
        T::Type{<:Real}=Float64)
    pair_ledgers = ConservativeTreeSubcycleLedger2D{T}[
        create_conservative_tree_subcycle_ledger_2d(T=T, ratio=schedule.ratio)
        for _ in 1:schedule.max_level
    ]
    return ConservativeTreeSubcycleLedgerBank2D{T}(schedule, pair_ledgers)
end

function create_conservative_tree_subcycle_ledger_bank_2d(max_level::Integer;
                                                          ratio::Integer=2,
                                                          T::Type{<:Real}=Float64)
    schedule = create_conservative_tree_subcycle_schedule_2d(
        max_level; ratio=ratio)
    return create_conservative_tree_subcycle_ledger_bank_2d(schedule; T=T)
end

function _check_conservative_tree_subcycle_spec_schedule_2d(
        spec::ConservativeTreeSpec2D,
        schedule::ConservativeTreeSubcycleSchedule2D)
    spec.max_level == schedule.max_level ||
        throw(ArgumentError("subcycle schedule max_level must match the tree spec"))
    schedule.ratio == 2 ||
        throw(ArgumentError("conservative-tree spatial subcycling requires ratio = 2"))
    return nothing
end

function create_conservative_tree_subcycle_spatial_ledger_bank_2d(
        spec::ConservativeTreeSpec2D;
        schedule::ConservativeTreeSubcycleSchedule2D=
            create_conservative_tree_subcycle_schedule_2d(spec.max_level),
        T::Type{<:Real}=Float64)
    _check_conservative_tree_subcycle_spec_schedule_2d(spec, schedule)
    route_packet_caches = [
        ConservativeTreeSubcycleRoutePacketCache2D{T}(
            Dict{Tuple{Int,Int},Int}(), Int[], Int[], T[])
        for _ in 1:spec.max_level
    ]
    refined_parent_ids_by_level = [Int[] for _ in 1:spec.max_level]
    inactive_refined_ids_by_level = [Int[] for _ in 0:spec.max_level]
    parent_ledger_slot = zeros(Int, length(spec.cells))

    @inbounds for (cell_id, cell) in pairs(spec.cells)
        children = spec.children[cell_id]
        children == (0, 0, 0, 0) && continue
        push!(inactive_refined_ids_by_level[cell.level + 1], cell_id)
        cell.level < spec.max_level || continue
        push!(refined_parent_ids_by_level[cell.level + 1], cell_id)
    end

    ledger_pairs = Vector{ConservativeTreeSubcyclePackedLedgerPair2D{T}}(
        undef, spec.max_level)
    for parent_level in 0:(spec.max_level - 1)
        parent_ids = refined_parent_ids_by_level[parent_level + 1]
        nparents = length(parent_ids)
        @inbounds for (slot, parent_id) in enumerate(parent_ids)
            parent_ledger_slot[parent_id] = slot
        end
        ledger_pairs[parent_level + 1] =
            ConservativeTreeSubcyclePackedLedgerPair2D{T}(
                parent_ids,
                zeros(T, 2, 2, 9, schedule.ratio, nparents),
                zeros(T, 9, schedule.ratio, nparents))
    end

    return ConservativeTreeSubcycleSpatialLedgerBank2D{T}(
        spec, schedule, ledger_pairs, parent_ledger_slot,
        route_packet_caches, Int[], Matrix{Int}[],
        refined_parent_ids_by_level,
        inactive_refined_ids_by_level)
end

function conservative_tree_subcycle_events_at_tick_2d(
        schedule::ConservativeTreeSubcycleSchedule2D,
        tick::Integer)
    t = Int(tick)
    0 <= t <= schedule.finest_ticks ||
        throw(ArgumentError("tick is outside the schedule"))
    return [event for event in schedule.events if event.tick == t]
end

function conservative_tree_subcycle_advance_counts_2d(
        schedule::ConservativeTreeSubcycleSchedule2D)
    counts = zeros(Int, schedule.max_level + 1)
    @inbounds for event in schedule.events
        event.phase == :advance || continue
        counts[event.src_level + 1] += 1
    end
    return counts
end

function conservative_tree_subcycle_sync_counts_2d(
        schedule::ConservativeTreeSubcycleSchedule2D)
    counts = Dict{Tuple{Symbol,Int,Int},Int}()
    @inbounds for event in schedule.events
        event.phase == :advance && continue
        key = (event.phase, event.src_level, event.dst_level)
        counts[key] = get(counts, key, 0) + 1
    end
    return counts
end

function _check_conservative_tree_pair_level_2d(
        schedule::ConservativeTreeSubcycleSchedule2D,
        parent_level::Integer)
    parent = Int(parent_level)
    0 <= parent < schedule.max_level ||
        throw(ArgumentError("parent_level must identify an adjacent level pair"))
    return parent
end

function conservative_tree_subcycle_pair_ledger_2d(
        bank::ConservativeTreeSubcycleLedgerBank2D,
        parent_level::Integer)
    parent = _check_conservative_tree_pair_level_2d(
        bank.schedule, parent_level)
    return bank.pair_ledgers[parent + 1]
end

function reset_conservative_tree_subcycle_bank_2d!(
        bank::ConservativeTreeSubcycleLedgerBank2D)
    for ledger in bank.pair_ledgers
        reset_conservative_tree_subcycle_ledger_2d!(ledger)
    end
    return bank
end

function reset_conservative_tree_subcycle_pair_2d!(
        bank::ConservativeTreeSubcycleLedgerBank2D,
        parent_level::Integer)
    reset_conservative_tree_subcycle_ledger_2d!(
        conservative_tree_subcycle_pair_ledger_2d(bank, parent_level))
    return bank
end

function conservative_tree_subcycle_spatial_pair_ledgers_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        parent_level::Integer)
    parent = _check_conservative_tree_pair_level_2d(
        bank.schedule, parent_level)
    pair = bank.ledger_pairs[parent + 1]
    ledgers = Vector{ConservativeTreeSubcycleLedger2D{eltype(pair.fine_to_coarse)}}(
        undef, length(pair.parent_ids))
    @inbounds for slot in eachindex(pair.parent_ids)
        ledgers[slot] = ConservativeTreeSubcycleLedger2D(
            bank.schedule.ratio,
            copy(@view(pair.coarse_to_fine[:, :, :, :, slot])),
            copy(@view(pair.fine_to_coarse[:, :, slot])))
    end
    return ledgers
end

function _conservative_tree_packed_ledger_pair_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        parent_level::Integer)
    parent = _check_conservative_tree_pair_level_2d(
        bank.schedule, parent_level)
    return bank.ledger_pairs[parent + 1]
end

function _conservative_tree_packed_ledger_slot_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        parent_cell_id::Integer)
    parent_id = Int(parent_cell_id)
    1 <= parent_id <= length(bank.spec.cells) ||
        throw(ArgumentError("parent_cell_id is outside the tree"))
    parent = bank.spec.cells[parent_id]
    _check_conservative_tree_pair_level_2d(bank.schedule, parent.level)
    slot = bank.parent_ledger_slot[parent_id]
    slot != 0 ||
        throw(ArgumentError("parent_cell_id does not identify a refined parent"))
    return slot
end

function conservative_tree_subcycle_spatial_ledger_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        parent_cell_id::Integer)
    parent_id = Int(parent_cell_id)
    1 <= parent_id <= length(bank.spec.cells) ||
        throw(ArgumentError("parent_cell_id is outside the tree"))
    parent = bank.spec.cells[parent_id]
    pair = _conservative_tree_packed_ledger_pair_2d(bank, parent.level)
    slot = _conservative_tree_packed_ledger_slot_2d(bank, parent_id)
    return ConservativeTreeSubcycleLedger2D(
        bank.schedule.ratio,
        copy(@view(pair.coarse_to_fine[:, :, :, :, slot])),
        copy(@view(pair.fine_to_coarse[:, :, slot])))
end

function reset_conservative_tree_subcycle_spatial_bank_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D)
    for pair in bank.ledger_pairs
        fill!(pair.coarse_to_fine, zero(eltype(pair.coarse_to_fine)))
        fill!(pair.fine_to_coarse, zero(eltype(pair.fine_to_coarse)))
    end
    for cache in bank.route_packet_caches
        _zero_conservative_tree_route_packet_cache_2d!(cache)
    end
    return bank
end

function reset_conservative_tree_subcycle_spatial_pair_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        parent_level::Integer)
    parent = _check_conservative_tree_pair_level_2d(
        bank.schedule, parent_level)
    pair = bank.ledger_pairs[parent + 1]
    fill!(pair.coarse_to_fine, zero(eltype(pair.coarse_to_fine)))
    fill!(pair.fine_to_coarse, zero(eltype(pair.fine_to_coarse)))
    _zero_conservative_tree_route_packet_cache_2d!(
        bank.route_packet_caches[parent + 1])
    return bank
end

function _zero_conservative_tree_route_packet_cache_2d!(
        cache::ConservativeTreeSubcycleRoutePacketCache2D)
    fill!(cache.packets, zero(eltype(cache.packets)))
    return cache
end

function _ensure_conservative_tree_route_packet_cache_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D{T},
        parent_level::Integer,
        dst_id::Integer,
        q::Integer) where T
    parent = _check_conservative_tree_pair_level_2d(
        bank.schedule, parent_level)
    cache = bank.route_packet_caches[parent + 1]
    key = (Int(dst_id), _check_d2q9_q(q))
    slot = get(cache.key_to_slot, key, 0)
    if slot == 0
        push!(cache.dst_ids, key[1])
        push!(cache.qs, key[2])
        slot = length(cache.dst_ids)
        cache.key_to_slot[key] = slot
        old_len = length(cache.packets)
        resize!(cache.packets, old_len + bank.schedule.ratio)
        @inbounds for idx in (old_len + 1):length(cache.packets)
            cache.packets[idx] = zero(T)
        end
    end
    return slot
end

function prepare_conservative_tree_subcycle_route_packet_cache_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        table::ConservativeTreeRouteTable2D)
    _check_conservative_tree_subcycle_route_table_2d(table)
    resize!(bank.route_packet_slot_by_route, length(table.routes))
    fill!(bank.route_packet_slot_by_route, 0)
    resize!(bank.inactive_route_packet_slots_by_level,
            bank.spec.max_level + 1)
    @inbounds for level in 0:bank.spec.max_level
        ids = bank.inactive_refined_ids_by_level[level + 1]
        slots = isassigned(bank.inactive_route_packet_slots_by_level,
                           level + 1) ?
                bank.inactive_route_packet_slots_by_level[level + 1] :
                zeros(Int, 0, 9)
        if size(slots, 1) != length(ids) || size(slots, 2) != 9
            slots = zeros(Int, length(ids), 9)
        else
            fill!(slots, 0)
        end
        bank.inactive_route_packet_slots_by_level[level + 1] = slots
    end

    @inbounds for route_id in table.interface_routes
        route = table.routes[route_id]
        route.kind == COALESCE_FACE || route.kind == COALESCE_CORNER || continue
        route.dst == 0 && continue
        child = bank.spec.cells[route.src]
        child.level > 0 || continue
        parent = bank.spec.cells[child.parent]
        slot = _ensure_conservative_tree_route_packet_cache_2d!(
            bank, parent.level, route.dst, route.q)
        bank.route_packet_slot_by_route[route_id] = slot
    end
    @inbounds for parent_level in 0:(bank.spec.max_level - 1)
        child_level = parent_level + 1
        inactive_ids = bank.inactive_refined_ids_by_level[child_level + 1]
        inactive_slots = bank.inactive_route_packet_slots_by_level[child_level + 1]
        for (local_idx, src_id) in enumerate(inactive_ids)
            src = bank.spec.cells[src_id]
            src.active && continue
            for q in 1:9
                dst_id, _ = _conservative_tree_inactive_parent_coalesce_route_spec_2d(
                    bank.spec, src_id, q)
                dst_id == 0 && continue
                inactive_slots[local_idx, q] =
                    _ensure_conservative_tree_route_packet_cache_2d!(
                    bank, parent_level, dst_id, q)
            end
        end
    end
    return bank
end

function _conservative_tree_subcycle_accumulate_fine_to_coarse_packet_unchecked_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        F::AbstractMatrix,
        src_id::Integer,
        dst_id::Integer,
        q::Integer,
        weight,
        kind::RouteKind,
        substep::Integer,
        route_packet_slot::Integer=0;
        alpha=1)
    kind == COALESCE_FACE || kind == COALESCE_CORNER ||
        throw(ArgumentError("route must be a fine-to-coarse coalesce route"))

    spec = bank.spec
    child_id = Int(src_id)
    child = spec.cells[child_id]
    child.level > 0 ||
        throw(ArgumentError("fine-to-coarse route source must have a parent"))
    parent_id = child.parent
    parent = spec.cells[parent_id]
    dst_cell_id = Int(dst_id)
    dst_cell_id == 0 || spec.cells[dst_cell_id].level == parent.level ||
        throw(ArgumentError("fine-to-coarse route destination level mismatch"))
    pair = _conservative_tree_packed_ledger_pair_2d(bank, parent.level)
    slot = _conservative_tree_packed_ledger_slot_2d(bank, parent_id)
    step = Int(substep)
    1 <= step <= bank.schedule.ratio ||
        throw(ArgumentError("substep must be inside 1:$(bank.schedule.ratio)"))
    qi = _check_d2q9_q(q)
    packet = reconstructed_integrated_D2Q9_packet(
        @view(F[child_id, :]), child.metrics.volume, qi, weight;
        alpha=alpha) / bank.schedule.ratio
    pair.fine_to_coarse[qi, step, slot] += packet
    dst_cell_id != 0 ||
        throw(ArgumentError("fine-to-coarse route must have a spatial destination"))
    packet_slot = Int(route_packet_slot)
    if packet_slot == 0
        packet_slot = _ensure_conservative_tree_route_packet_cache_2d!(
            bank, parent.level, dst_cell_id, qi)
    end
    cache = bank.route_packet_caches[parent.level + 1]
    cache.packets[(packet_slot - 1) * bank.schedule.ratio + step] += packet
    return bank
end

function _conservative_tree_subcycle_accumulate_fine_to_coarse_packet_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        F::AbstractMatrix,
        src_id::Integer,
        dst_id::Integer,
        q::Integer,
        weight,
        kind::RouteKind,
        substep::Integer,
        route_packet_slot::Integer=0;
        alpha=1)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    return _conservative_tree_subcycle_accumulate_fine_to_coarse_packet_unchecked_2d!(
        bank, F, src_id, dst_id, q, weight, kind, substep,
        route_packet_slot; alpha=alpha)
end

@inline function _conservative_tree_child_slot_2d(ix::Int, iy::Int)
    1 <= ix <= 2 || throw(ArgumentError("child ix must be 1 or 2"))
    1 <= iy <= 2 || throw(ArgumentError("child iy must be 1 or 2"))
    return ix + 2 * (iy - 1)
end

@inline function _conservative_tree_child_index_in_parent_2d(
        parent::ConservativeTreeCell2D,
        child::ConservativeTreeCell2D)
    child.level == parent.level + 1 ||
        throw(ArgumentError("child cell is not one level below parent"))
    ix = child.i - 2 * parent.i + 2
    iy = child.j - 2 * parent.j + 2
    1 <= ix <= 2 && 1 <= iy <= 2 ||
        throw(ArgumentError("child cell is not inside parent"))
    return ix, iy
end

function _check_conservative_tree_subcycle_spatial_F_2d(
        F::AbstractMatrix,
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D)
    _check_conservative_tree_F_2d(F, bank.spec)
    return nothing
end

function _check_conservative_tree_subcycle_route_table_2d(
        table::ConservativeTreeRouteTable2D)
    return table
end

function _check_conservative_tree_leaf_solid_mask_2d(
        spec::ConservativeTreeSpec2D,
        is_solid::AbstractArray{Bool,2})
    leaf_nx = _conservative_tree_level_size_2d(spec.Nx, spec.max_level)
    leaf_ny = _conservative_tree_level_size_2d(spec.Ny, spec.max_level)
    size(is_solid) == (leaf_nx, leaf_ny) ||
        throw(ArgumentError("is_solid must match the finest leaf-equivalent grid"))
    return is_solid
end

@inline function _conservative_tree_cell_leaf_bounds_2d(
        spec::ConservativeTreeSpec2D,
        cell::ConservativeTreeCell2D)
    scale = 1 << (spec.max_level - cell.level)
    i0 = (cell.i - 1) * scale + 1
    i1 = cell.i * scale
    j0 = (cell.j - 1) * scale + 1
    j1 = cell.j * scale
    return i0, i1, j0, j1
end

function _conservative_tree_cell_solid_status_2d(
        spec::ConservativeTreeSpec2D,
        cell::ConservativeTreeCell2D,
        is_solid::AbstractArray{Bool,2})
    i0, i1, j0, j1 = _conservative_tree_cell_leaf_bounds_2d(spec, cell)
    any_solid = false
    all_solid = true
    @inbounds for j in j0:j1, i in i0:i1
        solid = is_solid[i, j]
        any_solid |= solid
        all_solid &= solid
    end
    all_solid && return :solid
    any_solid && return :partial
    return :fluid
end

function _conservative_tree_cell_is_solid_2d(
        spec::ConservativeTreeSpec2D,
        cell::ConservativeTreeCell2D,
        is_solid::Union{Nothing,AbstractArray{Bool,2}})
    is_solid === nothing && return false
    status = _conservative_tree_cell_solid_status_2d(spec, cell, is_solid)
    status == :partial &&
        throw(ArgumentError("active AMR-D solid cells must be fully resolved by refinement"))
    return status == :solid
end

function validate_conservative_tree_solid_mask_resolved_2d(
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D,
        is_solid::AbstractArray{Bool,2})
    _check_conservative_tree_leaf_solid_mask_2d(spec, is_solid)
    @inbounds for cell_id in spec.active_cells
        status = _conservative_tree_cell_solid_status_2d(
            spec, spec.cells[cell_id], is_solid)
        status == :partial &&
            throw(ArgumentError("AMR-D solid mask cuts active cell $cell_id; refine the solid band"))
    end
    @inbounds for route_id in table.interface_routes
        route = table.routes[route_id]
        for cell_id in (route.src, route.dst)
            cell_id == 0 && continue
            status = _conservative_tree_cell_solid_status_2d(
                spec, spec.cells[cell_id], is_solid)
            status == :fluid && continue
            throw(ArgumentError("AMR-D solid mask touches an interface route; keep refinement interfaces away from solids"))
        end
    end
    return is_solid
end

