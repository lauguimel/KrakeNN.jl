function stream_conservative_tree_subcycled_buffered_routes_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D;
        boundary::Symbol=:skip,
        u_south=0,
        u_north=0,
        rho_wall=1,
        alpha_c2f=1,
        alpha_f2c=1,
        pre_stream_level! = nothing,
        schedule = nothing,
        route_bank = nothing,
        state_bank = nothing,
        Fsource = nothing,
        Fscratch = nothing,
        is_solid=nothing)
    _check_conservative_tree_stream_args_2d(Fout, Fin, spec)
    policy = _check_conservative_tree_boundary_policy_2d(boundary)
    is_solid === nothing ||
        validate_conservative_tree_solid_mask_resolved_2d(
            spec, table, is_solid)
    if spec.max_level == 0
        return stream_conservative_tree_routes_F_2d!(
            Fout, Fin, spec, table; boundary=boundary, u_south=u_south,
            u_north=u_north, rho_wall=rho_wall)
    end

    schedule_run = schedule === nothing ?
        create_conservative_tree_subcycle_schedule_2d(spec.max_level) :
        schedule
    _check_conservative_tree_subcycle_spec_schedule_2d(spec, schedule_run)
    route_bank_run = route_bank === nothing ?
        create_conservative_tree_subcycle_spatial_ledger_bank_2d(
            spec; schedule=schedule_run, T=eltype(Fout)) :
        route_bank
    state_bank_run = state_bank === nothing ?
        create_conservative_tree_subcycle_buffer_bank_2d(
            spec; schedule=schedule_run, T=eltype(Fout)) :
        state_bank
    route_bank_run.spec === spec ||
        throw(ArgumentError("route_bank must belong to spec"))
    route_bank_run.schedule === schedule_run ||
        throw(ArgumentError("route_bank schedule must match schedule"))
    state_bank_run.spec === spec ||
        throw(ArgumentError("state_bank must belong to spec"))
    state_bank_run.schedule === schedule_run ||
        throw(ArgumentError("state_bank schedule must match schedule"))
    prepare_conservative_tree_subcycle_route_packet_cache_2d!(
        route_bank_run, table)
    reset_conservative_tree_subcycle_spatial_bank_2d!(route_bank_run)
    reset_conservative_tree_subcycle_buffer_bank_2d!(state_bank_run)
    conservative_tree_subcycle_store_active_owned_2d!(state_bank_run, Fin)

    Fsource_run = Fsource === nothing ? similar(Fout) : Fsource
    Fscratch_run = Fscratch === nothing ? similar(Fout) : Fscratch
    _check_conservative_tree_F_2d(Fsource_run, spec)
    _check_conservative_tree_F_2d(Fscratch_run, spec)

    for event in schedule_run.events
        if event.phase == :sync_down
            reset_conservative_tree_subcycle_spatial_pair_2d!(
                route_bank_run, event.src_level)
            parent_owned = state_bank_run.levels[event.src_level + 1].owned
            conservative_tree_subcycle_sync_down_routes_F_2d!(
                route_bank_run, event, parent_owned, table; alpha=alpha_c2f)
        elseif event.phase == :advance
            level = event.src_level
            buffers = state_bank_run.levels[level + 1]
            fill!(Fsource_run, zero(eltype(Fsource_run)))
            _copy_conservative_tree_level_rows_2d!(
                Fsource_run, buffers.owned, spec, level)
            conservative_tree_subcycle_apply_restriction_to_inactive_level_F_2d!(
                Fsource_run, state_bank_run, level)
            pre_stream_level! === nothing ||
                pre_stream_level!(Fsource_run, spec, level, event)

            if level > 0
                conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
                    route_bank_run, event, Fsource_run, table; alpha=alpha_f2c)
                conservative_tree_subcycle_accumulate_inactive_parent_routes_F_2d!(
                    route_bank_run, event, Fsource_run; alpha=alpha_f2c)
            end

            fill!(Fscratch_run, zero(eltype(Fscratch_run)))
            _stream_conservative_tree_direct_level_routes_F_2d!(
                Fscratch_run, Fsource_run, spec, table, level, policy;
                u_south=u_south, u_north=u_north, rho_wall=rho_wall,
                is_solid=is_solid)
            if level > 0
                fill!(buffers.ghost_from_coarse,
                      zero(eltype(buffers.ghost_from_coarse)))
                conservative_tree_subcycle_apply_child_advance_injection_F_2d!(
                    buffers.ghost_from_coarse, route_bank_run, event)
                conservative_tree_subcycle_add_and_clear_ghost_to_F_level_2d!(
                    Fscratch_run, state_bank_run, level)
            end
            conservative_tree_subcycle_add_and_clear_reflux_to_F_level_2d!(
                Fscratch_run, state_bank_run, level)
            _copy_conservative_tree_active_level_rows_2d!(
                buffers.owned, Fscratch_run, spec, level)
        elseif event.phase == :sync_up
            parent_level = _check_subcycle_spatial_sync_up_event_2d(
                route_bank_run, event)
            conservative_tree_subcycle_restrict_level_2d!(
                state_bank_run, parent_level)
            parent_reflux = state_bank_run.levels[parent_level + 1].reflux_to_coarse
            conservative_tree_subcycle_apply_sync_up_F_2d!(
                parent_reflux, route_bank_run, event; boundary=boundary,
                u_south=u_south, u_north=u_north, rho_wall=rho_wall)
        else
            throw(ArgumentError("unknown subcycle event phase $(event.phase)"))
        end
    end

    conservative_tree_subcycle_collect_active_owned_F_2d!(Fout, state_bank_run)
    return Fout
end

function diagnose_conservative_tree_subcycled_rest_2d(
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D;
        boundary::Symbol=:bounceback,
        T::Type{<:Real}=Float64)
    Fin = allocate_conservative_tree_F_2d(spec; T=T)
    Fout = allocate_conservative_tree_F_2d(spec; T=T)
    w = weights(D2Q9())
    @inbounds for cell_id in spec.active_cells
        volume = T(spec.cells[cell_id].metrics.volume)
        for q in 1:9
            Fin[cell_id, q] = T(w[q]) * volume
        end
    end

    stream_conservative_tree_subcycled_routes_F_2d!(
        Fout, Fin, spec, table; boundary=boundary)

    active_initial = sum(Fin[spec.active_cells, :])
    active_final = sum(Fout[spec.active_cells, :])
    level_drift = zeros(T, spec.max_level + 1)
    @inbounds for level in 0:spec.max_level
        ids = [id for id in spec.active_cells if spec.cells[id].level == level]
        level_drift[level + 1] = sum(Fout[ids, :]) - sum(Fin[ids, :])
    end
    orientation_drift = active_population_sums_F_2d(Fout, spec) -
                        active_population_sums_F_2d(Fin, spec)
    return (active_initial=active_initial,
            active_final=active_final,
            active_drift=active_final - active_initial,
            max_active_abs=maximum(abs.(Fout[spec.active_cells, :] .-
                                        Fin[spec.active_cells, :])),
            level_drift=level_drift,
            orientation_drift=orientation_drift)
end

function conservative_tree_subcycle_sync_down_face_2d!(
        bank::ConservativeTreeSubcycleLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        Fq,
        q::Integer,
        face::Symbol)
    parent = _check_subcycle_sync_down_event_2d(bank.schedule, event)
    ledger = conservative_tree_subcycle_pair_ledger_2d(bank, parent)
    conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(
        ledger, Fq, q, face)
    return ledger
end

function conservative_tree_subcycle_sync_down_corner_2d!(
        bank::ConservativeTreeSubcycleLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        Fq,
        q::Integer,
        corner::Symbol)
    parent = _check_subcycle_sync_down_event_2d(bank.schedule, event)
    ledger = conservative_tree_subcycle_pair_ledger_2d(bank, parent)
    conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(
        ledger, Fq, q, corner)
    return ledger
end

function conservative_tree_subcycle_accumulate_advance_face_2d!(
        bank::ConservativeTreeSubcycleLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        fine_half_step::AbstractArray{<:Any,3},
        q::Integer,
        face::Symbol)
    parent, substep = _check_subcycle_child_advance_event_2d(
        bank.schedule, event)
    ledger = conservative_tree_subcycle_pair_ledger_2d(bank, parent)
    conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
        ledger, fine_half_step, q, face, substep)
    return ledger
end

function conservative_tree_subcycle_accumulate_advance_corner_2d!(
        bank::ConservativeTreeSubcycleLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        fine_half_step::AbstractArray{<:Any,3},
        q::Integer,
        corner::Symbol)
    parent, substep = _check_subcycle_child_advance_event_2d(
        bank.schedule, event)
    ledger = conservative_tree_subcycle_pair_ledger_2d(bank, parent)
    conservative_tree_subcycle_accumulate_fine_to_coarse_corner_2d!(
        ledger, fine_half_step, q, corner, substep)
    return ledger
end

function conservative_tree_subcycle_sync_up_ledger_2d(
        bank::ConservativeTreeSubcycleLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D)
    parent = _check_subcycle_sync_up_event_2d(bank.schedule, event)
    return conservative_tree_subcycle_pair_ledger_2d(bank, parent)
end

"""
    reset_conservative_tree_subcycle_ledger_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.reset_conservative_tree_subcycle_ledger_2d!)
```
"""
function reset_conservative_tree_subcycle_ledger_2d!(
        ledger::ConservativeTreeSubcycleLedger2D)
    ledger.coarse_to_fine .= 0
    ledger.fine_to_coarse .= 0
    return ledger
end

"""
    conservative_tree_subcycle_weights_2d(ledger::ConservativeTreeSubcycleLedger2D{T}) where T

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_subcycle_weights_2d)
```
"""
function conservative_tree_subcycle_weights_2d(ledger::ConservativeTreeSubcycleLedger2D{T}) where T
    ledger.ratio == 2 ||
        throw(ArgumentError("conservative-tree subcycle weights require ratio = 2"))
    return fill(inv(T(ledger.ratio)), ledger.ratio)
end

@inline function _check_subcycle_step_2d(ledger::ConservativeTreeSubcycleLedger2D,
                                         substep::Integer)
    step = Int(substep)
    1 <= step <= ledger.ratio ||
        throw(ArgumentError("substep must be inside 1:$(ledger.ratio)"))
    return step
end

"""
    conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!)
```
"""
function conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(
        ledger::ConservativeTreeSubcycleLedger2D,
        Fq,
        q::Integer,
        face::Symbol)
    qi = _check_d2q9_q(Int(q))
    weights = conservative_tree_subcycle_weights_2d(ledger)
    @inbounds for substep in 1:ledger.ratio
        split_coarse_to_fine_face_F_2d!(
            @view(ledger.coarse_to_fine[:, :, :, substep]),
            Fq * weights[substep], qi, face)
    end
    return ledger
end

"""
    conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!)
```
"""
function conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(
        ledger::ConservativeTreeSubcycleLedger2D,
        Fq,
        q::Integer,
        corner::Symbol)
    qi = _check_d2q9_q(Int(q))
    weights = conservative_tree_subcycle_weights_2d(ledger)
    @inbounds for substep in 1:ledger.ratio
        split_coarse_to_fine_corner_F_2d!(
            @view(ledger.coarse_to_fine[:, :, :, substep]),
            Fq * weights[substep], qi, corner)
    end
    return ledger
end

"""
    conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!)
```
"""
function conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
        ledger::ConservativeTreeSubcycleLedger2D,
        fine_half_step::AbstractArray{<:Any,3},
        q::Integer,
        face::Symbol,
        substep::Integer)
    _check_child_block_2d(fine_half_step, "fine_half_step")
    qi = _check_d2q9_q(Int(q))
    step = _check_subcycle_step_2d(ledger, substep)
    ledger.fine_to_coarse[qi, step] +=
        coalesce_fine_to_coarse_face_F(fine_half_step, qi, face)
    return ledger
end

"""
    conservative_tree_subcycle_accumulate_fine_to_coarse_corner_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_subcycle_accumulate_fine_to_coarse_corner_2d!)
```
"""
function conservative_tree_subcycle_accumulate_fine_to_coarse_corner_2d!(
        ledger::ConservativeTreeSubcycleLedger2D,
        fine_half_step::AbstractArray{<:Any,3},
        q::Integer,
        corner::Symbol,
        substep::Integer)
    _check_child_block_2d(fine_half_step, "fine_half_step")
    qi = _check_d2q9_q(Int(q))
    step = _check_subcycle_step_2d(ledger, substep)
    ledger.fine_to_coarse[qi, step] +=
        coalesce_fine_to_coarse_corner_F(fine_half_step, qi, corner)
    return ledger
end

"""
    conservative_tree_subcycle_orientation_sums_2d(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_subcycle_orientation_sums_2d)
```
"""
function conservative_tree_subcycle_orientation_sums_2d(
        ledger::ConservativeTreeSubcycleLedger2D{T}) where T
    coarse_to_fine = zeros(T, 9)
    fine_to_coarse = zeros(T, 9)
    @inbounds for substep in 1:ledger.ratio, q in 1:9
        fine_to_coarse[q] += ledger.fine_to_coarse[q, substep]
        for j in 1:2, i in 1:2
            coarse_to_fine[q] += ledger.coarse_to_fine[i, j, q, substep]
        end
    end
    return (coarse_to_fine=coarse_to_fine, fine_to_coarse=fine_to_coarse)
end

"""
    conservative_tree_subcycle_total_sums_2d(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_subcycle_total_sums_2d)
```
"""
function conservative_tree_subcycle_total_sums_2d(
        ledger::ConservativeTreeSubcycleLedger2D)
    sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
    return (coarse_to_fine=sum(sums.coarse_to_fine),
            fine_to_coarse=sum(sums.fine_to_coarse))
end
