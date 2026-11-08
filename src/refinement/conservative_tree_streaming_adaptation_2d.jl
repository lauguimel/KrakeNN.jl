"""
    stream_composite_routes_interior_F_2d!(coarse_out, patch_out,
                                           coarse_in, patch_in, topology;
                                           clear=true)

Scatter integrated D2Q9 populations along the non-boundary routes of a
`ConservativeTreeTopology2D`.

This is a surgical native-composite streaming primitive for the first AMR
milestone. It conserves every non-boundary route packet exactly up to roundoff.
Boundary routes are explicit in the topology and intentionally skipped here;
native periodic/wall/Zou-He closures are added in a later milestone.
"""
function stream_composite_routes_interior_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D;
        clear::Bool=true,
        coarse_prolongation::Symbol=:flat)
    return _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :skip, clear;
                                          coarse_prolongation=coarse_prolongation)
end

"""
    stream_composite_routes_periodic_x_F_2d!(coarse_out, patch_out,
                                             coarse_in, patch_in, topology;
                                             clear=true)

Scatter integrated D2Q9 populations along all interior routes and wrap
boundary packets that leave through the periodic x direction.

Packets leaving through y boundaries are still skipped. This keeps the
Milestone-1 boundary surface explicit: periodic x is native, wall and inlet/
outlet closures are added by later surgical patches.
"""
function stream_composite_routes_periodic_x_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D;
        clear::Bool=true,
        coarse_prolongation::Symbol=:flat)
    return _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :periodic_x, clear;
                                          coarse_prolongation=coarse_prolongation)
end

"""
    stream_composite_routes_periodic_x_wall_y_F_2d!(coarse_out, patch_out,
                                                    coarse_in, patch_in,
                                                    topology; clear=true)

Scatter integrated D2Q9 populations with periodic x wrapping and stationary
no-slip bounce-back for packets leaving through y boundaries.

This is still a transport-only primitive. Moving-wall and inlet/outlet
corrections belong to later boundary patches.
"""
function stream_composite_routes_periodic_x_wall_y_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D;
        clear::Bool=true,
        coarse_prolongation::Symbol=:flat)
    return _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :periodic_x_wall_y, clear;
                                          coarse_prolongation=coarse_prolongation)
end

"""
    stream_composite_routes_periodic_x_moving_wall_y_F_2d!(
        coarse_out, patch_out, coarse_in, patch_in, topology;
        u_south=0, u_north=0, rho_wall=1,
        volume_coarse=1, volume_fine=0.25, clear=true)

Scatter integrated D2Q9 populations with periodic x wrapping and moving-wall
bounce-back corrections on y boundaries.
"""
function stream_composite_routes_periodic_x_moving_wall_y_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D;
        u_south=0,
        u_north=0,
        rho_wall=1,
        volume_coarse=1,
        volume_fine=0.25,
        clear::Bool=true,
        coarse_prolongation::Symbol=:flat)
    return _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :periodic_x_moving_wall_y,
                                          clear;
                                          u_south=u_south,
                                          u_north=u_north,
                                          rho_wall=rho_wall,
                                          volume_coarse=volume_coarse,
                                          volume_fine=volume_fine,
                                          coarse_prolongation=coarse_prolongation)
end

"""
    stream_composite_routes_zou_he_x_wall_y_F_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.stream_composite_routes_zou_he_x_wall_y_F_2d!)
```
"""
function stream_composite_routes_zou_he_x_wall_y_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D;
        u_in=0,
        rho_out=1,
        rho_wall=1,
        volume_coarse=1,
        volume_fine=0.25,
        clear::Bool=true,
        coarse_prolongation::Symbol=:flat)
    _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                   coarse_in, patch_in,
                                   topology, :open_x_wall_y, clear;
                                   rho_wall=rho_wall,
                                   volume_coarse=volume_coarse,
                                   volume_fine=volume_fine,
                                   coarse_prolongation=coarse_prolongation)
    apply_composite_zou_he_west_F_2d!(
        coarse_out, patch_out, u_in, volume_coarse, volume_fine)
    apply_composite_zou_he_pressure_east_F_2d!(
        coarse_out, patch_out, volume_coarse, volume_fine; rho_out=rho_out)
    return coarse_out, patch_out
end

"""
    stream_composite_routes_zou_he_x_wall_y_solid_F_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.stream_composite_routes_zou_he_x_wall_y_solid_F_2d!)
```
"""
function stream_composite_routes_zou_he_x_wall_y_solid_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D,
        is_solid::AbstractArray{Bool,2};
        u_in=0,
        rho_out=1,
        rho_wall=1,
        volume_coarse=1,
        volume_fine=0.25,
        clear::Bool=true,
        coarse_prolongation::Symbol=:flat)
    _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                   coarse_in, patch_in,
                                   topology, :open_x_wall_y, clear;
                                   rho_wall=rho_wall,
                                   volume_coarse=volume_coarse,
                                   volume_fine=volume_fine,
                                   is_solid=is_solid,
                                   coarse_prolongation=coarse_prolongation)
    apply_composite_zou_he_west_F_2d!(
        coarse_out, patch_out, is_solid, u_in, volume_coarse, volume_fine)
    apply_composite_zou_he_pressure_east_F_2d!(
        coarse_out, patch_out, is_solid, volume_coarse, volume_fine;
        rho_out=rho_out)
    return coarse_out, patch_out
end

"""
    stream_composite_routes_periodic_x_wall_y_solid_F_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.stream_composite_routes_periodic_x_wall_y_solid_F_2d!)
```
"""
function stream_composite_routes_periodic_x_wall_y_solid_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D,
        is_solid::AbstractArray{Bool,2};
        clear::Bool=true,
        coarse_prolongation::Symbol=:flat)
    return _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :periodic_x_wall_y, clear;
                                          is_solid=is_solid,
                                          coarse_prolongation=coarse_prolongation)
end

"""
    collide_Guo_composite_solid_F_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_Guo_composite_solid_F_2d!)
```
"""
function collide_Guo_composite_solid_F_2d!(
        coarse_F::AbstractArray{<:Any,3},
        patch::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D,
        is_solid::AbstractArray{Bool,2},
        volume_coarse,
        volume_fine,
        omega_coarse,
        omega_fine,
        Fx,
        Fy)
    _check_route_stream_topology_layout(topology, coarse_F, patch)
    _check_route_solid_mask_layout(topology, coarse_F, patch, is_solid)

    @inbounds for id in topology.active_cells
        cell = topology.cells[id]
        _cell_is_solid_2d(cell, is_solid) && continue
        volume = cell.level == 0 ? volume_coarse : volume_fine
        omega = cell.level == 0 ? omega_coarse : omega_fine
        collide_Guo_integrated_D2Q9!(_cell_view_2d(coarse_F, patch, cell),
                                     volume, omega, Fx, Fy)
    end
    coalesce_patch_to_shadow_F_2d!(patch)
    return coarse_F, patch
end

"""
    regrid_conservative_tree_patch_F_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.regrid_conservative_tree_patch_F_2d!)
```
"""
function regrid_conservative_tree_patch_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D)
    size(coarse_out) == size(coarse_in) ||
        throw(ArgumentError("coarse_out and coarse_in must have the same size"))
    _check_composite_coarse_layout(coarse_in, patch_in)
    _check_composite_coarse_layout(coarse_out, patch_out)

    leaf = similar(coarse_in, 2 * size(coarse_in, 1), 2 * size(coarse_in, 2), 9)
    composite_to_leaf_F_2d!(leaf, coarse_in, patch_in)
    leaf_to_composite_F_2d!(coarse_out, patch_out, leaf)
    return coarse_out, patch_out
end

function _source_parent_leaf_block_F_2d!(
        leaf_block::AbstractArray{<:Any,3},
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        I::Int,
        J::Int)
    _check_child_block_2d(leaf_block, "leaf_block")
    if _inside_range(I, J, patch_in.parent_i_range, patch_in.parent_j_range)
        il, jl = _patch_local_parent_index(patch_in, I, J)
        leaf_block .= _child_block_view(patch_in.fine_F, il, jl)
    else
        _explode_limited_linear_composite_F_2d!(leaf_block, coarse_in, patch_in, I, J)
    end
    return leaf_block
end

"""
    regrid_conservative_tree_patch_direct_F_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.regrid_conservative_tree_patch_direct_F_2d!)
```
"""
function regrid_conservative_tree_patch_direct_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D)
    size(coarse_out) == size(coarse_in) ||
        throw(ArgumentError("coarse_out and coarse_in must have the same size"))
    _check_composite_coarse_layout(coarse_in, patch_in)
    _check_composite_coarse_layout(coarse_out, patch_out)
    coalesce_patch_to_shadow_F_2d!(patch_in)

    coarse_out .= 0
    patch_out.fine_F .= 0
    patch_out.coarse_shadow_F .= 0
    leaf_block = zeros(promote_type(eltype(coarse_in), eltype(patch_in.fine_F)),
                       2, 2, 9)

    @inbounds for J in axes(coarse_in, 2), I in axes(coarse_in, 1)
        _source_parent_leaf_block_F_2d!(leaf_block, coarse_in, patch_in, I, J)
        if _inside_range(I, J, patch_out.parent_i_range, patch_out.parent_j_range)
            il, jl = _patch_local_parent_index(patch_out, I, J)
            _child_block_view(patch_out.fine_F, il, jl) .= leaf_block
        else
            coalesce_F_2d!(@view(coarse_out[I, J, :]), leaf_block)
        end
    end

    coalesce_patch_to_shadow_F_2d!(patch_out)
    return coarse_out, patch_out
end

"""
    conservative_tree_solid_mask_patch_range_2d(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_solid_mask_patch_range_2d)
```
"""
function conservative_tree_solid_mask_patch_range_2d(
        is_solid::AbstractArray{Bool,2};
        pad::Int=1)
    nx_leaf = size(is_solid, 1)
    ny_leaf = size(is_solid, 2)
    iseven(nx_leaf) && iseven(ny_leaf) ||
        throw(ArgumentError("is_solid dimensions must be even leaf-grid sizes"))
    pad >= 0 || throw(ArgumentError("pad must be nonnegative"))
    any(is_solid) || throw(ArgumentError("is_solid contains no solid cells"))

    i_min_leaf = typemax(Int)
    i_max_leaf = typemin(Int)
    j_min_leaf = typemax(Int)
    j_max_leaf = typemin(Int)
    @inbounds for j in axes(is_solid, 2), i in axes(is_solid, 1)
        if is_solid[i, j]
            i_min_leaf = min(i_min_leaf, i)
            i_max_leaf = max(i_max_leaf, i)
            j_min_leaf = min(j_min_leaf, j)
            j_max_leaf = max(j_max_leaf, j)
        end
    end

    nx = nx_leaf >>> 1
    ny = ny_leaf >>> 1
    i_min = max(1, cld(i_min_leaf, 2) - pad)
    i_max = min(nx, cld(i_max_leaf, 2) + pad)
    j_min = max(1, cld(j_min_leaf, 2) - pad)
    j_max = min(ny, cld(j_max_leaf, 2) + pad)
    return (i_range=i_min:i_max, j_range=j_min:j_max)
end

"""
    conservative_tree_indicator_patch_range_2d(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_indicator_patch_range_2d)
```
"""
function conservative_tree_indicator_patch_range_2d(
        indicator::AbstractArray{<:Real,2};
        threshold::Real,
        pad::Int=1)
    nx = size(indicator, 1)
    ny = size(indicator, 2)
    nx > 0 && ny > 0 || throw(ArgumentError("indicator must be nonempty"))
    isfinite(threshold) || throw(ArgumentError("threshold must be finite"))
    threshold >= 0 || throw(ArgumentError("threshold must be nonnegative"))
    pad >= 0 || throw(ArgumentError("pad must be nonnegative"))

    i_min_hit = typemax(Int)
    i_max_hit = typemin(Int)
    j_min_hit = typemax(Int)
    j_max_hit = typemin(Int)
    hit = false
    @inbounds for j in axes(indicator, 2), i in axes(indicator, 1)
        value = indicator[i, j]
        isfinite(value) || throw(ArgumentError("indicator contains non-finite values"))
        if abs(value) > threshold
            hit = true
            i_min_hit = min(i_min_hit, i)
            i_max_hit = max(i_max_hit, i)
            j_min_hit = min(j_min_hit, j)
            j_max_hit = max(j_max_hit, j)
        end
    end
    hit || throw(ArgumentError("indicator has no cells above threshold"))

    i_min = max(1, i_min_hit - pad)
    i_max = min(nx, i_max_hit + pad)
    j_min = max(1, j_min_hit - pad)
    j_max = min(ny, j_max_hit + pad)
    return (i_range=i_min:i_max, j_range=j_min:j_max)
end

"""
    conservative_tree_gradient_indicator_2d(field::AbstractArray{<:Real,2})

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_gradient_indicator_2d)
```
"""
function conservative_tree_gradient_indicator_2d(field::AbstractArray{<:Real,2})
    nx = size(field, 1)
    ny = size(field, 2)
    nx > 0 && ny > 0 || throw(ArgumentError("field must be nonempty"))
    T = promote_type(Float64, eltype(field))
    indicator = zeros(T, nx, ny)

    @inbounds for j in axes(field, 2), i in axes(field, 1)
        isfinite(field[i, j]) || throw(ArgumentError("field contains non-finite values"))
        if nx == 1
            dx = zero(T)
        elseif i == first(axes(field, 1))
            dx = T(field[i + 1, j] - field[i, j])
        elseif i == last(axes(field, 1))
            dx = T(field[i, j] - field[i - 1, j])
        else
            dx = T(field[i + 1, j] - field[i - 1, j]) / T(2)
        end

        if ny == 1
            dy = zero(T)
        elseif j == first(axes(field, 2))
            dy = T(field[i, j + 1] - field[i, j])
        elseif j == last(axes(field, 2))
            dy = T(field[i, j] - field[i, j - 1])
        else
            dy = T(field[i, j + 1] - field[i, j - 1]) / T(2)
        end
        indicator[i, j] = sqrt(dx * dx + dy * dy)
    end
    return indicator
end

"""
    conservative_tree_hysteresis_patch_range_2d(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_hysteresis_patch_range_2d)
```
"""
function conservative_tree_hysteresis_patch_range_2d(
        current_i_range::AbstractUnitRange{<:Integer},
        current_j_range::AbstractUnitRange{<:Integer},
        target_i_range::AbstractUnitRange{<:Integer},
        target_j_range::AbstractUnitRange{<:Integer};
        shrink_margin::Int=1)
    isempty(current_i_range) && throw(ArgumentError("current_i_range must be nonempty"))
    isempty(current_j_range) && throw(ArgumentError("current_j_range must be nonempty"))
    isempty(target_i_range) && throw(ArgumentError("target_i_range must be nonempty"))
    isempty(target_j_range) && throw(ArgumentError("target_j_range must be nonempty"))
    shrink_margin >= 0 || throw(ArgumentError("shrink_margin must be nonnegative"))

    current_i = Int(first(current_i_range)):Int(last(current_i_range))
    current_j = Int(first(current_j_range)):Int(last(current_j_range))
    target_i = Int(first(target_i_range)):Int(last(target_i_range))
    target_j = Int(first(target_j_range)):Int(last(target_j_range))

    grows = first(target_i) < first(current_i) ||
            last(target_i) > last(current_i) ||
            first(target_j) < first(current_j) ||
            last(target_j) > last(current_j)
    if grows || shrink_margin == 0
        return (i_range=target_i, j_range=target_j)
    end

    can_shrink_i = first(target_i) >= first(current_i) + shrink_margin &&
                   last(target_i) <= last(current_i) - shrink_margin
    can_shrink_j = first(target_j) >= first(current_j) + shrink_margin &&
                   last(target_j) <= last(current_j) - shrink_margin
    if can_shrink_i && can_shrink_j
        return (i_range=target_i, j_range=target_j)
    end
    return (i_range=current_i, j_range=current_j)
end

"""
    conservative_tree_velocity_gradient_patch_range_2d(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_velocity_gradient_patch_range_2d)
```
"""
function conservative_tree_velocity_gradient_patch_range_2d(
        coarse_F::AbstractArray{T,3},
        patch::ConservativeTreePatch2D{T};
        threshold::Real,
        volume_leaf::T=T(0.25),
        force_x::T=zero(T),
        force_y::T=zero(T),
        pad_leaf::Int=0,
        pad_parent::Int=0,
        shrink_margin::Int=1) where T
    _check_composite_coarse_layout(coarse_F, patch)
    pad_leaf >= 0 || throw(ArgumentError("pad_leaf must be nonnegative"))
    pad_parent >= 0 || throw(ArgumentError("pad_parent must be nonnegative"))

    velocity = composite_leaf_velocity_field_2d(
        coarse_F, patch; volume_leaf=volume_leaf, force_x=force_x, force_y=force_y)
    indicator = conservative_tree_gradient_indicator_2d(velocity.ux)
    leaf_ranges = conservative_tree_indicator_patch_range_2d(
        indicator; threshold=threshold, pad=pad_leaf)

    nx = size(coarse_F, 1)
    ny = size(coarse_F, 2)
    i_min = max(1, cld(first(leaf_ranges.i_range), 2) - pad_parent)
    i_max = min(nx, cld(last(leaf_ranges.i_range), 2) + pad_parent)
    j_min = max(1, cld(first(leaf_ranges.j_range), 2) - pad_parent)
    j_max = min(ny, cld(last(leaf_ranges.j_range), 2) + pad_parent)
    return conservative_tree_hysteresis_patch_range_2d(
        patch.parent_i_range, patch.parent_j_range, i_min:i_max, j_min:j_max;
        shrink_margin=shrink_margin)
end

"""
    adapt_conservative_tree_patch_to_velocity_gradient_2d(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.adapt_conservative_tree_patch_to_velocity_gradient_2d)
```
"""
function adapt_conservative_tree_patch_to_velocity_gradient_2d(
        coarse_F::AbstractArray{T,3},
        patch::ConservativeTreePatch2D{T};
        threshold::Real,
        volume_leaf::T=T(0.25),
        force_x::T=zero(T),
        force_y::T=zero(T),
        pad_leaf::Int=0,
        pad_parent::Int=0,
        shrink_margin::Int=1) where T
    ranges = conservative_tree_velocity_gradient_patch_range_2d(
        coarse_F, patch; threshold=threshold, volume_leaf=volume_leaf,
        force_x=force_x, force_y=force_y, pad_leaf=pad_leaf,
        pad_parent=pad_parent, shrink_margin=shrink_margin)
    patch_out = create_conservative_tree_patch_2d(
        ranges.i_range, ranges.j_range; T=T)
    coarse_out = similar(coarse_F)
    regrid_conservative_tree_patch_direct_F_2d!(coarse_out, patch_out, coarse_F, patch)
    return (coarse_F=coarse_out, patch=patch_out)
end

"""
    adapt_conservative_tree_patch_to_solid_mask_2d(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.adapt_conservative_tree_patch_to_solid_mask_2d)
```
"""
function adapt_conservative_tree_patch_to_solid_mask_2d(
        coarse_F::AbstractArray{T,3},
        patch::ConservativeTreePatch2D{T},
        is_solid::AbstractArray{Bool,2};
        pad::Int=1) where T
    _check_composite_coarse_layout(coarse_F, patch)
    size(is_solid) == (2 * size(coarse_F, 1), 2 * size(coarse_F, 2)) ||
        throw(ArgumentError("is_solid must have size (2*Nx, 2*Ny)"))
    ranges = conservative_tree_solid_mask_patch_range_2d(is_solid; pad=pad)
    patch_out = create_conservative_tree_patch_2d(
        ranges.i_range, ranges.j_range; T=T)
    coarse_out = similar(coarse_F)
    regrid_conservative_tree_patch_direct_F_2d!(coarse_out, patch_out, coarse_F, patch)
    return (coarse_F=coarse_out, patch=patch_out)
end

"""
    vertical_facing_step_solid_mask_leaf_2d(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.vertical_facing_step_solid_mask_leaf_2d)
```
"""
function vertical_facing_step_solid_mask_leaf_2d(
        Nx::Int,
        Ny::Int,
        step_i_range::AbstractUnitRange{<:Integer},
        step_height::Int)
    Nx > 0 || throw(ArgumentError("Nx must be positive"))
    Ny > 0 || throw(ArgumentError("Ny must be positive"))
    isempty(step_i_range) && throw(ArgumentError("step_i_range must be nonempty"))
    first(step_i_range) >= 1 && last(step_i_range) <= Nx ||
        throw(ArgumentError("step_i_range must be inside 1:Nx"))
    1 <= step_height < Ny ||
        throw(ArgumentError("step_height must be inside 1:Ny-1"))

    mask = falses(Nx, Ny)
    @inbounds for j in 1:step_height, i in Int(first(step_i_range)):Int(last(step_i_range))
        mask[i, j] = true
    end
    return mask
end

