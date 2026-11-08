function _check_composite_pair_layout(coarse_out::AbstractArray{<:Any,3},
                                      patch_out::ConservativeTreePatch2D,
                                      coarse_in::AbstractArray{<:Any,3},
                                      patch_in::ConservativeTreePatch2D)
    size(coarse_out) == size(coarse_in) ||
        throw(ArgumentError("coarse_out and coarse_in must have the same size"))
    patch_out.parent_i_range == patch_in.parent_i_range ||
        throw(ArgumentError("patch_out and patch_in must have the same parent_i_range"))
    patch_out.parent_j_range == patch_in.parent_j_range ||
        throw(ArgumentError("patch_out and patch_in must have the same parent_j_range"))
    size(patch_out.fine_F) == size(patch_in.fine_F) ||
        throw(ArgumentError("patch_out and patch_in fine arrays must have the same size"))
    _check_composite_coarse_layout(coarse_in, patch_in)
    _check_composite_coarse_layout(coarse_out, patch_out)
    return nothing
end

function _check_leaf_layout(leaf_F::AbstractArray{<:Any,3},
                            coarse_F::AbstractArray{<:Any,3})
    size(leaf_F) == (2 * size(coarse_F, 1), 2 * size(coarse_F, 2), 9) ||
        throw(ArgumentError("leaf_F must have size (2*Nx_coarse, 2*Ny_coarse, 9)"))
    return nothing
end

"""
    composite_to_leaf_F_2d!(leaf_F, coarse_F, patch)

Expand a composite fixed-tree state to a uniform leaf grid. Active fine leaves
are copied inside `patch`; active coarse cells outside the patch are uniformly
exploded to their four ratio-2 leaves. Inactive coarse cells covered by the
patch are ignored.
"""
function composite_to_leaf_F_2d!(leaf_F::AbstractArray{<:Any,3},
                                 coarse_F::AbstractArray{<:Any,3},
                                 patch::ConservativeTreePatch2D)
    _check_composite_coarse_layout(coarse_F, patch)
    _check_leaf_layout(leaf_F, coarse_F)
    coalesce_patch_to_shadow_F_2d!(patch)

    @inbounds for J in axes(coarse_F, 2), I in axes(coarse_F, 1)
        i0 = 2 * I - 1
        j0 = 2 * J - 1
        leaf_block = @view leaf_F[i0:i0+1, j0:j0+1, :]

        if _inside_range(I, J, patch.parent_i_range, patch.parent_j_range)
            il, jl = _patch_local_parent_index(patch, I, J)
            fine_block = _child_block_view(patch.fine_F, il, jl)
            leaf_block .= fine_block
        else
            _explode_limited_linear_composite_F_2d!(leaf_block, coarse_F, patch, I, J)
        end
    end
    return leaf_F
end

"""
    leaf_to_composite_F_2d!(coarse_F, patch, leaf_F)

Restrict a uniform leaf grid back to the composite fixed-tree representation.
Outside `patch`, each 2x2 leaf block is coalesced to an active coarse cell.
Inside `patch`, the leaf values are copied to active fine leaves. The inactive
coarse parent region is zeroed in `coarse_F`.
"""
function leaf_to_composite_F_2d!(coarse_F::AbstractArray{<:Any,3},
                                 patch::ConservativeTreePatch2D,
                                 leaf_F::AbstractArray{<:Any,3})
    _check_composite_coarse_layout(coarse_F, patch)
    _check_leaf_layout(leaf_F, coarse_F)
    coarse_F .= 0

    @inbounds for J in axes(coarse_F, 2), I in axes(coarse_F, 1)
        i0 = 2 * I - 1
        j0 = 2 * J - 1
        leaf_block = @view leaf_F[i0:i0+1, j0:j0+1, :]

        if _inside_range(I, J, patch.parent_i_range, patch.parent_j_range)
            il, jl = _patch_local_parent_index(patch, I, J)
            fine_block = _child_block_view(patch.fine_F, il, jl)
            fine_block .= leaf_block
        else
            coalesce_F_2d!(@view(coarse_F[I, J, :]), leaf_block)
        end
    end

    coalesce_patch_to_shadow_F_2d!(patch)
    return coarse_F, patch
end

"""
    stream_composite_fully_periodic_leaf_F_2d!(coarse_out, patch_out,
                                               coarse_in, patch_in)

Conservative prototype stream step for a fixed-tree composite state. The state
is expanded to a uniform ratio-2 leaf grid, streamed periodically by one leaf
cell, then restricted back to coarse-outside/fine-inside ownership.

This is a topology/invariant canary, not the final physical subcycled
coarse/fine time integrator.
"""
function stream_composite_fully_periodic_leaf_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D)
    _check_composite_pair_layout(coarse_out, patch_out, coarse_in, patch_in)

    leaf_in = similar(coarse_in, 2 * size(coarse_in, 1), 2 * size(coarse_in, 2), 9)
    leaf_out = similar(leaf_in)
    composite_to_leaf_F_2d!(leaf_in, coarse_in, patch_in)
    stream_fully_periodic_F_2d!(leaf_out, leaf_in)
    leaf_to_composite_F_2d!(coarse_out, patch_out, leaf_out)
    return coarse_out, patch_out
end

"""
    stream_composite_periodic_x_wall_y_leaf_F_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.stream_composite_periodic_x_wall_y_leaf_F_2d!)
```
"""
function stream_composite_periodic_x_wall_y_leaf_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D)
    _check_composite_pair_layout(coarse_out, patch_out, coarse_in, patch_in)

    leaf_in = similar(coarse_in, 2 * size(coarse_in, 1), 2 * size(coarse_in, 2), 9)
    leaf_out = similar(leaf_in)
    composite_to_leaf_F_2d!(leaf_in, coarse_in, patch_in)
    stream_periodic_x_wall_y_F_2d!(leaf_out, leaf_in)
    leaf_to_composite_F_2d!(coarse_out, patch_out, leaf_out)
    return coarse_out, patch_out
end

"""
    stream_composite_periodic_x_moving_wall_y_leaf_F_2d!(

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.stream_composite_periodic_x_moving_wall_y_leaf_F_2d!)
```
"""
function stream_composite_periodic_x_moving_wall_y_leaf_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D;
        u_south=0,
        u_north=0,
        rho_wall=1,
        volume_leaf=1)
    _check_composite_pair_layout(coarse_out, patch_out, coarse_in, patch_in)

    leaf_in = similar(coarse_in, 2 * size(coarse_in, 1), 2 * size(coarse_in, 2), 9)
    leaf_out = similar(leaf_in)
    composite_to_leaf_F_2d!(leaf_in, coarse_in, patch_in)
    stream_periodic_x_moving_wall_y_F_2d!(leaf_out, leaf_in;
        u_south=u_south, u_north=u_north, rho_wall=rho_wall, volume=volume_leaf)
    leaf_to_composite_F_2d!(coarse_out, patch_out, leaf_out)
    return coarse_out, patch_out
end

"""
    collide_BGK_composite_F_2d!(coarse_F, patch, volume_coarse, volume_fine,
                                omega_coarse, omega_fine)

Collide only active cells of a fixed-tree composite state: coarse cells outside
the refined parent range and fine leaves inside `patch`.
"""
function collide_BGK_composite_F_2d!(coarse_F::AbstractArray{<:Any,3},
                                     patch::ConservativeTreePatch2D,
                                     volume_coarse,
                                     volume_fine,
                                     omega_coarse,
                                     omega_fine)
    _check_composite_coarse_layout(coarse_F, patch)

    @inbounds for j in axes(coarse_F, 2), i in axes(coarse_F, 1)
        _inside_range(i, j, patch.parent_i_range, patch.parent_j_range) && continue
        collide_BGK_integrated_D2Q9!(@view(coarse_F[i, j, :]), volume_coarse, omega_coarse)
    end
    collide_BGK_integrated_D2Q9!(patch.fine_F, volume_fine, omega_fine)
    coalesce_patch_to_shadow_F_2d!(patch)
    return coarse_F, patch
end

"""
    collide_Guo_composite_F_2d!(coarse_F::AbstractArray{<:Any,3},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_Guo_composite_F_2d!)
```
"""
function collide_Guo_composite_F_2d!(coarse_F::AbstractArray{<:Any,3},
                                     patch::ConservativeTreePatch2D,
                                     volume_coarse,
                                     volume_fine,
                                     omega_coarse,
                                     omega_fine,
                                     Fx,
                                     Fy)
    _check_composite_coarse_layout(coarse_F, patch)

    @inbounds for j in axes(coarse_F, 2), i in axes(coarse_F, 1)
        _inside_range(i, j, patch.parent_i_range, patch.parent_j_range) && continue
        collide_Guo_integrated_D2Q9!(@view(coarse_F[i, j, :]), volume_coarse,
                                     omega_coarse, Fx, Fy)
    end
    collide_Guo_integrated_D2Q9!(patch.fine_F, volume_fine, omega_fine, Fx, Fy)
    coalesce_patch_to_shadow_F_2d!(patch)
    return coarse_F, patch
end

"""
    ConservativeTreeMacroFlow2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeMacroFlow2D
```
"""
struct ConservativeTreeMacroFlow2D{T}
    flow::Symbol
    coarse_F::Array{T,3}
    patch::ConservativeTreePatch2D{T}
    ux_profile::Vector{T}
    analytic_ux_profile::Vector{T}
    l2_error::T
    linf_error::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    steps::Int
end

"""
    ConservativeTreeCylinderResult2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeCylinderResult2D
```
"""
struct ConservativeTreeCylinderResult2D{T}
    coarse_F::Array{T,3}
    patch::ConservativeTreePatch2D{T}
    is_solid_leaf::BitMatrix
    Fx_drag::T
    Fy_drag::T
    Cd::T
    u_ref::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    steps::Int
    avg_window::Int
end

"""
    ConservativeTreeCylinderChannelResult2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeCylinderChannelResult2D
```
"""
struct ConservativeTreeCylinderChannelResult2D{T}
    coarse_F::Array{T,3}
    patch::ConservativeTreePatch2D{T}
    is_solid_leaf::BitMatrix
    Fx_drag::T
    Fy_drag::T
    Cd::T
    u_in::T
    ux_mean::T
    omega::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    steps::Int
    avg_window::Int
end

"""
    ConservativeTreeSolidFlowResult2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeSolidFlowResult2D
```
"""
struct ConservativeTreeSolidFlowResult2D{T}
    flow::Symbol
    coarse_F::Array{T,3}
    patch::ConservativeTreePatch2D{T}
    is_solid_leaf::BitMatrix
    ux_mean::T
    uy_mean::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    steps::Int
end

"""
    composite_leaf_mean_ux_profile(coarse_F::AbstractArray{T,3},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.composite_leaf_mean_ux_profile)
```
"""
function composite_leaf_mean_ux_profile(coarse_F::AbstractArray{T,3},
                                        patch::ConservativeTreePatch2D{T};
                                        volume_leaf::T=T(0.25),
                                        force_x::T=zero(T)) where T
    leaf = zeros(T, 2 * size(coarse_F, 1), 2 * size(coarse_F, 2), 9)
    composite_to_leaf_F_2d!(leaf, coarse_F, patch)

    profile = zeros(T, size(leaf, 2))
    @inbounds for j in axes(leaf, 2)
        ux_sum = zero(T)
        for i in axes(leaf, 1)
            cell = @view leaf[i, j, :]
            rho = mass_F(cell) / volume_leaf
            ux_sum += (momentum_F(cell)[1] / volume_leaf + force_x / 2) / rho
        end
        profile[j] = ux_sum / T(size(leaf, 1))
    end
    return profile
end

"""
    composite_leaf_velocity_field_2d(coarse_F::AbstractArray{T,3},

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.composite_leaf_velocity_field_2d)
```
"""
function composite_leaf_velocity_field_2d(coarse_F::AbstractArray{T,3},
                                          patch::ConservativeTreePatch2D{T};
                                          volume_leaf::T=T(0.25),
                                          force_x::T=zero(T),
                                          force_y::T=zero(T)) where T
    leaf = zeros(T, 2 * size(coarse_F, 1), 2 * size(coarse_F, 2), 9)
    composite_to_leaf_F_2d!(leaf, coarse_F, patch)

    ux = zeros(T, size(leaf, 1), size(leaf, 2))
    uy = similar(ux)
    @inbounds for j in axes(leaf, 2), i in axes(leaf, 1)
        cell = @view leaf[i, j, :]
        rho = mass_F(cell) / volume_leaf
        rho > zero(T) || throw(ArgumentError("leaf cell density must be positive"))
        mx, my = momentum_F(cell)
        ux[i, j] = (mx / volume_leaf + force_x / 2) / rho
        uy[i, j] = (my / volume_leaf + force_y / 2) / rho
    end
    return (ux=ux, uy=uy)
end

"""
    couette_analytic_profile_2d(ny::Int, U)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.couette_analytic_profile_2d)
```
"""
function couette_analytic_profile_2d(ny::Int, U)
    ny >= 2 || throw(ArgumentError("ny must be >= 2"))
    T = typeof(float(U))
    return [T(U) * T(j - 1) / T(ny - 1) for j in 1:ny]
end

"""
    poiseuille_analytic_profile_2d(ny::Int, Fx, omega; rho=1)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.poiseuille_analytic_profile_2d)
```
"""
function poiseuille_analytic_profile_2d(ny::Int, Fx, omega; rho=1)
    ny >= 2 || throw(ArgumentError("ny must be >= 2"))
    T = promote_type(typeof(float(Fx)), typeof(float(omega)), typeof(float(rho)))
    nu = (one(T) / T(omega) - T(0.5)) / T(3)
    H = T(ny)
    return [
        T(Fx) / (T(2) * T(rho) * nu) *
        (T(j) - T(0.5)) * (H + T(0.5) - T(j))
        for j in 1:ny
    ]
end

function _profile_errors(profile::AbstractVector, reference::AbstractVector)
    length(profile) == length(reference) ||
        throw(ArgumentError("profile and reference must have the same length"))
    T = promote_type(eltype(profile), eltype(reference))
    l2 = sqrt(sum((T(profile[i]) - T(reference[i]))^2 for i in eachindex(profile)) /
              T(length(profile)))
    linf = maximum(abs(T(profile[i]) - T(reference[i])) for i in eachindex(profile))
    return l2, linf
end

function _leaf_fluid_mass_F(F::AbstractArray{T,3},
                            is_solid::AbstractArray{Bool,2}) where T
    _check_solid_mask_layout(F, is_solid)
    total = zero(T)
    @inbounds for q in 1:9, j in axes(F, 2), i in axes(F, 1)
        is_solid[i, j] && continue
        total += F[i, j, q]
    end
    return total
end

function _leaf_fluid_mean_ux_F(F::AbstractArray{T,3},
                               is_solid::AbstractArray{Bool,2};
                               volume::T,
                               force_x::T=zero(T)) where T
    _check_solid_mask_layout(F, is_solid)
    ux_sum = zero(T)
    n_fluid = 0
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        is_solid[i, j] && continue
        cell = @view F[i, j, :]
        rho = mass_F(cell) / volume
        ux_sum += (momentum_F(cell)[1] / volume + force_x / 2) / rho
        n_fluid += 1
    end
    n_fluid > 0 || throw(ArgumentError("solid mask leaves no fluid cells"))
    return ux_sum / T(n_fluid)
end

function _leaf_fluid_mean_velocity_F(F::AbstractArray{T,3},
                                     is_solid::AbstractArray{Bool,2};
                                     volume::T,
                                     force_x::T=zero(T),
                                     force_y::T=zero(T)) where T
    _check_solid_mask_layout(F, is_solid)
    ux_sum = zero(T)
    uy_sum = zero(T)
    n_fluid = 0
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        is_solid[i, j] && continue
        cell = @view F[i, j, :]
        rho = mass_F(cell) / volume
        p = momentum_F(cell)
        ux_sum += (p[1] / volume + force_x / 2) / rho
        uy_sum += (p[2] / volume + force_y / 2) / rho
        n_fluid += 1
    end
    n_fluid > 0 || throw(ArgumentError("solid mask leaves no fluid cells"))
    return ux_sum / T(n_fluid), uy_sum / T(n_fluid)
end

"""
    run_conservative_tree_couette_macroflow_2d(;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.run_conservative_tree_couette_macroflow_2d)
```
"""
