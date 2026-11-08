"""
    run_conservative_tree_vfs_route_native_2d(;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.run_conservative_tree_vfs_route_native_2d)
```
"""
function run_conservative_tree_vfs_route_native_2d(;
        Nx::Int=28,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=5:12,
        patch_j_range::UnitRange{Int}=1:8,
        step_i_range::UnitRange{Int}=14:19,
        step_height_leaf::Int=8,
        Fx=2e-5,
        Fy=0.0,
        omega=1.0,
        rho=1.0,
        steps::Int=900,
        T::Type{<:Real}=Float64)
    Fx = T(Fx)
    Fy = T(Fy)
    omega = T(omega)
    rho = T(rho)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    is_solid = vertical_facing_step_solid_mask_leaf_2d(
        2 * Nx, 2 * Ny, step_i_range, step_height_leaf)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(Nx, Ny, patch)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)

    _check_route_solid_mask_layout(topology, coarse, patch, is_solid)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    for _ in 1:steps
        collide_Guo_composite_solid_F_2d!(
            coarse, patch, topology, is_solid,
            volume_coarse, volume_fine, omega, omega, Fx, Fy)
        stream_composite_routes_periodic_x_wall_y_solid_F_2d!(
            coarse_next, patch_next, coarse, patch, topology, is_solid;
            coarse_prolongation=:limited_linear)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_final = _leaf_fluid_mass_F(leaf, is_solid)
    ux_mean, uy_mean = _leaf_fluid_mean_velocity_F(
        leaf, is_solid; volume=volume_fine, force_x=Fx, force_y=Fy)

    return ConservativeTreeSolidFlowResult2D{T}(
        :vfs_route_native, coarse, patch, is_solid, ux_mean, uy_mean,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

"""
    run_conservative_tree_vfs_mask_adaptive_route_native_2d(;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.run_conservative_tree_vfs_mask_adaptive_route_native_2d)
```
"""
function run_conservative_tree_vfs_mask_adaptive_route_native_2d(;
        Nx::Int=28,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=5:12,
        patch_j_range::UnitRange{Int}=1:8,
        step_i_range::UnitRange{Int}=14:19,
        step_height_leaf::Int=8,
        regrid_every::Int=120,
        pad::Int=1,
        Fx=2e-5,
        Fy=0.0,
        omega=1.0,
        rho=1.0,
        steps::Int=500,
        T::Type{<:Real}=Float64)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    regrid_every > 0 || throw(ArgumentError("regrid_every must be positive"))
    Fx = T(Fx)
    Fy = T(Fy)
    omega = T(omega)
    rho = T(rho)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    is_solid = vertical_facing_step_solid_mask_leaf_2d(
        2 * Nx, 2 * Ny, step_i_range, step_height_leaf)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(Nx, Ny, patch)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)

    _check_route_solid_mask_layout(topology, coarse, patch, is_solid)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    patch_history = Tuple{UnitRange{Int},UnitRange{Int}}[
        (patch.parent_i_range, patch.parent_j_range)
    ]
    regrid_count = 0

    for step in 1:steps
        collide_Guo_composite_solid_F_2d!(
            coarse, patch, topology, is_solid,
            volume_coarse, volume_fine, omega, omega, Fx, Fy)
        stream_composite_routes_periodic_x_wall_y_solid_F_2d!(
            coarse_next, patch_next, coarse, patch, topology, is_solid;
            coarse_prolongation=:limited_linear)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch

        if step < steps && step % regrid_every == 0
            ranges = conservative_tree_solid_mask_patch_range_2d(is_solid; pad=pad)
            if ranges.i_range != patch.parent_i_range ||
                    ranges.j_range != patch.parent_j_range
                adapted = adapt_conservative_tree_patch_to_solid_mask_2d(
                    coarse, patch, is_solid; pad=pad)
                coarse = adapted.coarse_F
                patch = adapted.patch
                patch_next = create_conservative_tree_patch_2d(
                    ranges.i_range, ranges.j_range; T=T)
                coarse_next = similar(coarse)
                topology = create_conservative_tree_topology_2d(Nx, Ny, patch)
                _check_route_solid_mask_layout(topology, coarse, patch, is_solid)
                regrid_count += 1
                push!(patch_history, (patch.parent_i_range, patch.parent_j_range))
            end
        end
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_final = _leaf_fluid_mass_F(leaf, is_solid)
    ux_mean, uy_mean = _leaf_fluid_mean_velocity_F(
        leaf, is_solid; volume=volume_fine, force_x=Fx, force_y=Fy)

    return ConservativeTreeSolidAdaptiveRun2D{T}(
        :vfs_mask_adaptive_route_native, coarse, patch, is_solid, patch_history,
        ux_mean, uy_mean, mass_initial, mass_final, mass_final - mass_initial,
        steps, regrid_every, regrid_count)
end
