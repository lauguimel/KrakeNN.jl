function run_conservative_tree_couette_macroflow_2d(;
        Nx::Int=16,
        Ny::Int=12,
        patch_i_range::UnitRange{Int}=6:11,
        patch_j_range::UnitRange{Int}=4:9,
        U=0.04,
        omega=1.0,
        rho=1.0,
        steps::Int=3000,
        T::Type{<:Real}=Float64)
    U = T(U)
    omega = T(omega)
    rho = T(rho)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    mass_initial = active_mass_F(coarse, patch)

    for _ in 1:steps
        collide_BGK_composite_F_2d!(coarse, patch, volume_coarse, volume_fine, omega, omega)
        stream_composite_periodic_x_moving_wall_y_leaf_F_2d!(
            coarse_next, patch_next, coarse, patch;
            u_north=U, volume_leaf=volume_fine, rho_wall=rho)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    profile = composite_leaf_mean_ux_profile(coarse, patch; volume_leaf=volume_fine)
    analytic = couette_analytic_profile_2d(length(profile), U)
    l2, linf = _profile_errors(profile, analytic)
    mass_final = active_mass_F(coarse, patch)

    return ConservativeTreeMacroFlow2D{T}(
        :couette, coarse, patch, profile, analytic, l2, linf,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

"""
    run_conservative_tree_poiseuille_macroflow_2d(;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.run_conservative_tree_poiseuille_macroflow_2d)
```
"""
function run_conservative_tree_poiseuille_macroflow_2d(;
        Nx::Int=18,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=7:12,
        patch_j_range::UnitRange{Int}=5:10,
        Fx=5e-5,
        omega=1.0,
        rho=1.0,
        steps::Int=5000,
        T::Type{<:Real}=Float64)
    Fx = T(Fx)
    omega = T(omega)
    rho = T(rho)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    mass_initial = active_mass_F(coarse, patch)

    for _ in 1:steps
        collide_Guo_composite_F_2d!(coarse, patch, volume_coarse, volume_fine,
                                    omega, omega, Fx, zero(T))
        stream_composite_periodic_x_wall_y_leaf_F_2d!(coarse_next, patch_next,
                                                      coarse, patch)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    profile = composite_leaf_mean_ux_profile(coarse, patch;
                                             volume_leaf=volume_fine,
                                             force_x=Fx)
    analytic = poiseuille_analytic_profile_2d(length(profile), Fx, omega; rho=rho)
    l2, linf = _profile_errors(profile, analytic)
    mass_final = active_mass_F(coarse, patch)

    return ConservativeTreeMacroFlow2D{T}(
        :poiseuille, coarse, patch, profile, analytic, l2, linf,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

function _run_conservative_tree_periodic_solid_force_macroflow_2d(
        flow::Symbol,
        is_solid::BitMatrix;
        Nx::Int,
        Ny::Int,
        patch_i_range::UnitRange{Int},
        patch_j_range::UnitRange{Int},
        Fx,
        Fy,
        omega,
        rho,
        steps::Int,
        T::Type{<:Real})
    Fx = T(Fx)
    Fy = T(Fy)
    omega = T(omega)
    rho = T(rho)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    volume_coarse = one(T)
    volume_fine = T(0.25)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)
    leaf_post = similar(leaf)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    for _ in 1:steps
        composite_to_leaf_F_2d!(leaf, coarse, patch)
        collide_Guo_integrated_D2Q9!(leaf, is_solid, volume_fine, omega, Fx, Fy)
        stream_periodic_x_wall_y_solid_F_2d!(leaf_post, leaf, is_solid)
        leaf_to_composite_F_2d!(coarse_next, patch_next, leaf_post)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_final = _leaf_fluid_mass_F(leaf, is_solid)
    ux_mean, uy_mean = _leaf_fluid_mean_velocity_F(
        leaf, is_solid; volume=volume_fine, force_x=Fx, force_y=Fy)
    return ConservativeTreeSolidFlowResult2D{T}(
        flow, coarse, patch, is_solid, ux_mean, uy_mean,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

function _run_conservative_tree_open_solid_macroflow_2d(
        flow::Symbol,
        is_solid::BitMatrix;
        Nx::Int,
        Ny::Int,
        patch_i_range::UnitRange{Int},
        patch_j_range::UnitRange{Int},
        u_in,
        omega,
        rho,
        steps::Int,
        T::Type{<:Real})
    u_in = T(u_in)
    omega = T(omega)
    rho = T(rho)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    volume_coarse = one(T)
    volume_fine = T(0.25)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)
    leaf_post = similar(leaf)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, u_in, zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, u_in, zero(T))
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    for _ in 1:steps
        composite_to_leaf_F_2d!(leaf, coarse, patch)
        stream_bounceback_xy_solid_F_2d!(leaf_post, leaf, is_solid)
        apply_zou_he_west_F_2d!(leaf_post, u_in, volume_fine, is_solid)
        apply_zou_he_pressure_east_F_2d!(leaf_post, volume_fine, is_solid; rho_out=rho)
        collide_BGK_integrated_D2Q9!(leaf_post, is_solid, volume_fine, omega)
        leaf_to_composite_F_2d!(coarse_next, patch_next, leaf_post)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_final = _leaf_fluid_mass_F(leaf, is_solid)
    ux_mean, uy_mean = _leaf_fluid_mean_velocity_F(leaf, is_solid; volume=volume_fine)
    return ConservativeTreeSolidFlowResult2D{T}(
        flow, coarse, patch, is_solid, ux_mean, uy_mean,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

"""
    run_conservative_tree_square_obstacle_macroflow_2d(;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.run_conservative_tree_square_obstacle_macroflow_2d)
```
"""
function run_conservative_tree_square_obstacle_macroflow_2d(;
        Nx::Int=24,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=8:17,
        patch_j_range::UnitRange{Int}=4:11,
        obstacle_i_range::UnitRange{Int}=22:27,
        obstacle_j_range::UnitRange{Int}=12:17,
        Fx=2e-5,
        Fy=0.0,
        omega=1.0,
        rho=1.0,
        steps::Int=1200,
        T::Type{<:Real}=Float64)
    is_solid = square_solid_mask_leaf_2d(2 * Nx, 2 * Ny,
                                         obstacle_i_range, obstacle_j_range)
    return _run_conservative_tree_periodic_solid_force_macroflow_2d(
        :square_obstacle, is_solid; Nx=Nx, Ny=Ny,
        patch_i_range=patch_i_range, patch_j_range=patch_j_range,
        Fx=Fx, Fy=Fy, omega=omega, rho=rho, steps=steps, T=T)
end

"""
    run_conservative_tree_bfs_macroflow_2d(;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.run_conservative_tree_bfs_macroflow_2d)
```
"""
function run_conservative_tree_bfs_macroflow_2d(;
        Nx::Int=28,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=1:12,
        patch_j_range::UnitRange{Int}=1:8,
        step_i_leaf::Int=16,
        step_height_leaf::Int=8,
        u_in=0.03,
        omega=1.0,
        rho=1.0,
        steps::Int=800,
        T::Type{<:Real}=Float64)
    is_solid = backward_facing_step_solid_mask_leaf_2d(2 * Nx, 2 * Ny,
                                                       step_i_leaf,
                                                       step_height_leaf)
    return _run_conservative_tree_open_solid_macroflow_2d(
        :bfs, is_solid; Nx=Nx, Ny=Ny,
        patch_i_range=patch_i_range, patch_j_range=patch_j_range,
        u_in=u_in, omega=omega, rho=rho, steps=steps, T=T)
end

"""
    run_conservative_tree_cylinder_macroflow_2d(;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.run_conservative_tree_cylinder_macroflow_2d)
```
"""
function run_conservative_tree_cylinder_macroflow_2d(;
        Nx::Int=24,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=8:17,
        patch_j_range::UnitRange{Int}=4:11,
        cx_leaf=(2 * Nx + 1) / 2,
        cy_leaf=(2 * Ny + 1) / 2,
        radius_leaf=3.0,
        Fx=2e-5,
        omega=1.0,
        rho=1.0,
        steps::Int=1200,
        avg_window::Int=300,
        T::Type{<:Real}=Float64)
    Fx = T(Fx)
    omega = T(omega)
    rho = T(rho)
    radius_leaf = T(radius_leaf)
    cx_leaf = T(cx_leaf)
    cy_leaf = T(cy_leaf)

    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    volume_coarse = one(T)
    volume_fine = T(0.25)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)
    leaf_post = similar(leaf)
    is_solid = cylinder_solid_mask_leaf_2d(size(leaf, 1), size(leaf, 2),
                                           cx_leaf, cy_leaf, radius_leaf)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    Fx_sum = zero(T)
    Fy_sum = zero(T)
    n_avg = 0
    for step in 1:steps
        composite_to_leaf_F_2d!(leaf, coarse, patch)
        collide_Guo_integrated_D2Q9!(leaf, is_solid, volume_fine, omega, Fx, zero(T))
        stream_periodic_x_wall_y_solid_F_2d!(leaf_post, leaf, is_solid)

        if step > steps - avg_window
            drag = compute_drag_mea_solid_F_2d(leaf, leaf_post, is_solid)
            Fx_sum += T(drag.Fx)
            Fy_sum += T(drag.Fy)
            n_avg += 1
        end

        leaf_to_composite_F_2d!(coarse_next, patch_next, leaf_post)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_final = _leaf_fluid_mass_F(leaf, is_solid)
    u_ref = _leaf_fluid_mean_ux_F(leaf, is_solid; volume=volume_fine, force_x=Fx)
    Fx_drag = Fx_sum / T(n_avg)
    Fy_drag = Fy_sum / T(n_avg)
    diameter = T(2) * radius_leaf
    Cd = T(2) * Fx_drag / (rho * max(abs(u_ref), eps(T))^2 * diameter)

    return ConservativeTreeCylinderResult2D{T}(
        coarse, patch, is_solid, Fx_drag, Fy_drag, Cd, u_ref,
        mass_initial, mass_final, mass_final - mass_initial, steps, avg_window)
end

"""
    run_conservative_tree_cylinder_channel_macroflow_2d(;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.run_conservative_tree_cylinder_channel_macroflow_2d)
```
"""
function run_conservative_tree_cylinder_channel_macroflow_2d(;
        Nx::Int=24,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=4:9,
        patch_j_range::UnitRange{Int}=4:11,
        cx_leaf=(2 * Nx) / 4,
        cy_leaf=(2 * Ny + 1) / 2,
        radius_leaf=3.0,
        u_in=0.03,
        Re=nothing,
        omega=1.0,
        rho=1.0,
        steps::Int=2000,
        avg_window::Int=500,
        T::Type{<:Real}=Float64)
    u_in = T(u_in)
    rho = T(rho)
    radius_leaf = T(radius_leaf)
    cx_leaf = T(cx_leaf)
    cy_leaf = T(cy_leaf)
    if Re === nothing
        omega = T(omega)
    else
        nu = u_in * T(2) * radius_leaf / T(Re)
        omega = one(T) / (T(3) * nu + T(0.5))
    end

    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    volume_coarse = one(T)
    volume_fine = T(0.25)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)
    leaf_post = similar(leaf)
    is_solid = cylinder_solid_mask_leaf_2d(size(leaf, 1), size(leaf, 2),
                                           cx_leaf, cy_leaf, radius_leaf)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, u_in, zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, u_in, zero(T))
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    Fx_sum = zero(T)
    Fy_sum = zero(T)
    n_avg = 0
    for step in 1:steps
        composite_to_leaf_F_2d!(leaf, coarse, patch)
        stream_bounceback_xy_solid_F_2d!(leaf_post, leaf, is_solid)
        apply_zou_he_west_F_2d!(leaf_post, u_in, volume_fine, is_solid)
        apply_zou_he_pressure_east_F_2d!(leaf_post, volume_fine, is_solid; rho_out=rho)

        if step > steps - avg_window
            drag = compute_drag_mea_solid_F_2d(leaf, leaf_post, is_solid)
            Fx_sum += T(drag.Fx)
            Fy_sum += T(drag.Fy)
            n_avg += 1
        end

        collide_BGK_integrated_D2Q9!(leaf_post, is_solid, volume_fine, omega)
        leaf_to_composite_F_2d!(coarse_next, patch_next, leaf_post)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_final = _leaf_fluid_mass_F(leaf, is_solid)
    ux_mean, _ = _leaf_fluid_mean_velocity_F(leaf, is_solid; volume=volume_fine)
    Fx_drag = Fx_sum / T(n_avg)
    Fy_drag = Fy_sum / T(n_avg)
    diameter = T(2) * radius_leaf
    Cd = T(2) * Fx_drag / (rho * u_in^2 * diameter)

    return ConservativeTreeCylinderChannelResult2D{T}(
        coarse, patch, is_solid, Fx_drag, Fy_drag, Cd, u_in, ux_mean, omega,
        mass_initial, mass_final, mass_final - mass_initial, steps, avg_window)
end
