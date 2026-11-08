"""
    validate_conservative_tree_route_native_phase_p_2d(;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.validate_conservative_tree_route_native_phase_p_2d)
```
"""
function validate_conservative_tree_route_native_phase_p_2d(;
        steps::Int=1000,
        T::Type{<:Real}=Float64)
    couette_route = run_conservative_tree_couette_route_native_2d(
        ; steps=steps, T=T)
    couette_oracle = run_conservative_tree_couette_macroflow_2d(
        ; Nx=size(couette_route.coarse_F, 1),
        Ny=size(couette_route.coarse_F, 2),
        patch_i_range=couette_route.patch.parent_i_range,
        patch_j_range=couette_route.patch.parent_j_range,
        steps=steps, T=T)
    couette_l2, couette_linf = _profile_errors(
        couette_route.ux_profile, couette_oracle.ux_profile)

    poiseuille_route = run_conservative_tree_poiseuille_route_native_2d(
        ; steps=steps, T=T)
    poiseuille_oracle = run_conservative_tree_poiseuille_macroflow_2d(
        ; Nx=size(poiseuille_route.coarse_F, 1),
        Ny=size(poiseuille_route.coarse_F, 2),
        patch_i_range=poiseuille_route.patch.parent_i_range,
        patch_j_range=poiseuille_route.patch.parent_j_range,
        steps=steps, T=T)
    poiseuille_l2, poiseuille_linf = _profile_errors(
        poiseuille_route.ux_profile, poiseuille_oracle.ux_profile)

    return (
        couette=(
            route=couette_route,
            oracle=couette_oracle,
            l2_delta=couette_l2,
            linf_delta=couette_linf,
        ),
        poiseuille=(
            route=poiseuille_route,
            oracle=poiseuille_oracle,
            l2_delta=poiseuille_l2,
            linf_delta=poiseuille_linf,
        ),
    )
end

"""
    run_conservative_tree_square_obstacle_route_native_2d(;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.run_conservative_tree_square_obstacle_route_native_2d)
```
"""
function run_conservative_tree_square_obstacle_route_native_2d(;
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
        coarse_route_mode::Symbol=:leaf_equivalent,
        T::Type{<:Real}=Float64)
    Fx = T(Fx)
    Fy = T(Fy)
    omega = T(omega)
    rho = T(rho)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    is_solid = square_solid_mask_leaf_2d(2 * Nx, 2 * Ny,
                                         obstacle_i_range, obstacle_j_range)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(
        Nx, Ny, patch; coarse_route_mode=coarse_route_mode)
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
        :square_obstacle_route_native, coarse, patch, is_solid, ux_mean, uy_mean,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

"""
    run_conservative_tree_cylinder_obstacle_route_native_2d(;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.run_conservative_tree_cylinder_obstacle_route_native_2d)
```
"""
function run_conservative_tree_cylinder_obstacle_route_native_2d(;
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
        coarse_route_mode::Symbol=:leaf_equivalent,
        T::Type{<:Real}=Float64)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    avg_window > 0 || throw(ArgumentError("avg_window must be positive"))
    Fx = T(Fx)
    omega = T(omega)
    rho = T(rho)
    cx_leaf = T(cx_leaf)
    cy_leaf = T(cy_leaf)
    radius_leaf = T(radius_leaf)
    volume_coarse = one(T)
    volume_fine = T(0.25)
    avg_window_i = min(avg_window, steps)

    is_solid = cylinder_solid_mask_leaf_2d(2 * Nx, 2 * Ny,
                                           cx_leaf, cy_leaf, radius_leaf)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(
        Nx, Ny, patch; coarse_route_mode=coarse_route_mode)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)
    leaf_next = similar(leaf)

    _check_route_solid_mask_layout(topology, coarse, patch, is_solid)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    Fx_sum = zero(T)
    Fy_sum = zero(T)
    n_avg = 0
    for step in 1:steps
        collide_Guo_composite_solid_F_2d!(
            coarse, patch, topology, is_solid,
            volume_coarse, volume_fine, omega, omega, Fx, zero(T))
        stream_composite_routes_periodic_x_wall_y_solid_F_2d!(
            coarse_next, patch_next, coarse, patch, topology, is_solid;
            coarse_prolongation=:limited_linear)

        if step > steps - avg_window_i
            composite_to_leaf_F_2d!(leaf, coarse, patch)
            composite_to_leaf_F_2d!(leaf_next, coarse_next, patch_next)
            drag = compute_drag_mea_solid_F_2d(leaf, leaf_next, is_solid)
            Fx_sum += T(drag.Fx)
            Fy_sum += T(drag.Fy)
            n_avg += 1
        end

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
        mass_initial, mass_final, mass_final - mass_initial, steps, avg_window_i)
end

"""
    ConservativeTreeBenchmarkRow2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeBenchmarkRow2D
```
"""
struct ConservativeTreeBenchmarkRow2D
    flow::Symbol
    method::Symbol
    Nx::Int
    Ny::Int
    steps::Int
    ux_mean::Float64
    uy_mean::Float64
    mass_rel_drift::Float64
    elapsed_s::Float64
end

function _conservative_tree_benchmark_row_2d(flow::Symbol,
                                             method::Symbol,
                                             Nx::Int,
                                             Ny::Int,
                                             result,
                                             elapsed_s::Real)
    ux = hasproperty(result, :ux_mean) ? getproperty(result, :ux_mean) :
         getproperty(result, :u_ref)
    uy = hasproperty(result, :uy_mean) ? getproperty(result, :uy_mean) : zero(ux)
    mass_initial = getproperty(result, :mass_initial)
    mass_drift = getproperty(result, :mass_drift)
    rel = abs(mass_drift) / max(abs(mass_initial), eps(typeof(float(mass_initial))))
    return ConservativeTreeBenchmarkRow2D(
        flow, method, Nx, Ny, getproperty(result, :steps),
        Float64(ux), Float64(uy), Float64(rel), Float64(elapsed_s))
end

"""
    benchmark_conservative_tree_cartesian_vs_amr_2d(;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.benchmark_conservative_tree_cartesian_vs_amr_2d)
```
"""
function benchmark_conservative_tree_cartesian_vs_amr_2d(;
        flows::Tuple=(:bfs, :square, :cylinder),
        steps::Int=240,
        T::Type{<:Real}=Float64)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    rows = ConservativeTreeBenchmarkRow2D[]
    for flow in flows
        if flow == :bfs
            leaf = nothing
            leaf_elapsed = @elapsed leaf = run_conservative_tree_bfs_macroflow_2d(
                ; steps=steps, T=T)
            route = nothing
            route_elapsed = @elapsed route = run_conservative_tree_bfs_route_native_2d(
                ; steps=steps, T=T)
            push!(rows, _conservative_tree_benchmark_row_2d(
                :bfs, :leaf_oracle, 28, 14, leaf, leaf_elapsed))
            push!(rows, _conservative_tree_benchmark_row_2d(
                :bfs, :amr_route_native, 28, 14, route, route_elapsed))
        elseif flow == :square
            leaf = nothing
            leaf_elapsed = @elapsed leaf = run_conservative_tree_square_obstacle_macroflow_2d(
                ; steps=steps, T=T)
            route = nothing
            route_elapsed = @elapsed route = run_conservative_tree_square_obstacle_route_native_2d(
                ; steps=steps, T=T)
            push!(rows, _conservative_tree_benchmark_row_2d(
                :square, :leaf_oracle, 24, 14, leaf, leaf_elapsed))
            push!(rows, _conservative_tree_benchmark_row_2d(
                :square, :amr_route_native, 24, 14, route, route_elapsed))
        elseif flow == :cylinder
            leaf = nothing
            leaf_elapsed = @elapsed leaf = run_conservative_tree_cylinder_macroflow_2d(
                ; steps=steps, avg_window=min(steps, 60), T=T)
            route = nothing
            route_elapsed = @elapsed route = run_conservative_tree_cylinder_obstacle_route_native_2d(
                ; steps=steps, avg_window=min(steps, 60), T=T)
            push!(rows, _conservative_tree_benchmark_row_2d(
                :cylinder, :leaf_oracle, 24, 14, leaf, leaf_elapsed))
            push!(rows, _conservative_tree_benchmark_row_2d(
                :cylinder, :amr_route_native, 24, 14, route, route_elapsed))
        else
            throw(ArgumentError("unsupported conservative-tree benchmark flow: $flow"))
        end
    end
    return rows
end

"""
    ConservativeTreeConvergenceRow2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeConvergenceRow2D
```
"""
struct ConservativeTreeConvergenceRow2D
    flow::Symbol
    method::Symbol
    scale::Int
    Nx::Int
    Ny::Int
    steps::Int
    ux_mean::Float64
    uy_mean::Float64
    Fx_drag::Float64
    Fy_drag::Float64
    Cd::Float64
    mass_rel_drift::Float64
    elapsed_s::Float64
end

function _scale_parent_range_2d(range::UnitRange{Int}, scale::Int)
    scale > 0 || throw(ArgumentError("scale must be positive"))
    return ((first(range) - 1) * scale + 1):(last(range) * scale)
end

function _scale_leaf_range_2d(range::UnitRange{Int}, scale::Int)
    scale > 0 || throw(ArgumentError("scale must be positive"))
    return ((first(range) - 1) * scale + 1):(last(range) * scale)
end

function _coarsen_leaf_range_2d(range::UnitRange{Int})
    isempty(range) && throw(ArgumentError("range must be nonempty"))
    return ((first(range) + 1) >>> 1):((last(range) + 1) >>> 1)
end

function _conservative_tree_obstacle_convergence_row_2d(
        flow::Symbol,
        method::Symbol,
        scale::Int,
        Nx::Int,
        Ny::Int,
        result,
        elapsed_s::Real)
    mass_initial = getproperty(result, :mass_initial)
    mass_drift = getproperty(result, :mass_drift)
    rel = abs(mass_drift) / max(abs(mass_initial), eps(typeof(float(mass_initial))))
    if flow == :cylinder
        return ConservativeTreeConvergenceRow2D(
            flow, method, scale, Nx, Ny, getproperty(result, :steps),
            Float64(getproperty(result, :u_ref)),
            0.0,
            Float64(getproperty(result, :Fx_drag)),
            Float64(getproperty(result, :Fy_drag)),
            Float64(getproperty(result, :Cd)),
            Float64(rel),
            Float64(elapsed_s))
    end

    return ConservativeTreeConvergenceRow2D(
        flow, method, scale, Nx, Ny, getproperty(result, :steps),
        Float64(getproperty(result, :ux_mean)),
        Float64(getproperty(result, :uy_mean)),
        NaN,
        NaN,
        NaN,
        Float64(rel),
        Float64(elapsed_s))
end

function _conservative_tree_obstacle_steps(scale::Int,
                                           base_steps::Int,
                                           step_exponent::Real)
    scale > 0 || throw(ArgumentError("scale must be positive"))
    base_steps > 0 || throw(ArgumentError("base_steps must be positive"))
    step_exponent >= 0 || throw(ArgumentError("step_exponent must be nonnegative"))
    return max(1, round(Int, base_steps * scale^step_exponent))
end

function _conservative_tree_obstacle_patch_ranges_2d(flow::Symbol,
                                                     scale::Int,
                                                     Nx::Int,
                                                     Ny::Int,
                                                     patch_strategy::Symbol)
    scale > 0 || throw(ArgumentError("scale must be positive"))
    if patch_strategy == :default
        if flow == :square || flow == :cylinder
            return (
                i_range=_scale_parent_range_2d(8:17, scale),
                j_range=_scale_parent_range_2d(4:11, scale),
            )
        end
    elseif patch_strategy == :interface_buffered
        if flow == :square || flow == :cylinder
            return (
                i_range=_scale_parent_range_2d(3:22, scale),
                j_range=1:Ny,
            )
        end
    else
        throw(ArgumentError("unsupported obstacle patch_strategy: $patch_strategy"))
    end
    throw(ArgumentError("unsupported obstacle convergence flow: $flow"))
end

"""
    convergence_conservative_tree_obstacles_2d(;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.convergence_conservative_tree_obstacles_2d)
```
"""
function convergence_conservative_tree_obstacles_2d(;
        flows::Tuple=(:square, :cylinder),
        scales::Tuple=(1, 2),
        base_steps::Int=1200,
        step_exponent::Real=1,
        avg_window::Int=300,
        patch_strategy::Symbol=:default,
        include_coarse_cartesian::Bool=false,
        T::Type{<:Real}=Float64)
    avg_window > 0 || throw(ArgumentError("avg_window must be positive"))
    rows = ConservativeTreeConvergenceRow2D[]

    for scale in scales
        scale > 0 || throw(ArgumentError("scales must contain positive integers"))
        steps = _conservative_tree_obstacle_steps(scale, base_steps, step_exponent)
        avg = min(avg_window * scale, steps)

        for flow in flows
            if flow == :square
                Nx = 24 * scale
                Ny = 14 * scale
                patch_ranges = _conservative_tree_obstacle_patch_ranges_2d(
                    :square, scale, Nx, Ny, patch_strategy)
                patch_i = patch_ranges.i_range
                patch_j = patch_ranges.j_range
                obstacle_i = _scale_leaf_range_2d(22:27, scale)
                obstacle_j = _scale_leaf_range_2d(12:17, scale)

                if include_coarse_cartesian
                    Nx_coarse = max(2, Nx >>> 1)
                    Ny_coarse = max(2, Ny >>> 1)
                    coarse_obstacle_i = _coarsen_leaf_range_2d(obstacle_i)
                    coarse_obstacle_j = _coarsen_leaf_range_2d(obstacle_j)
                    coarse_cart = nothing
                    coarse_elapsed = @elapsed coarse_cart =
                        run_conservative_tree_square_obstacle_macroflow_2d(
                            ; Nx=Nx_coarse, Ny=Ny_coarse,
                            patch_i_range=1:Nx_coarse,
                            patch_j_range=1:Ny_coarse,
                            obstacle_i_range=coarse_obstacle_i,
                            obstacle_j_range=coarse_obstacle_j,
                            steps=steps, T=T)
                    push!(rows, _conservative_tree_obstacle_convergence_row_2d(
                        :square, :cartesian_coarse, scale,
                        Nx_coarse, Ny_coarse, coarse_cart, coarse_elapsed))
                end

                leaf = nothing
                leaf_elapsed = @elapsed leaf = run_conservative_tree_square_obstacle_macroflow_2d(
                    ; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                    patch_j_range=patch_j, obstacle_i_range=obstacle_i,
                    obstacle_j_range=obstacle_j, steps=steps, T=T)
                route = nothing
                route_elapsed = @elapsed route = run_conservative_tree_square_obstacle_route_native_2d(
                    ; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                    patch_j_range=patch_j, obstacle_i_range=obstacle_i,
                    obstacle_j_range=obstacle_j, steps=steps, T=T)
                push!(rows, _conservative_tree_obstacle_convergence_row_2d(
                    :square, :leaf_oracle, scale, Nx, Ny, leaf, leaf_elapsed))
                push!(rows, _conservative_tree_obstacle_convergence_row_2d(
                    :square, :amr_route_native, scale, Nx, Ny, route, route_elapsed))
            elseif flow == :cylinder
                Nx = 24 * scale
                Ny = 14 * scale
                patch_ranges = _conservative_tree_obstacle_patch_ranges_2d(
                    :cylinder, scale, Nx, Ny, patch_strategy)
                patch_i = patch_ranges.i_range
                patch_j = patch_ranges.j_range
                cx_leaf = (2 * Nx + 1) / 2
                cy_leaf = (2 * Ny + 1) / 2
                radius_leaf = 3 * scale

                if include_coarse_cartesian
                    Nx_coarse = max(2, Nx >>> 1)
                    Ny_coarse = max(2, Ny >>> 1)
                    coarse_cart = nothing
                    coarse_elapsed = @elapsed coarse_cart =
                        run_conservative_tree_cylinder_macroflow_2d(
                            ; Nx=Nx_coarse, Ny=Ny_coarse,
                            patch_i_range=1:Nx_coarse,
                            patch_j_range=1:Ny_coarse,
                            cx_leaf=(2 * Nx_coarse + 1) / 2,
                            cy_leaf=(2 * Ny_coarse + 1) / 2,
                            radius_leaf=radius_leaf / 2,
                            steps=steps, avg_window=avg, T=T)
                    push!(rows, _conservative_tree_obstacle_convergence_row_2d(
                        :cylinder, :cartesian_coarse, scale,
                        Nx_coarse, Ny_coarse, coarse_cart, coarse_elapsed))
                end

                leaf = nothing
                leaf_elapsed = @elapsed leaf = run_conservative_tree_cylinder_macroflow_2d(
                    ; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                    patch_j_range=patch_j, cx_leaf=cx_leaf, cy_leaf=cy_leaf,
                    radius_leaf=radius_leaf, steps=steps, avg_window=avg, T=T)
                route = nothing
                route_elapsed = @elapsed route = run_conservative_tree_cylinder_obstacle_route_native_2d(
                    ; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                    patch_j_range=patch_j, cx_leaf=cx_leaf, cy_leaf=cy_leaf,
                    radius_leaf=radius_leaf, steps=steps, avg_window=avg, T=T)
                push!(rows, _conservative_tree_obstacle_convergence_row_2d(
                    :cylinder, :leaf_oracle, scale, Nx, Ny, leaf, leaf_elapsed))
                push!(rows, _conservative_tree_obstacle_convergence_row_2d(
                    :cylinder, :amr_route_native, scale, Nx, Ny, route, route_elapsed))
            else
                throw(ArgumentError("unsupported obstacle convergence flow: $flow"))
            end
        end
    end

    return rows
end

