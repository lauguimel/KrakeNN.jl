# AMR-D .krk validation and runtime dispatch helpers.

"""
    ConservativeTreeAMRDKrkCase2D

Public type or module in the grid-refinement and conservative-tree AMR API.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.ConservativeTreeAMRDKrkCase2D
```
"""
struct ConservativeTreeAMRDKrkCase2D
    name::String
    flow::Symbol
    geometry::Symbol
    boundary_policy::Symbol
    wall_model::Symbol
    max_level::Int
    refine_count::Int
    diagnostics::Vector{Symbol}
    spec_supported::Bool
    runtime_supported::Bool
    runtime_status::Symbol
    reason::String
end

function _amr_d_setup_from_source_2d(source)
    return source isa AbstractString ? load_kraken(source) : source
end

function _amr_d_boundary_type_map_2d(setup)
    faces = Dict{Symbol,Symbol}()
    for bc in getproperty(setup, :boundaries)
        faces[getproperty(bc, :face)] = getproperty(bc, :type)
    end
    return faces
end

@inline function _amr_d_face_is_2d(faces::Dict{Symbol,Symbol},
                                   face::Symbol,
                                   bc_type::Symbol)
    return get(faces, face, :missing) == bc_type
end

"""
    conservative_tree_amr_d_boundary_policy_2d(source)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_amr_d_boundary_policy_2d)
```
"""
function conservative_tree_amr_d_boundary_policy_2d(source)
    setup = _amr_d_setup_from_source_2d(source)
    faces = _amr_d_boundary_type_map_2d(setup)
    periodic_x = _amr_d_face_is_2d(faces, :west, :periodic) &&
                 _amr_d_face_is_2d(faces, :east, :periodic)
    wall_y = _amr_d_face_is_2d(faces, :south, :wall) &&
             _amr_d_face_is_2d(faces, :north, :wall)
    moving_wall_y = _amr_d_face_is_2d(faces, :south, :wall) &&
                    _amr_d_face_is_2d(faces, :north, :velocity)
    open_x_wall_y = _amr_d_face_is_2d(faces, :west, :velocity) &&
                    _amr_d_face_is_2d(faces, :east, :pressure) &&
                    wall_y
    closed_box = _amr_d_face_is_2d(faces, :west, :wall) &&
                 _amr_d_face_is_2d(faces, :east, :wall) &&
                 wall_y

    periodic_x && wall_y && return :periodic_x_wall_y
    periodic_x && moving_wall_y && return :periodic_x_moving_wall_y
    open_x_wall_y && return :open_x_wall_y
    closed_box && return :bounceback
    return :unsupported
end

"""
    conservative_tree_amr_d_geometry_2d(source)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_amr_d_geometry_2d)
```
"""
function conservative_tree_amr_d_geometry_2d(source)
    setup = _amr_d_setup_from_source_2d(source)
    obstacles = filter(r -> getproperty(r, :kind) == :obstacle,
                       getproperty(setup, :regions))
    isempty(obstacles) && return :channel
    names = lowercase(join((getproperty(r, :name) for r in obstacles), " "))
    occursin("cylinder", names) && return :cylinder
    occursin("square", names) && return :square
    occursin("step", names) && return :step
    return :obstacle
end

function _amr_d_flow_kind_2d(setup, geometry::Symbol, boundary_policy::Symbol)
    diagnostics = getproperty(setup, :diagnostics)
    columns = diagnostics === nothing ? Symbol[] : getproperty(diagnostics, :columns)

    geometry == :step && return :bfs
    if geometry == :cylinder
        boundary_policy == :open_x_wall_y && :lift in columns &&
            return :cylinder_lift
        return :cylinder
    end
    geometry == :square && return :square
    geometry != :channel && return :solid_obstacle
    boundary_policy == :periodic_x_moving_wall_y && return :couette
    boundary_policy == :periodic_x_wall_y && return :poiseuille
    boundary_policy == :open_x_wall_y && return :open_channel
    return :unknown
end

function _amr_d_wall_model_2d(geometry::Symbol)
    geometry == :channel && return :none
    return :halfway_bounceback_mask
end

function _amr_d_spec_max_level_2d(setup)
    spec = create_conservative_tree_spec_from_krk_2d(setup)
    return spec.max_level
end

function _amr_d_one_level_route_runtime_status_2d(flow::Symbol,
                                                  geometry::Symbol,
                                                  boundary_policy::Symbol)
    if geometry == :channel &&
            boundary_policy in (:periodic_x_wall_y,
                                :periodic_x_moving_wall_y,
                                :open_x_wall_y)
        return true, :route_native_one_level_channel, "one-level AMR-D channel route is available"
    elseif flow in (:square, :cylinder) &&
            boundary_policy == :periodic_x_wall_y
        return true, :route_native_one_level_solid, "one-level AMR-D solid mask route is available"
    elseif flow == :bfs && boundary_policy == :open_x_wall_y
        return true, :route_native_one_level_open_solid, "one-level AMR-D open solid route is available"
    end
    return false, :unsupported_boundary, "AMR-D has no route-native runtime for this boundary set"
end

function _amr_d_runtime_status_2d(flow::Symbol,
                                  geometry::Symbol,
                                  boundary_policy::Symbol,
                                  max_level::Int)
    max_level <= 1 &&
        return _amr_d_one_level_route_runtime_status_2d(
            flow, geometry, boundary_policy)

    if geometry == :channel &&
            boundary_policy in (:periodic_x_wall_y,
                                :periodic_x_moving_wall_y) &&
            1 <= max_level <= 4
        return true, :subcycled_nested_channel,
               "nested AMR-D channel scheduler is available for levels 1:4"
    elseif flow in (:square, :cylinder) &&
            boundary_policy == :periodic_x_wall_y &&
            1 <= max_level <= 4
        return true, :subcycled_nested_solid,
               "nested AMR-D solid-mask scheduler is available when the solid is fully refined"
    elseif geometry != :channel
        return false, :nested_obstacle_runtime_pending,
               "nested obstacle AMR-D runtime is not closed yet"
    elseif boundary_policy == :open_x_wall_y
        return false, :nested_open_channel_runtime_pending,
               "nested open-channel AMR-D runtime is not closed yet"
    end
    return false, :unsupported_nested_case,
           "nested AMR-D runtime does not support this case"
end

"""
    conservative_tree_amr_d_case_from_krk_2d(source)

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_amr_d_case_from_krk_2d)
```
"""
function conservative_tree_amr_d_case_from_krk_2d(source)
    setup = _amr_d_setup_from_source_2d(source)
    boundary_policy = conservative_tree_amr_d_boundary_policy_2d(setup)
    geometry = conservative_tree_amr_d_geometry_2d(setup)
    flow = _amr_d_flow_kind_2d(setup, geometry, boundary_policy)
    wall_model = _amr_d_wall_model_2d(geometry)
    diagnostics = getproperty(setup, :diagnostics)
    columns = diagnostics === nothing ? Symbol[] : copy(getproperty(diagnostics, :columns))
    refine_count = length(getproperty(setup, :refinements))

    spec_supported = true
    max_level = 0
    spec_reason = "static conservative-tree spec builds"
    try
        max_level = _amr_d_spec_max_level_2d(setup)
    catch err
        spec_supported = false
        spec_reason = sprint(showerror, err)
    end

    runtime_supported = false
    runtime_status = :invalid_static_spec
    runtime_reason = spec_reason
    if spec_supported
        runtime_supported, runtime_status, runtime_reason =
            _amr_d_runtime_status_2d(flow, geometry, boundary_policy, max_level)
    end

    return ConservativeTreeAMRDKrkCase2D(
        String(getproperty(setup, :name)), flow, geometry, boundary_policy,
        wall_model, max_level, refine_count, columns, spec_supported,
        runtime_supported, runtime_status, runtime_reason)
end

"""
    conservative_tree_amr_d_support_matrix_2d()

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.conservative_tree_amr_d_support_matrix_2d)
```
"""
function conservative_tree_amr_d_support_matrix_2d()
    return [
        (feature=:periodic_x_wall_y, single_patch=true, nested=true,
         wall_model=:none, note="channel Poiseuille"),
        (feature=:periodic_x_moving_wall_y, single_patch=true, nested=true,
         wall_model=:none, note="channel Couette"),
        (feature=:open_x_wall_y, single_patch=true, nested=false,
         wall_model=:none, note="BFS/open-channel nesting still pending"),
        (feature=:halfway_bounceback_solid_mask, single_patch=true, nested=true,
         wall_model=:halfway_bounceback_mask,
         note="periodic square/cylinder nesting requires solid fully resolved away from interfaces"),
        (feature=:ibb, single_patch=false, nested=false, wall_model=:unsupported,
         note="IBB is available in other KrakeNN paths, not AMR-D route-native D"),
        (feature=:libb, single_patch=false, nested=false, wall_model=:unsupported,
         note="LIBB is available in body-fit/SLBM/cartesian paths, not AMR-D route-native D"),
    ]
end

function _amr_d_var_2d(setup, name::Symbol, default)
    vars = getproperty(setup, :user_vars)
    return haskey(vars, name) ? vars[name] : default
end

function _amr_d_constant_expr_2d(expr, default)
    try
        return Float64(evaluate(expr))
    catch
        return default
    end
end

function _amr_d_body_force_2d(setup, name::Symbol, default)
    body = getproperty(getproperty(setup, :physics), :body_force)
    haskey(body, name) || return _amr_d_var_2d(setup, name, default)
    return _amr_d_constant_expr_2d(body[name], _amr_d_var_2d(setup, name, default))
end

function _amr_d_boundary_value_2d(setup,
                                  face::Symbol,
                                  bc_type::Symbol,
                                  name::Symbol,
                                  default)
    for bc in getproperty(setup, :boundaries)
        getproperty(bc, :face) == face || continue
        getproperty(bc, :type) == bc_type || continue
        values = getproperty(bc, :values)
        haskey(values, name) || continue
        return _amr_d_constant_expr_2d(values[name], default)
    end
    return default
end

function _amr_d_single_patch_ranges_2d(setup)
    ranges = conservative_tree_patch_ranges_from_krk_refines_2d(
        getproperty(setup, :domain), getproperty(setup, :refinements))
    length(ranges) == 1 ||
        throw(ArgumentError("AMR-D route-native one-level .krk cases need exactly one Refine block"))
    return only(ranges)
end

"""
    run_conservative_tree_amr_d_case_from_krk_2d(source;

Public function in the grid-refinement and conservative-tree AMR API.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.run_conservative_tree_amr_d_case_from_krk_2d)
```
"""
function run_conservative_tree_amr_d_case_from_krk_2d(source;
        steps_override=nothing,
        mass_guard_rtol=nothing,
        T::Type{<:AbstractFloat}=Float64)
    setup = _amr_d_setup_from_source_2d(source)
    case = conservative_tree_amr_d_case_from_krk_2d(setup)
    case.spec_supported ||
        throw(ArgumentError("AMR-D .krk static spec is invalid: $(case.reason)"))
    case.runtime_supported ||
        throw(ArgumentError("AMR-D .krk runtime is not supported: $(case.reason)"))

    steps = steps_override === nothing ? getproperty(setup, :max_steps) :
        Int(steps_override)
    spec = create_conservative_tree_spec_from_krk_2d(setup)
    omega = _amr_d_var_2d(setup, :omega, 1.2)
    rho0 = _amr_d_var_2d(setup, :rho0, 1.0)
    krk_mass_guard_rtol = _amr_d_var_2d(setup, :mass_guard_rtol, nothing)
    resolved_mass_guard_rtol = mass_guard_rtol === nothing ?
        krk_mass_guard_rtol : mass_guard_rtol

    if case.runtime_status == :route_native_one_level_channel
        domain = getproperty(setup, :domain)
        Nx = Int(getproperty(domain, :Nx))
        Ny = Int(getproperty(domain, :Ny))
        patch_i, patch_j = _amr_d_single_patch_ranges_2d(setup)
        if case.flow == :poiseuille
            Fx = _amr_d_body_force_2d(setup, :Fx, 1e-6)
            return run_conservative_tree_poiseuille_route_native_2d(
                Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                patch_j_range=patch_j, Fx=Fx, omega=omega, rho=rho0,
                steps=steps, T=T)
        elseif case.flow == :couette
            U = _amr_d_boundary_value_2d(
                setup, :north, :velocity, :ux, _amr_d_var_2d(setup, :U, 1e-3))
            return run_conservative_tree_couette_route_native_2d(
                Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                patch_j_range=patch_j, U=U, omega=omega, rho=rho0,
                steps=steps, T=T)
        end
    elseif case.runtime_status == :route_native_one_level_open_solid
        domain = getproperty(setup, :domain)
        Nx = Int(getproperty(domain, :Nx))
        Ny = Int(getproperty(domain, :Ny))
        patch_i, patch_j = _amr_d_single_patch_ranges_2d(setup)
        u_in = _amr_d_boundary_value_2d(
            setup, :west, :velocity, :ux, _amr_d_var_2d(setup, :u_in, 0.03))
        rho_out = _amr_d_boundary_value_2d(
            setup, :east, :pressure, :rho, 1.0)
        return run_conservative_tree_bfs_route_native_2d(
            Nx=Nx, Ny=Ny, patch_i_range=patch_i, patch_j_range=patch_j,
            step_i_leaf=round(Int, _amr_d_var_2d(setup, :step_i_leaf, 16)),
            step_height_leaf=round(Int, _amr_d_var_2d(
                setup, :step_height_leaf, 8)),
            u_in=u_in, rho_out=rho_out, omega=omega, rho=rho0,
            steps=steps, T=T)
    elseif case.runtime_status == :route_native_one_level_solid
        domain = getproperty(setup, :domain)
        Nx = Int(getproperty(domain, :Nx))
        Ny = Int(getproperty(domain, :Ny))
        patch_i, patch_j = _amr_d_single_patch_ranges_2d(setup)
        Fx = _amr_d_body_force_2d(setup, :Fx, 2e-5)
        if case.flow == :square
            Fy = _amr_d_body_force_2d(setup, :Fy, 0.0)
            return run_conservative_tree_square_obstacle_route_native_2d(
                Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                patch_j_range=patch_j,
                obstacle_i_range=round(Int, _amr_d_var_2d(
                    setup, :obstacle_i0, 22)):round(Int, _amr_d_var_2d(
                    setup, :obstacle_i1, 27)),
                obstacle_j_range=round(Int, _amr_d_var_2d(
                    setup, :obstacle_j0, 12)):round(Int, _amr_d_var_2d(
                    setup, :obstacle_j1, 17)),
                Fx=Fx, Fy=Fy, omega=omega, rho=rho0, steps=steps, T=T)
        elseif case.flow == :cylinder
            return run_conservative_tree_cylinder_obstacle_route_native_2d(
                Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                patch_j_range=patch_j,
                cx_leaf=_amr_d_var_2d(setup, :cx_leaf, (2 * Nx + 1) / 2),
                cy_leaf=_amr_d_var_2d(setup, :cy_leaf, (2 * Ny + 1) / 2),
                radius_leaf=_amr_d_var_2d(setup, :radius_leaf, 3.0),
                Fx=Fx, omega=omega, rho=rho0, steps=steps,
                avg_window=round(Int, _amr_d_var_2d(
                    setup, :avg_window, max(1, div(steps, 4)))),
                T=T)
        end
    elseif case.runtime_status == :subcycled_nested_solid
        leaf_nx = Int(getproperty(getproperty(setup, :domain), :Nx)) <<
            case.max_level
        leaf_ny = Int(getproperty(getproperty(setup, :domain), :Ny)) <<
            case.max_level
        Fx = _amr_d_body_force_2d(setup, :Fx, 2e-5)
        if case.flow == :square
            Fy = _amr_d_body_force_2d(setup, :Fy, 0.0)
            is_solid = square_solid_mask_leaf_2d(
                leaf_nx, leaf_ny,
                round(Int, _amr_d_var_2d(setup, :obstacle_i0, 22)):
                round(Int, _amr_d_var_2d(setup, :obstacle_i1, 27)),
                round(Int, _amr_d_var_2d(setup, :obstacle_j0, 12)):
                round(Int, _amr_d_var_2d(setup, :obstacle_j1, 17)))
            return run_conservative_tree_solid_obstacle_subcycled_2d(
                flow=:square_obstacle_subcycled, max_level=case.max_level,
                spec=spec, is_solid_leaf=is_solid, steps=steps,
                omega=omega, Fx=Fx, Fy=Fy, rho0=rho0,
                mass_guard_rtol=resolved_mass_guard_rtol, T=T)
        elseif case.flow == :cylinder
            is_solid = cylinder_solid_mask_leaf_2d(
                leaf_nx, leaf_ny,
                _amr_d_var_2d(setup, :cx_leaf, (leaf_nx + 1) / 2),
                _amr_d_var_2d(setup, :cy_leaf, (leaf_ny + 1) / 2),
                _amr_d_var_2d(setup, :radius_leaf, 3.0))
            return run_conservative_tree_solid_obstacle_subcycled_2d(
                flow=:cylinder_obstacle_subcycled,
                max_level=case.max_level, spec=spec,
                is_solid_leaf=is_solid, steps=steps, omega=omega,
                Fx=Fx, Fy=0.0, rho0=rho0,
                mass_guard_rtol=resolved_mass_guard_rtol, T=T)
        end
    elseif case.flow == :poiseuille
        Fx = _amr_d_body_force_2d(setup, :Fx, 1e-6)
        Fy = _amr_d_body_force_2d(setup, :Fy, 0.0)
        return run_conservative_tree_poiseuille_subcycled_2d(
            max_level=case.max_level, spec=spec, steps=steps, omega=omega,
            Fx=Fx, Fy=Fy, rho0=rho0,
            mass_guard_rtol=resolved_mass_guard_rtol, T=T)
    elseif case.flow == :couette
        U = _amr_d_boundary_value_2d(
            setup, :north, :velocity, :ux, _amr_d_var_2d(setup, :U, 1e-3))
        return run_conservative_tree_couette_subcycled_2d(
            max_level=case.max_level, spec=spec, steps=steps, omega=omega,
            U=U, rho0=rho0, mass_guard_rtol=resolved_mass_guard_rtol,
            T=T)
    end

    throw(ArgumentError("AMR-D .krk runtime helper currently dispatches " *
                        "nested channel Poiseuille/Couette only"))
end
