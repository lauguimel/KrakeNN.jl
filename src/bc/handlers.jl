"""
    BoundaryHandler

Pre-compiled boundary condition data for one face.
"""
struct BoundaryHandler
    face::Symbol
    type::Symbol
    # Pre-compiled expression functions (or nothing)
    ux_fn::Union{Function, Nothing}
    uy_fn::Union{Function, Nothing}
    rho_fn::Union{Function, Nothing}
    # Flags
    is_spatial_ux::Bool
    is_spatial_uy::Bool
    is_time_dep_ux::Bool
    is_time_dep_uy::Bool
    is_time_dep_rho::Bool
    # Pre-allocated arrays for spatial BCs (on backend)
    ux_arr::Union{AbstractArray, Nothing}
    uy_arr::Union{AbstractArray, Nothing}
    rho_arr::Union{AbstractArray, Nothing}
end

"""Build pre-compiled boundary handlers."""
function _build_boundary_handlers(setup::SimulationSetup, dx, dy, Nx, Ny, ::Type{T},
                                  backend) where T
    handlers = BoundaryHandler[]

    for bc in setup.boundaries
        bc.type == :periodic && continue

        # Determine face size for array allocation
        face_size = (bc.face in (:north, :south)) ? Nx : Ny

        ux_fn = haskey(bc.values, :ux) ? bc.values[:ux].func : nothing
        uy_fn = haskey(bc.values, :uy) ? bc.values[:uy].func : nothing
        rho_fn = haskey(bc.values, :rho) ? bc.values[:rho].func : nothing

        is_sp_ux = haskey(bc.values, :ux) && is_spatial(bc.values[:ux])
        is_sp_uy = haskey(bc.values, :uy) && is_spatial(bc.values[:uy])
        is_td_ux = haskey(bc.values, :ux) && is_time_dependent(bc.values[:ux])
        is_td_uy = haskey(bc.values, :uy) && is_time_dependent(bc.values[:uy])
        is_td_rho = haskey(bc.values, :rho) && is_time_dependent(bc.values[:rho])

        needs_spatial = is_sp_ux || is_sp_uy || is_td_ux || is_td_uy

        ux_arr = needs_spatial ? KernelAbstractions.zeros(backend, T, face_size) : nothing
        uy_arr = needs_spatial ? KernelAbstractions.zeros(backend, T, face_size) : nothing
        rho_arr = (haskey(bc.values, :rho) && (is_spatial(bc.values[:rho]) || is_td_rho)) ?
            KernelAbstractions.zeros(backend, T, face_size) : nothing

        # Pre-compute static spatial arrays
        if needs_spatial && !is_td_ux && !is_td_uy
            _fill_bc_arrays!(ux_arr, uy_arr, ux_fn, uy_fn, bc.face,
                             dx, dy, Nx, Ny, setup.domain, T)
        end

        push!(handlers, BoundaryHandler(
            bc.face, bc.type,
            ux_fn, uy_fn, rho_fn,
            is_sp_ux, is_sp_uy,
            is_td_ux, is_td_uy, is_td_rho,
            ux_arr, uy_arr, rho_arr
        ))
    end

    return handlers
end

function _libb_velocity_profile_host(bc::BoundarySetup, setup::SimulationSetup,
                                     dx, dy, Nx, Ny, ::Type{T}) where T
    if any(is_time_dependent, values(bc.values))
        throw(ArgumentError("time-dependent velocity BCs are not supported with STL wall=libb"))
    end

    dom = setup.domain
    if bc.face in (:west, :east)
        profile = zeros(T, Ny)
        x_val = bc.face == :west ? dx / 2 : dom.Lx - dx / 2
        for j in 1:Ny
            y_val = (j - 0.5) * dy
            kw = (; x=x_val, y=y_val, Lx=dom.Lx, Ly=dom.Ly,
                    Nx=Float64(Nx), Ny=Float64(Ny), dx=dx, dy=dy, t=0.0)
            profile[j] = haskey(bc.values, :ux) ?
                T(evaluate(bc.values[:ux]; kw...)) : zero(T)
        end
    elseif bc.face in (:south, :north)
        profile = zeros(T, Nx)
        y_val = bc.face == :south ? dy / 2 : dom.Ly - dy / 2
        for i in 1:Nx
            x_val = (i - 0.5) * dx
            kw = (; x=x_val, y=y_val, Lx=dom.Lx, Ly=dom.Ly,
                    Nx=Float64(Nx), Ny=Float64(Ny), dx=dx, dy=dy, t=0.0)
            profile[i] = haskey(bc.values, :uy) ?
                T(evaluate(bc.values[:uy]; kw...)) : zero(T)
        end
    else
        throw(ArgumentError("Boundary face ':$(bc.face)' is not valid for 2D STL wall=libb"))
    end
    return profile
end

function _libb_pressure_value(bc::BoundarySetup, ::Type{T}) where T
    if haskey(bc.values, :rho)
        expr = bc.values[:rho]
        (is_spatial(expr) || is_time_dependent(expr)) && throw(ArgumentError(
            "spatial or time-dependent pressure BCs are not supported with STL wall=libb"))
        return T(evaluate(expr; t=0.0))
    end
    return one(T)
end

function _build_libb_bc_rebuild_spec_2d(setup::SimulationSetup, dx, dy,
                                        Nx, Ny, ::Type{T}, backend) where T
    west = HalfwayBB()
    east = HalfwayBB()
    south = HalfwayBB()
    north = HalfwayBB()

    for bc in setup.boundaries
        face_bc = if bc.type == :wall
            HalfwayBB()
        elseif bc.type == :velocity
            profile = _libb_velocity_profile_host(bc, setup, dx, dy, Nx, Ny, T)
            ZouHeVelocity(_copy_to_backend(backend, T, profile))
        elseif bc.type == :pressure
            # East outlet: extrapolation rebuild (the Zou-He east closure keeps a
            # non-decaying outlet mode on this path); other faces keep Zou-He.
            bc.face == :east ? NeqExtrapolationPressure(_libb_pressure_value(bc, T)) :
                               ZouHePressure(_libb_pressure_value(bc, T))
        elseif bc.type == :periodic
            throw(ArgumentError("periodic domain boundaries are not supported with STL wall=libb"))
        else
            throw(ArgumentError("Boundary type ':$(bc.type)' is not supported with STL wall=libb"))
        end

        if bc.face == :west
            west = face_bc
        elseif bc.face == :east
            east = face_bc
        elseif bc.face == :south
            south = face_bc
        elseif bc.face == :north
            north = face_bc
        else
            throw(ArgumentError("Boundary face ':$(bc.face)' is not valid for 2D STL wall=libb"))
        end
    end

    return BCSpec2D(; west=west, east=east, south=south, north=north)
end

"""Fill BC arrays with spatial profile for a given face."""
function _fill_bc_arrays!(ux_arr, uy_arr, ux_fn, uy_fn, face::Symbol,
                          dx, dy, Nx, Ny, domain::DomainSetup, ::Type{T};
                          t::Float64=0.0) where T
    Lx, Ly = domain.Lx, domain.Ly

    if face in (:north, :south)
        cpu_ux = zeros(T, Nx)
        cpu_uy = zeros(T, Nx)
        y_val = face == :south ? dy / 2 : Ly - dy / 2
        for i in 1:Nx
            x_val = (i - 0.5) * dx
            kw = (; x=x_val, y=y_val, Lx=Lx, Ly=Ly,
                    Nx=Float64(Nx), Ny=Float64(Ny), dx=dx, dy=dy, t=t)
            ux_fn !== nothing && (cpu_ux[i] = T(Base.invokelatest(ux_fn; kw...)))
            uy_fn !== nothing && (cpu_uy[i] = T(Base.invokelatest(uy_fn; kw...)))
        end
        ux_arr !== nothing && copyto!(ux_arr, cpu_ux)
        uy_arr !== nothing && copyto!(uy_arr, cpu_uy)
    else  # :west or :east
        cpu_ux = zeros(T, Ny)
        cpu_uy = zeros(T, Ny)
        x_val = face == :west ? dx / 2 : Lx - dx / 2
        for j in 1:Ny
            y_val = (j - 0.5) * dy
            kw = (; x=x_val, y=y_val, Lx=Lx, Ly=Ly,
                    Nx=Float64(Nx), Ny=Float64(Ny), dx=dx, dy=dy, t=t)
            ux_fn !== nothing && (cpu_ux[j] = T(Base.invokelatest(ux_fn; kw...)))
            uy_fn !== nothing && (cpu_uy[j] = T(Base.invokelatest(uy_fn; kw...)))
        end
        ux_arr !== nothing && copyto!(ux_arr, cpu_ux)
        uy_arr !== nothing && copyto!(uy_arr, cpu_uy)
    end
end

"""Apply boundary conditions at a given timestep."""
function _apply_boundary_conditions!(f, handlers::Vector{BoundaryHandler},
                                     step::Int, Nx, Ny, dx, dy,
                                     domain::DomainSetup, ::Type{T}) where T
    for h in handlers
        h.type == :periodic && continue

        if h.type == :wall && h.ux_fn === nothing && h.uy_fn === nothing
            # Pure wall — handled by streaming bounce-back, but apply explicit
            # bounce-back for faces not covered by the streaming kernel
            _apply_wall_bc!(f, h.face, Nx, Ny)
        elseif h.type == :velocity
            # Re-evaluate time-dependent BCs
            if h.is_time_dep_ux || h.is_time_dep_uy
                _fill_bc_arrays!(h.ux_arr, h.uy_arr, h.ux_fn, h.uy_fn, h.face,
                                 dx, dy, Nx, Ny, domain, T;
                                 t=Float64(step))
            end

            if h.ux_arr !== nothing  # spatial BC
                _apply_velocity_spatial!(f, h, Nx, Ny)
            else  # scalar BC
                ux_val = h.ux_fn !== nothing ? Base.invokelatest(h.ux_fn; t=Float64(step)) : 0.0
                uy_val = h.uy_fn !== nothing ? Base.invokelatest(h.uy_fn; t=Float64(step)) : 0.0
                _apply_velocity_scalar!(f, h.face, ux_val, uy_val, Nx, Ny)
            end
        elseif h.type == :pressure
            rho_val = h.rho_fn !== nothing ? Base.invokelatest(h.rho_fn; t=Float64(step)) : 1.0
            _apply_pressure_bc!(f, h.face, rho_val, Nx, Ny)
        end
    end
end

"""Apply wall BC on a specific face."""
function _apply_wall_bc!(f, face::Symbol, Nx, Ny)
    # stream_2d! already handles bounce-back at domain edges.
    # For periodic streaming kernels, explicit bounce-back is needed.
    # The bounce-back is embedded in the streaming step for wall boundaries.
    # Nothing extra needed here — the streaming kernel handles it.
end

"""Apply scalar velocity BC."""
function _apply_velocity_scalar!(f, face::Symbol, ux_val, uy_val, Nx, Ny)
    if face == :north
        apply_zou_he_north_2d!(f, ux_val, Nx, Ny)
    elseif face == :south
        apply_zou_he_south_2d!(f, ux_val, Nx)
    elseif face == :west
        apply_zou_he_west_2d!(f, ux_val, Nx, Ny)
    end
end

"""Apply spatial velocity BC."""
function _apply_velocity_spatial!(f, h::BoundaryHandler, Nx, Ny)
    if h.face == :north
        apply_zou_he_north_spatial_2d!(f, h.ux_arr, h.uy_arr, Nx, Ny)
    elseif h.face == :south
        apply_zou_he_south_spatial_2d!(f, h.ux_arr, h.uy_arr, Nx)
    elseif h.face == :west
        apply_zou_he_west_spatial_2d!(f, h.ux_arr, h.uy_arr, Nx, Ny)
    end
end

"""Apply pressure BC.

Supported faces on the generic 2D route: `:west` (inlet) and `:east` (outlet),
both via Zou-He with a prescribed density and zero transverse velocity. Any
other face raises: a pressure condition that is silently dropped produces a
plausible-looking field for a problem the user did not ask for.
"""
function _apply_pressure_bc!(f, face::Symbol, rho_val, Nx, Ny)
    if face == :east
        apply_zou_he_pressure_east_2d!(f, Nx, Ny; ρ_out=rho_val)
    elseif face == :west
        apply_zou_he_pressure_west_2d!(f, Nx, Ny; ρ_in=rho_val)
    else
        error("Pressure boundary condition on face ':$(face)' is not supported " *
              "by the generic 2D runner. Supported pressure faces: :west, :east. " *
              "For a pressure condition on :north or :south, either rotate the " *
              "domain or use a driver that implements it.")
    end
end
