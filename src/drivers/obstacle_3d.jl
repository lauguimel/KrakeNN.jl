# Plain backend allocation: the D3Q19 pull brick (PullHalfwayBB_3D) clamps its
# eager-ifelse neighbour reads, so f_in/f_out need no halo padding on CPU.

function _libb_pressure_value_3d(bc, ::Type{T}) where T
    if haskey(bc.values, :rho)
        expr = bc.values[:rho]
        (is_spatial(expr) || is_time_dependent(expr)) && throw(ArgumentError(
            "spatial or time-dependent pressure BCs are not supported with 3D STL wall=libb"))
        return T(evaluate(expr; t=0.0))
    end
    return one(T)
end

function _libb_velocity_profile_host_3d(bc, setup, dx, Nx, Ny, Nz,
                                        ::Type{T}) where T
    if any(is_time_dependent, values(bc.values))
        throw(ArgumentError("time-dependent velocity BCs are not supported with 3D STL wall=libb"))
    end
    bc.face == :west || throw(ArgumentError(
        "3D STL wall=libb currently supports velocity BCs only on the west face"))

    dom = setup.domain
    dy = dom.Ly / Ny
    dz = dom.Lz / Nz
    profile = zeros(T, Ny, Nz)
    x_val = dx / 2
    for k in 1:Nz, j in 1:Ny
        y_val = (j - 0.5) * dy
        z_val = (k - 0.5) * dz
        kw = (; x=x_val, y=y_val, z=z_val,
              Lx=dom.Lx, Ly=dom.Ly, Lz=dom.Lz,
              Nx=Float64(Nx), Ny=Float64(Ny), Nz=Float64(Nz),
              dx=dx, dy=dy, dz=dz, t=0.0)
        profile[j, k] = haskey(bc.values, :ux) ?
            T(evaluate(bc.values[:ux]; kw...)) : zero(T)
    end
    return profile
end

function _west_libb_velocity_profile_host_3d(setup, dx, Nx, Ny, Nz,
                                             ::Type{T}) where T
    for bc in setup.boundaries
        if bc.face === :west && bc.type === :velocity
            return _libb_velocity_profile_host_3d(bc, setup, dx, Nx, Ny, Nz, T)
        end
    end
    throw(ArgumentError("3D STL wall=libb requires a west velocity inlet"))
end

function _build_libb_bc_rebuild_spec_3d(setup, dx, Nx, Ny, Nz,
                                        ::Type{T}, backend) where T
    west = HalfwayBB()
    east = HalfwayBB()
    south = HalfwayBB()
    north = HalfwayBB()
    bottom = HalfwayBB()
    top = HalfwayBB()

    for bc in setup.boundaries
        face_bc = if bc.type == :wall
            HalfwayBB()
        elseif bc.type == :velocity
            bc.face === :west || throw(ArgumentError(
                "3D STL wall=libb currently supports velocity BCs only on the west face"))
            profile = _libb_velocity_profile_host_3d(bc, setup, dx, Nx, Ny, Nz, T)
            dev = KernelAbstractions.allocate(backend, T, Ny, Nz)
            copyto!(dev, profile)
            ZouHeVelocity(dev)
        elseif bc.type == :pressure
            bc.face === :east || throw(ArgumentError(
                "3D STL wall=libb currently supports pressure BCs only on the east face"))
            ZouHePressure(_libb_pressure_value_3d(bc, T))
        elseif bc.type == :periodic
            throw(ArgumentError("periodic domain boundaries are not supported with 3D STL wall=libb"))
        else
            throw(ArgumentError("Boundary type ':$(bc.type)' is not supported with 3D STL wall=libb"))
        end

        if bc.face === :west
            west = face_bc
        elseif bc.face === :east
            east = face_bc
        elseif bc.face === :south
            south = face_bc
        elseif bc.face === :north
            north = face_bc
        elseif bc.face === :bottom
            bottom = face_bc
        elseif bc.face === :top
            top = face_bc
        else
            throw(ArgumentError("Boundary face ':$(bc.face)' is not valid for 3D STL wall=libb"))
        end
    end

    return BCSpec3D(; west=west, east=east, south=south, north=north,
                    bottom=bottom, top=top)
end

function run_obstacle_libb_3d(setup;
                              backend=KernelAbstractions.CPU(),
                              T::Type{<:AbstractFloat}=Float64)
    dom = setup.domain
    Nx, Ny, Nz = dom.Nx, dom.Ny, dom.Nz
    dx = dom.Lx / Nx
    ν = T(setup.physics.params[:nu])

    is_solid_h = zeros(Bool, Nx, Ny, Nz)
    _apply_geometry_3d!(is_solid_h, setup, Float64(dx))
    q_wall_h = _precompute_stl_libb_q_wall_3d(is_solid_h, setup, dx, T)
    u_profile_h = _west_libb_velocity_profile_host_3d(setup, dx, Nx, Ny, Nz, T)

    f_in_h = zeros(T, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 1:19
        f_in_h[i, j, k, q] = equilibrium(D3Q19(), one(T),
                                          u_profile_h[j, k],
                                          zero(T), zero(T), q)
    end

    q_wall = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny, Nz)
    uw_x = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    uw_y = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    uw_z = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    f_in = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    f_out = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    ρ = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    ux = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    uy = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    uz = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)

    copyto!(q_wall, q_wall_h)
    copyto!(is_solid, is_solid_h)
    fill!(uw_x, zero(T)); fill!(uw_y, zero(T)); fill!(uw_z, zero(T))
    copyto!(f_in, f_in_h)
    fill!(ρ, one(T)); fill!(ux, zero(T))
    fill!(uy, zero(T)); fill!(uz, zero(T))

    bcspec = _build_libb_bc_rebuild_spec_3d(setup, dx, Nx, Ny, Nz, T, backend)

    Fx_sum = 0.0
    Fy_sum = 0.0
    Fz_sum = 0.0
    n_avg = 0

    # Drag is averaged over the final window only (steady-state estimate).
    # Computing it every step would launch an extra reduction kernel per
    # step for no benefit; mirrors run_sphere_libb_3d's avg_window pattern.
    avg_window = max(1, setup.max_steps ÷ 5)
    for step in 1:setup.max_steps
        fused_trt_libb_v2_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                    q_wall, uw_x, uw_y, uw_z,
                                    Nx, Ny, Nz, ν)
        # Every lateral face is a half-way wall of the fused step (periodic
        # faces are refused above), so the inlet/outlet rim is rebuilt too.
        apply_bc_rebuild_3d!(f_out, f_in, bcspec, ν, Nx, Ny, Nz;
                             halfway_walls=(:south, :north, :bottom, :top))

        if step > setup.max_steps - avg_window
            drag = compute_drag_libb_3d(f_out, q_wall, Nx, Ny, Nz)
            Fx_sum += drag.Fx
            Fy_sum += drag.Fy
            Fz_sum += drag.Fz
            n_avg += 1
        end

        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    Fx_avg = n_avg == 0 ? 0.0 : Fx_sum / n_avg
    Fy_avg = n_avg == 0 ? 0.0 : Fy_sum / n_avg
    Fz_avg = n_avg == 0 ? 0.0 : Fz_sum / n_avg

    # Drag coefficient on the obstacle. Frontal area = projected silhouette of
    # the solid onto the y-z (cross-flow) plane — convention-free and matches
    # the analytic A=πR² for a sphere. Cd = 2·Fx / (u_ref²·A_frontal).
    frontal = 0
    @inbounds for k in 1:Nz, j in 1:Ny
        any_solid = false
        for i in 1:Nx
            is_solid_h[i, j, k] && (any_solid = true; break)
        end
        any_solid && (frontal += 1)
    end
    A_frontal = Float64(frontal)
    u_ref = Float64(maximum(abs, u_profile_h))
    Cd = (A_frontal > 0 && u_ref > 0) ? 2.0 * Fx_avg / (u_ref^2 * A_frontal) : 0.0
    D_eq = A_frontal > 0 ? 2.0 * sqrt(A_frontal / π) : 0.0

    return (; ρ=Array(ρ), ux=Array(ux), uy=Array(uy), uz=Array(uz),
            Fx=Fx_avg, Fy=Fy_avg, Fz=Fz_avg, Cd=Cd, A=A_frontal, D=D_eq,
            q_wall=Array(q_wall), is_solid=Array(is_solid),
            inlet_profile=u_profile_h, u_ref=u_ref)
end
