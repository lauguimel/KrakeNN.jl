
# =========================================================================
# 3D refined runner (D3Q19 + Refine blocks)
# =========================================================================

"""Dispatch grid-refined cases for D3Q19 (3D). Grid refinement is under development and not validated."""
function _run_refined_3d(setup::SimulationSetup;
                          backend=KernelAbstractions.CPU(), T=Float64,
                          callback::Union{Nothing,Function}=nothing,
                          callback_every::Int=100)
    is_thermal = :thermal in setup.modules
    dom = setup.domain
    Nx, Ny, Nz = dom.Nx, dom.Ny, dom.Nz
    Lx, Ly, Lz = dom.Lx, dom.Ly, dom.Lz
    dx = T(Lx / Nx)
    ν = T(setup.physics.params[:nu])
    ω = T(1.0 / (3.0 * ν + 0.5))

    # Initialize 3D base grid
    config = LBMConfig(D3Q19(); Nx=Nx, Ny=Ny, Nz=Nz, ν=Float64(ν), u_lid=0.0,
                       max_steps=setup.max_steps, output_interval=1000)
    state = initialize_3d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ = state.ρ; ux = state.ux; uy = state.uy; uz = state.uz
    is_solid = state.is_solid

    _apply_geometry_3d!(is_solid, setup, Float64(dx))

    if setup.initial !== nothing
        _apply_initial_conditions_3d!(f_in, f_out, setup, Float64(dx), T)
    end

    # Create 3D patches from Refine blocks
    patches = RefinementPatch3D{T}[]
    for rs in setup.refinements
        region_6 = rs.is_3d ? rs.region_3d :
            (Float64(rs.region[1]), Float64(rs.region[2]), 0.0,
             Float64(rs.region[3]), Float64(rs.region[4]), Float64(Lz))
        patch = create_patch_3d(rs.name, 1, rs.ratio, region_6,
            Nx, Ny, Nz, Float64(dx), Float64(ω), T; backend=backend)
        push!(patches, patch)
    end
    domain = create_refined_domain_3d(Nx, Ny, Nz, Float64(dx), Float64(ω), patches)

    # Apply geometry on fine patches
    for patch in patches
        _apply_patch_geometry_3d!(patch, setup)
    end

    # Initialize patch interiors from coarse state
    compute_macroscopic_3d!(ρ, ux, uy, uz, f_in)
    for patch in patches
        prolongate_f_rescaled_full_3d!(
            patch.f_in, f_in, ρ, ux, uy, uz,
            patch.ratio, patch.Nx_inner, patch.Ny_inner, patch.Nz_inner,
            patch.n_ghost,
            first(patch.parent_i_range), first(patch.parent_j_range), first(patch.parent_k_range),
            Nx, Ny, Nz, Float64(ω), Float64(patch.omega))
        copyto!(patch.f_out, patch.f_in)
        compute_macroscopic_3d!(patch.rho, patch.ux, patch.uy, patch.uz, patch.f_in)
    end

    # Build wall face list for BCs
    wall_faces = Symbol[bc.face for bc in setup.boundaries if bc.type == :wall]

    # Build patch flow BCs (bounce-back on faces touching domain walls)
    bc_patch_fns = build_patch_flow_bcs_3d(patches, Lx, Ly, Lz, Nx; wall_faces=wall_faces)

    # Build coarse boundary closure
    bc_base_fn = (f) -> begin
        for face in wall_faces
            apply_bounce_back_wall_3d!(f, Nx, Ny, Nz, face)
        end
    end

    # Collide closure
    collide_fn = (f, is_s) -> collide_3d!(f, is_s, ω)
    macro_fn = compute_macroscopic_3d!

    # --- Thermal setup ---
    g_in = nothing; g_out = nothing; Temp = nothing
    thermals = nothing
    ω_T = T(0); β_g_val = T(0); T_ref_buoy = T(0)
    bc_thermal_patch_fns = nothing
    thermal_bc_face_fns = Function[]

    if is_thermal
        params = setup.physics.params
        Pr = T(get(params, :Pr, 0.71))
        α_thermal = haskey(params, :alpha) ? T(params[:alpha]) : ν / Pr
        ω_T = T(1.0 / (3.0 * Float64(α_thermal) + 0.5))

        # Detect temperature BCs
        thermal_face_bcs = Dict{Symbol, T}()
        for bc in setup.boundaries
            if haskey(bc.values, :T)
                T_val = T(evaluate(bc.values[:T]))
                thermal_face_bcs[bc.face] = T_val
            end
        end
        T_hot = isempty(thermal_face_bcs) ? T(1) : maximum(values(thermal_face_bcs))
        T_cold = isempty(thermal_face_bcs) ? T(0) : minimum(values(thermal_face_bcs))
        ΔT = T_hot - T_cold
        abs(ΔT) < eps(T) && (ΔT = T(1))

        if haskey(params, :gbeta_DT)
            β_g_val = T(params[:gbeta_DT]) / ΔT
        else
            H = T(max(Nx, Ny, Nz))
            Ra = T(get(params, :Ra, 1e4))
            β_g_val = Ra * ν * α_thermal / (ΔT * H^3)
        end
        T_ref_buoy = (T_hot + T_cold) / T(2)

        # Allocate thermal arrays
        g_in  = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz, 19)
        g_out = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz, 19)
        Temp  = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz)

        # Initialize g to linear temperature profile
        w_lat = weights(D3Q19())
        T_mid = (T_hot + T_cold) / T(2)
        g_cpu = zeros(T, Nx, Ny, Nz, 19)
        hot_face = :west
        for (face, tv) in thermal_face_bcs
            tv ≈ T_hot && (hot_face = face)
        end
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            t_frac = if hot_face in (:west, :east)
                T(hot_face == :west ? (i - 1) / max(Nx - 1, 1) : (Nx - i) / max(Nx - 1, 1))
            elseif hot_face in (:south, :north)
                T(hot_face == :south ? (j - 1) / max(Ny - 1, 1) : (Ny - j) / max(Ny - 1, 1))
            else  # bottom/top
                T(hot_face == :bottom ? (k - 1) / max(Nz - 1, 1) : (Nz - k) / max(Nz - 1, 1))
            end
            T_init = T_hot - (T_hot - T_cold) * t_frac
            for q in 1:19
                g_cpu[i, j, k, q] = T(w_lat[q]) * T_init
            end
        end
        copyto!(g_in, g_cpu)
        copyto!(g_out, g_cpu)

        # Create thermal patch arrays
        thermals = ThermalPatchArrays3D{T}[
            create_thermal_patch_arrays_3d(p, Float64(ω_T);
                T_init=Float64(T_mid), backend=backend) for p in patches]

        # Initialize patch thermal from coarse
        compute_temperature_3d!(Temp, g_in)
        for (pidx, patch) in enumerate(patches)
            fill_thermal_full_3d!(patch, thermals[pidx], g_in, Nx, Ny, Nz)
        end

        # Build thermal patch BCs
        bc_thermal_patch_fns = build_patch_thermal_bcs_3d(
            patches, Lx, Ly, Lz, Nx, thermal_face_bcs)

        # Build coarse thermal BC closures
        for (face, T_val) in thermal_face_bcs
            if face == :south
                push!(thermal_bc_face_fns, (g, nx, ny, nz) -> apply_fixed_temp_south_3d!(g, T_val, nx, nz))
            elseif face == :north
                push!(thermal_bc_face_fns, (g, nx, ny, nz) -> apply_fixed_temp_north_3d!(g, T_val, nx, ny, nz))
            elseif face == :west
                push!(thermal_bc_face_fns, (g, nx, ny, nz) -> apply_fixed_temp_west_3d!(g, T_val, ny, nz))
            elseif face == :east
                push!(thermal_bc_face_fns, (g, nx, ny, nz) -> apply_fixed_temp_east_3d!(g, T_val, nx, ny, nz))
            elseif face == :bottom
                push!(thermal_bc_face_fns, (g, nx, ny, nz) -> apply_fixed_temp_bottom_3d!(g, T_val, nx, ny))
            elseif face == :top
                push!(thermal_bc_face_fns, (g, nx, ny, nz) -> apply_fixed_temp_top_3d!(g, T_val, nx, ny, nz))
            end
        end
    end

    # --- Build fused thermal step and coarse BC closures ---
    thermal_fused_step_fn = nothing
    thermal_bc_coarse_fn = nothing
    if is_thermal
        let wf=wall_faces, bc_base=bc_base_fn, ρ_=ρ, ux_=ux, uy_=uy, uz_=uz,
            is_s=is_solid, ω_=ω, ωT_=ω_T, βg_=β_g_val, Tref_=T_ref_buoy,
            tbcfns=thermal_bc_face_fns, Nx_=Nx, Ny_=Ny, Nz_=Nz

            thermal_fused_step_fn = (f_o, f_i, g_o, g_i, Te, nx, ny, nz) -> begin
                stream_3d!(f_o, f_i, nx, ny, nz)
                stream_3d!(g_o, g_i, nx, ny, nz)
                bc_base(f_o)
                for bfn in tbcfns
                    bfn(g_o, nx, ny, nz)
                end
                compute_temperature_3d!(Te, g_o)
                compute_macroscopic_3d!(ρ_, ux_, uy_, uz_, f_o)
                collide_thermal_3d!(g_o, ux_, uy_, uz_, ωT_)
                collide_boussinesq_3d!(f_o, Te, is_s, ω_, βg_, Tref_)
            end

            thermal_bc_coarse_fn = (f, g, Te, nx, ny, nz) -> begin
                for face in wf
                    apply_bounce_back_wall_3d!(f, Nx_, Ny_, Nz_, face)
                end
                for bfn in tbcfns
                    bfn(g, nx, ny, nz)
                end
            end
        end
    end

    # Output setup
    vtk_out = _find_output(setup, :vtk)
    pvd = nothing; output_dir = ""
    if vtk_out !== nothing
        output_dir = setup_output_dir(vtk_out.directory)
        pvd = create_pvd(joinpath(output_dir, setup.name))
    end

    # --- Time loop ---
    for step in 1:setup.max_steps
        if is_thermal
            f_in, f_out, g_in, g_out = advance_thermal_refined_step_3d!(
                domain, thermals,
                f_in, f_out, g_in, g_out, ρ, ux, uy, uz, Temp, is_solid;
                fused_step_fn=thermal_fused_step_fn,
                omega_T_coarse=Float64(ω_T),
                β_g=Float64(β_g_val),
                T_ref_buoy=Float64(T_ref_buoy),
                bc_thermal_patch_fns=bc_thermal_patch_fns,
                bc_flow_patch_fns=bc_patch_fns,
                bc_coarse_fn=thermal_bc_coarse_fn)
        else
            f_in, f_out = advance_refined_step_3d!(
                domain, f_in, f_out, ρ, ux, uy, uz, is_solid;
                stream_fn=stream_3d!, collide_fn=collide_fn, macro_fn=macro_fn,
                bc_base_fn=bc_base_fn, bc_patch_fns=bc_patch_fns)
        end

        # Re-apply coarse BCs after restriction
        bc_base_fn(f_in)
        compute_macroscopic_3d!(ρ, ux, uy, uz, f_in)
        if is_thermal
            for bfn in thermal_bc_face_fns
                bfn(g_in, Nx, Ny, Nz)
            end
            compute_temperature_3d!(Temp, g_in)
        end

        # VTK output
        if vtk_out !== nothing && step % vtk_out.interval == 0
            _write_output_3d(ρ, ux, uy, uz, setup, vtk_out, pvd, output_dir, Float64(dx), step)
        end

        # Callback
        if callback !== nothing && step % callback_every == 0
            cb_state = is_thermal ?
                (; rho=Array(ρ), ux=Array(ux), uy=Array(uy), uz=Array(uz), Temp=Array(Temp)) :
                (; rho=Array(ρ), ux=Array(ux), uy=Array(uy), uz=Array(uz))
            callback(step, cb_state)
        end
    end

    # Finalize outputs
    pvd !== nothing && vtk_save(pvd)

    if is_thermal
        return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), uz=Array(uz),
                Temp=Array(Temp), setup=setup)
    end
    return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), uz=Array(uz), setup=setup)
end

# --- 3D geometry helpers ---

"""Apply initial conditions to 3D grid."""
function _apply_initial_conditions_3d!(f_in, f_out, setup::SimulationSetup,
                                        dx::Float64, ::Type{T}) where T
    setup.initial === nothing && return
    # For now, 3D initial conditions default to equilibrium (u=0, ρ=1)
    # Custom IC functions can be added later
end

# --- Refinement helpers ---

"""Auto-detect which patch faces touch domain walls and build BC closures."""
function _build_patch_flow_bcs(patches::Vector{RefinementPatch{T}},
                                setup::SimulationSetup) where T
    Lx, Ly = setup.domain.Lx, setup.domain.Ly
    tol = Lx / setup.domain.Nx * 0.01

    bc_map = Dict{Symbol, BoundarySetup}()
    for bc in setup.boundaries
        bc.type == :periodic && continue
        bc_map[bc.face] = bc
    end

    patch_bcs = Dict{Int, Function}()
    for (pidx, patch) in enumerate(patches)
        closures = Function[]

        for (face, condition) in [
            (:west,  Float64(patch.x_min) <= tol),
            (:east,  Float64(patch.x_max) >= Lx - tol),
            (:south, Float64(patch.y_min) <= tol),
            (:north, Float64(patch.y_max) >= Ly - tol),
        ]
            condition || continue
            bc = get(bc_map, face, nothing)
            bc === nothing && continue
            cl = _make_patch_face_bc(face, bc, T)
            cl !== nothing && push!(closures, cl)
        end

        if !isempty(closures)
            # Capture closures in a local let-block to avoid closure issues
            let cls = closures
                patch_bcs[pidx] = (f, Nx_p, Ny_p) -> begin
                    for cl in cls
                        cl(f, Nx_p, Ny_p)
                    end
                end
            end
        end
    end
    return patch_bcs
end

"""Create a BC closure for one face of a patch.

Only wall (bounce-back) BCs are applied directly on patches. Velocity and
pressure BCs are handled implicitly through the ghost-fill prolongation from
the coarse grid, which already has these BCs applied. Applying Zou-He or
pressure BCs directly on the patch would target ghost rows (outside the
physical domain), producing incorrect results.
"""
function _make_patch_face_bc(face::Symbol, bc::BoundarySetup, ::Type{T}) where T
    if bc.type == :wall
        return (f, Nx_p, Ny_p) -> apply_bounce_back_wall_2d!(f, Nx_p, Ny_p, face)
    else
        # velocity / pressure / periodic: handled by ghost fill from coarse grid
        return nothing
    end
end

"""Auto-detect which patch faces touch domain walls with thermal BCs."""
function _build_patch_thermal_bcs(patches::Vector{RefinementPatch{T}},
                                   setup::SimulationSetup, ::Type{T}) where T
    Lx, Ly = setup.domain.Lx, setup.domain.Ly
    tol = Lx / setup.domain.Nx * 0.01

    # Collect faces with fixed temperature
    thermal_face_bcs = Dict{Symbol, T}()
    for bc in setup.boundaries
        if haskey(bc.values, :T)
            thermal_face_bcs[bc.face] = T(evaluate(bc.values[:T]))
        end
    end

    patch_bcs = Dict{Int, Function}()
    for (pidx, patch) in enumerate(patches)
        closures = Function[]

        for (face, condition) in [
            (:west,  Float64(patch.x_min) <= tol),
            (:east,  Float64(patch.x_max) >= Lx - tol),
            (:south, Float64(patch.y_min) <= tol),
            (:north, Float64(patch.y_max) >= Ly - tol),
        ]
            condition || continue
            T_val = get(thermal_face_bcs, face, nothing)
            T_val === nothing && continue
            cl = _make_patch_thermal_bc(face, T_val)
            push!(closures, cl)
        end

        if !isempty(closures)
            let cls = closures
                patch_bcs[pidx] = (g, Nx_p, Ny_p) -> begin
                    for cl in cls
                        cl(g, Nx_p, Ny_p)
                    end
                end
            end
        end
    end
    return patch_bcs
end

"""Create a thermal BC closure for one face of a patch (anti-bounce-back Dirichlet)."""
function _make_patch_thermal_bc(face::Symbol, T_val)
    if face == :south
        return (g, Nx_p, Ny_p) -> apply_fixed_temp_south_2d!(g, T_val, Nx_p)
    elseif face == :north
        return (g, Nx_p, Ny_p) -> apply_fixed_temp_north_2d!(g, T_val, Nx_p, Ny_p)
    elseif face == :west
        return (g, Nx_p, Ny_p) -> apply_fixed_temp_west_2d!(g, T_val, Ny_p)
    elseif face == :east
        return (g, Nx_p, Ny_p) -> apply_fixed_temp_east_2d!(g, T_val, Nx_p, Ny_p)
    end
end

"""Compute Nusselt number from the finest patch touching a heated wall."""
function _compute_nusselt_from_patches(domain::RefinedDomain{T},
                                        thermals::Vector{ThermalPatchArrays{T}},
                                        Temp_coarse, setup, ::Type{T}) where T
    # Find temperature extremes from boundary BCs
    T_hot = T(-Inf); T_cold = T(Inf)
    hot_face = :west
    for bc in setup.boundaries
        if haskey(bc.values, :T)
            T_val = T(evaluate(bc.values[:T]))
            if T_val > T_hot
                T_hot = T_val
                hot_face = bc.face
            end
            if T_val < T_cold
                T_cold = T_val
            end
        end
    end
    ΔT = T_hot - T_cold
    if abs(ΔT) < eps(T) || isinf(T_hot)
        return T(NaN)
    end

    # Use physical domain size for consistent units with patch.dx
    H = T(max(setup.domain.Lx, setup.domain.Ly))

    # Find the patch that touches the hot wall (for fine-grid gradient)
    Lx, Ly = setup.domain.Lx, setup.domain.Ly
    tol = Lx / setup.domain.Nx * 0.01

    for (pidx, patch) in enumerate(domain.patches)
        touches_hot = (hot_face == :west  && Float64(patch.x_min) <= tol) ||
                      (hot_face == :east  && Float64(patch.x_max) >= Lx - tol) ||
                      (hot_face == :south && Float64(patch.y_min) <= tol) ||
                      (hot_face == :north && Float64(patch.y_max) >= Ly - tol)
        touches_hot || continue

        T_fine = Array(thermals[pidx].Temp)
        ng = patch.n_ghost
        dx_f = T(patch.dx)

        # Compute wall-normal gradient via 2nd-order one-sided FD
        if hot_face == :west
            Ny_fine = patch.Ny_inner
            Nu_arr = zeros(T, Ny_fine)
            for jf in 1:Ny_fine
                j = jf + ng
                i1, i2, i3 = ng + 1, ng + 2, ng + 3
                dTdn = (-3 * T_fine[i1, j] + 4 * T_fine[i2, j] - T_fine[i3, j]) / (2 * dx_f)
                Nu_arr[jf] = -H * dTdn / ΔT
            end
            return sum(Nu_arr[2:end-1]) / max(Ny_fine - 2, 1)
        elseif hot_face == :east
            Ny_fine = patch.Ny_inner
            Nu_arr = zeros(T, Ny_fine)
            Nx_p = patch.Nx
            for jf in 1:Ny_fine
                j = jf + ng
                i1, i2, i3 = Nx_p - ng, Nx_p - ng - 1, Nx_p - ng - 2
                dTdn = (3 * T_fine[i1, j] - 4 * T_fine[i2, j] + T_fine[i3, j]) / (2 * dx_f)
                Nu_arr[jf] = H * dTdn / ΔT
            end
            return sum(Nu_arr[2:end-1]) / max(Ny_fine - 2, 1)
        elseif hot_face == :south
            Nx_fine = patch.Nx_inner
            Nu_arr = zeros(T, Nx_fine)
            for if_ in 1:Nx_fine
                i = if_ + ng
                j1, j2, j3 = ng + 1, ng + 2, ng + 3
                dTdn = (-3 * T_fine[i, j1] + 4 * T_fine[i, j2] - T_fine[i, j3]) / (2 * dx_f)
                Nu_arr[if_] = -H * dTdn / ΔT
            end
            return sum(Nu_arr[2:end-1]) / max(Nx_fine - 2, 1)
        elseif hot_face == :north
            Nx_fine = patch.Nx_inner
            Nu_arr = zeros(T, Nx_fine)
            Ny_p = patch.Ny
            for if_ in 1:Nx_fine
                i = if_ + ng
                j1, j2, j3 = Ny_p - ng, Ny_p - ng - 1, Ny_p - ng - 2
                dTdn = (3 * T_fine[i, j1] - 4 * T_fine[i, j2] + T_fine[i, j3]) / (2 * dx_f)
                Nu_arr[if_] = H * dTdn / ΔT
            end
            return sum(Nu_arr[2:end-1]) / max(Nx_fine - 2, 1)
        end
    end

    # No patch on the hot wall: fall back to coarse-grid Nusselt
    T_cpu = Array(Temp_coarse)
    dx_c = T(setup.domain.Lx / setup.domain.Nx)
    if hot_face == :west
        Ny_c = setup.domain.Ny
        Nu_arr = zeros(T, Ny_c)
        for j in 1:Ny_c
            dTdn = (-3*T_cpu[1,j] + 4*T_cpu[2,j] - T_cpu[3,j]) / (2*dx_c)
            Nu_arr[j] = -H * dTdn / ΔT
        end
        return sum(Nu_arr[2:end-1]) / max(Ny_c - 2, 1)
    end
    return T(NaN)
end
