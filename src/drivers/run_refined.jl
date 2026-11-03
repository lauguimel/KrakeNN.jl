
"""Dispatch grid-refined cases (2D and 3D).

Handles both D2Q9 and D3Q19 lattices with isothermal and thermal modules.

Grid refinement is under development and not validated.
"""
function _run_refined(setup::SimulationSetup;
                      backend=KernelAbstractions.CPU(), T=Float64,
                      callback::Union{Nothing,Function}=nothing,
                      callback_every::Int=100)
    if setup.lattice === :D3Q19
        return _run_refined_3d(setup; backend=backend, T=T,
                               callback=callback, callback_every=callback_every)
    end

    is_thermal = :thermal in setup.modules

    # --- Generic isothermal refined path ---
    dom = setup.domain
    Nx, Ny = dom.Nx, dom.Ny
    dx = T(dom.Lx / Nx)
    dy = T(dom.Ly / Ny)
    ν = T(setup.physics.params[:nu])
    ω = T(1.0 / (3.0 * ν + 0.5))

    # Initialize base grid
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=Float64(ν), u_lid=0.0,
                       max_steps=setup.max_steps, output_interval=1000)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    _apply_geometry!(is_solid, setup, Float64(dx), Float64(dy))

    if setup.initial !== nothing
        _apply_initial_conditions!(f_in, f_out, setup, Float64(dx), Float64(dy), T)
    end

    # Body force (Guo)
    has_body_force = !isempty(setup.physics.body_force)
    Fx_val = T(0); Fy_val = T(0)
    if has_body_force
        Fx_val = haskey(setup.physics.body_force, :Fx) ?
            T(evaluate(setup.physics.body_force[:Fx])) : T(0)
        Fy_val = haskey(setup.physics.body_force, :Fy) ?
            T(evaluate(setup.physics.body_force[:Fy])) : T(0)
    end

    # Create patches from Refine blocks
    patches = RefinementPatch{T}[]
    for rs in setup.refinements
        patch = create_patch(rs.name, 1, rs.ratio,
            (Float64(rs.region[1]), Float64(rs.region[2]),
             Float64(rs.region[3]), Float64(rs.region[4])),
            Nx, Ny, Float64(dx), Float64(ω), T; backend=backend)
        push!(patches, patch)
    end
    domain = create_refined_domain(Nx, Ny, Float64(dx), Float64(ω), patches)

    # Apply geometry on fine patches (re-evaluate at fine resolution)
    for patch in patches
        _apply_patch_geometry!(patch, setup)
    end

    # Initialize patch interiors from coarse state
    compute_macroscopic_2d!(ρ, ux, uy, f_in)
    for patch in patches
        prolongate_f_rescaled_full_2d!(
            patch.f_in, f_in, ρ, ux, uy,
            patch.ratio, patch.Nx_inner, patch.Ny_inner,
            patch.n_ghost, first(patch.parent_i_range), first(patch.parent_j_range),
            Nx, Ny, Float64(ω), Float64(patch.omega))
        copyto!(patch.f_out, patch.f_in)
        compute_macroscopic_2d!(patch.rho, patch.ux, patch.uy, patch.f_in)
    end

    # --- Thermal refined setup (if :thermal module) ---
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

        # Detect temperature BCs from boundaries
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
        if abs(ΔT) < eps(T)
            ΔT = T(1)
        end

        # Use pre-computed gbeta_DT from Setup helper when available;
        # gbeta_DT = β·g·ΔT already in lattice units (consistent with ν, α, L_ref).
        if haskey(params, :gbeta_DT)
            β_g_val = T(params[:gbeta_DT]) / ΔT
        else
            H = T(max(Nx, Ny))
            Ra = T(get(params, :Ra, 1e4))
            β_g_val = Ra * ν * α_thermal / (ΔT * H^3)
        end
        T_ref_buoy = (T_hot + T_cold) / T(2)

        # Allocate thermal arrays on base grid
        g_in  = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
        g_out = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
        Temp  = KernelAbstractions.zeros(backend, T, Nx, Ny)

        # Initialize g to linear temperature profile between hot/cold walls
        w_lat = weights(D2Q9())
        T_mid = (T_hot + T_cold) / T(2)
        g_cpu = zeros(T, Nx, Ny, 9)
        # Detect gradient direction from thermal BCs
        hot_face = :west
        for (face, tv) in thermal_face_bcs
            tv ≈ T_hot && (hot_face = face)
        end
        for j in 1:Ny, i in 1:Nx
            t_frac = if hot_face in (:west, :east)
                T(hot_face == :west ? (i - 1) / max(Nx - 1, 1) :
                                      (Nx - i) / max(Nx - 1, 1))
            else  # south/north
                T(hot_face == :south ? (j - 1) / max(Ny - 1, 1) :
                                       (Ny - j) / max(Ny - 1, 1))
            end
            T_init = T_hot - (T_hot - T_cold) * t_frac
            for q in 1:9
                g_cpu[i, j, q] = T(w_lat[q]) * T_init
            end
        end
        copyto!(g_in, g_cpu)
        copyto!(g_out, g_cpu)

        # Create thermal patch arrays
        thermals = ThermalPatchArrays{T}[
            create_thermal_patch_arrays(p, Float64(ω_T);
                T_init=Float64(T_mid), backend=backend) for p in patches]

        # Initialize patch thermal from coarse
        compute_temperature_2d!(Temp, g_in)
        for (pidx, patch) in enumerate(patches)
            fill_thermal_full!(patch, thermals[pidx], g_in, Nx, Ny)
        end

        # Build thermal patch BCs (fixed-T on faces touching domain walls)
        bc_thermal_patch_fns = _build_patch_thermal_bcs(patches, setup, T)

        # Build coarse thermal BC closures
        for (face, T_val) in thermal_face_bcs
            if face == :south
                push!(thermal_bc_face_fns, (g, nx, ny) -> apply_fixed_temp_south_2d!(g, T_val, nx))
            elseif face == :north
                push!(thermal_bc_face_fns, (g, nx, ny) -> apply_fixed_temp_north_2d!(g, T_val, nx, ny))
            elseif face == :west
                push!(thermal_bc_face_fns, (g, nx, ny) -> apply_fixed_temp_west_2d!(g, T_val, ny))
            elseif face == :east
                push!(thermal_bc_face_fns, (g, nx, ny) -> apply_fixed_temp_east_2d!(g, T_val, nx, ny))
            end
        end
    end

    # Auto-detect patch BCs
    bc_patch_fns = _build_patch_flow_bcs(patches, setup)

    # Build closures
    stream_fn! = _select_streaming_kernel(setup)
    bc_handlers = _build_boundary_handlers(setup, Float64(dx), Float64(dy), Nx, Ny, T, backend)

    collide_fn = if has_body_force
        (f, is_s) -> collide_guo_2d!(f, is_s, ω, Fx_val, Fy_val)
    else
        (f, is_s) -> collide_2d!(f, is_s, ω)
    end

    macro_fn = if has_body_force
        (r, u, v, f) -> compute_macroscopic_forced_2d!(r, u, v, f, Fx_val, Fy_val)
    else
        compute_macroscopic_2d!
    end

    bc_base_fn = (f) -> _apply_boundary_conditions!(f, bc_handlers, 0, Nx, Ny,
                                                     Float64(dx), Float64(dy), dom, T)

    # Patch collide with force scaling (F/ratio)
    patch_collide_fns = nothing
    patch_macro_fn = nothing
    if has_body_force
        patch_collide_fns = Dict{Int, Function}()
        for (pidx, patch) in enumerate(patches)
            Fx_f = Fx_val / T(patch.ratio)
            Fy_f = Fy_val / T(patch.ratio)
            ω_f = patch.omega
            patch_collide_fns[pidx] = (f, is_s) -> collide_guo_2d!(f, is_s, ω_f, Fx_f, Fy_f)
        end
        patch_macro_fn = (r, u, v, f) -> compute_macroscopic_forced_2d!(r, u, v, f,
                                                                         Fx_val, Fy_val)
    end

    # --- Thermal fused step and coarse BC closures ---
    thermal_fused_step_fn = nothing
    thermal_bc_coarse_fn = nothing
    if is_thermal
        # Collect wall faces for explicit bounce-back after restriction
        wall_faces = Symbol[bc.face for bc in setup.boundaries if bc.type == :wall]

        let sfn=stream_fn!, bch=bc_handlers, d_=dom, dx_=dx, dy_=dy,
            ρ_=ρ, ux_=ux, uy_=uy, is_s=is_solid, ω_=ω, ωT_=ω_T,
            βg_=β_g_val, Tref_=T_ref_buoy, tbcfns=thermal_bc_face_fns, T_=T,
            wf=wall_faces, Nx_=Nx, Ny_=Ny

            thermal_fused_step_fn = (f_o, f_i, g_o, g_i, Te, nx, ny) -> begin
                sfn(f_o, f_i, nx, ny)
                sfn(g_o, g_i, nx, ny)
                _apply_boundary_conditions!(f_o, bch, 0, nx, ny,
                    Float64(dx_), Float64(dy_), d_, T_)
                for bfn in tbcfns
                    bfn(g_o, nx, ny)
                end
                compute_temperature_2d!(Te, g_o)
                compute_macroscopic_2d!(ρ_, ux_, uy_, f_o)
                collide_thermal_2d!(g_o, ux_, uy_, ωT_)
                collide_boussinesq_2d!(f_o, Te, is_s, ω_, βg_, Tref_)
            end

            thermal_bc_coarse_fn = (f, g, Te, nx, ny) -> begin
                # Explicit bounce-back on wall faces (stream_2d! handles it during
                # the fused step, but restriction overwrites wall cells)
                for face in wf
                    apply_bounce_back_wall_2d!(f, Nx_, Ny_, face)
                end
                _apply_boundary_conditions!(f, bch, 0, nx, ny,
                    Float64(dx_), Float64(dy_), d_, T_)
                for bfn in tbcfns
                    bfn(g, nx, ny)
                end
            end
        end
    end

    # Output setup
    vtk_out = _find_output(setup, :vtk)
    pvd = nothing
    output_dir = ""
    if vtk_out !== nothing
        output_dir = setup_output_dir(vtk_out.directory)
        pvd = create_pvd(joinpath(output_dir, setup.name))
    end
    png_out = _find_output(setup, :png)
    gif_out = _find_output(setup, :gif)
    gif_frames = _init_gif_frames(gif_out)
    _check_image_backend(png_out, gif_out)

    # --- Time loop ---
    for step in 1:setup.max_steps
        if is_thermal
            f_in, f_out, g_in, g_out = advance_thermal_refined_step!(
                domain, thermals,
                f_in, f_out, g_in, g_out, ρ, ux, uy, Temp, is_solid;
                fused_step_fn=thermal_fused_step_fn,
                omega_T_coarse=Float64(ω_T),
                β_g=Float64(β_g_val),
                T_ref_buoy=Float64(T_ref_buoy),
                bc_thermal_patch_fns=bc_thermal_patch_fns,
                bc_flow_patch_fns=bc_patch_fns,
                bc_coarse_fn=thermal_bc_coarse_fn)
        else
            f_in, f_out = advance_refined_step!(domain, f_in, f_out, ρ, ux, uy, is_solid;
                stream_fn=stream_fn!, collide_fn=collide_fn, macro_fn=macro_fn,
                bc_base_fn=bc_base_fn, bc_patch_fns=bc_patch_fns,
                patch_collide_fns=patch_collide_fns, patch_macro_fn=patch_macro_fn)
        end

        # Re-apply coarse BCs after restriction (patches may overwrite wall cells)
        _apply_boundary_conditions!(f_in, bc_handlers, step, Nx, Ny,
                                    Float64(dx), Float64(dy), dom, T)
        compute_macroscopic_2d!(ρ, ux, uy, f_in)
        if is_thermal
            for bfn in thermal_bc_face_fns
                bfn(g_in, Nx, Ny)
            end
            compute_temperature_2d!(Temp, g_in)
        end

        # VTK output
        if vtk_out !== nothing && step % vtk_out.interval == 0
            _write_output(ρ, ux, uy, setup, vtk_out, pvd, output_dir, Float64(dx), step)
        end

        # PNG/GIF
        _maybe_save_png(png_out, ρ, ux, uy, setup, output_dir, step)
        _maybe_collect_gif(gif_out, gif_frames, ρ, ux, uy, step)

        # Callback
        if callback !== nothing && step % callback_every == 0
            cb_state = is_thermal ?
                (; rho=Array(ρ), ux=Array(ux), uy=Array(uy), Temp=Array(Temp)) :
                (; rho=Array(ρ), ux=Array(ux), uy=Array(uy))
            callback(step, cb_state)
        end
    end

    # Finalize outputs
    pvd !== nothing && vtk_save(pvd)
    _maybe_save_gif(gif_out, gif_frames, setup, output_dir)

    if is_thermal
        # Compute Nusselt from the finest patch touching a heated wall
        Nu = _compute_nusselt_from_patches(domain, thermals, Temp, setup, T)
        return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), Temp=Array(Temp),
                Nu=Nu, setup=setup)
    end
    return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), setup=setup)
end
