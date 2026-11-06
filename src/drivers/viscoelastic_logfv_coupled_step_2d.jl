"""
    _run_viscoelastic_logfv_step_channel_coupled_2d(geom_h; shear_length, kwargs...)

Shared coupled log-FV polymer driver behind the public `run_viscoelastic_logfv_*_coupled_2d`
wrappers, which forward their keywords to it. Only the stop option is described here.

# Stop option

- `stop`: `:none` (default) runs exactly `max_steps` steps and computes the same
  values as without the option. `:cd_change` samples the total drag coefficient
  `Cd = Cd_s + Cd_p - Cd_bsd` every `check_every` steps, from step 1 and independently
  of the averaging window, and stops when two consecutive samples satisfy
  `|Cd_k - Cd_(k-1)| <= tol * |Cd_k|`. It needs the drag keywords
  (`drag_cx`, `drag_cy`, `drag_radius`, `drag_u_ref`), which the cylinder wrapper sets.
  `max_steps` stays the hard cap.
- `tol`: relative tolerance of the `:cd_change` rule (default `2e-4`).
- `check_every`: steps between two samples. Default `nothing`: `round(Int, Nx / u_mean)`
  steps, one flow-through of the domain at the mean velocity. Set it to the window the
  stop criterion is written for.
- `step_callback(step, fields)`: called every step as before. If it returns exactly
  `true`, the run stops after that step. Any other return value is ignored, as before.

The result carries, in addition to its previous fields, `converged::Bool` (the
`:cd_change` rule was met, or the callback returned `true`), `stop_reason` (`:rule`,
`:callback`, or `:max_steps`), `stop_step` (steps run), `cd_history` (vector of
`(step, Cd)` samples, empty with `stop = :none`) and `Cd_last` (last sample, `NaN`
without samples).

`Cd` and the `Cd_*`, `F*` fields keep their meaning: the mean over the window samples
(the last `avg_window` steps of `max_steps`). A run stopped inside that window averages
the samples taken so far. A run stopped before the window has none, so they take the
values of the last `cd_history` sample (`Cd == Cd_last`). Same rule as
`run_cylinder_libb_2d`.
"""
function _run_viscoelastic_logfv_step_channel_coupled_2d(
    geom_h;
    shear_length::Real,
    nu_s::Real=0.08,
    nu_p::Real=0.02,
    lambda::Real=5.0,
    polymer_model=:oldroydb,
    L_max::Real=10.0,
    u_mean::Real=0.01,
    Fx_body::Real=2e-7,
    bsd_fraction::Real=1.0,
    polymer_substeps=:auto,
    subcycle_relative_tolerance::Real=0.01,
    max_deformation_increment::Real=0.05,
    max_memory_deformation_increment::Real=0.07,
    max_polymer_substeps::Integer=64,
    max_steps::Integer=60,
    avg_window::Union{Nothing,Integer}=nothing,
    drag_stride::Integer=1,
    diagnostic_stride::Integer=0,
    embedded_gradient::Bool=false,
    embedded_advection::Bool=false,
    embedded_force::Bool=false,
    embedded_drag::Bool=false,
    embedded_geometry=:qwall,
    embedded_circle_samples::Integer=32,
    force_boundary_fill::Symbol=:bc_aware,
    advection_scheme::Symbol=:rusanov,
    wall_bc::Symbol=:halfwayBB,
    step_callback::Union{Nothing,Function}=nothing,
    stop::Symbol=:none,
    tol::Real=2e-4,
    check_every::Union{Nothing,Integer}=nothing,
    drag_cx::Union{Nothing,Real}=nothing,
    drag_cy::Union{Nothing,Real}=nothing,
    drag_radius::Union{Nothing,Real}=nothing,
    drag_u_ref::Union{Nothing,Real}=nothing,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    @trace_enter :driver_step_entry
    shear_length > 0 || throw(ArgumentError("shear_length must be positive"))
    nu_s > 0 || throw(ArgumentError("nu_s must be positive"))
    nu_p >= 0 || throw(ArgumentError("nu_p must be non-negative"))
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    0 <= bsd_fraction <= 1 || throw(ArgumentError("bsd_fraction must be in [0, 1]"))
    max_steps >= 0 || throw(ArgumentError("max_steps must be non-negative"))
    drag_stride > 0 || throw(ArgumentError("drag_stride must be positive"))
    stop in (:none, :cd_change) ||
        throw(ArgumentError("stop must be :none or :cd_change, got $(stop)"))
    tol >= 0 || throw(ArgumentError("tol must be non-negative"))
    diagnostic_stride >= 0 || throw(ArgumentError("diagnostic_stride must be non-negative"))
    embedded_geometry_symbol = Symbol(embedded_geometry)
    embedded_geometry_symbol in (:qwall, :circle) ||
        throw(ArgumentError("embedded_geometry must be :qwall or :circle"))
    embedded_circle_samples > 0 ||
        throw(ArgumentError("embedded_circle_samples must be positive"))
    force_boundary_fill in (:bc_aware, :nearest, :none) ||
        throw(ArgumentError("force_boundary_fill must be :bc_aware, :nearest, or :none"))
    advection_scheme_symbol = Symbol(replace(lowercase(String(advection_scheme)), '-' => '_'))
    advection_scheme_symbol in (:rusanov, :muscl_superbee, :muscl_superbee_relax) ||
        throw(ArgumentError("advection_scheme must be :rusanov, :muscl_superbee, or :muscl_superbee_relax"))
    wall_bc in (:halfwayBB, :bouzidi_fl, :bouzidi_fl_twopass) ||
        throw(ArgumentError("wall_bc must be :halfwayBB, :bouzidi_fl, or :bouzidi_fl_twopass"))
    if embedded_geometry_symbol === :circle &&
       (isnothing(drag_cx) || isnothing(drag_cy) || isnothing(drag_radius))
        throw(ArgumentError(
            "embedded_geometry=:circle requires drag_cx, drag_cy, and drag_radius",
        ))
    end

    geom = transfer_step_geometry_2d(geom_h, backend)
    Nx, Ny = geom_h.Nx, geom_h.Ny
    is_solid = geom.is_solid
    q_wall = geom.q_wall
    is_solid_h = geom_h.is_solid
    dx = one(T)
    dy = one(T)
    embedded_circle_cx = isnothing(drag_cx) ? T(NaN) : T(drag_cx) + dx / T(2)
    embedded_circle_cy = isnothing(drag_cy) ? T(NaN) : T(drag_cy) + dy / T(2)
    embedded_h = if embedded_geometry_symbol === :circle
        # The FVFD circle lowering samples control volumes centered at
        # (i-0.5,j-0.5), while the LBM q_wall/is_solid mask is node-centered
        # at (i-1,j-1).  Shift the FVFD circle into that coordinate frame so
        # every LBM-fluid cell has a positive FVFD fluid volume.
        fvfd_embedded_boundary_from_circle_2d(
            Nx, Ny, embedded_circle_cx, embedded_circle_cy, T(drag_radius);
            FT=T, samples=embedded_circle_samples,
        )
    else
        fvfd_embedded_boundary_from_qwall_2d(geom_h.q_wall; FT=T)
    end
    embedded = fvfd_transfer_embedded_boundary_2d(embedded_h, backend, T)

    nu_s_t = T(nu_s)
    nu_p_t = T(nu_p)
    nu_total_t = nu_s_t + nu_p_t
    bsd_t = T(bsd_fraction)
    lambda_t = T(lambda)
    model_cfg = _logfv_polymer_model_config(polymer_model, L_max, T)
    model_code = model_cfg.model_code
    L2_t = model_cfg.L2
    prefactor_t = nu_p_t / lambda_t
    Fx_body_t = T(Fx_body)
    nu_lbm_t = nu_s_t + bsd_t * nu_p_t
    nu_lbm_t > zero(T) || throw(ArgumentError("nu_s + bsd_fraction * nu_p must be positive"))
    shear_length_t = T(shear_length)

    u_profile_h = parabolic_face_profile_2d(geom_h; face=:west, mean_velocity=T(u_mean), FT=T)
    u_profile = KernelAbstractions.allocate(backend, T, Ny)
    copyto!(u_profile, u_profile_h)
    bcspec = default_step_bcspec_2d(geom, u_profile, one(T))

    f_in = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_out = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_in_h = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        ux0 = (is_solid_h[i, j] || embedded_h.cut_count[i, j] > 0) ?
              zero(T) : u_profile_h[j]
        f_in_h[i, j, q] = equilibrium(D2Q9(), one(T), ux0, zero(T), q)
    end
    copyto!(f_in, f_in_h)
    fill!(f_out, zero(T))

    rho = KernelAbstractions.zeros(backend, T, Nx, Ny)
    ux = KernelAbstractions.zeros(backend, T, Nx, Ny)
    uy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    uwx = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    uwy = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)

    psixx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixx_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixx_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    west_xx = KernelAbstractions.zeros(backend, T, Ny)
    west_xy = KernelAbstractions.zeros(backend, T, Ny)
    west_yy = KernelAbstractions.zeros(backend, T, Ny)
    east_xx = KernelAbstractions.zeros(backend, T, Ny)
    east_xy = KernelAbstractions.zeros(backend, T, Ny)
    east_yy = KernelAbstractions.zeros(backend, T, Ny)
    south_xx = KernelAbstractions.zeros(backend, T, Nx)
    south_xy = KernelAbstractions.zeros(backend, T, Nx)
    south_yy = KernelAbstractions.zeros(backend, T, Nx)
    north_xx = KernelAbstractions.zeros(backend, T, Nx)
    north_xy = KernelAbstractions.zeros(backend, T, Nx)
    north_yy = KernelAbstractions.zeros(backend, T, Nx)
    ux_east = KernelAbstractions.zeros(backend, T, Ny)
    uy_south = KernelAbstractions.zeros(backend, T, Nx)
    uy_north = KernelAbstractions.zeros(backend, T, Nx)
    ux_face = KernelAbstractions.zeros(backend, T, Nx + 1, Ny)
    uy_face = KernelAbstractions.zeros(backend, T, Nx, Ny + 1)
    dudx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dudy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_total = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_total = KernelAbstractions.zeros(backend, T, Nx, Ny)
    drag_tx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    drag_ty = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tau_bsd_xx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tau_bsd_xy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tau_bsd_yy = KernelAbstractions.zeros(backend, T, Nx, Ny)

    inlet_shear_estimate = T(4) * abs(T(u_mean)) / shear_length_t
    body_shear_estimate = abs(Fx_body_t) * T(Ny) / (T(2) * max(nu_total_t, eps(T)))
    max_grad_norm_estimate = max(inlet_shear_estimate, body_shear_estimate)
    subcycle_estimate = logfv_oldroydb_subcycle_estimate(
        Float64(max_grad_norm_estimate),
        Float64(lambda_t),
        1.0;
        relative_tolerance=Float64(subcycle_relative_tolerance),
        max_deformation_increment=Float64(max_deformation_increment),
        max_memory_deformation_increment=Float64(max_memory_deformation_increment),
        min_substeps=1,
        max_substeps=max_polymer_substeps,
    )
    selected_polymer_substeps = if polymer_substeps === :auto
        subcycle_estimate.recommended
    elseif polymer_substeps isa Integer
        polymer_substeps >= 1 || throw(ArgumentError("polymer_substeps must be >= 1"))
        polymer_substeps
    else
        throw(ArgumentError("polymer_substeps must be an integer or :auto"))
    end
    dt_poly = one(T) / T(selected_polymer_substeps)
    drag_enabled = !isnothing(drag_cx) && !isnothing(drag_cy) &&
                   !isnothing(drag_radius) && !isnothing(drag_u_ref)
    avg_window_i = isnothing(avg_window) ? max(1, max_steps) : Int(avg_window)
    avg_window_i >= 1 || throw(ArgumentError("avg_window must be >= 1"))
    drag_start = max_steps - min(max_steps, avg_window_i)
    Fx_s_sum = 0.0
    Fy_s_sum = 0.0
    Fx_p_sum = 0.0
    Fy_p_sum = 0.0
    Fx_bsd_sum = 0.0
    Fy_bsd_sum = 0.0
    n_drag = 0
    check_n = isnothing(check_every) ?
        max(1, round(Int, Nx / Float64(u_mean))) : Int(check_every)
    check_n >= 1 || throw(ArgumentError("check_every must be >= 1"))
    stop === :cd_change && !drag_enabled &&
        throw(ArgumentError("stop = :cd_change requires drag_cx, drag_cy, drag_radius and drag_u_ref"))
    cd_history = Tuple{Int,Float64}[]
    nan_force = (Fx=NaN, Fy=NaN)
    last_drag_s = nan_force; last_drag_p = nan_force; last_drag_bsd = nan_force
    stop_reason = :max_steps
    completed_steps = 0
    first_nonfinite_step = 0
    first_nonfinite_field = :none
    first_nonfinite_i = 0
    first_nonfinite_j = 0
    logfv_bc = fvfd_openx_wally_bcspec_2d()
    fvfd_geometry = FVFDGeometry2D(
        is_solid, embedded, FVFDPatch2D(dx, dy), logfv_bc,
    )
    ux_face_bc = FVFDFieldBC2D(u_profile, ux_east, uy_south, uy_north)
    uy_face_bc = FVFDFieldBC2D(u_profile, ux_east, uy_south, uy_north)
    wall_gradient_sides = WallGradientSides(uy_south, uy_north, nothing, nothing)

    logfv_add_constant_force_fluid_2d!(fx_total, fy_total, is_solid, Fx_body_t, zero(T); sync=false)
    logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f_in, fx_total, fy_total; sync=false)

    for step in 1:max_steps
        completed_steps = step
        logfv_copy_column_profile_2d!(ux_east, ux, Nx; sync=false)
        logfv_copy_column_profile_2d!(east_xx, psixx, Nx; sync=false)
        logfv_copy_column_profile_2d!(east_xy, psixy, Nx; sync=false)
        logfv_copy_column_profile_2d!(east_yy, psiyy, Nx; sync=false)
        if embedded_advection
            logfv_cell_velocity_to_faces_embedded_2d!(
                ux_face, uy_face, ux, uy, fvfd_geometry, ux_face_bc, uy_face_bc;
                sync=false,
            )
        else
            logfv_cell_velocity_to_faces_bc_aware_2d!(
                ux_face, uy_face, ux, uy, is_solid,
                u_profile, ux_east, uy_south, uy_north, logfv_bc;
                sync=false,
            )
        end
        logfv_advect_upwind_bc_aware_2d!(
            psixx_adv, psixy_adv, psiyy_adv,
            psixx, psixy, psiyy,
            west_xx, west_xy, west_yy,
            east_xx, east_xy, east_yy,
            south_xx, south_xy, south_yy,
            north_xx, north_xy, north_yy,
            ux_face, uy_face, is_solid, dx, dy, logfv_bc, one(T);
            sync=false,
            advection_scheme=advection_scheme_symbol,
        )
        if embedded_gradient
            fvfd_velocity_gradient_embedded_2d!(
                dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, logfv_bc, embedded;
                sync=false,
            )
        else
            fvfd_velocity_gradient_2d!(
                dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, logfv_bc;
                sync=false,
            )
        end
        # M51 helper REVERTED here (M53d audit): step is shared by cylinder,
        # square_channel, bfs → applying wall-position gradient broke
        # M5e/M7d/M8h. M48 cylinder R-sweep showed the helper was anyway
        # not flattening the U-shape (net ~zero impact), so removing.
        # Cavity keeps its own correction at cavity_driver_2d.jl:221.

        psixx_work, psixy_work, psiyy_work = psixx_adv, psixy_adv, psiyy_adv
        for _ in 1:selected_polymer_substeps
            logfv_step_constitutive_log_2d!(
                psixx_next, psixy_next, psiyy_next,
                psixx_work, psixy_work, psiyy_work,
                dudx, dudy, dvdx, dvdy,
                lambda_t, dt_poly, model_code, L2_t;
                sync=false,
            )
            psixx_work, psixx_next = psixx_next, psixx_work
            psixy_work, psixy_next = psixy_next, psixy_work
            psiyy_work, psiyy_next = psiyy_next, psiyy_work
        end
        psixx, psixx_adv = psixx_work, psixx
        psixy, psixy_adv = psixy_work, psixy
        psiyy, psiyy_adv = psiyy_work, psiyy

        logfv_stress_from_log_2d!(
            tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor_t;
            model_code, L2=L2_t, sync=false,
        )
        if embedded_force
            logfv_polymer_force_embedded_bc_aware_2d!(
                fx_poly, fy_poly, tauxx, tauxy, tauyy, fvfd_geometry;
                sync=false,
            )
            # Option-A rescale: the embedded kernel returns force per
            # fluid volume; the LBM Guo source consumes force per lattice cell.
            fvfd_scale_by_cell_fraction_2d!(
                fx_poly, fy_poly, embedded, is_solid; sync=false,
            )
        else
            logfv_polymer_force_bc_aware_2d!(
                fx_poly, fy_poly, tauxx, tauxy, tauyy, is_solid, dx, dy, logfv_bc;
                sync=false,
            )
        end
        logfv_bsd_correct_force_bc_aware_2d!(
            fx_total, fy_total, fx_poly, fy_poly, ux, uy, is_solid, bsd_t, nu_p_t, dx, dy,
            logfv_bc;
            sync=false,
        )
        if force_boundary_fill === :nearest
            logfv_fill_nearest_boundary_2d!(fx_total, fy_total; sync=false)
        end
        logfv_add_constant_force_fluid_2d!(fx_total, fy_total, is_solid, Fx_body_t, zero(T); sync=false)

        fused_trt_libb_v2_guo_field_step!(
            f_out, f_in, rho, ux, uy, is_solid, q_wall, uwx, uwy, fx_total, fy_total,
            Nx, Ny, nu_lbm_t; wall_bc=wall_bc,
        )
        apply_bc_rebuild_2d!(f_out, f_in, bcspec, nu_lbm_t, Nx, Ny)
        in_window = drag_enabled && step > drag_start &&
                    ((step - drag_start - 1) % drag_stride == 0 || step == max_steps)
        is_check = stop === :cd_change && step % check_n == 0
        if in_window || is_check
            drag_s = compute_drag_libb_mei_2d(f_out, q_wall, uwx, uwy, Nx, Ny)
            drag_p = if embedded_drag
                logfv_embedded_wall_traction_2d!(
                    drag_tx, drag_ty, tauxx, tauxy, tauyy, fvfd_geometry; sync=true,
                )
                (Fx=Float64(sum(Array(drag_tx))), Fy=Float64(sum(Array(drag_ty))))
            else
                compute_polymeric_drag_2d(
                    tauxx, tauxy, tauyy, q_wall, Nx, Ny;
                    cx=Float64(drag_cx),
                    cy=Float64(drag_cy),
                    radius=Float64(drag_radius),
                    extrapolate=true,
                    reconstruction_order=2,
                )
            end
            drag_bsd = if embedded_drag
                logfv_bsd_stress_from_gradient_2d!(
                    tau_bsd_xx, tau_bsd_xy, tau_bsd_yy,
                    dudx, dudy, dvdx, dvdy, bsd_t * nu_p_t; sync=false,
                )
                logfv_embedded_wall_traction_2d!(
                    drag_tx, drag_ty, tau_bsd_xx, tau_bsd_xy, tau_bsd_yy,
                    fvfd_geometry; sync=true,
                )
                (Fx=Float64(sum(Array(drag_tx))), Fy=Float64(sum(Array(drag_ty))))
            else
                _logfv_compute_bsd_drag_2d(
                    dudx, dudy, dvdx, dvdy, q_wall, Nx, Ny;
                    cx=Float64(drag_cx),
                    cy=Float64(drag_cy),
                    radius=Float64(drag_radius),
                    zeta_nu_p=Float64(bsd_t * nu_p_t),
                    reconstruction_order=2,
                )
            end
            if in_window
                Fx_s_sum += drag_s.Fx
                Fy_s_sum += drag_s.Fy
                Fx_p_sum += drag_p.Fx
                Fy_p_sum += drag_p.Fy
                Fx_bsd_sum += drag_bsd.Fx
                Fy_bsd_sum += drag_bsd.Fy
                n_drag += 1
            end
            if is_check
                cd_sample = 2.0 * (drag_s.Fx + drag_p.Fx - drag_bsd.Fx) /
                            (Float64(drag_u_ref)^2 * 2.0 * Float64(drag_radius))
                push!(cd_history, (step, cd_sample))
                last_drag_s = drag_s; last_drag_p = drag_p; last_drag_bsd = drag_bsd
            end
        end
        logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f_out, fx_total, fy_total; sync=false)
        if diagnostic_stride > 0 &&
           (step == 1 || step % diagnostic_stride == 0 || step == max_steps)
            KernelAbstractions.synchronize(backend)
            finite_diag = _logfv_first_nonfinite_field_2d(
                is_solid_h,
                :rho => rho,
                :ux => ux,
                :uy => uy,
                :psixx => psixx,
                :psixy => psixy,
                :psiyy => psiyy,
                :tauxx => tauxx,
                :tauxy => tauxy,
                :tauyy => tauyy,
                :fx_poly => fx_poly,
                :fy_poly => fy_poly,
                :fx_total => fx_total,
                :fy_total => fy_total,
            )
            if !finite_diag.finite
                first_nonfinite_step = step
                first_nonfinite_field = finite_diag.field
                first_nonfinite_i = finite_diag.i
                first_nonfinite_j = finite_diag.j
                break
            end
        end
        if step_callback !== nothing
            KernelAbstractions.synchronize(backend)
            if step_callback(step, (; rho, ux, uy,
                psixx, psixy, psiyy, tauxx, tauxy, tauyy,
                fx_poly, fy_poly, fx_total, fy_total,
                dudx, dudy, dvdx, dvdy, f_out, q_wall, uwx, uwy, is_solid_h)) === true
                stop_reason = :callback
            end
        end
        if stop_reason === :max_steps && is_check && length(cd_history) >= 2
            cd_new = cd_history[end][2]; cd_old = cd_history[end - 1][2]
            if isfinite(cd_new) && abs(cd_new - cd_old) <= tol * abs(cd_new)
                stop_reason = :rule
            end
        end
        f_in, f_out = f_out, f_in
        stop_reason === :max_steps || break
    end
    logfv_stress_from_log_2d!(
        tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor_t;
        model_code, L2=L2_t,
    )
    KernelAbstractions.synchronize(backend)

    rho_cpu = Array(rho)
    ux_cpu = Array(ux)
    uy_cpu = Array(uy)
    psixx_cpu = Array(psixx)
    psixy_cpu = Array(psixy)
    psiyy_cpu = Array(psiyy)
    tauxx_cpu = Array(tauxx)
    tauxy_cpu = Array(tauxy)
    tauyy_cpu = Array(tauyy)
    fx_poly_cpu = Array(fx_poly)
    fy_poly_cpu = Array(fy_poly)
    fx_total_cpu = Array(fx_total)
    fy_total_cpu = Array(fy_total)

    conf_diag = _logfv_conformation_diagnostics_2d(
        psixx_cpu, psixy_cpu, psiyy_cpu, is_solid_h, model_code, L2_t,
    )
    min_c_eig = conf_diag.min_c_eig
    max_c_trace = conf_diag.max_c_trace
    min_fene_denom = conf_diag.min_fene_denom
    max_fene_factor = conf_diag.max_fene_factor
    max_speed = 0.0
    for j in 1:Ny, i in 1:Nx
        if !is_solid_h[i, j]
            max_speed = max(max_speed, hypot(Float64(ux_cpu[i, j]), Float64(uy_cpu[i, j])))
        end
    end
    fluid_mask = .!is_solid_h
    max_abs_psi = max(maximum(abs, psixx_cpu), maximum(abs, psixy_cpu), maximum(abs, psiyy_cpu))
    max_abs_tau = max(maximum(abs, tauxx_cpu), maximum(abs, tauxy_cpu), maximum(abs, tauyy_cpu))
    max_abs_poly_force = max(maximum(abs, fx_poly_cpu), maximum(abs, fy_poly_cpu))
    max_abs_total_force = max(maximum(abs, fx_total_cpu), maximum(abs, fy_total_cpu))
    n_force = n_drag > 0 || !isempty(cd_history)
    Fx_s = n_drag > 0 ? Fx_s_sum / n_drag : last_drag_s.Fx
    Fy_s = n_drag > 0 ? Fy_s_sum / n_drag : last_drag_s.Fy
    Fx_p = n_drag > 0 ? Fx_p_sum / n_drag : last_drag_p.Fx
    Fy_p = n_drag > 0 ? Fy_p_sum / n_drag : last_drag_p.Fy
    Fx_bsd = n_drag > 0 ? Fx_bsd_sum / n_drag : last_drag_bsd.Fx
    Fy_bsd = n_drag > 0 ? Fy_bsd_sum / n_drag : last_drag_bsd.Fy
    Fx_drag = n_force ? Fx_s + Fx_p - Fx_bsd : NaN
    Fy_drag = n_force ? Fy_s + Fy_p - Fy_bsd : NaN
    drag_diameter = isnothing(drag_radius) ? NaN : 2.0 * Float64(drag_radius)
    drag_speed = isnothing(drag_u_ref) ? NaN : Float64(drag_u_ref)
    Cd_s = n_force ? 2.0 * Fx_s / (drag_speed^2 * drag_diameter) : NaN
    Cd_p = n_force ? 2.0 * Fx_p / (drag_speed^2 * drag_diameter) : NaN
    Cd_bsd = n_force ? 2.0 * Fx_bsd / (drag_speed^2 * drag_diameter) : NaN
    Cd = n_force ? Cd_s + Cd_p - Cd_bsd : NaN
    fluid_cell_fractions = embedded_h.cell_fraction[.!is_solid_h]
    embedded_min_fluid_cell_fraction = isempty(fluid_cell_fractions) ?
        NaN : Float64(minimum(fluid_cell_fractions))
    embedded_zero_fluid_cell_fraction_count = count(
        <=(sqrt(eps(T))), fluid_cell_fractions,
    )
    embedded_circle_normal_alignment = if embedded_geometry_symbol === :circle
        _logfv_embedded_circle_normal_alignment_2d(
            embedded_h, Float64(embedded_circle_cx), Float64(embedded_circle_cy),
        )
    else
        (min=NaN, mean=NaN, samples=0)
    end

    return (;
        geometry=geom_h,
        Nx,
        Ny,
        nu_s=Float64(nu_s_t),
        nu_p=Float64(nu_p_t),
        nu_total=Float64(nu_total_t),
        nu_lbm=Float64(nu_lbm_t),
        lambda=Float64(lambda_t),
        polymer_model=model_cfg.polymer_model,
        L_max=model_cfg.L_max,
        u_mean=Float64(u_mean),
        Fx_body=Float64(Fx_body_t),
        bsd_fraction=Float64(bsd_t),
        max_steps,
        completed_steps,
        polymer_substeps=selected_polymer_substeps,
        requested_polymer_substeps=polymer_substeps,
        diagnostic_stride,
        embedded_gradient,
        embedded_advection,
        embedded_force,
        embedded_drag,
        embedded_geometry=embedded_geometry_symbol,
        embedded_circle_samples,
        embedded_cut_count=count(embedded_h.cut_count .> 0),
        embedded_wall_length=Float64(sum(embedded_h.wall_fraction)),
        embedded_min_fluid_cell_fraction,
        embedded_zero_fluid_cell_fraction_count,
        embedded_normal_radial_min=embedded_circle_normal_alignment.min,
        embedded_normal_radial_mean=embedded_circle_normal_alignment.mean,
        embedded_normal_radial_samples=embedded_circle_normal_alignment.samples,
        force_boundary_fill,
        advection_scheme=advection_scheme_symbol,
        first_nonfinite_step,
        first_nonfinite_field,
        first_nonfinite_i,
        first_nonfinite_j,
        subcycle_estimate,
        max_grad_norm_estimate=Float64(max_grad_norm_estimate),
        rho=rho_cpu,
        ux=ux_cpu,
        uy=uy_cpu,
        psixx=psixx_cpu,
        psixy=psixy_cpu,
        psiyy=psiyy_cpu,
        tauxx=tauxx_cpu,
        tauxy=tauxy_cpu,
        tauyy=tauyy_cpu,
        fx_poly=fx_poly_cpu,
        fy_poly=fy_poly_cpu,
        fx_total=fx_total_cpu,
        fy_total=fy_total_cpu,
        is_solid=is_solid_h,
        min_c_eig,
        max_c_trace,
        min_fene_denom,
        max_fene_factor,
        max_speed,
        max_abs_psi,
        max_abs_tau,
        max_abs_poly_force,
        max_abs_total_force,
        Fx_s,
        Fy_s,
        Fx_p,
        Fy_p,
        Fx_bsd,
        Fy_bsd,
        Fx_drag,
        Fy_drag,
        Cd_s,
        Cd_p,
        Cd_bsd,
        Cd,
        n_drag_samples=n_drag,
        converged=stop_reason !== :max_steps,
        stop_reason,
        stop_step=completed_steps,
        cd_history,
        Cd_last=isempty(cd_history) ? NaN : cd_history[end][2],
        rho_min=minimum(rho_cpu[fluid_mask]),
        rho_max=maximum(rho_cpu[fluid_mask]),
    )
end

