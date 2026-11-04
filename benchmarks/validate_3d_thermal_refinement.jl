# ==========================================================================
# Validation suite: 3D thermal + generic thermal refinement + 3D refinement
#
# Part A — Natural convection 2D: uniform vs refined vs generic-refined (.krk)
#           Reference: De Vahl Davis (1983), Ra=1e3 → Nu=1.118
# Part B — Natural convection 3D: uniform convergence
#           Reference: Fusegi et al. (1991), Ra=1e3 → Nu=1.085
# Part C — Cylinder 2D: uniform Cd vs refined (stability + flow quality)
#           Reference: Schäfer & Turek (1996), Re=20, Cd=5.58
# Part D — 3D refinement: lid-driven cavity with center patch (stability)
# ==========================================================================

using Kraken
using Printf

const NU_REF_2D = 1.118     # De Vahl Davis Ra=1e3
const NU_REF_3D = 1.085     # Fusegi Ra=1e3
const CD_REF    = 5.58       # Schäfer & Turek Re=20

nsteps_natconv(N) = max(5000, round(Int, 8 * N^2))
nsteps_cylinder(Ny) = max(10000, round(Int, 12 * Ny^2))

header(title) = begin
    println("\n" * "="^70)
    println(title)
    println("="^70)
end

# ==========================================================================
# Part A — Natural Convection 2D
# ==========================================================================

function validate_natconv_2d(; Ns=[32, 64, 128])
    header("PART A — NATURAL CONVECTION 2D  (Ra=1e3, Pr=0.71, Nu_ref=$NU_REF_2D)")

    # A.1 Global refinement (uniform grids)
    println("\n  A.1 Uniform (global refinement)")
    @printf("  %5s  %8s  %8s  %7s  %5s\n", "N", "steps", "Nu", "err%", "order")
    errs = Float64[]
    for N in Ns
        r = run_natural_convection_2d(; N=N, Ra=1e3, Pr=0.71,
                                        max_steps=nsteps_natconv(N))
        e = abs(r.Nu - NU_REF_2D) / NU_REF_2D
        push!(errs, e)
        ord = length(errs) > 1 ? log2(errs[end-1] / errs[end]) : NaN
        @printf("  %5d  %8d  %8.4f  %6.2f%%  %5s\n",
                N, nsteps_natconv(N), r.Nu, e*100,
                isnan(ord) ? "  -" : @sprintf("%.2f", ord))
    end

    # A.2 Local refinement (dedicated driver, wall patches)
    println("\n  A.2 Refined — dedicated driver (wall_fraction=0.2, ratio=2)")
    @printf("  %5s  %8s  %8s  %7s  %5s\n", "N_base", "steps", "Nu", "err%", "order")
    errs_r = Float64[]
    for N in Ns[1:min(2, end)]  # 32, 64 only (refined is more expensive)
        r = run_natural_convection_refined_2d(; N=N, Ra=1e3, Pr=0.71,
                max_steps=nsteps_natconv(N), ratio=2, wall_fraction=0.2)
        e = abs(r.Nu - NU_REF_2D) / NU_REF_2D
        push!(errs_r, e)
        ord = length(errs_r) > 1 ? log2(errs_r[end-1] / errs_r[end]) : NaN
        @printf("  %5d  %8d  %8.4f  %6.2f%%  %5s\n",
                N, nsteps_natconv(N), r.Nu, e*100,
                isnan(ord) ? "  -" : @sprintf("%.2f", ord))
    end

    # A.3 Generic thermal refined (.krk path)
    println("\n  A.3 Refined — generic .krk path (Module thermal + Refine)")
    @printf("  %5s  %8s  %8s  %7s  %12s\n", "N_base", "steps", "Nu", "err%", "T_range")
    for N in Ns[1:min(2, end)]
        steps = nsteps_natconv(N)
        setup = Kraken.parse_kraken("""
        Simulation natconv_generic D2Q9
        Module thermal
        Domain L = 1.0 x 1.0  N = $N x $N
        Setup rayleigh = 1e3, prandtl = 0.71
        Boundary west  wall T = 1.0
        Boundary east  wall T = 0.0
        Boundary south wall
        Boundary north wall
        Refine west_wall { region = [0.0, 0.0, 0.2, 1.0], ratio = 2, parent = base }
        Refine east_wall { region = [0.8, 0.0, 1.0, 1.0], ratio = 2, parent = base }
        Run $steps steps
        """)
        r = run_simulation(setup)
        e = abs(r.Nu - NU_REF_2D) / NU_REF_2D
        Tmin, Tmax = extrema(r.Temp)
        @printf("  %5d  %8d  %8.4f  %6.2f%%  [%.3f,%.3f]\n",
                N, steps, r.Nu, e*100, Tmin, Tmax)
    end
end

# ==========================================================================
# Part B — Natural Convection 3D
# ==========================================================================

function validate_natconv_3d(; Ns=[16, 24])
    header("PART B — NATURAL CONVECTION 3D  (Ra=1e3, Pr=0.71, Nu_ref=$NU_REF_3D)")

    println("\n  B.1 Uniform (global refinement)")
    @printf("  %5s  %8s  %8s  %7s  %5s\n", "N", "steps", "Nu", "err%", "order")
    errs = Float64[]
    for N in Ns
        steps = max(5000, round(Int, 12 * N^2))
        r = run_natural_convection_3d(; N=N, Ra=1e3, Pr=0.71, max_steps=steps)
        e = abs(r.Nu - NU_REF_3D) / NU_REF_3D
        push!(errs, e)
        ord = length(errs) > 1 ? log2(errs[end-1] / errs[end]) : NaN
        @printf("  %5d  %8d  %8.4f  %6.2f%%  %5s\n",
                N, steps, r.Nu, e*100,
                isnan(ord) ? "  -" : @sprintf("%.2f", ord))
    end
end

# ==========================================================================
# Part C — Cylinder in channel 2D
# ==========================================================================

function validate_cylinder_2d(; Nys=[40, 80], u_in=0.05)
    header("PART C — CYLINDER IN CHANNEL 2D  (Re=20, D/H≈0.25, Cd_ref=$CD_REF)")

    # C.1 Uniform — quantitative Cd
    println("\n  C.1 Uniform (MEA drag)")
    @printf("  %5s  %5s  %8s  %8s  %7s\n", "Ny", "Nx", "steps", "Cd", "err%")
    for Ny in Nys
        Nx = 4 * Ny
        R = round(Int, 0.125 * Ny)
        nu = u_in * 2R / 20.0
        steps = nsteps_cylinder(Ny)
        r = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R, u_in=u_in, ν=nu,
                              max_steps=steps, avg_window=min(2000, steps÷5))
        e = abs(r.Cd - CD_REF) / CD_REF * 100
        @printf("  %5d  %5d  %8d  %8.4f  %6.2f%%\n", Ny, Nx, steps, r.Cd, e)
    end

    # C.2 Refined via .krk (stability + flow quality)
    println("\n  C.2 Refined via .krk (ratio=2 around cylinder)")
    @printf("  %5s  %5s  %8s  %8s  %8s  %12s\n",
            "Ny", "Nx", "steps", "|ux|max", "|uy|max", "ρ_range")
    for Ny in Nys
        Nx = 4 * Ny
        R = round(Int, 0.125 * Ny)
        nu = u_in * 2R / 20.0
        steps = nsteps_cylinder(Ny)
        Lx = Float64(Nx); Ly = Float64(Ny)
        R_phys = Float64(R); cx = 0.2 * Lx; cy = 0.5 * Ly
        margin = 3.0 * R_phys

        setup = Kraken.parse_kraken("""
        Simulation cyl_refined D2Q9
        Domain L = $Lx x $Ly  N = $Nx x $Ny
        Define R = $R_phys
        Define cx = $cx
        Define cy = $cy
        Physics nu = $nu
        Obstacle cylinder { (x - cx)^2 + (y - cy)^2 <= R^2 }
        Boundary west velocity(ux = $u_in, uy = 0)
        Boundary east pressure(rho = 1.0)
        Boundary south wall
        Boundary north wall
        Refine cyl_zone { region = [$(max(0,cx-margin)), $(max(0,cy-margin)), $(min(Lx,cx+2margin)), $(min(Ly,cy+margin))], ratio = 2 }
        Run $steps steps
        """)
        r = run_simulation(setup)
        @printf("  %5d  %5d  %8d  %8.4f  %8.4f  [%.4f,%.4f]\n",
                Ny, Nx, steps,
                maximum(abs, r.ux), maximum(abs, r.uy),
                minimum(r.ρ), maximum(r.ρ))
    end
end

# ==========================================================================
# Part D — 3D Refinement: lid-driven cavity stability
# ==========================================================================

function validate_refinement_3d(; N=16, max_steps=500)
    header("PART D — 3D REFINEMENT  (lid-driven cavity N=$N, ratio=2)")

    ν = 0.1; ω = 1.0 / (3.0 * ν + 0.5); dx = 1.0

    config = LBMConfig(D3Q19(); Nx=N, Ny=N, Nz=N, ν=ν, u_lid=0.1, max_steps=max_steps)
    state = initialize_3d(config, Float64)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy, uz = state.ρ, state.ux, state.uy, state.uz
    is_solid = state.is_solid

    # Center patch (away from walls — no patch BC needed)
    patch = create_patch_3d("center", 1, 2,
        (4.0, 4.0, 4.0, 12.0, 12.0, 12.0), N, N, N, dx, ω, Float64)
    domain = create_refined_domain_3d(N, N, N, dx, ω, [patch])

    compute_macroscopic_3d!(ρ, ux, uy, uz, f_in)
    prolongate_f_rescaled_full_3d!(
        patch.f_in, f_in, ρ, ux, uy, uz,
        patch.ratio, patch.Nx_inner, patch.Ny_inner, patch.Nz_inner,
        patch.n_ghost,
        first(patch.parent_i_range), first(patch.parent_j_range),
        first(patch.parent_k_range), N, N, N, ω, Float64(patch.omega))
    copyto!(patch.f_out, patch.f_in)
    compute_macroscopic_3d!(patch.rho, patch.ux, patch.uy, patch.uz, patch.f_in)

    @printf("\n  Patch: %d×%d×%d (inner %d×%d×%d), ω_fine=%.4f\n",
            patch.Nx, patch.Ny, patch.Nz,
            patch.Nx_inner, patch.Ny_inner, patch.Nz_inner, patch.omega)
    @printf("  %5s  %10s  %10s  %12s  %6s\n",
            "step", "ux_max", "uy_max", "ρ_range", "ok")

    for step in 1:max_steps
        f_in, f_out = advance_refined_step_3d!(domain, f_in, f_out, ρ, ux, uy, uz, is_solid;
            stream_fn=stream_3d!,
            collide_fn=(f, is_s) -> collide_3d!(f, is_s, ω),
            macro_fn=compute_macroscopic_3d!,
            bc_base_fn=(f) -> apply_zou_he_top_3d!(f, 0.1, N, N, N))
        apply_zou_he_top_3d!(f_in, 0.1, N, N, N)
        compute_macroscopic_3d!(ρ, ux, uy, uz, f_in)

        if step % (max_steps ÷ 5) == 0 || step == 1
            ρ_cpu = Array(ρ); ux_cpu = Array(ux); uy_cpu = Array(uy)
            ok = !any(isnan, ρ_cpu) && minimum(ρ_cpu) > 0.5
            @printf("  %5d  %10.6f  %10.6f  [%.4f,%.4f]  %6s\n",
                    step, maximum(abs, ux_cpu), maximum(abs, uy_cpu),
                    minimum(ρ_cpu), maximum(ρ_cpu), ok ? "✓" : "✗")
            if !ok
                println("  *** UNSTABLE — stopping early")
                return false
            end
        end
    end

    println("  → PASSED: $max_steps steps stable")
    return true
end

# ==========================================================================
# Part E — 3D Thermal Refinement: conduction + natural convection
# ==========================================================================

function validate_thermal_refinement_3d(; N=16, max_steps=3000)
    header("PART E — 3D THERMAL REFINEMENT  (wall patches, ratio=2)")

    FT = Float64
    ν = FT(0.05); Pr = FT(0.71)
    α_thermal = ν / Pr
    ω_f = FT(1.0 / (3.0 * ν + 0.5))
    ω_T = FT(1.0 / (3.0 * α_thermal + 0.5))
    dx = FT(1.0); L = FT(N)
    T_hot = FT(1.0); T_cold = FT(0.0); ΔT = T_hot - T_cold
    T_ref = FT(0.5); Ra = FT(1e3)
    β_g = Ra * ν * α_thermal / (ΔT * L^3)

    Nx, Ny, Nz = N, N, N
    config = LBMConfig(D3Q19(); Nx=Nx, Ny=Ny, Nz=Nz, ν=Float64(ν),
                       u_lid=0.0, max_steps=max_steps)

    # --- E.1 Pure conduction + west wall patch: T profile preserved ---
    println("\n  E.1 Pure conduction 3D + wall Refine (β_g=0, stability)")

    state = initialize_3d(config, FT)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy, uz = state.ρ, state.ux, state.uy, state.uz
    is_solid = state.is_solid

    w = weights(D3Q19())
    g_in  = zeros(FT, Nx, Ny, Nz, 19)
    g_out = zeros(FT, Nx, Ny, Nz, 19)
    Temp  = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        T_init = FT(T_hot - ΔT * (i - 0.5) / Nx)   # half-way walls at x = 0.5, Nx + 0.5
        for q in 1:19; g_in[i,j,k,q] = FT(w[q]) * T_init; end
    end; copyto!(g_out, g_in)

    # West wall patch
    wall_frac = 0.25
    patch = create_patch_3d("west", 1, 2,
        (0.0, 0.0, 0.0, Float64(wall_frac * L), Float64(L), Float64(L)),
        Nx, Ny, Nz, Float64(dx), Float64(ω_f), FT)
    domain = create_refined_domain_3d(Nx, Ny, Nz, Float64(dx), Float64(ω_f), [patch])

    th = create_thermal_patch_arrays_3d(patch, Float64(ω_T); T_init=Float64(T_ref))
    thermals = ThermalPatchArrays3D{FT}[th]

    bc_th = build_patch_thermal_bcs_3d(domain.patches, Float64(L), Float64(L), Float64(L), Nx,
        Dict(:west => Float64(T_hot), :east => Float64(T_cold)))
    bc_flow = build_patch_flow_bcs_3d(domain.patches, Float64(L), Float64(L), Float64(L), Nx;
        wall_faces=[:west, :east, :south, :north, :bottom, :top])

    # Init patch from coarse
    compute_macroscopic_3d!(ρ, ux, uy, uz, f_in)
    compute_temperature_3d!(Temp, g_in)
    prolongate_f_rescaled_full_3d!(
        patch.f_in, f_in, ρ, ux, uy, uz,
        patch.ratio, patch.Nx_inner, patch.Ny_inner, patch.Nz_inner, patch.n_ghost,
        first(patch.parent_i_range), first(patch.parent_j_range), first(patch.parent_k_range),
        Nx, Ny, Nz, Float64(ω_f), Float64(patch.omega))
    copyto!(patch.f_out, patch.f_in)
    compute_macroscopic_3d!(patch.rho, patch.ux, patch.uy, patch.uz, patch.f_in)
    fill_thermal_full_3d!(patch, th, g_in, Nx, Ny, Nz)

    fused_step = (fo, fi, go, gi, Te, nx, ny, nz) -> begin
        stream_3d!(fo, fi, nx, ny, nz)
        stream_3d!(go, gi, nx, ny, nz)
        apply_fixed_temp_west_3d!(go, T_hot, ny, nz)
        apply_fixed_temp_east_3d!(go, T_cold, nx, ny, nz)
        compute_temperature_3d!(Te, go)
        compute_macroscopic_3d!(ρ, ux, uy, uz, fo)
        collide_thermal_3d!(go, ux, uy, uz, ω_T)
        collide_3d!(fo, is_solid, ω_f)
    end

    # Post-restriction thermal call on the patched (west) face only: the
    # half-way kernel flips the incoming slots, so the unpatched east wall,
    # already set post-stream in the fused step, must not get a second call.
    bc_coarse = (f, g, Te, nx, ny, nz) -> begin
        apply_fixed_temp_west_3d!(g, T_hot, ny, nz)
    end

    for step in 1:max_steps
        f_in, f_out, g_in, g_out = advance_thermal_refined_step_3d!(
            domain, thermals, f_in, f_out, g_in, g_out,
            ρ, ux, uy, uz, Temp, is_solid;
            fused_step_fn=fused_step,
            omega_T_coarse=Float64(ω_T),
            β_g=0.0, T_ref_buoy=Float64(T_ref),
            bc_thermal_patch_fns=bc_th, bc_flow_patch_fns=bc_flow,
            bc_coarse_fn=bc_coarse)
    end

    compute_temperature_3d!(Temp, g_in)
    T_cpu = Array(Temp)
    jm = N ÷ 2; km = N ÷ 2
    T_line = T_cpu[:, jm, km]
    T_exact = [FT(T_hot - ΔT * (i - 0.5) / Nx) for i in 1:Nx]
    max_err = maximum(abs.(T_line .- T_exact))
    ux_max = maximum(abs, ux); uy_max = maximum(abs, uy)
    @printf("  T profile error: %.4e, |ux|=%.2e, |uy|=%.2e\n", max_err, ux_max, uy_max)
    @printf("  T range: [%.4f, %.4f]\n", minimum(T_cpu), maximum(T_cpu))
    pass_cond = !any(isnan, T_cpu) && ux_max < 1e-10
    @printf("  → %s (zero velocity, T bounded)\n", pass_cond ? "PASSED" : "FAILED")

    # --- E.2 NatConv 3D + wall patches: stability + Nu ---
    println("\n  E.2 Natural convection 3D + wall Refine (Ra=1e3)")

    state2 = initialize_3d(config, FT)
    f_in2, f_out2 = state2.f_in, state2.f_out
    ρ2, ux2, uy2, uz2 = state2.ρ, state2.ux, state2.uy, state2.uz
    is_solid2 = state2.is_solid

    g_in2  = zeros(FT, Nx, Ny, Nz, 19)
    g_out2 = zeros(FT, Nx, Ny, Nz, 19)
    Temp2  = zeros(FT, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        T_init = FT(T_hot - ΔT * (i - 0.5) / Nx)   # half-way walls at x = 0.5, Nx + 0.5
        for q in 1:19; g_in2[i,j,k,q] = FT(w[q]) * T_init; end
    end; copyto!(g_out2, g_in2)

    # West + east wall patches
    p_west = create_patch_3d("west", 1, 2,
        (0.0, 0.0, 0.0, Float64(wall_frac * L), Float64(L), Float64(L)),
        Nx, Ny, Nz, Float64(dx), Float64(ω_f), FT)
    p_east = create_patch_3d("east", 1, 2,
        (Float64((1 - wall_frac) * L), 0.0, 0.0, Float64(L), Float64(L), Float64(L)),
        Nx, Ny, Nz, Float64(dx), Float64(ω_f), FT)
    domain2 = create_refined_domain_3d(Nx, Ny, Nz, Float64(dx), Float64(ω_f), [p_west, p_east])

    th_w = create_thermal_patch_arrays_3d(p_west, Float64(ω_T); T_init=Float64(T_ref))
    th_e = create_thermal_patch_arrays_3d(p_east, Float64(ω_T); T_init=Float64(T_ref))
    thermals2 = ThermalPatchArrays3D{FT}[th_w, th_e]

    bc_th2 = build_patch_thermal_bcs_3d(domain2.patches, Float64(L), Float64(L), Float64(L), Nx,
        Dict(:west => Float64(T_hot), :east => Float64(T_cold)))
    bc_flow2 = build_patch_flow_bcs_3d(domain2.patches, Float64(L), Float64(L), Float64(L), Nx;
        wall_faces=[:west, :east, :south, :north, :bottom, :top])

    compute_macroscopic_3d!(ρ2, ux2, uy2, uz2, f_in2)
    compute_temperature_3d!(Temp2, g_in2)
    for (pidx, patch) in enumerate(domain2.patches)
        prolongate_f_rescaled_full_3d!(
            patch.f_in, f_in2, ρ2, ux2, uy2, uz2,
            patch.ratio, patch.Nx_inner, patch.Ny_inner, patch.Nz_inner, patch.n_ghost,
            first(patch.parent_i_range), first(patch.parent_j_range), first(patch.parent_k_range),
            Nx, Ny, Nz, Float64(ω_f), Float64(patch.omega))
        copyto!(patch.f_out, patch.f_in)
        compute_macroscopic_3d!(patch.rho, patch.ux, patch.uy, patch.uz, patch.f_in)
        fill_thermal_full_3d!(patch, thermals2[pidx], g_in2, Nx, Ny, Nz)
    end

    fused_step2 = (fo, fi, go, gi, Te, nx, ny, nz) -> begin
        stream_3d!(fo, fi, nx, ny, nz)
        stream_3d!(go, gi, nx, ny, nz)
        apply_fixed_temp_west_3d!(go, T_hot, ny, nz)
        apply_fixed_temp_east_3d!(go, T_cold, nx, ny, nz)
        compute_temperature_3d!(Te, go)
        compute_macroscopic_3d!(ρ2, ux2, uy2, uz2, fo)
        collide_thermal_3d!(go, ux2, uy2, uz2, ω_T)
        collide_boussinesq_3d!(fo, Te, is_solid2, ω_f, β_g, T_ref)
    end

    # Both walls carry a patch here, so both get the post-restriction call.
    bc_coarse2 = (f, g, Te, nx, ny, nz) -> begin
        apply_fixed_temp_west_3d!(g, T_hot, ny, nz)
        apply_fixed_temp_east_3d!(g, T_cold, nx, ny, nz)
    end

    @printf("  %6s  %10s  %10s  %12s  %6s\n",
            "step", "ux_max", "uy_max", "T_range", "ok")
    for step in 1:max_steps
        f_in2, f_out2, g_in2, g_out2 = advance_thermal_refined_step_3d!(
            domain2, thermals2, f_in2, f_out2, g_in2, g_out2,
            ρ2, ux2, uy2, uz2, Temp2, is_solid2;
            fused_step_fn=fused_step2,
            omega_T_coarse=Float64(ω_T),
            β_g=Float64(β_g), T_ref_buoy=Float64(T_ref),
            bc_thermal_patch_fns=bc_th2, bc_flow_patch_fns=bc_flow2,
            bc_coarse_fn=bc_coarse2)

        if step % (max_steps ÷ 5) == 0 || step == 1
            T_cpu2 = Array(Temp2); ux_cpu = Array(ux2); uy_cpu = Array(uy2)
            ok = !any(isnan, T_cpu2) && minimum(T_cpu2) >= -0.1 && maximum(T_cpu2) <= 1.1
            @printf("  %6d  %10.6f  %10.6f  [%.4f,%.4f]  %6s\n",
                    step, maximum(abs, ux_cpu), maximum(abs, uy_cpu),
                    minimum(T_cpu2), maximum(T_cpu2), ok ? "✓" : "✗")
            if !ok
                println("  *** UNSTABLE — stopping early")
                return false
            end
        end
    end

    # Nusselt from fine west patch (better wall gradient resolution)
    T_fine = Array(thermals2[1].Temp)
    ng = domain2.patches[1].n_ghost
    Ny_fine = domain2.patches[1].Ny_inner
    Nz_fine = domain2.patches[1].Nz_inner
    dx_f = FT(domain2.patches[1].dx)
    Nu_arr = zeros(FT, Ny_fine, Nz_fine)
    for kf in 1:Nz_fine, jf in 1:Ny_fine
        j = jf + ng; k = kf + ng
        i1, i2, i3 = ng + 1, ng + 2, ng + 3
        dTdx = (-3 * T_fine[i1,j,k] + 4 * T_fine[i2,j,k] - T_fine[i3,j,k]) / (2 * dx_f)
        Nu_arr[jf, kf] = -L * dTdx / ΔT
    end
    Nu = sum(Nu_arr[2:end-1, 2:end-1]) / max((Ny_fine-2)*(Nz_fine-2), 1)
    err = abs(Nu - NU_REF_3D) / NU_REF_3D * 100
    @printf("\n  Nu (fine patch):  %8.4f  (err %.2f%% vs Fusegi %.3f)\n", Nu, err, NU_REF_3D)
    @printf("  → %d steps stable\n", max_steps)
    return true
end

# ==========================================================================
# Main
# ==========================================================================

function run_validation(; quick=false)
    println("Kraken.jl — Validation: 3D thermal + refinement")
    println("CPU-only, Float64\n")

    if quick
        validate_natconv_2d(; Ns=[32, 64])
        validate_natconv_3d(; Ns=[16])
        validate_cylinder_2d(; Nys=[40])
        validate_refinement_3d(; N=16, max_steps=200)
        validate_thermal_refinement_3d(; N=12, max_steps=1500)
    else
        validate_natconv_2d(; Ns=[32, 64, 128])
        validate_natconv_3d(; Ns=[16, 24])
        validate_cylinder_2d(; Nys=[40, 80])
        validate_refinement_3d(; N=16, max_steps=500)
        validate_thermal_refinement_3d(; N=16, max_steps=3000)
    end

    println("\n" * "="^70)
    println("VALIDATION COMPLETE")
    println("="^70)
end

if abspath(PROGRAM_FILE) == @__FILE__
    quick = "--quick" in ARGS
    run_validation(; quick=quick)
end
