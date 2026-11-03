using Kraken
using Metal
using KernelAbstractions
using Printf

const NU_REF_3D = 1.085  # Fusegi Ra=1e3

function validate_thermal_ref_3d_gpu(; N=16, backend=MetalBackend(), FT=Float32)
    println("=== 3D Thermal Refinement — GPU validation (N=$N, $FT) ===\n")

    ν = FT(0.05); Pr = FT(0.71)
    α = ν / Pr
    ω_f = FT(1.0 / (3ν + 0.5)); ω_T = FT(1.0 / (3α + 0.5))
    dx = FT(1.0); L = FT(N)
    T_hot = FT(1.0); T_cold = FT(0.0); ΔT = T_hot - T_cold
    T_ref = FT(0.5); Ra = FT(1e3)
    β_g = Ra * ν * α / (ΔT * L^3)
    Nx, Ny, Nz = N, N, N
    w = weights(D3Q19())
    nsteps = max(5000, round(Int, 12 * N^2))

    # --- E.1 Conduction + wall patch (β_g=0) ---
    println("E.1 Pure conduction + west wall patch (β_g=0)")
    config = LBMConfig(D3Q19(); Nx=N, Ny=N, Nz=N, ν=Float64(ν), u_lid=0.0, max_steps=nsteps)
    state = initialize_3d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy, uz = state.ρ, state.ux, state.uy, state.uz
    is_solid = state.is_solid

    g_in  = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    g_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    Temp  = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)

    g_cpu = zeros(FT, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        T_init = FT(T_hot - ΔT * (i - 0.5) / Nx)   # half-way walls at x = 0.5, Nx + 0.5
        for q in 1:19; g_cpu[i,j,k,q] = FT(w[q]) * T_init; end
    end
    copyto!(g_in, g_cpu); copyto!(g_out, g_cpu)

    wall_frac = 0.25
    patch = create_patch_3d("west", 1, 2,
        (0.0, 0.0, 0.0, Float64(wall_frac * L), Float64(L), Float64(L)),
        Nx, Ny, Nz, Float64(dx), Float64(ω_f), FT; backend=backend)
    domain = create_refined_domain_3d(Nx, Ny, Nz, Float64(dx), Float64(ω_f), [patch])

    th = create_thermal_patch_arrays_3d(patch, Float64(ω_T); T_init=Float64(T_ref), backend=backend)
    thermals = ThermalPatchArrays3D{FT}[th]

    bc_th = build_patch_thermal_bcs_3d(domain.patches, Float64(L), Float64(L), Float64(L), Nx,
        Dict(:west => Float64(T_hot), :east => Float64(T_cold)))
    bc_flow = build_patch_flow_bcs_3d(domain.patches, Float64(L), Float64(L), Float64(L), Nx;
        wall_faces=[:west, :east, :south, :north, :bottom, :top])

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

    fused = (fo, fi, go, gi, Te, nx, ny, nz) -> begin
        stream_3d!(fo, fi, nx, ny, nz); stream_3d!(go, gi, nx, ny, nz)
        apply_fixed_temp_west_3d!(go, T_hot, ny, nz)
        apply_fixed_temp_east_3d!(go, T_cold, nx, ny, nz)
        compute_temperature_3d!(Te, go); compute_macroscopic_3d!(ρ, ux, uy, uz, fo)
        collide_thermal_3d!(go, ux, uy, uz, ω_T)
        collide_3d!(fo, is_solid, ω_f)
    end
    # Post-restriction thermal call on the patched (west) face only: the
    # half-way kernel flips the incoming slots, so the unpatched east wall,
    # already set post-stream in `fused`, must not get a second call.
    bc_c = (f, g, Te, nx, ny, nz) -> begin
        apply_fixed_temp_west_3d!(g, T_hot, ny, nz)
    end

    t0 = time()
    for step in 1:nsteps
        f_in, f_out, g_in, g_out = advance_thermal_refined_step_3d!(
            domain, thermals, f_in, f_out, g_in, g_out, ρ, ux, uy, uz, Temp, is_solid;
            fused_step_fn=fused, omega_T_coarse=Float64(ω_T),
            β_g=0.0, T_ref_buoy=Float64(T_ref),
            bc_thermal_patch_fns=bc_th, bc_flow_patch_fns=bc_flow, bc_coarse_fn=bc_c)
    end
    dt = time() - t0

    compute_temperature_3d!(Temp, g_in)
    T_cpu = Array(Temp)
    jm = N÷2; km = N÷2
    T_line = T_cpu[:, jm, km]
    T_exact = [FT(T_hot - ΔT * (i - 0.5) / Nx) for i in 1:Nx]
    max_err = maximum(abs.(T_line .- T_exact))
    ux_max = maximum(abs, Array(ux))
    @printf("  %d steps in %.1fs (%.0f steps/s)\n", nsteps, dt, nsteps/dt)
    @printf("  T error: %.4e, |ux|=%.2e\n", max_err, ux_max)
    @printf("  T range: [%.4f, %.4f]\n", minimum(T_cpu), maximum(T_cpu))
    @printf("  → %s\n\n", max_err < 0.02 && !any(isnan, T_cpu) ? "PASSED" : "FAILED")

    # --- E.2 NatConv + west/east wall patches ---
    println("E.2 Natural convection + wall patches (Ra=1e3)")

    state2 = initialize_3d(config, FT; backend=backend)
    f_in2, f_out2 = state2.f_in, state2.f_out
    ρ2, ux2, uy2, uz2 = state2.ρ, state2.ux, state2.uy, state2.uz
    is_solid2 = state2.is_solid

    g_in2  = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    g_out2 = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    Temp2  = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)

    g_cpu2 = zeros(FT, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        T_init = FT(T_hot - ΔT * (i - 0.5) / Nx)   # half-way walls at x = 0.5, Nx + 0.5
        for q in 1:19; g_cpu2[i,j,k,q] = FT(w[q]) * T_init; end
    end
    copyto!(g_in2, g_cpu2); copyto!(g_out2, g_cpu2)

    p_w = create_patch_3d("west", 1, 2,
        (0.0, 0.0, 0.0, Float64(wall_frac*L), Float64(L), Float64(L)),
        Nx, Ny, Nz, Float64(dx), Float64(ω_f), FT; backend=backend)
    p_e = create_patch_3d("east", 1, 2,
        (Float64((1-wall_frac)*L), 0.0, 0.0, Float64(L), Float64(L), Float64(L)),
        Nx, Ny, Nz, Float64(dx), Float64(ω_f), FT; backend=backend)
    domain2 = create_refined_domain_3d(Nx, Ny, Nz, Float64(dx), Float64(ω_f), [p_w, p_e])

    th_w = create_thermal_patch_arrays_3d(p_w, Float64(ω_T); T_init=Float64(T_ref), backend=backend)
    th_e = create_thermal_patch_arrays_3d(p_e, Float64(ω_T); T_init=Float64(T_ref), backend=backend)
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

    fused2 = (fo, fi, go, gi, Te, nx, ny, nz) -> begin
        stream_3d!(fo, fi, nx, ny, nz); stream_3d!(go, gi, nx, ny, nz)
        apply_fixed_temp_west_3d!(go, T_hot, ny, nz)
        apply_fixed_temp_east_3d!(go, T_cold, nx, ny, nz)
        compute_temperature_3d!(Te, go); compute_macroscopic_3d!(ρ2, ux2, uy2, uz2, fo)
        collide_thermal_3d!(go, ux2, uy2, uz2, ω_T)
        collide_boussinesq_3d!(fo, Te, is_solid2, ω_f, β_g, T_ref)
    end
    # Both walls carry a patch here, so both get the post-restriction call.
    bc_c2 = (f, g, Te, nx, ny, nz) -> begin
        apply_fixed_temp_west_3d!(g, T_hot, ny, nz)
        apply_fixed_temp_east_3d!(g, T_cold, nx, ny, nz)
    end

    t0 = time()
    for step in 1:nsteps
        f_in2, f_out2, g_in2, g_out2 = advance_thermal_refined_step_3d!(
            domain2, thermals2, f_in2, f_out2, g_in2, g_out2,
            ρ2, ux2, uy2, uz2, Temp2, is_solid2;
            fused_step_fn=fused2, omega_T_coarse=Float64(ω_T),
            β_g=Float64(β_g), T_ref_buoy=Float64(T_ref),
            bc_thermal_patch_fns=bc_th2, bc_flow_patch_fns=bc_flow2, bc_coarse_fn=bc_c2)

        if step % (nsteps ÷ 10) == 0 || step == 1
            T_cpu2 = Array(Temp2); ux_cpu = Array(ux2); uy_cpu = Array(uy2)
            ok = !any(isnan, T_cpu2)
            @printf("  step %5d: |ux|=%.4f |uy|=%.4f T=[%.4f,%.4f] %s\n",
                    step, maximum(abs, ux_cpu), maximum(abs, uy_cpu),
                    minimum(T_cpu2), maximum(T_cpu2), ok ? "✓" : "NaN!")
            if !ok
                return
            end
        end
    end
    dt = time() - t0

    compute_temperature_3d!(Temp2, g_in2)
    compute_macroscopic_3d!(ρ2, ux2, uy2, uz2, f_in2)

    # Nu from fine west patch
    T_fine = Array(thermals2[1].Temp)
    ng = domain2.patches[1].n_ghost
    Ny_f = domain2.patches[1].Ny_inner
    Nz_f = domain2.patches[1].Nz_inner
    dx_f = FT(domain2.patches[1].dx)
    Nu_arr = zeros(FT, Ny_f, Nz_f)
    for kf in 1:Nz_f, jf in 1:Ny_f
        j = jf + ng; k = kf + ng
        i1, i2, i3 = ng+1, ng+2, ng+3
        dTdx = (-3*T_fine[i1,j,k] + 4*T_fine[i2,j,k] - T_fine[i3,j,k]) / (2*dx_f)
        Nu_arr[jf, kf] = -L * dTdx / ΔT
    end
    Nu_fine = sum(Nu_arr[2:end-1, 2:end-1]) / max((Ny_f-2)*(Nz_f-2), 1)

    # Nu from coarse grid: second-order one-sided gradient through the
    # half-way wall value and the first two nodes.
    T_c = Array(Temp2)
    Nu_c_arr = zeros(FT, Ny, Nz)
    for k in 1:Nz, j in 1:Ny
        dTdx = (-8*T_hot + 9*T_c[1,j,k] - T_c[2,j,k]) / FT(3)
        Nu_c_arr[j, k] = -L * dTdx / ΔT
    end
    Nu_coarse = sum(Nu_c_arr[2:end-1, 2:end-1]) / max((Ny-2)*(Nz-2), 1)

    # Reference: uniform Nu at same N
    r_uni = run_natural_convection_3d(; N=N, Ra=1e3, Pr=0.71, max_steps=nsteps,
                                        backend=backend, FT=FT)

    @printf("  %d steps in %.1fs (%.0f steps/s)\n", nsteps, dt, nsteps/dt)
    @printf("  T range: [%.4f, %.4f]\n", minimum(T_c), maximum(T_c))
    @printf("  |ux|=%.4f, |uy|=%.4f\n", maximum(abs, Array(ux2)), maximum(abs, Array(uy2)))
    @printf("\n  Nu comparison (Ra=1e3, N=%d):\n", N)
    @printf("    Fusegi ref:     %8.4f\n", Float64(NU_REF_3D))
    @printf("    Uniform:        %8.4f  (err %.2f%%)\n", Float64(r_uni.Nu), abs(r_uni.Nu-NU_REF_3D)/NU_REF_3D*100)
    @printf("    Refined coarse: %8.4f  (err %.2f%%)\n", Float64(Nu_coarse), abs(Nu_coarse-NU_REF_3D)/NU_REF_3D*100)
    @printf("    Refined fine:   %8.4f  (err %.2f%%)\n", Float64(Nu_fine), abs(Nu_fine-NU_REF_3D)/NU_REF_3D*100)
    @printf("\n  → STABLE (%d steps)\n", nsteps)
end

validate_thermal_ref_3d_gpu(; N=16)
