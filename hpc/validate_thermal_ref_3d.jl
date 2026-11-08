using KrakeNN
using CUDA
using KernelAbstractions
using Printf

if !CUDA.functional()
    error("CUDA is not functional — check GPU allocation and drivers")
end
println("GPU: ", CUDA.name(CUDA.device()))
println("VRAM: ", round(CUDA.totalmem(CUDA.device()) / 1e9, digits=1), " GB")

const NU_REF_3D = 1.085  # Fusegi Ra=1e3

function run_validation(; N, backend, FT=Float32)
    println("\n", "="^60)
    @printf("N=%d, %s, %d steps\n", N, string(FT), max(5000, round(Int, 12*N^2)))
    println("="^60)

    ν = FT(0.05); Pr = FT(0.71); α = ν / Pr
    ω_f = FT(1.0 / (3ν + 0.5)); ω_T = FT(1.0 / (3α + 0.5))
    dx = FT(1.0); L = FT(N)
    T_hot = FT(1.0); T_cold = FT(0.0); ΔT = T_hot - T_cold
    T_ref = FT(0.5); Ra = FT(1e3)
    β_g = Ra * ν * α / (ΔT * L^3)
    Nx, Ny, Nz = N, N, N
    w = weights(D3Q19())
    nsteps = max(5000, round(Int, 12 * N^2))
    wall_frac = 0.25

    # --- E.1 Conduction + full-domain wall patch ---
    println("\nE.1 Pure conduction + west wall patch (full-domain)")
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
        T_init = FT(T_hot - ΔT*(i-1)/(Nx-1))
        for q in 1:19; g_cpu[i,j,k,q] = FT(w[q]) * T_init; end
    end
    copyto!(g_in, g_cpu); copyto!(g_out, g_cpu)

    # Full-domain patch (no margin) — stencil_clamped removed in 21ae88b
    patch = create_patch_3d("west", 1, 2,
        (0.0, 0.0, 0.0, Float64(wall_frac*L), Float64(L), Float64(L)),
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
    bc_c = (f, g, Te, nx, ny, nz) -> begin
        apply_fixed_temp_west_3d!(g, T_hot, ny, nz)
        apply_fixed_temp_east_3d!(g, T_cold, nx, ny, nz)
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
    T_exact = [FT(T_hot - ΔT*(i-1)/(Nx-1)) for i in 1:Nx]
    max_err = maximum(abs.(T_line .- T_exact))
    @printf("  %d steps in %.1fs (%.0f steps/s)\n", nsteps, dt, nsteps/dt)
    @printf("  T error: %.4e, |ux|=%.2e\n", max_err, maximum(abs, Array(ux)))
    @printf("  → %s\n", max_err < 0.02 && !any(isnan, T_cpu) ? "PASSED" : "FAILED")

    # --- E.2 NatConv + west/east wall patches ---
    println("\nE.2 Natural convection + wall patches (Ra=1e3)")

    state2 = initialize_3d(config, FT; backend=backend)
    f2i, f2o = state2.f_in, state2.f_out
    ρ2, ux2, uy2, uz2 = state2.ρ, state2.ux, state2.uy, state2.uz
    is2 = state2.is_solid

    g2i = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    g2o = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
    Tm2 = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)

    g_cpu2 = zeros(FT, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        T_init = FT(T_hot - ΔT*(i-1)/(Nx-1))
        for q in 1:19; g_cpu2[i,j,k,q] = FT(w[q]) * T_init; end
    end
    copyto!(g2i, g_cpu2); copyto!(g2o, g_cpu2)

    # Full-domain wall patches (no margin) — stencil_clamped removed in 21ae88b
    pw = create_patch_3d("west", 1, 2,
        (0.0, 0.0, 0.0, Float64(wall_frac*L), Float64(L), Float64(L)),
        Nx, Ny, Nz, Float64(dx), Float64(ω_f), FT; backend=backend)
    pe = create_patch_3d("east", 1, 2,
        (Float64((1-wall_frac)*L), 0.0, 0.0, Float64(L), Float64(L), Float64(L)),
        Nx, Ny, Nz, Float64(dx), Float64(ω_f), FT; backend=backend)
    dom2 = create_refined_domain_3d(Nx, Ny, Nz, Float64(dx), Float64(ω_f), [pw, pe])

    thw = create_thermal_patch_arrays_3d(pw, Float64(ω_T); T_init=Float64(T_ref), backend=backend)
    the = create_thermal_patch_arrays_3d(pe, Float64(ω_T); T_init=Float64(T_ref), backend=backend)
    ths2 = ThermalPatchArrays3D{FT}[thw, the]

    bct2 = build_patch_thermal_bcs_3d(dom2.patches, Float64(L), Float64(L), Float64(L), Nx,
        Dict(:west => Float64(T_hot), :east => Float64(T_cold)))
    bcf2 = build_patch_flow_bcs_3d(dom2.patches, Float64(L), Float64(L), Float64(L), Nx;
        wall_faces=[:west, :east, :south, :north, :bottom, :top])

    compute_macroscopic_3d!(ρ2, ux2, uy2, uz2, f2i)
    compute_temperature_3d!(Tm2, g2i)
    for (pidx, p) in enumerate(dom2.patches)
        prolongate_f_rescaled_full_3d!(
            p.f_in, f2i, ρ2, ux2, uy2, uz2,
            p.ratio, p.Nx_inner, p.Ny_inner, p.Nz_inner, p.n_ghost,
            first(p.parent_i_range), first(p.parent_j_range), first(p.parent_k_range),
            Nx, Ny, Nz, Float64(ω_f), Float64(p.omega))
        copyto!(p.f_out, p.f_in)
        compute_macroscopic_3d!(p.rho, p.ux, p.uy, p.uz, p.f_in)
        fill_thermal_full_3d!(p, ths2[pidx], g2i, Nx, Ny, Nz)
    end

    fs2 = (fo, fi, go, gi, Te, nx, ny, nz) -> begin
        stream_3d!(fo, fi, nx, ny, nz); stream_3d!(go, gi, nx, ny, nz)
        apply_fixed_temp_west_3d!(go, T_hot, ny, nz)
        apply_fixed_temp_east_3d!(go, T_cold, nx, ny, nz)
        compute_temperature_3d!(Te, go); compute_macroscopic_3d!(ρ2, ux2, uy2, uz2, fo)
        collide_thermal_3d!(go, ux2, uy2, uz2, ω_T)
        collide_boussinesq_3d!(fo, Te, is2, ω_f, β_g, T_ref)
    end
    bcc2 = (f, g, Te, nx, ny, nz) -> begin
        apply_fixed_temp_west_3d!(g, T_hot, ny, nz)
        apply_fixed_temp_east_3d!(g, T_cold, nx, ny, nz)
    end

    t0 = time()
    stable = true
    for step in 1:nsteps
        f2i, f2o, g2i, g2o = advance_thermal_refined_step_3d!(
            dom2, ths2, f2i, f2o, g2i, g2o, ρ2, ux2, uy2, uz2, Tm2, is2;
            fused_step_fn=fs2, omega_T_coarse=Float64(ω_T),
            β_g=Float64(β_g), T_ref_buoy=Float64(T_ref),
            bc_thermal_patch_fns=bct2, bc_flow_patch_fns=bcf2, bc_coarse_fn=bcc2)

        if step % (nsteps ÷ 10) == 0 || step == 1
            Tc = Array(Tm2)
            ok = !any(isnan, Tc)
            @printf("  step %6d: |ux|=%.4f |uy|=%.4f T=[%.4f,%.4f] %s\n",
                    step, maximum(abs, Array(ux2)), maximum(abs, Array(uy2)),
                    minimum(Tc), maximum(Tc), ok ? "✓" : "NaN!")
            if !ok; stable = false; break; end
        end
    end
    dt = time() - t0
    @printf("  %d steps in %.1fs (%.0f steps/s)\n", nsteps, dt, nsteps/dt)

    if stable
        # Nu from fine west patch
        Tf = Array(ths2[1].Temp)
        ng = dom2.patches[1].n_ghost
        Nyf = dom2.patches[1].Ny_inner; Nzf = dom2.patches[1].Nz_inner
        dxf = FT(dom2.patches[1].dx)
        Nu_arr = zeros(FT, Nyf, Nzf)
        for kf in 1:Nzf, jf in 1:Nyf
            j = jf+ng; k = kf+ng
            dTdx = (-3*Tf[ng+1,j,k] + 4*Tf[ng+2,j,k] - Tf[ng+3,j,k]) / (2*dxf)
            Nu_arr[jf, kf] = -L * dTdx / ΔT
        end
        Nu_fine = sum(Nu_arr[2:end-1, 2:end-1]) / max((Nyf-2)*(Nzf-2), 1)

        # Uniform reference
        r_uni = run_natural_convection_3d(; N=N, Ra=1e3, Pr=0.71, max_steps=nsteps,
                                            backend=backend, FT=FT)

        @printf("\n  Nu comparison (Ra=1e3, N=%d):\n", N)
        @printf("    Fusegi ref:     %8.4f\n", Float64(NU_REF_3D))
        @printf("    Uniform:        %8.4f  (err %.2f%%)\n",
                Float64(r_uni.Nu), abs(r_uni.Nu-NU_REF_3D)/NU_REF_3D*100)
        @printf("    Refined (fine): %8.4f  (err %.2f%%)\n",
                Float64(Nu_fine), abs(Nu_fine-NU_REF_3D)/NU_REF_3D*100)
    end

    println(stable ? "\n  → STABLE" : "\n  → UNSTABLE (NaN)")
end

function run_krk_validation(; N, nsteps, backend)
    println("\n", "="^60)
    @printf("E.3 KRK dispatch: NatConv 3D refined, N=%d, %d steps\n", N, nsteps)
    println("="^60)
    L = Float64(N)
    wf = 0.25

    krk = """
        Simulation natconv_ref_3d D3Q19
        Domain L = $L x $L x $L  N = $N x $N x $N
        Physics nu = 0.05  Pr = 0.71  Ra = 1000
        Module thermal
        Boundary west  wall(T = 1.0)
        Boundary east  wall(T = 0.0)
        Boundary south wall
        Boundary north wall
        Boundary bottom wall
        Boundary top wall
        Refine west_wall { region = [0.0, 0.0, 0.0, $(wf*L), $L, $L], ratio = 2 }
        Refine east_wall { region = [$((1-wf)*L), 0.0, 0.0, $L, $L, $L], ratio = 2 }
        Run $nsteps steps
    """

    t0 = time()
    result = run_simulation(parse_kraken(krk); backend=backend, T=Float32)
    dt = time() - t0
    @printf("  %d steps in %.1fs\n", nsteps, dt)

    T_cpu = result.Temp
    ok = !any(isnan, T_cpu)
    @printf("  NaN: %s, max|u|=%.4f\n", ok ? "no" : "YES", maximum(sqrt.(result.ux.^2 .+ result.uy.^2)))

    # Coarse-grid Nu (rough estimate)
    H = Float32(N)
    Nu_arr = zeros(Float32, N, N)
    for k in 1:N, j in 1:N
        dTdx = (-3*T_cpu[1,j,k] + 4*T_cpu[2,j,k] - T_cpu[3,j,k]) / 2f0
        Nu_arr[j, k] = -H * dTdx
    end
    Nu = sum(Nu_arr[2:end-1, 2:end-1]) / max((N-2)*(N-2), 1)
    @printf("  Nu (coarse) = %.4f  ref=1.085 (Fusegi)\n", Nu)
    println(ok ? "  → PASSED" : "  → FAILED")
end

# Run for N=24 and N=32
backend = CUDABackend()
run_validation(; N=24, backend=backend)
run_validation(; N=32, backend=backend)
run_krk_validation(; N=32, nsteps=max(5000, 12*32^2), backend=backend)
