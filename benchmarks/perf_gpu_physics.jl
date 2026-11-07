# GPU performance benchmark for different physics modules
# Measures MLUPS for BGK, Guo, thermal, rheology, and VOF multiphase
using KrakeNN
using Printf
using KernelAbstractions

const HAS_CUDA = try
    @eval using CUDA
    CUDA.functional()
catch
    false
end

const HAS_METAL = try
    @eval using Metal
    Metal.functional()
catch
    false
end

function bench_bgk(; N=256, steps=200, backend=KernelAbstractions.CPU(), FT=Float64)
    config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=0.1, u_lid=0.05, max_steps=steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    omega_val = FT(omega(config))
    is_solid = state.is_solid

    # Warmup
    for _ in 1:5
        stream_2d!(f_out, f_in, N, N)
        collide_2d!(f_out, is_solid, omega_val)
        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    t = @elapsed begin
        for _ in 1:steps
            stream_2d!(f_out, f_in, N, N)
            collide_2d!(f_out, is_solid, omega_val)
            f_in, f_out = f_out, f_in
        end
        KernelAbstractions.synchronize(backend)
    end
    return N * N * steps / t / 1e6
end

function bench_guo(; N=256, steps=200, backend=KernelAbstractions.CPU(), FT=Float64)
    config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=0.1, u_lid=0.0, max_steps=steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    is_solid = state.is_solid
    Fx = KernelAbstractions.zeros(backend, FT, N, N)
    Fy = KernelAbstractions.zeros(backend, FT, N, N)
    fill!(Fx, FT(1e-5))
    omega_val = FT(omega(config))

    for _ in 1:5
        stream_periodic_x_wall_y_2d!(f_out, f_in, N, N)
        collide_guo_field_2d!(f_out, is_solid, omega_val, Fx, Fy)
        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    t = @elapsed begin
        for _ in 1:steps
            stream_periodic_x_wall_y_2d!(f_out, f_in, N, N)
            collide_guo_field_2d!(f_out, is_solid, omega_val, Fx, Fy)
            f_in, f_out = f_out, f_in
        end
        KernelAbstractions.synchronize(backend)
    end
    return N * N * steps / t / 1e6
end

function bench_thermal(; N=256, steps=200, backend=KernelAbstractions.CPU(), FT=Float64)
    nu = FT(0.05)
    alpha_t = nu  # Pr=1
    omega_f = FT(1.0 / (3.0 * nu + 0.5))
    omega_T = FT(1.0 / (3.0 * alpha_t + 0.5))
    beta_g = FT(1e-4)
    T_ref = FT(0.5)

    config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=Float64(nu), u_lid=0.0, max_steps=steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    rho, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    g_in  = KernelAbstractions.zeros(backend, FT, N, N, 9)
    g_out = KernelAbstractions.zeros(backend, FT, N, N, 9)
    Temp  = KernelAbstractions.zeros(backend, FT, N, N)

    w = KrakeNN.weights(D2Q9())
    g_cpu = zeros(FT, N, N, 9)
    for j in 1:N, i in 1:N, q in 1:9
        g_cpu[i, j, q] = FT(w[q]) * FT(0.5)
    end
    copyto!(g_in, g_cpu)
    copyto!(g_out, g_cpu)

    # Warmup
    for _ in 1:5
        stream_periodic_x_wall_y_2d!(f_out, f_in, N, N)
        stream_periodic_x_wall_y_2d!(g_out, g_in, N, N)
        compute_temperature_2d!(Temp, g_out)
        compute_macroscopic_2d!(rho, ux, uy, f_out)
        collide_thermal_2d!(g_out, ux, uy, omega_T)
        collide_boussinesq_2d!(f_out, Temp, is_solid, omega_f, beta_g, T_ref)
        f_in, f_out = f_out, f_in
        g_in, g_out = g_out, g_in
    end
    KernelAbstractions.synchronize(backend)

    t = @elapsed begin
        for _ in 1:steps
            stream_periodic_x_wall_y_2d!(f_out, f_in, N, N)
            stream_periodic_x_wall_y_2d!(g_out, g_in, N, N)
            compute_temperature_2d!(Temp, g_out)
            compute_macroscopic_2d!(rho, ux, uy, f_out)
            collide_thermal_2d!(g_out, ux, uy, omega_T)
            collide_boussinesq_2d!(f_out, Temp, is_solid, omega_f, beta_g, T_ref)
            f_in, f_out = f_out, f_in
            g_in, g_out = g_out, g_in
        end
        KernelAbstractions.synchronize(backend)
    end
    return N * N * steps / t / 1e6
end

function bench_rheology(; N=256, steps=200, backend=KernelAbstractions.CPU(), FT=Float64)
    config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=0.1, u_lid=0.0, max_steps=steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    is_solid = state.is_solid
    tau_field = KernelAbstractions.ones(backend, FT, N, N)
    Fx = KernelAbstractions.zeros(backend, FT, N, N)
    Fy = KernelAbstractions.zeros(backend, FT, N, N)
    fill!(Fx, FT(1e-5))
    fill!(tau_field, FT(0.8))

    rheology = PowerLaw(FT(0.1), FT(0.7); nu_min=FT(1e-5), nu_max=FT(5.0))

    for _ in 1:5
        stream_periodic_x_wall_y_2d!(f_out, f_in, N, N)
        collide_rheology_guo_2d!(f_out, is_solid, rheology, tau_field, Fx, Fy)
        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    t = @elapsed begin
        for _ in 1:steps
            stream_periodic_x_wall_y_2d!(f_out, f_in, N, N)
            collide_rheology_guo_2d!(f_out, is_solid, rheology, tau_field, Fx, Fy)
            f_in, f_out = f_out, f_in
        end
        KernelAbstractions.synchronize(backend)
    end
    return N * N * steps / t / 1e6
end

function bench_vof(; N=256, steps=200, backend=KernelAbstractions.CPU(), FT=Float64)
    config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=0.1, u_lid=0.0, max_steps=steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    rho, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    C     = KernelAbstractions.zeros(backend, FT, N, N)
    C_new = KernelAbstractions.zeros(backend, FT, N, N)
    nx_n  = KernelAbstractions.zeros(backend, FT, N, N)
    ny_n  = KernelAbstractions.zeros(backend, FT, N, N)
    kappa = KernelAbstractions.zeros(backend, FT, N, N)
    Fx_st = KernelAbstractions.zeros(backend, FT, N, N)
    Fy_st = KernelAbstractions.zeros(backend, FT, N, N)

    # Initialize droplet
    R = N ÷ 5
    cx, cy = N ÷ 2, N ÷ 2
    C_cpu = zeros(FT, N, N)
    w = KrakeNN.weights(D2Q9())
    f_cpu = zeros(FT, N, N, 9)
    for j in 1:N, i in 1:N
        r = sqrt(FT((i - cx)^2 + (j - cy)^2))
        C_cpu[i, j] = FT(0.5) * (one(FT) - tanh((r - FT(R)) / FT(2)))
        rho_init = C_cpu[i, j] * FT(1.0) + (1 - C_cpu[i, j]) * FT(0.001)
        for q in 1:9
            f_cpu[i, j, q] = FT(w[q]) * rho_init
        end
    end
    copyto!(C, C_cpu)
    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)

    sigma_val = FT(0.01)

    # Warmup
    for _ in 1:5
        stream_fully_periodic_2d!(f_out, f_in, N, N)
        compute_macroscopic_2d!(rho, ux, uy, f_out)
        advect_vof_step!(C, C_new, ux, uy, N, N)
        copyto!(C, C_new)
        compute_vof_normal_2d!(nx_n, ny_n, C, N, N)
        compute_hf_curvature_2d!(kappa, C, nx_n, ny_n, N, N)
        compute_surface_tension_2d!(Fx_st, Fy_st, kappa, C, sigma_val, N, N)
        collide_twophase_2d!(f_out, C, Fx_st, Fy_st, is_solid;
                             ρ_l=1.0, ρ_g=0.001, ν_l=0.1, ν_g=0.1)
        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    t = @elapsed begin
        for _ in 1:steps
            stream_fully_periodic_2d!(f_out, f_in, N, N)
            compute_macroscopic_2d!(rho, ux, uy, f_out)
            advect_vof_step!(C, C_new, ux, uy, N, N)
            copyto!(C, C_new)
            compute_vof_normal_2d!(nx_n, ny_n, C, N, N)
            compute_hf_curvature_2d!(kappa, C, nx_n, ny_n, N, N)
            compute_surface_tension_2d!(Fx_st, Fy_st, kappa, C, sigma_val, N, N)
            collide_twophase_2d!(f_out, C, Fx_st, Fy_st, is_solid;
                                 ρ_l=1.0, ρ_g=0.001, ν_l=0.1, ν_g=0.1)
            f_in, f_out = f_out, f_in
        end
        KernelAbstractions.synchronize(backend)
    end
    return N * N * steps / t / 1e6
end

function run_physics_benchmark(; gpu=false)
    N = 256
    steps = 200

    physics = [
        ("BGK",       bench_bgk),
        ("Guo force", bench_guo),
        ("Thermal",   bench_thermal),
        ("Rheology",  bench_rheology),
        ("VOF",       bench_vof),
    ]

    println("\n=== Physics Performance Comparison (N=$N, $steps steps) ===")

    # CPU
    println("\n--- CPU ---")
    @printf("  %-12s   %10s\n", "Physics", "MLUPS")
    @printf("  %-12s   %10s\n", "----------", "--------")

    cpu_mlups = Float64[]
    for (name, fn) in physics
        mlups = fn(; N=N, steps=steps)
        push!(cpu_mlups, mlups)
        @printf("  %-12s   %10.1f\n", name, mlups)
    end

    # GPU
    if gpu
        gpu_backend = nothing
        if HAS_CUDA
            gpu_backend = CUDABackend()
            println("\n--- CUDA GPU ---")
        elseif HAS_METAL
            gpu_backend = MetalBackend()
            println("\n--- Metal GPU ---")
        end

        if !isnothing(gpu_backend)
            @printf("  %-12s   %10s   %8s\n", "Physics", "MLUPS", "Speedup")
            @printf("  %-12s   %10s   %8s\n", "----------", "--------", "-------")

            for (idx, (name, fn)) in enumerate(physics)
                mlups_gpu = fn(; N=N, steps=steps, backend=gpu_backend)
                speedup = cpu_mlups[idx] > 0 ? mlups_gpu / cpu_mlups[idx] : NaN
                @printf("  %-12s   %10.1f   %7.1fx\n", name, mlups_gpu, speedup)
            end
        else
            println("\nNo GPU backend available. Skipping GPU benchmark.")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    gpu = "--gpu" in ARGS
    run_physics_benchmark(; gpu=gpu)
end
