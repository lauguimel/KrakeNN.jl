#!/usr/bin/env julia
# ===========================================================================
# Benchmark: PLIC (MYC / ELVIRA) vs MUSCL-Superbee VOF advection
#
# Compares schemes on circle rotation and reversed vortex,
# reports metrics alongside Basilisk reference values.
# ===========================================================================

using Kraken
using KernelAbstractions
using Printf

const backend = KernelAbstractions.CPU()
const FT = Float64

# ===================================================================
# Geometric VOF init: exact circle fraction per cell (sub-sampling)
# ===================================================================

function init_circle_fraction!(C, cx, cy, R, dx)
    Nx, Ny = size(C)
    nsub = 8  # 8×8 sub-sampling per cell
    C_cpu = zeros(FT, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        count = 0
        for sj in 1:nsub, si in 1:nsub
            x = (i - 1 + (si - 0.5) / nsub) * dx
            y = (j - 1 + (sj - 0.5) / nsub) * dx
            if (x - cx)^2 + (y - cy)^2 <= R^2
                count += 1
            end
        end
        C_cpu[i,j] = FT(count) / FT(nsub^2)
    end
    copyto!(C, C_cpu)
end

# ===================================================================
# Helper: run PLIC advection with selectable reconstruction
# ===================================================================

function run_plic_advection(; Nx, Ny, max_steps, velocity_fn,
                              init_C_fn=nothing, init_circle=nothing,
                              dt=1.0, recon::Symbol=:myc)
    dx = FT(1)
    C      = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_new  = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    nx_n   = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ny_n   = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    cc_f   = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ux     = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    uy     = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    if init_circle !== nothing
        init_circle_fraction!(C, init_circle.cx, init_circle.cy, init_circle.R, dx)
    else
        init_vof_field!(C, init_C_fn, dx, FT)
    end
    C0 = Array(C)

    is_time_dep = try
        velocity_fn(FT(0.5)*dx, FT(0.5)*dx, FT(0))
        velocity_fn(FT(0.5)*dx, FT(0.5)*dx, FT(1))
        true
    catch; false; end

    fill_velocity_field!(ux, uy, velocity_fn, dx, FT(0), backend, FT)

    for step in 1:max_steps
        t = FT(step - 1) * dt
        if is_time_dep && step > 1
            fill_velocity_field!(ux, uy, velocity_fn, dx, t, backend, FT)
        end
        advect_vof_plic_step!(C, C_new, nx_n, ny_n, cc_f, ux, uy, Nx, Ny;
                               step=step, recon=recon)
        copyto!(C, C_new)
    end

    return (C=Array(C), C0=C0)
end

# ===================================================================
# Helper: run MUSCL advection
# ===================================================================

function run_muscl_advection(; Nx, Ny, max_steps, velocity_fn, init_C_fn, dt=1.0)
    result = run_advection_2d(; Nx=Nx, Ny=Ny, max_steps=max_steps, dt=dt,
                                velocity_fn=velocity_fn, init_C_fn=init_C_fn)
    return (C=result.C, C0=result.C0)
end

# ===================================================================
# Metrics
# ===================================================================

function compute_metrics(C, C0)
    err = abs.(C .- C0)
    L1  = sum(err) / length(C)
    Linf = maximum(err)
    mass0 = sum(C0)
    mass  = sum(C)
    mass_err = abs(mass - mass0) / mass0
    cmin = minimum(C)
    cmax = maximum(C)
    return (; L1, Linf, mass_err, cmin, cmax)
end

# ===================================================================
# Test 1: Circle rotation (1 full revolution)
# ===================================================================

function bench_circle_rotation(N)
    R  = N * 0.234
    cx, cy = N / 2.0, N / 2.0
    ω  = 2π / (4N)

    velocity_fn(x, y, t) = (-(y - cy) * ω, (x - cx) * ω)
    init_fn(x, y) = begin
        r = sqrt((x - cx)^2 + (y - cy)^2)
        r <= R ? 1.0 : 0.0
    end
    circ = (cx=cx, cy=cy, R=R)

    nsteps = 4N

    t_muscl = @elapsed r_muscl = run_muscl_advection(; Nx=N, Ny=N, max_steps=nsteps,
                                                       velocity_fn=velocity_fn, init_C_fn=init_fn)
    t_myc   = @elapsed r_myc   = run_plic_advection(; Nx=N, Ny=N, max_steps=nsteps,
                                                      velocity_fn=velocity_fn, init_circle=circ,
                                                      recon=:myc)
    t_elv   = @elapsed r_elv   = run_plic_advection(; Nx=N, Ny=N, max_steps=nsteps,
                                                      velocity_fn=velocity_fn, init_circle=circ,
                                                      recon=:elvira)

    m_muscl = compute_metrics(r_muscl.C, r_muscl.C0)
    m_myc   = compute_metrics(r_myc.C,  r_myc.C0)
    m_elv   = compute_metrics(r_elv.C,  r_elv.C0)

    return (; N, nsteps, m_muscl, m_myc, m_elv, t_muscl, t_myc, t_elv)
end

# ===================================================================
# Test 2: Reversed vortex (LeVeque-style time reversal)
# ===================================================================

function bench_reversed_vortex(N)
    R  = N * 0.2
    cx, cy = N / 2.0, 0.75 * N
    T_period = 8.0

    function velocity_fn(x, y, t)
        xn = x / N; yn = y / N
        scale = cos(π * t / T_period) * 0.5
        vx = -sin(π * xn) * cos(π * yn) * scale
        vy =  cos(π * xn) * sin(π * yn) * scale
        return (vx, vy)
    end

    init_fn(x, y) = begin
        r = sqrt((x - cx)^2 + (y - cy)^2)
        r <= R ? 1.0 : 0.0
    end
    circ = (cx=cx, cy=cy, R=R)

    nsteps = round(Int, T_period)

    t_muscl = @elapsed r_muscl = run_muscl_advection(; Nx=N, Ny=N, max_steps=nsteps,
                                                       velocity_fn=velocity_fn, init_C_fn=init_fn)
    t_myc   = @elapsed r_myc   = run_plic_advection(; Nx=N, Ny=N, max_steps=nsteps,
                                                      velocity_fn=velocity_fn, init_circle=circ,
                                                      recon=:myc)
    t_elv   = @elapsed r_elv   = run_plic_advection(; Nx=N, Ny=N, max_steps=nsteps,
                                                      velocity_fn=velocity_fn, init_circle=circ,
                                                      recon=:elvira)

    m_muscl = compute_metrics(r_muscl.C, r_muscl.C0)
    m_myc   = compute_metrics(r_myc.C,  r_myc.C0)
    m_elv   = compute_metrics(r_elv.C,  r_elv.C0)

    return (; N, nsteps, m_muscl, m_myc, m_elv, t_muscl, t_myc, t_elv)
end

# ===================================================================
# Run benchmarks
# ===================================================================

function print_row(label, N, nsteps, m, t)
    @printf("  %-10s  %4d  %5d  %.2e   %.2e   %.2e   [%.4f, %.4f]  %5.2fs\n",
            label, N, nsteps, m.L1, m.Linf, m.mass_err, m.cmin, m.cmax, t)
end

function main()
    println("=" ^ 100)
    println("  VOF Advection Benchmark: MUSCL-Superbee vs PLIC-MYC vs PLIC-ELVIRA")
    println("  (MYC + CFL sub-stepping + ELVIRA second-order reconstruction)")
    println("=" ^ 100)

    # --- Circle rotation ---
    println("\n▶ Circle rotation (1 full revolution, sharp disk init)")
    println("  " * "-"^96)
    @printf("  %-10s  %4s  %5s  %-9s   %-9s   %-9s   %-17s  %s\n",
            "Scheme", "N", "Steps", "L1", "L∞", "Δm/m", "C range", "Time")
    println("  " * "-"^96)

    for N in [32, 64, 128]
        r = bench_circle_rotation(N)
        print_row("MUSCL", r.N, r.nsteps, r.m_muscl, r.t_muscl)
        print_row("PLIC-MYC", r.N, r.nsteps, r.m_myc, r.t_myc)
        print_row("PLIC-ELVIRA", r.N, r.nsteps, r.m_elv, r.t_elv)
        println("  " * "-"^96)
    end

    println("  Basilisk ref (rotate.c, 1/4 rotation ×4 ≈ full rotation estimate):")
    println("    N=32:  L1≈8.4e-3   L∞≈3.0e-1  Δm=0  C∈[0,1]  (×4 extrapolation)")
    println("    N=64:  L1≈2.9e-3   L∞≈2.0e-1  Δm=0  C∈[0,1]")
    println("    N=128: L1≈1.5e-3   L∞≈1.8e-1  Δm=0  C∈[0,1]")

    # --- Reversed vortex ---
    println("\n▶ Reversed vortex (time reversal, sharp disk init)")
    println("  " * "-"^96)
    @printf("  %-10s  %4s  %5s  %-9s   %-9s   %-9s   %-17s  %s\n",
            "Scheme", "N", "Steps", "L1", "L∞", "Δm/m", "C range", "Time")
    println("  " * "-"^96)

    for N in [32, 64, 128]
        r = bench_reversed_vortex(N)
        print_row("MUSCL", r.N, r.nsteps, r.m_muscl, r.t_muscl)
        print_row("PLIC-MYC", r.N, r.nsteps, r.m_myc, r.t_myc)
        print_row("PLIC-ELVIRA", r.N, r.nsteps, r.m_elv, r.t_elv)
        println("  " * "-"^96)
    end

    println("  Basilisk ref (reversed.c, full cycle T=15):")
    println("    N=32:  L1=5.77e-2  L∞=1.00     Δm=0  C∈[0,1]")
    println("    N=64:  L1=1.10e-2  L∞=1.00     Δm=0  C∈[0,1]")
    println("    N=128: L1=1.45e-3  L∞=5.01e-1  Δm=0  C∈[0,1]")

    println("\n" * "=" ^ 100)
end

main()
