# =============================================================================
# KrakeNN LI-BB + trt_rates fix validation on Aqua (CUDA / H100).
#
# Runs the canonical benchmarks with a single CUDA backend:
#   - Schäfer-Turek 2D-1 (parabolic Re=20) grid-refinement at D=40, 80, 160.
#   - Schäfer-Turek 2D-2 (parabolic Re=100 unsteady) at D=40.
#   - 3D sphere Re=20 uniform at R=12.
#   - MLUPS summary to compare against Metal (see project memory).
#
# Invoked from `hpc/validate_libb_cuda.pbs` on Aqua. Prints a
# rST-friendly results block.
# =============================================================================

using Pkg; Pkg.activate(dirname(@__DIR__))
using KrakeNN
using CUDA, KernelAbstractions
using FFTW

@assert CUDA.functional()
backend = CUDA.CUDABackend()
T = Float64     # H100 likes Float64 — no FP32 advantage at this throughput

function _logln(args...)
    println(args...); flush(stdout)
end

_logln("=" ^ 78)
_logln("Aqua CUDA validation: LI-BB V2 + trt_rates fix")
_logln("GPU: $(CUDA.name(CUDA.device()))")
_logln("Julia: $VERSION,  KrakeNN: $(pathof(KrakeNN))")
_logln("=" ^ 78)

# -----------------------------------------------------------------------------
# ST 2D-1 grid-refinement (parabolic Re=20)
# -----------------------------------------------------------------------------
_logln("\n## Schäfer-Turek 2D-1 (parabolic Re=20, ref Cd = 5.57..5.59)\n")
_logln("| D_lu | Nx × Ny     | Cd      | ΔCd/Cd_ref | t (s) | MLUPS |")
_logln("|------|-------------|---------|------------|-------|-------|")

Cd_ref = 5.58
for D_lu in (40, 80)
    Ny = Int(round(0.41 / 0.1 * D_lu))
    Nx = Int(round(2.2 / 0.1 * D_lu))
    cx = 2 * D_lu; cy = 2 * D_lu
    radius = D_lu ÷ 2
    u_max = 0.06
    u_mean = (2/3) * u_max
    ν = u_mean * D_lu / 20.0

    max_steps = D_lu == 40 ? 60_000 : 120_000
    avg_window = max_steps ÷ 6

    t0 = time()
    res = run_cylinder_libb_2d(; Nx=Nx, Ny=Ny, radius=radius,
                                 cx=cx, cy=cy, u_in=u_max, ν=ν,
                                 inlet=:parabolic, ρ_out=1.0,
                                 max_steps=max_steps, avg_window=avg_window,
                                 backend=backend, T=T)
    CUDA.synchronize()
    dt = time() - t0
    mlups = max_steps * Nx * Ny / dt / 1e6
    rel_err = (res.Cd - Cd_ref) / Cd_ref * 100
    _logln("| $(lpad(D_lu,4)) | $(lpad("$(Nx)×$(Ny)",11)) | $(lpad(round(res.Cd, digits=3),7)) | $(lpad(round(rel_err, digits=2),9)) % | $(lpad(round(dt, digits=1),5)) | $(lpad(round(mlups, digits=1),5)) |")
end

# -----------------------------------------------------------------------------
# ST 2D-2 unsteady (parabolic Re=100)
# -----------------------------------------------------------------------------
_logln("\n## Schäfer-Turek 2D-2 (unsteady Re=100, ref Cd_max=3.22-3.24,")
_logln("                       Cl_max=0.98-1.02, St=0.295-0.305)\n")

let
    D_lu = 40
    Ny = Int(round(0.41 / 0.1 * D_lu))
    Nx = Int(round(2.2 / 0.1 * D_lu))
    cx = 2 * D_lu; cy = 2 * D_lu
    radius = D_lu ÷ 2
    u_max = 0.06; u_mean = (2/3) * u_max
    Re = 100.0
    ν = u_mean * D_lu / Re
    St_ref = 0.30
    T_period = D_lu / (St_ref * u_mean)
    transient_periods = 5
    measure_periods   = 20
    max_steps = Int(round((transient_periods + measure_periods) * T_period))
    measure_start = Int(round(transient_periods * T_period))

    q_wall_h, is_solid_h = KrakeNN.precompute_q_wall_cylinder(Nx, Ny, cx, cy, radius; FT=T)
    u_prof_h = [T(4)*T(u_max)*T(j-1)*T(Ny-j)/T(Ny-1)^2 for j in 1:Ny]

    q_wall   = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    uw_x     = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    uw_y     = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_in     = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_out    = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    ρ  = KernelAbstractions.allocate(backend, T, Nx, Ny)
    ux = KernelAbstractions.allocate(backend, T, Nx, Ny)
    uy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    u_prof = KernelAbstractions.allocate(backend, T, Ny)
    copyto!(q_wall, q_wall_h); copyto!(is_solid, is_solid_h)
    fill!(uw_x, zero(T)); fill!(uw_y, zero(T))
    copyto!(u_prof, u_prof_h)
    f_in_h = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_in_h[i,j,q] = KrakeNN.equilibrium(KrakeNN.D2Q9(), one(T),
                                            u_prof_h[j], zero(T), q)
    end
    copyto!(f_in, f_in_h); fill!(f_out, zero(T))
    fill!(ρ, one(T)); fill!(ux, zero(T)); fill!(uy, zero(T))

    bcspec = BCSpec2D(; west = ZouHeVelocity(u_prof),
                        east = ZouHePressure(T(1.0)))

    Cd_t = Float64[]; Cl_t = Float64[]
    D_phys = 2 * radius; u_ref = u_mean
    t0 = time()
    let f_in = f_in, f_out = f_out
        for step in 1:max_steps
            KrakeNN.fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                             q_wall, uw_x, uw_y, Nx, Ny, T(ν))
            apply_bc_rebuild_2d!(f_out, f_in, bcspec, ν, Nx, Ny)
            if step >= measure_start
                drag = KrakeNN.compute_drag_libb_mei_2d(f_out, q_wall, uw_x, uw_y, Nx, Ny)
                push!(Cd_t, 2*drag.Fx/(u_ref^2 * D_phys))
                push!(Cl_t, 2*drag.Fy/(u_ref^2 * D_phys))
            end
            f_in, f_out = f_out, f_in
        end
        CUDA.synchronize()
    end
    dt = time() - t0
    mlups = max_steps * Nx * Ny / dt / 1e6

    Cd_mean = sum(Cd_t) / length(Cd_t)
    Cd_max  = maximum(Cd_t)
    Cl_max  = maximum(abs.(Cl_t))
    N = length(Cl_t)
    F = abs.(fft(Cl_t .- sum(Cl_t)/N))
    f_peak_idx = argmax(F[2:N÷2]) + 1
    f_peak_lu  = (f_peak_idx - 1) / N
    St = f_peak_lu * D_phys / u_mean

    _logln("  D_lu=$D_lu  steps=$max_steps  dt=$(round(dt, digits=1)) s  MLUPS=$(round(mlups, digits=1))")
    _logln("  Cd_mean = $(round(Cd_mean, digits=3))")
    _logln("  Cd_max  = $(round(Cd_max,  digits=3))  (ref 3.22-3.24)")
    _logln("  Cl_max  = $(round(Cl_max,  digits=3))  (ref 0.98-1.02)")
    _logln("  St      = $(round(St,      digits=3))  (ref 0.295-0.305)")
end

# -----------------------------------------------------------------------------
# 3D sphere Re=20 uniform
# -----------------------------------------------------------------------------
_logln("\n## 3D sphere Re=20 uniform (ref Cd ≈ 2.6 free-stream,")
_logln("                             +confinement for ducted)\n")
for radius in (12,)
    Ny = 6 * radius; Nz = 6 * radius; Nx = 3 * Ny
    D = 2 * radius
    u_in = 0.04
    ν = u_in * D / 20.0

    t0 = time()
    res = run_sphere_libb_3d(; Nx=Nx, Ny=Ny, Nz=Nz, radius=radius,
                              cx=Nx÷4, cy=Ny÷2, cz=Nz÷2,
                              u_in=u_in, ν=ν,
                              max_steps=30_000, avg_window=5_000,
                              backend=backend, T=T)
    CUDA.synchronize()
    dt = time() - t0
    mlups = 30_000 * Nx * Ny * Nz / dt / 1e6
    _logln("  R=$radius  Nx×Ny×Nz=$(Nx)×$(Ny)×$(Nz)  Cd=$(round(res.Cd, digits=3))  dt=$(round(dt, digits=1)) s  MLUPS=$(round(mlups, digits=1))")
end

_logln("\n=== DONE ===")
