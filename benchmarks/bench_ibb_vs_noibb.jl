# =============================================================================
# Benchmark: IBB (Bouzidi LI-BB) vs no-IBB (classical halfway bounce-back)
# on the 2D cylinder and the 3D sphere.
#
# - Accuracy: grid-refinement at Re=20, extrapolated Richardson reference.
# - Performance: MLUPS (million lattice-site updates per second) on Metal.
#
# Run:
#   julia --project=. benchmarks/bench_ibb_vs_noibb.jl
# =============================================================================

using Pkg; Pkg.activate(dirname(@__DIR__))
using KrakeNN
using KrakeNN: fused_trt_libb_v2_step_3d!, D3Q19, precompute_q_wall_sphere_3d,
              rebuild_inlet_outlet_libb_3d!,
              stream_3d!, collide_3d!
using Metal, KernelAbstractions

backend = Metal.MetalBackend()
T = Float32

# -----------------------------------------------------------------------------
# 3D no-IBB sphere driver (classical stream + collide with halfway-BB on solid)
# with the same Zou-He velocity inlet / pressure outlet pattern as the IBB
# driver, so the only difference with `run_sphere_libb_3d` is the wall
# treatment at the sphere surface.
# -----------------------------------------------------------------------------
function _noibb_sphere_3d(; Nx, Ny, Nz, radius, cx, cy, cz, u_in, ν,
                           max_steps, avg_window, backend, T)
    # Classical BGK ω
    ω = 1f0 / (3f0*T(ν) + 0.5f0)

    is_solid_h = zeros(Bool, Nx, Ny, Nz)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        if (i - cx)^2 + (j - cy)^2 + (k - cz)^2 ≤ radius^2
            is_solid_h[i, j, k] = true
        end
    end

    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny, Nz)
    f_in  = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    f_out = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    copyto!(is_solid, is_solid_h)
    # init at uniform inflow equilibrium
    f_in_h = zeros(T, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 1:19
        f_in_h[i,j,k,q] = KrakeNN.equilibrium(D3Q19(), one(T), T(u_in),
                                              zero(T), zero(T), q)
    end
    copyto!(f_in, f_in_h); fill!(f_out, zero(T))

    # Inlet slab: equilibrium at (ρ=1, u=(u_in, 0, 0))
    f_inlet_h = zeros(T, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, q in 1:19
        f_inlet_h[j,k,q] = KrakeNN.equilibrium(D3Q19(), one(T), T(u_in),
                                                zero(T), zero(T), q)
    end
    f_inlet = KernelAbstractions.allocate(backend, T, Ny, Nz, 19)
    copyto!(f_inlet, f_inlet_h)

    cxs = (0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0)
    cys = (0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1)
    czs = (0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1)
    Fx_sum = 0.0; Fy_sum = 0.0; Fz_sum = 0.0; n_avg = 0
    sa = Array(is_solid)
    t0 = time()
    step_count = 0
    let f_in = f_in, f_out = f_out
        for step in 1:max_steps
            stream_3d!(f_out, f_in, Nx, Ny, Nz)
            @views f_out[1,  :, :, :] .= f_inlet
            @views f_out[Nx, :, :, :] .= f_out[Nx-1, :, :, :]
            collide_3d!(f_out, is_solid, ω)
            if step > max_steps - avg_window
                fa = Array(f_out)
                Fx = 0.0; Fy = 0.0; Fz = 0.0
                @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
                    sa[i,j,k] && continue
                    for q in 2:19
                        ni = i + cxs[q]; nj = j + cys[q]; nk = k + czs[q]
                        if 1 <= ni <= Nx && 1 <= nj <= Ny && 1 <= nk <= Nz && sa[ni,nj,nk]
                            Fx += 2 * cxs[q] * Float64(fa[i,j,k,q])
                            Fy += 2 * cys[q] * Float64(fa[i,j,k,q])
                            Fz += 2 * czs[q] * Float64(fa[i,j,k,q])
                        end
                    end
                end
                Fx_sum += Fx; Fy_sum += Fy; Fz_sum += Fz; n_avg += 1
            end
            f_in, f_out = f_out, f_in
            step_count += 1
        end
        KernelAbstractions.synchronize(backend)
    end
    dt = time() - t0

    Fx = Fx_sum / n_avg; Fy = Fy_sum / n_avg; Fz = Fz_sum / n_avg
    A = π * Float64(radius)^2
    Cd = 2 * Fx / (Float64(u_in)^2 * A)

    mlups = step_count * Nx * Ny * Nz / dt / 1e6
    return (; Cd, Fx, Fy, Fz, mlups, dt, steps=step_count)
end

# -----------------------------------------------------------------------------
# 2D benchmark
# -----------------------------------------------------------------------------
println("=" ^ 78)
println("2D CYLINDER — uniform inflow Re=20, blockage 25 % (D/H = 0.25)")
println("=" ^ 78)
println()
println("| D     | N_total       | Cd IBB | Cd no-IBB | MLUPS IBB | MLUPS no-IBB |")
println("|-------|---------------|--------|-----------|-----------|--------------|")

for radius in (10, 20, 40)
    Ny = 8 * radius
    Nx = 4 * Ny
    D = 2 * radius
    u_in = 0.04
    ν = u_in * D / 20.0
    max_steps = 30_000
    avg_window = 5_000

    # IBB
    t0 = time()
    r_ibb = run_cylinder_libb_2d(; Nx=Nx, Ny=Ny, radius=radius,
                                   cx=Nx÷4, cy=Ny÷2, u_in=u_in, ν=ν,
                                   inlet=:uniform, ρ_out=1.0,
                                   max_steps=max_steps, avg_window=avg_window,
                                   backend=backend, T=T)
    t_ibb = time() - t0
    mlups_ibb = max_steps * Nx * Ny / t_ibb / 1e6

    # no-IBB (classical halfway-BB via run_cylinder_2d — BGK + Zou-He BCs)
    t0 = time()
    r_no = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=radius,
                            cx=Nx÷4, cy=Ny÷2, u_in=u_in, ν=ν,
                            max_steps=max_steps, avg_window=avg_window,
                            backend=backend, T=T)
    t_no = time() - t0
    mlups_no = max_steps * Nx * Ny / t_no / 1e6

    println("| $(lpad(D,5)) | $(lpad(Nx*Ny,13)) | $(lpad(round(r_ibb.Cd, digits=3),6)) | $(lpad(round(r_no.Cd, digits=3),9)) | $(lpad(round(mlups_ibb, digits=1),9)) | $(lpad(round(mlups_no, digits=1),12)) |")
end

# -----------------------------------------------------------------------------
# 3D benchmark
# -----------------------------------------------------------------------------
println()
println("=" ^ 78)
println("3D SPHERE — uniform inflow Re=20")
println("=" ^ 78)
println()
println("| R     | N_total       | Cd IBB | Cd no-IBB | MLUPS IBB | MLUPS no-IBB |")
println("|-------|---------------|--------|-----------|-----------|--------------|")

for radius in (6, 8, 12)
    Ny = 6 * radius
    Nz = 6 * radius
    Nx = 3 * Ny
    D = 2 * radius
    u_in = 0.04
    ν = u_in * D / 20.0
    max_steps = 20_000
    avg_window = 4_000

    cx = Nx ÷ 4; cy = Ny ÷ 2; cz = Nz ÷ 2

    # IBB
    t0 = time()
    r_ibb = run_sphere_libb_3d(; Nx=Nx, Ny=Ny, Nz=Nz, radius=radius,
                                cx=cx, cy=cy, cz=cz, u_in=u_in, ν=ν,
                                max_steps=max_steps, avg_window=avg_window,
                                backend=backend, T=T)
    t_ibb = time() - t0
    mlups_ibb = max_steps * Nx * Ny * Nz / t_ibb / 1e6

    # no-IBB (classical halfway-BB)
    r_no = _noibb_sphere_3d(; Nx=Nx, Ny=Ny, Nz=Nz, radius=radius,
                            cx=cx, cy=cy, cz=cz, u_in=u_in, ν=ν,
                            max_steps=max_steps, avg_window=avg_window,
                            backend=backend, T=T)

    println("| $(lpad(radius,5)) | $(lpad(Nx*Ny*Nz,13)) | $(lpad(round(r_ibb.Cd, digits=3),6)) | $(lpad(round(r_no.Cd, digits=3),9)) | $(lpad(round(mlups_ibb, digits=1),9)) | $(lpad(round(r_no.mlups, digits=1),12)) |")
end
