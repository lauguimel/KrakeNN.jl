# =============================================================================
# MLUPS-only benchmark: pure fused_trt_libb_v2 kernel throughput, no drag,
# no host transfers. Times a tight loop of `n_steps` after a warmup.
#
# Run on Aqua via `hpc/bench_mlups_only.pbs` → writes output_mlups.log.
# =============================================================================

using Pkg; Pkg.activate(dirname(@__DIR__))
using KrakeNN
using KrakeNN: fused_trt_libb_v2_step!, fused_trt_libb_v2_step_3d!, D2Q9, D3Q19,
              precompute_q_wall_cylinder, precompute_q_wall_sphere_3d,
              apply_bc_rebuild_2d!, apply_bc_rebuild_3d!,
              BCSpec2D, BCSpec3D, ZouHeVelocity, ZouHePressure
using CUDA, KernelAbstractions

_log(args...) = (println(args...); flush(stdout))
@assert CUDA.functional()
backend = CUDA.CUDABackend()

_log("=" ^ 78)
_log("MLUPS-only benchmark on $(CUDA.name(CUDA.device()))")
_log("Pure kernel throughput — no drag, no host transfers.")
_log("=" ^ 78)

function bench_2d(; Nx, Ny, radius, ν, u_max, warmup, steps, T)
    cx = Nx ÷ 4; cy = Ny ÷ 2
    q_wall_h, is_solid_h = precompute_q_wall_cylinder(Nx, Ny, cx, cy, radius; FT=T)
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
        f_in_h[i,j,q] = KrakeNN.equilibrium(D2Q9(), one(T), u_prof_h[j], zero(T), q)
    end
    copyto!(f_in, f_in_h); fill!(f_out, zero(T))
    fill!(ρ, one(T)); fill!(ux, zero(T)); fill!(uy, zero(T))
    bcspec = BCSpec2D(; west=ZouHeVelocity(u_prof), east=ZouHePressure(T(1.0)))

    let f_in = f_in, f_out = f_out
        for _ in 1:warmup
            fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                     q_wall, uw_x, uw_y, Nx, Ny, T(ν))
            apply_bc_rebuild_2d!(f_out, f_in, bcspec, ν, Nx, Ny)
            f_in, f_out = f_out, f_in
        end
        CUDA.synchronize()
        t0 = time()
        for _ in 1:steps
            fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                     q_wall, uw_x, uw_y, Nx, Ny, T(ν))
            apply_bc_rebuild_2d!(f_out, f_in, bcspec, ν, Nx, Ny)
            f_in, f_out = f_out, f_in
        end
        CUDA.synchronize()
        dt = time() - t0
        mlups = steps * Nx * Ny / dt / 1e6
        return (; dt, mlups)
    end
end

function bench_3d(; Nx, Ny, Nz, radius, ν, u_in, warmup, steps, T)
    cx = Nx ÷ 4; cy = Ny ÷ 2; cz = Nz ÷ 2
    q_wall_h, is_solid_h = precompute_q_wall_sphere_3d(Nx, Ny, Nz, cx, cy, cz,
                                                         radius; FT=T)
    q_wall   = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny, Nz)
    uw_x     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    uw_y     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    uw_z     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    f_in     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    f_out    = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19)
    ρ  = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    ux = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    uy = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    uz = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
    u_prof = KernelAbstractions.allocate(backend, T, Ny, Nz)
    copyto!(q_wall, q_wall_h); copyto!(is_solid, is_solid_h)
    fill!(uw_x, zero(T)); fill!(uw_y, zero(T)); fill!(uw_z, zero(T))
    fill!(u_prof, T(u_in))
    f_in_h = zeros(T, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 1:19
        f_in_h[i,j,k,q] = KrakeNN.equilibrium(D3Q19(), one(T), T(u_in),
                                              zero(T), zero(T), q)
    end
    copyto!(f_in, f_in_h); fill!(f_out, zero(T))
    fill!(ρ, one(T)); fill!(ux, zero(T)); fill!(uy, zero(T)); fill!(uz, zero(T))
    bcspec = BCSpec3D(; west=ZouHeVelocity(u_prof), east=ZouHePressure(T(1.0)))

    let f_in = f_in, f_out = f_out
        for _ in 1:warmup
            fused_trt_libb_v2_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                        q_wall, uw_x, uw_y, uw_z, Nx, Ny, Nz, T(ν))
            apply_bc_rebuild_3d!(f_out, f_in, bcspec, ν, Nx, Ny, Nz)
            f_in, f_out = f_out, f_in
        end
        CUDA.synchronize()
        t0 = time()
        for _ in 1:steps
            fused_trt_libb_v2_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                        q_wall, uw_x, uw_y, uw_z, Nx, Ny, Nz, T(ν))
            apply_bc_rebuild_3d!(f_out, f_in, bcspec, ν, Nx, Ny, Nz)
            f_in, f_out = f_out, f_in
        end
        CUDA.synchronize()
        dt = time() - t0
        mlups = steps * Nx * Ny * Nz / dt / 1e6
        return (; dt, mlups)
    end
end

# -----------------------------------------------------------------------------
# 2D scan (Float32 + Float64) at Re=20, fixed D/H=0.25, radius varying.
# -----------------------------------------------------------------------------
_log("\n## 2D cylinder (pure kernel MLUPS, Re=20)\n")
_log("| T       | D_lu | Nx × Ny     |   cells | MLUPS |")
_log("|---------|------|-------------|---------|-------|")
for T in (Float32, Float64), D_lu in (40, 80, 160, 320)
    Ny = Int(round(0.41 / 0.1 * D_lu))
    Nx = Int(round(2.2 / 0.1 * D_lu))
    radius = D_lu ÷ 2
    u_max = 0.06; u_mean = (2/3) * u_max
    ν = u_mean * D_lu / 20.0
    r = bench_2d(; Nx=Nx, Ny=Ny, radius=radius, ν=ν, u_max=u_max,
                   warmup=200, steps=2000, T=T)
    _log("| $(lpad(string(T),7)) | $(lpad(D_lu,4)) | $(lpad("$(Nx)×$(Ny)",11)) | $(lpad(Nx*Ny,7)) | $(lpad(round(r.mlups, digits=0),5)) |")
end

# -----------------------------------------------------------------------------
# 3D scan
# -----------------------------------------------------------------------------
_log("\n## 3D sphere (pure kernel MLUPS, Re=20)\n")
_log("| T       |  R | Nx × Ny × Nz     |    cells  | MLUPS |")
_log("|---------|----|------------------|-----------|-------|")
for T in (Float32, Float64), R in (12, 20, 32)
    Ny = 6 * R; Nz = 6 * R; Nx = 3 * Ny
    u_in = 0.04
    ν = u_in * 2R / 20.0
    r = bench_3d(; Nx=Nx, Ny=Ny, Nz=Nz, radius=R, ν=ν, u_in=u_in,
                   warmup=100, steps=1000, T=T)
    _log("| $(lpad(string(T),7)) | $(lpad(R,2)) | $(lpad("$(Nx)×$(Ny)×$(Nz)",16)) | $(lpad(Nx*Ny*Nz,9)) | $(lpad(round(r.mlups, digits=0),5)) |")
end

_log("\n=== DONE ===")
