using KrakeNN, CUDA, KernelAbstractions
using KrakeNN: LBMSpec, PullSLBM, SolidInert, ApplyLiBBPrePhase,
              Moments, CollideTRTDirect, WriteMoments, build_lbm_kernel

# ===========================================================================
# SLBM paper benchmarks on CUDA (Aqua H100).
#
# Runs:
# 1. Schäfer-Turek 2D-1 convergence: uniform D=20/40/80 + stretched
# 2. AD proof-of-concept: dKE/dν + dFx/dR via Enzyme
#
# ===========================================================================

FT = Float64  # High precision on H100

function run_st(; Nx, Ny, x_str=0.0, y_str=0.0, label, steps=80_000)
    Lx = 2.2; Ly = 0.41; cx_p = 0.2; cy_p = 0.2; R_p = 0.05
    xdir = x_str > 0 ? :left : :none
    ydir = y_str > 0 ? :both : :none
    mesh = stretched_box_mesh(; x_min=0.0, x_max=Lx, y_min=0.0, y_max=Ly,
                                Nx=Nx, Ny=Ny, x_stretch=x_str, y_stretch=y_str,
                                x_stretch_dir=xdir, y_stretch_dir=ydir, FT=FT)
    use_local_cfl = (x_str > 0 || y_str > 0)
    geom_h = build_slbm_geometry(mesh; local_cfl=use_local_cfl)
    dx_ref = mesh.dx_ref; D_lu = 2*R_p/dx_ref
    u_max = min(0.04, 0.1*20.0/D_lu); u_mean = 2*u_max/3; ν = u_mean*D_lu/20.0

    is_solid_h = zeros(Bool, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        dx = mesh.X[i,j]-cx_p; dy = mesh.Y[i,j]-cy_p
        dx*dx + dy*dy ≤ R_p^2 && (is_solid_h[i,j] = true)
    end
    qw_h, uwx_h, uwy_h = precompute_q_wall_slbm_cylinder_2d(mesh, is_solid_h, cx_p, cy_p, R_p)

    backend = CUDABackend()
    is_solid = CuArray(is_solid_h); q_wall = CuArray(FT.(qw_h))
    uwx = CuArray(FT.(uwx_h)); uwy = CuArray(FT.(uwy_h))
    i_dep = CuArray(FT.(geom_h.i_dep)); j_dep = CuArray(FT.(geom_h.j_dep))

    spec = LBMSpec(PullSLBM(), SolidInert(), ApplyLiBBPrePhase(),
                   Moments(), CollideTRTDirect(), WriteMoments())
    kernel! = build_lbm_kernel(backend, spec)
    sp, sm = trt_rates(ν)

    u_prof_h = FT[FT(4*u_max*(j-1)*(Ny-j)/(Ny-1)^2) for j in 1:Ny]
    u_prof = CuArray(u_prof_h)
    bc = BCSpec2D(west=ZouHeVelocity(u_prof), east=ZouHePressure(FT(1)))

    f_in = CuArray(zeros(FT, Nx, Ny, 9)); f_out = similar(f_in)
    ρ = CuArray(ones(FT, Nx, Ny)); ux = CuArray(zeros(FT, Nx, Ny)); uy = similar(ux)
    f_init = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_init[i,j,q] = FT(KrakeNN.equilibrium(D2Q9(), 1.0, Float64(u_prof_h[min(j,Ny)]), 0.0, q))
    end
    copyto!(f_in, f_init)

    avg_win = steps ÷ 4; Fx_sum = 0.0; n_avg = 0
    t0 = time()
    for step in 1:steps
        kernel!(f_out, ρ, ux, uy, f_in, is_solid, q_wall, uwx, uwy,
                i_dep, j_dep, Nx, Ny, FT(sp), FT(sm), false, false; ndrange=(Nx, Ny))
        apply_bc_rebuild_2d!(f_out, f_in, bc, ν, Nx, Ny)
        if step > steps - avg_win
            Fx, _ = compute_drag_libb_mei_2d(f_out, q_wall, uwx, uwy, Nx, Ny)
            Fx_sum += Fx; n_avg += 1
        end
        f_in, f_out = f_out, f_in
    end
    CUDA.synchronize()
    elapsed = time() - t0; mlups = Nx*Ny*steps/elapsed/1e6
    Cd = 2*(Fx_sum/n_avg)/(1.0*u_mean^2*D_lu)
    err = 100*abs(Cd - 5.58)/5.58
    cells = Nx*Ny
    println("$label $(Nx)×$(Ny) ($(cells) cells): Cd=$(round(Cd,digits=4)), err=$(round(err,digits=2))%, $(round(mlups,digits=0)) MLUPS, $(round(elapsed,digits=1))s")
end

println("=== SLBM Schäfer-Turek 2D-1 on H100 (CUDA Float64) ===\n")

println("--- Uniform convergence (baseline) ---")
run_st(Nx=441, Ny=83, label="Uniform D=20")
run_st(Nx=881, Ny=165, label="Uniform D=40")
run_st(Nx=1761, Ny=329, label="Uniform D=80")

println("\n--- Stretched (local CFL) — same or fewer cells ---")
run_st(Nx=441, Ny=83, x_str=1.0, label="Stretch 441×83 s=1.0")
run_st(Nx=550, Ny=100, x_str=1.0, label="Stretch 550×100 s=1.0")
run_st(Nx=660, Ny=120, x_str=0.8, label="Stretch 660×120 s=0.8")
run_st(Nx=881, Ny=165, x_str=1.0, label="Stretch 881×165 s=1.0")

println("\n--- Stretched high-res ---")
run_st(Nx=1761, Ny=329, x_str=1.0, label="Stretch 1761×329 s=1.0")

println("\n=== Done ===")
