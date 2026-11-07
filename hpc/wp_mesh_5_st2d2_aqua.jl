# WP-MESH-5 on Aqua (CUDA Float64). Same matrix as the local script but
# pinned to CUDABackend, longer sample windows.
using KrakeNN, CUDA, KernelAbstractions, Gmsh
using FFTW: rfft, rfftfreq

const Lx, Ly         = 2.2, 0.41
const cx_p, cy_p, R_p = 0.2, 0.2, 0.05
const Re_target       = 100.0
const u_max           = 0.04
const u_mean          = (2/3) * u_max
const Cd_ref          = 3.23
const Cl_ref          = 0.706
const St_ref          = 0.30
T = Float64

backend = CUDABackend()

function _setup_lattice(D_lu)
    dx_ref = 2 * R_p / D_lu
    Nx = round(Int, Lx / dx_ref) + 1
    Ny = round(Int, Ly / dx_ref) + 1
    cx_lu = cx_p / dx_ref; cy_lu = cy_p / dx_ref; R_lu = R_p / dx_ref
    ν = u_mean * D_lu / Re_target
    return (; Nx, Ny, dx_ref, cx_lu, cy_lu, R_lu, ν)
end

function _inlet_profile(Ny)
    u_prof_h = T[T(u_max) * 4 * (T(j-1) * (T(Ny-1) - T(j-1))) / T(Ny-1)^2 for j in 1:Ny]
    u_prof = CuArray(u_prof_h)
    return u_prof, u_prof_h
end

function _init_f(Nx, Ny, u_prof_h, is_solid_h)
    f_h = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        u = is_solid_h[i, j] ? 0.0 : Float64(u_prof_h[j])
        f_h[i, j, q] = T(KrakeNN.equilibrium(D2Q9(), 1.0, u, 0.0, q))
    end
    return f_h
end

function _strouhal(cl_history, dt_lu, D_lu)
    cl = cl_history .- (sum(cl_history) / length(cl_history))
    F  = abs.(rfft(cl))
    n  = length(cl)
    fs = rfftfreq(n, 1 / dt_lu)
    idx = argmax(F[2:end]) + 1
    return fs[idx] * D_lu / u_mean
end

function _cd_cl_rms(cd_history, cl_history)
    cd_mean = sum(cd_history) / length(cd_history)
    cl_mean = sum(cl_history) / length(cl_history)
    cl_rms  = sqrt(sum((cl_history .- cl_mean).^2) / length(cl_history))
    return (cd_mean, cl_rms)
end

function _step_loop_cart(s, is_solid_h, q_wall_h, label, steps, sample_every, sample_window)
    is_solid = CuArray(is_solid_h); q_wall = CuArray(T.(q_wall_h))
    uw_x = CuArray(zeros(T, s.Nx, s.Ny, 9)); uw_y = CuArray(zeros(T, s.Nx, s.Ny, 9))
    u_prof, u_prof_h = _inlet_profile(s.Ny)
    bcspec = BCSpec2D(; west=ZouHeVelocity(u_prof), east=ZouHePressure(one(T)),
                        south=HalfwayBB(), north=HalfwayBB())
    fa = CuArray(_init_f(s.Nx, s.Ny, u_prof_h, is_solid_h))
    fb = similar(fa); fill!(fb, zero(T))
    ρ  = CUDA.ones(T, s.Nx, s.Ny); ux = CUDA.zeros(T, s.Nx, s.Ny); uy = CUDA.zeros(T, s.Nx, s.Ny)
    cd_h = Float64[]; cl_h = Float64[]
    norm = u_mean^2 * (s.R_lu * 2)
    t0 = time()
    for step in 1:steps
        fused_trt_libb_v2_step!(fb, fa, ρ, ux, uy, is_solid, q_wall, uw_x, uw_y, s.Nx, s.Ny, s.ν)
        apply_bc_rebuild_2d!(fb, fa, bcspec, s.ν, s.Nx, s.Ny)
        if step > steps - sample_window && step % sample_every == 0
            Fx, Fy = compute_drag_libb_mei_2d(fb, q_wall, uw_x, uw_y, s.Nx, s.Ny)
            push!(cd_h, 2 * Fx / norm); push!(cl_h, 2 * Fy / norm)
        end
        fa, fb = fb, fa
    end
    CUDA.synchronize()
    elapsed = time() - t0
    cd, clr = _cd_cl_rms(cd_h, cl_h)
    St = _strouhal(cl_h, sample_every, 2 * s.R_lu)
    cells = s.Nx * s.Ny
    mlups = cells * steps / elapsed / 1e6
    println("[$label cells=$cells] Cd=$(round(cd,digits=3)) Cl_RMS=$(round(clr,digits=3)) St=$(round(St,digits=3)) | $(round(mlups,digits=0)) MLUPS $(round(elapsed,digits=1))s")
    return (; cells, cd, cl_rms=clr, St, mlups, elapsed,
            err_cd=100*abs(cd-Cd_ref)/Cd_ref,
            err_clr=100*abs(clr-Cl_ref)/Cl_ref,
            err_st=100*abs(St-St_ref)/St_ref)
end

run_A(D_lu, steps, sw, se) = begin
    s = _setup_lattice(D_lu)
    is_solid_h = zeros(Bool, s.Nx, s.Ny)
    for j in 1:s.Ny, i in 1:s.Nx
        (i-1.0-s.cx_lu)^2 + (j-1.0-s.cy_lu)^2 ≤ s.R_lu^2 && (is_solid_h[i,j] = true)
    end
    q_wall_h = zeros(T, s.Nx, s.Ny, 9)
    cxs=KrakeNN.velocities_x(D2Q9()); cys=KrakeNN.velocities_y(D2Q9())
    for j in 1:s.Ny, i in 1:s.Nx
        is_solid_h[i,j] && continue
        for q in 2:9
            in_,jn_ = i+cxs[q], j+cys[q]
            (in_<1||in_>s.Nx||jn_<1||jn_>s.Ny) && continue
            is_solid_h[in_,jn_] && (q_wall_h[i,j,q] = 0.5)
        end
    end
    _step_loop_cart(s, is_solid_h, q_wall_h, "A D=$D_lu", steps, se, sw)
end

run_B(D_lu, steps, sw, se) = begin
    s = _setup_lattice(D_lu)
    qwh, ish = precompute_q_wall_cylinder(s.Nx, s.Ny, s.cx_lu, s.cy_lu, s.R_lu)
    _step_loop_cart(s, ish, qwh, "B D=$D_lu", steps, se, sw)
end

run_C(D_lu, steps, sw, se) = begin
    s = _setup_lattice(D_lu)
    fpath = tempname() * ".msh"
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0); gmsh.model.add("st_block")
        gmsh.model.geo.addPoint(0,0,0,0.1,1); gmsh.model.geo.addPoint(Lx,0,0,0.1,2)
        gmsh.model.geo.addPoint(Lx,Ly,0,0.1,3); gmsh.model.geo.addPoint(0,Ly,0,0.1,4)
        gmsh.model.geo.addLine(1,2,1); gmsh.model.geo.addLine(2,3,2)
        gmsh.model.geo.addLine(3,4,3); gmsh.model.geo.addLine(4,1,4)
        gmsh.model.geo.addCurveLoop([1,2,3,4],1); gmsh.model.geo.addPlaneSurface([1],1)
        gmsh.model.geo.mesh.setTransfiniteCurve(1,s.Nx); gmsh.model.geo.mesh.setTransfiniteCurve(3,s.Nx)
        gmsh.model.geo.mesh.setTransfiniteCurve(2,s.Ny); gmsh.model.geo.mesh.setTransfiniteCurve(4,s.Ny)
        gmsh.model.geo.mesh.setTransfiniteSurface(1); gmsh.model.geo.mesh.setRecombine(2,1)
        gmsh.model.geo.synchronize(); gmsh.model.mesh.generate(2); gmsh.write(fpath)
    finally; gmsh.finalize(); end
    mesh, _ = load_gmsh_mesh_2d(fpath)
    geom_h = build_slbm_geometry(mesh)
    geom = transfer_slbm_geometry(geom_h, backend)
    is_solid_h = zeros(Bool, mesh.Nξ, mesh.Nη)
    for j in 1:mesh.Nη, i in 1:mesh.Nξ
        x=mesh.X[i,j]; y=mesh.Y[i,j]
        (x-cx_p)^2+(y-cy_p)^2 ≤ R_p^2 && (is_solid_h[i,j]=true)
    end
    qw_h, uwx_h, uwy_h = precompute_q_wall_slbm_cylinder_2d(mesh, is_solid_h, cx_p, cy_p, R_p)
    is_solid = CuArray(is_solid_h); q_wall = CuArray(T.(qw_h))
    uw_x = CuArray(T.(uwx_h)); uw_y = CuArray(T.(uwy_h))
    u_prof, u_prof_h = _inlet_profile(mesh.Nη)
    bcspec = BCSpec2D(; west=ZouHeVelocity(u_prof), east=ZouHePressure(one(T)),
                        south=HalfwayBB(), north=HalfwayBB())
    fa = CuArray(_init_f(mesh.Nξ, mesh.Nη, u_prof_h, is_solid_h)); fb=similar(fa); fill!(fb,0)
    ρ=CUDA.ones(T, mesh.Nξ, mesh.Nη); ux=CUDA.zeros(T, mesh.Nξ, mesh.Nη); uy=CUDA.zeros(T, mesh.Nξ, mesh.Nη)
    cd_h=Float64[]; cl_h=Float64[]
    norm = u_mean^2 * (s.R_lu * 2)
    t0=time()
    for step in 1:steps
        slbm_trt_libb_step!(fb, fa, ρ, ux, uy, is_solid, q_wall, uw_x, uw_y, geom, s.ν)
        apply_bc_rebuild_2d!(fb, fa, bcspec, s.ν, mesh.Nξ, mesh.Nη)
        if step > steps - sw && step % se == 0
            Fx, Fy = compute_drag_libb_mei_2d(fb, q_wall, uw_x, uw_y, mesh.Nξ, mesh.Nη)
            push!(cd_h, 2 * Fx / norm); push!(cl_h, 2 * Fy / norm)
        end
        fa, fb = fb, fa
    end
    CUDA.synchronize()
    elapsed = time() - t0
    cd, clr = _cd_cl_rms(cd_h, cl_h)
    St = _strouhal(cl_h, se, 2 * s.R_lu)
    cells = mesh.Nξ * mesh.Nη
    mlups = cells * steps / elapsed / 1e6
    println("[C D=$D_lu cells=$cells] Cd=$(round(cd,digits=3)) Cl_RMS=$(round(clr,digits=3)) St=$(round(St,digits=3)) | $(round(mlups,digits=0)) MLUPS $(round(elapsed,digits=1))s")
    return (; cells, cd, cl_rms=clr, St, mlups, elapsed,
            err_cd=100*abs(cd-Cd_ref)/Cd_ref,
            err_clr=100*abs(clr-Cl_ref)/Cl_ref,
            err_st=100*abs(St-St_ref)/St_ref)
end

println("=== WP-MESH-5 — ST 2D-2 Re=100 (CUDA H100, FP64) ===")
println("Reference: Cd≈3.23, Cl_RMS≈0.706, St≈0.30\n")

for D_lu in (20, 40, 80)
    steps = D_lu == 20 ? 80_000 : D_lu == 40 ? 160_000 : 320_000
    sw    = steps ÷ 2
    se    = D_lu == 80 ? 20 : 10
    println("--- D_lu = $D_lu (steps=$steps sample_every=$se) ---")
    try; run_A(D_lu, steps, sw, se); catch e; println("  A D=$D_lu FAILED: ", sprint(showerror, e)[1:min(end,200)]); end
    try; run_B(D_lu, steps, sw, se); catch e; println("  B D=$D_lu FAILED: ", sprint(showerror, e)[1:min(end,200)]); end
    try; run_C(D_lu, steps, sw, se); catch e; println("  C D=$D_lu FAILED: ", sprint(showerror, e)[1:min(end,200)]); end
end

println("\n=== Done ===")
