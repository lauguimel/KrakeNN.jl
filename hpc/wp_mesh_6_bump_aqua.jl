# WP-MESH-6 — cylinder-in-cross-flow Re=100 (Williamson 1996),
# matrix on Aqua H100 (CUDA Float64).
#
#   Domain    : 1.0 × 0.5 (symmetric, blockage 10 %)
#   Cylinder  : centered (0.5, 0.25), R = 0.025
#   Re = u_max · D / ν = 100
#   Reference : Cd ≈ 1.4   Cl_RMS ≈ 0.33   St ≈ 0.165
#               (Williamson 1996, Park 1998, Henderson 1995)
#
# Three baselines × three resolutions, ALL with the SAME total cell
# count at each resolution so the comparison is "what does each kernel
# give for the same compute budget":
#
#   (A) cartesian_mesh  + halfway-BB             — uniform Cartesian
#   (B) cartesian_mesh  + LI-BB v2 (sub-cell q_w) — uniform Cartesian
#   (C) gmsh Bump 0.1   + SLBM + LI-BB v2          — structured non-regular,
#                                                   cells densified at the
#                                                   centre (cylinder)
#
# Per run we record Cd_mean, Cl_RMS, Strouhal St over the second half
# of the trajectory.

using KrakeNN, CUDA, KernelAbstractions, Gmsh
using FFTW: rfft, rfftfreq

const Lx, Ly         = 1.0, 0.5
const cx_p, cy_p     = 0.5, 0.245   # cy slightly off-axis (10% of D below
                                    # the centreline) to break symmetry and
                                    # trigger vortex shedding without having
                                    # to wait for numerical noise to amplify.
                                    # Same trick as Schäfer-Turek 2D-2.
const R_p            = 0.025
const Re_target      = 100.0
const u_max          = 0.04
const u_mean         = u_max               # uniform inlet here, not parabolic
const Cd_ref         = 1.4
const Cl_ref         = 0.33
const St_ref         = 0.165
const BUMP_COEFS     = (0.1, 0.5)   # 0.1 = extreme stretching (ratio ~93×, instability study)
                                    # 0.5 = moderate stretching (ratio ~10×, paper demo)
T = Float64

backend = CUDABackend()

function _setup(D_lu)
    dx_ref = 2 * R_p / D_lu
    Nx = round(Int, Lx / dx_ref) + 1
    Ny = round(Int, Ly / dx_ref) + 1
    cx_lu = cx_p / dx_ref; cy_lu = cy_p / dx_ref; R_lu = R_p / dx_ref
    ν = u_mean * D_lu / Re_target
    return (; Nx, Ny, dx_ref, cx_lu, cy_lu, R_lu, ν)
end

function _inlet_uniform(Ny)
    u_prof_h = fill(T(u_max), Ny)
    return CuArray(u_prof_h), u_prof_h
end

function _init_f(Nx, Ny, u_prof_h, is_solid_h)
    f_h = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        u = is_solid_h[i, j] ? 0.0 : Float64(u_prof_h[j])
        f_h[i, j, q] = T(KrakeNN.equilibrium(D2Q9(), 1.0, u, 0.0, q))
    end
    return f_h
end

function _strouhal(cl, dt_lu, D_lu)
    cl_z = cl .- (sum(cl) / length(cl))
    F = abs.(rfft(cl_z))
    fs = rfftfreq(length(cl_z), 1 / dt_lu)
    idx = argmax(F[2:end]) + 1
    return fs[idx] * D_lu / Float64(u_mean)
end

function _stats(cd, cl)
    cd_mean = sum(cd) / length(cd)
    cl_mean = sum(cl) / length(cl)
    cl_rms  = sqrt(sum((cl .- cl_mean).^2) / length(cl))
    return cd_mean, cl_rms
end

function _format(label, cells, cd, clr, St, mlups, elapsed)
    err_cd  = 100*abs(cd - Cd_ref)/Cd_ref
    err_clr = 100*abs(clr - Cl_ref)/Cl_ref
    err_st  = 100*abs(St - St_ref)/St_ref
    println("[$label cells=$cells] Cd=$(round(cd,digits=3)) (err $(round(err_cd,digits=2))%)  ",
            "Cl_RMS=$(round(clr,digits=3)) (err $(round(err_clr,digits=2))%)  ",
            "St=$(round(St,digits=3)) (err $(round(err_st,digits=2))%)  | ",
            "$(round(mlups,digits=0)) MLUPS  $(round(elapsed,digits=1))s")
end

function _step_loop_cart(label, s, is_solid_h, q_wall_h, steps, sample_every, sample_window)
    is_solid = CuArray(is_solid_h); q_wall = CuArray(T.(q_wall_h))
    uw_x = CuArray(zeros(T, s.Nx, s.Ny, 9)); uw_y = CuArray(zeros(T, s.Nx, s.Ny, 9))
    u_prof, u_prof_h = _inlet_uniform(s.Ny)
    bcspec = BCSpec2D(; west=ZouHeVelocity(u_prof), east=ZouHePressure(one(T)),
                        south=HalfwayBB(), north=HalfwayBB())
    fa = CuArray(_init_f(s.Nx, s.Ny, u_prof_h, is_solid_h))
    fb = similar(fa); fill!(fb, zero(T))
    ρ = CUDA.ones(T, s.Nx, s.Ny); ux = CUDA.zeros(T, s.Nx, s.Ny); uy = CUDA.zeros(T, s.Nx, s.Ny)
    cd_h = Float64[]; cl_h = Float64[]
    norm = u_mean^2 * (s.R_lu * 2)
    t0 = time()
    for step in 1:steps
        fused_trt_libb_v2_step!(fb, fa, ρ, ux, uy, is_solid, q_wall, uw_x, uw_y, s.Nx, s.Ny, s.ν)
        apply_bc_rebuild_2d!(fb, fa, bcspec, s.ν, s.Nx, s.Ny)
        if step > steps - sample_window && step % sample_every == 0
            Fx, Fy = compute_drag_libb_mei_2d(fb, q_wall, uw_x, uw_y, s.Nx, s.Ny)
            push!(cd_h, 2*Fx/norm); push!(cl_h, 2*Fy/norm)
        end
        fa, fb = fb, fa
    end
    CUDA.synchronize()
    elapsed = time() - t0
    cd, clr = _stats(cd_h, cl_h)
    St = _strouhal(cl_h, sample_every, 2 * s.R_lu)
    cells = s.Nx * s.Ny
    mlups = cells * steps / elapsed / 1e6
    _format(label, cells, cd, clr, St, mlups, elapsed)
    return (; cells, cd, cl_rms=clr, St, mlups, elapsed)
end

run_A(D_lu, steps, sw, se) = begin
    s = _setup(D_lu)
    is_solid_h = zeros(Bool, s.Nx, s.Ny)
    for j in 1:s.Ny, i in 1:s.Nx
        (i-1.0-s.cx_lu)^2 + (j-1.0-s.cy_lu)^2 ≤ s.R_lu^2 && (is_solid_h[i,j] = true)
    end
    q_wall_h = zeros(T, s.Nx, s.Ny, 9)
    cxs = KrakeNN.velocities_x(D2Q9()); cys = KrakeNN.velocities_y(D2Q9())
    for j in 1:s.Ny, i in 1:s.Nx
        is_solid_h[i,j] && continue
        for q in 2:9
            in_, jn_ = i + cxs[q], j + cys[q]
            (in_<1 || in_>s.Nx || jn_<1 || jn_>s.Ny) && continue
            is_solid_h[in_, jn_] && (q_wall_h[i,j,q] = T(0.5))
        end
    end
    _step_loop_cart("A D=$D_lu", s, is_solid_h, q_wall_h, steps, se, sw)
end

run_B(D_lu, steps, sw, se) = begin
    s = _setup(D_lu)
    qwh, ish = precompute_q_wall_cylinder(s.Nx, s.Ny, s.cx_lu, s.cy_lu, s.R_lu)
    _step_loop_cart("B D=$D_lu", s, ish, qwh, steps, se, sw)
end

run_C(D_lu, steps, sw, se; bump_coef=0.1, label_suffix="") = begin
    s = _setup(D_lu)
    fpath = tempname() * ".msh"
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0); gmsh.model.add("bump")
        gmsh.model.geo.addPoint(0.0, 0.0, 0.0, 0.1, 1)
        gmsh.model.geo.addPoint(Lx,  0.0, 0.0, 0.1, 2)
        gmsh.model.geo.addPoint(Lx,  Ly,  0.0, 0.1, 3)
        gmsh.model.geo.addPoint(0.0, Ly,  0.0, 0.1, 4)
        gmsh.model.geo.addLine(1,2,1); gmsh.model.geo.addLine(2,3,2)
        gmsh.model.geo.addLine(3,4,3); gmsh.model.geo.addLine(4,1,4)
        gmsh.model.geo.addCurveLoop([1,2,3,4],1); gmsh.model.geo.addPlaneSurface([1],1)
        gmsh.model.geo.mesh.setTransfiniteCurve(1, s.Nx, "Bump", bump_coef)
        gmsh.model.geo.mesh.setTransfiniteCurve(3, s.Nx, "Bump", bump_coef)
        gmsh.model.geo.mesh.setTransfiniteCurve(2, s.Ny, "Bump", bump_coef)
        gmsh.model.geo.mesh.setTransfiniteCurve(4, s.Ny, "Bump", bump_coef)
        gmsh.model.geo.mesh.setTransfiniteSurface(1); gmsh.model.geo.mesh.setRecombine(2,1)
        gmsh.model.geo.synchronize(); gmsh.model.mesh.generate(2); gmsh.write(fpath)
    finally
        gmsh.finalize()
    end
    mesh, _ = load_gmsh_mesh_2d(fpath)
    geom_h = build_slbm_geometry(mesh)
    geom = transfer_slbm_geometry(geom_h, backend)
    is_solid_h = zeros(Bool, mesh.Nξ, mesh.Nη)
    for j in 1:mesh.Nη, i in 1:mesh.Nξ
        x = mesh.X[i,j]; y = mesh.Y[i,j]
        (x-cx_p)^2 + (y-cy_p)^2 ≤ R_p^2 && (is_solid_h[i,j] = true)
    end
    qw_h, uwx_h, uwy_h = precompute_q_wall_slbm_cylinder_2d(mesh, is_solid_h, cx_p, cy_p, R_p)
    is_solid = CuArray(is_solid_h); q_wall = CuArray(T.(qw_h))
    uw_x = CuArray(T.(uwx_h)); uw_y = CuArray(T.(uwy_h))
    u_prof, u_prof_h = _inlet_uniform(mesh.Nη)
    bcspec = BCSpec2D(; west=ZouHeVelocity(u_prof), east=ZouHePressure(one(T)),
                        south=HalfwayBB(), north=HalfwayBB())
    fa = CuArray(_init_f(mesh.Nξ, mesh.Nη, u_prof_h, is_solid_h)); fb=similar(fa); fill!(fb,0)
    ρ = CUDA.ones(T, mesh.Nξ, mesh.Nη); ux = CUDA.zeros(T, mesh.Nξ, mesh.Nη); uy = CUDA.zeros(T, mesh.Nξ, mesh.Nη)
    cd_h = Float64[]; cl_h = Float64[]
    norm = u_mean^2 * (s.R_lu * 2)
    t0 = time()
    for step in 1:steps
        slbm_trt_libb_step!(fb, fa, ρ, ux, uy, is_solid, q_wall, uw_x, uw_y, geom, s.ν)
        apply_bc_rebuild_2d!(fb, fa, bcspec, s.ν, mesh.Nξ, mesh.Nη)
        if step > steps - sw && step % se == 0
            Fx, Fy = compute_drag_libb_mei_2d(fb, q_wall, uw_x, uw_y, mesh.Nξ, mesh.Nη)
            push!(cd_h, 2*Fx/norm); push!(cl_h, 2*Fy/norm)
        end
        fa, fb = fb, fa
    end
    CUDA.synchronize()
    elapsed = time() - t0
    cd, clr = _stats(cd_h, cl_h)
    St = _strouhal(cl_h, se, 2 * s.R_lu)
    cells = mesh.Nξ * mesh.Nη
    mlups = cells * steps / elapsed / 1e6
    _format("C$(label_suffix) D=$D_lu", cells, cd, clr, St, mlups, elapsed)
    return (; cells, cd, cl_rms=clr, St, mlups, elapsed)
end

println("=== WP-MESH-6 — Cylinder cross-flow Re=100 (CUDA H100, FP64) ===")
println("Reference (Williamson 1996, Park 1998): Cd≈$Cd_ref  Cl_RMS≈$Cl_ref  St≈$St_ref")
println("Domain $Lx × $Ly  cylinder ($cx_p, $cy_p) R=$R_p  blockage = $(round(2*R_p/Ly*100, digits=1))%")
println("Bump coefs = $BUMP_COEFS\n")

for D_lu in (20, 40, 80)
    steps = D_lu == 20 ? 80_000 : D_lu == 40 ? 160_000 : 320_000
    sw = steps ÷ 2
    se = D_lu == 80 ? 20 : 10
    println("--- D_lu=$D_lu  steps=$steps  sample_every=$se  window=$sw ---")
    try; run_A(D_lu, steps, sw, se); catch e; println("  A FAILED: ", sprint(showerror, e)[1:min(end,200)]); end
    try; run_B(D_lu, steps, sw, se); catch e; println("  B FAILED: ", sprint(showerror, e)[1:min(end,200)]); end
    for bc in BUMP_COEFS
        suffix = "[b=$(bc)]"
        try; run_C(D_lu, steps, sw, se; bump_coef=bc, label_suffix=suffix)
        catch e; println("  C$suffix FAILED: ", sprint(showerror, e)[1:min(end,200)]); end
    end
end

println("\n=== Done ===")
