# WP-MESH-6 Chemin B — single long run of (C) SLBM+LI-BB on Bump 0.1.
#
# Goal: the matrix run (wp_mesh_6_bump_aqua.jl) showed (C) at D=40 BUMP=0.1
# gives Cd=1.63 (matches Cartesian) but Cl_RMS≈0 (no shedding). The dx_ref
# for Bump 0.1 is ~7× smaller than the Cartesian dx_ref, so the effective
# physical time per step is ~7× smaller. With the same step count, (C)
# covers ~1.5 flow-through times vs (A)(B)'s ~8 — shedding has not yet
# triggered in (C). This script re-runs (C) alone with 10× more steps
# (1.6M) to let the shedding transient develop.

using KrakeNN, CUDA, KernelAbstractions, Gmsh
using FFTW: rfft, rfftfreq

const Lx, Ly         = 1.0, 0.5
const cx_p, cy_p     = 0.5, 0.245
const R_p            = 0.025
const Re_target      = 100.0
const u_max          = 0.04
const u_mean         = u_max
const Cd_ref         = 1.4
const Cl_ref         = 0.33
const St_ref         = 0.165
const BUMP_COEF      = 0.1
T = Float64
backend = CUDABackend()

D_lu = 40
steps = 1_600_000       # 10× the matrix run — ~15 flow-through times on Bump 0.1
sample_every = 20
sample_window = 800_000

dx_ref = 2 * R_p / D_lu
Nx = round(Int, Lx / dx_ref) + 1
Ny = round(Int, Ly / dx_ref) + 1
ν = u_mean * D_lu / Re_target

println("=== WP-MESH-6 Chemin B — (C) Bump 0.1 long run ===")
println("Reference (Williamson 1996): Cd≈$Cd_ref  Cl_RMS≈$Cl_ref  St≈$St_ref")
println("D_lu=$D_lu  steps=$steps  sample_every=$sample_every  window=$sample_window")

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
    gmsh.model.geo.mesh.setTransfiniteCurve(1, Nx, "Bump", BUMP_COEF)
    gmsh.model.geo.mesh.setTransfiniteCurve(3, Nx, "Bump", BUMP_COEF)
    gmsh.model.geo.mesh.setTransfiniteCurve(2, Ny, "Bump", BUMP_COEF)
    gmsh.model.geo.mesh.setTransfiniteCurve(4, Ny, "Bump", BUMP_COEF)
    gmsh.model.geo.mesh.setTransfiniteSurface(1); gmsh.model.geo.mesh.setRecombine(2,1)
    gmsh.model.geo.synchronize(); gmsh.model.mesh.generate(2); gmsh.write(fpath)
finally
    gmsh.finalize()
end

mesh, _ = load_gmsh_mesh_2d(fpath)
println("Bump mesh loaded: $(mesh.Nξ) × $(mesh.Nη) nodes  dx_ref=$(mesh.dx_ref)")

geom_h = build_slbm_geometry(mesh)
geom = transfer_slbm_geometry(geom_h, backend)

is_solid_h = zeros(Bool, mesh.Nξ, mesh.Nη)
for j in 1:mesh.Nη, i in 1:mesh.Nξ
    x = mesh.X[i,j]; y = mesh.Y[i,j]
    (x-cx_p)^2 + (y-cy_p)^2 ≤ R_p^2 && (is_solid_h[i,j] = true)
end
println("solid cells: $(sum(is_solid_h))")

qw_h, uwx_h, uwy_h = precompute_q_wall_slbm_cylinder_2d(mesh, is_solid_h, cx_p, cy_p, R_p)
is_solid = CuArray(is_solid_h); q_wall = CuArray(T.(qw_h))
uw_x = CuArray(T.(uwx_h)); uw_y = CuArray(T.(uwy_h))

u_prof_h = fill(T(u_max), mesh.Nη)
u_prof = CuArray(u_prof_h)
bcspec = BCSpec2D(; west=ZouHeVelocity(u_prof), east=ZouHePressure(one(T)),
                    south=HalfwayBB(), north=HalfwayBB())

f_h = zeros(T, mesh.Nξ, mesh.Nη, 9)
for j in 1:mesh.Nη, i in 1:mesh.Nξ, q in 1:9
    u = is_solid_h[i,j] ? 0.0 : Float64(u_prof_h[j])
    f_h[i,j,q] = T(KrakeNN.equilibrium(D2Q9(), 1.0, u, 0.0, q))
end
fa = CuArray(f_h); fb = similar(fa); fill!(fb, zero(T))
ρ = CUDA.ones(T, mesh.Nξ, mesh.Nη); ux = CUDA.zeros(T, mesh.Nξ, mesh.Nη); uy = CUDA.zeros(T, mesh.Nξ, mesh.Nη)

function _timeloop!(fa, fb, ρ, ux, uy, is_solid, q_wall, uw_x, uw_y, geom, bcspec,
                    ν, Nξ, Nη, steps, sample_every, sample_window, norm_)
    cd_h = Float64[]; cl_h = Float64[]
    for step in 1:steps
        slbm_trt_libb_step!(fb, fa, ρ, ux, uy, is_solid, q_wall, uw_x, uw_y, geom, ν)
        apply_bc_rebuild_2d!(fb, fa, bcspec, ν, Nξ, Nη)
        if step > steps - sample_window && step % sample_every == 0
            Fx, Fy = compute_drag_libb_mei_2d(fb, q_wall, uw_x, uw_y, Nξ, Nη)
            push!(cd_h, 2*Fx/norm_); push!(cl_h, 2*Fy/norm_)
        end
        fa, fb = fb, fa
        if step % 200_000 == 0
            CUDA.synchronize()
            rho_h = Array(ρ); ux_h = Array(ux)
            println("  step $step : ρ ∈ [$(round(minimum(rho_h),digits=4)), $(round(maximum(rho_h),digits=4))]  ",
                    "ux ∈ [$(round(minimum(ux_h),digits=4)), $(round(maximum(ux_h),digits=4))]")
        end
    end
    return cd_h, cl_h
end

norm_ = u_mean^2 * (R_p * 2 / mesh.dx_ref)
t0 = time()
cd_h, cl_h = _timeloop!(fa, fb, ρ, ux, uy, is_solid, q_wall, uw_x, uw_y, geom,
                         bcspec, ν, mesh.Nξ, mesh.Nη, steps, sample_every,
                         sample_window, norm_)
CUDA.synchronize()
elapsed = time() - t0

cd_mean = sum(cd_h) / length(cd_h)
cl_mean = sum(cl_h) / length(cl_h)
cl_rms  = sqrt(sum((cl_h .- cl_mean).^2) / length(cl_h))
cl_z = cl_h .- cl_mean
F = abs.(rfft(cl_z))
fs = rfftfreq(length(cl_z), 1 / sample_every)
idx = argmax(F[2:end]) + 1
St = fs[idx] * (2 * R_p / mesh.dx_ref) / Float64(u_mean)

err_cd  = 100*abs(cd_mean - Cd_ref)/Cd_ref
err_clr = 100*abs(cl_rms - Cl_ref)/Cl_ref
err_st  = 100*abs(St - St_ref)/St_ref
cells = mesh.Nξ * mesh.Nη
mlups = cells * steps / elapsed / 1e6
println("\n[C long D=$D_lu cells=$cells]  Cd=$(round(cd_mean,digits=3)) (err $(round(err_cd,digits=2))%)  ",
        "Cl_RMS=$(round(cl_rms,digits=3)) (err $(round(err_clr,digits=2))%)  ",
        "St=$(round(St,digits=3)) (err $(round(err_st,digits=2))%)  | ",
        "$(round(mlups,digits=0)) MLUPS  $(round(elapsed,digits=1))s")

println("\n=== Done ===")
