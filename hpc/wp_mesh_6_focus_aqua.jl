# WP-MESH-6 extension — cylinder-focused single-block (approach D).
#
# Same physical setup and sampling protocol as `wp_mesh_6_bump_aqua.jl`
# (cylinder Re=100, domain 1.0 × 0.5, blockage 10%, uniform inlet
# u=0.04, Cd/Cl_RMS/St over the 2nd half of the trajectory). The only
# change is the mesh generator:
#
#   (C) gmsh Bump — nodes clustered toward the rectangle BOUNDARIES
#   (D) cylinder_focused_mesh — nodes clustered toward the CYLINDER
#       CENTRE (cx_p, cy_p) in physical space
#
# Both (C) and (D) are single-block body-confirming (cylinder carved
# out with LI-BB); they only differ in where the available cells are
# spent. (D) is the paper-relevant comparison: most published cylinder
# codes cluster cells on the body, not on the outer walls.
#
# Strength sweep: strength∈(1.0, 2.0) — moderate vs strong clustering.
# The effective stretching ratio between near-wall and far-field cells
# is roughly `exp(2*strength)`, so strength=1 → ~7×, strength=2 → ~55×.

using KrakeNN, CUDA, KernelAbstractions
using FFTW: rfft, rfftfreq

const Lx, Ly         = 1.0, 0.5
const cx_p, cy_p     = 0.5, 0.245   # same off-axis centre as (C) to trigger shedding
const R_p            = 0.025
const Re_target      = 100.0
const u_max          = 0.04
const u_mean         = u_max
const Cd_ref         = 1.4
const Cl_ref         = 0.33
const St_ref         = 0.165
const FOCUS_STRENGTHS = (1.0, 2.0)
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

run_D(D_lu, steps, sw, se; strength=1.0, label_suffix="") = begin
    s = _setup(D_lu)
    mesh = cylinder_focused_mesh(; x_min=0.0, x_max=Lx, y_min=0.0, y_max=Ly,
                                    Nx=s.Nx, Ny=s.Ny,
                                    cx=cx_p, cy=cy_p, strength=strength, FT=T)
    # Local-CFL + local τ: on a stretched mesh the smallest cells drive the
    # effective relaxation time toward the stability floor (τ→0.5). Using
    # local_cfl in the geometry and `compute_local_omega_2d` + `_step_local_2d!`
    # lets the kernel use a per-cell τ_local = τ_global · (dx_local/dx_ref)²
    # (quadratic scaling), which is the same pipeline validated in 2026-04-17/18
    # for local-CFL SLBM (memory: project_slbm_local_cfl).
    geom_h = build_slbm_geometry(mesh; local_cfl=true)
    geom = transfer_slbm_geometry(geom_h, backend)
    sp_h, sm_h = compute_local_omega_2d(mesh; ν=Float64(s.ν),
                                          scaling=:quadratic, τ_floor=0.51)
    sp_field = CuArray(T.(sp_h)); sm_field = CuArray(T.(sm_h))
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
        slbm_trt_libb_step_local_2d!(fb, fa, ρ, ux, uy, is_solid,
                                       q_wall, uw_x, uw_y, geom, sp_field, sm_field)
        apply_bc_rebuild_2d!(fb, fa, bcspec, s.ν, mesh.Nξ, mesh.Nη;
                              sp_field=sp_field, sm_field=sm_field)
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
    # Report the effective cylinder resolution: minimum edge length near
    # the body, divided by dx_nominal = 2R/D_lu. For a uniform mesh this
    # is 1; for strength=1 it is below 1 (finer near cylinder).
    min_dx = Float64(mesh.dx_ref)
    dx_nominal = 2 * R_p / D_lu
    stretch_ratio = dx_nominal / min_dx
    println("  (D$(label_suffix)): dx_min=$(round(min_dx, sigdigits=4))  ",
            "stretch_ratio (dx_nominal/dx_min)=$(round(stretch_ratio, digits=2))")
    _format("D$(label_suffix) D=$D_lu", cells, cd, clr, St, mlups, elapsed)
    return (; cells, cd, cl_rms=clr, St, mlups, elapsed, stretch_ratio)
end

println("=== WP-MESH-6 (D) — cylinder-focused single-block (CUDA H100, FP64) ===")
println("Reference (Williamson 1996, Park 1998): Cd≈$Cd_ref  Cl_RMS≈$Cl_ref  St≈$St_ref")
println("Domain $Lx × $Ly  cylinder ($cx_p, $cy_p) R=$R_p  blockage = $(round(2*R_p/Ly*100, digits=1))%")
println("Focus strengths = $FOCUS_STRENGTHS\n")

for D_lu in (20, 40, 80)
    steps = D_lu == 20 ? 80_000 : D_lu == 40 ? 160_000 : 320_000
    sw = steps ÷ 2
    se = D_lu == 80 ? 20 : 10
    println("--- D_lu=$D_lu  steps=$steps  sample_every=$se  window=$sw ---")
    for ss in FOCUS_STRENGTHS
        suffix = "[s=$(ss)]"
        try
            run_D(D_lu, steps, sw, se; strength=ss, label_suffix=suffix)
        catch e
            println("  D$suffix FAILED: ", sprint(showerror, e)[1:min(end,200)])
        end
    end
end

println("\n=== Done ===")
