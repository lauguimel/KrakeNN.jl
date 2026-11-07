# Local Metal smoke test for approach (D) — cylinder-focused single-block.
# Validates the cylinder_focused_mesh → build_slbm_geometry → SLBM+LI-BB
# pipeline on Float32/Metal at small D (no convergence claim, just no NaN
# and sensible Cd order-of-magnitude).
#
# Usage: julia --project=. tmp/smoke_focus_local_metal.jl
# Env   : D_LU (default 20), STEPS (default 5000), STRENGTH (default 1.0)

using KrakeNN, KernelAbstractions, Metal

T = Float32
backend = MetalBackend()
DeviceArray = Metal.MtlArray

const Lx, Ly = 1.0f0, 0.5f0
const cx_p, cy_p = 0.5f0, 0.245f0
const R_p = 0.025f0
const Re_target = 100.0f0
const u_max = 0.04f0
const u_mean = u_max

D_lu     = parse(Int,     get(ENV, "D_LU",     "20"))
steps    = parse(Int,     get(ENV, "STEPS",    "5000"))
strength = parse(Float64, get(ENV, "STRENGTH", "1.0"))

dx_ref = 2 * R_p / D_lu
Nx = round(Int, Lx / dx_ref) + 1
Ny = round(Int, Ly / dx_ref) + 1
ν = T(u_mean * D_lu / Re_target)

println("=== Smoke (D) — cylinder-focused local Metal Float32 ===")
println("D_lu=$D_lu  Nx=$Nx  Ny=$Ny  strength=$strength  steps=$steps")

mesh = cylinder_focused_mesh(; x_min=0.0, x_max=Float64(Lx),
                               y_min=0.0, y_max=Float64(Ly),
                               Nx=Nx, Ny=Ny,
                               cx=Float64(cx_p), cy=Float64(cy_p),
                               strength=strength, FT=T)
println("mesh: dx_ref=$(mesh.dx_ref)  (Cartesian equivalent would be $(T(dx_ref)))")
stretch_ratio = T(dx_ref) / mesh.dx_ref
println("stretch ratio (dx_nominal / dx_min) = $(round(Float64(stretch_ratio), digits=2))")

geom_h = build_slbm_geometry(mesh; local_cfl=true)
geom = transfer_slbm_geometry(geom_h, backend)

# Local-τ fields (quadratic scaling): τ_local = τ_global * (dx_local/dx_ref)^2
sp_h, sm_h = compute_local_omega_2d(mesh; ν=Float64(ν), scaling=:quadratic, τ_floor=0.51)
sp_field = DeviceArray(T.(sp_h)); sm_field = DeviceArray(T.(sm_h))
n_unstable = sum(sp_h .> 1.99) + sum(sm_h .> 1.99)
println("local ω: sp ∈ [$(round(minimum(sp_h),digits=3)), $(round(maximum(sp_h),digits=3))], unstable=$n_unstable")

is_solid_h = zeros(Bool, mesh.Nξ, mesh.Nη)
for j in 1:mesh.Nη, i in 1:mesh.Nξ
    x = mesh.X[i,j]; y = mesh.Y[i,j]
    (x-cx_p)^2 + (y-cy_p)^2 ≤ R_p^2 && (is_solid_h[i,j] = true)
end
println("solid cells: $(sum(is_solid_h))")

qw_h, uwx_h, uwy_h = precompute_q_wall_slbm_cylinder_2d(mesh, is_solid_h, Float64(cx_p),
                                                         Float64(cy_p), Float64(R_p); FT=T)
is_solid = DeviceArray(is_solid_h); q_wall = DeviceArray(qw_h)
uw_x = DeviceArray(uwx_h); uw_y = DeviceArray(uwy_h)

u_prof = DeviceArray(fill(T(u_max), mesh.Nη))
bcspec = BCSpec2D(; west=ZouHeVelocity(u_prof), east=ZouHePressure(one(T)),
                    south=HalfwayBB(), north=HalfwayBB())

f_h = zeros(T, mesh.Nξ, mesh.Nη, 9)
for j in 1:mesh.Nη, i in 1:mesh.Nξ, q in 1:9
    u = is_solid_h[i,j] ? T(0) : T(u_max)
    f_h[i,j,q] = T(KrakeNN.equilibrium(D2Q9(), one(T), u, zero(T), q))
end
fa = DeviceArray(f_h); fb = similar(fa); fill!(fb, zero(T))
ρ = KernelAbstractions.allocate(backend, T, mesh.Nξ, mesh.Nη); fill!(ρ, one(T))
ux = KernelAbstractions.allocate(backend, T, mesh.Nξ, mesh.Nη); fill!(ux, zero(T))
uy = KernelAbstractions.allocate(backend, T, mesh.Nξ, mesh.Nη); fill!(uy, zero(T))

function _run_loop!(fa, fb, ρ, ux, uy, is_solid, q_wall, uw_x, uw_y,
                     geom, sp, sm, ν, bcspec, mesh, steps, sample_every, sample_window,
                     norm)
    cd_h = Float64[]; cl_h = Float64[]
    for step in 1:steps
        slbm_trt_libb_step_local_2d!(fb, fa, ρ, ux, uy, is_solid, q_wall, uw_x, uw_y,
                                       geom, sp, sm)
        apply_bc_rebuild_2d!(fb, fa, bcspec, ν, mesh.Nξ, mesh.Nη;
                              sp_field=sp, sm_field=sm)
        if step > steps - sample_window && step % sample_every == 0
            Fx, Fy = compute_drag_libb_mei_2d(fb, q_wall, uw_x, uw_y, mesh.Nξ, mesh.Nη)
            push!(cd_h, 2 * Float64(Fx) / norm)
            push!(cl_h, 2 * Float64(Fy) / norm)
        end
        fa, fb = fb, fa
        if step % 200 == 0
            KernelAbstractions.synchronize(get_backend(fa))
            rho_h = Array(ρ)
            if any(isnan, rho_h)
                error("NaN detected at step $step")
            end
            println("  step $step  ρ ∈ [$(round(minimum(rho_h), digits=4)), $(round(maximum(rho_h), digits=4))]")
        end
    end
    return fa, cd_h, cl_h
end

norm = u_mean^2 * (R_p * 2)
sample_every = 10
sample_window = steps ÷ 2

t0 = time()
fa, cd_h, cl_h = _run_loop!(fa, fb, ρ, ux, uy, is_solid, q_wall, uw_x, uw_y,
                             geom, sp_field, sm_field, ν, bcspec, mesh, steps,
                             sample_every, sample_window, norm)
KernelAbstractions.synchronize(backend)
elapsed = time() - t0

cd_mean = sum(cd_h) / length(cd_h)
cl_mean = sum(cl_h) / length(cl_h)
cl_rms = sqrt(sum((cl_h .- cl_mean).^2) / length(cl_h))
cells = mesh.Nξ * mesh.Nη
mlups = cells * steps / elapsed / 1e6

println("\n=== Smoke (D) results (strength=$strength, D_lu=$D_lu, $steps steps) ===")
println("  Cd_mean  = $(round(cd_mean, digits=4))")
println("  Cl_RMS   = $(round(cl_rms,  digits=4))")
println("  elapsed  = $(round(elapsed, digits=1))s   $(round(mlups, digits=0)) MLUPS")
println("  cells    = $cells   dx_min=$(mesh.dx_ref)")
println("PASS (no NaN).")
