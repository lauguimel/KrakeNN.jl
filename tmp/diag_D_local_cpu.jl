# Diagnose (D) cylinder_focused NaN locally on CPU F64.
# Replicate the Aqua driver pipeline at small D to see where NaN appears.

using KrakeNN, KernelAbstractions

const T = Float64
const backend = KernelAbstractions.CPU()
const Lx, Ly = 1.0, 0.5
const cx_p, cy_p = 0.5, 0.245
const R_p = 0.025
const u_max = 0.04
const Re = 100.0

D_lu = 20                    # small test
steps = 2000
strength = 1.0                # start with moderate clustering

dx_ref = 2 * R_p / D_lu
Nx = round(Int, Lx / dx_ref) + 1
Ny = round(Int, Ly / dx_ref) + 1
R_lu = R_p / dx_ref
ν = u_max * D_lu / Re

println("=== (D) local diagnosis ===")
println("D_lu=$D_lu  Nx=$Nx  Ny=$Ny  dx_ref=$dx_ref")
println("R_lu=$R_lu  ν=$(round(ν, sigdigits=4))  Re=$Re  τ_ref=$(round(3ν+0.5, sigdigits=4))")
println("strength=$strength")

mesh = cylinder_focused_mesh(; x_min=0.0, x_max=Lx, y_min=0.0, y_max=Ly,
                                Nx=Nx, Ny=Ny, cx=cx_p, cy=cy_p,
                                strength=strength, FT=T)
println("  mesh dx_ref=$(round(mesh.dx_ref, sigdigits=4))  (smallest cell)")

local_cfl = parse(Bool, get(ENV, "LOCAL_CFL", "true"))
scaling = Symbol(get(ENV, "SCALING", "quadratic"))   # :quadratic, :none, :linear
println("  local_cfl=$local_cfl  scaling=$scaling")
geom = build_slbm_geometry(mesh; local_cfl=local_cfl)

τ_floor_test = parse(Float64, get(ENV, "TAU_FLOOR", "0.55"))
println("  τ_floor = $τ_floor_test")
sp, sm = compute_local_omega_2d(mesh; ν=Float64(ν), scaling=scaling, τ_floor=τ_floor_test)
τ_p = 1 ./ sp; τ_m = 1 ./ sm
println("  τ_plus range: $(round(minimum(τ_p), sigdigits=4)) .. $(round(maximum(τ_p), sigdigits=4))")
println("  τ_minus range: $(round(minimum(τ_m), sigdigits=4)) .. $(round(maximum(τ_m), sigdigits=4))")
println("  $(count(x -> x <= 0.511, τ_p)) cells hit τ_floor")

is_solid = zeros(Bool, Nx, Ny)
for j in 1:Ny, i in 1:Nx
    x = mesh.X[i, j]; y = mesh.Y[i, j]
    (x - cx_p)^2 + (y - cy_p)^2 ≤ R_p^2 && (is_solid[i, j] = true)
end
println("  solid cells: $(sum(is_solid))")

q_wall, uwx, uwy = precompute_q_wall_slbm_cylinder_2d(mesh, is_solid, cx_p, cy_p, R_p; FT=T)
println("  q_wall cut-link cells: $(count(any.(q_wall[i,j,:] .> 0 for i in 1:Nx, j in 1:Ny)))")

# Init
u_prof = fill(T(u_max), Ny)
f_in = zeros(T, Nx, Ny, 9)
for j in 1:Ny, i in 1:Nx, q in 1:9
    u = is_solid[i, j] ? 0.0 : Float64(u_max)
    f_in[i, j, q] = T(KrakeNN.equilibrium(D2Q9(), 1.0, u, 0.0, q))
end
f_out = similar(f_in); fill!(f_out, 0)
ρ = ones(T, Nx, Ny); ux = zeros(T, Nx, Ny); uy = zeros(T, Nx, Ny)

bcspec = BCSpec2D(; west=ZouHeVelocity(u_prof), east=ZouHePressure(one(T)),
                    south=HalfwayBB(), north=HalfwayBB())

function _run_sim!(f_in, f_out, ρ, ux, uy, is_solid, q_wall, uwx, uwy, geom, sp, sm, bcspec, ν, Nx, Ny, mesh, τ_p, τ_m, steps)
    for step in 1:steps
        slbm_trt_libb_step_local_2d!(f_out, f_in, ρ, ux, uy, is_solid,
                                       q_wall, uwx, uwy, geom, sp, sm)
        apply_bc_rebuild_2d!(f_out, f_in, bcspec, ν, Nx, Ny;
                              sp_field=sp, sm_field=sm)
        if any(isnan, ρ) || any(isnan, ux) || any(isnan, uy) || any(isnan, f_out)
            println("  NaN at step $step")
            for j in 1:Ny, i in 1:Nx
                if isnan(ρ[i, j]) || isnan(ux[i, j]) || isnan(uy[i, j])
                    println("    NaN at (i=$i, j=$j)  x=$(round(mesh.X[i,j], digits=4))  y=$(round(mesh.Y[i,j], digits=4))")
                    println("      τ_p_local=$(round(τ_p[i,j], sigdigits=4))  τ_m_local=$(round(τ_m[i,j], sigdigits=4))  is_solid=$(is_solid[i,j])")
                    break
                end
            end
            return step, f_in, f_out
        end
        if step in (10, 100, 500, 1000, 2000)
            println("  step $step: ρ ∈ [$(round(minimum(ρ), digits=4)), $(round(maximum(ρ), digits=4))]  u_max=$(round(maximum(sqrt.(ux.^2 .+ uy.^2)), digits=4))")
        end
        f_in, f_out = f_out, f_in
    end
    return 0, f_in, f_out
end

println("\nStarting $steps steps …")
nan_step, f_in, f_out = _run_sim!(f_in, f_out, ρ, ux, uy, is_solid, q_wall, uwx, uwy, geom, sp, sm, bcspec, ν, Nx, Ny, mesh, τ_p, τ_m, steps)
println(nan_step == 0 ? "\n✅ $steps steps no NaN" : "\n❌ NaN at step $nan_step")
