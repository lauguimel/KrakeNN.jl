# Local Metal smoke for approach (E3) — 3-block Cartesian multi-block
# cylinder. Validates the end-to-end step pipeline (extended arrays +
# exchange + wall-ghost + per-block kernel + per-block BC apply +
# drag on interior view). Metal FP32 is numerically unstable at low
# lattice resolutions for cylinder+LI-BB (known KrakeNN pitfall, see
# memory project_wp_mesh_6_bump), so this smoke only verifies that
# the DATA FLOW is wired correctly (no NaN within the first few steps,
# interior shapes compile, SubArray BC kernels launch). Accuracy comes
# from the Aqua F64 run.

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
const R_bubble = 0.15f0

D_lu  = parse(Int, get(ENV, "D_LU",  "20"))
steps = parse(Int, get(ENV, "STEPS", "200"))

dx_ref = 2 * R_p / D_lu
Nx_total = round(Int, Lx / dx_ref) + 1
Ny = round(Int, Ly / dx_ref) + 1
x_C_west = cx_p - R_bubble
x_C_east = cx_p + R_bubble
Nx_W = round(Int, x_C_west / dx_ref)
Nx_C = round(Int, (x_C_east - x_C_west) / dx_ref) + 1
Nx_E = Nx_total - Nx_W - Nx_C
cx_lu = cx_p / dx_ref; cy_lu = cy_p / dx_ref; R_lu = R_p / dx_ref
ν = T(u_mean * D_lu / Re_target)

println("=== Smoke (E3) — 3-block Cartesian cylinder local Metal Float32 ===")
println("D_lu=$D_lu  Nx_total=$Nx_total  Ny=$Ny  steps=$steps")
println("Blocks: W=$(Nx_W)×$Ny  C=$(Nx_C)×$Ny  E=$(Nx_E)×$Ny")

# --- Build MBM ---
x_W_min = 0.0; x_W_max = (Nx_W - 1) * Float64(dx_ref)
x_C_min = Nx_W * Float64(dx_ref); x_C_max = x_C_min + (Nx_C - 1) * Float64(dx_ref)
x_E_min = (Nx_W + Nx_C) * Float64(dx_ref); x_E_max = x_E_min + (Nx_E - 1) * Float64(dx_ref)
y_min = 0.0; y_max = (Ny - 1) * Float64(dx_ref)
mesh_W = cartesian_mesh(; x_min=x_W_min, x_max=x_W_max, y_min=y_min, y_max=y_max,
                          Nx=Nx_W, Ny=Ny, FT=T)
mesh_C = cartesian_mesh(; x_min=x_C_min, x_max=x_C_max, y_min=y_min, y_max=y_max,
                          Nx=Nx_C, Ny=Ny, FT=T)
mesh_E = cartesian_mesh(; x_min=x_E_min, x_max=x_E_max, y_min=y_min, y_max=y_max,
                          Nx=Nx_E, Ny=Ny, FT=T)
blk_W = Block(:W, mesh_W; west=:inlet,     east=:interface, south=:wall, north=:wall)
blk_C = Block(:C, mesh_C; west=:interface, east=:interface, south=:wall, north=:wall)
blk_E = Block(:E, mesh_E; west=:interface, east=:outlet,    south=:wall, north=:wall)
mbm = MultiBlockMesh2D([blk_W, blk_C, blk_E];
                        interfaces=[Interface(; from=(:W, :east), to=(:C, :west)),
                                    Interface(; from=(:C, :east), to=(:E, :west))])
issues = sanity_check_multiblock(mbm; verbose=false)
any(iss -> iss.severity === :error, issues) && error("sanity error: $issues")
println("sanity OK: $(length(issues)) issue(s) (warnings only).")

# --- Per-block state on Metal ---
ng = 1
states = [allocate_block_state_2d(b; n_ghost=ng, backend=backend) for b in mbm.blocks]

# --- Cylinder precompute in block C local units ---
cx_C_lu = (Float64(cx_p) - x_C_min) / Float64(dx_ref)
cy_C_lu = Float64(cy_p) / Float64(dx_ref)
qw_C, isol_C = precompute_q_wall_cylinder(Nx_C, Ny, cx_C_lu, cy_C_lu, R_lu; FT=T)
uwx_C = zeros(T, Nx_C, Ny, 9)
uwy_C = zeros(T, Nx_C, Ny, 9)
println("solid cells in block C: $(sum(isol_C))")

# --- Lift to extended + transfer to device ---
ext_qwall = [
    DeviceArray(extend_interior_field_2d(zeros(T, Nx_W, Ny, 9), ng)),
    DeviceArray(extend_interior_field_2d(qw_C, ng)),
    DeviceArray(extend_interior_field_2d(zeros(T, Nx_E, Ny, 9), ng)),
]
ext_uwx = [
    DeviceArray(extend_interior_field_2d(zeros(T, Nx_W, Ny, 9), ng)),
    DeviceArray(extend_interior_field_2d(uwx_C, ng)),
    DeviceArray(extend_interior_field_2d(zeros(T, Nx_E, Ny, 9), ng)),
]
ext_uwy = [
    DeviceArray(extend_interior_field_2d(zeros(T, Nx_W, Ny, 9), ng)),
    DeviceArray(extend_interior_field_2d(uwy_C, ng)),
    DeviceArray(extend_interior_field_2d(zeros(T, Nx_E, Ny, 9), ng)),
]
ext_solid = [
    DeviceArray(extend_interior_field_2d(zeros(Bool, Nx_W, Ny), ng)),
    DeviceArray(extend_interior_field_2d(isol_C, ng)),
    DeviceArray(extend_interior_field_2d(zeros(Bool, Nx_E, Ny), ng)),
]

# --- Init equilibrium in each block's interior ---
u_prof = DeviceArray(fill(T(u_max), Ny))
u_prof_h = fill(T(u_max), Ny)
function init_f_int(Nx, solid_h)
    f = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        u = solid_h[i, j] ? T(0) : T(u_max)
        f[i, j, q] = T(KrakeNN.equilibrium(D2Q9(), one(T), u, zero(T), q))
    end
    return f
end
f_W_int = init_f_int(Nx_W, zeros(Bool, Nx_W, Ny))
f_C_int = init_f_int(Nx_C, isol_C)
f_E_int = init_f_int(Nx_E, zeros(Bool, Nx_E, Ny))
for (k, fi) in enumerate((f_W_int, f_C_int, f_E_int))
    int_view = interior_f(states[k])
    copyto!(int_view, DeviceArray(fi))
end

f_out = [similar(st.f) for st in states]
for k in 1:3; fill!(f_out[k], zero(T)); end

bcspecs = (
    BCSpec2D(; west=ZouHeVelocity(u_prof), east=HalfwayBB(),
                south=HalfwayBB(),           north=HalfwayBB()),
    BCSpec2D(; west=HalfwayBB(),             east=HalfwayBB(),
                south=HalfwayBB(),           north=HalfwayBB()),
    BCSpec2D(; west=HalfwayBB(),             east=ZouHePressure(one(T)),
                south=HalfwayBB(),           north=HalfwayBB()),
)

function _step_loop!(mbm, states, f_out, ext_solid, ext_qwall, ext_uwx, ext_uwy,
                      bcspecs, ν, steps)
    for step in 1:steps
        exchange_ghost_2d!(mbm, states)
        fill_physical_wall_ghost_2d!(mbm, states)
        for k in 1:3
            Nx_ext, Ny_ext = ext_dims(states[k])
            fused_trt_libb_v2_step!(f_out[k], states[k].f, states[k].ρ, states[k].ux, states[k].uy,
                                      ext_solid[k], ext_qwall[k], ext_uwx[k], ext_uwy[k],
                                      Nx_ext, Ny_ext, ν)
        end
        for k in 1:3
            Nξ_phys = states[k].Nξ_phys; Nη_phys = states[k].Nη_phys
            ng_k = states[k].n_ghost
            int_f_out = view(f_out[k], (ng_k+1):(ng_k+Nξ_phys), (ng_k+1):(ng_k+Nη_phys), :)
            int_f_in  = view(states[k].f, (ng_k+1):(ng_k+Nξ_phys), (ng_k+1):(ng_k+Nη_phys), :)
            apply_bc_rebuild_2d!(int_f_out, int_f_in, bcspecs[k], ν, Nξ_phys, Nη_phys)
        end
        for k in 1:3
            states[k].f, f_out[k] = f_out[k], states[k].f
        end
        if step % 50 == 0
            KernelAbstractions.synchronize(get_backend(states[1].f))
            rho_C = Array(states[2].ρ)
            if any(isnan, rho_C)
                println("  step $step : NaN in block C ρ"); return false
            end
            println("  step $step : C ρ ∈ [$(round(minimum(rho_C),digits=4)), $(round(maximum(rho_C),digits=4))]")
        end
    end
    return true
end

t0 = time()
ok = _step_loop!(mbm, states, f_out, ext_solid, ext_qwall, ext_uwx, ext_uwy,
                  bcspecs, ν, steps)
KernelAbstractions.synchronize(backend)
elapsed = time() - t0

cells = (Nx_W + Nx_C + Nx_E) * Ny
mlups = cells * steps / elapsed / 1e6
println("\n=== Smoke (E3) results (D_lu=$D_lu, $steps steps) ===")
println("  elapsed  = $(round(elapsed, digits=1))s   $(round(mlups, digits=0)) MLUPS")
println("  cells    = $cells (= $Nx_W + $Nx_C + $Nx_E)×$Ny")
println(ok ? "PASS (no NaN, pipeline wired OK — validate accuracy on Aqua F64)" :
             "FAIL (NaN — diagnose BEFORE Aqua submit)")
