# Investigate the WP-MESH-5 anomaly: Cl_RMS = 0.59 % (B) vs 0.01 % (C)
# at D=80 on the same uniform mesh. Test:
#   1. Are the q_w identical between Cart precompute and SLBM precompute?
#   2. If we feed the SAME q_w to both kernels, does the gap persist?
# Run on Metal D=40 (faster than D=80 but still in the converged regime).

using KrakeNN, Metal, KernelAbstractions, Gmsh

const Lx, Ly = 2.2f0, 0.41f0
const cx_p, cy_p, R_p = 0.2f0, 0.2f0, 0.05f0
const D_lu, Re = 40, 100
const u_max = 0.04f0
const u_mean = (2/3) * u_max
T = Float32

backend = MetalBackend()

dx_ref = T(2 * R_p / D_lu)
Nx = round(Int, Lx / dx_ref) + 1
Ny = round(Int, Ly / dx_ref) + 1
cx_lu = T(cx_p / dx_ref); cy_lu = T(cy_p / dx_ref); R_lu = T(R_p / dx_ref)
ν = T(u_mean * D_lu / Re)
println("D=$D_lu, Nx×Ny=$(Nx)×$(Ny), ν=$ν")

# (B) q_w via Cartesian precompute (lattice units)
qw_B, is_solid_B = precompute_q_wall_cylinder(Nx, Ny, cx_lu, cy_lu, R_lu; FT=T)

# (C) q_w via SLBM curvilinear precompute (physical units, Cartesian mesh)
mesh_C = cartesian_mesh(; x_min=0.0, x_max=Float64(Lx), y_min=0.0, y_max=Float64(Ly),
                          Nx=Nx, Ny=Ny, FT=T)
is_solid_C_h = zeros(Bool, Nx, Ny)
for j in 1:Ny, i in 1:Nx
    x = mesh_C.X[i,j]; y = mesh_C.Y[i,j]
    (x - cx_p)^2 + (y - cy_p)^2 ≤ R_p^2 && (is_solid_C_h[i,j] = true)
end
qw_C, uwx_C, uwy_C = precompute_q_wall_slbm_cylinder_2d(mesh_C, is_solid_C_h, cx_p, cy_p, R_p; FT=T)

println("\n=== q_w comparison ===")
println("max|qw_B − qw_C|             = ", maximum(abs.(qw_B .- qw_C)))
println("nnz(qw_B)                    = ", count(qw_B .> 0))
println("nnz(qw_C)                    = ", count(qw_C .> 0))
println("max|is_solid_B − is_solid_C| = ", maximum(Int.(is_solid_B) .- Int.(is_solid_C_h)))

# Indices where they differ
diff_qw = abs.(qw_B .- qw_C)
n_diff = count(diff_qw .> 1e-6)
println("nodes with |Δq_w| > 1e-6     = $n_diff / $(length(qw_B))")
if n_diff > 0
    iqs = findall(diff_qw .> 1e-6)
    for k in 1:min(5, length(iqs))
        i, j, q = Tuple(iqs[k])
        println("  (i=$i, j=$j, q=$q):  qw_B=$(qw_B[i,j,q])  qw_C=$(qw_C[i,j,q])  Δ=$(diff_qw[i,j,q])")
    end
end

println("\n=== is_solid comparison ===")
diff_is = is_solid_B .⊻ is_solid_C_h
println("nodes where is_solid differs = $(count(diff_is)) / $(length(is_solid_B))")
if count(diff_is) > 0
    iqs = findall(diff_is)
    for k in 1:min(5, length(iqs))
        i, j = Tuple(iqs[k])
        # in lattice (precompute_q_wall_cylinder), test is (i-1, j-1) vs centre lattice
        x_lu = i - 1.0; y_lu = j - 1.0
        d_lu = (x_lu - Float64(cx_lu))^2 + (y_lu - Float64(cy_lu))^2
        x_phys = mesh_C.X[i,j]; y_phys = mesh_C.Y[i,j]
        d_phys = (x_phys - cx_p)^2 + (y_phys - cy_p)^2
        println("  (i=$i, j=$j) is_solid_B=$(is_solid_B[i,j]) is_solid_C=$(is_solid_C_h[i,j])  d_lu²=$d_lu (R²=$(Float64(R_lu)^2))  d_phys²=$d_phys (R²=$(R_p^2))")
    end
end
