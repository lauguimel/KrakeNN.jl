using KrakeNN

Nx_single = 10
Nx_A, Nx_B = 6, 5
Ny = 5
Ly = Float64(Ny - 1); Lx_A = Float64(Nx_A - 1); Lx_single = Float64(Nx_single - 1)
ν = 0.1; ω = 1.0 / (3ν + 0.5); u0 = 0.05

function init_eq(Nx, Ny, ρ0, ux0, uy0)
    f = zeros(Float64, Nx, Ny, 9)
    usq = ux0 * ux0 + uy0 * uy0
    @inbounds for j in 1:Ny, i in 1:Nx, q in 1:9
        f[i, j, q] = KrakeNN.feq_2d(Val(q), ρ0, ux0, uy0, usq)
    end
    return f
end

fref_in = init_eq(Nx_single, Ny, 1.0, u0, 0.0)
fref_out = similar(fref_in)
ρref = ones(Nx_single, Ny); uxref = fill(u0, Nx_single, Ny); uyref = zeros(Nx_single, Ny)
is_solid_ref = zeros(Bool, Nx_single, Ny)
fused_bgk_step!(fref_out, fref_in, ρref, uxref, uyref, is_solid_ref, Nx_single, Ny, ω)

println("Δ in single-block (should be 0 for uniform eq init):")
println("  max abs diff f_out vs f_in: ", maximum(abs.(fref_out .- fref_in)))
println("  f_in[3,3,1]=$(fref_in[3,3,1]) vs f_out[3,3,1]=$(fref_out[3,3,1])")
println("  f_in[1,3,2]=$(fref_in[1,3,2]) vs f_out[1,3,2]=$(fref_out[1,3,2])  (west wall)")
println("  f_in[10,3,4]=$(fref_in[10,3,4]) vs f_out[10,3,4]=$(fref_out[10,3,4])  (east wall)")
println()

# Single-step A only
fA_in = init_eq(Nx_A, Ny, 1.0, u0, 0.0)
fA_out = similar(fA_in)
ρA = ones(Nx_A, Ny); uxA = fill(u0, Nx_A, Ny); uyA = zeros(Nx_A, Ny)
is_solid_A = zeros(Bool, Nx_A, Ny)
fused_bgk_step!(fA_out, fA_in, ρA, uxA, uyA, is_solid_A, Nx_A, Ny, ω)

println("Δ in block A (should only differ from single-block at east edge):")
for i in 1:Nx_A
    err = maximum(abs.(fA_out[i, :, :] .- fref_out[i, :, :]))
    println("  col i=$i  max|diff| = $err")
end

# Single-step B
fB_in = init_eq(Nx_B, Ny, 1.0, u0, 0.0)
fB_out = similar(fB_in)
ρB = ones(Nx_B, Ny); uxB = fill(u0, Nx_B, Ny); uyB = zeros(Nx_B, Ny)
is_solid_B = zeros(Bool, Nx_B, Ny)
fused_bgk_step!(fB_out, fB_in, ρB, uxB, uyB, is_solid_B, Nx_B, Ny, ω)
println("\nΔ in block B (B's i=1..5 = single's i=6..10):")
for i in 1:Nx_B
    err = maximum(abs.(fB_out[i, :, :] .- fref_out[i + Nx_A - 1, :, :]))
    println("  B i=$i (single i=$(i + Nx_A - 1))  max|diff| = $err")
end

# Build mbm + exchange
mesh_A = cartesian_mesh(; x_min=0.0, x_max=Float64(Nx_A-1), y_min=0.0, y_max=Ly, Nx=Nx_A, Ny=Ny)
mesh_B = cartesian_mesh(; x_min=Float64(Nx_A-1), x_max=Float64(Nx_single-1), y_min=0.0, y_max=Ly, Nx=Nx_B, Ny=Ny)
blk_A = Block(:A, mesh_A; west=:inlet, east=:interface, south=:wall, north=:wall)
blk_B = Block(:B, mesh_B; west=:interface, east=:outlet, south=:wall, north=:wall)
iface = Interface(; from=(:A, :east), to=(:B, :west))
mbm = MultiBlockMesh2D([blk_A, blk_B]; interfaces=[iface])
exchange_ghost_2d!(mbm, [fA_out, fB_out])
println("\nAfter exchange:")
for q in 1:9
    err_A_col6 = maximum(abs.(fA_out[Nx_A, :, q] .- fref_out[Nx_A, :, q]))
    err_B_col1 = maximum(abs.(fB_out[1, :, q] .- fref_out[Nx_A, :, q]))
    println("  q=$q  A[6] err=$err_A_col6  B[1] err=$err_B_col1")
end
