# =============================================================================
# Schäfer-Turek 2D-1 benchmark (steady cylinder, Re=20, parabolic inflow)
#
# Reference (Schäfer & Turek 1996, Notes Num. Fluid Mech. 52):
#   Cd = 5.57 .. 5.59
#   Cl = 0.0104 .. 0.0110
#   ΔP = 0.1172 .. 0.1176
#
# Geometry (physical units):
#   Domain [0, 2.2] × [0, 0.41], cylinder D=0.1, center (0.2, 0.2).
#   Inlet u(y) = 4 U y (H-y)/H², U_max = 0.3 m/s (U_mean = 0.2 m/s).
#   Re = U_mean D / ν = 20, so ν = 0.001 m²/s.
#
# Lattice mapping (D = N_D lattice units):
#   L_x = 22 N_D, L_y = (0.41/0.1) N_D ≈ 4.1 N_D
#   cx = 2 N_D, cy = 2 N_D (asymmetric, ST convention)
#   U_max_lu = 0.06 (Ma ≈ 0.1 — acceptable); ν_lu = U_mean_lu D_lu / Re
#
# Refinement scan: D = 20, 40, 80 (MLUPS and Cd). Larger grids stream to
# Aqua (CUDA) via the campaign skill; this script is for the Metal local
# pre-flight.
# =============================================================================

using Pkg; Pkg.activate(dirname(@__DIR__))
using KrakeNN
using Metal, KernelAbstractions

backend = Metal.MetalBackend()
T = Float32

println("=" ^ 78)
println("Schäfer-Turek 2D-1: parabolic Re=20, asymmetric cylinder (D/H ≈ 0.244)")
println("Reference Cd = 5.57-5.59  (Schäfer & Turek 1996)")
println("=" ^ 78)
println()
println("| D_lu | Nx × Ny     | Cd      | ΔCd/Cd_ref | t (s) | MLUPS |")
println("|------|-------------|---------|------------|-------|-------|")

Cd_ref = 5.58

for D_lu in (20, 40, 80)
    # Physical→lattice factor: Δx_phys = 0.1 / D_lu
    H_phys = 0.41; L_phys = 2.2; D_phys = 0.1
    Ny = Int(round(H_phys / D_phys * D_lu))   # ≈ 4.1 × D_lu
    Nx = Int(round(L_phys / D_phys * D_lu))   # = 22 × D_lu
    # Cylinder position: (0.2, 0.2) in physical → (2·D_lu, 2·D_lu) lattice
    cx = 2 * D_lu
    cy = 2 * D_lu
    radius = D_lu ÷ 2

    u_max = 0.06                              # centerline, Ma≈0.1
    u_mean = (2/3) * u_max                    # 0.04
    ν = u_mean * D_lu / 20.0                  # Re = 20

    max_steps = D_lu == 20 ? 40_000 :
                D_lu == 40 ? 60_000 : 100_000
    avg_window = max_steps ÷ 4

    t0 = time()
    res = run_cylinder_libb_2d(; Nx=Nx, Ny=Ny, radius=radius,
                                 cx=cx, cy=cy, u_in=u_max, ν=ν,
                                 inlet=:parabolic, ρ_out=1.0,
                                 max_steps=max_steps, avg_window=avg_window,
                                 backend=backend, T=T)
    dt = time() - t0
    mlups = max_steps * Nx * Ny / dt / 1e6
    rel_err = (res.Cd - Cd_ref) / Cd_ref * 100
    println("| $(lpad(D_lu,4)) | $(lpad("$(Nx)×$(Ny)",11)) | $(lpad(round(res.Cd, digits=3),7)) | $(lpad(round(rel_err, digits=2),9)) % | $(lpad(round(dt, digits=1),5)) | $(lpad(round(mlups, digits=1),5)) |")
end
