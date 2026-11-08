using Test
using KrakeNN

# ==========================================================================
# Schäfer-Turek-style cylinder with LI-BB V2 (Bouzidi pre-phase + TRT)
# and Zou-He velocity inlet / pressure outlet reconstructed pre-collision
# (`rebuild_inlet_outlet_libb_2d!`) to bypass the kernel's halfway-BB
# fallback corruption at the non-wall boundaries.
#
# With the BC rebuild + the trt_rates label fix (viscosity was 13× too
# high before) the flow field is physically correct and the MEA drag
# matches the BGK baseline within a few percent across Re = 5..40.
# ==========================================================================

@testset "Cylinder LI-BB V2 — Schäfer-Turek scaffold" begin
    Nx, Ny = 300, 80
    radius = 10
    u_in = 0.04
    Re = 20.0
    D = 2 * radius
    ν = u_in * D / Re

    result = run_cylinder_libb_2d(; Nx=Nx, Ny=Ny, radius=radius,
                                    cx=Nx÷4, cy=Ny÷2,
                                    u_in=u_in, ν=ν, inlet=:uniform,
                                    max_steps=20_000, avg_window=5_000)

    @test !any(isnan, result.ρ)
    @test !any(isnan, result.ux)
    # Flow symmetric about the cylinder axis (small lift ratio).
    @test abs(result.Fy / result.Fx) < 0.05
    # Density bounded at Ma ≈ 0.07.
    @test maximum(result.ρ) < 1.3
    @test minimum(result.ρ) > 0.8
    # Velocity accelerates correctly through the gap (blockage 25 %).
    # Mass conservation at steady state: u_avg in the gap ≈ u_in · Ny / (Ny - D).
    cx = Nx ÷ 4
    gap = [result.ux[cx, j] for j in 1:Ny if !result.is_solid[cx, j]]
    u_gap_mean = sum(gap) / length(gap)
    u_gap_expected = u_in * Ny / (Ny - D)
    @test 0.7 * u_gap_expected < u_gap_mean < 1.3 * u_gap_expected
    # Drag: Cd for uniform inflow Re=20, D/H=25 % is ≈ 5 (cf. Sen,
    # Mittal, Biswas 2009 for confined cylinders). BGK baseline gives
    # 5.3 on the same geometry; V2 + LI-BB gives ≈ 5 (0.94 × BGK).
    @test 3.0 < result.Cd < 8.0

    @info "Cylinder LI-BB Re=20 (scaffold)" Cd=result.Cd u_gap=u_gap_mean lift_ratio=abs(result.Fy/result.Fx) maxρm1=maximum(abs.(result.ρ .- 1))
end
