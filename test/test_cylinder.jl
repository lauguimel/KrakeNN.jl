using Test
using KrakeNN

@testset "Cylinder 2D Re=20" begin
    # Flow around cylinder in confined channel (blockage D/H = 25%)
    # Schäfer & Turek benchmark: Cd ≈ 5.57 for similar confinement at Re=20
    Nx, Ny = 300, 80
    radius = 10
    u_in = 0.04
    Re = 20.0
    D = 2 * radius
    ν = u_in * D / Re

    result = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=radius, u_in=u_in, ν=ν,
                              max_steps=50000, avg_window=5000)

    @test !any(isnan, result.ρ)

    # Cd in range [3, 8] — confined channel at Re=20 (Schäfer-Turek ~5.57)
    @test result.Cd > 3.0 && result.Cd < 8.0

    # Symmetric flow (low lift): |Fy/Fx| < 0.05
    @test abs(result.Fy / result.Fx) < 0.05

    # Density close to 1 (low compressibility error)
    @test maximum(result.ρ) < 1.1
    @test minimum(result.ρ) > 0.9

    @info "Cylinder Re=20: Cd = $(round(result.Cd, digits=2)), lift ratio = $(round(abs(result.Fy/result.Fx), digits=4))"
end
