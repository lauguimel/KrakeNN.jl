using Test
using KrakeNN

@testset "Poiseuille 2D body force" begin
    Nx, Ny = 4, 32
    ν = 0.1
    Fx = 1e-5

    result = run_poiseuille_2d(; Nx=Nx, Ny=Ny, ν=ν, Fx=Fx, max_steps=10000)

    # Analytical parabolic profile (half-way bounce-back: walls at y=0.5, y=Ny+0.5)
    u_analytical = [Fx / (2ν) * (j - 0.5) * (Ny + 0.5 - j) for j in 1:Ny]
    u_numerical = result.ux[2, :]  # any x slice (periodic)

    # Interior points only (skip wall nodes j=1 and j=Ny)
    u_max = maximum(u_analytical)
    errors = abs.(u_numerical[2:end-1] .- u_analytical[2:end-1])
    max_rel_err = maximum(errors) / u_max

    @test max_rel_err < 0.02  # 2% L∞ relative error
    @info "Poiseuille 2D: L∞ relative error = $(round(max_rel_err, digits=5))"

    # Mass conservation
    @test abs(sum(result.ρ) - Nx * Ny) / (Nx * Ny) < 0.001
end
