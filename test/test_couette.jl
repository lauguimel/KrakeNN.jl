using Test
using KrakeNN

@testset "Couette 2D" begin
    Nx, Ny = 4, 32
    ν = 0.1
    u_wall = 0.05

    result = run_couette_2d(; Nx=Nx, Ny=Ny, ν=ν, u_wall=u_wall, max_steps=30000)

    # Analytical linear profile (Zou-He on-node: walls at j=1 and j=Ny)
    # u(j) = u_wall * (1 - (j-1)/(Ny-1))
    H = Ny - 1
    u_analytical = [u_wall * (1 - (j - 1) / H) for j in 1:Ny]
    u_numerical = result.ux[2, :]

    # Interior points
    errors = abs.(u_numerical[2:end-1] .- u_analytical[2:end-1])
    max_rel_err = maximum(errors) / u_wall

    @test max_rel_err < 1e-10  # exact solution (machine precision)
    @info "Couette 2D: L∞ relative error = $(round(max_rel_err, digits=5))"

    # uy should be essentially zero
    @test maximum(abs.(result.uy)) < 1e-6
end
