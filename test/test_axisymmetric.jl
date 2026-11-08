using Test
using KrakeNN

@testset "Axisymmetric LBM" begin

    @testset "Hagen-Poiseuille pipe flow" begin
        # Pipe flow: u_z(r) = Fz/(4ν) * (R² - r²)
        # R = Nr - 0.5 (half-way BB, wall at j=Nr+0.5, axis at j=0.5)
        Nz, Nr = 4, 32
        ν = 0.1
        Fz = 1e-5

        result = run_hagen_poiseuille_2d(; Nz=Nz, Nr=Nr, ν=ν, Fz=Fz, max_steps=15000)

        R = Nr - 0.5  # pipe radius (half-way BB)
        # Analytical: u_z(j) = Fz/(4ν) * (R² - r²) where r = j - 0.5
        u_analytical = [Fz / (4ν) * (R^2 - (j - 0.5)^2) for j in 1:Nr]

        u_numerical = result.uz[2, :]  # any z-slice (periodic)

        # Compare interior points (skip wall at j=Nr)
        u_max = maximum(u_analytical)
        errors = abs.(u_numerical[1:end-1] .- u_analytical[1:end-1])
        max_rel_err = maximum(errors) / u_max

        @test !any(isnan, result.uz)

        # Specular axis BC + per-node Guo force correction → quantitative accuracy
        @test max_rel_err < 0.10  # 10% tolerance (3% typical at Nr=32)

        @info "Hagen-Poiseuille: L∞ error = $(round(max_rel_err*100, digits=1))%, u_max = $(round(maximum(u_numerical), sigdigits=4)) (analytical $(round(u_max, sigdigits=4)))"
    end
end
