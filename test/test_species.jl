using Test
using KrakeNN

@testset "Species transport" begin

    @testset "Diffusion 1D" begin
        # Same as heat conduction: C=1 at south, C=0 at north, periodic x
        Nx, Ny = 4, 32
        D_coeff = 0.1  # diffusivity
        ω_D = Float64(1.0 / (3.0 * D_coeff + 0.5))

        h_in  = zeros(Float64, Nx, Ny, 9)
        h_out = zeros(Float64, Nx, Ny, 9)
        C     = zeros(Float64, Nx, Ny)
        ux    = zeros(Float64, Nx, Ny)
        uy    = zeros(Float64, Nx, Ny)

        w = KrakeNN.weights(D2Q9())
        for j in 1:Ny, i in 1:Nx, q in 1:9
            h_in[i, j, q] = w[q] * 0.5
        end
        copy!(h_out, h_in)

        for step in 1:5000
            KrakeNN.stream_periodic_x_wall_y_2d!(h_out, h_in, Nx, Ny)
            apply_fixed_conc_south_2d!(h_out, 1.0, Nx)
            apply_fixed_conc_north_2d!(h_out, 0.0, Nx, Ny)
            collide_species_2d!(h_out, ux, uy, ω_D)
            h_in, h_out = h_out, h_in
        end

        compute_concentration_2d!(C, h_in)
        C_profile = C[2, :]
        C_analytical = [1.0 - (j - 0.5) / Ny for j in 1:Ny]

        err = maximum(abs.(C_profile .- C_analytical))
        @test err < 0.02
        @info "Species diffusion 1D: max error = $(round(err, digits=5))"
    end
end
