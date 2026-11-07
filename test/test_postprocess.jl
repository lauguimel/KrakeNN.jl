using Test

@testset "Postprocess helpers" begin

    # --- Helper: build a mock result NamedTuple ---
    function make_result(; Nx=8, Ny=8, Lx=1.0, Ly=1.0,
                          ux=nothing, uy=nothing, ρ=nothing,
                          user_vars=Dict{Symbol,Any}())
        ux = isnothing(ux) ? zeros(Nx, Ny) : ux
        uy = isnothing(uy) ? zeros(Nx, Ny) : uy
        ρ  = isnothing(ρ)  ? ones(Nx, Ny)  : ρ
        domain = (Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
        setup = (domain=domain, user_vars=user_vars)
        return (ux=ux, uy=uy, ρ=ρ, setup=setup)
    end

    @testset "extract_line along y" begin
        Nx, Ny = 8, 10
        Lx, Ly = 2.0, 5.0
        dx, dy = Lx / Nx, Ly / Ny
        ux = zeros(Nx, Ny)
        # Linear profile along y at each x
        for j in 1:Ny, i in 1:Nx
            ux[i, j] = Float64(j)
        end
        result = make_result(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, ux=ux)

        # Extract along y at x ≈ midpoint (at=0.5 => x_pos = 0.5*2.0 = 1.0)
        prof = KrakeNN.extract_line(result, :ux, :y; at=0.5)
        @test prof.axis == :y
        @test length(prof.values) == Ny
        @test length(prof.coord) == Ny
        # coord should be cell centers: (j-0.5)*dy
        @test prof.coord[1] ≈ 0.5 * dy
        @test prof.coord[end] ≈ (Ny - 0.5) * dy
        # Values should be 1:Ny at the extracted x column
        i_expected = clamp(round(Int, 1.0 / dx + 0.5), 1, Nx)
        @test prof.values == ux[i_expected, :]
    end

    @testset "extract_line along x" begin
        Nx, Ny = 6, 4
        Lx, Ly = 3.0, 2.0
        dx, dy = Lx / Nx, Ly / Ny
        uy = zeros(Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            uy[i, j] = Float64(i) * 10.0
        end
        result = make_result(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, uy=uy)

        prof = KrakeNN.extract_line(result, :uy, :x; at=0.5)
        @test prof.axis == :x
        @test length(prof.values) == Nx
        j_expected = clamp(round(Int, (0.5 * Ly) / dy + 0.5), 1, Ny)
        @test prof.values == uy[:, j_expected]
    end

    @testset "extract_line invalid axis" begin
        result = make_result()
        @test_throws ArgumentError KrakeNN.extract_line(result, :ux, :z)
    end

    @testset "field_error with function — L2" begin
        Nx, Ny = 10, 10
        Lx, Ly = 1.0, 1.0
        dx, dy = Lx / Nx, Ly / Ny
        # ux = exact analytical => error should be 0
        ux = zeros(Nx, Ny)
        analytical(x, y) = 3.0 * x + 2.0 * y
        for j in 1:Ny, i in 1:Nx
            ux[i, j] = analytical((i - 0.5) * dx, (j - 0.5) * dy)
        end
        result = make_result(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, ux=ux)
        err = KrakeNN.field_error(result, :ux, analytical; norm=:L2)
        @test err.error < 1e-14
        @test err.norm == :L2
    end

    @testset "field_error with function — Linf" begin
        Nx, Ny = 8, 8
        Lx, Ly = 1.0, 1.0
        dx, dy = Lx / Nx, Ly / Ny
        # Add a known offset
        offset = 0.01
        analytical(x, y) = x * y
        ux = zeros(Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            ux[i, j] = analytical((i - 0.5) * dx, (j - 0.5) * dy) + offset
        end
        result = make_result(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, ux=ux)
        err = KrakeNN.field_error(result, :ux, analytical; norm=:Linf)
        ref_max = maximum(abs.(ux .- offset))
        @test err.error ≈ offset / ref_max
    end

    @testset "field_error with function — L1" begin
        Nx, Ny = 4, 4
        ux = ones(Nx, Ny) * 2.0
        result = make_result(Nx=Nx, Ny=Ny, ux=ux)
        # Analytical = 2.0 everywhere => error = 0
        err = KrakeNN.field_error(result, :ux, (x, y) -> 2.0; norm=:L1)
        @test err.error < 1e-14
    end

    @testset "probe" begin
        Nx, Ny = 10, 10
        Lx, Ly = 2.0, 2.0
        dx, dy = Lx / Nx, Ly / Ny
        ρ = zeros(Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            ρ[i, j] = 100.0 * i + j
        end
        result = make_result(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, ρ=ρ)

        # Probe at cell center (i=5, j=3) => x=(5-0.5)*0.2=0.9, y=(3-0.5)*0.2=0.5
        val = KrakeNN.probe(result, :ρ, 0.9, 0.5)
        @test val == ρ[5, 3]

        # Probe at domain corner
        val_corner = KrakeNN.probe(result, :ρ, 0.0, 0.0)
        @test val_corner == ρ[1, 1]

        # Probe beyond domain (clamped)
        val_far = KrakeNN.probe(result, :ρ, 100.0, 100.0)
        @test val_far == ρ[Nx, Ny]
    end

    @testset "domain_stats" begin
        Nx, Ny = 4, 4
        ux = ones(Nx, Ny) * 0.1
        uy = ones(Nx, Ny) * 0.2
        ρ  = ones(Nx, Ny) * 1.001
        result = make_result(Nx=Nx, Ny=Ny, ux=ux, uy=uy, ρ=ρ)

        stats = KrakeNN.domain_stats(result)
        @test stats.max_ux ≈ 0.1
        @test stats.max_uy ≈ 0.2
        @test stats.max_u  ≈ sqrt(0.1^2 + 0.2^2)
        @test stats.mean_rho ≈ 1.001
        @test stats.mass_error ≈ 0.001
    end

    @testset "domain_stats zero velocity" begin
        result = make_result()
        stats = KrakeNN.domain_stats(result)
        @test stats.max_u == 0.0
        @test stats.mean_rho ≈ 1.0
        @test stats.mass_error ≈ 0.0
    end

end
