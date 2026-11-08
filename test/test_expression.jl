using Test

# Use KrakeNN if available, otherwise include directly for standalone testing
if !isdefined(@__MODULE__, :KrakenExpr)
    include(joinpath(@__DIR__, "..", "src", "io", "expression.jl"))
end

@testset "KrakenExpr" begin

    @testset "Basic arithmetic" begin
        e = parse_kraken_expr("2 + 3")
        @test evaluate(e) ≈ 5.0

        e = parse_kraken_expr("x + y")
        @test evaluate(e; x=1.0, y=2.0) ≈ 3.0

        e = parse_kraken_expr("x * y - z")
        @test evaluate(e; x=3.0, y=4.0, z=2.0) ≈ 10.0

        e = parse_kraken_expr("x^2 + y^2")
        @test evaluate(e; x=3.0, y=4.0) ≈ 25.0
    end

    @testset "Math functions" begin
        e = parse_kraken_expr("sin(x)")
        @test evaluate(e; x=π/2) ≈ 1.0

        e = parse_kraken_expr("cos(0.0)")
        @test evaluate(e) ≈ 1.0

        e = parse_kraken_expr("sqrt(x)")
        @test evaluate(e; x=9.0) ≈ 3.0

        e = parse_kraken_expr("exp(0.0)")
        @test evaluate(e) ≈ 1.0

        e = parse_kraken_expr("abs(x)")
        @test evaluate(e; x=-5.0) ≈ 5.0

        e = parse_kraken_expr("tanh(0.0)")
        @test evaluate(e) ≈ 0.0
    end

    @testset "Constants" begin
        e = parse_kraken_expr("pi")
        @test evaluate(e) ≈ π

        e = parse_kraken_expr("2 * pi")
        @test evaluate(e) ≈ 2π

        e = parse_kraken_expr("sin(2 * pi * x)")
        @test evaluate(e; x=0.25) ≈ 1.0 atol=1e-12
    end

    @testset "User-defined variables (Define)" begin
        uv = Dict{Symbol,Float64}(:U => 0.05, :H => 2.5)
        e = parse_kraken_expr("4 * U * y * (H - y) / H^2", uv)
        # Parabolic profile at y = H/2 → max velocity = U
        @test evaluate(e; y=1.25) ≈ 0.05 atol=1e-12
        # Zero at walls
        @test evaluate(e; y=0.0) ≈ 0.0 atol=1e-12
        @test evaluate(e; y=2.5) ≈ 0.0 atol=1e-12
    end

    @testset "Domain variables" begin
        e = parse_kraken_expr("x / Lx + y / Ly")
        @test evaluate(e; x=0.5, Lx=1.0, y=0.5, Ly=2.0) ≈ 0.75

        e = parse_kraken_expr("sin(2 * pi * x / Lx) * sin(pi * y / Ly)")
        @test evaluate(e; x=0.25, Lx=1.0, y=0.5, Ly=1.0) ≈ 1.0 atol=1e-12
    end

    @testset "Time-dependent expressions" begin
        e = parse_kraken_expr("0.1 * sin(2 * pi * t / 5000)")
        @test is_time_dependent(e)
        @test !is_spatial(e)
        @test evaluate(e; t=1250.0) ≈ 0.1 atol=1e-12

        e = parse_kraken_expr("x + t")
        @test is_time_dependent(e)
        @test is_spatial(e)
    end

    @testset "Spatial detection" begin
        e = parse_kraken_expr("x^2 + y^2")
        @test is_spatial(e)
        @test !is_time_dependent(e)

        e = parse_kraken_expr("0.1")
        @test !is_spatial(e)
        @test !is_time_dependent(e)
    end

    @testset "Boolean / geometry conditions" begin
        e = parse_kraken_expr("(x - 2.5)^2 + (y - 1.25)^2 <= 0.5^2")
        @test evaluate(e; x=2.5, y=1.25) == true   # center
        @test evaluate(e; x=0.0, y=0.0) == false    # far away

        e = parse_kraken_expr("x < 3.0 && y < 0.5")
        @test evaluate(e; x=2.0, y=0.3) == true
        @test evaluate(e; x=4.0, y=0.3) == false
    end

    @testset "Comparisons" begin
        e = parse_kraken_expr("ifelse(x > 0.5, 1.0, 0.0)")
        @test evaluate(e; x=0.8) ≈ 1.0
        @test evaluate(e; x=0.2) ≈ 0.0
    end

    @testset "Variable collection" begin
        e = parse_kraken_expr("x + y * t + Lx")
        @test :x in e.variables
        @test :y in e.variables
        @test :t in e.variables
        @test :Lx in e.variables
    end

    @testset "Security — reject unsafe expressions" begin
        @test_throws ArgumentError parse_kraken_expr("run(`ls`)")
        @test_throws ArgumentError parse_kraken_expr("eval(:(1+1))")
        @test_throws ArgumentError parse_kraken_expr("open(\"file\")")
        @test_throws ArgumentError parse_kraken_expr("ccall(:puts, Cint, (Cstring,), \"hello\")")
        @test_throws ArgumentError parse_kraken_expr("Base.rm(\"file\")")
        # Unknown variables should be rejected
        @test_throws ArgumentError parse_kraken_expr("unknown_var + 1")
    end

    @testset "has_variable utility" begin
        e = parse_kraken_expr("x + y")
        @test has_variable(e, :x)
        @test has_variable(e, :y)
        @test !has_variable(e, :z)
        @test !has_variable(e, :t)
    end

    @testset "Edge cases" begin
        # Pure constant
        e = parse_kraken_expr("42.0")
        @test evaluate(e) ≈ 42.0

        # Negative number
        e = parse_kraken_expr("-x")
        @test evaluate(e; x=5.0) ≈ -5.0

        # Nested functions
        e = parse_kraken_expr("sqrt(abs(x))")
        @test evaluate(e; x=-4.0) ≈ 2.0

        # Division
        e = parse_kraken_expr("x / (y + 1)")
        @test evaluate(e; x=10.0, y=4.0) ≈ 2.0
    end
end
