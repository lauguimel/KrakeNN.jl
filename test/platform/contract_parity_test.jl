using Test
using KrakeNN

# A method that declares nothing → exercises the default (fail-safe) capabilities().
# (structs are top-level only in Julia, so it lives outside the @testset.)
struct _DummyMethod <: KrakeNN.AbstractMethod end

@testset "platform contract (Phase 0/0b)" begin

    @testset "capabilities introspection" begin
        @test capabilities(LBM()) == Set((ForwardSolve, GPUExecution, SteadyAdjoint, SteadyResidual))
        @test capabilities(_DummyMethod()) == Set{Capability}()   # default = empty (fail-safe)
    end

    @testset "solve/sample parity with run_simulation" begin
        krk = """
            Simulation parity D2Q9
            Domain L = 0.25 x 1.0  N = 8 x 16
            Physics nu = 0.1  Fx = 1e-5
            Boundary x periodic
            Boundary south wall
            Boundary north wall
            Run 200 steps
        """
        # Separate setups so any setup mutation can't couple the two runs.
        ref = run_simulation(parse_kraken(krk))
        sol = solve(parse_kraken(krk), LBM())

        @test sol isa LBMSolution
        @test sol isa AbstractSolution

        # Behaviour-preserving: the wrapped result is bit-for-bit identical.
        @test sol.result.ρ  == ref.ρ
        @test sol.result.ux == ref.ux
        @test sol.result.uy == ref.uy

        # sample is a faithful pass-through (same array object; indexed access matches).
        @test sample(sol, :ux)         === sol.result.ux
        @test sample(sol, :ux, :)      === sol.result.ux
        @test sample(sol, :uy, (2, 2)) == ref.uy[2, 2]
    end
end

@testset "platform observables (Phase 1)" begin
    krk = """
        Simulation parity D2Q9
        Domain L = 0.25 x 1.0  N = 8 x 16
        Physics nu = 0.1  Fx = 1e-5
        Boundary x periodic
        Boundary south wall
        Boundary north wall
        Run 200 steps
    """
    sol = solve(parse_kraken(krk), LBM())
    ref = sol.result

    # observe goes through sample only — values match the raw field.
    @test observe(sol, FieldProbe(:ux, (2, 2))).value == ref.ux[2, 2]
    @test observe(sol, LineProfile(:uy, [(2, j) for j in 1:4])).value == [ref.uy[2, j] for j in 1:4]
    @test observe(sol, FieldReduction(:ρ, sum)).value == sum(ref.ρ)

    pred = observe(sol, FieldProbe(:ux, (2, 2)))
    @test pred isa Prediction
    @test pred.observable isa FieldProbe

    # predict = solve + observe (LBM is deterministic → identical to observing a fresh solve).
    pr = predict(parse_kraken(krk), LBM(), FieldReduction(:ρ, sum))
    @test pr.value == observe(solve(parse_kraken(krk), LBM()), FieldReduction(:ρ, sum)).value
end
