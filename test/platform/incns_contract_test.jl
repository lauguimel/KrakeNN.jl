# Platform-contract parity for the IncNS wrapper (mirrors
# test/platform/contract_parity_test.jl for LBM). The wrapper must be
# behaviour-preserving: solve(params, IncNS(:projection)) is bit-identical to
# calling solve_incns_projection directly with the same keywords, and
# sample is a faithful pass-through on the wrapped NamedTuple.

using Test
using Kraken
using KernelAbstractions

@testset "IncNS platform contract" begin

    @testset "capabilities introspection + driver validation" begin
        # SteadyResidual = the C1 steady FV residual (fv_steady_residual.jl),
        # declared only by the driver that solves it (decision 2026-11-04)
        @test capabilities(IncNS(:projection)) == Set((ForwardSolve,))
        @test capabilities(IncNS(:simple)) == Set((ForwardSolve,))
        @test capabilities(IncNS(:fv_newton)) == Set((ForwardSolve, SteadyResidual, SteadyAdjoint, AssembledJacobian))
        # CPU forward solve only: the CUDA seam methods are manual-load.
        @test GPUExecution ∉ capabilities(IncNS(:projection))
        @test_throws ArgumentError IncNS(:not_a_driver)
    end

    @testset "solve/sample parity with solve_incns_projection (16², 5 steps)" begin
        nu = 0.1
        params = (; nx = 16, ny = 16, Lx = 2pi, Ly = 2pi, nu,
                  dt = 1.0e-2, nsteps = 5,
                  bc_x = :periodic, bc_y = :periodic,
                  u0 = (x, y) -> sin(x) * cos(y),
                  v0 = (x, y) -> -cos(x) * sin(y),
                  p0 = (x, y) -> 0.25 * (cos(2x) + cos(2y)),
                  scheme = :cn, backend = CPU())

        ref = solve_incns_projection(; params...)
        sol = solve(params, IncNS(:projection))

        @test sol isa IncNSSolution
        @test sol isa AbstractSolution

        # Behaviour-preserving: bit-for-bit identical wrapped result.
        @test sol.result.u == ref.u
        @test sol.result.v == ref.v
        @test sol.result.p == ref.p
        @test sol.result.t_final == ref.t_final
        @test sol.result.nfactorizations == ref.nfactorizations
        @test sol.result.nlinsolves == ref.nlinsolves

        # sample is a faithful pass-through (same array object; indexed access matches).
        @test sample(sol, :u)         === sol.result.u
        @test sample(sol, :u, :)      === sol.result.u
        @test sample(sol, :v, (2, 2)) == ref.v[2, 2]
        @test sample(sol, :p, (3, 4)) == ref.p[3, 4]
    end
end
