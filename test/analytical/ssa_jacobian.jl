# T1: the coloured dual Jacobian of the SSA residual equals the dense ForwardDiff
# Jacobian (12 x 12 grid, Weertman and learned drag, rough non-converged state).
using Test
if !isdefined(@__MODULE__, :ssa_stream_problem)
    include(joinpath(@__DIR__, "ssa_helpers.jl"))
end

@testset "SSA Jacobian (T1)" begin
    rng = MersenneTwister(7)
    ld, theta = ssa_learned_drag()
    for (name, drag) in (("Weertman", SSAWeertman(SSA_C, 1 / 3)), ("learned", ld))
        @testset "$name" begin
            p = ssa_stream_problem(12; drag)
            n = ssa_nunknowns(p)
            U = 80 .* rand(rng, n) .+ 5.0 .* randn(rng, n)
            dense = ForwardDiff.jacobian(x -> ssa_residual(p, x), U)
            s = ssa_newton_solver(p)
            ssa_jacobian!(s, U)
            J = Matrix(s.J)
            d = maximum(abs, J .- dense)
            @info "SSA Jacobian: max |coloured - dense|" name d maxJ = maximum(abs, dense) colors = s.asm.ncolors
            @test d <= 1e-10 * maximum(abs, dense)
            # structure: no dense entry outside the declared pattern
            S = ssa_jacobian_pattern(p)
            @test all(iszero(dense[i, j]) || S[i, j] for i in 1:n, j in 1:n)
        end
    end
end
