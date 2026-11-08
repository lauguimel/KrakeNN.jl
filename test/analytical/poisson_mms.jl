using Test

# Auto-skips under `using KrakeNN` (runtests.jl); unexported helpers are
# KrakeNN.-qualified below, so the package-test path never double-includes src.
if !isdefined(@__MODULE__, :solve_poisson_dirichlet)
    include(joinpath(@__DIR__, "..", "..", "src", "solve", "poisson.jl"))
end

const POISSON_MMS_NS = (16, 32, 64, 128)

dirichlet_exact(x, y) = sin(pi * x) * sin(pi * y)
dirichlet_rhs(x, y) = 2.0 * pi^2 * sin(pi * x) * sin(pi * y)

neumann_exact(x, y) = cos(pi * x) * cos(pi * y)
neumann_rhs(x, y) = 2.0 * pi^2 * cos(pi * x) * cos(pi * y)

function convergence_result(solve_case, u_exact)
    errors = Float64[]
    orders = Float64[]

    previous_error = NaN
    for N in POISSON_MMS_NS
        u = solve_case(N)
        err = KrakeNN.l2_error(u, u_exact, N)
        push!(errors, err)

        if !isnan(previous_error)
            push!(orders, log2(previous_error / err))
        end
        previous_error = err
    end

    mean_order = sum(orders) / length(orders)
    return errors, orders, mean_order
end

@testset "Poisson regular Cartesian MMS" begin
    @testset "Unpinned Neumann singularity" begin
        A, b = KrakeNN.assemble_poisson_neumann_unpinned(16, neumann_rhs)
        max_row_sum = maximum(abs.(vec(sum(A; dims=2))))
        @test max_row_sum < 1.0e-10
        @test_throws Exception KrakeNN.solve_poisson(A, b, 16)
    end

    @testset "Dirichlet second-order convergence" begin
        errors, orders, mean_order = convergence_result(
            N -> solve_poisson_dirichlet(N, dirichlet_rhs),
            dirichlet_exact,
        )

        @test 1.8 <= mean_order <= 2.2
        @info "Poisson Dirichlet MMS convergence" N=POISSON_MMS_NS errors=errors orders=orders mean_order=mean_order
    end

    @testset "Pinned Neumann second-order convergence" begin
        errors, orders, mean_order = convergence_result(
            N -> solve_poisson_neumann(N, neumann_rhs, neumann_exact),
            neumann_exact,
        )

        @test 1.8 <= mean_order <= 2.2
        @info "Poisson pinned Neumann MMS convergence" N=POISSON_MMS_NS errors=errors orders=orders mean_order=mean_order
    end
end
