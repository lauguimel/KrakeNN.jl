using Test
# nnz/nonzeros — no longer inherited once the include below auto-skips.
# Loaded through KrakeNN (SparseArrays is a KrakeNN dep, not a test dep; a bare
# `using SparseArrays` fails in the Pkg.test sandbox on Julia >= 1.12).
using KrakeNN.SparseArrays

# Auto-skips under `using KrakeNN` (runtests.jl); unexported helpers are
# KrakeNN.-qualified below, so the package-test path never double-includes src.
if !isdefined(@__MODULE__, :assemble_poisson_embedded)
    include(joinpath(@__DIR__, "..", "..", "src", "solve", "poisson_embedded.jl"))
end

const POISSON_EMBEDDED_NS = (16, 32, 64, 128)

embedded_sine_exact(x, y) = sin(pi * x) * sin(pi * y)
embedded_sine_rhs(x, y) = 2.0 * pi^2 * sin(pi * x) * sin(pi * y)

embedded_quad_exact(x, y) = x^2 + y^2
embedded_quad_rhs(x, y) = -4.0

function embedded_regular_fractions(N::Integer)
    return ones(N + 1, N), ones(N, N + 1), ones(N, N)
end

function embedded_sparse_max_abs(A)
    return nnz(A) == 0 ? 0.0 : maximum(abs.(nonzeros(A)))
end

function embedded_convergence_result(solve_case, error_case)
    errors = Float64[]
    orders = Float64[]

    previous_error = NaN
    for N in POISSON_EMBEDDED_NS
        err = error_case(N, solve_case(N))
        push!(errors, err)

        if !isnan(previous_error)
            push!(orders, log2(previous_error / err))
        end
        previous_error = err
    end

    mean_order = sum(orders) / length(orders)
    return errors, orders, mean_order
end

@testset "Poisson embedded cut-cell MMS" begin
    @testset "Regular limit matches rung 1" begin
        N_matrix = 8
        fx, fy, vf = embedded_regular_fractions(N_matrix)
        A_embedded, b_embedded = assemble_poisson_embedded(
            N_matrix, fx, fy, vf, embedded_sine_rhs;
            outer_bc=:dirichlet,
            outer_dirichlet=(x, y) -> 0.0,
        )
        A_regular, b_regular = KrakeNN.assemble_poisson_dirichlet(N_matrix, embedded_sine_rhs)

        @test embedded_sparse_max_abs(A_embedded - A_regular) == 0.0
        @test maximum(abs.(b_embedded - b_regular)) == 0.0

        errors, orders, mean_order = embedded_convergence_result(
            N -> begin
                fx, fy, vf = embedded_regular_fractions(N)
                solve_poisson_embedded(
                    N, fx, fy, vf, embedded_sine_rhs;
                    outer_bc=:dirichlet,
                    outer_dirichlet=(x, y) -> 0.0,
                )
            end,
            (N, u) -> KrakeNN.l2_error(u, embedded_sine_exact, N),
        )

        @test 1.8 <= mean_order <= 2.2
        @info "Embedded Poisson regular-limit MMS convergence" N=POISSON_EMBEDDED_NS errors=errors orders=orders mean_order=mean_order
    end

    @testset "All-Neumann embedded nullspace" begin
        N = 64
        fx, fy, vf = KrakeNN.tilted_half_plane_fractions(N)
        A, b = assemble_poisson_embedded(
            N, fx, fy, vf, (x, y) -> 0.0;
            outer_bc=:neumann,
            embedded_bc=:neumann,
        )

        max_row_sum = KrakeNN.fluid_row_sum_max(A, N, vf)
        @test max_row_sum < 1.0e-10

        pin_value = 1.25
        k0 = KrakeNN.first_fluid_dof(vf, N)
        A_pinned, b_pinned = pin_reference_dof(A, b, k0, pin_value)
        u = KrakeNN.solve_poisson(A_pinned, b_pinned, N)
        max_deviation = KrakeNN.fluid_constant_deviation(u, pin_value, N, vf)

        @test max_deviation < 1.0e-9
        @info "Embedded Poisson all-Neumann nullspace" N=N max_row_sum=max_row_sum pin_dof=k0 max_deviation=max_deviation
    end

    @testset "Embedded Dirichlet quadratic MMS convergence" begin
        errors, orders, mean_order = embedded_convergence_result(
            N -> begin
                fx, fy, vf = KrakeNN.tilted_half_plane_fractions(N)
                u = solve_poisson_embedded(
                    N, fx, fy, vf, embedded_quad_rhs;
                    outer_bc=:dirichlet,
                    embedded_bc=:dirichlet,
                    outer_dirichlet=embedded_quad_exact,
                    embedded_dirichlet=embedded_quad_exact,
                )
                return u, vf
            end,
            (N, result) -> begin
                u, vf = result
                KrakeNN.fluid_l2_error(u, embedded_quad_exact, N, vf)
            end,
        )

        @test mean_order >= 1.5
        @info "Embedded Poisson Dirichlet MMS convergence" N=POISSON_EMBEDDED_NS errors=errors orders=orders mean_order=mean_order
    end
end
