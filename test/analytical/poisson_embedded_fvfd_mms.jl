using Test
using KrakeNN

if !isdefined(@__MODULE__, :assemble_poisson_embedded)
    include(joinpath(@__DIR__, "..", "..", "src", "solve", "poisson_embedded.jl"))
end

if !isdefined(@__MODULE__, :fractions_from_fvfd)
    include(joinpath(@__DIR__, "..", "..", "src", "solve", "poisson_embedded_fvfd.jl"))
end

const POISSON_EMBEDDED_FVFD_NS = (16, 32, 64, 128)
const POISSON_EMBEDDED_FVFD_RESULTS = Dict{Symbol,Any}()

fvfd_quad_exact(x, y) = x^2 + y^2
fvfd_quad_rhs(x, y) = -4.0
fvfd_zero_rhs(x, y) = 0.0

function fvfd_a2_halfplane_boundary(N::Integer)
    nx = cos(pi / 6)
    ny = sin(pi / 6)
    offset = -Float64(N) * (nx * 0.43 + ny * 0.52)
    return KrakeNN.fvfd_embedded_boundary_from_halfplane_2d(N, N, nx, ny, offset; FT=Float64)
end

function fvfd_convergence_result(solve_case, error_case)
    errors = Float64[]
    orders = Float64[]
    previous_error = NaN

    for N in POISSON_EMBEDDED_FVFD_NS
        err = error_case(N, solve_case(N))
        push!(errors, err)
        if !isnan(previous_error)
            push!(orders, log2(previous_error / err))
        end
        previous_error = err
    end

    return errors, orders, sum(orders) / length(orders)
end

function fvfd_max_abs_diff(actual, expected)
    Δ = abs.(actual .- expected)
    max_diff, idx = findmax(Δ)
    return max_diff, Tuple(idx)
end

function fvfd_assert_matches_a2(name::Symbol, actual, expected; atol::Float64=1.0e-12)
    size(actual) == size(expected) ||
        throw(DimensionMismatch("$(name) size mismatch: got $(size(actual)), expected $(size(expected))"))
    max_diff, idx = fvfd_max_abs_diff(actual, expected)
    if !(max_diff < atol)
        throw(ErrorException(
            "$(name) mismatch at $(idx): actual=$(actual[idx...]) " *
            "expected=$(expected[idx...]) diff=$(max_diff) tol=$(atol)",
        ))
    end
    @test max_diff < atol
    return max_diff
end

function fvfd_orientation_crosscheck(N::Integer=32)
    eb = fvfd_a2_halfplane_boundary(N)
    vf_real, fx_real, fy_real = fractions_from_fvfd(eb)
    fx_a2, fy_a2, vf_a2 = KrakeNN.tilted_half_plane_fractions(N)

    vf_diff = fvfd_assert_matches_a2(:vol_frac, vf_real, vf_a2)
    fx_diff = fvfd_assert_matches_a2(:face_frac_x, fx_real, fx_a2)
    fy_diff = fvfd_assert_matches_a2(:face_frac_y, fy_real, fy_a2)
    max_diff = max(vf_diff, fx_diff, fy_diff)

    @info "FVFD embedded Poisson orientation cross-check" N=N vol_frac=vf_diff face_frac_x=fx_diff face_frac_y=fy_diff max_diff=max_diff
    return (; N, vol_frac=vf_diff, face_frac_x=fx_diff, face_frac_y=fy_diff, max_diff)
end

function fvfd_dirichlet_mms()
    errors, orders, mean_order = fvfd_convergence_result(
        N -> begin
            eb = fvfd_a2_halfplane_boundary(N)
            vf, _, _ = fractions_from_fvfd(eb)
            A, b = assemble_poisson_embedded_from_fvfd(
                eb, fvfd_quad_rhs;
                outer_bc=:dirichlet,
                embedded_bc=:dirichlet,
                outer_dirichlet=fvfd_quad_exact,
                embedded_dirichlet=fvfd_quad_exact,
            )
            return KrakeNN.solve_poisson(A, b, N), vf
        end,
        (N, result) -> begin
            u, vf = result
            KrakeNN.fluid_l2_error(u, fvfd_quad_exact, N, vf)
        end,
    )

    @test mean_order >= 1.5
    @info "FVFD-lowered embedded Dirichlet MMS convergence" N=POISSON_EMBEDDED_FVFD_NS errors=errors orders=orders mean_order=mean_order
    return (; N=POISSON_EMBEDDED_FVFD_NS, errors, orders, mean_order)
end

function fvfd_neumann_nullspace(; N::Integer=64)
    eb = fvfd_a2_halfplane_boundary(N)
    vf, _, _ = fractions_from_fvfd(eb)
    A, b = assemble_poisson_embedded_from_fvfd(
        eb, fvfd_zero_rhs;
        outer_bc=:neumann,
        embedded_bc=:neumann,
    )

    max_row_sum = KrakeNN.fluid_row_sum_max(A, N, vf)
    @test max_row_sum < 1.0e-9

    pin_value = 1.25
    k0 = KrakeNN.first_fluid_dof(vf, N)
    A_pinned, b_pinned = pin_reference_dof(A, b, k0, pin_value)
    u = KrakeNN.solve_poisson(A_pinned, b_pinned, N)
    max_deviation = KrakeNN.fluid_constant_deviation(u, pin_value, N, vf)
    @test max_deviation < 1.0e-8

    @info "FVFD-lowered embedded Neumann nullspace" N=N max_row_sum=max_row_sum pin_dof=k0 max_deviation=max_deviation
    return (; N, max_row_sum, pin_dof=k0, max_deviation)
end

function fvfd_fraction_arrays_in_unit_interval(arrays...)
    return all(all(value -> 0.0 <= Float64(value) <= 1.0, array) for array in arrays)
end

function fvfd_circle_smoke(; N::Integer=32)
    eb = KrakeNN.fvfd_embedded_boundary_from_circle_2d(
        N, N, 0.5 * Float64(N), 0.5 * Float64(N), 0.25 * Float64(N);
        FT=Float64,
    )
    vf, fx, fy = fractions_from_fvfd(eb)
    @test fvfd_fraction_arrays_in_unit_interval(vf, fx, fy)

    A, b = assemble_poisson_embedded_from_fvfd(
        eb, fvfd_zero_rhs;
        outer_bc=:neumann,
        embedded_bc=:neumann,
    )
    max_row_sum = KrakeNN.fluid_row_sum_max(A, N, vf)
    @test max_row_sum < 1.0e-9

    pin_value = -0.75
    k0 = KrakeNN.first_fluid_dof(vf, N)
    A_pinned, b_pinned = pin_reference_dof(A, b, k0, pin_value)
    u = KrakeNN.solve_poisson(A_pinned, b_pinned, N)
    max_deviation = KrakeNN.fluid_constant_deviation(u, pin_value, N, vf)
    @test max_deviation < 1.0e-8

    @info "FVFD circle adapter smoke" N=N max_row_sum=max_row_sum pin_dof=k0 max_deviation=max_deviation
    return (; N, max_row_sum, pin_dof=k0, max_deviation)
end

@testset "Poisson embedded FVFD adapter MMS" begin
    orientation = fvfd_orientation_crosscheck(32)
    POISSON_EMBEDDED_FVFD_RESULTS[:orientation] = orientation

    dirichlet = fvfd_dirichlet_mms()
    POISSON_EMBEDDED_FVFD_RESULTS[:dirichlet] = dirichlet

    neumann = fvfd_neumann_nullspace(; N=64)
    POISSON_EMBEDDED_FVFD_RESULTS[:neumann] = neumann

    circle = fvfd_circle_smoke(; N=32)
    POISSON_EMBEDDED_FVFD_RESULTS[:circle] = circle
end
