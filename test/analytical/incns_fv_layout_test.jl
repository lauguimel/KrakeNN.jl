# T1 storage/derivative invariants, CPU Float64. Gates are exact equality:
# scalar identities and remapped flow blocks describe the same discrete operator.
using Test, Kraken, Random, LinearAlgebra
using Kraken.SparseArrays

# Independent transcription of the original physical dependency rules. This
# deliberately does not call the production dependency-declaration helpers.
function _fvlayout_reference_pattern(p)
    g = p.grid
    function neighbor(c, d)
        c == (0, 0) && return c
        i, j = c
        ii, jj = i + (d == 1) - (d == 2), j + (d == 3) - (d == 4)
        if !(1 <= ii <= g.Nx && 1 <= jj <= g.Ny)
            bc = (g.bcE, g.bcW, g.bcN, g.bcS)[d]
            bc == FV_BC_PERIODIC || return (0, 0)
            ii, jj = mod1(ii, g.Nx), mod1(jj, g.Ny)
        end
        return p.tag[ii, jj] == FV_CELL_SOLID ? (0, 0) : (ii, jj)
    end
    expected = falses(fv_nunknowns(p), fv_nunknowns(p))
    for j in 1:g.Ny, i in 1:g.Nx
        P = (i, j)
        E, W, N, S = ntuple(d -> neighbor(P, d), 4)
        star = (P, E, W, N, S)
        axis2 = (neighbor(E, 1), neighbor(W, 2), neighbor(N, 3), neighbor(S, 4))
        for equation in 1:3
            row = fv_index(g, i, j, equation)
            if p.tag[i, j] == FV_CELL_SOLID || (equation == 3 && p.tag[i, j] == FV_CELL_PINNED)
                expected[row, row] = true
                continue
            end
            supports = if equation == 1
                (star, (star..., neighbor(N, 1), neighbor(N, 2), neighbor(S, 1), neighbor(S, 2)), (star..., axis2...))
            elseif equation == 2
                ((star..., neighbor(E, 3), neighbor(E, 4), neighbor(W, 3), neighbor(W, 4)), star, (star..., axis2...))
            else
                ((P, E, W), (P, N, S), (star..., axis2...))
            end
            for comp in 1:3, c in supports[comp]
                c == (0, 0) && continue
                expected[row, fv_index(g, c..., comp)] = true
            end
        end
    end
    return sparse(expected)
end

function _fvlayout_map(p, q)
    g = q.grid
    physical = [fv_index(g, i, j, c) for j in 1:g.Ny for i in 1:g.Nx for c in 1:3]
    append!(physical, fv_nunknowns(g)+1:fv_nunknowns(q))
    scalar = [fv_index(g, i, j, 4) for j in 1:g.Ny for i in 1:g.Nx]
    return physical, scalar
end

@testset "FV field layout: offsets and independent dependency rules" begin
    for ns in (0, 1, 2)
        layout = Kraken.FVFieldLayout{ns}()
        @test isbitstype(typeof(layout))
        @test Kraken._fv_nfields(layout) == 3 + ns
        @test Kraken._fv_velocity_slot(layout, 2) == 2
        @test Kraken._fv_pressure_slot(layout) == 3
        @test Kraken._fv_scalar_slot(layout, 1) == 4
    end
    @test_throws ArgumentError Kraken.FVFieldLayout{-1}()
    bc = (W=:periodic, E=:periodic, S=:wall, N=:wall)
    for obstacle in (false, true)
        solid = falses(7, 7)
        obstacle && (solid[[1, 7], 3:4] .= true)
        p = fv_steady_problem(7, 7; bc, solid)
        S = fv_jacobian_pattern(p)
        @test S == _fvlayout_reference_pattern(p)
        if !obstacle
            rows = [fv_index(p.grid, 4, 4, c) for c in 1:3]
            @test vec(sum(S[rows, :]; dims=2)) == [23, 23, 15]
        end
        @test count(S[fv_index(p.grid, 1, 1, 3), :]) == 1
        @test isbitstype(typeof(p.grid))
        g = p.grid
        @test FVGrid2D(g.Nx, g.Ny, g.h, g.conv, g.bcE, g.bcW, g.bcN, g.bcS) == g
        @test FVGrid2D{Float64}(g.Nx, g.Ny, g.h, g.conv, g.bcE, g.bcW, g.bcN, g.bcS) == g
    end
end

@testset "FV one scalar: exact flow blocks, maps, mass tails and contractions" begin
    bc = (W=:periodic, E=:periodic, S=:wall, N=:wall)
    cy = CarreauYasuda(0.3, 0.05, 1.2, 2.0, 0.7)
    law = LearnedViscosity(TinyMLP{2,2,1}(); nu_ref=0.3, monotone=true)
    theta = law_init(law; rng=MersenneTwister(41))
    theta[3:4] .= (0.3, -0.2)
    field = [0.2i + 0.1j for i in 1:5, j in 1:4]
    ib = fv_immersed_circle(10, 10, 5.0, 5.0, 2.0; h=1.0, mass=:M3)
    solid = falses(5, 4); solid[3, 3] = true
    cases = (
        fv_steady_problem(5, 4; h=1.0, bc, solid, nu=0.3),
        fv_steady_problem(5, 4; h=1.0, bc, rheology=cy, gnf_eps=0.01),
        fv_steady_problem(5, 4; h=1.0, bc, viscosity=FVLawViscosity(law, theta, (field,))),
        fv_steady_problem(10, 10; h=1.0, bc, immersed=ib, wall_flux=0.0, nu=0.3))
    for (case, p) in enumerate(cases)
        @testset "case $case" begin
            q = Kraken._fv_with_layout(p, Kraken.FVFieldLayout{1}())
            g = q.grid
            @test fv_nunknowns(g) == 4g.Nx*g.Ny
            @test fv_nunknowns(q)-fv_nunknowns(g) == fv_nunknowns(p)-fv_nunknowns(p.grid)
            ids, scalar = _fvlayout_map(p, q)
            U = 0.1randn(MersenneTwister(92), fv_nunknowns(p))
            X = 0.2randn(MersenneTwister(93), fv_nunknowns(q)); X[ids] .= U
            R, RX = fv_residual(U, p), fv_residual(X, q)
            @test RX[ids] == R
            @test RX[scalar] == X[scalar]
            @test fv_fields(q, X) == fv_fields(p, U)
            @test fv_adapt(Array, q).grid === q.grid
            @test Kraken._fv_with(q; nu=copy(q.nu)).grid === q.grid
            source = fv_param_problem(FVUniformSource(2), q, [0.7])
            @test source.src[scalar] == q.src[scalar]
            @test all(==(0.7), source.src[2:4:fv_nunknowns(g)])
            @test source.grid === g
            S0, S1 = fv_jacobian_pattern(p), fv_jacobian_pattern(q)
            @test S1[ids, ids] == S0
            @test S1[scalar, scalar] == sparse(I, length(scalar), length(scalar))
            @test iszero(S1[scalar, ids]) && iszero(S1[ids, scalar])
            s0, s1 = fv_newton_solver(p), fv_newton_solver(q)
            J0, J1 = fv_jacobian!(s0, U), fv_jacobian!(s1, X)
            @test J1[ids, ids] == J0
            @test J1[scalar, scalar] == sparse(I, length(scalar), length(scalar))
            @test iszero(J1[scalar, ids]) && iszero(J1[ids, scalar])
            shifted = copy(fv_jacobian!(s1, X; shift=0.5))
            @test diag(shifted)[scalar] == ones(length(scalar))
            sig = Kraken._fv_state_sig(s1.norms, 0.0, X)
            Y = copy(X); Y[last(scalar)] += 1.0
            @test Kraken._fv_state_sig(s1.norms, 0.0, Y) != sig
            @test maximum(abs, RX) == last(Kraken._fv_norms(s1.norms, RX))
            params, values = if case == 2
                FVProblemMap(fv_with_gnf_parameters, 5), fv_gnf_parameters(p)
            elseif case == 3
                FVLawParameters(length(theta)), theta
            else
                FVViscosityField(), vec(p.nu)
            end
            @test fv_param_problem(params, q, values).grid === g
            lambda = randn(MersenneTwister(94), length(U))
            w0, w1 = fv_adjoint_workspace(s0, params), fv_adjoint_workspace(s1, params)
            copyto!(w0.λ, lambda); fill!(w1.λ, 0.3); w1.λ[ids] .= lambda
            a = Kraken._fv_param_cotangent!(w0, params, s0, U, values)
            b = Kraken._fv_param_cotangent!(w1, params, s1, X, values)
            @test a == b
            fv_set_params!(s1, params, values)
            @test s1.prob.grid === g
            if case == 4
                @test first(q.immersed.mass.rowcol[q.immersed.mass.rowcol .> fv_nunknowns(g)]) == fv_nunknowns(g)+1
                @test q.immersed.mass.Q == p.immersed.mass.Q
                @test q.immersed.mass.quadrature_error == p.immersed.mass.quadrature_error
                for obj in (FVWallForce(1;kind=:residual_flux), FVWallForce(1;kind=:surface))
                    @test fv_objective(obj, X, q) == fv_objective(obj, U, p)
                    value0, gu0 = Kraken._fv_objective_grad_u(obj, U, p)
                    value1, gu1 = Kraken._fv_objective_grad_u(obj, X, q)
                    @test value0 == value1 && gu1[ids] == gu0
                    @test iszero(gu1[scalar])
                    old_support = fv_objective_support(obj, p)
                    support = fv_objective_support(obj, q)
                    @test filter(k -> k in ids, support) == sort!(ids[old_support])
                    # Every cell in the original physical wall stencil gains
                    # one declared scalar direction, currently with zero gradient.
                    cells = unique([(k-1)÷3+1 for k in old_support if k <= fv_nunknowns(p.grid)])
                    @test filter(k -> k in scalar, support) == sort!(4 .* cells)
                end
            end
        end
    end
end

@testset "FV equation scales and Newton reports" begin
    p = Kraken._fv_with_layout(fv_steady_problem(3, 2; conv=false), Kraken.FVFieldLayout{1}())
    solver = fv_newton_solver(p)
    nb = solver.norms
    for invalid in ([1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, -1.0, 1.0],
                    [1.0, Inf, 1.0, 1.0], [NaN, 1.0, 1.0, 1.0])
        @test_throws Exception Kraken._fv_equation_scales!(nb, invalid)
        @test nb.equations.host_scales == ones(4)
    end
    scales = [2.0, 4.0, 8.0, 16.0]
    Kraken._fv_equation_scales!(nb, scales)
    R = repeat([2.0, 8.0, 24.0, 64.0], 6)
    eq = Kraken._fv_equation_norms(nb, R)
    @test eq.inf == [1.0, 2.0, 3.0, 4.0]
    @test last(Kraken._fv_norms(nb, R)) == 64.0
    @test eq.l2 == sqrt.([6.0, 24.0, 54.0, 96.0])
    X = zeros(fv_nunknowns(p)); X[4:4:end] .= 2.0
    stats = fv_newton!(solver, X; atol=1e-12, rtol=0.0)
    @test stats.converged && stats.iterations == 1
    @test iszero(X)
    @test stats.equation_scales == scales
    @test length(stats.equation_l2) == length(stats.history) == 2
    @test first(stats.equation_inf) == [0.0, 0.0, 0.0, 0.125]
    @test iszero(last(stats.equation_inf))
    @test first(stats.inf) == 0.125
    # Adjoint diagnostics report the raw full residual, even at a deliberately
    # unconverged scalar state and with nonunit Newton report scales.
    X[4] = 2.0
    objective = FVLinearFunctional([1], [1.0])
    gradient = fv_steady_gradient(solver, X, [1.0], objective; params=FVUniformViscosity())
    @test gradient.residual_inf == 2.0
    @test iszero(gradient.gU[4:4:end])
    # More than one reduction tile, with an appended constraint-like row.
    big = Kraken._FVNormBuffers(Array, 5; per=2, nextra=1, layout=Kraken.FVFieldLayout{1}())
    state = Float64.(mod.(1:21, 17) .- 8)
    @test Kraken._fv_state_sig(big, 0.0, state) == (0.0, sum(state), sum(abs2, state))
    eq = Kraken._fv_equation_norms(big, state)
    @test eq.l2[3] == sqrt(sum(abs2, state[3:4:20]) + state[end]^2)
    before = Kraken._fv_state_sig(big, 0.0, state)
    state[end] += 1.0
    @test Kraken._fv_state_sig(big, 0.0, state) != before
end
