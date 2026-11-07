# C2 of the FV Newton roadmap (src/methods/inc_ns/fv_newton.jl): coloured
# dual Jacobian of the C1 residual + Newton through the linear-solve seam. CPU
# checks (the assembly runs the same KernelAbstractions kernels as on GPU, on
# the KA CPU backend):
#   1. pattern built from the stencil ⊇ support of the dense ForwardDiff
#      Jacobian, and tight on Navier-Stokes states; coloured assembly == dense
#      Jacobian == C1 box-coloured helper, bit for bit, in CSC and CSR order,
#      for every chunk size; J·v vs central differences;
#   2. Stokes (linear): one Newton step reaches the linear-solver floor;
#      periodic Poiseuille vs the analytical parabola;
#   3. Navier-Stokes cavity Re = 100: quadratic convergence to round-off,
#      ONE symbolic analysis (lin_factorize) then numeric refactorizations
#      (UMFPACK symbolic object unchanged), Ghia profiles, SIMPLE agreement;
#      Re = 1000 with and without pseudo-transient continuation;
#   4. the pressure-pin row scale does not change the UMFPACK solve;
#   5. platform contract: IncNS(:fv_newton).
# GPU (cuDSS) contract: incns_fv_newton_gpu.jl.

using Test
using LinearAlgebra
using Random
using ForwardDiff
using KrakeNN
using KrakeNN.SparseArrays

if !isdefined(@__MODULE__, :_fvc1_sparse_jacobian)
    include(joinpath(@__DIR__, "fv_steady_helpers.jl"))
end
if !isdefined(@__MODULE__, :_fvc2_ghia_error)
    include(joinpath(@__DIR__, "fv_newton_helpers.jl"))
end

_fvc2_dense_jacobian(p, U) =
    ForwardDiff.jacobian((R, X) -> fv_residual!(R, X, p), zeros(length(U)), U)

# quadratic-convergence gate on a residual history r_0, r_1, ...: every step
# taken from e_k = r_k / r_0 <= 1e-2 satisfies e_{k+1} <= C e_k^2 or lands on
# the round-off floor (e_{k+1} <= 1e-14)
function _fvc2_quadratic(hist; C = 10.0)
    e = hist ./ hist[1]
    steps = [(e[k], e[k+1]) for k in 1:length(e)-1 if e[k] <= 1e-2]
    return length(steps) >= 2 && all(b <= C * a^2 || b <= 1e-14 for (a, b) in steps)
end

@testset "IncNS steady FV Newton (C2, CPU)" begin

    @testset "state signature reuses norm buffers across partial blocks" begin
        for ncell in (1, 255, 256, 257, 513)
            nb = KrakeNN._FVNormBuffers(Array, ncell)
            # Exactly representable integer sums cover all three components,
            # including the last cell of a non-full reduction block.
            U = Float64.(mod.(1:3ncell, 17) .- 8)
            ref = (0.125, sum(U), sum(abs2, U))
            @test KrakeNN._fv_state_sig(nb, 0.125, U) == ref
            KrakeNN._fv_norms(nb, U)
            @test KrakeNN._fv_state_sig(nb, 0.125, U) == ref
            U[end] += 1
            @test KrakeNN._fv_state_sig(nb, 0.0, U) == (0.0, sum(U), sum(abs2, U))
        end
    end

    @testset "pattern, colouring and coloured dual assembly" begin
        rng = MersenneTwister(11)
        solid = zeros(Bool, 14, 8); solid[5:7, 1:3] .= true; solid[10:11, 6:8] .= true
        cases = (
            ("cavity NS nu(x,y) 12x10",
             fv_steady_problem(12, 10; nu = (x, y) -> 0.01 * (1 + x * y), boundary_velocity = _fvc2_lid), true),
            ("masked inlet/outlet channel 14x8",
             fv_steady_problem(14, 8; nu = (x, y) -> 0.05 * (1 + x * y), solid, bc = (W = :inlet, E = :outlet, S = :wall, N = :wall),
                               boundary_velocity = (s, x, y) -> s === :W ? (4y * (1 - y), 0.0) : (0.0, 0.0)), true),
            ("x-periodic channel 12x7 (12 not a multiple of 5)",
             fv_steady_problem(12, 7; nu = (x, y) -> 0.1 + 0.05 * y + 0.02 * sin(x),
                               bc = (W = :periodic, E = :periodic, S = :wall, N = :wall),
                               source = (x, y) -> (1.0, 0.0, 0.0)), false),
            ("doubly periodic Stokes 10x10",
             fv_steady_problem(10, 10; nu = (x, y) -> 1 + 0.3 * sin(6x), conv = false,
                               bc = (W = :periodic, E = :periodic, S = :periodic, N = :periodic)), true),
        )
        for (name, p, box_ok) in cases
            @testset "$name" begin
                S = fv_jacobian_pattern(p)
                colors, ncol = fv_jacobian_coloring(S)
                # max 23 entries per row (interior momentum rows)
                @test maximum(diff(sparse(transpose(S)).colptr)) <= 23
                # J0 measured 35-39 colours on cavities up to 1024²; the doubly
                # periodic 10x10 wrap needs 43 (measured 2026-11-04)
                @test ncol <= 45
                n = fv_nunknowns(p)
                s = fv_newton_solver(p)
                for trial in 1:2
                    U = randn(rng, n)
                    Jd = _fvc2_dense_jacobian(p, U)
                    # superset: no nonzero of the dense Jacobian outside the pattern
                    @test all(S[k] for k in findall(!iszero, Jd))
                    fv_jacobian!(s, U)
                    @test Matrix(s.J) == Jd                 # bit for bit
                    box_ok && @test _fvc1_sparse_jacobian(p, U) == sparse(Jd)
                    if p.grid.conv == 1
                        # tight on Navier-Stokes states with a generic nu(x, y):
                        # every stored entry is nonzero (a nu independent of x
                        # cancels dRv/du_N exactly, of y dRu/dv_E: values, not
                        # structure)
                        @test count(!iszero, nonzeros(s.J)) == nnz(S)
                    end
                    # chunk sizes and CSR storage order (the cuDSS layout) on CPU
                    for (chunk, order) in ((1, :csc), (8, :csr), (5, :csr))
                        asm = fv_jacobian_assembler(S, colors, ncol; chunk, order)
                        nz = zeros(nnz(S))
                        fv_assemble_jacobian!(nz, asm, U, p)
                        ref = order === :csc ? nonzeros(s.J) : nonzeros(sparse(transpose(s.J)))
                        @test nz == ref
                    end
                    # J v against central differences of the residual
                    v = randn(rng, n); e = 1e-6
                    fd = (fv_residual(U .+ e .* v, p) .- fv_residual(U .- e .* v, p)) ./ (2e)
                    @test norm(s.J * v - fd) / norm(fd) < 1e-7
                end
            end
        end
    end

    @testset "Stokes: one Newton step; periodic Poiseuille" begin
        # inlet/outlet channel, constant nu, Stokes: R is affine in U, so one
        # Newton step from rest lands on the linear-solver floor
        p = fv_steady_problem(24, 12; nu = 1.0, conv = false,
                              bc = (W = :inlet, E = :outlet, S = :wall, N = :wall),
                              boundary_velocity = (s, x, y) -> s === :W ? (6y * (1 - y), 0.0) : (0.0, 0.0))
        r = solve_incns_fv_newton(; problem = p, atol = 0.0, rtol = 0.0, maxit = 1)
        @test r.iterations == 1
        @test r.residual_history[end] <= 1e-12
        @test r.stats.alpha == [1.0]
        # x-periodic channel, body force f, Navier-Stokes: u = f y (1 - y) / (2 nu)
        f, nu, Ny = 1.0, 0.1, 16
        q = fv_steady_problem(8, Ny; nu, bc = (W = :periodic, E = :periodic, S = :wall, N = :wall),
                              source = (x, y) -> (f, 0.0, 0.0))
        r = solve_incns_fv_newton(; problem = q)
        @test r.converged
        y = [(j - 0.5) / Ny for j in 1:Ny]
        ue = f .* y .* (1 .- y) ./ (2nu)
        @test norm(r.u .- ue') / norm(repeat(ue', 8)) <= 0.01       # V1: L2 <= 1 %
        @test maximum(abs, r.v) < 1e-12
    end

    @testset "cavity Re = 100: quadratic convergence, one analysis, Ghia, SIMPLE" begin
        N = 32
        p = fv_steady_problem(N, N; nu = 0.01, boundary_velocity = _fvc2_lid)
        s = fv_newton_solver(p)
        U = zeros(fv_nunknowns(p))
        st1 = fv_newton!(s, U; maxit = 1)
        @test s.n_factorize == 1 && s.n_refactorize == 0
        F = s.cache.factor
        sym = F.symbolic.p                      # UMFPACK symbolic analysis
        st = fv_newton!(s, U; atol = 1e-15, rtol = 0.0, maxit = 15)
        hist = vcat(st1.history[1], st.history)
        @info "C2 Newton, cavity Re=100 32² (CPU)" residual_2norm = hist inf_norm = st.inf alpha = st.alpha
        @test st.converged
        @test length(hist) - 1 <= 7
        @test hist[end] < 1e-14 && st.inf[end] < 1e-15       # round-off level
        @test st.cont[end] < 1e-15 && st.mom[end] < 1e-15
        @test _fvc2_quadratic(hist)
        # the analysis ran once; every later iteration refactorized the same
        # UMFPACK object, whose symbolic analysis was kept
        @test s.n_factorize == 1
        @test s.n_refactorize == length(hist) - 2
        @test s.cache.factor === F && F.symbolic.p == sym
        # finalize: the cached factors are those of J at the converged state
        fv_newton_finalize!(s, U)
        b = randn(MersenneTwister(2), length(U))
        Jd = _fvc2_dense_jacobian(p, U)
        x = similar(b); lin_solve!(x, s.cache, b)
        @test norm(Jd * x - b) / norm(b) < 1e-12
        lin_solve!(x, s.cache, b; transpose = true)          # C4 adjoint path
        @test norm(transpose(Jd) * x - b) / norm(b) < 1e-12
        @test s.n_factorize == 1
        # agreement with the SIMPLE cavity driver at discretisation level
        # (gate and measurement as in incns_fv_steady_residual.jl)
        u, v, _ = fv_fields(p, U)
        sp = solve_incns_cavity(; nx = N, ny = N, Re = 100.0, tol = 1e-9, maxiter = 20000)
        @test sp.converged
        dif(a, b) = maximum(abs, (a[N ÷ 2, :] .+ a[N ÷ 2 + 1, :]) ./ 2 .- (b[N ÷ 2, :] .+ b[N ÷ 2 + 1, :]) ./ 2)
        @test dif(u, sp.u) < 5e-3
        @test dif(permutedims(v), permutedims(sp.v)) < 5e-3
        # Ghia Re = 100 at 128² (V2 gate 5 %; measured 2026-11-04 macOS CPU:
        # 0.0053 on u, 0.0072 on v, 5 iterations)
        q = fv_steady_problem(128, 128; nu = 0.01, boundary_velocity = _fvc2_lid)
        r = solve_incns_fv_newton(; problem = q)
        @test r.converged && r.iterations <= 7
        eu, ev = _fvc2_ghia_error(r.u, r.v, 100.0)
        @test eu <= 0.05 && ev <= 0.05
        @test r.stats.n_factorize == 1
    end

    @testset "cavity Re = 1000 32²: Newton and pseudo-transient continuation" begin
        p = fv_steady_problem(32, 32; nu = 1e-3, boundary_velocity = _fvc2_lid)
        r = solve_incns_fv_newton(; problem = p, maxit = 30)
        @test r.converged && r.iterations <= 12
        @test _fvc2_quadratic(r.residual_history)
        rp = solve_incns_fv_newton(; problem = p, maxit = 30, dtau0 = 1.0, dtau_max = 1e4)
        @test rp.converged
        @test rp.stats.shift[1] > 0 && rp.stats.shift[end] == 0     # continuation switched off
        @test rp.residual_history[end] < 1e-12
        @test norm(rp.U - r.U, Inf) < 1e-10                         # same discrete solution
    end

    @testset "pressure-pin row scale does not change the UMFPACK solve" begin
        # see benchmarks/fv_c2/pin_scaling.jl (receipt results/pin_scaling_cpu.txt)
        # at a mid-Newton state (two steps from rest), as in the benchmark
        p = fv_steady_problem(32, 32; nu = 0.01, boundary_velocity = _fvc2_lid)
        r = solve_incns_fv_newton(; problem = p, maxit = 2)
        U = r.U
        s = r.solver
        J = copy(fv_jacobian!(s, U))
        R = fv_residual(U, p)
        rp = 3
        x1 = lu(J) \ R
        for sc in (J[6, 6], 1e-6, 1e6)
            D = ones(length(R)); D[rp] = sc
            @test (lu(Diagonal(D) * J) \ (D .* R)) ≈ x1 rtol = 1e-13
        end
    end

    @testset "platform contract: IncNS(:fv_newton)" begin
        @test capabilities(IncNS(:fv_newton)) == Set((ForwardSolve, SteadyResidual, SteadyAdjoint, AssembledJacobian))
        @test SteadyResidual ∉ capabilities(IncNS(:cavity))
        p = fv_steady_problem(16, 16; nu = 0.01, boundary_velocity = _fvc2_lid)
        sol = solve((; problem = p), IncNS(:fv_newton))
        @test sol isa IncNSSolution
        ref = solve_incns_fv_newton(; problem = p)
        @test sample(sol, :u) == ref.u && sample(sol, :p) == ref.p
        @test norm(residual(p, IncNS(:fv_newton), Array(sample(sol, :U))), Inf) < 1e-12
    end
end
