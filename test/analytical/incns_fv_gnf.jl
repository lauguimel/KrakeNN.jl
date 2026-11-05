# C3 of the FV Newton roadmap (src/methods/inc_ns/fv_gnf.jl): generalised-
# Newtonian viscosity inside the steady FV residual, Jacobian through ν(U).
# CPU checks (same KernelAbstractions kernels as on GPU, KA CPU backend):
#   1. GNF pattern (exact stencil of the face strain rate) ⊇ support of the
#      dense ForwardDiff Jacobian and tight on random states; coloured assembly
#      == dense Jacobian bit for bit, CSC and CSR, for power-law shear-thinning,
#      shear-thickening, Carreau-Yasuda, Cross, Herschel-Bulkley; finite at rest;
#   2. law parameters as a vector θ; ∂R/∂θ by duals through the kernel vs
#      central differences (the C4 path);
#   3. n = 1 reproduces the Newtonian C2 residual bit for bit and its Newton
#      solution to round-off;
#   4. power-law Poiseuille: FV vs the regularised 1D reference (V1 ≤ 2 %),
#      second-order refinement, ε → 0 towards the analytical profile;
#      quadratic Newton convergence from rest, one symbolic analysis;
#   5. 2D: power-law cavity (quadratic convergence) and the Carreau-Yasuda
#      obstacle channel against Kraken's LBM GNF driver
#      (run_gnf_periodic_steady), with the gap shrinking under refinement.
# GPU (cuDSS) contract: incns_fv_gnf_gpu.jl.

using Test
using LinearAlgebra
using Random
using ForwardDiff
using Kraken
using Kraken.SparseArrays

if !isdefined(@__MODULE__, :_fvc2_ghia_error)
    include(joinpath(@__DIR__, "fv_newton_helpers.jl"))
end

_fvc3_dense_jacobian(p, U) =
    ForwardDiff.jacobian((R, X) -> fv_residual!(R, X, p), zeros(length(U)), U)

# quadratic-convergence gate, as in incns_fv_newton.jl: every step taken from
# e_k = r_k / r_0 <= 1e-2 satisfies e_{k+1} <= C e_k^2 or lands on the
# round-off floor; at least two such steps. The floor is 1e-13 here, not 1e-14:
# the power-law channels reach 1.2e-14 (Ny = 16) and 3.1e-14 (Ny = 32) relative
# from e = 9e-10 (measured 2026-11-05), round-off of a small r_0.
function _fvc3_quadratic(hist; C = 10.0, floor = 1e-13)
    e = hist ./ hist[1]
    steps = [(e[k], e[k+1]) for k in 1:length(e)-1 if e[k] <= 1e-2]
    return length(steps) >= 2 && all(b <= C * a^2 || b <= floor for (a, b) in steps)
end

# regularised GNF Poiseuille, 1D reference: walls at y = 0 and H, body force G,
# shear stress τ(y) = G (H/2 - y); γ̇(τ) inverts τ = ν(√(γ̇² + ε²)) γ̇ (monotone
# for the laws used) by bisection, u(y) = ∫₀^y γ̇ by the midpoint rule on M
# points (M = 4000: quadrature error < 1e-7 relative, checked against the
# closed form for ε = 0)
function _fvc3_gdot(law, eps, tau)
    f(gd) = effective_viscosity(law, sqrt(gd^2 + eps^2)) * gd - tau
    lo, hi = 0.0, 1.0
    while f(hi) < 0
        hi *= 2
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        f(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end
function _fvc3_uref(law, eps, G, H, y; M = 4000)
    y = min(y, H - y)
    dy = y / M
    return sum(_fvc3_gdot(law, eps, G * (H / 2 - (k - 0.5) * dy)) for k in 1:M; init = 0.0) * dy
end
# unregularised power law
_fvc3_uan(K, n, G, H, y) = n / (n + 1) * (G / K)^(1 / n) * ((H / 2)^(1 + 1 / n) - abs(y - H / 2)^(1 + 1 / n))

_fvc3_channel(law, eps, Ny; G = 1.0, Nx = 4) =
    fv_steady_problem(Nx, Ny; rheology = law, gnf_eps = eps,
                      bc = (W = :periodic, E = :periodic, S = :wall, N = :wall),
                      source = (x, y) -> (G, 0.0, 0.0))

# LBM GNF T4 geometry (test/ad/gnf_adjoint_enzyme_test.jl): periodic channel
# 8m × 4m with two half-cylinders, Carreau-Yasuda of contrast 50, lattice units
# (h = 1). The LBM half-way bounce-back wall lies on the cell faces of the
# staircase mask, the FV no-slip faces are the same faces.
function _fvc3_obstacle(m)
    nu0 = 0.17; Nx, Ny = 8m, 4m
    solid = [(((i - 0.5) / m - 4)^2 + ((j - 0.5) / m)^2 < 1.5^2) ||
             (((i - 0.5) / m - 4)^2 + ((j - 0.5) / m - 4)^2 < 1.5^2) for i in 1:Nx, j in 1:Ny]
    g = 0.3 * 5 * nu0^2 / m^3
    closure = CarreauYasuda(nu0, 0.02 * nu0, 5.0 * m^2 / nu0, 2.0, 0.7)
    return (; Nx, Ny, solid, g, closure)
end

const _FVC3_PL_THIN = PowerLaw(0.1, 0.5; nu_min = 0.0, nu_max = Inf)
const _FVC3_PL_THICK = PowerLaw(0.1, 1.5; nu_min = 0.0, nu_max = Inf)
const _FVC3_CY = CarreauYasuda(0.1, 0.002, 3.0, 2.0, 0.4)

@testset "IncNS steady FV GNF (C3, CPU)" begin

    @testset "GNF Jacobian: exact-stencil pattern, coloured assembly == dense ForwardDiff" begin
        rng = MersenneTwister(31)
        solid = zeros(Bool, 14, 8); solid[5:7, 1:3] .= true; solid[10:11, 6:8] .= true
        geoms = (
            ("cavity 12x10", (; boundary_velocity = _fvc2_lid), (12, 10)),
            ("masked inlet/outlet channel 14x8",
             (; solid, bc = (W = :inlet, E = :outlet, S = :wall, N = :wall),
              boundary_velocity = (s, x, y) -> s === :W ? (4y * (1 - y), 0.0) : (0.0, 0.0)), (14, 8)),
            ("x-periodic channel 12x7",
             (; bc = (W = :periodic, E = :periodic, S = :wall, N = :wall),
              source = (x, y) -> (1.0, 0.0, 0.0)), (12, 7)),
            ("doubly periodic Stokes 10x10",
             (; conv = false, bc = (W = :periodic, E = :periodic, S = :periodic, N = :periodic),
              source = (x, y) -> (sin(2pi * y), 0.0, 0.0)), (10, 10)),
        )
        laws = (("power law n = 0.5", _FVC3_PL_THIN, 1e-2),
                ("power law n = 1.5", _FVC3_PL_THICK, 1e-2),
                ("Carreau-Yasuda, eps = 0", _FVC3_CY, 0.0),
                ("Cross", Cross(0.1, 0.002, 2.0, 0.8), 1e-2),
                ("Herschel-Bulkley", HerschelBulkley(0.02, 0.05, 0.6; m_reg = 50.0), 1e-2))
        for (lname, law, eps) in laws, (gname, kw, (Nx, Ny)) in geoms
            @testset "$lname, $gname" begin
                p = fv_steady_problem(Nx, Ny; rheology = law, gnf_eps = eps, kw...)
                @test p.visc isa FVGNFViscosity
                S = fv_jacobian_pattern(p)
                colors, ncol = fv_jacobian_coloring(S)
                @test maximum(diff(sparse(transpose(S)).colptr)) <= 27   # 3x3 block: 27 per interior row
                @test ncol <= 45
                # the Newtonian pattern is contained in the GNF one
                p0 = fv_steady_problem(Nx, Ny; nu = 0.1, kw...)
                S0 = fv_jacobian_pattern(p0)
                @test all(S[k] for k in findall(S0))
                n = fv_nunknowns(p)
                s = fv_newton_solver(p)
                for trial in 1:2
                    U = trial == 1 ? zeros(n) : 0.3 .* randn(rng, n)
                    Jd = _fvc3_dense_jacobian(p, U)
                    @test all(isfinite, Jd)                   # finite at rest too
                    @test all(S[k] for k in findall(!iszero, Jd))
                    fv_jacobian!(s, U)
                    @test Matrix(s.J) == Jd                   # bit for bit
                    if trial == 2 && !(p.grid.conv == 0 && p.grid.bcN == FV_BC_PERIODIC)
                        # tight: every stored entry is nonzero on a random state
                        @test count(!iszero, nonzeros(s.J)) == nnz(S)
                    end
                    if trial == 2
                        asm = fv_jacobian_assembler(S, colors, ncol; chunk = 8, order = :csr)
                        nz = zeros(nnz(S))
                        fv_assemble_jacobian!(nz, asm, U, p)
                        @test nz == nonzeros(sparse(transpose(s.J)))
                        v = randn(rng, n); e = 1e-6
                        fd = (fv_residual(U .+ e .* v, p) .- fv_residual(U .- e .* v, p)) ./ (2e)
                        @test norm(s.J * v - fd) / norm(fd) < 1e-6
                    end
                end
            end
        end
        # interior-row count on a large grid (no boundary effect): 27 entries
        p = fv_steady_problem(40, 40; rheology = _FVC3_PL_THIN, gnf_eps = 1e-2,
                              bc = (W = :periodic, E = :periodic, S = :periodic, N = :periodic))
        S = fv_jacobian_pattern(p)
        @test nnz(S) == 27 * size(S, 1) - 26     # pinned continuity row: 1 entry
        @test_throws ArgumentError fv_steady_problem(8, 8; rheology = _FVC3_PL_THIN, gnf_eps = 0.0)
        @test_throws ArgumentError fv_steady_problem(8, 8; rheology = _FVC3_PL_THIN, gnf_eps = -1.0)
    end

    @testset "law parameters as a vector; ∂R/∂θ by duals (C4 path)" begin
        p = fv_steady_problem(12, 10; rheology = _FVC3_CY, gnf_eps = 1e-3, boundary_velocity = _fvc2_lid)
        @test fv_gnf_parameter_names(p) == (:eta_0, :eta_inf, :lambda, :a, :n)
        θ = fv_gnf_parameters(p)
        @test θ == [0.1, 0.002, 3.0, 2.0, 0.4]
        @test fv_with_gnf_parameters(p, θ).visc === p.visc
        q = fv_with_gnf_parameters(p, θ)
        @test q.nu === p.nu && q.src === p.src                   # arrays shared
        U = 0.3 .* randn(MersenneTwister(5), fv_nunknowns(p))
        @test fv_residual(U, q) == fv_residual(U, p)
        dRdθ = ForwardDiff.jacobian(t -> fv_residual(U, fv_with_gnf_parameters(p, t)), θ)
        for k in (1, 2, 3, 5)
            e = 1e-6 * θ[k]
            tp = copy(θ); tp[k] += e; tm = copy(θ); tm[k] -= e
            fd = (fv_residual(U, fv_with_gnf_parameters(p, tp)) .-
                  fv_residual(U, fv_with_gnf_parameters(p, tm))) ./ (2e)
            @test norm(dRdθ[:, k] - fd) / norm(fd) < 1e-6
        end
        pl = _fvc3_channel(_FVC3_PL_THIN, 1e-2, 8)
        @test fv_gnf_parameter_names(pl) == (:K, :n, :nu_min, :nu_max)
        Up = 0.1 .* randn(MersenneTwister(6), fv_nunknowns(pl))
        D = ForwardDiff.jacobian(t -> fv_residual(Up, fv_with_gnf_parameters(pl, [t[1], t[2], 0.0, Inf])),
                                 [0.1, 0.5])
        e = 1e-7
        fdK = (fv_residual(Up, _fvc3_channel(PowerLaw(0.1 + e, 0.5; nu_min = 0.0, nu_max = Inf), 1e-2, 8)) .-
               fv_residual(Up, _fvc3_channel(PowerLaw(0.1 - e, 0.5; nu_min = 0.0, nu_max = Inf), 1e-2, 8))) ./ (2e)
        @test norm(D[:, 1] - fdK) / norm(fdK) < 1e-6
    end

    @testset "n = 1 reproduces the Newtonian C2 result" begin
        nu = 0.01
        pn = fv_steady_problem(32, 32; nu, boundary_velocity = _fvc2_lid)
        pg = fv_steady_problem(32, 32; rheology = PowerLaw(nu, 1.0; nu_min = 0.0, nu_max = Inf),
                               gnf_eps = 1e-3, boundary_velocity = _fvc2_lid)
        U = randn(MersenneTwister(8), fv_nunknowns(pn))
        @test fv_residual(U, pg) == fv_residual(U, pn)          # bit for bit
        # Jacobian: same values on the Newtonian pattern, zeros elsewhere
        sn = fv_newton_solver(pn); sg = fv_newton_solver(pg)
        @test nnz(sg.J) > nnz(sn.J)
        fv_jacobian!(sn, U); fv_jacobian!(sg, U)
        @test sg.J == sn.J
        rn = solve_incns_fv_newton(; problem = pn)
        rg = solve_incns_fv_newton(; problem = pg)
        @test rn.converged && rg.converged
        @test rg.iterations == rn.iterations
        @test norm(rg.U - rn.U, Inf) <= 1e-12 * norm(rn.U, Inf)
        k = findall(>(1e-10), rn.residual_history)
        @test maximum(abs.(rg.residual_history[k] ./ rn.residual_history[k] .- 1)) < 1e-8
    end

    @testset "power-law Poiseuille: regularised reference, refinement, ε → 0, quadratic Newton" begin
        K, G = 0.1, 1.0      # wall shear rate (G/(2K))^(1/n): 25 (n = 0.5), 6.8 (n = 1.5)
        for (n, law) in ((0.5, _FVC3_PL_THIN), (1.5, _FVC3_PL_THICK))
            errs = Float64[]
            for Ny in (16, 32, 64)
                eps = 1e-2
                p = _fvc3_channel(law, eps, Ny; G)
                r = solve_incns_fv_newton(; problem = p, maxit = 40)
                y = [(j - 0.5) / Ny for j in 1:Ny]
                ur = [_fvc3_uref(law, eps, G, 1.0, yy) for yy in y]
                u = r.u[1, :]
                e = norm(u - ur) / norm(ur)
                push!(errs, e)
                @info "C3 power-law Poiseuille" n Ny eps iterations = r.iterations rel_L2_vs_regularised = e residual_2norm = r.residual_history
                @test r.converged
                @test e <= 0.02                                  # V1 gate: 2 %
                @test maximum(abs, r.v) < 1e-12
                @test maximum(abs.(r.u .- r.u[1:1, :])) < 1e-12   # x-invariant
                @test _fvc3_quadratic(r.residual_history)
                @test r.stats.n_factorize == 1
            end
            # second order (measured 2026-11-05, macOS CPU: ratios 4.0 / 4.0)
            @test errs[1] / errs[2] >= 3.5 && errs[2] / errs[3] >= 3.5
            # ε → 0: the regularised model tends to the power law, and the FV
            # solution at Ny = 64 with ε = 1e-4 to the analytical profile at
            # discretisation level (measured 2026-11-05: 7.7e-4 / 2.3e-4 for
            # n = 0.5 / 1.5; gate 2e-3)
            y = [(j - 0.5) / 64 for j in 1:64]
            ua = _fvc3_uan.(K, n, G, 1.0, y)
            dref = [norm([_fvc3_uref(law, eps, G, 1.0, yy) for yy in y] - ua) / norm(ua) for eps in (1e-1, 1e-2, 1e-3)]
            @test dref[1] > dref[2] > dref[3]
            r = solve_incns_fv_newton(; problem = _fvc3_channel(law, 1e-4, 64; G), maxit = 40)
            @test r.converged
            @test norm(r.u[1, :] - ua) / norm(ua) <= 2e-3
        end
    end

    @testset "2D: power-law cavity, quadratic Newton" begin
        # lid-driven cavity, shear-thinning n = 0.5, K = 0.01 (Re_K = U^(2-n) L^n / K = 100)
        p = fv_steady_problem(32, 32; rheology = PowerLaw(0.01, 0.5; nu_min = 0.0, nu_max = Inf),
                              gnf_eps = 1e-2, boundary_velocity = _fvc2_lid)
        s = fv_newton_solver(p)
        U = zeros(fv_nunknowns(p))
        st = fv_newton!(s, U; atol = 1e-14, rtol = 0.0, maxit = 30)
        @info "C3 Newton, power-law cavity n=0.5 32² (CPU)" residual_2norm = st.history alpha = st.alpha
        @test st.converged
        @test _fvc3_quadratic(st.history)
        @test st.inf[end] < 1e-14
        @test s.n_factorize == 1
        # finalize: cached factors are those of the exact GNF Jacobian (C4 entry)
        fv_newton_finalize!(s, U)
        Jd = _fvc3_dense_jacobian(p, U)
        b = randn(MersenneTwister(3), length(U)); x = similar(b)
        lin_solve!(x, s.cache, b; transpose = true)
        @test norm(transpose(Jd) * x - b) / norm(b) < 1e-12
    end

    @testset "Carreau-Yasuda obstacle channel vs the LBM GNF driver" begin
        # FV (h = 1, lattice units) and run_gnf_periodic_steady on the same
        # staircase geometry, law and force. Measured 2026-11-05 (macOS CPU),
        # relative L2 gap of u over fluid cells: 16.7 % (m = 4), 5.0 % (m = 8),
        # 1.6 % (m = 16); flux gap 15 %, 4.4 %, 1.4 %. The FV flux moves by
        # 2.3 % then 0.07 % under refinement, the LBM flux by 7.8 % then 3.0 %:
        # the gap is mostly the LBM staircase error, and it shrinks.
        gaps = Float64[]
        for m in (4, 8)
            c = _fvc3_obstacle(m)
            rl = run_gnf_periodic_steady(; is_solid = c.solid, closure = c.closure, Fx = c.g,
                                         tol = 1e-10, check_every = 20, fused = true)
            @test rl.converged
            p = fv_steady_problem(c.Nx, c.Ny; h = 1.0, rheology = c.closure, gnf_eps = 0.0, solid = c.solid,
                                  bc = (W = :periodic, E = :periodic, S = :wall, N = :wall),
                                  source = (x, y) -> (c.g, 0.0, 0.0))
            r = solve_incns_fv_newton(; problem = p, maxit = 40)
            @test r.converged
            @test _fvc3_quadratic(r.residual_history)
            fl = .!c.solid
            du = norm((r.u .- rl.ux)[fl]) / norm(rl.ux[fl])
            dq = abs(sum(r.u[1, :]) - sum(rl.ux[1, :])) / sum(rl.ux[1, :])
            @info "C3 CY obstacle channel FV vs LBM" m newton_iterations = r.iterations lbm_steps = rl.steps rel_L2_u = du flux_gap = dq residual_2norm = r.residual_history
            push!(gaps, du)
            m == 8 && @test du <= 0.06 && dq <= 0.05
        end
        @test gaps[2] <= gaps[1] / 2.5
    end

    @testset "platform contract: IncNS(:fv_newton) with a GNF problem" begin
        p = _fvc3_channel(_FVC3_PL_THIN, 1e-2, 16)
        sol = solve((; problem = p), IncNS(:fv_newton))
        @test norm(residual(p, IncNS(:fv_newton), Array(sample(sol, :U))), Inf) < 1e-12
    end
end
