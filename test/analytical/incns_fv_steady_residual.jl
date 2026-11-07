# C1 of the FV Newton roadmap: steady colocated FV residual as one
# KernelAbstractions kernel, generic in the element type
# (src/methods/inc_ns/fv_steady_residual.jl). CPU checks:
#   1. pressure pins (one per closed fluid pocket);
#   2. exact discrete solutions, built independently of the kernel, give R = 0;
#   3. manufactured solutions (variable viscosity, cross shear): velocity order 2
#      for Stokes, >= 1 for Navier-Stokes (first-order upwind);
#   4. element-type genericity: ForwardDiff duals in the state and in the
#      parameters agree with finite differences; Float32 agrees with Float64;
#      the Jacobian support lies in the 5x5 cell box (the colouring C2 relies on);
#   5. Newton on this residual vs the SIMPLE cavity driver (different
#      discretisations, agreement at discretisation level only);
#   6. platform contract (`SteadyResidual`, declared by IncNS(:fv_newton) only).
# Bit-for-bit parity with the J0 prototype is a separate script
# (benchmarks/fv_c1/j0_parity.jl, receipt in benchmarks/fv_c1/results/).
# GPU parity and allocations: incns_fv_steady_residual_gpu.jl.

using Test
using LinearAlgebra
using Random
using ForwardDiff
using KrakeNN

if !isdefined(@__MODULE__, :_fvc1_sparse_jacobian)
    include(joinpath(@__DIR__, "fv_steady_helpers.jl"))
end

const _FVC1_XPER = (W = :periodic, E = :periodic, S = :wall, N = :wall)
const _FVC1_WALLS = (W = :wall, E = :wall, S = :wall, N = :wall)
_fvc1_lid(s, x, y) = s === :N ? (1.0, 0.0) : (0.0, 0.0)

@testset "IncNS steady FV residual (C1, CPU)" begin

    @testset "pressure pins: one per closed fluid pocket" begin
        cav = fv_steady_problem(8, 8)
        @test findall(==(FV_CELL_PINNED), vec(cav.tag)) == [1]
        ch = fv_steady_problem(8, 4; bc = (W = :inlet, E = :outlet, S = :wall, N = :wall))
        @test count(==(FV_CELL_PINNED), ch.tag) == 0
        per = fv_steady_problem(8, 4; bc = _FVC1_XPER)
        @test count(==(FV_CELL_PINNED), per.tag) == 1
        # a solid column splits the cavity into two closed pockets
        solid = zeros(Bool, 9, 6); solid[5, :] .= true
        two = fv_steady_problem(9, 6; solid)
        @test findall(==(FV_CELL_PINNED), vec(two.tag)) == [1, 6]
        @test count(==(FV_CELL_SOLID), two.tag) == 6
        # an open channel holding a sealed fluid bubble: the bubble alone is pinned
        solid = zeros(Bool, 10, 7); solid[4:8, 2:6] .= true; solid[5:7, 3:5] .= false
        bub = fv_steady_problem(10, 7; solid, bc = (W = :inlet, E = :outlet, S = :wall, N = :wall))
        @test findall(==(FV_CELL_PINNED), vec(bub.tag)) == [5 + (3 - 1) * 10]
        # pins are recomputed from scratch
        @test fv_pressure_pins!(bub.tag, bub.grid) == 1
        @test_throws ArgumentError _fvc1_sparse_jacobian(per, zeros(fv_nunknowns(per)))  # 8 % 5 != 0
        @test_throws ArgumentError fv_steady_problem(4, 4; bc = (W = :periodic, E = :wall, S = :wall, N = :wall))
        @test_throws ArgumentError fv_steady_problem(4, 4; bc = (W = :slip, E = :wall, S = :wall, N = :wall))
        @test_throws DimensionMismatch fv_steady_problem(4, 4; solid = zeros(Bool, 3, 4))
        @test_throws DimensionMismatch fv_residual!(zeros(10), zeros(48), cav)
    end

    @testset "exact discrete solutions give R = 0" begin
        # (a) uniform flow in a doubly periodic box, Navier-Stokes
        N = 7
        p = fv_steady_problem(N, N; nu = 0.3,
                              bc = (W = :periodic, E = :periodic, S = :periodic, N = :periodic))
        U = fv_state(p, (x, y) -> 0.7, (x, y) -> -0.4, (x, y) -> 0.0)
        @test norm(fv_residual(U, p), Inf) < 1e-15

        # (b) Couette flow with nu(y), x-periodic, moving north wall, NS. The
        # discrete profile is built from the constant-shear-stress condition
        # tau = nu_f (u_{j+1} - u_j)/h (interior) and the half-cell wall gradients.
        Nx, Ny = 6, 9; h = 1 / Ny; tau = 0.8
        nuy = [1 + 0.5 * sin(3 * (j - 0.5) * h) for j in 1:Ny]
        u = zeros(Ny); u[1] = tau * h / (2 * nuy[1])
        for j in 1:(Ny - 1)
            u[j + 1] = u[j] + tau * h / ((nuy[j] + nuy[j + 1]) / 2)
        end
        ulid = u[Ny] + tau * h / (2 * nuy[Ny])
        p = fv_steady_problem(Nx, Ny; nu = [nuy[j] for i in 1:Nx, j in 1:Ny], bc = _FVC1_XPER,
                              boundary_velocity = (s, x, y) -> s === :N ? (ulid, 0.0) : (0.0, 0.0))
        U = fv_state(p, [u[j] for i in 1:Nx, j in 1:Ny], zeros(Nx, Ny), zeros(Nx, Ny))
        @test norm(fv_residual(U, p), Inf) < 1e-14

        # (c) body-force Poiseuille with nu(y), x-periodic, NS: profile from an
        # independent tridiagonal solve of the 1D flux balance
        #   nu_n (u_N - u_P) + nu_s (u_S - u_P) = -h^2 f   (wall: nu_P * 2 (0 - u_P))
        fx = 2.0
        A = zeros(Ny, Ny)
        for j in 1:Ny
            for (jj, wall) in ((j + 1, j == Ny), (j - 1, j == 1))
                if wall
                    A[j, j] += 2 * nuy[j]
                else
                    nf = (nuy[j] + nuy[jj]) / 2
                    A[j, j] += nf; A[j, jj] -= nf
                end
            end
        end
        up = A \ fill(h^2 * fx, Ny)
        p = fv_steady_problem(Nx, Ny; nu = [nuy[j] for i in 1:Nx, j in 1:Ny], bc = _FVC1_XPER,
                              source = (x, y) -> (fx, 0.0, 0.0))
        U = fv_state(p, [up[j] for i in 1:Nx, j in 1:Ny], zeros(Nx, Ny), zeros(Nx, Ny))
        @test norm(fv_residual(U, p), Inf) < 1e-12 * h^2 * fx   # round-off of the h^2 f scale
        # discrete Poiseuille vs the analytic parabola at constant nu (V1-type check)
        p1 = fv_steady_problem(10, 32; nu = 1.0, bc = _FVC1_XPER, source = (x, y) -> (fx, 0.0, 0.0))
        U1, it = _fvc1_newton(p1, zeros(fv_nunknowns(p1)))
        uu, _, _ = fv_fields(p1, U1)
        yc = ((1:32) .- 0.5) ./ 32
        uan = fx / 2 .* yc .* (1 .- yc)
        @test it <= 2                                   # Stokes-like: one Newton step
        @test norm(uu[3, :] - uan) / norm(uan) < 0.01   # V1 gate (1 %)
    end

    @testset "manufactured solutions, variable viscosity" begin
        # Measured 2026-11-04 (macOS CPU, Float64): Stokes x-periodic/y-walls
        # velocity L2 orders 2.04, 2.02 (10->20->40), pressure 0.96, 1.47; all walls
        # 1.98, 2.00; NS velocity 1.68, 1.49 (tends to 1, first-order upwind).
        # The pressure error concentrates in the wall rows (first order there,
        # second order inside: half-cell wall closure of the normal stress).
        Ns = (10, 20, 40)
        st = [_fvc1_mms_errors(N, _FVC1_XPER, false) for N in Ns]
        ou = [log2(st[k][1] / st[k + 1][1]) for k in 1:2]
        op = log2(st[2][2] / st[3][2])
        @test all(>=(1.9), ou)       # plan C1: Stokes, variable nu, order 2
        @test op >= 1.3
        @test all(s -> s[3] <= 2, st)  # linear problem: one Newton step (+ check)
        sw = [_fvc1_mms_errors(N, _FVC1_WALLS, false) for N in Ns]
        @test all(>=(1.9), [log2(sw[k][1] / sw[k + 1][1]) for k in 1:2])
        ns = [_fvc1_mms_errors(N, _FVC1_XPER, true) for N in Ns[2:3]]
        @test log2(ns[1][1] / ns[2][1]) >= 1.0   # plan C1: NS order 1
        @test ns[2][1] < 5e-3
    end

    @testset "generic element type" begin
        rng = MersenneTwister(11)
        solid = zeros(Bool, 12, 9); solid[5:7, 1:4] .= true
        cases = (
            fv_steady_problem(9, 9; nu = (x, y) -> 0.02 * (1 + x * y), boundary_velocity = _fvc1_lid),
            fv_steady_problem(12, 9; nu = 0.05, solid,
                              bc = (W = :inlet, E = :outlet, S = :wall, N = :wall),
                              boundary_velocity = (s, x, y) -> s === :W ? (4y * (1 - y), 0.0) : (0.0, 0.0)),
            fv_steady_problem(10, 10; nu = (x, y) -> 1 + 0.3 * sin(6x), bc = (W = :periodic, E = :periodic,
                              S = :periodic, N = :periodic), source = (x, y) -> (cos(6y), 0.0, 0.0)),
        )
        for p in cases
            n = fv_nunknowns(p)
            U = randn(rng, n)
            Jd = ForwardDiff.jacobian(x -> fv_residual(x, p), U)
            # J v by duals vs central finite differences
            v = randn(rng, n); e = 1e-6
            jv_fd = (fv_residual(U .+ e .* v, p) .- fv_residual(U .- e .* v, p)) ./ (2e)
            @test norm(Jd * v - jv_fd) / norm(jv_fd) < 1e-7
            # support of dR/dU lies in the 5x5 cell box: the box-coloured
            # sparse Jacobian reproduces the dense one exactly
            @test Matrix(_fvc1_sparse_jacobian(p, U)) == Jd
            # duals in the parameters (C4 path): d R / d(scale of nu)
            pd = FVSteadyProblem2D(p.grid, p.tag, p.nu .* ForwardDiff.Dual(1.0, 1.0), p.bvel, p.src)
            Rd = fv_residual(U, pd)
            @test eltype(Rd) <: ForwardDiff.Dual
            ps(s) = FVSteadyProblem2D(p.grid, p.tag, p.nu .* s, p.bvel, p.src)
            dfd = (fv_residual(U, ps(1 + e)) .- fv_residual(U, ps(1 - e))) ./ (2e)
            @test norm(ForwardDiff.partials.(Rd, 1) .- dfd) / norm(dfd) < 1e-7
            @test ForwardDiff.value.(Rd) == fv_residual(U, p)
        end
        # Float32 arrays and state: same residual to single precision
        p64 = cases[1]
        p32 = fv_steady_problem(9, 9; nu = (x, y) -> 0.02 * (1 + x * y), boundary_velocity = _fvc1_lid,
                                T = Float32)
        U = randn(rng, fv_nunknowns(p64))
        R32 = fv_residual(Float32.(U), p32)
        @test eltype(R32) == Float32
        @test norm(R32 - fv_residual(U, p64)) / norm(fv_residual(U, p64)) < 1e-5
    end

    @testset "Newton on the FV residual vs SIMPLE cavity (Re = 100, 32²)" begin
        # Different discretisations (SIMPLE: relaxed Rhie-Chow, its own
        # convection); measured 2026-11-04 macOS CPU: max centreline difference
        # 1.2e-3 (u) and 1.3e-3 (v). Gate 5e-3 of the lid velocity.
        N = 32
        p = fv_steady_problem(N, N; nu = 0.01, boundary_velocity = _fvc1_lid)
        U, it = _fvc1_newton(p, zeros(fv_nunknowns(p)))
        @test it <= 8
        @test norm(fv_residual(U, p), Inf) < 1e-11
        u, v, _ = fv_fields(p, U)
        s = solve_incns_cavity(; nx = N, ny = N, Re = 100.0, tol = 1e-9, maxiter = 20000)
        @test s.converged
        uc = (u[N ÷ 2, :] .+ u[N ÷ 2 + 1, :]) ./ 2
        us = (s.u[N ÷ 2, :] .+ s.u[N ÷ 2 + 1, :]) ./ 2
        vc = (v[:, N ÷ 2] .+ v[:, N ÷ 2 + 1]) ./ 2
        vs = (s.v[:, N ÷ 2] .+ s.v[:, N ÷ 2 + 1]) ./ 2
        @test maximum(abs, uc - us) < 5e-3
        @test maximum(abs, vc - vs) < 5e-3
    end

    @testset "platform contract: SteadyResidual" begin
        @test SteadyResidual in capabilities(IncNS(:fv_newton))
        # decision 2026-11-04: SteadyResidual is driver-dependent, the SIMPLE
        # drivers no longer declare it
        @test SteadyResidual ∉ capabilities(IncNS(:cavity))
        p = fv_steady_problem(6, 5; nu = 0.1, boundary_velocity = _fvc1_lid)
        U = randn(MersenneTwister(3), fv_nunknowns(p))
        @test residual(p, IncNS(:fv_newton), U) == fv_residual(U, p)
        @test residual(p, IncNS(:fv_newton), U, nothing) == fv_residual(U, p)
        @test_throws ArgumentError residual(p, IncNS(:cavity), U)
        # host round trip of the problem arrays
        q = fv_adapt(Array, p)
        @test q.nu == p.nu && q.tag == p.tag && q.grid === p.grid
        u, v, pp = fv_fields(p, U)
        @test fv_state(p, u, v, pp) == U
        @test fv_index(p.grid, 2, 3, 3) == 3 * (1 + 2 * 6) + 3
    end
end
