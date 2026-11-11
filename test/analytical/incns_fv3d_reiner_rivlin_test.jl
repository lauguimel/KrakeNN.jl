# Second normal stress difference in the 3D steady FV residual
# (FVReinerRivlinViscosity, tau = 2 eta D + 4 Psi2 D.D), CPU only, Float64:
#   (a) stress hook in simple shear: N1 = 0, N2 = Psi2 gdot^2; Giesekus closed
#       forms against the textbook expressions; tabulated law with c = 1;
#   (b) Psi2 = 0 reduces to the eta model: residual and Jacobian bit for bit,
#       Newton solution to round-off, for a GNF eta and a law eta;
#   (c) coloured Jacobian == dense ForwardDiff Jacobian, pattern superset, and
#       J d against centred finite differences of the residual;
#   (d) Psi2 = c eta: the continuous forcing is a pure gradient, so a
#       rectilinear duct flow stays rectilinear. The FV forcing is not an exact
#       discrete gradient: the secondary flow is a discretisation error, small
#       against the one of a constant Psi2 = c eta_0 (not proportional) and
#       decreasing with refinement (gates set from the expected O(h) to O(h^2)
#       decay, not measured);
#   (e) adjoint gradient along Psi2 (and eta) parameters vs centred FD of the
#       converged objective, and adjoint == forward tangent;
#   (f) 16 x 16 duct slice, closed-form Giesekus (alpha = 0.4, Wi = 0.4):
#       eight-vortex sign pattern of the reference viscoelastic solution (flow
#       from the corners to the centre along the diagonals, toward the walls
#       along the wall bisectors), mirror symmetries, magnitude window;
#   (g) inference and zero allocation of the cell residual with the new model.
using Test
using LinearAlgebra
using Random
using ForwardDiff
using KrakeNN

const _RR_BC = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)
const _RR_NEWTON = (; atol = 1e-14, rtol = 1e-15, maxit = 60)

_rr_dense_jacobian(p, U) = ForwardDiff.jacobian((R, X) -> fv_residual!(R, X, p), zeros(length(U)), U)

# 1 x N x N periodic slice of a square duct of side 1, mean axial velocity 1
_rr_slice(N, visc; conv = true, F0 = 50.0) =
    fv_steady_problem(1, N, N; h = 1 / N, bc = _RR_BC, conv, viscosity = visc,
                      source = (x, y, z) -> (F0, 0.0, 0.0, 0.0), u_mean_target = 1.0)

_rr_giesekus(lam, alpha; eta_s = 1.0, eta_p = 1.0, eps = 1e-4) =
    FVReinerRivlinViscosity(FVLawViscosity(GiesekusViscosityLaw(), [eta_s, eta_p, lam, alpha]; eps),
                            FVLawViscosity(GiesekusPsi2Law(), [eta_p, lam, alpha]; eps))

function _rr_solve(p, U = KrakeNN._fv_initial_force!(zeros(fv_nunknowns(p)), p); s = fv_newton_solver(p))
    st = fv_newton!(s, U; _RR_NEWTON...)
    st.converged || st.stagnated || error("Newton did not converge")
    return s, U, st
end

# secondary speed |(v, w)| on the slice (Nx = 1)
function _rr_secondary(p, U)
    _, v, w, _ = fv_fields(p, U)
    return v[1, :, :], w[1, :, :], sqrt.(v[1, :, :] .^ 2 .+ w[1, :, :] .^ 2)
end

@testset "3D FV Reiner-Rivlin stress (Psi2)" begin
    @testset "(a) simple shear and Giesekus closed forms" begin
        gd = 0.7; psi = -0.3
        G = (0.0, gd, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)     # u = gd y
        tx = KrakeNN._fv3_add_dd((0.0, 0.0, 0.0), psi, G, 1, 1)
        ty = KrakeNN._fv3_add_dd((0.0, 0.0, 0.0), psi, G, 2, 1)
        tz = KrakeNN._fv3_add_dd((0.0, 0.0, 0.0), psi, G, 3, 1)
        @test tx[1] - ty[2] == 0                              # N1 = 0
        @test ty[2] - tz[3] ≈ psi * gd^2 rtol = 1e-14         # N2 = Psi2 gdot^2
        @test tx[2] == 0 && tx[3] == 0 && ty[1] == 0 && tz[1] == 0   # no shear stress added
        @test collect(KrakeNN._fv3_add_dd((1.0, 2.0, 3.0), psi, G, 2, -1)) ≈ [1.0, 2.0 - psi * gd^2, 3.0] rtol = 1e-14
        a = 0.4
        # textbook form (cancellation at small Wi: not below Wi = 0.1 at rtol 1e-8)
        for wi in (0.1, 0.3, 1.0, 5.0)
            chi = sqrt((sqrt(1 + 16a * (1 - a) * wi^2) - 1) / (8a * (1 - a) * wi^2))
            f = (1 - chi) / (1 + (1 - 2a) * chi)
            # lambda = 2, gdot = wi / 2, eta_p0 = 1.5, eta_s = 0.5
            @test law_value(GiesekusViscosityLaw(), (wi / 2,), [0.5, 1.5, 2.0, a]) ≈
                  0.5 + 1.5 * (1 - f)^2 / (1 + (1 - 2a) * f) rtol = 1e-8
            @test law_value(GiesekusPsi2Law(), (wi / 2,), [1.5, 2.0, a]) ≈ -1.5 * 2.0 * f / wi^2 rtol = 1e-8
        end
        @test law_value(GiesekusPsi2Law(), (1e-12,), [1.0, 1.0, a]) ≈ -a rtol = 1e-12   # g(0) = alpha
        tab = TabulatedGiesekusPsi2Law(collect(-1.5:0.25:1.0))
        @test law_nparams(tab) == 14
        for gdot in (1e-3, 0.2, 1.0, 40.0)
            @test law_value(tab, (gdot,), [1.0, 1.0, a, ones(11)...]) ≈
                  law_value(GiesekusPsi2Law(), (gdot,), [1.0, 1.0, a]) rtol = 1e-14
        end
        c = collect(range(0.8, 1.2; length = 11))
        ps(gdot) = law_value(tab, (gdot,), [1.0, 1.0, a, c...]) / law_value(GiesekusPsi2Law(), (gdot,), [1.0, 1.0, a])
        @test ps(1e-4) ≈ 0.8 rtol = 1e-12                      # clamped below the first node
        @test ps(1e3) ≈ 1.2 rtol = 1e-12                       # clamped above the last node
        @test ps(exp10(-0.375)) ≈ (c[5] + c[6]) / 2 rtol = 1e-12   # midpoint of a hat
        @test_throws ArgumentError TabulatedGiesekusPsi2Law([0.0, 0.0])
    end

    @testset "(b) Psi2 = 0 reduces to the eta model" begin
        rng = MersenneTwister(5)
        cy = CarreauYasuda(2.0, 0.2, 1.5, 2.0, 0.5)
        etas = (("GNF eta", FVGNFViscosity(cy, 1e-3)),
                ("law eta", FVLawViscosity(GiesekusViscosityLaw(), [1.0, 1.0, 0.4, 0.4]; eps = 1e-4)))
        for (name, eta) in etas
            @testset "$name" begin
                rr = FVReinerRivlinViscosity(eta, FVLawViscosity(GiesekusPsi2Law(), [0.0, 0.4, 0.4]; eps = 1e-4))
                p0 = _rr_slice(6, eta); p1 = _rr_slice(6, rr)
                n = fv_nunknowns(p0)
                @test fv_nunknowns(p1) == n
                for trial in 1:2
                    U = 0.5 .* randn(rng, n)
                    R0 = zeros(n); R1 = zeros(n)
                    fv_residual!(R0, U, p0); fv_residual!(R1, U, p1)
                    @test R1 == R0
                    @test _rr_dense_jacobian(p1, U) == _rr_dense_jacobian(p0, U)
                end
                _, U0, st0 = _rr_solve(p0)
                _, U1, st1 = _rr_solve(p1)
                @test last(st0.inf) < 1e-11 && last(st1.inf) < 1e-11
                @test norm(U1 - U0, Inf) <= 1e-12 * norm(U0, Inf)
                @test maximum(_rr_secondary(p1, U1)[3]) <= 1e-12   # GNF: no secondary flow
            end
        end
    end

    @testset "(c) Jacobian: coloured == dense, centred FD" begin
        rng = MersenneTwister(7)
        rr = _rr_giesekus(0.5, 0.3)
        probs = (("generic 3 x 4 x 3", fv_steady_problem(3, 4, 3; h = 0.25, bc = _RR_BC, viscosity = rr,
                                                         boundary_velocity = (s, x, y, z) ->
                                                             s === :T ? (0.3, 0.1, 0.0) : (0.0, 0.0, 0.0))),
                 ("slice 1 x 5 x 5 flow rate", _rr_slice(5, rr)))
        for (name, p) in probs
            @testset "$name" begin
                S = fv_jacobian_pattern(p)
                s = fv_newton_solver(p)
                n = fv_nunknowns(p)
                for trial in 1:2
                    U = 0.5 .* randn(rng, n)
                    Jd = _rr_dense_jacobian(p, U)
                    @test all(S[k] for k in findall(!iszero, Jd))
                    fv_jacobian!(s, U)
                    @test Matrix(s.J) == Jd
                    d = randn(rng, n)
                    e = 1e-6
                    Rp = zeros(n); Rm = zeros(n)
                    fv_residual!(Rp, U .+ e .* d, p); fv_residual!(Rm, U .- e .* d, p)
                    fd = (Rp .- Rm) ./ (2e)
                    Jdd = Jd * d
                    @test norm(fd - Jdd, Inf) <= 1e-6 * norm(Jdd, Inf)
                end
            end
        end
    end

    @testset "(d) Psi2 = c eta: secondary flow is a discretisation error" begin
        # eta: Carreau (shear thinning, so the constant-Psi2 forcing is not a gradient)
        eta0, etainf, lam, a, nidx = 2.0, 0.2, 1.5, 2.0, 0.5
        c = -0.2
        eta = FVGNFViscosity(CarreauYasuda(eta0, etainf, lam, a, nidx), 1e-3)
        prop = FVReinerRivlinViscosity(eta, FVLawViscosity(ClassicalLaw(CarreauYasuda(eta0, etainf, lam, a, nidx)),
                                                         [c * eta0, c * etainf, lam, a, nidx]; eps = 1e-3))
        cons = FVReinerRivlinViscosity(eta, FVLawViscosity(ClassicalLaw(CarreauYasuda(eta0, etainf, lam, a, nidx)),
                                                         [c * eta0, c * eta0, lam, a, nidx]; eps = 1e-3))
        ratio = Dict{Int,Float64}()
        for N in (8, 16)
            _, Up, stp = _rr_solve(_rr_slice(N, prop; conv = false))
            _, Uc, stc = _rr_solve(_rr_slice(N, cons; conv = false))
            @test last(stp.inf) < 1e-11 && last(stc.inf) < 1e-11
            sp = maximum(_rr_secondary(_rr_slice(N, prop), Up)[3])
            sc = maximum(_rr_secondary(_rr_slice(N, cons), Uc)[3])
            @test sc > 1e-5                                   # the constant-Psi2 case does drive a flow
            ratio[N] = sp / sc
            @info "Psi2 = c eta vs constant Psi2" N max_sec_prop = sp max_sec_const = sc ratio = ratio[N]
        end
        @test ratio[16] <= 0.25
        @test ratio[16] <= 0.75 * ratio[8]
    end

    @testset "(e) adjoint vs centred FD and tangent" begin
        N = 8
        lam, alpha = 0.4, 0.4
        rr = _rr_giesekus(lam, alpha)
        base = _rr_slice(N, rr)
        map = FVReinerRivlinParameters(rr)
        @test map.neta == 4 && map.npsi == 3 && fv_nparams(map, base) == 7
        θ = [1.0, 1.0, lam, alpha, 1.0, lam, alpha]
        g = base.grid
        idx = Int[]
        for j in 2:3, comp in (2, 3)
            push!(idx, fv_index(g, 1, j, j, comp))
        end
        obj = FVLinearFunctional(idx, ones(length(idx)))
        U0 = KrakeNN._fv_initial_force!(zeros(fv_nunknowns(base)), base)
        s, U, st = _rr_solve(base, copy(U0))
        @test last(st.inf) < 1e-11
        @test fv_objective(obj, U, base) > 0                  # diagonal flow toward the centre
        ws = fv_adjoint_workspace(s, map)
        r = fv_steady_gradient(s, U, θ, obj; params = map, ws)
        @test r.residual_inf <= 1e-12
        @test all(isfinite, r.grad)
        sfd = fv_newton_solver(base)
        function J_at(θp)
            fv_set_params!(sfd, map, θp)
            _, Up, _ = _rr_solve(sfd.prob, copy(U0); s = sfd)
            return fv_objective(obj, Up, sfd.prob)
        end
        rng = MersenneTwister(13)
        dirs = (Float64.(1:7 .== 5), Float64.(1:7 .== 6), Float64.(1:7 .== 3), randn(rng, 7))
        for d in dirs
            gdd = dot(r.grad, d)
            errs = Float64[]
            for step in (1e-3, 1e-4)
                ε = step / norm(d, Inf)
                push!(errs, abs((J_at(θ .+ ε .* d) - J_at(θ .- ε .* d)) / (2ε) - gdd))
            end
            @test minimum(errs) <= 1e-8 + 1e-5 * abs(gdd)
            @info "Reiner-Rivlin adjoint vs FD" d gdd err = minimum(errs)
        end
        d = dirs[end]
        dU = fv_steady_tangent(s, U, θ, d; params = map, ws)
        tg = dot(Array(r.gU), dU) + dot(r.explicit, d)
        @test abs(tg - dot(r.grad, d)) <= 1e-10 * max(abs(tg), norm(r.grad) * norm(d))
        @test_throws DimensionMismatch fv_adjoint_workspace(s, FVReinerRivlinParameters(4, 2))
    end

    @testset "(f) 16 x 16 Giesekus slice: eight-vortex sign pattern" begin
        N = 16
        p = _rr_slice(N, _rr_giesekus(0.4, 0.4))
        _, U, st = _rr_solve(p)
        @test last(st.inf) < 1e-11
        v, w, sp = _rr_secondary(p, U)
        m = maximum(sp)
        @test 1e-4 < m < 1e-2
        @info "Giesekus slice" N max_sec = m iterations = st.iterations
        tol = 1e-6 * m + 1e-13
        M = N + 1
        # mirror y -> 1 - y, z -> 1 - z and the diagonal y <-> z
        @test maximum(abs(v[j, k] + v[M - j, k]) for j in 1:N, k in 1:N) <= tol
        @test maximum(abs(v[j, k] - v[j, M - k]) for j in 1:N, k in 1:N) <= tol
        @test maximum(abs(w[j, k] - w[M - j, k]) for j in 1:N, k in 1:N) <= tol
        @test maximum(abs(w[j, k] + w[j, M - k]) for j in 1:N, k in 1:N) <= tol
        @test maximum(abs(v[j, k] - w[k, j]) for j in 1:N, k in 1:N) <= tol
        # diagonal, lower-left quadrant: from the corner toward the centre
        @test all(v[j, j] > 0 && w[j, j] > 0 for j in 2:6)
        # wall bisectors (between cells N/2 and N/2 + 1): toward the walls
        @test all((v[j, 8] + v[j, 9]) / 2 < 0 for j in 3:6)
        @test all((w[8, k] + w[9, k]) / 2 < 0 for k in 3:6)
    end

    @testset "(g) cell residual: inference and allocation" begin
        S = ForwardDiff.Dual{Nothing,Float64,2}
        p = _rr_slice(4, _rr_giesekus(0.4, 0.4))
        for U in (fill(0.1, fv_nunknowns(p)), fill(S(0.1), fv_nunknowns(p)))
            T = promote_type(eltype(U), eltype(p.nu), KrakeNN._fv_visc_eltype(p.visc))
            R = zeros(T, length(U))
            @inferred KrakeNN._fv3_cell_residual(U, p.tag, p.nu, p.bvel, p.src, p.grid, 1, 2, 2, p.visc)
            cell!() = KrakeNN._fv3_residual_cell!(R, U, p.tag, p.nu, p.bvel, p.src, p.grid, p.visc, 1, 2, 2)
            cell!()
            @test (@allocated cell!()) == 0
        end
        @test_throws ArgumentError _rr_slice(4, FVReinerRivlinViscosity(FVNewtonianViscosity(), p.visc.psi2))
    end
end
