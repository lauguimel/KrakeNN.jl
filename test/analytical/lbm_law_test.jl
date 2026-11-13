# Scalar laws (ClassicalLaw, DenseNet) in the 2D LBM GNF step through
# LBMLawViscosity, and the steady adjoint gradient with respect to the law
# parameters. CPU, ~20 s after compilation.
#
#   (a) ClassicalLaw(CarreauYasuda) with eps = 0 reproduces the CarreauYasuda
#       closure bit for bit (f, tau), split and fused paths, Poiseuille 4x16 and
#       the obstacle channel 16x8; same per-cell adjoint gradient as the analytic
#       parameter route (gnf_parameter_gradient).
#   (b) DenseNet law, periodic channel H = 1, body force G = 1, Ny = 16 and 32:
#       LBM (diffusive scaling dt = dx^2) vs FV (same law, theta, eps).
#       Measured 2026-11-09: relative L2 gap of u 8.5e-3 (Ny 16), 2.1e-3 (Ny 32);
#       mean-velocity gap 9.3e-3, 2.3e-3 (second order, ratio 4.0).
#   (c) LBM adjoint gradient (all weights) vs central differences on random
#       directions, steps 1e-2..1e-5: relative error 2.7e-5, 2.8e-7, 1.5e-8,
#       5.6e-8 (V shape, minimum at 1e-4).
#   (d) LBM vs FV gradient of J = mean u: relative gap 1.2e-2 (Ny 16),
#       3.1e-3 (Ny 32), shrinking with the discretisation gap.
# Bounds below are about twice the measured values.
using Test, KrakeNN, Random, LinearAlgebra, Printf

const _LL_G = 1.0
const _LL_NX = 4
const _LL_EPS = 1e-3

function _ll_net()
    net = DenseNet(1, (8, 8); activation=:tanh, output=:exp, scale=(5.0,))
    theta = law_init(net; rng=MersenneTwister(3)) .* 10   # nu varies by ~25 % in the channel
    theta[end] = log(0.1)
    return net, theta
end

# Law units: H = 1. Lattice: dx = 1/Ny, dt = dx^2, so nu_lat = nu, gamma = gamma_lat/dt.
_ll_closure(net, theta, Ny) =
    LBMLawViscosity(net, theta; eps=_LL_EPS, nu_scale=1.0, gamma_scale=Float64(Ny)^2)
function _ll_lbm(net, theta, Ny; tol=1e-12, f_init=nothing, tau_init=nothing,
                 max_steps=1_000_000)
    return run_gnf_periodic_steady(; is_solid=falses(_LL_NX, Ny),
        closure=_ll_closure(net, theta, Ny), Fx=_LL_G / Ny^3, tol, check_every=20,
        fused=true, f_init, tau_init, max_steps)
end
_ll_fvprob(net, theta, Ny) = fv_steady_problem(_LL_NX, Ny; h=1 / Ny,
    bc=(W=:periodic, E=:periodic, S=:wall, N=:wall), source=(x, y) -> (_LL_G, 0.0, 0.0),
    viscosity=FVLawViscosity(net, theta; eps=_LL_EPS))
_ll_rel(a, b) = norm(a .- b) / max(norm(b), eps(Float64))

@testset "Scalar laws in the LBM GNF step" begin
    @testset "(a) ClassicalLaw(CarreauYasuda) == CarreauYasuda closure, bit for bit" begin
        cy = CarreauYasuda(0.17, 0.0034, 20.0, 2.0, 0.7)
        law = ClassicalLaw(cy)
        cl = LBMLawViscosity(law, law_init(law); eps=0.0)
        @test gnf_viscosity(cl, 0.0) == gnf_viscosity(cy, 0.0)
        @test gnf_viscosity(cl, 3.7e-5) == gnf_viscosity(cy, 3.7e-5)
        m = 2
        obstacle = [(((i - 0.5) / m - 4)^2 + ((j - 0.5) / m)^2 < 1.5^2) ||
                    (((i - 0.5) / m - 4)^2 + ((j - 0.5) / m - 4)^2 < 1.5^2) for i in 1:8m, j in 1:4m]
        for (solid, Fx) in ((falses(4, 16), 1e-5), (obstacle, 0.3 * 5 * 0.17^2 / m^3)),
            fused in (false, true)
            kw = (; is_solid=solid, Fx, tol=1e-10, check_every=20, fused)
            r0 = run_gnf_periodic_steady(; closure=cy, kw...)
            r1 = run_gnf_periodic_steady(; closure=cl, kw...)
            @test r0.converged && r1.steps == r0.steps
            @test r1.f == r0.f && r1.tau == r0.tau && r1.nu == r0.nu
        end
        # Adjoint: law pullback == analytic-parameter ForwardDiff route, same λ.
        r = run_gnf_periodic_steady(; is_solid=obstacle, closure=cy, Fx=0.3 * 5 * 0.17^2 / m^3,
                                    tol=1e-10, check_every=20, fused=true)
        Nx, Ny = size(obstacle)
        gu = (ones(Nx, Ny) ./ (Nx * Ny), zeros(Nx, Ny))
        res = gnf_steady_law_gradient(r.f, r.tau, gu; is_solid=obstacle, closure=cl,
                                      Fx=0.3 * 5 * 0.17^2 / m^3, solver=:gmres, tol=1e-12,
                                      restart=1000, maxiter=20_000)
        @test res.stats.converged
        θ = law_init(law)
        closure_of = t -> CarreauYasuda(t[1], t[2], t[3], t[4], t[5])
        gref, _ = gnf_parameter_gradient(res.lin, res.stats.λ, θ, closure_of)
        err = _ll_rel(res.grad, gref)
        @info "LBM law (a) CY pullback vs analytic route" relative = err
        @test err <= 1e-12
    end

    net, theta = _ll_net()
    @testset "constructor contract" begin
        @test_throws ArgumentError LBMLawViscosity(net, theta; eps=0.0)
        @test_throws DimensionMismatch LBMLawViscosity(net, theta[1:end-1])
        @test_throws ArgumentError LBMLawViscosity(DenseNet(2, (3,)), zeros(13))
        @test_throws ArgumentError LBMLawViscosity(net, theta; nu_scale=0.0)
        c32 = KrakeNN.gnf_convert_closure(Float32, LBMLawViscosity(net, theta))
        @test eltype(c32.theta) === Float32 && c32.eps isa Float32
    end

    gaps = Float64[]; qgaps = Float64[]; ggaps = Float64[]
    base16 = Ref{Any}(nothing)
    @testset "(b, d) DenseNet channel: LBM vs FV, values and gradients, Ny = $Ny" for Ny in (16, 32)
        rl = _ll_lbm(net, theta, Ny)
        @test rl.converged && rl.finite
        ul = rl.ux .* Ny                       # u = u_lat dx/dt
        p = _ll_fvprob(net, theta, Ny)
        s = fv_newton_solver(p)
        U = zeros(fv_nunknowns(p))
        @test fv_newton!(s, U; atol=1e-13, rtol=0.0, maxit=30).converged
        uf = [U[fv_index(p.grid, i, j, 1)] for i in 1:_LL_NX, j in 1:Ny]
        push!(gaps, _ll_rel(uf, ul))
        push!(qgaps, abs(sum(uf) - sum(ul)) / sum(ul))
        numin, numax = extrema(Array(rl.nu))
        @test numax / numin > 1.2              # the law is not Newtonian on this flow
        # J = mean u, on both solvers
        nc = _LL_NX * Ny
        gu = (fill(Float64(Ny) / nc, _LL_NX, Ny), zeros(_LL_NX, Ny))
        gl = gnf_steady_law_gradient(rl.f, rl.tau, gu; is_solid=falses(_LL_NX, Ny),
            closure=_ll_closure(net, theta, Ny), Fx=_LL_G / Ny^3, solver=:gmres, tol=1e-12,
            restart=1000, maxiter=20_000)
        @test gl.stats.converged
        @test length(gl.grad) == law_nparams(net) && all(isfinite, gl.grad)
        map = FVLawParameters(length(theta))
        obj = FVLinearFunctional([fv_index(p.grid, i, j, 1) for i in 1:_LL_NX for j in 1:Ny],
                                 fill(1 / nc, nc))
        gf = fv_steady_gradient(s, U, theta, obj; params=map, ws=fv_adjoint_workspace(s, map))
        push!(ggaps, _ll_rel(gl.grad, gf.grad))
        @info "LBM law (b, d) DenseNet channel" Ny lbm_steps = rl.steps nu_range = (numin, numax) rel_L2_u = gaps[end] mean_u_gap = qgaps[end] grad_gap = ggaps[end] t_lin = gl.t_lin t_adjoint = gl.t_adjoint t_grad = gl.t_grad
        Ny == 16 && (base16[] = (; rl, grad=gl.grad))
    end
    @test gaps[1] <= 1.7e-2 && gaps[2] <= 4.2e-3
    @test qgaps[1] <= 1.9e-2 && qgaps[2] <= 4.6e-3
    @test gaps[2] <= gaps[1] / 3                    # second order: measured ratio 4.0
    @test ggaps[1] <= 2.5e-2 && ggaps[2] <= 6.2e-3
    @test ggaps[2] <= ggaps[1] / 3

    @testset "(c) LBM adjoint vs central differences, all weights (Ny = 16)" begin
        Ny = 16
        rng = MersenneTwister(11)
        J(r) = sum(r.ux .* Ny) / (_LL_NX * Ny)
        for k in 1:2
            d = normalize(randn(rng, length(theta)))
            exact = dot(base16[].grad, d)
            errs = Float64[]
            for h in (1e-2, 1e-3, 1e-4, 1e-5)
                Jp = J(_ll_lbm(net, theta .+ h .* d, Ny; tol=0.0, f_init=base16[].rl.f,
                               tau_init=base16[].rl.tau, max_steps=6000))
                Jm = J(_ll_lbm(net, theta .- h .* d, Ny; tol=0.0, f_init=base16[].rl.f,
                               tau_init=base16[].rl.tau, max_steps=6000))
                push!(errs, abs((Jp - Jm) / (2h) - exact) / abs(exact))
            end
            @info "LBM law (c) adjoint vs FD" direction = k steps = (1e-2, 1e-3, 1e-4, 1e-5) rel_err = Tuple(errs)
            @test minimum(errs) <= 1e-7
            @test errs[1] > 10 * errs[3]             # truncation branch of the V
        end
    end
end
