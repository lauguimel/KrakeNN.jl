# Scalar laws (ClassicalLaw, DenseNet) in the 3D D3Q19 GNF step through
# LBMLawViscosity (`effective_viscosity(c, γ̇) = gnf_viscosity(c, γ̇²/2)`, the
# 2D convention on I1 = S:S). CPU.
#
#   (a) ClassicalLaw(CarreauYasuda) with eps = 0 reproduces the CarreauYasuda
#       closure bit for bit in `run_gnf_channel_3d` (f, sp, sm, u, steps), for
#       halfway (q_w = 1/2) and LI-BB (q_w = 0.3) walls.
#   (b) DenseNet law, plane channel H = 1 (flow x, walls normal to y, periodic
#       in x and in the spanwise direction z), body force G = 1, N = 16 and 32
#       wall-normal nodes, diffusive scaling dt = dx^2: 3D LBM vs the FV
#       solution with the same law, theta and eps (2D FV plane channel: the
#       flow is invariant in z), gap shrinking at second order.
#   (d) 3D steady adjoint (`gnf3d_steady_law_gradient`) of J = mean u on the
#       same channels (reduced periodic lattice taken from (b), re-converged by
#       `run_gnf3d_periodic_steady`) vs the FV gradient with the same law: gap
#       shrinking at second order.
#   (c) 3D adjoint gradient (all weights) vs central differences on random
#       directions, steps 1e-2..1e-5 (N = 16): V-shaped error, minimum < 1e-6.
using Test, KrakeNN, Random, LinearAlgebra, Printf

const _L3_G = 1.0
const _L3_EPS = 1e-3
const _L3_FVNX = 4

function _l3_net()
    net = DenseNet(1, (8, 8); activation=:tanh, output=:exp, scale=(5.0,))
    theta = law_init(net; rng=MersenneTwister(3)) .* 10   # nu varies by ~25 % in the channel
    theta[end] = log(0.1)
    return net, theta
end

# Law units: H = 1. Lattice: dx = 1/N, dt = dx^2, so nu_lat = nu, gamma = gamma_lat/dt.
_l3_closure(net, theta, N) =
    LBMLawViscosity(net, theta; eps=_L3_EPS, nu_scale=1.0, gamma_scale=Float64(N)^2)
_l3_lbm(rheology, N; max_steps=400_000, tol_ss=1e-11) =
    run_gnf_channel_3d(; rheology, N, nu_init=0.1, F=_L3_G / N^3, flow_axis=:x,
                       wall_axis=:y, n_transverse=4, max_steps, tol_ss, check_every=100)
_l3_fvprob(net, theta, N) = fv_steady_problem(_L3_FVNX, N; h=1 / N,
    bc=(W=:periodic, E=:periodic, S=:wall, N=:wall), source=(x, y) -> (_L3_G, 0.0, 0.0),
    viscosity=FVLawViscosity(net, theta; eps=_L3_EPS))
_l3_rel(a, b) = norm(a .- b) / max(norm(b), eps(Float64))

@testset "Scalar laws in the 3D LBM GNF step" begin
    @testset "(a) ClassicalLaw(CarreauYasuda) == CarreauYasuda closure, bit for bit (3D)" begin
        cy = CarreauYasuda(0.17, 0.0034, 20.0, 2.0, 0.7)
        law = ClassicalLaw(cy)
        cl = LBMLawViscosity(law, law_init(law); eps=0.0)
        for g in (0.0, 3.7e-5, 0.02, 1.3)
            @test KrakeNN.effective_viscosity(cl, g) == KrakeNN.effective_viscosity(cy, g)
        end
        for q_w in (0.5, 0.3)
            kw = (; N=8, nu_init=0.17, F=2e-3, q_w, max_steps=3000, tol_ss=1e-10,
                  check_every=100)
            r0 = run_gnf_channel_3d(; rheology=cy, kw...)
            r1 = run_gnf_channel_3d(; rheology=cl, kw...)
            @test r0.converged && r1.steps == r0.steps
            @test r1.fields.f == r0.fields.f
            @test r1.fields.sp == r0.fields.sp && r1.fields.sm == r0.fields.sm
            @test r1.fields.ux == r0.fields.ux && r1.u == r0.u && r1.nu == r0.nu
            numin, numax = extrema(r0.nu)
            @info "LBM3D law (a) CY bit identity" q_w steps = r0.steps nu_range = (numin, numax) max_abs_f_diff = maximum(abs.(r1.fields.f .- r0.fields.f))
            @test numax / numin > 1.01             # the closure is not Newtonian here
        end
    end

    net, theta = _l3_net()
    gaps = Float64[]; qgaps = Float64[]; ggaps = Float64[]
    base16 = Ref{Any}(nothing)
    @testset "(b) DenseNet plane channel: 3D LBM vs FV, N = $N" for N in (16, 32)
        t0 = time()
        rl = _l3_lbm(_l3_closure(net, theta, N), N)
        t_lbm = time() - t0
        @test rl.converged
        @test all(isfinite, rl.u)
        ul = rl.u .* N                          # u = u_lat dx/dt
        p = _l3_fvprob(net, theta, N)
        s = fv_newton_solver(p)
        U = zeros(fv_nunknowns(p))
        @test fv_newton!(s, U; atol=1e-13, rtol=0.0, maxit=30).converged
        uf = [sum(U[fv_index(p.grid, i, j, 1)] for i in 1:_L3_FVNX) / _L3_FVNX for j in 1:N]
        push!(gaps, _l3_rel(uf, ul))
        push!(qgaps, abs(sum(uf) - sum(ul)) / sum(ul))
        numin, numax = extrema(rl.nu)
        @test numax / numin > 1.2               # the law is not Newtonian on this flow
        @info "LBM3D law (b) DenseNet channel" N lbm_steps = rl.steps t_lbm nu_range = (numin, numax) rel_L2_u = gaps[end] mean_u_gap = qgaps[end]

        # (d) adjoint on the reduced periodic lattice (interior of the padded channel)
        cl = _l3_closure(net, theta, N)
        F = (_L3_G / N^3, 0.0, 0.0)
        solid = Array{Bool}(Array(rl.fields.is_solid)[2:3, :, 2:3])
        f0 = Array(rl.fields.f)[2:3, :, 2:3, :]
        tau0 = inv.(Array(rl.fields.sp)[2:3, :, 2:3])
        rp = run_gnf3d_periodic_steady(; is_solid=solid, closure=cl, F, tol=1e-12,
                                       f_init=f0, tau_init=tau0, max_steps=200_000)
        @test rp.converged
        up = [sum(rp.ux[:, j, :]) / 4 for j in 2:N+1]
        @test _l3_rel(up, rl.u) <= 1e-8         # same fixed point as run_gnf_channel_3d
        fl = .!solid; nfl = count(fl)
        gux = zeros(size(solid)); gux[fl] .= N / nfl   # J = mean physical u over fluid nodes
        t1 = time()
        gl = gnf3d_steady_law_gradient(rp.f, rp.tau, (gux, zeros(size(solid)), zeros(size(solid)));
                                       is_solid=solid, closure=cl, F, solver=:gmres,
                                       tol=1e-12, restart=1000, maxiter=20_000)
        t_adj = time() - t1
        @test gl.stats.converged
        @test length(gl.grad) == law_nparams(net) && all(isfinite, gl.grad)
        lawp = FVLawParameters(length(theta))
        nc = _L3_FVNX * N
        obj = FVLinearFunctional([fv_index(p.grid, i, j, 1) for i in 1:_L3_FVNX for j in 1:N],
                                 fill(1 / nc, nc))
        gf = fv_steady_gradient(s, U, theta, obj; params=lawp, ws=fv_adjoint_workspace(s, lawp))
        push!(ggaps, _l3_rel(gl.grad, gf.grad))
        @info "LBM3D law (d) adjoint gradient vs FV" N gmres_iters = gl.stats.iters relres = gl.stats.relres grad_gap = ggaps[end] t_lin = gl.t_lin t_adjoint = gl.t_adjoint t_grad = gl.t_grad t_total = t_adj
        N == 16 && (base16[] = (; rp, grad=gl.grad, solid, F))
    end
    ratio = gaps[1] / gaps[2]; qratio = qgaps[1] / qgaps[2]
    @info "LBM3D law (b) second-order check" gap_ratio = ratio mean_gap_ratio = qratio
    @test gaps[1] <= 2e-2 && qgaps[1] <= 2e-2
    @test ratio >= 3.0                          # second order: ratio ≈ 4 expected
    @test qratio >= 3.0
    gratio = ggaps[1] / ggaps[2]
    @info "LBM3D law (d) gradient second-order check" grad_gaps = Tuple(ggaps) grad_gap_ratio = gratio
    @test ggaps[1] <= 3e-2
    @test gratio >= 3.0                         # shrinking with the discretisation gap

    @testset "(c) 3D adjoint vs central differences, all weights (N = 16)" begin
        N = 16
        b = base16[]
        fl = .!b.solid; nfl = count(fl)
        J = r -> N * sum(r.ux[fl]) / nfl
        rng = MersenneTwister(11)
        for k in 1:2
            d = normalize(randn(rng, length(theta)))
            exact = dot(b.grad, d)
            errs = Float64[]
            for h in (1e-2, 1e-3, 1e-4, 1e-5)
                Jh = t -> J(run_gnf3d_periodic_steady(; is_solid=b.solid,
                    closure=_l3_closure(net, theta .+ t .* d, N), F=b.F, tol=0.0,
                    f_init=b.rp.f, tau_init=b.rp.tau, max_steps=6000))
                push!(errs, abs((Jh(h) - Jh(-h)) / (2h) - exact) / abs(exact))
            end
            @info "LBM3D law (c) adjoint vs FD" direction = k steps = (1e-2, 1e-3, 1e-4, 1e-5) rel_err = Tuple(errs)
            @test minimum(errs) <= 1e-6
            @test errs[1] > 10 * errs[3]             # truncation branch of the V
        end
    end
end
