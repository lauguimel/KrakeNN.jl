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
    gaps = Float64[]; qgaps = Float64[]
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
    end
    ratio = gaps[1] / gaps[2]; qratio = qgaps[1] / qgaps[2]
    @info "LBM3D law (b) second-order check" gap_ratio = ratio mean_gap_ratio = qratio
    @test gaps[1] <= 2e-2 && qgaps[1] <= 2e-2
    @test ratio >= 3.0                          # second order: ratio ≈ 4 expected
    @test qratio >= 3.0
end
