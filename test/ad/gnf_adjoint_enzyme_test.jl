using Test
using LinearAlgebra
using Printf
using Random
using KernelAbstractions
using KrakeNN
using Enzyme

# T4: the hand-written VJP of the GNF step (gnf_vjp!, local
# ForwardDiff blocks + transposed stream) against Enzyme reverse mode on the
# full CPU step. The step is rewritten here as plain loops over the same
# per-cell function (gnf_cell_update) and the same stream map (_spxwy_src),
# and checked bit for bit against gnf_step! before it is differentiated.
# Case: periodic obstacle channel at m = 4 (32×16), reference Carreau–Yasuda parameters
# (contrast η0/η∞ = 50), at the forward fixed point; 3 random vectors.
# Full tier only (needs Enzyme): run with
#   KRAKEN_ONLY=ad/gnf_adjoint_enzyme_test.jl

const _GAE_OPP = (1, 4, 5, 2, 3, 8, 9, 6, 7)

function _gae_step_plain!(fo, to, fi, ti, solid, closure, Fx)
    Nx, Ny = size(solid)
    @inbounds for j in 1:Ny, i in 1:Nx
        if solid[i, j]
            for q in 1:9
                fo[i, j, _GAE_OPP[q]] = fi[KrakeNN._spxwy_src(i, j, q, Nx, Ny)]
            end
            to[i, j] = ti[i, j]
        else
            fp = KrakeNN.SVector{9,Float64}(ntuple(q -> fi[KrakeNN._spxwy_src(i, j, q, Nx, Ny)], Val(9)))
            fout, τn, _, _ = gnf_cell_update(fp, ti[i, j], Fx, 0.0, closure)
            for q in 1:9
                fo[i, j, q] = fout[q]
            end
            to[i, j] = τn
        end
    end
    return nothing
end

@testset "T4 GNF VJP vs Enzyme (32×16, CY contrast 50)" begin
    t0 = time()
    m = 4; nu0 = 0.17; Nx, Ny = 8m, 4m; N = Nx * Ny
    solid = [(((i - 0.5) / m - 4)^2 + ((j - 0.5) / m)^2 < 1.5^2) ||
             (((i - 0.5) / m - 4)^2 + ((j - 0.5) / m - 4)^2 < 1.5^2) for i in 1:Nx, j in 1:Ny]
    g = 0.3 * 5 * nu0^2 / m^3            # same scaled force as gnf_adjoint_test.jl
    closure = CarreauYasuda(nu0, 0.02 * nu0, 5.0 * m^2 / nu0, 2.0, 0.7)
    r = run_gnf_periodic_steady(; is_solid=solid, closure=closure, Fx=g, tol=1e-10,
                                check_every=20, fused=true)
    @test r.converged
    f = Array(r.f); tau = Array(r.tau)

    # plain step == gnf_step! (split and fused), bit for bit
    fo = similar(f); to = similar(tau)
    _gae_step_plain!(fo, to, f, tau, solid, closure, g)
    for fused in (false, true)
        f2 = similar(f); t2 = copy(tau); I1 = similar(tau); nu = similar(tau)
        gnf_step!(f2, f, t2, I1, nu, solid, closure, fill(g, Nx, Ny), zeros(Nx, Ny); fused)
        @test f2 == fo
        @test t2 == to
    end

    lin = gnf_linearise(f, tau, solid, closure, g)
    op = GNFStepOperator(lin)
    rng = MersenneTwister(4)
    worst = 0.0
    for k in 1:3
        v = randn(rng, 10N)
        dfo = reshape(v[1:9N], Nx, Ny, 9); dto = reshape(v[9N+1:10N], Nx, Ny)
        dfi = zeros(Nx, Ny, 9); dti = zeros(Nx, Ny)
        Enzyme.autodiff(Enzyme.Reverse, _gae_step_plain!, Enzyme.Const,
                        Enzyme.Duplicated(similar(f), copy(dfo)), Enzyme.Duplicated(similar(tau), copy(dto)),
                        Enzyme.Duplicated(copy(f), dfi), Enzyme.Duplicated(copy(tau), dti),
                        Enzyme.Const(solid), Enzyme.Const(closure), Enzyme.Const(g))
        ref = vcat(vec(dfi), vec(dti))
        y = similar(v)
        gnf_vjp!(y, op, v; project=false)
        e = norm(y .- ref) / norm(ref)
        ea = norm((y .- ref) .* lin.active) / norm(ref .* lin.active)
        @printf("  T4 v%d: |VJP - Enzyme|/|Enzyme| = %.2e (active components %.2e)\n", k, e, ea)
        worst = max(worst, e)
    end
    @test worst <= 1e-12
    @printf("  T4 wall (incl. compilation) %.1f s\n", time() - t0)
end
