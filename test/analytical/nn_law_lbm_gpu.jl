# DenseNet law in the 2D LBM GNF step on CUDA (LBMLawViscosity).
#   (a) CPU/GPU parity of the forward run (same number of steps, split and fused
#       kernels) on the DenseNet channel of lbm_law_test.jl, and of the steady
#       adjoint weight gradient (operator and GMRES on the device, blocks and
#       pullback on the host);
#   (b) cost: steps per second of the fused step on an Nx x Ny channel and the
#       host time of gnf_linearise and gnf_law_gradient.
# Self-gated like nn_law_gpu.jl: skipped without CUDA; KRAKEN_REQUIRE_GPU=1 turns
# the skip into a failure. Command (cluster):
#   KRAKEN_REQUIRE_GPU=1 julia --project=. test/analytical/nn_law_lbm_gpu.jl
# Environment: KRAKEN_NNLAW_LBM_N (Ny for (b), default 512, Nx = 2 Ny);
# KRAKEN_NNLAW_CPU_SMOKE=1 runs the same code on the CPU (script check only, not
# a GPU receipt; use a small KRAKEN_NNLAW_LBM_N such as 32).
using Test, KrakeNN, LinearAlgebra, Random, Printf

const _NLG_SMOKE = get(ENV, "KRAKEN_NNLAW_CPU_SMOKE", "0") in ("1", "true")
const _NLG_REQUIRE_GPU = get(ENV, "KRAKEN_REQUIRE_GPU", "0") in ("1", "true")
# CUDA is loaded here, at top level, never inside a function.
const _NLG_GPU_OK = _NLG_SMOKE || (try
    @eval Main using CUDA
    @eval Main CUDA.functional()
catch
    false
end) === true

_nlg_backend() = _NLG_SMOKE ? KrakeNN.KernelAbstractions.CPU() : Main.CUDA.CUDABackend()
_nlg_to() = _NLG_SMOKE ? Array : Main.CUDA.CuArray
_nlg_sync() = _NLG_SMOKE ? nothing : Main.CUDA.synchronize()
_nlg_rel(a, b) = norm(a .- b) / max(norm(b), eps(Float64))

function _nlg_net()
    net = DenseNet(1, (8, 8); activation=:tanh, output=:exp, scale=(5.0,))
    theta = law_init(net; rng=MersenneTwister(3)) .* 10
    theta[end] = log(0.1)
    return net, theta
end
_nlg_closure(net, theta, Ny) =
    LBMLawViscosity(net, theta; eps=1e-3, nu_scale=1.0, gamma_scale=Float64(Ny)^2)

@testset "DenseNet LBM law (CUDA)" begin
    if !_NLG_GPU_OK
        if _NLG_REQUIRE_GPU
            @test _NLG_GPU_OK
        else
            @test_skip "requires CUDA.functional()"
        end
    else
        net, theta = _nlg_net()
        @testset "(a) CPU/GPU parity, forward and weight gradient" begin
            Nx, Ny = 4, 16
            solid = falses(Nx, Ny)
            for fused in (false, true)
                kw = (; is_solid=solid, Fx=1.0 / Ny^3, tol=0.0, check_every=20, max_steps=4000, fused)
                rc = run_gnf_periodic_steady(; closure=_nlg_closure(net, theta, Ny), kw...)
                rg = run_gnf_periodic_steady(; closure=_nlg_closure(net, _nlg_to()(theta), Ny),
                                             backend=_nlg_backend(), kw...)
                e = _nlg_rel(Array(rg.f), Array(rc.f))
                @info "LBM DenseNet forward parity" fused relative = e
                @test e <= 1e-12
            end
            r = run_gnf_periodic_steady(; closure=_nlg_closure(net, theta, Ny), is_solid=solid,
                                        Fx=1.0 / Ny^3, tol=1e-12, check_every=20, fused=true)
            @test r.converged
            gu = (fill(Float64(Ny) / (Nx * Ny), Nx, Ny), zeros(Nx, Ny))
            kw = (; is_solid=solid, Fx=1.0 / Ny^3, solver=:gmres, tol=1e-12, restart=1000,
                  maxiter=20_000)
            gc = gnf_steady_law_gradient(r.f, r.tau, gu; closure=_nlg_closure(net, theta, Ny), kw...)
            gg = gnf_steady_law_gradient(_nlg_to()(r.f), _nlg_to()(r.tau), gu;
                closure=_nlg_closure(net, _nlg_to()(theta), Ny), backend=_nlg_backend(), kw...)
            e = _nlg_rel(gg.grad, gc.grad)
            @info "LBM DenseNet gradient parity" relative = e iters_cpu = gc.stats.iters iters_gpu = gg.stats.iters
            @test gg.stats.converged
            @test e <= 1e-8
        end

        @testset "(b) cost" begin
            Ny = parse(Int, get(ENV, "KRAKEN_NNLAW_LBM_N", "512")); Nx = 2Ny
            solid = falses(Nx, Ny)
            cl = _nlg_closure(net, _nlg_to()(theta), Ny)
            kw = (; is_solid=solid, Fx=1.0 / Ny^3, tol=0.0, check_every=1000, fused=true,
                  backend=_nlg_backend())
            run_gnf_periodic_steady(; closure=cl, max_steps=10, kw...)      # compile
            _nlg_sync()
            nsteps = 2000
            r = run_gnf_periodic_steady(; closure=cl, max_steps=nsteps, kw...)
            mlups = Nx * Ny * nsteps / r.wall_time / 1e6
            lin = gnf_linearise(r.f, r.tau, solid, cl, 1.0 / Ny^3)
            s = randn(MersenneTwister(2), Nx, Ny)
            gnf_law_gradient(lin, s, cl)
            tg = @elapsed gnf_law_gradient(lin, s, cl)
            @printf("LBM DenseNet cost %s, %d x %d: fused step %.1f MLUPS, host linearise %.2f s, host pullback %.3f s (%d threads)\n",
                    _NLG_SMOKE ? "CPU smoke" : "CUDA", Nx, Ny, mlups, lin.time, tg, Threads.nthreads())
            @test r.finite && isfinite(mlups)
        end
    end
end
