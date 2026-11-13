# Steady adjoint of the 3D D3Q19 GNF step on CUDA (src/ad/gnf3d_adjoint*.jl),
# DenseNet law through LBMLawViscosity, Float64. CPU/GPU parity of
#   (1) the forward run `run_gnf3d_periodic_steady` (fixed number of steps),
#   (2) the matrix-free products `gnf3d_jvp!` / `gnf3d_vjp!` (projected and
#       unprojected) on the same host blocks, and the mass gauge on the device,
#   (3) the full law gradient `gnf3d_steady_law_gradient`: GPU bordered GMRES and
#       GPU K-step vs CPU bordered GMRES (blocks and pullback on the host, solve
#       on the device).
# Two cases: the 4×6×3 lattice of test/ad/gnf3d_adjoint_test.jl (wall plane and
# one solid voxel) and a 12×10×8 channel (walls j = 1, j = 10) with a voxel sphere.
# Float32 products are printed for information only (no gate).
# Self-gated like nn_law_lbm_gpu.jl: skipped without CUDA; KRAKEN_REQUIRE_GPU=1
# turns the skip into a failure. Command (cluster):
#   KRAKEN_REQUIRE_GPU=1 julia --project=. test/analytical/gnf3d_adjoint_gpu.jl
# KRAKEN_GNF3D_CPU_SMOKE=1 runs the same code on the CPU (script check only, not a
# GPU receipt).
using Test, KrakeNN, LinearAlgebra, Random

const _G3G_SMOKE = get(ENV, "KRAKEN_GNF3D_CPU_SMOKE", "0") in ("1", "true")
const _G3G_REQUIRE_GPU = get(ENV, "KRAKEN_REQUIRE_GPU", "0") in ("1", "true")
# CUDA is loaded here, at top level, never inside a function.
const _G3G_GPU_OK = _G3G_SMOKE || (try
    @eval Main using CUDA
    @eval Main CUDA.functional()
catch
    false
end) === true

_g3g_backend() = _G3G_SMOKE ? KrakeNN.KernelAbstractions.CPU() : Main.CUDA.CUDABackend()
_g3g_to() = _G3G_SMOKE ? Array : Main.CUDA.CuArray
_g3g_rel(a, b) = norm(Array(a) .- Array(b)) / max(norm(Array(b)), eps(Float64))

function _g3g_net()
    net = DenseNet(1, (6,); activation=:tanh, output=:exp, scale=(1.0,))
    θ = law_init(net; rng=MersenneTwister(5))
    θ[end] = log(0.12)
    return net, θ
end
_g3g_closure(net, θ) = LBMLawViscosity(net, θ; eps=1e-3)

function _g3g_cases()
    a = falses(4, 6, 3); a[:, 1, :] .= true; a[2, 4, 2] = true
    b = falses(12, 10, 8); b[:, 1, :] .= true; b[:, 10, :] .= true
    for k in 1:8, j in 1:10, i in 1:12
        (i - 6.5)^2 + (j - 5.5)^2 + (k - 4.5)^2 <= 4.0 && (b[i, j, k] = true)
    end
    return (("4x6x3 wall+voxel", Array{Bool}(a), (1e-3, 0.0, 0.0)),
            ("12x10x8 channel+sphere", Array{Bool}(b), (5e-4, 0.0, 0.0)))
end

@testset "3D GNF steady adjoint (CUDA)" begin
    if !_G3G_GPU_OK
        if _G3G_REQUIRE_GPU
            @test _G3G_GPU_OK
        else
            @test_skip "requires CUDA.functional()"
        end
    else
        net, θ = _g3g_net()
        clh = _g3g_closure(net, θ)
        cld = _g3g_closure(net, _g3g_to()(θ))
        be = _g3g_backend()
        @testset "$name" for (name, solid, F) in _g3g_cases()
            fl = .!solid; nfl = count(fl)
            # (1) forward parity, fixed number of steps from the default start
            kw = (; is_solid=solid, F, nu_init=0.12, tol=0.0, check_every=100, max_steps=500)
            rc = run_gnf3d_periodic_steady(; closure=clh, kw...)
            rg = run_gnf3d_periodic_steady(; closure=cld, backend=be, kw...)
            ef = _g3g_rel(rg.f, rc.f)
            @info "GNF3D GPU forward parity" case = name steps = rg.steps relative = ef
            @test ef <= 1e-12

            # steady state (CPU) for the products and the gradient
            r = run_gnf3d_periodic_steady(; is_solid=solid, closure=clh, F, nu_init=0.12,
                                          tol=1e-12, max_steps=200_000)
            @test r.converged
            lin = gnf3d_linearise(r.f, r.tau, solid, clh, F)

            # (2) operator products on the same blocks
            opc = GNF3DStepOperator(lin)
            opg = GNF3DStepOperator(lin; backend=be)
            n = opc.n
            rng = MersenneTwister(31)
            x = randn(rng, n); xd = _g3g_to()(x)
            yc = similar(x); yd = similar(xd)
            for project in (false, true)
                gnf3d_jvp!(yc, opc, x; project); gnf3d_jvp!(yd, opg, xd; project)
                ej = _g3g_rel(yd, yc)
                gnf3d_vjp!(yc, opc, x; project); gnf3d_vjp!(yd, opg, xd; project)
                ev = _g3g_rel(yd, yc)
                @info "GNF3D GPU product parity" case = name project jvp = ej vjp = ev
                @test ej <= 1e-13
                @test ev <= 1e-13
            end
            yg = similar(opg.chat); gnf3d_vjp!(yg, opg, opg.chat)
            @test norm(Array(yg) .- Array(opg.chat)) <= 1e-13

            # Float32 products (information only)
            op32 = GNF3DStepOperator(lin; backend=be, T=Float32)
            y32 = similar(xd); gnf3d_vjp!(y32, op32, xd); gnf3d_vjp!(yd, opg, xd)
            @info "GNF3D GPU Float32 vjp (no gate)" case = name relative = _g3g_rel(y32, yd)

            # (3) full gradient of J = mean u_x over fluid nodes
            gux = zeros(size(solid)); gux[fl] .= 1 / nfl
            z = zeros(size(solid)); gu = (gux, z, z)
            kwg = (; is_solid=solid, F, solver=:gmres, tol=1e-12, restart=1000, maxiter=20_000)
            gc = gnf3d_steady_law_gradient(r.f, r.tau, gu; closure=clh, kwg...)
            gg = gnf3d_steady_law_gradient(r.f, r.tau, gu; closure=cld, backend=be, kwg...)
            gk = gnf3d_steady_law_gradient(r.f, r.tau, gu; is_solid=solid, F, closure=cld,
                                           backend=be, solver=:kstep, tol=1e-11)
            eg = _g3g_rel(gg.grad, gc.grad); ek = _g3g_rel(gk.grad, gc.grad)
            @info "GNF3D GPU gradient parity" case = name n_fluid = nfl steps = r.steps iters_cpu = gc.stats.iters iters_gpu_gmres = gg.stats.iters iters_gpu_kstep = gk.stats.iters gmres = eg kstep = ek t_adjoint_gpu_gmres = gg.t_adjoint t_adjoint_gpu_kstep = gk.t_adjoint
            @test gc.stats.converged && gg.stats.converged
            @test eg <= 1e-8
            @test gk.stats.converged && ek <= 1e-8
        end
    end
end
