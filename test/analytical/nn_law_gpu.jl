# DenseNet in-kernel law on CUDA (FV residual / Jacobian / parameter gradient).
#   (a) CPU/GPU parity of residual, Jacobian and the contracted weight gradient
#       for a mixed-activation DenseNet inside FVLawViscosity;
#   (b) cost table, widths (8,16,32,64) x depth (2,3,4) on a 256^2 grid:
#       compile time of the first residual+Jacobian call and time per Jacobian
#       assembly (median of 5, device-synchronised).
# Self-gated like incns_fv_energy_gpu.jl: skipped without CUDA/CUDSS;
# KRAKEN_REQUIRE_GPU=1 turns the skip into a failure. Command (cluster):
#   KRAKEN_REQUIRE_GPU=1 julia --project=. test/analytical/nn_law_gpu.jl
# Environment: KRAKEN_NNLAW_N (grid size for (b), default 256);
# KRAKEN_NNLAW_CPU_SMOKE=1 runs the same code on the CPU (script check only,
# not a GPU receipt; use a small KRAKEN_NNLAW_N such as 24).
using Test, KrakeNN, LinearAlgebra, Random, SparseArrays, Printf, Statistics

const _NNL_SMOKE = get(ENV, "KRAKEN_NNLAW_CPU_SMOKE", "0") in ("1", "true")
const _NNL_REQUIRE_GPU = get(ENV, "KRAKEN_REQUIRE_GPU", "0") in ("1", "true")
const _NNL_GPU_OK = _NNL_SMOKE || (try
    @eval Main using CUDA, CUDSS
    @eval Main CUDA.functional()
catch
    false
end) === true

_nnl_to() = _NNL_SMOKE ? Array : Main.CUDA.CuArray
_nnl_tag() = _NNL_SMOKE ? CPUBackendTag() : CUDABackendTag()
_nnl_sync() = _NNL_SMOKE ? nothing : Main.CUDA.synchronize()
_nnl_host(J::SparseMatrixCSC) = J
_nnl_host(J) = sparse(transpose(SparseMatrixCSC(size(J)...,
    Array(J.rowPtr), Array(J.colVal), Array(J.nzVal))))
_nnl_rel(a, b) = norm(a - b) / max(norm(a), norm(b), eps(Float64))

function _nnl_problem(law, theta, N; fields=())
    return fv_steady_problem(N, N; h=1.0, bc=(W=:periodic, E=:periodic, S=:wall, N=:wall),
        source=(x, y) -> (0.02, 0.0, 0.0),
        viscosity=FVLawViscosity(law, theta, fields; eps=1e-3))
end

function _nnl_law(nin, width, depth, act)
    law = DenseNet(nin, ntuple(_ -> width, depth); activation=act, output=:exp)
    theta = law_init(law; rng=MersenneTwister(7))
    theta[end] = log(0.3)
    return law, theta
end

@testset "DenseNet FV law (CUDA)" begin
    if !_NNL_GPU_OK
        if _NNL_REQUIRE_GPU
            @test _NNL_GPU_OK
        else
            @test_skip "requires CUDA.functional()"
        end
    else
        @testset "(a) CPU/GPU parity" begin
            N = 12
            f1 = [0.2*(i-0.5)+0.1*(j-0.5) for i in 1:N, j in 1:N]
            for (hidden, act) in (((5, 4), (:tanh, :softplus)), ((6, 6, 4), (:relu, :sigmoid, :tanh)))
                law = DenseNet(2, hidden; activation=act, output=:exp)
                theta = law_init(law; rng=MersenneTwister(5)); theta[end] = log(0.3)
                p = _nnl_problem(law, theta, N; fields=(f1,))
                sc = fv_newton_solver(p)
                sg = fv_newton_solver(p; to=_nnl_to(), linsolve=_nnl_tag())
                Uc = 0.05 .* randn(MersenneTwister(1), fv_nunknowns(p)); Ug = _nnl_to()(Uc)
                Rc = similar(Uc); Rg = similar(Ug)
                fv_residual!(Rc, Uc, sc.prob); fv_residual!(Rg, Ug, sg.prob)
                @test _nnl_rel(Rc, Array(Rg)) <= 1e-12
                fv_jacobian!(sc, Uc); fv_jacobian!(sg, Ug)
                @test _nnl_rel(Matrix(sc.J), Matrix(_nnl_host(sg.J))) <= 1e-12
                # Converged state, then contracted weight gradient.
                Uc = zeros(fv_nunknowns(p)); Ug = _nnl_to()(Uc)
                @test fv_newton!(sc, Uc; atol=1e-12, rtol=0.0, maxit=20).converged
                @test fv_newton!(sg, Ug; atol=1e-12, rtol=0.0, maxit=20).converged
                @test _nnl_rel(Uc, Array(Ug)) <= 1e-10
                map = FVLawParameters(length(theta))
                wc = fv_adjoint_workspace(sc, map); wg = fv_adjoint_workspace(sg, map)
                obj = FVLinearFunctional([fv_index(p.grid, 2, 4, 1)], [1.0])
                gc = fv_steady_gradient(sc, Uc, theta, obj; params=map, ws=wc)
                gg = fv_steady_gradient(sg, Ug, theta, obj; params=map, ws=wg)
                err = _nnl_rel(gc.grad, gg.grad)
                @info "DenseNet CPU/GPU gradient parity" hidden act relative=err
                @test err <= 1e-10
            end
        end

        @testset "(b) compile / assembly cost table" begin
            N = parse(Int, get(ENV, "KRAKEN_NNLAW_N", "256"))
            rows = String[]
            @printf("DenseNet cost, %d^2 grid, tanh hidden, exp output (%s)\n", N,
                    _NNL_SMOKE ? "CPU smoke" : "CUDA")
            @printf("%6s %6s %8s %14s %16s\n", "width", "depth", "nparams",
                    "compile_s", "jacobian_ms")
            for depth in (2, 3, 4), width in (8, 16, 32, 64)
                law, theta = _nnl_law(1, width, depth, :tanh)
                p = _nnl_problem(law, theta, N)
                s = fv_newton_solver(p; to=_nnl_to(), linsolve=_nnl_tag())
                U = _nnl_to()(zeros(fv_nunknowns(p))); R = similar(U)
                t_compile = @elapsed begin
                    fv_residual!(R, U, s.prob)
                    fv_jacobian!(s, U)
                    _nnl_sync()
                end
                ts = Float64[]
                for _ in 1:5
                    push!(ts, @elapsed begin
                        fv_jacobian!(s, U)
                        _nnl_sync()
                    end)
                end
                @printf("%6d %6d %8d %14.2f %16.2f\n", width, depth, law_nparams(law),
                        t_compile, 1e3 * median(ts))
                @test isfinite(t_compile) && all(isfinite, nonzeros(_nnl_host(s.J)))
            end
        end
    end
end
