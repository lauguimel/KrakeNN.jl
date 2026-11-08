# GPU / cuDSS contract of the SSA solver (src/methods/ssa/ssa.jl):
#   - the coloured dual Jacobian assembled on the device equals the CPU one;
#   - Newton on CUDABackend + cuDSS reproduces the CPU solution (Weertman, n = 3);
#   - the learned-drag adjoint gradient on the GPU matches central finite differences
#     (relative error < 1e-5) and the CPU gradient; ONE cuDSS analysis for the whole run.
# Self-gated like incns_fv_adjoint_gpu.jl: skipped when CUDA/CUDSS are not loadable or
# CUDA is not functional; KRAKEN_REQUIRE_GPU=1 turns the skip into a failure.
# Size: KRAKEN_SSA_N (default 24; the cluster runs 64).
using Test
if !isdefined(@__MODULE__, :ssa_stream_problem)
    include(joinpath(@__DIR__, "ssa_helpers.jl"))
end
if !isdefined(@__MODULE__, :ssa_learned_gradient_check)
    include(joinpath(@__DIR__, "ssa_learned_adjoint.jl"))
end

const _SSA_GPU_OK = (try
    @eval Main using CUDA, CUDSS
    @eval Main CUDA.functional()
catch
    false
end) === true

const _SSA_REQUIRE_GPU = get(ENV, "KRAKEN_REQUIRE_GPU", "0") in ("1", "true")

_ssa_host(J) = sparse(transpose(SparseMatrixCSC(size(J)..., Array(J.rowPtr), Array(J.colVal),
                                                Array(J.nzVal))))

@testset "SSA solver (GPU)" begin
    if !_SSA_GPU_OK
        if _SSA_REQUIRE_GPU
            @error "KRAKEN_REQUIRE_GPU=1 but CUDA/CUDSS is not functional"
            @test _SSA_GPU_OK
        else
            @info "Skipping SSA GPU tests (CUDA not functional here)"
            @test_skip "SSA GPU contract requires CUDA.functional()"
        end
    else
        CU = Main.CUDA
        N = ssa_test_n()
        to = CU.CuArray; lin = CUDABackendTag()
        p = ssa_stream_problem(N)
        @testset "Jacobian parity CPU/GPU" begin
            U = 20 .* rand(MersenneTwister(5), ssa_nunknowns(p)) .+ 1.0
            sc = ssa_newton_solver(p)
            sg = ssa_newton_solver(p; to, linsolve = lin)
            ssa_jacobian!(sc, U); ssa_jacobian!(sg, to(U))
            Jc = Matrix(sc.J); Jg = Matrix(_ssa_host(sg.J))
            @test maximum(abs, Jc .- Jg) <= 1e-12 * maximum(abs, Jc)
        end
        @testset "Newton parity CPU/GPU (Weertman, n = 3)" begin
            rc = ssa_solve(p)
            rg = ssa_solve(p; to, linsolve = lin)
            println("SSA GPU Newton: N = $N, CPU its = $(rc.iterations), GPU its = $(rg.iterations), ",
                    "final |R|_inf GPU = $(rg.stats.inf[end])")
            @test rc.converged && rg.converged
            @test norm(Array(rg.U) - Array(rc.U), Inf) <= 1e-8 * norm(Array(rc.U), Inf)
            @test rg.solver.n_factorize == 1
        end
        @testset "learned drag: GPU adjoint vs FD and vs CPU" begin
            gg = ssa_learned_gradient_check(; to, linsolve = lin, nweights = 6)
            gc = ssa_learned_gradient_check(; nweights = 6)
            println("SSA GPU T4: rel err adjoint vs FD ", gg.rel, " V-shape ", gg.vshape)
            @test maximum(gg.rel) < 1e-5
            @test norm(gg.grad - gc.grad, Inf) <= 1e-7 * norm(gc.grad, Inf)
            @test gg.solver.n_factorize == 1
        end
    end
end
