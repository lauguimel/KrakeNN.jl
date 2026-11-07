# Prepared C5 M3 CUDA/cuDSS contract. Execution requires explicit GPU opt-in.
# Float64; frozen 1e-12 relative parity, zero hot device allocation.
using Test, LinearAlgebra, KrakeNN

if !isdefined(@__MODULE__, :_c5gpu_problem)
    include(joinpath(@__DIR__, "incns_fv_immersed_gpu_helpers.jl"))
end

const _C5_GPU_REQUIRED = get(ENV, "KRAKEN_REQUIRE_GPU", "0") in ("1", "true")
const _C5_GPU_ENABLED = _C5_GPU_REQUIRED || get(ENV, "KRAKEN_TEST_GPU", "0") in ("1", "true")
const _C5_GPU_READY = _C5_GPU_ENABLED && (try
    @eval Main using CUDA, CUDSS
    Main.CUDA.functional()
catch
    false
end)
if _C5_GPU_READY
    @eval _c5gpu_allocated(f) = Main.CUDA.@allocated f()
end

@testset "C5 immersed Newton and both adjoints (GPU opt-in)" begin
    if !_C5_GPU_READY
        if _C5_GPU_REQUIRED
            @test _C5_GPU_READY
        else
            @test_skip "Requires explicit GPU opt-in and functional CUDA/cuDSS"
        end
    else
        CU = Main.CUDA
        CU.allowscalar(false)
        p = _c5gpu_problem(16)
        @test p.immersed.closure.mass == 0x03
        @test fv_nunknowns(p) == 3 * 16^2 + 1
        @test only(p.immersed.mass.Q) == 0.0
        opts = (; atol=1e-14, rtol=1e-14, maxit=6)
        sc = fv_newton_solver(p)
        sg = fv_newton_solver(p; to=CU.CuArray, linsolve=CUDABackendTag())
        Uc = zeros(fv_nunknowns(p)); Ug = CU.zeros(Float64, length(Uc))
        @test fv_newton!(sc, Uc; opts...).converged
        @test fv_newton!(sg, Ug; opts...).converged
        @test norm(Array(Ug)-Uc, Inf) <= 1e-12 * norm(Uc, Inf)
        @test abs(only(Array(Ug)[3 * 16^2 + 1:end]) - only(Uc[3 * 16^2 + 1:end])) <= 1e-12
        Rc = fv_residual(Uc, p)
        @test norm(Array(fv_residual(Ug, sg.prob))-Rc, Inf) <= 1e-12
        map = FVUniformViscosity()
        ws = fv_adjoint_workspace(sg, map)
        for (kind, obj) in _c5gpu_objectives()
            @testset "$kind parity and hot adjoint" begin
                rc = fv_steady_gradient(sc, Uc, [0.1], obj; params=map)
                rg = fv_steady_gradient(sg, Ug, [0.1], obj; params=map, ws)
                @test abs(rc.J) > 0 && norm(rc.grad, Inf) > 0
                @test abs(rg.J-rc.J) <= 1e-12 * abs(rc.J)
                @test norm(rg.grad-rc.grad, Inf) <= 1e-12 * norm(rc.grad, Inf)
                fv_steady_gradient(sg, Ug, [0.1], obj; params=map, ws, finalize=false)
                @test _c5gpu_allocated(() -> begin
                    fv_steady_gradient(sg, Ug, [0.1], obj; params=map, ws, finalize=false)
                    CU.synchronize()
                end) == 0
            end
        end
        @test sg.n_factorize == 1
        # Force actual Newton work on both warm-up and measured calls.
        Ug .+= 1e-4
        @test fv_newton!(sg, Ug; opts...).converged
        Ug .+= 1e-4
        receipt = Ref{Any}()
        bytes = _c5gpu_allocated(() -> begin
            receipt[] = fv_newton!(sg, Ug; opts...)
            CU.synchronize()
        end)
        @test receipt[].converged
        @test receipt[].iterations > 0
        @test bytes == 0
        @test norm(Array(Ug)-Uc, Inf) <= 1e-12 * norm(Uc, Inf)
    end
end
