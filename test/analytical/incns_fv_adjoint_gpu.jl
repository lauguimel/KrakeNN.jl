# GPU / cuDSS contract of the C4 FV adjoint (src/methods/inc_ns/fv_adjoint.jl):
#   - the transposed cuDSS solve is available (CUDSS.jl >= 1.0) and solves
#     Jᵀ λ = g with the Newton factors (no second analysis);
#   - gradients on CUDABackend (parameter duals through the same residual
#     kernel, coloured ν-field gather) equal the CPU gradients for scalar
#     parameters, the viscosity field and a generic FVProblemMap;
#   - with a preallocated workspace the device allocates nothing
#     (`CUDA.@allocated == 0`) in the transposed solve, the ν-field and scalar
#     parameter passes, fv_steady_adjoint and fv_steady_gradient; the device free
#     memory is stable over 10 gradient evaluations (R11 reduced contract).
#
# Self-gated like incns_fv_newton_gpu.jl: skipped when CUDA/CUDSS are not
# loadable or CUDA is not functional; KRAKEN_REQUIRE_GPU=1 turns the skip into a
# failure.

using Test
using LinearAlgebra
using Random
using Kraken
using Kraken.SparseArrays

if !isdefined(@__MODULE__, :_fvc2_ghia_error)
    include(joinpath(@__DIR__, "fv_newton_helpers.jl"))
end

const _FVC4_GPU_OK = (try
    @eval Main using CUDA, CUDSS
    @eval Main CUDA.functional()
catch
    false
end) === true

const _FVC4_REQUIRE_GPU = get(ENV, "KRAKEN_REQUIRE_GPU", "0") in ("1", "true")

if _FVC4_GPU_OK
    @eval _fvc4_gpu_allocated(f) = Main.CUDA.@allocated f()
end

_fvc4_host(J) = sparse(transpose(SparseMatrixCSC(size(J)..., Array(J.rowPtr), Array(J.colVal),
                                                 Array(J.nzVal))))

@testset "IncNS steady FV adjoint (C4, GPU)" begin
    if !_FVC4_GPU_OK
        if _FVC4_REQUIRE_GPU
            @error "KRAKEN_REQUIRE_GPU=1 but CUDA/CUDSS is not functional"
            @test _FVC4_GPU_OK
        else
            @info "Skipping C4 FV adjoint GPU tests (CUDA not functional here)"
            @test_skip "C4 FV adjoint GPU contract requires CUDA.functional()"
        end
    else
        CU = Main.CUDA
        N = 64
        p = fv_steady_problem(N, N; nu = (x, y) -> 0.01 * (1 + x * y), boundary_velocity = _fvc2_lid)
        newton = (; atol = 1e-14, rtol = 1e-15, maxit = 40)
        sc = fv_newton_solver(p)
        Uc = zeros(fv_nunknowns(p))
        @test fv_newton!(sc, Uc; newton...).converged
        sg = fv_newton_solver(p; to = CU.CuArray, linsolve = CUDABackendTag())
        Ug = CU.zeros(Float64, fv_nunknowns(p))
        @test fv_newton!(sg, Ug; newton...).converged
        @test norm(Array(Ug) - Uc, Inf) < 1e-10

        @testset "transposed solve with the Newton factors" begin
            @test lin_supports_transpose(sg.cache)
            fv_newton_finalize!(sg, Ug)
            @test sg.n_factorize == 1
            rng = MersenneTwister(3)
            b = randn(rng, length(Uc)); x = similar(Ug)
            lin_solve!(x, sg.cache, CU.CuArray(b); transpose = true)
            Jh = _fvc4_host(sg.J)
            @test norm(transpose(Jh) * Array(x) - b) / norm(b) < 1e-10
        end

        lp = LineProfile(:u, [(N ÷ 2, j) for j in 4:4:N])
        objs = (("lid drag", FVWallForce(1; region = :N)),
                ("u profile misfit", FVStateObs(p, lp, zeros(length(lp.indices)))))
        poly = FVProblemMap((q, θ) -> fv_param_problem(FVViscosityField(), q,
                                vec(Float64[1 + ((i - 0.5) / N) * ((j - 0.5) / N) for i in 1:N, j in 1:N]) .* θ[1] .+
                                vec(Float64[((j - 0.5) / N)^2 for i in 1:N, j in 1:N]) .* θ[2]), 2)
        for (oname, obj) in objs
            @testset "CPU/GPU gradient parity: $oname" begin
                for (mname, map, θ) in (("lid speed", FVSideVelocity(p, :N), [1.0]),
                                        ("nu field", FVViscosityField(), nothing),
                                        ("FVProblemMap", poly, [0.01, 0.0]))
                    rc = fv_steady_gradient(sc, Uc, θ, obj; params = map)
                    rg = fv_steady_gradient(sg, Ug, θ, obj; params = map)
                    rel = norm(rg.grad - rc.grad, Inf) / norm(rc.grad, Inf)
                    @info "C4 GPU parity" oname mname rel
                    @test rel < 1e-8
                    @test abs(rg.J - rc.J) <= 1e-10 * max(abs(rc.J), 1)
                    @test sg.n_factorize == 1
                end
            end
        end

        @testset "uniform viscosity CPU/GPU gradient parity" begin
            # Solve at the uniform parameter itself; the main case has a
            # spatial viscosity field and cannot reuse that state here.
            q = fv_steady_problem(N, N; nu = 0.01, boundary_velocity = _fvc2_lid)
            snc = fv_newton_solver(q)
            sng = fv_newton_solver(q; to = CU.CuArray, linsolve = CUDABackendTag())
            Unc = copy(Uc); Ung = copy(Ug)
            @test fv_newton!(snc, Unc; newton...).converged
            @test fv_newton!(sng, Ung; newton...).converged
            map = FVUniformViscosity()
            ws = fv_adjoint_workspace(sng, map)
            for (_, obj) in objs
                rc = fv_steady_gradient(snc, Unc, [0.01], obj; params = map)
                rg = fv_steady_gradient(sng, Ung, [0.01], obj; params = map, ws)
                @test norm(rg.grad - rc.grad, Inf) <= 1e-8 * norm(rc.grad, Inf)
                @test _fvc4_gpu_allocated(() -> fv_steady_gradient(sng, Ung, [0.01], obj;
                                                                  params = map, ws)) == 0
            end
        end

        @testset "zero device allocation with a workspace" begin
            obj = FVWallForce(1; region = :N)
            wsf = fv_adjoint_workspace(sg, FVViscosityField())
            wss = fv_adjoint_workspace(sg, FVSideVelocity(p, :N))
            map = FVSideVelocity(p, :N)
            fv_steady_gradient(sg, Ug, nothing, obj; params = FVViscosityField(), ws = wsf)  # warm-up
            fv_steady_gradient(sg, Ug, [1.0], obj; params = map, ws = wss)
            gU = CU.CuArray(fv_steady_gradient(sc, Uc, nothing, obj; params = FVViscosityField()).gU)
            @test _fvc4_gpu_allocated(() -> (lin_solve!(wsf.λ, sg.cache, wsf.g; transpose = true);
                                             CU.synchronize())) == 0
            @test _fvc4_gpu_allocated(() -> Kraken._fv_param_cotangent!(wsf, FVViscosityField(), sg, Ug, nothing)) == 0
            @test _fvc4_gpu_allocated(() -> Kraken._fv_param_cotangent!(wss, map, sg, Ug, [1.0])) == 0
            @test _fvc4_gpu_allocated(() -> Kraken._fv_state_sig(sg.norms, 0.0, Ug)) == 0
            @test _fvc4_gpu_allocated(() -> fv_steady_adjoint(sg, Ug, gU; params = FVViscosityField(), ws = wsf)) == 0
            @test _fvc4_gpu_allocated(() -> fv_steady_gradient(sg, Ug, nothing, obj;
                                                               params = FVViscosityField(), ws = wsf)) == 0
            fv_steady_gradient(sg, Ug, nothing, obj;
                               params = FVViscosityField(), ws = wsf, finalize = false)
            @test _fvc4_gpu_allocated(() -> fv_steady_gradient(sg, Ug, nothing, obj;
                               params = FVViscosityField(), ws = wsf, finalize = false)) == 0
            @test _fvc4_gpu_allocated(() -> fv_steady_gradient(sg, Ug, [1.0], obj;
                                                               params = map, ws = wss)) == 0
            frees = Int[]
            for it in 1:10
                fv_steady_gradient(sg, Ug, nothing, obj; params = FVViscosityField(), ws = wsf)
                CU.synchronize()
                push!(frees, CU.free_memory())
            end
            spread = (maximum(frees) - minimum(frees)) / 2^20
            @info "C4 GPU gradient loop" free_memory_spread_MiB = spread
            @test spread <= 64
        end
    end
end
