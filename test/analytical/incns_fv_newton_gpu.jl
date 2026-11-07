# GPU / cuDSS contract of the C2 Newton solver (src/methods/inc_ns/fv_newton.jl):
#   - the coloured dual assembly on CUDABackend, written into the values of a
#     CuSparseMatrixCSR, equals the CPU (CSC) assembly bit for bit (to 1e-12);
#   - Newton on GPU (cuDSS through the seam) follows the CPU residual history
#     and reaches the same state; ONE lin_factorize (analysis), then
#     refactorizations; the finalized factors solve J x = b and Jᵀ x = b;
#   - the per-iteration hot path allocates nothing on the device
#     (`CUDA.@allocated == 0` for assembly, refactorization, solve, residual,
#     norms, and for whole Newton iterations once the cache exists), and the
#     device free memory is stable over 10 iterations (R11, reduced contract).
#
# Self-gated like incns_fv_steady_residual_gpu.jl: skipped when CUDA/CUDSS are
# not loadable or CUDA is not functional; KRAKEN_REQUIRE_GPU=1 turns the skip
# into a failure.

using Test
using LinearAlgebra
using Random
using KrakeNN
using KrakeNN.SparseArrays

if !isdefined(@__MODULE__, :_fvc2_ghia_error)
    include(joinpath(@__DIR__, "fv_newton_helpers.jl"))
end

const _FVC2_GPU_OK = (try
    @eval Main using CUDA, CUDSS
    @eval Main CUDA.functional()
catch
    false
end) === true

const _FVC2_REQUIRE_GPU = get(ENV, "KRAKEN_REQUIRE_GPU", "0") in ("1", "true")

if _FVC2_GPU_OK
    @eval _fvc2_gpu_allocated(f) = Main.CUDA.@allocated f()
end

# host CSC copy of a device CSR matrix (CSR(A) == CSC(Aᵀ))
_fvc2_host(J) = sparse(transpose(SparseMatrixCSC(size(J)..., Array(J.rowPtr), Array(J.colVal),
                                                 Array(J.nzVal))))

@testset "IncNS steady FV Newton (C2, GPU)" begin
    if !_FVC2_GPU_OK
        if _FVC2_REQUIRE_GPU
            @error "KRAKEN_REQUIRE_GPU=1 but CUDA/CUDSS is not functional"
            @test _FVC2_GPU_OK
        else
            @info "Skipping C2 FV Newton GPU tests (CUDA not functional here)"
            @test_skip "C2 FV Newton GPU contract requires CUDA.functional()"
        end
    else
        CU = Main.CUDA
        rng = MersenneTwister(7)
        solid = zeros(Bool, 64, 32); solid[20:28, 1:14] .= true; solid[40:44, 18:32] .= true
        cases = (
            ("cavity NS nu(x,y) 64²",
             fv_steady_problem(64, 64; nu = (x, y) -> 0.01 * (1 + x * y), boundary_velocity = _fvc2_lid)),
            ("masked inlet/outlet channel 64x32",
             fv_steady_problem(64, 32; nu = 0.02, solid,
                               bc = (W = :inlet, E = :outlet, S = :wall, N = :wall),
                               boundary_velocity = (s, x, y) -> s === :W ? (4y * (1 - y), 0.0) : (0.0, 0.0))),
        )
        for (name, p) in cases
            @testset "assembly parity: $name" begin
                sc = fv_newton_solver(p)
                sg = fv_newton_solver(p; to = CU.CuArray, linsolve = CUDABackendTag())
                @test nameof(typeof(sg.J)) === :CuSparseMatrixCSR
                for trial in 1:2
                    U = randn(rng, fv_nunknowns(p))
                    fv_jacobian!(sc, U)
                    fv_jacobian!(sg, CU.CuArray(U))
                    Jg = _fvc2_host(sg.J)
                    @test Jg.colptr == sc.J.colptr && rowvals(Jg) == rowvals(sc.J)
                    @test norm(nonzeros(Jg) - nonzeros(sc.J), Inf) <= 1e-12 * norm(nonzeros(sc.J), Inf)
                end
            end
        end

        @testset "Newton on cuDSS: cavity Re = 100 64²" begin
            p = fv_steady_problem(64, 64; nu = 0.01, boundary_velocity = _fvc2_lid)
            rc = solve_incns_fv_newton(; problem = p)
            sg = fv_newton_solver(p; to = CU.CuArray, linsolve = CUDABackendTag())
            U = CU.zeros(Float64, fv_nunknowns(p))
            st = fv_newton!(sg, U)
            @info "C2 Newton, cavity Re=100 64² (GPU, cuDSS)" residual_2norm = st.history
            @test st.converged
            @test st.iterations == rc.iterations
            k = findall(>(1e-10), rc.residual_history)
            @test maximum(abs.(st.history[k] ./ rc.residual_history[k] .- 1)) < 1e-6
            @test st.inf[end] < 1e-12
            @test norm(Array(U) - Array(rc.U), Inf) < 1e-10
            @test sg.n_factorize == 1 && sg.n_refactorize == st.iterations - 1
            # finalize at the converged state: direct and transposed solves
            fv_newton_finalize!(sg, U)
            @test sg.n_factorize == 1
            Jh = _fvc2_host(sg.J)
            b = randn(rng, length(U)); bd = CU.CuArray(b); x = similar(bd)
            lin_solve!(x, sg.cache, bd)
            @test norm(Jh * Array(x) - b) / norm(b) < 1e-10
            if lin_supports_transpose(sg.cache)
                lin_solve!(x, sg.cache, bd; transpose = true)
                @test norm(transpose(Jh) * Array(x) - b) / norm(b) < 1e-10
            else
                @info "cuDSS transposed solve unavailable (CUDSS.jl < 1.0): skipped"
            end

            # zero device allocation in the hot path (cache exists)
            R = sg.R; dU = sg.dU
            @test _fvc2_gpu_allocated(() -> fv_jacobian!(sg, U)) == 0
            @test _fvc2_gpu_allocated(() -> lin_refactorize!(sg.cache)) == 0
            @test _fvc2_gpu_allocated(() -> (lin_solve!(dU, sg.cache, R); CU.synchronize())) == 0
            @test _fvc2_gpu_allocated(() -> fv_residual!(R, U, sg.prob)) == 0
            @test _fvc2_gpu_allocated(() -> KrakeNN._fv_norms(sg.norms, R)) == 0
            # whole Newton iterations from a perturbed state
            U .+= 1e-3 .* CU.CuArray(randn(rng, length(U)))
            @test _fvc2_gpu_allocated(() -> fv_newton!(sg, U; maxit = 4)) == 0
            # free memory over 10 iterations of assembly + refactorize + solve
            frees = Int[]
            for it in 1:10
                fv_jacobian!(sg, U); lin_refactorize!(sg.cache)
                lin_solve!(dU, sg.cache, R); CU.synchronize()
                push!(frees, CU.free_memory())
            end
            spread = (maximum(frees) - minimum(frees)) / 2^20
            @info "C2 GPU hot loop" free_memory_spread_MiB = spread
            @test spread <= 64          # R11 reduced contract ("stability")
        end
    end
end
