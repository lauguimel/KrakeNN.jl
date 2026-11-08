# GPU / cuDSS contract of the C3 GNF path (src/methods/inc_ns/fv_gnf.jl):
#   - the GNF residual on CUDABackend equals the CPU one to 1e-12, including
#     with a law of ForwardDiff duals (∂R/∂θ, the C4 path);
#   - the coloured dual assembly of the GNF Jacobian on CUDABackend, written
#     into a CuSparseMatrixCSR, equals the CPU (CSC) assembly to 1e-12;
#   - Newton on cuDSS for the power-law cavity follows the CPU residual
#     history and reaches the same state; ONE analysis, then refactorizations;
#   - the per-iteration hot path allocates nothing on the device.
#
# Self-gated like incns_fv_newton_gpu.jl: skipped when CUDA/CUDSS are not
# loadable or CUDA is not functional; KRAKEN_REQUIRE_GPU=1 turns the skip into
# a failure.

using Test
using LinearAlgebra
using Random
using ForwardDiff
using KrakeNN
using KrakeNN.SparseArrays

if !isdefined(@__MODULE__, :_fvc2_ghia_error)
    include(joinpath(@__DIR__, "fv_newton_helpers.jl"))
end

const _FVC3_GPU_OK = (try
    @eval Main using CUDA, CUDSS
    @eval Main CUDA.functional()
catch
    false
end) === true

const _FVC3_REQUIRE_GPU = get(ENV, "KRAKEN_REQUIRE_GPU", "0") in ("1", "true")

if _FVC3_GPU_OK
    @eval _fvc3_gpu_allocated(f) = Main.CUDA.@allocated f()
end

_fvc3_host(J) = sparse(transpose(SparseMatrixCSC(size(J)..., Array(J.rowPtr), Array(J.colVal),
                                                 Array(J.nzVal))))

@testset "IncNS steady FV GNF (C3, GPU)" begin
    if !_FVC3_GPU_OK
        if _FVC3_REQUIRE_GPU
            @error "KRAKEN_REQUIRE_GPU=1 but CUDA/CUDSS is not functional"
            @test _FVC3_GPU_OK
        else
            @info "Skipping C3 FV GNF GPU tests (CUDA not functional here)"
            @test_skip "C3 FV GNF GPU contract requires CUDA.functional()"
        end
    else
        CU = Main.CUDA
        rng = MersenneTwister(17)
        thin = PowerLaw(0.01, 0.5; nu_min = 0.0, nu_max = Inf)
        solid = zeros(Bool, 64, 32); solid[20:28, 1:14] .= true; solid[40:44, 18:32] .= true
        cases = (
            ("power-law cavity 64²",
             fv_steady_problem(64, 64; rheology = thin, gnf_eps = 1e-2, boundary_velocity = _fvc2_lid)),
            ("Carreau-Yasuda masked inlet/outlet channel 64x32",
             fv_steady_problem(64, 32; rheology = CarreauYasuda(0.05, 0.001, 2.0, 2.0, 0.5), solid,
                               bc = (W = :inlet, E = :outlet, S = :wall, N = :wall),
                               boundary_velocity = (s, x, y) -> s === :W ? (4y * (1 - y), 0.0) : (0.0, 0.0))),
        )
        for (name, p) in cases
            @testset "residual and assembly parity: $name" begin
                pd = fv_adapt(CU.CuArray, p)
                sc = fv_newton_solver(p)
                sg = fv_newton_solver(p; to = CU.CuArray, linsolve = CUDABackendTag())
                for trial in 1:2
                    U = 0.3 .* randn(rng, fv_nunknowns(p))
                    Ud = CU.CuArray(U)
                    Rc = fv_residual(U, p)
                    @test norm(Array(fv_residual(Ud, pd)) - Rc, Inf) <= 1e-12 * max(norm(Rc, Inf), 1)
                    fv_jacobian!(sc, U)
                    fv_jacobian!(sg, Ud)
                    Jg = _fvc3_host(sg.J)
                    @test Jg.colptr == sc.J.colptr && rowvals(Jg) == rowvals(sc.J)
                    @test norm(nonzeros(Jg) - nonzeros(sc.J), Inf) <= 1e-12 * norm(nonzeros(sc.J), Inf)
                    # law of duals on the device: ∂R/∂θ (C4 path)
                    θ = fv_gnf_parameters(p)
                    D = ForwardDiff.Dual{Nothing}
                    θd = [D(θ[k], ForwardDiff.Partials((k == 1 ? 1.0 : 0.0,))) for k in eachindex(θ)]
                    Rdc = fv_residual(U, fv_with_gnf_parameters(p, θd))
                    Rdg = Array(fv_residual(Ud, fv_with_gnf_parameters(pd, θd)))
                    dc = ForwardDiff.partials.(Rdc, 1); dg = ForwardDiff.partials.(Rdg, 1)
                    @test norm(dg - dc, Inf) <= 1e-12 * max(norm(dc, Inf), 1)
                end
            end
        end

        @testset "Newton on cuDSS: power-law cavity 64²" begin
            p = cases[1][2]
            rc = solve_incns_fv_newton(; problem = p, maxit = 40)
            sg = fv_newton_solver(p; to = CU.CuArray, linsolve = CUDABackendTag())
            U = CU.zeros(Float64, fv_nunknowns(p))
            st = fv_newton!(sg, U; maxit = 40)
            @info "C3 Newton, power-law cavity n=0.5 64² (GPU, cuDSS)" residual_2norm = st.history
            @test rc.converged && st.converged
            @test st.iterations == rc.iterations
            k = findall(>(1e-10), rc.residual_history)
            @test maximum(abs.(st.history[k] ./ rc.residual_history[k] .- 1)) < 1e-6
            @test norm(Array(U) - Array(rc.U), Inf) < 1e-10
            @test sg.n_factorize == 1 && sg.n_refactorize == st.iterations - 1
            fv_newton_finalize!(sg, U)
            Jh = _fvc3_host(sg.J)
            b = randn(rng, length(U)); bd = CU.CuArray(b); x = similar(bd)
            lin_solve!(x, sg.cache, bd)
            @test norm(Jh * Array(x) - b) / norm(b) < 1e-10
            if lin_supports_transpose(sg.cache)
                lin_solve!(x, sg.cache, bd; transpose = true)
                @test norm(transpose(Jh) * Array(x) - b) / norm(b) < 1e-10
            end
            R = sg.R; dU = sg.dU
            @test _fvc3_gpu_allocated(() -> fv_jacobian!(sg, U)) == 0
            @test _fvc3_gpu_allocated(() -> fv_residual!(R, U, sg.prob)) == 0
            U .+= 1e-4 .* CU.CuArray(randn(rng, length(U)))
            @test _fvc3_gpu_allocated(() -> fv_newton!(sg, U; maxit = 4)) == 0
        end
    end
end
