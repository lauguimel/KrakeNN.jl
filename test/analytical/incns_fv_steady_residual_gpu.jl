# GPU contract of the C1 steady FV residual (src/methods/inc_ns/fv_steady_residual.jl):
#   - the same kernel on CUDABackend agrees with the CPU backend (Float64 state,
#     and a Dual{4} state as C2 will feed it), on a lid-driven cavity with
#     variable viscosity, a masked inlet/outlet channel and a periodic box;
#   - the hot path is allocation-free on the device (`CUDA.@allocated == 0`)
#     and the device free memory is stable over 10 evaluations (R11, reduced
#     contract).
#
# Self-gated like linear_solve_cudss_c0_gpu.jl: skipped when CUDA is not
# loadable or not functional; KRAKEN_REQUIRE_GPU=1 turns the skip into a failure.

using Test
using LinearAlgebra
using Random
using ForwardDiff
using KrakeNN

const _FVC1_GPU_OK = (try
    @eval Main using CUDA
    @eval Main CUDA.functional()
catch
    false
end) === true

const _FVC1_REQUIRE_GPU = get(ENV, "KRAKEN_REQUIRE_GPU", "0") in ("1", "true")

if _FVC1_GPU_OK
    @eval _fvc1_gpu_allocated(f) = Main.CUDA.@allocated f()
end

struct _FVC1GPUTag end

@testset "IncNS steady FV residual (C1, GPU)" begin
    if !_FVC1_GPU_OK
        if _FVC1_REQUIRE_GPU
            @error "KRAKEN_REQUIRE_GPU=1 but CUDA is not functional"
            @test _FVC1_GPU_OK
        else
            @info "Skipping C1 FV residual GPU tests (CUDA not functional here)"
            @test_skip "C1 FV residual GPU contract requires CUDA.functional()"
        end
    else
        CU = Main.CUDA
        rng = MersenneTwister(5)
        solid = zeros(Bool, 64, 32); solid[20:28, 1:14] .= true; solid[40:44, 18:32] .= true
        cases = (
            ("cavity NS nu(x,y) 96²",
             fv_steady_problem(96, 96; nu = (x, y) -> 0.01 * (1 + x * y),
                               boundary_velocity = (s, x, y) -> s === :N ? (1.0, 0.0) : (0.0, 0.0))),
            ("masked inlet/outlet channel 64x32",
             fv_steady_problem(64, 32; nu = 0.02, solid,
                               bc = (W = :inlet, E = :outlet, S = :wall, N = :wall),
                               boundary_velocity = (s, x, y) -> s === :W ? (4y * (1 - y), 0.0) : (0.0, 0.0))),
            ("periodic box Stokes 40²",
             fv_steady_problem(40, 40; nu = (x, y) -> 1 + 0.3 * sin(6x), conv = false,
                               bc = (W = :periodic, E = :periodic, S = :periodic, N = :periodic),
                               source = (x, y) -> (cos(6y), 0.0, 0.0))),
        )
        for (name, p) in cases
            @testset "$name" begin
                n = fv_nunknowns(p)
                U = randn(rng, n)
                dp = fv_adapt(CU.CuArray, p)
                dU = CU.CuArray(U)
                dR = CU.zeros(Float64, n)
                Rc = fv_residual(U, p)
                fv_residual!(dR, dU, dp)
                @test norm(Array(dR) - Rc, Inf) / norm(Rc, Inf) <= 1e-14

                # Dual{4} state (C2 seeds four colours per pass)
                D = ForwardDiff.Dual{_FVC1GPUTag,Float64,4}
                Ud = [D(U[k], ForwardDiff.Partials(ntuple(m -> m == (k % 4) + 1 ? 1.0 : 0.0, 4)))
                      for k in 1:n]
                Rdc = fv_residual(Ud, p)
                dUd = CU.CuArray(Ud)
                dRd = CU.CuArray(similar(Ud))
                fv_residual!(dRd, dUd, dp)
                Rdg = Array(dRd)
                @test norm(ForwardDiff.value.(Rdg) - ForwardDiff.value.(Rdc), Inf) /
                      norm(ForwardDiff.value.(Rdc), Inf) <= 1e-14
                pc = reduce(hcat, [collect(ForwardDiff.partials(r)) for r in Rdc])
                pg = reduce(hcat, [collect(ForwardDiff.partials(r)) for r in Rdg])
                @test norm(pg - pc, Inf) / norm(pc, Inf) <= 1e-14

                # R11 reduced contract: zero device allocation, stable free memory
                fv_residual!(dR, dU, dp); fv_residual!(dRd, dUd, dp)   # warm-up
                @test _fvc1_gpu_allocated(() -> fv_residual!(dR, dU, dp)) == 0
                @test _fvc1_gpu_allocated(() -> fv_residual!(dRd, dUd, dp)) == 0
                frees = Int[]
                for it in 1:10
                    fv_residual!(dR, dU, dp)
                    fv_residual!(dRd, dUd, dp)
                    push!(frees, CU.free_memory())
                end
                spread = maximum(frees) - minimum(frees)
                @info "C1 GPU residual" case = name free_memory_spread_MiB = spread / 2^20
                @test spread <= 2^21   # 2 MiB: driver-level noise; C0 measured 0
            end
        end
    end
end
