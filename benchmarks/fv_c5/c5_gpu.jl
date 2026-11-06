# Prepared only: CUDA/cuDSS M3 contract, then M1/M3 immersed Stokes timings.
# Run from ~/kraken_c5 with --project=env and KRAKEN_REQUIRE_GPU=1.
# There is deliberately no CPU timing mode for 256^2 or 512^2.
using CUDA, CUDSS, Test, LinearAlgebra, Statistics, Printf, Kraken
CUDA.functional() || error("C5 GPU validation requires functional CUDA")
CUDA.allowscalar(false)
println("GPU: ", CUDA.name(CUDA.device()), " CUDA ", pkgversion(CUDA),
        " CUDSS ", pkgversion(CUDSS), " Kraken ", pkgversion(Kraken))
flush(stdout)

const C5_TEST_DIR = joinpath(@__DIR__, "..", "..", "test", "analytical")
ENV["KRAKEN_REQUIRE_GPU"] = "1"
# Includes CPU compiler guards, then the small CPU/GPU comparison. A failed
# receipt stops execution before either large GPU timing is launched.
include(joinpath(C5_TEST_DIR, "incns_fv_adjoint_compiler.jl"))
include(joinpath(C5_TEST_DIR, "incns_fv_immersed_mass_compiler.jl"))
include(joinpath(C5_TEST_DIR, "incns_fv_immersed_gpu.jl"))

function c5_gpu_timing(N, mass)
    GC.gc(); CUDA.reclaim()
    CUDA.synchronize()
    free_before = Int(CUDA.free_memory())
    p = _c5gpu_problem(N; mass)
    s = fv_newton_solver(p; to=CUDA.CuArray, linsolve=CUDABackendTag())
    U = CUDA.zeros(Float64, fv_nunknowns(p))
    opts = (; atol=1e-12, rtol=1e-12, maxit=8)
    # Isolate the first symbolic + numeric factorization from assembly and
    # later numeric refactorizations. Memory differences below are retained
    # device usage after synchronization/reclaim, not peak memory estimates.
    fv_jacobian!(s, U)
    CUDA.synchronize(); CUDA.reclaim()
    free_unfactored = Int(CUDA.free_memory())
    first_factor_seconds = @elapsed begin
        Kraken._fv_factor!(s)
        CUDA.synchronize()
    end
    CUDA.reclaim()
    free_factored = Int(CUDA.free_memory())
    t_newton = @elapsed begin
        st = fv_newton!(s, U; opts...)
        CUDA.synchronize()
    end
    st.converged || error("C5 Newton failed at N=$N")
    @printf("N=%d mass=%s unknowns=%d nnz=%d colors=%d Newton=%.6f s iterations=%d residual_inf=%.4e\n",
        N, string(mass), length(U), s.asm.nnz, s.asm.ncolors, t_newton, st.iterations, st.inf[end])
    @printf("N=%d mass=%s first_factor=%.6f s numeric_refactor=%.6f s factor_retained=%d B system_retained=%d B\n",
        N, string(mass), first_factor_seconds, median(st.t_factor),
        free_unfactored-free_factored, free_before-free_factored)
    map = FVUniformViscosity()
    ws = fv_adjoint_workspace(s, map)
    for (kind, obj) in _c5gpu_objectives()
        first_time = @elapsed begin
            r = fv_steady_gradient(s, U, [0.1], obj; params=map, ws)
            CUDA.synchronize()
        end
        fv_steady_gradient(s, U, [0.1], obj; params=map, ws, finalize=false)
        CUDA.synchronize()
        times = Float64[]
        for _ in 1:3
            push!(times, @elapsed begin
                fv_steady_gradient(s, U, [0.1], obj; params=map, ws, finalize=false)
                CUDA.synchronize()
            end)
        end
        bytes = CUDA.@allocated begin
            fv_steady_gradient(s, U, [0.1], obj; params=map, ws, finalize=false)
            CUDA.synchronize()
        end
        @printf("N=%d mass=%s kind=%s J=%.12e dJ_dnu=%.12e first=%.6f s hot=%.6f s device=%d B\n",
            N, string(mass), string(kind), r.J, r.grad[1], first_time, median(times), bytes)
        bytes == 0 || error("C5 hot adjoint allocated device memory")
    end
    U .+= 1e-4
    fv_newton!(s, U; opts...).converged || error("C5 Newton warm-up failed")
    U .+= 1e-4
    receipt = Ref{Any}()
    newton_bytes = CUDA.@allocated begin
        receipt[] = fv_newton!(s, U; opts...)
        CUDA.synchronize()
    end
    receipt[].converged && receipt[].iterations > 0 || error("C5 measured Newton did no converged work")
    newton_bytes == 0 || error("C5 hot Newton allocated device memory")
    @printf("N=%d mass=%s hot_Newton_device=%d B free_device=%.3f GiB\n",
        N, string(mass), newton_bytes, CUDA.free_memory()/2.0^30)
    s.n_factorize == 1 || error("C5 unexpectedly repeated symbolic analysis")
    flush(stdout)
    return (;N, mass, unknowns=length(U), nonzeros=s.asm.nnz, colors=s.asm.ncolors,
        first_factor_seconds, numeric_refactor_seconds=median(st.t_factor),
        factor_retained_bytes=free_unfactored-free_factored,
        system_retained_bytes=free_before-free_factored,
        newton_seconds=t_newton, newton_iterations=st.iterations,
        residual_inf=st.inf[end], hot_newton_device_bytes=newton_bytes)
end

let records = NamedTuple[]
    for N in (256, 512), mass in (:M1, :M3)
        push!(records, c5_gpu_timing(N, mass))
    end
    output = get(ENV, "KRAKEN_C5_GPU_OUTPUT", joinpath(@__DIR__, "c5_gpu_comparison.csv"))
    open(output, "w") do io
        println(io, join(string.(keys(first(records))), ","))
        for record in records
            println(io, join(string.(values(record)), ","))
        end
    end
end
println("DONE status=0")
