# GPU direct-solve test for ext/KrakenCUDSSExt (cuDSS seam backend).
# Self-gated: requires CUDA + CUDSS loadable AND CUDA.functional() — skipped
# cleanly on CPU-only boxes (the guard wraps the LOAD, so real ext failures
# still surface on a GPU box).

using Test

const _CUDSS_GPU_OK = (try
    @eval Main using CUDA, CUDSS
    @eval Main CUDA.functional()
catch
    false
end) === true

cudss_dirichlet_exact(x, y) = sin(pi * x) * sin(pi * y)
cudss_dirichlet_rhs(x, y)   = 2.0 * pi^2 * sin(pi * x) * sin(pi * y)
cudss_neumann_rhs(x, y)     = 2.0 * pi^2 * cos(pi * x) * cos(pi * y)

@testset "Poisson cuDSS direct GPU (ext)" begin
    if !_CUDSS_GPU_OK
        @info "Skipping cuDSS GPU Poisson test and KrakenCUDSSExt activation check (CUDA+CUDSS not loadable or CUDA not functional here)"
        @test_skip "cuDSS GPU direct path requires CUDA.functional() + CUDSS"
    else
        # #61: with CUDA + CUDSS loaded the package extension must be active.
        # (A wrong weakdep UUID in Project.toml leaves it silently inactive.)
        @test Base.get_extension(Kraken, :KrakenCUDSSExt) !== nothing

        N = 64

        # Dirichlet: parity vs the CPU CHOLMOD seam on the SAME operator.
        u_gpu = solve_poisson_direct(cudss_dirichlet_rhs, N;
                                     method=CUDABackendTag())
        u_cpu = solve_poisson_dirichlet(N, cudss_dirichlet_rhs)
        linf = maximum(abs.(u_gpu .- u_cpu))
        @test linf <= 1e-8
        @test Kraken.l2_error(u_gpu, cudss_dirichlet_exact, N) <= 2e-3

        # Neumann + pin: gauge-fixed parity vs the CPU pinned path (both pin
        # DOF 1 to zero through the same seam contract).
        u_gpu_n = solve_poisson_direct(cudss_neumann_rhs, N; bc=:neumann,
                                       method=CUDABackendTag())
        u_cpu_n = solve_poisson_direct(cudss_neumann_rhs, N; bc=:neumann,
                                       method=CPUBackendTag())
        linf_n = maximum(abs.(u_gpu_n .- u_cpu_n))
        @test linf_n <= 1e-8

        @info "cuDSS GPU Poisson parity" N=N Linf_dirichlet=linf Linf_neumann=linf_n
    end
end
