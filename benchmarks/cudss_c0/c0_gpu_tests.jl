# C0 GPU receipts: the existing cuDSS / EHD GPU test files plus the new C0
# contract, run in a GPU environment (KrakeNN dev'd by path + CUDA + CUDSS +
# KernelAbstractions + Test). CUDA and CUDSS are loaded BEFORE KrakeNN so the
# extension triggers (Julia 1.12, see benchmarks/results/ehd/README.md).
#
#   KRAKEN_REQUIRE_GPU=1 KRAKEN_TEST_EHD_SIDEWALLS_CUDA=true \
#   KRAKEN_TEST_EC_RESTART_CUDA=true julia --project=<env> benchmarks/cudss_c0/c0_gpu_tests.jl
using CUDA, CUDSS
using Test, KernelAbstractions, LinearAlgebra
using KrakeNN

CUDA.functional() || error("CUDA not functional")
Base.get_extension(KrakeNN, :KrakeNNCUDSSExt) === nothing && error("KrakeNNCUDSSExt not loaded")
println("GPU: ", CUDA.name(CUDA.device()), "  CUDA.jl ", pkgversion(CUDA),
        "  CUDSS.jl ", pkgversion(CUDSS), "  KrakeNN ", pkgversion(KrakeNN))
const TA = joinpath(@__DIR__, "..", "..", "test", "analytical")

@testset "C0 GPU receipts" begin
    for f in ("poisson_cudss_gpu.jl",                 # seam parity (Dirichlet, pinned Neumann)
              "linear_solve_refactor_transpose.jl",   # CPU contract (reference)
              "linear_solve_cudss_c0_gpu.jl",         # C0 cuDSS contract
              "ehd_phi_gpu_parity_2d.jl",             # EHD :direct phi, cuDSS vs UMFPACK
              "ehd_transient_moment_2d.jl",
              "EC-SIDEWALLS.jl",                      # CUDA part opt-in by env var
              "EC-RESTART-CUDA.jl")                   # CUDA part opt-in by env var
        println("\n--- ", f, " ---"); flush(stdout)
        include(joinpath(TA, f))
    end
end
println("DONE")
