# C0 — EHD electroconvection time per step with the :direct (cuDSS) phi solve,
# and a bitwise fingerprint of the fields, so the SAME script run on the
# pre-C0 code (origin/dev/platform) and on feat/cudss-c0 gives the before/after
# timing and checks that the results are unchanged.
# Configuration of benchmarks/results/ehd/README.md (197 x 321, MRT, regularized
# charge, force projection xy; reference 1.39 ms/step on H100 with cuDSS).
# Uses only API present before C0.
#
#   julia --project=<env> benchmarks/cudss_c0/c0_ehd_step_bench.jl
# Env: C0_EHD_STEPS (timed steps, default 2000), C0_EHD_WARM (default 200).
using CUDA, CUDSS
using Kraken
using Printf

CUDA.functional() || error("CUDA not functional")
const NSTEP = parse(Int, get(ENV, "C0_EHD_STEPS", "2000"))
const NWARM = parse(Int, get(ENV, "C0_EHD_WARM", "200"))
println("GPU: ", CUDA.name(CUDA.device()), "  CUDA.jl ", pkgversion(CUDA),
        "  CUDSS.jl ", pkgversion(CUDSS), "  Kraken ", pkgversion(Kraken))

cfg = (; Nx = 197, Ny = 321, C = 10.0, M = 10.0, T = 166.0, ns_scheme = :mrt,
       charge_scheme = :regularized, phi_scheme = :direct, force_projection = :xy,
       backend = CUDA.CUDABackend(), FT = Float64)
s = init_state(ECState; cfg...)
println("phi setup type: ", nameof(typeof(s.poisson_setup)))
advance!(s, NWARM)
CUDA.synchronize()
t = @elapsed begin
    advance!(s, NSTEP)
    CUDA.synchronize()
end
@printf("EHD 197x321 :direct  %d steps  %.3f ms/step\n", NSTEP, 1e3 * t / NSTEP)

r = solution(s).result
for k in (:phi, :q, :ux, :uy, :Ex, :Ey)
    a = getproperty(r, k)
    @printf("fingerprint %-3s sum = %.17e  bits hash = %016x\n", k, sum(a),
            hash(reinterpret(UInt64, vec(Float64.(a)))))
end
println("DONE")
