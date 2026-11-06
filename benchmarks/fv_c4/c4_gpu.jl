# C4 GPU receipts: the C4 test files (CPU adjoint checks + GPU/cuDSS adjoint
# contract) with KRAKEN_REQUIRE_GPU=1, then the cost of one adjoint gradient
# relative to one Newton solve on the lid-driven cavity at 512² and 1024²
# (Re = 100, from rest).
#
#   KRAKEN_REQUIRE_GPU=1 julia --project=<env> benchmarks/fv_c4/c4_gpu.jl
#   C4_BENCH_CPU=1 julia --project benchmarks/fv_c4/c4_gpu.jl   (CPU smoke run,
#       32² and 64², UMFPACK, no CUDA needed, test files skipped)
#
# <env> holds Kraken (dev'd by path), CUDA 6, CUDSS 1 (transposed solve needs
# CUDSS.jl >= 1.0), ForwardDiff, KernelAbstractions, Test. Per case: Newton
# wall time and iterations; per parameter map (uniform ν, lid speed, ν field)
# and objective (lid drag, u-profile misfit): the first gradient (finalize =
# true: re-assembly + refactorization, workspace built inside) split into
# objective / finalize / transposed solve / parameter cotangent / explicit
# term, then the median of 5 hot gradients with a workspace and finalize =
# false, the ratios to the Newton solve and to one Newton iteration, and
# CUDA.@allocated of a hot gradient and each constituent phase. The built-in
# hot paths must allocate zero device bytes; host result arrays are excluded.
const CPU = get(ENV, "C4_BENCH_CPU", "0") in ("1", "true")
if !CPU
    using CUDA
    println("CUDA.functional() = ", CUDA.functional()); flush(stdout)
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
using Test, LinearAlgebra, Random, Printf, Statistics
using Kraken
const TA = joinpath(@__DIR__, "..", "..", "test", "analytical")
include(joinpath(TA, "fv_newton_helpers.jl"))
CPU || println("GPU: ", CUDA.name(CUDA.device()), "  CUDA.jl ", pkgversion(CUDA), "  CUDSS.jl ",
               pkgversion(CUDSS), "  Kraken ", pkgversion(Kraken), "  threads ", Threads.nthreads())
CPU && println("CPU smoke run, Kraken ", pkgversion(Kraken), ", threads ", Threads.nthreads())

const STATUS = Ref(0)
if !CPU
    println("\n--- C4 test files ---"); flush(stdout)
    try
        @testset "C4 GPU receipts" begin
            for f in ("incns_fv_adjoint.jl", "incns_fv_adjoint_gpu.jl")
                println("\n--- ", f, " ---"); flush(stdout)
                include(joinpath(TA, f))
            end
        end
    catch err
        println("TESTS FAILED: ", sprint(showerror, err)); STATUS[] = 1
    end
    flush(stdout)
end

sync() = CPU || CUDA.synchronize()
# macro expanded only when CUDA is loaded (the CPU smoke run has no CUDA)
CPU ? (devalloc(f) = -1) : @eval(devalloc(f) = CUDA.@allocated f())
gib(x) = x / 2^30

function allocation_phases(s, U, θ, obj, m, ws)
    hp = Kraken._fv_host_problem(s.prob)
    Uh = Vector{Float64}(Array(U))
    _, gU = Kraken._fv_objective_grad_u(obj, Uh, hp)
    θh = m isa FVViscosityField ? vec(hp.nu) : Vector{Float64}(θ)
    phases = (
        ("signature", () -> Kraken._fv_state_sig(s.norms, 0.0, U)),
        ("host copies", () -> (Kraken._fv_host_problem(s.prob), Array(U))),
        ("objective", () -> Kraken._fv_objective_grad_u(obj, Uh, hp)),
        ("RHS copy", () -> copyto!(ws.g, gU)),
        ("transpose", () -> lin_solve!(ws.λ, s.cache, ws.g; transpose = true)),
        ("cotangent", () -> Kraken._fv_param_cotangent!(ws, m, s, U, θ)),
        ("residual", () -> fv_residual!(s.R, U, s.prob)),
        ("norms", () -> Kraken._fv_norms(s.norms, s.R)),
        ("explicit", () -> Kraken._fv_objective_grad_theta(obj, m, Uh, hp, θh)))
    for (name, f) in phases
        sync()
        bytes = devalloc(f)
        @printf("    hot device allocation %-11s %d B\n", name, bytes)
        bytes == 0 || (STATUS[] = 1)
    end
end

function bench(N, Re)
    println("\n================ cavity N = $(N)², Re = $Re (from rest) ================"); flush(stdout)
    GC.gc(); CPU || CUDA.reclaim()
    p = fv_steady_problem(N, N; nu = 1 / Re, boundary_velocity = _fvc2_lid)
    s = CPU ? fv_newton_solver(p) : fv_newton_solver(p; to = CUDA.CuArray, linsolve = CUDABackendTag())
    n = fv_nunknowns(p)
    U = CPU ? zeros(n) : CUDA.zeros(Float64, n)
    t_newton = @elapsed st = fv_newton!(s, U; maxit = 40)
    tref = st.iterations > 1 ? median(st.t_factor[2:end]) : st.t_factor[1]
    t_it = tref + median(st.t_assembly) + median(st.t_solve)
    @printf("  unknowns %d | Newton converged %s in %d iterations, %.2f s wall | one iteration (assembly + refactorize + solve, median) %.3f s\n",
            n, st.converged, st.iterations, t_newton, t_it)
    lp = LineProfile(:u, [(N ÷ 2, j) for j in 1:max(1, N ÷ 16):N])
    objs = (("lid drag", FVWallForce(1; region = :N)),
            ("u profile", FVStateObs(p, lp, zeros(length(lp.indices)))))
    maps = (("uniform nu", FVUniformViscosity(), [1 / Re]),
            ("lid speed", FVSideVelocity(p, :N), [1.0]),
            ("nu field", FVViscosityField(), nothing))
    println("  map        | objective | first: total s (obj / finalize / Jᵀ solve / param / explicit) | ws build s | hot median s | hot/Newton | hot/iteration | hot device B")
    for (mname, map) in ((m[1], (m[2], m[3])) for m in maps), (oname, obj) in objs
        m, θ = map
        try
            t0 = time_ns()
            r = fv_steady_gradient(s, U, θ, obj; params = m)
            sync(); tf = (time_ns() - t0) / 1e9
            tw = @elapsed (ws = fv_adjoint_workspace(s, m); sync())
            fv_steady_gradient(s, U, θ, obj; params = m, ws, finalize = false)  # warm
            hot = Float64[]
            hot_obj = Float64[]; hot_explicit = Float64[]
            for _ in 1:5
                t1 = time_ns(); rh = fv_steady_gradient(s, U, θ, obj; params = m, ws, finalize = false); sync()
                push!(hot, (time_ns() - t1) / 1e9)
                push!(hot_obj, rh.t_objective); push!(hot_explicit, rh.t_explicit)
            end
            th = median(hot)
            a = devalloc(() -> fv_steady_gradient(s, U, θ, obj; params = m, ws, finalize = false))
            @printf("  %-10s | %-9s | %.3f (%.3f / %.3f / %.3f / %.3f / %.3f) | %.3f | %.4f | %.3f | %.2f | %d\n",
                    mname, oname, tf, r.t_objective, r.t_finalize, r.t_solve, r.t_param, r.t_explicit,
                    tw, th, th / t_newton, th / t_it, a)
            @printf("    hot objective %.6f s | explicit %.6f s | (objective + explicit)/iteration %.4f\n",
                    median(hot_obj), median(hot_explicit), median(hot_obj .+ hot_explicit) / t_it)
            if !CPU
                a == 0 || (STATUS[] = 1)
                allocation_phases(s, U, θ, obj, m, ws)
            end
        catch err
            println("  GRADIENT FAILED $mname / $oname: ", sprint(showerror, err)); STATUS[] = 1
        end
        flush(stdout)
    end
    @printf("  n_factorize after all gradients: %d (1 = no new analysis)\n", s.n_factorize)
    s.n_factorize == 1 || (STATUS[] = 1)
    CPU || @printf("  device free memory at end of case: %.2f GiB\n", gib(CUDA.free_memory()))
    s = nothing; U = nothing
    GC.gc(); CPU || CUDA.reclaim()
    return nothing
end

for N in (CPU ? (32, 64) : (512, 1024))
    try
        bench(N, 100.0)
    catch err
        println("BENCH FAILED N=$(N): ", sprint(showerror, err)); STATUS[] = 1
    end
end
println("DONE status=", STATUS[])
exit(STATUS[])
