# Cost of the 3D GNF steady adjoint (src/ad/gnf3d_adjoint*.jl) on CUDA, Float64.
# Case: N³ periodic box, walls j = 1 and j = N (plane channel, flow x) with a voxel
# sphere of radius N/6 at the centre, DenseNet law of lbm3d_law_test.jl
# (1 → 8 → 8 → 1) through LBMLawViscosity, body force F = 1/N³ (diffusive scaling).
# For each N it reports, in this order:
#   forward   steps to tol, wall time, MLUPS of the fused step (GPU);
#   linearise host time of gnf3d_linearise (20×21 ForwardDiff blocks, threads);
#   operator  device copy of the blocks; time of one vjp product (mean of 50);
#   adjoint   bordered GMRES and K-step solves of J = mean u_x: iterations,
#             explicit residual, wall time;
#   gradient  host pullback time;
#   memory    device used bytes (total − free) sampled after each phase, maximum
#             of the samples (a sampled peak, not the allocator high-water mark),
#             and host Sys.maxrss().
# Run on a full GPU for the numbers to mean anything (a MIG slice gives
# correctness only). Command (cluster):
#   julia -t 16 --project=. benchmarks/lbm3d_law/cost_gnf3d_adjoint.jl
# Environment:
#   KRAKEN_GNF3D_N         sizes, comma separated (default "32,64")
#   KRAKEN_GNF3D_TOL       forward steady tolerance (default 1e-10)
#   KRAKEN_GNF3D_MAXSTEPS  forward step budget (default 300000)
#   KRAKEN_GNF3D_RESTART   GMRES restart length (default 200; memory = restart × 20 N³ × 8 B)
#   KRAKEN_GNF3D_ATOL      adjoint tolerance (default 1e-10)
#   KRAKEN_GNF3D_SOLVERS   "gmres,kstep" (default) or a subset
#   KRAKEN_GNF3D_CPU_SMOKE=1  same code on the CPU (script check only; use N = 12)
using KrakeNN, LinearAlgebra, Random, Printf

const _C3_SMOKE = get(ENV, "KRAKEN_GNF3D_CPU_SMOKE", "0") in ("1", "true")
if !_C3_SMOKE
    using CUDA   # top level only, never inside a function
end

_c3_backend() = _C3_SMOKE ? KrakeNN.KernelAbstractions.CPU() : CUDA.CUDABackend()
_c3_to() = _C3_SMOKE ? Array : CUDA.CuArray
_c3_sync() = _C3_SMOKE ? nothing : CUDA.synchronize()
_c3_used() = _C3_SMOKE ? 0.0 : Float64(CUDA.total_memory() - CUDA.free_memory())

function _c3_net()
    net = DenseNet(1, (8, 8); activation=:tanh, output=:exp, scale=(5.0,))
    theta = law_init(net; rng=MersenneTwister(3)) .* 10
    theta[end] = log(0.1)
    return net, theta
end

function _c3_solid(N)
    s = falses(N, N, N); s[:, 1, :] .= true; s[:, N, :] .= true
    c = (N + 1) / 2; R = N / 6
    for k in 1:N, j in 1:N, i in 1:N
        (i - c)^2 + (j - c)^2 + (k - c)^2 <= R^2 && (s[i, j, k] = true)
    end
    return Array{Bool}(s)
end

function _c3_run(N, net, theta)
    tol = parse(Float64, get(ENV, "KRAKEN_GNF3D_TOL", "1e-10"))
    maxsteps = parse(Int, get(ENV, "KRAKEN_GNF3D_MAXSTEPS", "300000"))
    restart = parse(Int, get(ENV, "KRAKEN_GNF3D_RESTART", "200"))
    atol = parse(Float64, get(ENV, "KRAKEN_GNF3D_ATOL", "1e-10"))
    solvers = Symbol.(split(get(ENV, "KRAKEN_GNF3D_SOLVERS", "gmres,kstep"), ","))
    be = _c3_backend()
    solid = _c3_solid(N); fl = .!solid; nfl = count(fl)
    F = (1.0 / N^3, 0.0, 0.0)
    cld = LBMLawViscosity(net, _c3_to()(theta); eps=1e-3, nu_scale=1.0,
                          gamma_scale=Float64(N)^2)
    used = Float64[_c3_used()]

    run_gnf3d_periodic_steady(; is_solid=solid, closure=cld, F, tol=0.0, max_steps=10,
                              backend=be)                                   # compile
    _c3_sync()
    tf = @elapsed r = run_gnf3d_periodic_steady(; is_solid=solid, closure=cld, F, tol,
                                                 check_every=500, max_steps=maxsteps,
                                                 backend=be)
    push!(used, _c3_used())
    mlups = N^3 * r.steps / tf / 1e6
    @printf("N=%d forward: %d steps, converged=%s, %.2f s, %.1f MLUPS (fluid %d of %d nodes)\n",
            N, r.steps, r.converged, tf, mlups, nfl, N^3)

    gnf3d_linearise(r.f[1:2, 1:2, 1:2, :], r.tau[1:2, 1:2, 1:2], solid[1:2, 1:2, 1:2],
                    cld, F)                                                 # compile
    lin = gnf3d_linearise(r.f, r.tau, solid, cld, F)
    @printf("N=%d linearise: %.2f s on %d threads (host blocks %.2f GB)\n",
            N, lin.time, Threads.nthreads(), sizeof(lin.B) / 1e9)

    to = @elapsed (op = GNF3DStepOperator(lin; backend=be); _c3_sync())
    push!(used, _c3_used())
    x = _c3_to()(randn(MersenneTwister(1), op.n)); y = similar(x)
    gnf3d_vjp!(y, op, x); _c3_sync()
    tv = @elapsed (for _ in 1:50; gnf3d_vjp!(y, op, x); end; _c3_sync())
    @printf("N=%d operator: copy %.2f s (device blocks %.2f GB), vjp %.3f ms\n",
            N, to, sizeof(op.B) / 1e9, 1e3 * tv / 50)

    gux = zeros(size(solid)); gux[fl] .= N / nfl; z = zeros(size(solid))
    b = similar(op.chat, op.n); copyto!(b, gnf3d_velocity_rhs(lin, (gux, z, z)))
    grads = Dict{Symbol,Vector{Float64}}()
    for sv in solvers
        kw = sv === :gmres ? (; tol=atol, restart, maxiter=50_000) : (; tol=atol)
        ta = @elapsed ((λ, _, info) = gnf3d_adjoint_solve(op, b; solver=sv, kw...); _c3_sync())
        push!(used, _c3_used())
        s = gnf3d_nu_sensitivity(lin, λ)
        tg = @elapsed (grads[sv] = gnf3d_law_gradient(lin, s, cld))
        @printf("N=%d adjoint %s: converged=%s, %d iters, relres %.2e, %.2f s; pullback %.3f s\n",
                N, sv, info.converged, info.iters, info.relres, ta, tg)
    end
    if haskey(grads, :gmres) && haskey(grads, :kstep)
        @printf("N=%d gradient gap kstep vs gmres: %.2e\n", N,
                norm(grads[:kstep] .- grads[:gmres]) / norm(grads[:gmres]))
    end
    @printf("N=%d memory: device used (sampled max) %.2f GB, host maxrss %.2f GB\n",
            N, maximum(used) / 1e9, Sys.maxrss() / 1e9)
    return nothing
end

function main()
    net, theta = _c3_net()
    Ns = parse.(Int, split(get(ENV, "KRAKEN_GNF3D_N", "32,64"), ","))
    @printf("3D GNF adjoint cost, %s, %d host threads\n",
            _C3_SMOKE ? "CPU smoke" : "CUDA " * string(CUDA.name(CUDA.device())),
            Threads.nthreads())
    for N in Ns
        _c3_run(N, net, theta)
        GC.gc(); _C3_SMOKE || CUDA.reclaim()
    end
end

main()
