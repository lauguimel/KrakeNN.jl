# Prepared CUDA/cuDSS validation. Default size: 512^2, Float64, no CPU solve.
using Test, LinearAlgebra, Random, Statistics, Printf, Kraken

# Plain top-level imports: an `@eval using CUDA` inside the same expression as
# its first use hits the world-age trap and kills the job at start-up.
using CUDA, CUDSS
CUDA.functional() || error("FV law GPU validation requires functional CUDA")
const LAW_GPU_READY = true
const LAW_REQUIRE_GPU = true
law_device_allocated(f) = CUDA.@allocated f()

const LAW_NEWTON = (; atol=1e-12, rtol=1e-12, maxit=40)
const LAW_EPS = 1e-2

function law_channel(N; kwargs...)
    return fv_steady_problem(N, N; h=1/N, conv=false,
        bc=(W=:periodic, E=:periodic, S=:wall, N=:wall),
        source=(x,y)->(0.05,0.0,0.0), kwargs...)
end

function law_gpu_solve(p)
    s = fv_newton_solver(p; to=CUDA.CuArray, linsolve=CUDABackendTag())
    if s.prob.visc isa FVLawViscosity
        @test s.prob.visc.theta isa CUDA.CuArray
        @test all(field -> field isa CUDA.CuArray, s.prob.visc.fields)
        @test isbitstype(typeof(CUDA.cudaconvert(s.prob.visc)))
    end
    U = CUDA.zeros(Float64, fv_nunknowns(p))
    measured = @timed begin
        st = fv_newton!(s, U; LAW_NEWTON...)
        CUDA.synchronize()
        st
    end
    measured.value.converged || error("GPU Newton did not converge")
    return s, U, measured
end

function law_gpu_fd_objective!(s, U0, map, theta, obj)
    fv_set_params!(s, map, theta)
    U = copy(U0)
    st = fv_newton!(s, U; LAW_NEWTON...)
    st.converged || error("Perturbed GPU Newton did not converge")
    # The objective is evaluated on the host, matching fv_steady_gradient.
    return fv_objective(obj, Array(U), Kraken._fv_host_problem(s.prob))
end

function law_write_csv(path, records)
    open(path, "w") do io
        println(io, join(string.(keys(first(records))), ","))
        foreach(row -> println(io, join(string.(values(row)), ",")), records)
    end
end

function law_gpu_main()
    CUDA.allowscalar(false)
    N = parse(Int, get(ENV, "KRAKEN_LAW_N", "512"))
    N >= 4 || error("KRAKEN_LAW_N must be at least 4")
    output = get(ENV, "KRAKEN_LAW_OUTPUT", @__DIR__)
    mkpath(output)
    println("GPU=", CUDA.name(CUDA.device()), " N=", N, " precision=Float64 CUDA=",
        pkgversion(CUDA), " CUDSS=", pkgversion(CUDSS), " Kraken=", pkgversion(Kraken))
    flush(stdout)

    classical = CarreauYasuda(0.3, 0.03, 2.0, 2.0, 0.7)
    adapter = ClassicalLaw(classical)
    pg = law_channel(N; rheology=classical, gnf_eps=LAW_EPS)
    pc = law_channel(N; viscosity=FVLawViscosity(adapter, law_init(adapter), (); eps=LAW_EPS))
    sg, Ug, tg = law_gpu_solve(pg)
    sc, Uc, tc = law_gpu_solve(pc)
    history = NamedTuple[]
    @testset "GPU classical adapter reproduces FVGNF Newton" begin
        @test tg.value.converged && tc.value.converged
        @test length(tg.value.history) == length(tc.value.history)
        @test norm(Array(Ug)-Array(Uc), Inf) <= 1e-12 * max(norm(Array(Ug), Inf), eps())
        for k in eachindex(tg.value.history)
            a, b = tg.value.history[k], tc.value.history[k]
            # Frozen arithmetic parity gate; absolute floor covers final residuals.
            @test isapprox(a, b; rtol=1e-13, atol=1e-14)
            push!(history, (; iteration=k-1, gnf=a, classical=b, difference=b-a))
        end
    end
    law_write_csv(joinpath(output, "classical_history.csv"), history)
    # Release both large factor caches before preparing the learned system.
    sg = sc = Ug = Uc = nothing
    GC.gc(); CUDA.reclaim()

    temperature = [0.2*(i-0.5)/N + 0.1*(j-0.5)/N for i in 1:N, j in 1:N]
    law = LearnedViscosity(TinyMLP{2,2,1}(); nu_ref=0.3, monotone=true)
    theta = law_init(law; rng=MersenneTwister(33))
    p = law_channel(N; viscosity=FVLawViscosity(law, theta, (temperature,); eps=LAW_EPS))
    s, U, primal = law_gpu_solve(p)
    law_write_csv(joinpath(output, "learned_history.csv"),
        [(; iteration=k-1, residual=r) for (k,r) in enumerate(primal.value.history)])
    map = FVLawParameters(length(theta))
    ws = fv_adjoint_workspace(s, map; chunk=4)
    obj = fv_flow_rate(p, max(1,N÷2))
    first_gradient = @timed begin
        r = fv_steady_gradient(s, U, theta, obj; params=map, ws)
        CUDA.synchronize()
        r
    end
    result = first_gradient.value
    @test result.residual_inf <= 1e-10
    @test all(isfinite, result.grad)
    @test s.n_factorize == 1

    fd = NamedTuple[]
    sfd = fv_newton_solver(p; to=CUDA.CuArray, linsolve=CUDABackendTag())
    rng = MersenneTwister(41)
    @testset "GPU full steady gradient: two random directions" begin
        for direction in 1:2
            d = randn(rng, length(theta)); d ./= norm(d)
            adjoint = dot(result.grad, d)
            for step in (1e-4, 2e-5)
                plus = law_gpu_fd_objective!(sfd, U, map, theta .+ step.*d, obj)
                minus = law_gpu_fd_objective!(sfd, U, map, theta .- step.*d, obj)
                central = (plus-minus)/(2step)
                error = abs(central-adjoint)
                # Require a resolved derivative, then the requested relative gate.
                scale = max(abs(central), abs(adjoint))
                @test scale > 1e-10
                relative_error = error/max(scale, eps())
                @test relative_error <= 1e-6
                push!(fd, (; direction, step, adjoint, central, absolute_error=error, relative_error))
            end
        end
    end
    law_write_csv(joinpath(output, "gradient_directions.csv"), fd)
    sfd = nothing; GC.gc(); CUDA.reclaim()

    # Allocation measurements distinguish host bytes from device bytes.
    gradient() = begin
        fv_steady_gradient(s, U, theta, obj; params=map, ws, finalize=false)
        CUDA.synchronize()
    end
    gradient()
    warm = [@timed gradient() for _ in 1:3]
    device_bytes = law_device_allocated(gradient)
    @test device_bytes == 0
    U .+= 1e-6
    fv_newton!(s,U; LAW_NEWTON...).converged || error("Newton warm-up failed")
    U .+= 1e-6
    warm_newton = @timed begin
        st = fv_newton!(s,U; LAW_NEWTON...)
        CUDA.synchronize()
        st
    end
    @test warm_newton.value.converged && warm_newton.value.iterations > 0
    U .+= 1e-6
    receipt = Ref{Any}()
    newton_device_bytes = law_device_allocated(() -> begin
        receipt[] = fv_newton!(s,U; LAW_NEWTON...)
        CUDA.synchronize()
    end)
    @test receipt[].converged && receipt[].iterations > 0
    @test newton_device_bytes == 0
    records = [(; N, precision="Float64", parameters=length(theta), chunk=4,
        gnf_first_newton_s=tg.time, classical_first_newton_s=tc.time,
        learned_first_newton_s=primal.time, learned_first_newton_host_bytes=primal.bytes,
        learned_warm_newton_s=warm_newton.time, learned_warm_newton_host_bytes=warm_newton.bytes,
        learned_warm_newton_device_bytes=newton_device_bytes,
        first_gradient_s=first_gradient.time, warm_gradient_s=median(r.time for r in warm),
        warm_gradient_host_bytes=minimum(r.bytes for r in warm),
        warm_gradient_device_bytes=device_bytes, residual_inf=result.residual_inf)]
    law_write_csv(joinpath(output, "law_gpu_timings.csv"), records)
    show(stdout, MIME("text/plain"), only(records)); println()
    println("DONE status=0")
end

if LAW_GPU_READY
    law_gpu_main()
elseif LAW_REQUIRE_GPU
    error("KRAKEN_REQUIRE_GPU=1 but CUDA/cuDSS is unavailable or CUDA.functional() is false")
else
    @info "SKIPPED law GPU benchmark; no CPU simulation was started"
end
