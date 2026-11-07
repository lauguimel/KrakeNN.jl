# Isolated CPU cost of the existing chunked -lambda' * dR/dtheta path.
# No Newton solve, Jacobian assembly, transposed solve, or full objective gradient.
using Kraken, Random, LinearAlgebra, Statistics, Dates

function parameter_sweep_cost(width; N=64, chunk=4, samples=3)
    law = LearnedViscosity(TinyMLP{2,width,1}(); nu_ref=0.3, monotone=true)
    theta = law_init(law; rng=MersenneTwister(33))
    field = [0.2*(i-0.5)/N + 0.1*(j-0.5)/N for i in 1:N, j in 1:N]
    prob = fv_steady_problem(N,N; h=1/N, conv=false,
        viscosity=FVLawViscosity(law,theta,(field,); eps=1e-2))
    # Building the assembler is setup only; neither assemble nor factor is called.
    setup = @timed fv_newton_solver(prob; chunk)
    solver = setup.value
    map = FVLawParameters(length(theta))
    ws = fv_adjoint_workspace(solver,map; chunk)
    U = 0.03randn(MersenneTwister(71),fv_nunknowns(prob))
    copyto!(ws.λ,randn(MersenneTwister(72),length(U)))
    operation() = Kraken._fv_param_cotangent!(ws,map,solver,U,theta)
    cold = @timed operation()
    all(isfinite,cold.value) || error("Nonfinite parameter contraction")
    warm = [@timed operation() for _ in 1:samples]
    all(x -> x.value == cold.value,warm) || error("Non-reproducible parameter contraction")
    seconds = median(x.time for x in warm)
    bytes = minimum(x.bytes for x in warm)
    sweeps = cld(length(theta),chunk)
    return (; measured_utc=Dates.format(now(UTC),dateformat"yyyy-mm-ddTHH:MM:SS"),
        julia=string(VERSION), machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
        threads=Threads.nthreads(), backend="CPU", precision="Float64", N,
        inputs=2,width,hidden_layers=1,monotone=true,parameters=length(theta),chunk,sweeps,
        setup_seconds=setup.time,cold_contraction_seconds=cold.time,
        cold_compile_seconds=cold.compile_time,cold_recompile_seconds=cold.recompile_time,
        cold_contraction_host_bytes=cold.bytes,
        warm_contraction_seconds=seconds,warm_contraction_host_bytes=bytes,
        amortized_sweep_seconds=seconds/sweeps,amortized_sweep_host_bytes=bytes/sweeps,
        contraction_norm=norm(cold.value),samples)
end

function main()
    output = isempty(ARGS) ? joinpath(@__DIR__,"parameter_sweep_cost.csv") : only(ARGS)
    rows = NamedTuple[]
    for width in (1,8,35)
        row = parameter_sweep_cost(width)
        push!(rows,row)
        println("parameters=",row.parameters," chunk=",row.chunk," sweeps=",row.sweeps,
            " cold_s=",row.cold_contraction_seconds," warm_s=",row.warm_contraction_seconds,
            " warm_host_bytes=",row.warm_contraction_host_bytes)
        flush(stdout)
        # Preserve completed measurements if a later compilation is interrupted.
        open(output,"w") do io
            println(io,join(string.(keys(first(rows))),","))
            foreach(r -> println(io,join(string.(values(r)),",")),rows)
        end
    end
end

main()
