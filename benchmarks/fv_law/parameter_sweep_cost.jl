# Isolated CPU cost of both -lambda' * dR/dtheta paths at fixed U and lambda.
# No Newton solve, Jacobian assembly, transposed solve, or full objective gradient.
using KrakeNN, Random, LinearAlgebra, Statistics, Dates

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
    rows = NamedTuple[]
    values = Vector{Float64}[]
    for path in (:contraction,:duals)
        operation() = KrakeNN._fv_param_cotangent!(ws,map,solver,U,theta;law_gradient=path)
        cold = @timed operation()
        reference = copy(cold.value)
        push!(values,reference)
        all(isfinite,reference) || error("Nonfinite parameter contraction")
        warm = [@timed operation() for _ in 1:samples]
        all(x -> x.value == reference,warm) || error("Non-reproducible parameter contraction")
        seconds = median(x.time for x in warm)
        bytes = minimum(x.bytes for x in warm)
        sweeps = path == :duals ? cld(length(theta),chunk) : 1
        push!(rows,(; measured_utc=Dates.format(now(UTC),dateformat"yyyy-mm-ddTHH:MM:SS"),
            julia=string(VERSION), machine=Sys.MACHINE, cpu=Sys.CPU_NAME,
            threads=Threads.nthreads(), backend="CPU", precision="Float64", N,
            inputs=2,width,hidden_layers=1,monotone=true,parameters=length(theta),path,chunk,sweeps,
            setup_seconds=setup.time,cold_contraction_seconds=cold.time,
            cold_compile_seconds=cold.compile_time,cold_recompile_seconds=cold.recompile_time,
            cold_contraction_host_bytes=cold.bytes,
            warm_contraction_seconds=seconds,warm_contraction_host_bytes=bytes,
            amortized_sweep_seconds=seconds/sweeps,amortized_sweep_host_bytes=bytes/sweeps,
            contraction_norm=norm(reference),samples))
    end
    relative_error = norm(values[1]-values[2])/max(norm(values[1]),norm(values[2]),eps())
    relative_error <= 1e-12 || error("Contraction/duals mismatch: $relative_error")
    return [merge(row,(;relative_error)) for row in rows]
end

function main()
    output = isempty(ARGS) ? joinpath(@__DIR__,"parameter_sweep_cost.csv") : only(ARGS)
    # Retain the original L2 receipt verbatim and append the L3 comparison.
    # Re-running this script replaces only L3 measurements, never history.
    historical = readlines(joinpath(@__DIR__,"parameter_sweep_cost.csv"))
    headers = split(first(historical),',')
    last_historical = findfirst(==("samples"),headers)
    oldkeys = Tuple(Symbol.(headers[1:last_historical]))
    oldrows = [NamedTuple{oldkeys}(Tuple(split(line,',')[1:last_historical]))
               for line in historical[2:end]]
    rows = NamedTuple[]
    for width in (1,8,35)
        measured = parameter_sweep_cost(width)
        c,d = measured
        previous = only(filter(row -> parse(Int,row.width) == width,oldrows))
        row = merge(previous,(;l3_measured_utc=c.measured_utc,l3_julia=c.julia,
            l3_machine=c.machine,l3_cpu=c.cpu,l3_threads=c.threads,
            l3_contraction_cold_seconds=c.cold_contraction_seconds,
            l3_contraction_compile_seconds=c.cold_compile_seconds,
            l3_contraction_warm_seconds=c.warm_contraction_seconds,
            l3_contraction_host_bytes=c.warm_contraction_host_bytes,
            l3_duals_cold_seconds=d.cold_contraction_seconds,
            l3_duals_compile_seconds=d.cold_compile_seconds,
            l3_duals_warm_seconds=d.warm_contraction_seconds,
            l3_duals_host_bytes=d.warm_contraction_host_bytes,
            l3_contraction_norm=c.contraction_norm,l3_relative_error=c.relative_error,
            l3_samples=c.samples))
        push!(rows,row)
        for row in measured
            println("parameters=",row.parameters," path=",row.path," chunk=",row.chunk,
                " sweeps=",row.sweeps," cold_s=",row.cold_contraction_seconds,
                " warm_s=",row.warm_contraction_seconds,
                " warm_host_bytes=",row.warm_contraction_host_bytes,
                " relative_error=",row.relative_error)
        end
        flush(stdout)
    end
    # Replace the receipt only when all cases completed, preserving all three
    # historical rows even if a later compilation or numerical gate fails.
    open(output,"w") do io
        println(io,join(string.(keys(first(rows))),","))
        foreach(r -> println(io,join(string.(values(r)),",")),rows)
    end
end

main()
