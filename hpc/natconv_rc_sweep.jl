#!/usr/bin/env julia
"""
Parametric study: Rc sweep for natural convection cavity.
Compares KrakeNN LBM vs NatConv OpenFOAM results.

Pr=0.71, Ra ∈ {1e3, 1e4}, Rc ∈ {1, 2, 5, 10, 20, 50, 100}, N=512.
Requires CUDA — will error if GPU not available.
"""

using KrakeNN
using KernelAbstractions
using CUDA
using Printf
using DelimitedFiles

# Fail fast if CUDA not working
@assert CUDA.functional() "CUDA not functional — aborting"
println("  GPU: ", CUDA.name(CUDA.device()))

function nsteps(N, Ra)
    base_128 = Ra <= 1e3 ? 40_000 : (Ra <= 1e4 ? 80_000 : 160_000)
    return round(Int, base_128 * (N / 128)^2)
end

function load_natconv(path)
    if !isfile(path)
        @warn "NatConv results not found: $path"
        return nothing
    end
    data, header = readdlm(path, ','; header=true)
    cols = Dict(String(h) => i for (i, h) in enumerate(vec(header)))
    results = Dict{Tuple{Float64,Float64,Float64}, Float64}()
    for row in eachrow(data)
        converged = String(row[cols["converged"]]) == "True"
        converged || continue
        pr = Float64(row[cols["Pr"]])
        ra = Float64(row[cols["Ra"]])
        rc = Float64(row[cols["Rc"]])
        nu = Float64(row[cols["Nu"]])
        results[(pr, ra, rc)] = nu
    end
    return results
end

function run_rc_sweep()
    println("\n  KrakeNN — Rc Sweep: Natural Convection (Pr=0.71)\n")

    backend = CUDABackend()
    FT = Float64

    N = 512
    Pr = 0.71
    Ra_values = [1e3, 1e4]
    Rc_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    natconv = load_natconv(joinpath(homedir(),
        "NatConv/analysis_v2/newtonian_v2_results.csv"))

    results = []

    for Ra in Ra_values
        steps = nsteps(N, Ra)
        println("=" ^ 80)
        @printf("  Ra = %.0e, N = %d, %d steps\n", Ra, N, steps)
        println("=" ^ 80)
        @printf("  %-6s  %-10s  %-10s  %-8s  %-10s\n",
                "Rc", "Nu (LBM)", "Nu (OF)", "Err %", "Time (s)")
        println("  " * "-" ^ 50)

        for Rc in Rc_values
            t = @elapsed begin
                r = run_natural_convection_2d(; N=N, Ra=Ra, Pr=Pr, Rc=Rc,
                                                max_steps=steps,
                                                backend=backend, FT=FT)
            end

            Nu_of = nothing
            if natconv !== nothing
                key = (Pr, Ra, Rc)
                Nu_of = get(natconv, key, nothing)
            end

            err_str = "—"
            Nu_of_str = "—"
            if Nu_of !== nothing
                err = 100 * abs(r.Nu - Nu_of) / Nu_of
                err_str = @sprintf("%.2f", err)
                Nu_of_str = @sprintf("%.4f", Nu_of)
            end

            @printf("  %-6.0f  %-10.4f  %-10s  %-8s  %-10.1f\n",
                    Rc, r.Nu, Nu_of_str, err_str, t)

            push!(results, (Ra=Ra, Rc=Rc, Nu_lbm=r.Nu, Nu_of=Nu_of,
                           time=t, N=N, steps=steps))
        end
        println()
    end

    # Save CSV
    open("natconv_rc_sweep.csv", "w") do io
        println(io, "Ra,Rc,Pr,N,steps,Nu_lbm,Nu_of,err_pct,time_s")
        for r in results
            err = r.Nu_of !== nothing ? 100*abs(r.Nu_lbm - r.Nu_of)/r.Nu_of : NaN
            Nu_of_str = r.Nu_of !== nothing ? @sprintf("%.6f", r.Nu_of) : ""
            @printf(io, "%.0e,%.1f,%.2f,%d,%d,%.6f,%s,%.4f,%.1f\n",
                    r.Ra, r.Rc, Pr, r.N, r.steps, r.Nu_lbm, Nu_of_str, err, r.time)
        end
    end
    println("  Results saved to natconv_rc_sweep.csv")
end

run_rc_sweep()
