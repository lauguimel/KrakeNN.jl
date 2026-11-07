#!/usr/bin/env julia
"""
Natural convection in a differentially heated cavity — parametric study.

Compares KrakeNN LBM results against:
  - De Vahl Davis (1983) benchmark (Rc=1)
  - OpenFOAM/RheoTool results from NatConv project

Usage:
    julia --project examples/natural_convection_study.jl
"""

using KrakeNN
using KernelAbstractions
using Printf
using DelimitedFiles

const HAS_METAL = try
    @eval using Metal
    true
catch
    false
end

# ── Reference data ───────────────────────────────────────────────────────

const DE_VAHL_DAVIS = Dict(
    1e3 => 1.118,
    1e4 => 2.243,
    1e5 => 4.519,
    1e6 => 8.800,
)

"""Load NatConv OpenFOAM results from CSV."""
function load_natconv_results(path::String)
    if !isfile(path)
        @warn "NatConv results not found at $path"
        return nothing
    end
    data, header = readdlm(path, ','; header=true)
    cols = Dict(String(h) => i for (i, h) in enumerate(vec(header)))
    results = Dict{Tuple{Float64,Float64,Float64}, Float64}()
    for row in eachrow(data)
        pr = Float64(row[cols["Pr"]])
        ra = Float64(row[cols["Ra"]])
        rc = Float64(row[cols["Rc"]])
        nu = Float64(row[cols["Nu"]])
        converged = String(row[cols["converged"]]) == "True"
        converged && (results[(pr, ra, rc)] = nu)
    end
    return results
end

# ── Part 1: Validation against De Vahl Davis (Rc=1) ─────────────────────

function run_validation(; N=128, max_steps=80000)
    println("=" ^ 70)
    println("  PART 1: Validation vs De Vahl Davis (1983) — Rc = 1, Pr = 0.71")
    println("=" ^ 70)

    Ra_values = [1e3, 1e4, 1e5]

    @printf("  %-8s  %-10s  %-10s  %-8s\n", "Ra", "Nu (LBM)", "Nu (ref)", "Error %")
    println("  " * "-" ^ 42)

    results = Dict{Float64, Float64}()
    for Ra in Ra_values
        steps = Ra <= 1e3 ? max_steps ÷ 2 : max_steps
        r = run_natural_convection_2d(; N=N, Ra=Ra, Pr=0.71, Rc=1.0,
                                        max_steps=steps)
        Nu_ref = DE_VAHL_DAVIS[Ra]
        err = 100 * abs(r.Nu - Nu_ref) / Nu_ref
        @printf("  %-8.0e  %-10.4f  %-10.3f  %-8.2f\n", Ra, r.Nu, Nu_ref, err)
        results[Ra] = r.Nu
    end
    println()
    return results
end

# ── Part 2: CPU vs GPU benchmark ────────────────────────────────────────

function run_benchmark(; N=128, Ra=1e4, max_steps=20000)
    println("=" ^ 70)
    println("  PART 2: CPU vs GPU benchmark (Ra=$Ra, N=$N, $max_steps steps)")
    println("=" ^ 70)

    # CPU
    t_cpu = @elapsed begin
        r_cpu = run_natural_convection_2d(; N=N, Ra=Ra, Pr=0.71, Rc=1.0,
                                            max_steps=max_steps,
                                            backend=KernelAbstractions.CPU())
    end
    @printf("  CPU:   %.2f s  (Nu = %.4f)\n", t_cpu, r_cpu.Nu)

    # GPU (Metal on macOS)
    t_gpu = NaN
    r_gpu = nothing
    if HAS_METAL
        # Metal does not support Float64 — use Float32
        t_gpu = @elapsed begin
            r_gpu = run_natural_convection_2d(; N=N, Ra=Ra, Pr=0.71, Rc=1.0,
                                                max_steps=max_steps,
                                                backend=Metal.MetalBackend(),
                                                FT=Float32)
        end
        @printf("  Metal: %.2f s  (Nu = %.4f)\n", t_gpu, r_gpu.Nu)
        @printf("  Speedup: %.1fx\n", t_cpu / t_gpu)
    else
        println("  GPU benchmark skipped (Metal.jl not available).")
    end
    println()
    return (t_cpu=t_cpu, t_gpu=t_gpu)
end

# ── Part 3: Comparison with NatConv results ──────────────────────────────

function run_comparison(; N=128, max_steps=80000)
    println("=" ^ 70)
    println("  PART 3: Comparison with NatConv (OpenFOAM/RheoTool)")
    println("=" ^ 70)

    natconv_path = joinpath(homedir(),
        "Documents/Recherche/NatConv/analysis_v2/newtonian_v2_results.csv")
    natconv = load_natconv_results(natconv_path)

    if natconv === nothing
        println("  Skipped — NatConv results not found.\n")
        return nothing
    end

    # Run a subset: Pr=0.71, Rc=1, multiple Ra
    Pr = 0.71
    Rc = 1.0
    Ra_values = [1e3, 1e4, 1e5]

    @printf("  %-8s  %-10s  %-10s  %-8s\n", "Ra", "Nu (LBM)", "Nu (OF)", "Error %")
    println("  " * "-" ^ 42)

    for Ra in Ra_values
        key = (Pr, Ra, Rc)
        if !haskey(natconv, key)
            @printf("  %-8.0e  %-10s  %-10s  %-8s\n", Ra, "—", "missing", "—")
            continue
        end

        steps = Ra <= 1e3 ? max_steps ÷ 2 : max_steps
        r = run_natural_convection_2d(; N=N, Ra=Ra, Pr=Pr, Rc=Rc,
                                        max_steps=steps)
        Nu_of = natconv[key]
        err = 100 * abs(r.Nu - Nu_of) / Nu_of
        @printf("  %-8.0e  %-10.4f  %-10.4f  %-8.2f\n", Ra, r.Nu, Nu_of, err)
    end
    println()
end

# ── Main ─────────────────────────────────────────────────────────────────

function main()
    println("\n  KrakeNN — Natural Convection Parametric Study\n")

    run_validation(; N=128, max_steps=80000)
    run_benchmark(; N=128, Ra=1e4, max_steps=20000)
    run_comparison(; N=128, max_steps=80000)

    println("Done.")
end

main()
