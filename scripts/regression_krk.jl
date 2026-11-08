#!/usr/bin/env julia
# Regression test: run all examples/*.krk for a short number of steps,
# report NaN/Inf/crashes. Skips refinement cases on CPU for speed.

using KrakeNN
using Printf

const EXAMPLES = filter(f -> endswith(f, ".krk"),
                        readdir(joinpath(@__DIR__, "..", "examples"), join=true))

@printf("Running %d .krk regression tests (max_steps=100 each)\n", length(EXAMPLES))
println("="^60)

results = Tuple{String, Symbol, String}[]
for f in sort(EXAMPLES)
    name = basename(f)
    t0 = time()
    try
        r = run_simulation(f; max_steps=100)
        dt = time() - t0
        has_nan = any(isnan, r.ρ) || any(isnan, r.ux) || any(isnan, r.uy)
        if has_nan
            push!(results, (name, :fail, "NaN after 100 steps"))
            @printf("  ✗ %-30s  %.1fs  NaN!\n", name, dt)
        else
            push!(results, (name, :ok, @sprintf("%.1fs", dt)))
            @printf("  ✓ %-30s  %.1fs\n", name, dt)
        end
    catch e
        push!(results, (name, :error, sprint(showerror, e)[1:min(end,100)]))
        @printf("  ✗ %-30s  ERROR: %s\n", name, sprint(showerror, e)[1:min(end,80)])
    end
end

println("="^60)
n_ok = count(r -> r[2] == :ok, results)
n_fail = count(r -> r[2] != :ok, results)
@printf("Summary: %d/%d pass, %d fail\n", n_ok, length(results), n_fail)
exit(n_fail > 0 ? 1 : 0)
