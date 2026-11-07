#!/usr/bin/env julia
# Compare KrakeNN CIJ jet interfaces with Basilisk reference data
# Usage: julia --project compare_cij.jl [result_dir]

using KrakeNN
using Serialization
using Printf

const BASILISK_DATA = "/Users/guillaume/Documents/Recherche/Rheodrop/data/numerical/ds_num"

function load_kraken_result(result_dir)
    result_file = joinpath(result_dir, "result.jls")
    if isfile(result_file)
        return deserialize(result_file)
    end
    error("No result.jls found in $result_dir. Run the simulation first.")
end

function extract_breakup_length(contour; R0=1.0)
    # Breakup length = z position where jet first pinches off
    # Look for gaps in the contour (large jumps in z)
    z_vals = first.(contour)
    for i in 2:length(z_vals)
        if z_vals[i] - z_vals[i-1] > 2.0 * R0
            return z_vals[i-1]
        end
    end
    return z_vals[end]  # No breakup → return jet tip position
end

function compare_single_case(result_dir)
    result = load_kraken_result(result_dir)
    p = result.params
    R0 = p.R0
    u_lb = p.u_lb

    println("=" ^ 60)
    @printf("CIJ Jet Comparison: Re=%d, We=%d, δ=%.3f\n", p.Re, p.We, p.δ)
    @printf("  LBM: R0=%d, u_lb=%.3f, ν_l=%.5f, σ=%.2e\n", R0, u_lb, p.ν_l, p.σ_lb)
    @printf("  Domain: %d × %d, T_period=%.0f steps\n", p.Nz, p.Nr, p.T_period)
    println("  Breakup: ", result.breakup_detected ? "YES (step $(result.breakup_step))" : "NO")

    # Get last KrakeNN snapshot
    last_step = maximum(keys(result.interfaces))
    kraken_pts = result.interfaces[last_step]
    t_phys_krak = last_step * u_lb / R0

    println("\n  KrakeNN snapshot: step=$last_step, t_phys=$(@sprintf("%.2f", t_phys_krak))")
    println("  Interface points: $(length(kraken_pts))")

    if isempty(kraken_pts)
        println("  WARNING: No interface points — simulation may have diffused or diverged")
        return
    end

    # Normalize KrakeNN coordinates to physical units (R₀=1)
    z_krak = [pt[1] / R0 for pt in kraken_pts]
    r_krak = [pt[2] / R0 for pt in kraken_pts]

    @printf("  KrakeNN jet: z ∈ [%.1f, %.1f] R₀, r ∈ [%.3f, %.3f] R₀\n",
            minimum(z_krak), maximum(z_krak), minimum(r_krak), maximum(r_krak))

    # Find closest Basilisk snapshot
    bas_file = find_basilisk_snapshot(BASILISK_DATA, p.Re, p.δ, t_phys_krak)
    if bas_file === nothing
        println("  WARNING: No Basilisk reference found for t=$(t_phys_krak)")
        # Try to find any snapshot around breakup time
        for t_try in [150.0, 155.0, 152.0, 158.0]
            bas_file = find_basilisk_snapshot(BASILISK_DATA, p.Re, p.δ, t_try; tol=1.0)
            bas_file !== nothing && break
        end
    end

    if bas_file !== nothing
        println("  Basilisk file: ", basename(bas_file))
        bas_contour = load_basilisk_interface_contour(bas_file)
        z_bas = first.(bas_contour)
        r_bas = last.(bas_contour)
        @printf("  Basilisk jet: z ∈ [%.1f, %.1f] R₀, r ∈ [%.3f, %.3f] R₀\n",
                minimum(z_bas), maximum(z_bas), minimum(r_bas), maximum(r_bas))

        # Compare breakup lengths
        L_krak = maximum(z_krak)
        L_bas = extract_breakup_length(bas_contour)
        @printf("  Jet length: KrakeNN=%.1f R₀, Basilisk=%.1f R₀ (ratio=%.2f)\n",
                L_krak, L_bas, L_krak / L_bas)
    end

    # Write comparison data for gnuplot/matplotlib
    open(joinpath(result_dir, "interface_kraken.dat"), "w") do io
        println(io, "# z/R0  r/R0")
        for (z, r) in sort(collect(zip(z_krak, r_krak)), by=first)
            @printf(io, "%.6f %.6f\n", z, r)
        end
    end
    println("  Wrote interface_kraken.dat")

    if bas_file !== nothing
        open(joinpath(result_dir, "interface_basilisk.dat"), "w") do io
            println(io, "# z/R0  r/R0")
            bas_contour = load_basilisk_interface_contour(bas_file)
            for (z, r) in bas_contour
                @printf(io, "%.6f %.6f\n", z, r)
            end
        end
        println("  Wrote interface_basilisk.dat")
    end

    println("\n  To plot: gnuplot -e \"plot 'interface_kraken.dat' w p, 'interface_basilisk.dat' w l\"")
end

# Main
result_dir = length(ARGS) >= 1 ? ARGS[1] : "validation/cij_Re200"
compare_single_case(result_dir)
