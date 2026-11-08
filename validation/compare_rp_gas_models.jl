# Compare the three gas models (smooth, ghost, phasefield) on the
# Rayleigh-Plateau instability and validate against linear theory.
#
# The inviscid dispersion relation (Rayleigh 1878):
#   ω² = (σ / ρR₀³) · x · I₁(x)/I₀(x) · (1 - x²)
# where x = kR₀ = 2π/λ_ratio.
#
# We measure the growth rate from r_min(t) in the linear regime and
# compare with theory across several wavelengths.
#
# Usage:
#   julia --gcthreads=1 --project=. validation/compare_rp_gas_models.jl

using KrakeNN
using Printf

# --- Modified Bessel functions I₀, I₁ (polynomial approximation, Abramowitz & Stegun) ---

function besseli0(x::Float64)
    # Abramowitz & Stegun 9.8.1/9.8.2, |error| < 1.6e-7 for all x
    ax = abs(x)
    if ax < 3.75
        t = (x / 3.75)^2
        return 1.0 + t*(3.5156229 + t*(3.0899424 + t*(1.2067492 +
               t*(0.2659732 + t*(0.0360768 + t*0.0045813)))))
    else
        t = 3.75 / ax
        return (exp(ax) / sqrt(ax)) * (0.39894228 + t*(0.01328592 +
               t*(0.00225319 + t*(-0.00157565 + t*(0.00916281 +
               t*(-0.02057706 + t*(0.02635537 + t*(-0.01647633 +
               t*0.00392377))))))))
    end
end

function besseli1(x::Float64)
    # Abramowitz & Stegun 9.8.3/9.8.4
    ax = abs(x)
    if ax < 3.75
        t = (x / 3.75)^2
        val = ax * (0.5 + t*(0.87890594 + t*(0.51498869 + t*(0.15084934 +
              t*(0.02658733 + t*(0.00301532 + t*0.00032411))))))
        return x < 0 ? -val : val
    else
        t = 3.75 / ax
        val = (exp(ax) / sqrt(ax)) * (0.39894228 + t*(-0.03988024 +
              t*(-0.00362018 + t*(0.00163801 + t*(-0.01031555 +
              t*(0.02282967 + t*(-0.02895312 + t*(0.01787654 +
              t*(-0.00420059)))))))))
        return x < 0 ? -val : val
    end
end

# --- Rayleigh dispersion relation (inviscid) ---

"""
    rayleigh_growth_rate(kR0, σ, ρ, R0)

Inviscid growth rate ω for the Rayleigh-Plateau instability.
Returns ω (real) if the mode is unstable (kR₀ < 1), 0.0 otherwise.
"""
function rayleigh_growth_rate(kR0, σ, ρ, R0)
    if kR0 >= 1.0
        return 0.0  # stable mode
    end
    x = kR0
    ratio = besseli1(x) / besseli0(x)
    ω2 = (σ / (ρ * R0^3)) * x * ratio * (1.0 - x^2)
    return sqrt(max(0.0, ω2))
end

"""
    measure_growth_rate(r_min, times, R0, ε)

Extract growth rate from r_min(t) in the linear regime.
In the linear regime: r_min(t) ≈ R0 · (1 - ε·exp(ω·t))
So: ln(1 - r_min/R0) ≈ ln(ε) + ω·t
We fit the slope in the region where the perturbation is still small.
"""
function measure_growth_rate(r_min, times, R0, ε)
    # Compute amplitude: A(t) = 1 - r_min(t)/R0
    amplitudes = [1.0 - r / R0 for r in r_min]

    # Select linear regime: amplitude between ε and 5ε (well before nonlinear)
    valid = findall(a -> a > 0.5 * ε && a < min(10 * ε, 0.3), amplitudes)

    if length(valid) < 2
        @warn "Not enough data points in linear regime" n_valid=length(valid) amplitudes
        return NaN
    end

    # Linear fit of ln(amplitude) vs time
    log_amp = log.(amplitudes[valid])
    t = Float64.(times[valid])

    # Least squares: ln(A) = a + ω·t
    n = length(t)
    t_mean = sum(t) / n
    la_mean = sum(log_amp) / n
    num = sum((t .- t_mean) .* (log_amp .- la_mean))
    den = sum((t .- t_mean).^2)
    ω_measured = num / den

    return ω_measured
end

# --- Parameters ---

R0 = 20
ε = 0.05
σ = 0.01
ν = 0.05
ρ = 1.0  # ρ_l = ρ_g = 1 for clean comparison

# Wavelengths to scan: λ_ratio from 4.0 to 12.0
# (kR0 = 2π/λ_ratio: unstable for kR0 < 1, i.e. λ_ratio > 2π ≈ 6.28)
# We include some stable modes to verify decay
λ_ratios = [4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Simulation parameters
Nz = 256
Nr = 60
max_steps = 15000
output_interval = 200

gas_models = [:smooth, :ghost, :phasefield]

println("=" ^ 80)
println("Rayleigh-Plateau dispersion relation — 3 gas models vs theory")
println("=" ^ 80)
println()
@printf("R0=%d, ε=%.2f, σ=%.3f, ν=%.3f, ρ=%.1f\n", R0, ε, σ, ν, ρ)
@printf("Grid: %d × %d, max_steps=%d\n\n", Nz, Nr, max_steps)

# --- Theoretical dispersion curve ---
println("Theoretical growth rates (Rayleigh inviscid):")
println("-" ^ 50)
for λ_r in λ_ratios
    kR0 = 2π / λ_r
    ω_th = rayleigh_growth_rate(kR0, σ, ρ, R0)
    @printf("  λ/R₀ = %5.1f  →  kR₀ = %.3f  →  ω_th = %.6f", λ_r, kR0, ω_th)
    if kR0 >= 1.0
        println("  (STABLE)")
    else
        println()
    end
end
println()

# --- Run simulations ---

results = Dict{Symbol, Dict{Float64, NamedTuple}}()

for model in gas_models
    results[model] = Dict{Float64, NamedTuple}()

    println("=" ^ 60)
    println("Gas model: $model")
    println("=" ^ 60)

    for λ_r in λ_ratios
        @printf("  λ/R₀ = %.1f ... ", λ_r)
        flush(stdout)

        kwargs = (
            Nz=Nz, Nr=Nr, R0=R0, λ_ratio=λ_r, ε=ε,
            σ=σ, ν=ν, ρ_l=ρ, ρ_g=ρ,
            gas_model=model,
            max_steps=max_steps, output_interval=output_interval,
        )

        # Model-specific params
        if model == :smooth
            kwargs = (; kwargs..., n_smooth=3)
        elseif model == :ghost
            kwargs = (; kwargs..., n_ghost_layers=3)
        elseif model == :phasefield
            kwargs = (; kwargs..., W_pf=4.0, τ_g=0.7)
        end

        result = run_rp_hybrid_2d(; kwargs...)

        ω_meas = measure_growth_rate(result.r_min, result.times, Float64(R0), ε)
        kR0 = 2π / λ_r
        ω_th = rayleigh_growth_rate(kR0, σ, ρ, Float64(R0))

        if isnan(ω_meas)
            @printf("ω = NaN (not enough linear data)\n")
        else
            rel_err = ω_th > 0 ? abs(ω_meas - ω_th) / ω_th * 100 : NaN
            @printf("ω_meas = %.6f  ω_th = %.6f  err = %.1f%%\n", ω_meas, ω_th, rel_err)
        end

        results[model][λ_r] = (
            r_min=result.r_min, times=result.times,
            ω_meas=ω_meas, ω_th=ω_th, kR0=kR0
        )
    end
    println()
end

# --- Summary table ---

println()
println("=" ^ 90)
println("SUMMARY: Growth rate comparison (ω)")
println("=" ^ 90)
@printf("%-8s  %-6s  %-10s", "λ/R₀", "kR₀", "ω_theory")
for model in gas_models
    @printf("  %-12s", "ω_$model")
end
println()
println("-" ^ 90)

for λ_r in λ_ratios
    kR0 = 2π / λ_r
    ω_th = rayleigh_growth_rate(kR0, σ, ρ, Float64(R0))
    @printf("%-8.1f  %-6.3f  %-10.6f", λ_r, kR0, ω_th)
    for model in gas_models
        ω_m = results[model][λ_r].ω_meas
        if isnan(ω_m)
            @printf("  %-12s", "NaN")
        else
            @printf("  %-12.6f", ω_m)
        end
    end
    println()
end
println("-" ^ 90)

# --- r_min(t) data output for plotting ---

output_dir = joinpath(@__DIR__, "..", "results", "rp_comparison")
mkpath(output_dir)

# Write r_min(t) for each (model, λ_ratio) pair
for model in gas_models
    for λ_r in λ_ratios
        r = results[model][λ_r]
        fname = joinpath(output_dir, "rmin_$(model)_lambda$(λ_r).dat")
        open(fname, "w") do io
            println(io, "# t  r_min/R0  (model=$model, λ/R₀=$λ_r)")
            for (t, rm) in zip(r.times, r.r_min)
                @printf(io, "%d  %.6f\n", t, rm / R0)
            end
        end
    end
end

# Write dispersion data
fname = joinpath(output_dir, "dispersion.dat")
open(fname, "w") do io
    print(io, "# kR0  omega_theory")
    for model in gas_models
        print(io, "  omega_$model")
    end
    println(io)
    for λ_r in λ_ratios
        kR0 = 2π / λ_r
        ω_th = rayleigh_growth_rate(kR0, σ, ρ, Float64(R0))
        @printf(io, "%.4f  %.6f", kR0, ω_th)
        for model in gas_models
            ω_m = results[model][λ_r].ω_meas
            @printf(io, "  %.6f", isnan(ω_m) ? 0.0 : ω_m)
        end
        println(io)
    end
end

# Write theoretical curve (fine resolution)
fname_th = joinpath(output_dir, "dispersion_theory.dat")
open(fname_th, "w") do io
    println(io, "# kR0  omega_theory")
    for kR0 in range(0.01, 0.99, length=100)
        ω_th = rayleigh_growth_rate(kR0, σ, ρ, Float64(R0))
        @printf(io, "%.4f  %.6f\n", kR0, ω_th)
    end
end

println()
println("Data written to: $output_dir/")
println("  - rmin_<model>_lambda<λ>.dat  (r_min/R₀ vs t)")
println("  - dispersion.dat              (measured ω vs kR₀)")
println("  - dispersion_theory.dat       (Rayleigh curve, fine)")
println()
println("Done.")
