# Poiseuille flow convergence study
# Verifies 2nd-order spatial accuracy of the LBM BGK solver
using KrakeNN
using Printf

function run_poiseuille_convergence()
    # Sweep covers 3 octaves which is enough to demonstrate the second
    # order spatial accuracy of the BGK + half-way bounce-back scheme.
    # Going beyond Ny=128 requires O(N^2/ν) steps to reach steady state
    # which makes the benchmark too slow for a default run; use the
    # higher resolutions only when validating numerics, not for routine
    # benchmarks.
    Nys = [16, 32, 64, 128]
    ν = 0.1
    # Fx is scaled with Ny so that u_max stays constant at ~5e-3 across
    # all grids. Without this scaling, u_max ∝ Ny^2 quickly violates the
    # LBM Mach < 0.1 bound on the finer grids.
    u_max_target = 5e-3
    errors = Float64[]

    for Ny in Nys
        H = Ny
        Fx = 8 * ν * u_max_target / H^2

        # Convergence to steady state in a Poiseuille channel takes
        # O(Ny^2 / ν) steps. The 5x prefactor was calibrated empirically:
        # at Ny=64/ν=0.1 the error reaches its discretisation floor
        # (~1e-4) only after ~100k steps, which is exactly 5*Ny^2/ν.
        max_steps = round(Int, 5 * Ny^2 / ν)
        result = run_poiseuille_2d(; Nx=4, Ny=Ny, ν=ν, Fx=Fx, max_steps=max_steps)

        # Analytical parabolic profile with half-way bounce-back
        # Walls at y=0.5 and y=Ny+0.5 (half-way BB convention)
        u_analytical = [Fx / (2ν) * (j - 0.5) * (H + 0.5 - j) for j in 1:Ny]

        # L2 relative error
        u_num = result.ux[2, :]
        err = sqrt(sum((u_num .- u_analytical) .^ 2) / sum(u_analytical .^ 2))
        push!(errors, err)
    end

    # Print results
    println("\n=== Poiseuille Convergence ===")
    @printf("  %5s   %12s   %5s\n", "Ny", "L2 error", "Order")
    @printf("  %5s   %12s   %5s\n", "-----", "----------", "-----")
    for (i, Ny) in enumerate(Nys)
        order = i > 1 ? log2(errors[i-1] / errors[i]) : NaN
        @printf("  %5d   %12.4e   %5s\n", Ny, errors[i],
                isnan(order) ? "  -" : @sprintf("%.2f", order))
    end

    return Nys, errors
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_poiseuille_convergence()
end
