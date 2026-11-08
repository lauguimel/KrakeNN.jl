# Taylor-Green vortex decay — convergence study
# Verifies 2nd-order spatial accuracy via comparison with analytical decay
using KrakeNN
using Printf

function run_taylor_green_convergence()
    Ns = [16, 32, 64, 128]
    ν = 0.01
    u0 = 0.01
    max_steps = 1000
    errors = Float64[]

    for N in Ns
        result = run_taylor_green_2d(; N=N, ν=ν, u0=u0, max_steps=max_steps)

        # Analytical solution: exponential decay
        k = 2π / N
        decay = exp(-2 * ν * k^2 * max_steps)

        # Analytical ux field at final time.
        # Coordinates follow the driver: x = i - 1, y = j - 1 (0-based,
        # consistent with initialize_taylor_green_2d).
        ux_analytical = zeros(N, N)
        for j in 1:N, i in 1:N
            x = i - 1
            y = j - 1
            ux_analytical[i, j] = -u0 * cos(k * x) * sin(k * y) * decay
        end

        # L2 relative error
        diff = result.ux .- ux_analytical
        err = sqrt(sum(diff .^ 2) / sum(ux_analytical .^ 2))
        push!(errors, err)
    end

    println("\n=== Taylor-Green Convergence ===")
    @printf("  %5s   %12s   %5s\n", "N", "L2 error", "Order")
    @printf("  %5s   %12s   %5s\n", "-----", "----------", "-----")
    for (i, N) in enumerate(Ns)
        order = i > 1 ? log2(errors[i-1] / errors[i]) : NaN
        @printf("  %5d   %12.4e   %5s\n", N, errors[i],
                isnan(order) ? "  -" : @sprintf("%.2f", order))
    end

    return Ns, errors
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_taylor_green_convergence()
end
