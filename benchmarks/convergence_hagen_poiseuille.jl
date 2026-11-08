# Hagen-Poiseuille (axisymmetric) convergence study
# Verifies 2nd-order spatial accuracy of the axisymmetric LBM solver
using KrakeNN
using Printf

function run_hagen_poiseuille_convergence()
    Nrs = [8, 16, 32, 64, 128]
    Nz = 4
    ν = 0.1
    u_max_target = 5e-3
    errors = Float64[]

    for Nr in Nrs
        R = Nr
        # Scale Fz to keep u_max constant: u_max = Fz*R^2/(4*ν)
        Fz = 4 * ν * u_max_target / R^2
        max_steps = round(Int, 5 * Nr^2 / ν)

        result = run_hagen_poiseuille_2d(; Nz=Nz, Nr=Nr, ν=ν, Fz=Fz, max_steps=max_steps)

        # Analytical profile: u_z(r) = u_max * (1 - (r/R)^2)
        # with r = j - 0.5 (half-way BB), R = Nr
        u_analytical = [Fz * R^2 / (4ν) * (1 - ((j - 0.5) / R)^2) for j in 1:Nr]

        # L2 relative error
        u_num = result.uz[2, :]
        err = sqrt(sum((u_num .- u_analytical) .^ 2) / sum(u_analytical .^ 2))
        push!(errors, err)
    end

    # Print results
    println("\n=== Hagen-Poiseuille Convergence ===")
    @printf("  %5s   %12s   %5s\n", "Nr", "L2 error", "Order")
    @printf("  %5s   %12s   %5s\n", "-----", "----------", "-----")
    for (i, Nr) in enumerate(Nrs)
        order = i > 1 ? log2(errors[i-1] / errors[i]) : NaN
        @printf("  %5d   %12.4e   %5s\n", Nr, errors[i],
                isnan(order) ? "  -" : @sprintf("%.2f", order))
    end

    return Nrs, errors
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_hagen_poiseuille_convergence()
end
