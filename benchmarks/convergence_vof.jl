# VOF multiphase convergence study
# Laplace law validation: static droplet at different radii
# Measures pressure jump error and spurious currents
using KrakeNN
using Printf

function run_vof_convergence()
    N = 128
    sigma = 0.01
    nu = 0.1
    rho_l = 1.0
    rho_g = 0.001
    max_steps = 5000
    radii = [10, 15, 20, 30]

    println("\n=== Laplace Law Convergence (static droplet) ===")
    @printf("  %5s   %12s   %12s   %8s   %12s\n",
            "R", "dp_LBM", "dp_exact", "error%", "max|u|")
    @printf("  %5s   %12s   %12s   %8s   %12s\n",
            "-----", "----------", "----------", "------", "----------")

    for R in radii
        result = run_static_droplet_2d(; N=N, R=R, σ=sigma, ν=nu,
                                         ρ_l=rho_l, ρ_g=rho_g,
                                         max_steps=max_steps)

        dp_num = result.Δp
        dp_ana = result.Δp_analytical
        rel_err = abs(dp_num - dp_ana) / abs(dp_ana) * 100
        max_u = result.max_u_spurious

        @printf("  %5d   %12.6e   %12.6e   %7.2f%%   %12.4e\n",
                R, dp_num, dp_ana, rel_err, max_u)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_vof_convergence()
end
