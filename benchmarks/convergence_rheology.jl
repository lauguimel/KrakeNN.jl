# Non-Newtonian rheology convergence study
# Power-law Poiseuille flow for different power-law indices
using KrakeNN
using Printf

function run_rheology_convergence()
    Nx, Ny = 4, 32
    K = 0.1
    Fx_val = 1e-4
    max_steps = 50000
    n_values = [0.5, 0.7, 1.0, 1.5]

    println("\n=== Power-Law Poiseuille Convergence ===")
    @printf("  %5s   %12s   %12s   %12s\n", "n", "u_max_num", "u_max_ana", "L_inf err")
    @printf("  %5s   %12s   %12s   %12s\n", "-----", "---------", "---------", "---------")

    for n_pl in n_values
        # Analytical solution for power-law Poiseuille (half-way BB)
        H = Float64(Ny)
        u_analytical = zeros(Ny)
        for j in 1:Ny
            y = j - 0.5
            dist = abs(y - H / 2)
            u_analytical[j] = n_pl / (n_pl + 1) * (Fx_val / K)^(1 / n_pl) *
                              ((H / 2)^((n_pl + 1) / n_pl) - dist^((n_pl + 1) / n_pl))
        end
        u_max_ana = maximum(u_analytical)

        # Initialize LBM arrays
        f_in  = zeros(Float64, Nx, Ny, 9)
        f_out = zeros(Float64, Nx, Ny, 9)
        is_solid = falses(Nx, Ny)

        # Estimate initial viscosity
        gamma_est = u_max_ana / (H / 4)
        nu_est = K * max(gamma_est, 1e-8)^(n_pl - 1)
        tau_init = 3 * nu_est + 0.5
        tau_field = fill(tau_init, Nx, Ny)

        Fx_arr = fill(Float64(Fx_val), Nx, Ny)
        Fy_arr = zeros(Float64, Nx, Ny)

        # Initialize equilibrium at rest
        for j in 1:Ny, i in 1:Nx, q in 1:9
            f_in[i, j, q] = equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
        end
        copy!(f_out, f_in)

        rheology = PowerLaw(K, n_pl; nu_min=1e-5, nu_max=5.0)

        for step in 1:max_steps
            stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
            collide_rheology_guo_2d!(f_out, is_solid, rheology, tau_field, Fx_arr, Fy_arr)
            f_in, f_out = f_out, f_in
        end

        # Compute macroscopic
        rho = ones(Float64, Nx, Ny)
        ux  = zeros(Float64, Nx, Ny)
        uy  = zeros(Float64, Nx, Ny)
        compute_macroscopic_forced_2d!(rho, ux, uy, f_in, Fx_val, 0.0)

        u_num = ux[2, :]
        u_max_num = maximum(u_num)

        # Relative L_inf error on interior points (skip wall-adjacent)
        errs = abs.(u_num[3:end-2] .- u_analytical[3:end-2]) ./ u_max_ana
        max_err = maximum(errs)

        @printf("  %5.1f   %12.6e   %12.6e   %12.4e\n",
                n_pl, u_max_num, u_max_ana, max_err)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_rheology_convergence()
end
