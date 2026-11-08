# Thermal LBM convergence study
# 1) Heat conduction: grid convergence with linear T profile
# 2) Natural convection: Nu(Ra) vs De Vahl Davis reference
using KrakeNN
using Printf

function run_thermal_convergence()
    # --- Part 1: Heat conduction grid convergence ---
    Nys = [8, 16, 32, 64, 128]
    alpha = 0.1
    omega_T = 1.0 / (3.0 * alpha + 0.5)
    T_hot, T_cold = 1.0, 0.0
    errors = Float64[]

    for Ny in Nys
        Nx = 4
        max_steps = max(5000, 5 * Ny^2)

        g_in  = zeros(Float64, Nx, Ny, 9)
        g_out = zeros(Float64, Nx, Ny, 9)
        Temp  = zeros(Float64, Nx, Ny)
        ux    = zeros(Float64, Nx, Ny)
        uy    = zeros(Float64, Nx, Ny)

        w = KrakeNN.weights(D2Q9())
        for j in 1:Ny, i in 1:Nx, q in 1:9
            g_in[i, j, q] = w[q] * 0.5
        end
        copy!(g_out, g_in)

        for step in 1:max_steps
            KrakeNN.stream_periodic_x_wall_y_2d!(g_out, g_in, Nx, Ny)
            KrakeNN.apply_fixed_temp_south_2d!(g_out, T_hot, Nx)
            KrakeNN.apply_fixed_temp_north_2d!(g_out, T_cold, Nx, Ny)
            KrakeNN.collide_thermal_2d!(g_out, ux, uy, omega_T)
            g_in, g_out = g_out, g_in
        end

        KrakeNN.compute_temperature_2d!(Temp, g_in)
        T_profile = Temp[2, :]

        # Analytical: linear from T_hot at wall (y=0.5) to T_cold at wall (y=Ny+0.5)
        T_analytical = [T_hot - (j - 0.5) / Ny for j in 1:Ny]

        err = maximum(abs.(T_profile .- T_analytical))
        push!(errors, err)
    end

    println("\n=== Thermal Conduction Convergence ===")
    @printf("  %5s   %12s   %5s\n", "Ny", "L_inf error", "Order")
    @printf("  %5s   %12s   %5s\n", "-----", "-----------", "-----")
    for (i, Ny) in enumerate(Nys)
        order = i > 1 ? log2(errors[i-1] / errors[i]) : NaN
        @printf("  %5d   %12.4e   %5s\n", Ny, errors[i],
                isnan(order) ? "  -" : @sprintf("%.2f", order))
    end

    # --- Part 2: Natural convection Nu vs reference ---
    println("\n=== Natural Convection Nu(Ra) ===")
    @printf("  %8s   %8s   %8s   %8s\n", "Ra", "Nu_LBM", "Nu_ref", "error%")
    @printf("  %8s   %8s   %8s   %8s\n", "--------", "------", "------", "------")

    # De Vahl Davis (1983) reference values
    ra_cases = [(1e3, 1.118)]

    for (Ra, Nu_ref) in ra_cases
        result = run_natural_convection_2d(; N=64, Ra=Ra, Pr=0.71, Rc=1.0,
                                             max_steps=30000)
        rel_err = abs(result.Nu - Nu_ref) / Nu_ref * 100
        @printf("  %8.0e   %8.4f   %8.4f   %7.2f%%\n", Ra, result.Nu, Nu_ref, rel_err)
    end

    return errors
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_thermal_convergence()
end
