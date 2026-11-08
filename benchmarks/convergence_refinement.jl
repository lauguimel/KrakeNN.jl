# Grid refinement convergence benchmarks
# 1) Natural convection: Nu vs De Vahl Davis — uniform vs refined
# 2) Cylinder in channel: Cd (uniform) + flow stability (refined)
using KrakeNN
using Printf

const NU_REF_RA1E3 = 1.118   # De Vahl Davis (1983)
const CD_REF_RE20 = 5.58      # Schäfer & Turek (1996), D/H ≈ 0.25

nsteps_natconv(N) = max(5000, round(Int, 8 * N^2))
nsteps_cylinder(Ny) = max(10000, round(Int, 12 * Ny^2))

# =========================================================================
# Part 1 — Natural Convection (Ra=1e3)
# =========================================================================

function run_natconv_convergence(; Ns=[32, 64])
    println("\n" * "="^65)
    println("NATURAL CONVECTION — Ra=1e3, Pr=0.71, Nu_ref=$NU_REF_RA1E3")
    println("="^65)

    println("\n--- Uniform ---")
    @printf("  %5s  %8s  %8s  %7s  %5s\n", "N", "steps", "Nu", "err%", "order")
    errs_u = Float64[]
    for N in Ns
        r = run_natural_convection_2d(; N=N, Ra=1e3, Pr=0.71,
                                        max_steps=nsteps_natconv(N))
        e = abs(r.Nu - NU_REF_RA1E3) / NU_REF_RA1E3
        push!(errs_u, e)
        ord = length(errs_u) > 1 ? log2(errs_u[end-1] / errs_u[end]) : NaN
        @printf("  %5d  %8d  %8.4f  %6.2f%%  %5s\n",
                N, nsteps_natconv(N), r.Nu, e*100,
                isnan(ord) ? "  -" : @sprintf("%.2f", ord))
    end

    println("\n--- Refined (wall_fraction=0.2, ratio=2) ---")
    @printf("  %5s  %8s  %8s  %7s  %5s\n", "N_base", "steps", "Nu", "err%", "order")
    errs_r = Float64[]
    for N in Ns
        r = run_natural_convection_refined_2d(; N=N, Ra=1e3, Pr=0.71,
                max_steps=nsteps_natconv(N), ratio=2, wall_fraction=0.2)
        e = abs(r.Nu - NU_REF_RA1E3) / NU_REF_RA1E3
        push!(errs_r, e)
        ord = length(errs_r) > 1 ? log2(errs_r[end-1] / errs_r[end]) : NaN
        @printf("  %5d  %8d  %8.4f  %6.2f%%  %5s\n",
                N, nsteps_natconv(N), r.Nu, e*100,
                isnan(ord) ? "  -" : @sprintf("%.2f", ord))
    end

    return (errs_u, errs_r)
end

# =========================================================================
# Part 2 — Cylinder (Re=20)
# =========================================================================

function run_cylinder_convergence(; Nys=[40, 80], u_in=0.05)
    println("\n" * "="^65)
    println("CYLINDER IN CHANNEL — Re=20, D/H=0.25, Cd_ref=$CD_REF_RE20")
    println("="^65)

    println("\n--- Uniform (with MEA drag) ---")
    @printf("  %5s  %5s  %8s  %8s  %7s\n", "Ny", "Nx", "steps", "Cd", "err%")
    for Ny in Nys
        Nx = 4*Ny
        R = round(Int, 0.125*Ny)
        nu = u_in * 2R / 20.0
        steps = nsteps_cylinder(Ny)
        r = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R, u_in=u_in, ν=nu,
                              max_steps=steps, avg_window=min(2000, steps÷5))
        e = abs(r.Cd - CD_REF_RE20) / CD_REF_RE20 * 100
        @printf("  %5d  %5d  %8d  %8.4f  %6.2f%%\n", Ny, Nx, steps, r.Cd, e)
    end

    println("\n--- Refined via .krk (ratio=2 around cylinder) ---")
    @printf("  %5s  %5s  %8s  %8s  %8s  %10s\n",
            "Ny", "Nx", "steps", "|ux|_max", "|uy|_max", "ρ_range")
    for Ny in Nys
        Nx = 4*Ny
        R = round(Int, 0.125*Ny)
        nu = u_in * 2R / 20.0
        steps = nsteps_cylinder(Ny)
        Lx = Float64(Nx); Ly = Float64(Ny)
        R_phys = Float64(R); cx = 0.2*Lx; cy = 0.5*Ly
        margin = 3.0*R_phys

        setup = KrakeNN.parse_kraken("""
        Simulation cyl_ref D2Q9
        Domain L = $Lx x $Ly  N = $Nx x $Ny
        Define R = $R_phys
        Define cx = $cx
        Define cy = $cy
        Physics nu = $nu
        Obstacle cylinder { (x - cx)^2 + (y - cy)^2 <= R^2 }
        Boundary west velocity(ux = $u_in, uy = 0)
        Boundary east pressure(rho = 1.0)
        Boundary south wall
        Boundary north wall
        Refine cyl_zone { region = [$(max(0,cx-margin)), $(max(0,cy-margin)), $(min(Lx,cx+2margin)), $(min(Ly,cy+margin))], ratio = 2 }
        Run $steps steps
        """)
        r = run_simulation(setup)
        @printf("  %5d  %5d  %8d  %8.4f  %8.4f  [%.4f,%.4f]\n",
                Ny, Nx, steps,
                maximum(abs, r.ux), maximum(abs, r.uy),
                minimum(r.ρ), maximum(r.ρ))
    end
    # NOTE: Cd extraction from refined runner requires adding a MEA diagnostic
    # hook to advance_refined_step!. For now, the refined run validates flow
    # stability and field quality, not quantitative Cd.
end

# =========================================================================

function run_refinement_convergence(; Ns_natconv=[32, 64], Nys_cyl=[80])
    run_natconv_convergence(; Ns=Ns_natconv)
    run_cylinder_convergence(; Nys=Nys_cyl)
    println("\nDone.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_refinement_convergence()
end
