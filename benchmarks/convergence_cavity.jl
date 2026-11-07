# Cavity flow validation — Re sweep vs Ghia et al. (1982)
using KrakeNN
using Printf

# Ghia et al. (1982) reference data: (y/L, ux/U) on vertical centerline
const GHIA_Y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
                0.9531, 0.9609, 0.9688, 0.9766, 1.0000]

const GHIA_RE100_UX = [0.0000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                       -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                       0.68717, 0.73722, 0.78871, 0.84123, 1.00000]

const GHIA_RE400_UX = [0.0000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299,
                       -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093,
                       0.55892, 0.61756, 0.68439, 0.75837, 1.00000]

const GHIA_RE1000_UX = [0.0000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
                        -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304,
                        0.46604, 0.51117, 0.57492, 0.65928, 1.00000]

const GHIA_DATA = Dict(100 => GHIA_RE100_UX, 400 => GHIA_RE400_UX, 1000 => GHIA_RE1000_UX)

function run_cavity_sweep(; full::Bool=false)
    cases = [
        (Re=100, N=128, max_steps=60000),
        (Re=400, N=256, max_steps=200000),
    ]
    if full
        push!(cases, (Re=1000, N=256, max_steps=500000))
    end

    println("\n=== Cavity Re Sweep vs Ghia (1982) ===")
    @printf("  %6s   %5s   %12s\n", "Re", "N", "L∞ error")
    @printf("  %6s   %5s   %12s\n", "------", "-----", "----------")

    for c in cases
        u_lid = 0.1
        ν = u_lid * c.N / c.Re
        config = LBMConfig(D2Q9(); Nx=c.N, Ny=c.N, ν=ν, u_lid=u_lid, max_steps=c.max_steps)
        result = run_cavity_2d(config)

        # Interpolate centerline ux at Ghia y-positions
        mid_x = c.N ÷ 2
        ux_centerline = result.ux[mid_x, :] ./ u_lid  # normalize

        ghia_ux = GHIA_DATA[c.Re]
        max_err = 0.0
        for (yg, ug) in zip(GHIA_Y, ghia_ux)
            j = clamp(round(Int, yg * c.N) + 1, 1, c.N)
            err = abs(ux_centerline[j] - ug)
            max_err = max(max_err, err)
        end

        @printf("  %6d   %5d   %12.4e\n", c.Re, c.N, max_err)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    full = "--full" in ARGS || get(ENV, "FULL_MODE", "") == "1"
    run_cavity_sweep(; full=full)
end
