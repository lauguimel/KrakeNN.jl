# # Couette Flow (2D) — Tutorial
#
#
# ## Problem statement
#
# Plane Couette flow is the steady, shear-driven flow between two parallel
# plates.  One wall moves at velocity ``u_w`` while the other is stationary.
# The resulting velocity profile is linear:
#
# ```math
# u_x(y) = u_w \,\frac{y}{H}
# ```
#
# This is one of the rare cases where the LBM gives an **exact** solution:
# the D2Q9 equilibrium is quadratic in velocity, so a linear profile is
# reproduced at machine precision.
#
# ---
#
# ## Simulation file
#
# Download: [`couette.krk`](../assets/krk/couette.krk)
#
# ```
# Simulation couette D2Q9
# Domain  L = 0.125 x 1.0  N = 4 x 32
#
# Define u_wall = 0.05
#
# Physics nu = 0.1
#
# Boundary x periodic
# Boundary south wall
# Boundary north velocity(ux = u_wall, uy = 0)
#
# Run 10000 steps
# ```
#
# ## Run and post-process

using KrakeNN
using CairoMakie

krk = joinpath(@__DIR__, "..", "..", "..", "examples", "couette.krk")
result = run_simulation(krk)

# Extract the velocity profile along ``y`` at the domain centre:

prof = extract_line(result, :ux, :y; at=0.5)

# Check global statistics:

stats = domain_stats(result)
nothing #hide

# `stats.max_u`, `stats.mean_rho`, `stats.mass_error` — one call.
#
# ## Velocity profile
#
# Compare with the analytical solution.  `field_error` takes an expression
# string that can reference all `Define` variables from the `.krk` file:

err = field_error(result, :ux, "u_wall * (y - 0.5*dy) / (Ny*dy - dy)")
nothing #hide

# The analytical expression uses `u_wall` (from `Define`), `y`, `dy`, `Ny`
# — all resolved automatically from the `.krk` setup.

Ny = result.setup.domain.Ny
u_wall = result.setup.user_vars[:u_wall]
H = Ny - 1

fig = Figure(size=(600, 420))
ax = Axis(fig[1, 1];
    xlabel = "u_x  (lattice units)",
    ylabel = "y  (physical units)",
    title  = "Couette flow — Ny = $Ny, L₂ error = $(round(err.error, digits=2))")
lines!(ax, err.analytical_field[2, :], prof.coord; label="Analytical", linewidth=2)
scatter!(ax, prof.values, prof.coord; label="LBM (.krk)", markersize=8)
axislegend(ax; position=:rt)
fig
save("couette_profile.svg", fig) #hide

# ---
#
# ## Convergence study
#
# Sweep ``N_y`` using **parametric kwargs** — same `.krk` file, different
# resolution.  `field_error` computes the error at each level:

Ny_list = [16, 32, 64, 128]
errors  = Float64[]

for Ny_i in Ny_list
    nsteps = max(10_000, ceil(Int, 3 * (Ny_i - 1)^2 / 0.1))
    res = run_simulation(krk; Ny=Ny_i, max_steps=nsteps)
    e = field_error(res, :ux, "u_wall * (y - 0.5*dy) / (Ny*dy - dy)")
    push!(errors, e.error)
end

fig2 = Figure(size=(500, 400))
ax2  = Axis(fig2[1, 1];
    xlabel = "Ny", ylabel = "Relative L₂ error",
    title  = "Convergence — Couette flow",
    xscale = log10, yscale = log10)
scatterlines!(ax2, Float64.(Ny_list), errors;
    linewidth=2, markersize=10, label="LBM")
hlines!(ax2, [eps()]; linestyle=:dash, color=:gray, label="machine ε")
axislegend(ax2; position=:rt)
fig2
save("couette_convergence.svg", fig2) #hide

# ---
#
# ## Summary of post-processing helpers
#
# | Function | Purpose | Example |
# |----------|---------|---------|
# | `extract_line(result, :ux, :y; at=0.5)` | 1D profile along an axis | Centreline velocity |
# | `field_error(result, :ux, "expr")` | ``L_2``/``L_\infty`` error vs expression | Validation |
# | `probe(result, :ux, x, y)` | Point value | Specific location |
# | `domain_stats(result)` | Global stats (max\_u, mass error, ...) | Quick check |
#
# Expressions in `field_error` have access to all `Define` variables from the
# `.krk` file, plus `x`, `y`, `Lx`, `Ly`, `Nx`, `Ny`, `dx`, `dy`.
#
# ## References
#
# - [Zou & He (1997)](@cite zou1997pressure) — Zou–He boundary conditions
# - [Kruger *et al.* (2017)](@cite kruger2017lattice) — LBM textbook
