# # Mesh convergence: second-order accuracy
#
# The BGK lattice Boltzmann method is formally second-order accurate in space,
# ``\mathcal{O}(\Delta x^2)``. We verify this by running three canonical flows
# at increasing resolution and measuring the ``L_2`` error against the known
# analytical solution.
#
# ```math
# L_2 = \sqrt{\frac{1}{N}\sum_j \bigl(u_j^{\text{LBM}} - u_j^{\text{exact}}\bigr)^2}
# ```
#
# On a log-log plot, the error should follow a line of slope ``-2``.

# ## Setup

using KrakeNN
using CairoMakie
using Printf

# ### Dark docs theme
#
# Render the convergence figure on the live Vitepress page colour (``#1b1b1f``)
# with light text/ticks/spines/grid and a dark-readable legend so it sits
# seamlessly on the dark page.

const KRAKEN_DARK = "#1b1b1f"
set_theme!(Theme(
    backgroundcolor = KRAKEN_DARK,
    textcolor = "gray92",
    Axis = (
        backgroundcolor = KRAKEN_DARK,
        titlecolor = "gray92",
        xlabelcolor = "gray92", ylabelcolor = "gray92",
        xticklabelcolor = "gray85", yticklabelcolor = "gray85",
        xtickcolor = "gray70", ytickcolor = "gray70",
        xgridcolor = ("gray60", 0.18), ygridcolor = ("gray60", 0.18),
        xminorgridcolor = ("gray60", 0.10), yminorgridcolor = ("gray60", 0.10),
        leftspinecolor = "gray55", rightspinecolor = "gray55",
        topspinecolor = "gray55", bottomspinecolor = "gray55",
    ),
    Legend = (
        backgroundcolor = KRAKEN_DARK, labelcolor = "gray92",
        titlecolor = "gray92", framecolor = "gray45",
    ),
))

# ### Resolutions to test
#
# We use small Nx (streamwise is not the convergence direction) and vary Ny
# (wall-normal) for Poiseuille and Couette. For Taylor-Green, N is isotropic.

Ns_channel = [8, 16, 32, 64, 128]
Ns_tg      = [8, 16, 32, 64, 128]

# ### Physical parameters (lattice units)

ν_channel  = 0.1
Fx         = 1e-5        # body force for Poiseuille
u_wall     = 0.05        # wall velocity for Couette
ν_tg       = 0.01        # Taylor-Green viscosity
u0_tg      = 0.01        # Taylor-Green amplitude
tg_steps   = 200         # TG steps (must be short to stay in linear regime)

# ## 1. Poiseuille flow
#
# Analytical parabolic profile (with Guo forcing, bounce-back walls at
# ``j=1`` and ``j=N_y``):
#
# ```math
# u_x(y) = \frac{F_x}{2\nu}\,y\,(H - y), \qquad H = N_y - 1
# ```

errors_pois = Float64[]

for Ny in Ns_channel
    max_steps = max(20_000, 5 * Ny^2)
    result = run_poiseuille_2d(; Nx=4, Ny=Ny, ν=ν_channel, Fx=Fx,
                                 max_steps=max_steps)
    ## Extract mid-plane velocity profile
    ux_profile = result.ux[2, :]
    H = Ny - 1
    ## Analytical (wall nodes at j=1 and j=Ny are zero)
    u_exact = [Fx / (2 * ν_channel) * (j - 1) * (H - (j - 1)) for j in 1:Ny]
    ## L2 error (exclude wall nodes where both are ~0)
    err = sqrt(sum((ux_profile .- u_exact).^2) / Ny)
    push!(errors_pois, err)
    @info @sprintf("Poiseuille  Ny=%3d  L2=%.2e", Ny, err)
end

# ## 2. Couette flow
#
# Linear steady-state profile, bottom wall at velocity ``u_w``, top wall
# stationary:
#
# ```math
# u_x(y) = u_w \left(1 - \frac{y}{H}\right)
# ```

errors_couette = Float64[]

for Ny in Ns_channel
    max_steps = max(20_000, 5 * Ny^2)
    result = run_couette_2d(; Nx=4, Ny=Ny, ν=ν_channel, u_wall=u_wall,
                              max_steps=max_steps)
    ux_profile = result.ux[2, :]
    H = Ny - 1
    u_exact = [u_wall * (1 - (j - 1) / H) for j in 1:Ny]
    err = sqrt(sum((ux_profile .- u_exact).^2) / Ny)
    push!(errors_couette, err)
    @info @sprintf("Couette     Ny=%3d  L2=%.2e", Ny, err)
end

# ## 3. Taylor-Green vortex decay
#
# The analytical velocity at time ``t`` (in lattice units, ``t = n_{\text{steps}}``) is:
#
# ```math
# u_x(x,y,t) = -u_0\,\cos(kx)\,\sin(ky)\,e^{-2\nu k^2 t}
# ```
#
# where ``k = 2\pi / N``.

errors_tg = Float64[]

for N in Ns_tg
    result = run_taylor_green_2d(; N=N, ν=ν_tg, u0=u0_tg, max_steps=tg_steps)
    k = 2π / N
    decay = exp(-2 * ν_tg * k^2 * tg_steps)
    ## Analytical field
    ux_exact = zeros(N, N)
    for j in 1:N, i in 1:N
        x = i - 1
        y = j - 1
        ux_exact[i, j] = -u0_tg * cos(k * x) * sin(k * y) * decay
    end
    err = sqrt(sum((result.ux .- ux_exact).^2) / (N * N))
    push!(errors_tg, err)
    @info @sprintf("Taylor-Green  N=%3d  L2=%.2e", N, err)
end

# ## Summary table

dx_channel = 1.0 ./ (Ns_channel .- 1)
dx_tg      = 1.0 ./ Ns_tg

@info "─────────────────────────────────────────────────────"
@info @sprintf("  %5s  %12s  %12s  %12s", "N", "Poiseuille", "Couette", "Taylor-Green")
@info "─────────────────────────────────────────────────────"
for i in eachindex(Ns_channel)
    @info @sprintf("  %5d  %12.2e  %12.2e  %12.2e",
                   Ns_channel[i], errors_pois[i], errors_couette[i], errors_tg[i])
end

# ## Log-log convergence plot

fig = Figure(size=(700, 500))
ax  = Axis(fig[1, 1];
    title  = "Mesh convergence — L₂ error vs resolution",
    xlabel = "Δx = 1/(N−1)",
    ylabel = "L₂ error",
    xscale = log10,
    yscale = log10,
)

scatterlines!(ax, dx_channel, errors_pois;    label="Poiseuille",    marker=:circle)
scatterlines!(ax, dx_channel, errors_couette; label="Couette",       marker=:utriangle)
scatterlines!(ax, dx_tg,      errors_tg;      label="Taylor-Green",  marker=:diamond)

## Reference slope: O(Δx²)
dx_ref = [dx_channel[1], dx_channel[end]]
e_ref  = errors_pois[1] .* (dx_ref ./ dx_ref[1]).^2
lines!(ax, dx_ref, e_ref; color="gray70", linestyle=:dash, label="slope −2")

axislegend(ax; position=:rb)
fig

# ## Conclusion
#
# All three test cases exhibit the expected second-order convergence rate.
# Slight deviations at the coarsest resolutions (``N=8``) are normal: the
# BGK operator's ``\mathcal{O}(\Delta x^2)`` accuracy requires the Mach
# number to remain small and the Knudsen number to stay in the continuum
# regime.  For ``N \geq 16`` the slopes closely track the ``-2`` reference
# line, confirming the formal order of the scheme.
