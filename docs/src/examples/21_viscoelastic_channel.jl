# # Viscoelastic Channel Flow (Oldroyd-B)
#
# ```@raw html
# <DownloadMenu :files="[{label:'viscoelastic_channel.csv',href:'/downloads/viscoelastic_channel/viscoelastic_channel.csv'},{label:'viscoelastic_channel.py',href:'/downloads/viscoelastic_channel/viscoelastic_channel.py'}]" />
# ```
#
# ## Problem Statement
#
# Fully developed channel flow of an Oldroyd-B viscoelastic fluid driven by a
# body force ``F_x``.  The Oldroyd-B model splits the total stress into a
# Newtonian solvent contribution (viscosity ``\nu_s``) and a polymeric
# contribution described by the conformation tensor ``\mathbf{C}``:
#
# ```math
# \boldsymbol{\tau}_p = G\,(\mathbf{C} - \mathbf{I}), \qquad
# G = \frac{\nu_p}{\lambda}
# ```
#
# where ``\nu_p`` is the polymeric viscosity, ``\lambda`` is the relaxation
# time, and ``G`` is the elastic modulus.
#
# ### Analytical solution
#
# At steady state, the velocity profile is identical to the Newtonian
# Poiseuille solution with the **total** viscosity ``\nu = \nu_s + \nu_p``:
#
# ```math
# u_x(y) = \frac{F_x}{2\nu}\, y\,(H - y)
# ```
#
# The first normal stress difference, however, is non-zero and quadratic in
# the shear rate:
#
# ```math
# N_1(y) = \tau_{xx} - \tau_{yy} = 2\,\nu_p\,\lambda\,\dot\gamma(y)^2
# ```
#
# where ``\dot\gamma(y) = F_x\,(H/2 - y)/\nu``.  At the centreline
# ``\dot\gamma = 0`` and ``N_1 = 0``; near the walls ``N_1`` reaches its
# maximum.
#
# ### Log-conformation method
#
# Direct evolution of ``\mathbf{C}`` is numerically unstable at high
# Weissenberg numbers.  KrakeNN uses the **log-conformation** approach
# of [Fattal & Kupferman (2004)](@cite fattal2004constitutive): the
# evolution equation is reformulated for ``\boldsymbol{\Theta} = \log\mathbf{C}``,
# which is unconditionally symmetric positive-definite.
#
# The kernel [`evolve_logconf_2d!`](@ref) advances ``\boldsymbol{\Theta}`` by
# one time step, and [`compute_stress_from_logconf_2d!`](@ref) recovers the
# polymeric stress via matrix exponentiation.  See
# [Theory --- Viscoelastic Flows](@ref) for the full derivation.
#
# ### Why this test matters
#
# The viscoelastic Poiseuille test validates:
#
# 1. **Log-conformation evolution** --- ``\boldsymbol{\Theta}`` must converge
#    to the correct steady-state conformation tensor.
# 2. **Stress recovery** --- ``\boldsymbol{\tau}_p`` must match the analytical
#    ``N_1`` profile.
# 3. **Coupling** --- The polymeric stress divergence must feed back correctly
#    into the momentum equation without altering the Newtonian velocity profile.
#
# ---
#
# ## LBM Setup
#
# | Parameter | Symbol | Value |
# |-----------|--------|-------|
# | Lattice   | ---    | D2Q9  |
# | Domain    | ``N_x \times N_y`` | ``4 \times 32`` (periodic in ``x``, walls in ``y``) |
# | Solvent viscosity | ``\nu_s`` | 0.08 |
# | Polymeric viscosity | ``\nu_p`` | 0.02 |
# | Total viscosity | ``\nu`` | 0.10 |
# | Relaxation time | ``\lambda`` | 5.0 |
# | Elastic modulus | ``G`` | ``\nu_p / \lambda = 0.004`` |
# | Body force | ``F_x`` | ``10^{-5}`` |
# | Solvent relaxation | ``\omega_s`` | ``1/(3\nu_s + 0.5) \approx 1.316`` |
# | Time steps | --- | 30 000 |
#
# ---
#
# ## Code

using KrakeNN

Nx, Ny = 4, 32
ν_s = 0.08
ν_p = 0.02
ν_total = ν_s + ν_p
lambda = 5.0
G = ν_p / lambda
Fx_val = 1e-5
max_steps = 30000

ω_s = 1.0 / (3.0 * ν_s + 0.5)

## Initialize LBM
f_in  = zeros(Float64, Nx, Ny, 9)
f_out = zeros(Float64, Nx, Ny, 9)
is_solid = falses(Nx, Ny)
ux = zeros(Float64, Nx, Ny)
uy = zeros(Float64, Nx, Ny)
ρ  = ones(Float64, Nx, Ny)

for j in 1:Ny, i in 1:Nx, q in 1:9
    f_in[i,j,q] = KrakeNN.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
end
copy!(f_out, f_in)

## Log-conformation arrays: Θ = log(C), initialised to log(I) = 0
Θ_xx = zeros(Float64, Nx, Ny);  Θ_xy = zeros(Float64, Nx, Ny)
Θ_yy = zeros(Float64, Nx, Ny)
Θ_xx_new = similar(Θ_xx);  Θ_xy_new = similar(Θ_xy);  Θ_yy_new = similar(Θ_yy)

## Polymeric stress
tau_p_xx = zeros(Float64, Nx, Ny)
tau_p_xy = zeros(Float64, Nx, Ny)
tau_p_yy = zeros(Float64, Nx, Ny)
Fx_p = zeros(Float64, Nx, Ny)
Fy_p = zeros(Float64, Nx, Ny)

## Total force = body force + polymeric stress divergence
Fx_total = fill(Float64(Fx_val), Nx, Ny)
Fy_total = zeros(Float64, Nx, Ny)

## Time stepping
for step in 1:max_steps
    stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
    collide_guo_field_2d!(f_out, is_solid, Fx_total, Fy_total, Float64(ω_s))
    compute_macroscopic_2d!(ρ, ux, uy, f_out)

    evolve_logconf_2d!(Θ_xx_new, Θ_xy_new, Θ_yy_new,
                       Θ_xx, Θ_xy, Θ_yy,
                       ux, uy; lambda=lambda)
    copyto!(Θ_xx, Θ_xx_new)
    copyto!(Θ_xy, Θ_xy_new)
    copyto!(Θ_yy, Θ_yy_new)

    compute_stress_from_logconf_2d!(tau_p_xx, tau_p_xy, tau_p_yy,
                                    Θ_xx, Θ_xy, Θ_yy; G=G)
    compute_polymeric_force_2d!(Fx_p, Fy_p, tau_p_xx, tau_p_xy, tau_p_yy)

    Fx_total .= Fx_val .+ Fx_p
    Fy_total .= Fy_p
    f_in, f_out = f_out, f_in
end

# ---
#
# ## Results --- Velocity and N1 Profiles
#
# ### Velocity profile
#
# The velocity profile is identical to the Newtonian case with ``\nu = \nu_s + \nu_p``.

H = Float64(Ny)
u_analytical = [Fx_val / (2 * ν_total) * (j - 0.5) * (H + 0.5 - j) for j in 1:Ny]
u_numerical  = ux[2, :]

# ![Oldroyd-B channel velocity profile.  Blue: analytical Poiseuille with total viscosity.  Orange: LBM with log-conformation.  The profiles overlap, confirming that the polymeric stress does not alter the steady-state velocity.](viscoelastic_velocity.svg)
#
# ### First normal stress difference
#
# ``N_1 = \tau_{xx} - \tau_{yy}`` is quadratic in ``\dot\gamma`` and vanishes
# at the centreline.

N1_numerical = tau_p_xx[2, :] .- tau_p_yy[2, :]

N1_analytical = zeros(Ny)
for j in 1:Ny
    γ_dot = Fx_val / ν_total * abs((j - 0.5) - H / 2)
    N1_analytical[j] = 2 * ν_p * lambda * γ_dot^2
end

# ![First normal stress difference N1 across the channel.  Blue: analytical 2 nu_p lambda gamma_dot^2.  Orange: LBM.  N1 is zero at the centreline and maximum near the walls, confirming correct viscoelastic stress computation.](viscoelastic_N1.svg)
#
# The numerical ``N_1`` profile closely follows the analytical prediction.
# At the centreline ``\dot\gamma = 0`` and ``N_1 \approx 0``; near the walls
# ``N_1`` grows quadratically.  Small deviations near the wall nodes are
# expected due to the bounce-back boundary discretisation.
#
# ---
#
# ## References
#
# - [Theory --- Viscoelastic Flows](@ref) (page 15)
# - [Fattal & Kupferman (2004)](@cite fattal2004constitutive) --- Log-conformation method
# - [Kruger *et al.* (2017)](@cite kruger2017lattice) --- The Lattice Boltzmann Method

nothing #hide
