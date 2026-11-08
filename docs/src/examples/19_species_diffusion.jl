# # Species Diffusion
#
# ```@raw html
# <DownloadMenu :files="[{label:'species_diffusion.csv',href:'/downloads/species_diffusion/species_diffusion.csv'},{label:'species_diffusion.py',href:'/downloads/species_diffusion/species_diffusion.py'}]" />
# ```
#
# ## Problem Statement
#
# Pure diffusion of a passive scalar concentration ``C`` between two parallel
# plates held at fixed concentrations ``C_{\text{south}} = 1`` and
# ``C_{\text{north}} = 0``.  The flow is quiescent (``\mathbf{u} = 0``), so
# the problem reduces to the 1D steady diffusion equation:
#
# ```math
# \frac{\partial^2 C}{\partial y^2} = 0
# ```
#
# with the analytical solution:
#
# ```math
# C(y) = 1 - \frac{y}{H}
# ```
#
# where ``H = N_y`` is the channel height.  This is a linear profile from
# ``C = 1`` at the bottom wall to ``C = 0`` at the top wall.
#
# ### Double-distribution function approach
#
# The species concentration is tracked by a separate set of D2Q9 populations
# ``h_q`` with its own relaxation parameter ``\omega_D = 1/(3D + 0.5)``,
# where ``D`` is the mass diffusivity.  The equilibrium includes only the
# first-order velocity term:
#
# ```math
# h_q^{\text{eq}} = w_q\,C\left(1 + \frac{\mathbf{c}_q \cdot \mathbf{u}}{c_s^2}\right)
# ```
#
# Fixed-concentration boundary conditions use the anti-bounce-back method
# (Dirichlet BC for the scalar field).
#
# ### Why this test matters
#
# This test validates:
#
# 1. **Species collision kernel** --- [`collide_species_2d!`](@ref) must
#    correctly relax ``h_q`` towards the scalar equilibrium.
# 2. **Concentration recovery** --- [`compute_concentration_2d!`](@ref) must
#    sum the populations to recover ``C``.
# 3. **Dirichlet BCs** --- The anti-bounce-back scheme must impose the correct
#    wall concentrations.
# 4. **Mesh convergence** --- The error should decrease as ``\mathcal{O}(\Delta x^2)``.
#
# For details on the species transport model see
# [Theory --- Species Transport](@ref).
#
# ---
#
# ## LBM Setup
#
# | Parameter | Symbol | Value |
# |-----------|--------|-------|
# | Lattice   | ---    | D2Q9  |
# | Domain    | ``N_x \times N_y`` | ``4 \times 32`` (periodic in ``x``, walls in ``y``) |
# | Diffusivity | ``D`` | 0.1 |
# | Relaxation | ``\omega_D`` | ``1/(3D + 0.5) \approx 0.769`` |
# | South BC  | ``C`` | 1.0 (fixed) |
# | North BC  | ``C`` | 0.0 (fixed) |
# | Time steps | --- | 5 000 |
#
# ---
#
# ## Code

using KrakeNN

Nx, Ny = 4, 32
D_coeff = 0.1
ω_D = 1.0 / (3.0 * D_coeff + 0.5)

## Species populations (D2Q9)
h_in  = zeros(Float64, Nx, Ny, 9)
h_out = zeros(Float64, Nx, Ny, 9)
C     = zeros(Float64, Nx, Ny)
ux    = zeros(Float64, Nx, Ny)
uy    = zeros(Float64, Nx, Ny)

## Initialize to uniform C = 0.5
w = KrakeNN.weights(D2Q9())
for j in 1:Ny, i in 1:Nx, q in 1:9
    h_in[i, j, q] = w[q] * 0.5
end
copy!(h_out, h_in)

## Time stepping
for step in 1:5000
    KrakeNN.stream_periodic_x_wall_y_2d!(h_out, h_in, Nx, Ny)
    apply_fixed_conc_south_2d!(h_out, 1.0, Nx)
    apply_fixed_conc_north_2d!(h_out, 0.0, Nx, Ny)
    collide_species_2d!(h_out, ux, uy, ω_D)
    h_in, h_out = h_out, h_in
end

compute_concentration_2d!(C, h_in)

# ---
#
# ## Results --- Concentration Profile
#
# We compare the numerical profile at ``x = 2`` to the analytical linear
# solution.

C_profile    = C[2, :]
C_analytical = [1.0 - (j - 0.5) / Ny for j in 1:Ny]

# ![Species diffusion concentration profile.  Blue line: analytical linear profile C(y) = 1 - y/H.  Orange dots: LBM species transport.  The numerical solution matches the analytical solution to high accuracy.](species_diffusion_profile.svg)
#
# ---
#
# ## Convergence Study
#
# We run the diffusion problem at four resolutions and verify second-order
# convergence.
#
# ```math
# E_{L_2} = \sqrt{\frac{\sum_j (C_j^{\text{num}} - C_j^{\text{ana}})^2}
#                       {\sum_j (C_j^{\text{ana}})^2}}
# ```

Ny_list = [8, 16, 32, 64]
errors  = Float64[]

for Ny_i in Ny_list
    h_i  = zeros(Float64, 4, Ny_i, 9)
    h_o  = zeros(Float64, 4, Ny_i, 9)
    ux_i = zeros(Float64, 4, Ny_i)
    uy_i = zeros(Float64, 4, Ny_i)
    C_i  = zeros(Float64, 4, Ny_i)

    for j in 1:Ny_i, i in 1:4, q in 1:9
        h_i[i, j, q] = w[q] * 0.5
    end
    copy!(h_o, h_i)

    for step in 1:8000
        KrakeNN.stream_periodic_x_wall_y_2d!(h_o, h_i, 4, Ny_i)
        apply_fixed_conc_south_2d!(h_o, 1.0, 4)
        apply_fixed_conc_north_2d!(h_o, 0.0, 4, Ny_i)
        collide_species_2d!(h_o, ux_i, uy_i, ω_D)
        h_i, h_o = h_o, h_i
    end

    compute_concentration_2d!(C_i, h_i)
    C_num = C_i[2, :]
    C_ana = [1.0 - (j - 0.5) / Ny_i for j in 1:Ny_i]
    L2 = sqrt(sum((C_num .- C_ana).^2) / sum(C_ana.^2))
    push!(errors, L2)
end

# ![Convergence of species diffusion.  Log-log plot of L2 error vs Ny.  The errors follow a slope-2 line, confirming second-order convergence.](species_diffusion_convergence.svg)
#
# ---
#
# ## References
#
# - [Theory --- Species Transport](@ref) (page 17)
# - [Kruger *et al.* (2017)](@cite kruger2017lattice) --- The Lattice Boltzmann Method

nothing #hide
