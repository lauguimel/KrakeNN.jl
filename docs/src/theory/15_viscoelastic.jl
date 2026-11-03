# # Viscoelastic Constitutive Models
#
# Viscoelastic fluids — polymer solutions, melts, Boger fluids — exhibit
# both viscous and elastic behaviour.  Unlike Generalized Newtonian Fluids
# (which only modify the viscosity), viscoelastic models track an additional
# **conformation tensor** ``\mathbf{C}`` (or equivalently the polymeric
# stress ``\boldsymbol{\tau}_p``) that evolves alongside the LBM
# distributions.
#
# **References**:
# Krüger et al. (2017) [kruger2017lattice](@cite kruger2017lattice) §12,
# Fattal & Kupferman (2004) [fattal2004constitutive](@cite fattal2004constitutive).
#
# ## Total stress decomposition
#
# The total deviatoric stress is split into a Newtonian solvent contribution
# and a polymeric contribution:
# ```math
# \boldsymbol{\tau} = 2\,\eta_s\,\mathbf{S}
#     + \boldsymbol{\tau}_p
# ```
#
# where ``\eta_s = \rho\,\nu_s`` is the solvent viscosity.  The LBM
# collision handles the solvent part (via the relaxation rate
# ``\omega_s = 1/(3\nu_s + 1/2)``), while ``\boldsymbol{\tau}_p`` is
# evolved by a separate PDE and coupled to the LBM through a body force.
#
# ## Oldroyd-B model
#
# The simplest viscoelastic model couples a Newtonian solvent with a
# single-mode upper-convected Maxwell element:
# ```math
# \boldsymbol{\tau}_p + \lambda\,\overset{\nabla}{\boldsymbol{\tau}_p}
#     = 2\,\eta_p\,\mathbf{S}
# ```
#
# where ``\overset{\nabla}{\boldsymbol{\tau}_p}`` is the **upper-convected
# derivative**:
# ```math
# \overset{\nabla}{\boldsymbol{\tau}_p}
#     = \frac{\partial \boldsymbol{\tau}_p}{\partial t}
#     + \mathbf{u} \cdot \nabla \boldsymbol{\tau}_p
#     - \boldsymbol{\tau}_p \cdot \nabla\mathbf{u}
#     - (\nabla\mathbf{u})^T \cdot \boldsymbol{\tau}_p
# ```
#
# ``\lambda`` is the polymer relaxation time and ``\eta_p = \rho\,\nu_p``
# the polymer viscosity.  The **Weissenberg number**
# ``\mathrm{Wi} = \lambda\,\dot{\gamma}`` controls the degree of
# viscoelasticity.
#
# ```julia
# OldroydB(nu_s, nu_p, lambda)
# ```
#
# ## FENE-P model
#
# The Oldroyd-B model allows unlimited polymer stretching, which is
# unphysical.  The **Finitely Extensible Nonlinear Elastic** model with
# Peterlin closure (FENE-P) introduces a maximum extensibility ``L``:
# ```math
# \boldsymbol{\tau}_p = G\,f(\mathrm{tr}\,\mathbf{C})\,
#     (\mathbf{C} - \mathbf{I})
# ```
#
# with the Peterlin function:
# ```math
# f(\mathrm{tr}\,\mathbf{C})
#     = \frac{L^2}{L^2 - \mathrm{tr}\,\mathbf{C}}
# ```
#
# where ``G = \nu_p / \lambda`` is the elastic modulus and
# ``\mathbf{C}`` is the conformation tensor.  The Oldroyd-B model is
# recovered as ``L \to \infty``.
#
# ```julia
# FENEP(nu_s, nu_p, lambda, L_max)
# ```
#
# ## Saramito model
#
# The **Saramito (2007)** model extends Oldroyd-B with a yield stress,
# combining viscoelasticity with viscoplasticity.  Below the yield stress
# ``\tau_y``, the material behaves as a Kelvin-Voigt viscoelastic solid;
# above it, as an Oldroyd-B fluid.  This is achieved by modifying the
# relaxation term with a regularized yield function.
#
# ```julia
# Saramito(nu_s, nu_p, lambda, tau_y; m_reg=1000.0)
# ```
#
# ## Log-conformation formulation
#
# At high Weissenberg numbers (``\mathrm{Wi} \gg 1``), the conformation
# tensor ``\mathbf{C}`` can lose positive-definiteness due to exponential
# growth of eigenvalues, causing numerical blow-up.  The **log-conformation**
# approach of [fattal2004constitutive](@cite fattal2004constitutive)
# evolves ``\boldsymbol{\Theta} = \log \mathbf{C}`` instead:
# ```math
# \frac{\partial \boldsymbol{\Theta}}{\partial t}
#     + \mathbf{u} \cdot \nabla \boldsymbol{\Theta}
#     = \boldsymbol{\Omega}\,\boldsymbol{\Theta}
#     - \boldsymbol{\Theta}\,\boldsymbol{\Omega}
#     + 2\,\mathbf{B}
#     + \frac{1}{\lambda}\left(f\,e^{-\boldsymbol{\Theta}} - \mathbf{I}\right)
# ```
#
# where ``\boldsymbol{\Omega}`` and ``\mathbf{B}`` come from decomposing
# the velocity gradient ``\nabla\mathbf{u}`` in the eigenbasis of
# ``\mathbf{C} = \exp(\boldsymbol{\Theta})``.
#
# Since ``\boldsymbol{\Theta}`` is unconstrained (any symmetric matrix),
# ``\mathbf{C} = \exp(\boldsymbol{\Theta})`` is **guaranteed**
# positive-definite, eliminating the most common source of instability
# in viscoelastic simulations.
#
# ```julia
# OldroydB(nu_s, nu_p, lambda; formulation=LogConfFormulation())
# FENEP(nu_s, nu_p, lambda, L_max; formulation=LogConfFormulation())
# ```
#
# ## Eigendecomposition and matrix operations
#
# The log-conformation formulation requires efficient 2×2 symmetric matrix
# operations (eigendecomposition, exp, log) at every lattice node.
# Kraken provides GPU-optimized, branchless implementations:
#
# ```julia
# eigen_sym2x2(a11, a12, a22)       # → (λ1, λ2, e1x, e1y, e2x, e2y)
# mat_exp_sym2x2(a11, a12, a22)     # → (e11, e12, e22)
# mat_log_sym2x2(a11, a12, a22)     # → (l11, l12, l22)
# decompose_velocity_gradient(...)   # → (Ω12, B11, B22)
# ```
#
# All use analytical closed-form expressions with `atan`-based eigenvector
# computation, avoiding branches that would cause GPU warp divergence.
#
# ## Which code path uses which model
#
# Two implementations coexist in `src/`:
#
# - The explicit-Euler upwind kernels described in the rest of this page
#   (`evolve_stress_2d!`, `evolve_logconf_2d!`, `compute_polymeric_force_2d!`)
#   are driven by `run_viscoelastic_cylinder_2d`, which selects the
#   formulation with a symbol (`formulation=:logconf`), not with the
#   `OldroydB`/`FENEP` structs above.
# - The production 2D and 3D drivers (`run_viscoelastic_logfv_*`,
#   `run_viscoelastic_fvfd_*` and the `.krk` viscoelastic dispatch in
#   `simulation_runner.jl`) evolve the log-conformation on a finite-volume
#   grid, with MUSCL-superbee advection by default, and take a polymer
#   closure of type `LogConfOldroydB`, `LogConfFENEP`, `LogConfGiesekus`
#   or `LogConfPTT` (`src/drivers/viscoelastic_spec.jl`). The 3D sphere
#   driver only accepts `OldroydBSpec` (Oldroyd-B in direct conformation form);
#   it refuses log-conformation models.
# - `Saramito` exists as a type and can be built from a `.krk` file, but
#   no viscoelastic driver consumes it yet.
#
# ## Coupling with LBM
#
# The polymeric stress is coupled to the LBM collision through the
# **Guo forcing scheme**.  At each time step:
#
# 1. Compute macroscopic ``\rho``, ``\mathbf{u}`` from distributions
# 2. Evolve ``\boldsymbol{\tau}_p`` (or ``\boldsymbol{\Theta}``) using
#    upwind advection + source + relaxation (explicit Euler)
# 3. Compute the polymeric force as the divergence of the stress:
#    ```math
#    F_{p,i} = \frac{\partial \tau_{p,ij}}{\partial x_j}
#    ```
# 4. Add ``\mathbf{F}_p`` to the Guo forcing term in the BGK collision
#
# The force divergence is computed with central differences:
#
# ```julia
# compute_polymeric_force_2d!(Fx_p, Fy_p, tau_xx, tau_xy, tau_yy)
# ```
#
# ## Stress vs. log-conformation kernels
#
# Two evolution kernels are available, selected by the `formulation` field:
#
# | Formulation | Evolves | Stability | Cost |
# |:---|:---|:---|:---|
# | `StressFormulation()` | ``\boldsymbol{\tau}_p`` directly | Wi ≲ 1 | Lower |
# | `LogConfFormulation()` | ``\boldsymbol{\Theta} = \log\mathbf{C}`` | Wi ≫ 1 | Eigendecomposition |
#
# ```julia
# evolve_stress_2d!(tau_xx_new, tau_xy_new, tau_yy_new,
#                    tau_xx, tau_xy, tau_yy, ux, uy, nu_p, lambda)
#
# evolve_logconf_2d!(Theta_xx_new, Theta_xy_new, Theta_yy_new,
#                     Theta_xx, Theta_xy, Theta_yy, ux, uy;
#                     lambda=1.0, L_max=0.0)
# ```
#
# Both are fused advection + source + relaxation GPU kernels: a single
# kernel launch per time step for all three tensor components.

nothing  # suppress REPL output
