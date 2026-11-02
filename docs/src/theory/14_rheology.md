```@meta
EditURL = "14_rheology.jl"
```

# Non-Newtonian Rheology

Many fluids of practical interest — polymer solutions, blood, cement,
food products — exhibit a viscosity that depends on the local deformation
rate (shear-thinning or shear-thickening) or possess a yield stress.
Kraken.jl provides a modular rheology framework that plugs into the LBM
collision operator, with all models implemented as lightweight structs
suitable for direct GPU kernel arguments.

**References**:
Krüger et al. (2017) [kruger2017lattice](@cite kruger2017lattice) §12,
Gabbanelli et al. (2005) [gabbanelli2005lattice](@cite gabbanelli2005lattice),
Boyd et al. (2007) [boyd2007analysis](@cite boyd2007analysis).

## Generalized Newtonian fluids

A **Generalized Newtonian Fluid (GNF)** is defined by a viscosity that
depends only on the local shear rate magnitude ``\dot{\gamma}``:
```math
\boldsymbol{\tau} = 2\,\eta(\dot{\gamma})\,\mathbf{S}
```

where ``\mathbf{S} = (\nabla\mathbf{u} + \nabla\mathbf{u}^T)/2`` is
the strain-rate tensor and ``\dot{\gamma} = \sqrt{2\,\mathbf{S}:\mathbf{S}}``
is its second invariant.

In the LBM framework, the GNF approach modifies the collision step by
computing a **local relaxation rate** at each node:
```math
\omega(\mathbf{x}) = \frac{1}{3\,\nu(\dot{\gamma}) + 1/2}
```

This is a purely local operation — no additional PDEs need to be solved.

## Strain rate from non-equilibrium stress

A key advantage of the LBM is that the strain-rate tensor can be
computed **locally** from the non-equilibrium part of the distributions,
without any finite-difference stencil:
```math
\Pi_{\alpha\beta}^{\mathrm{neq}} = \sum_q (f_q - f_q^{\mathrm{eq}})\,
    e_{q\alpha}\,e_{q\beta}
```

```math
S_{\alpha\beta} = -\frac{\Pi_{\alpha\beta}^{\mathrm{neq}}}{2\,\rho\,c_s^2\,\tau}
```

The shear rate magnitude is then:
```math
\dot{\gamma} = \sqrt{2\,S_{\alpha\beta}\,S_{\alpha\beta}}
             = \sqrt{2\left(S_{xx}^2 + S_{yy}^2 + 2\,S_{xy}^2\right)}
    \quad \text{(2D)}
```

This is entirely local (no neighbour reads) and thus ideal for GPU
execution.  Both D2Q9 and D3Q19 implementations are provided:

```julia
strain_rate_magnitude_2d(f1,...,f9, feq1,...,feq9, rho, tau) → γ̇
strain_rate_magnitude_3d(f1,...,f19, feq1,...,feq19, rho, tau) → γ̇
```

!!! note "Implicit coupling"
    The strain rate depends on ``\tau``, which itself depends on ``\dot{\gamma}``
    through the rheological model.  In practice, a single explicit evaluation
    per time step (using ``\tau`` from the previous step) is sufficient for
    stability, since the LBM time step is small.

## Power-law model

The simplest GNF model:
```math
\eta(\dot{\gamma}) = K\,\dot{\gamma}^{\,n-1}
```

- ``n < 1``: **shear-thinning** (e.g. polymer solutions, blood)
- ``n > 1``: **shear-thickening** (e.g. dense suspensions)
- ``n = 1``: Newtonian (``\eta = K``)

The singularity at ``\dot{\gamma} \to 0`` is regularized by clamping:
``\nu \in [\nu_\min, \nu_\max]``.

```julia
PowerLaw(K, n; nu_min=1e-6, nu_max=10.0)
```

## Carreau-Yasuda model

A more physical model with finite zero-shear and infinite-shear
viscosities:
```math
\eta(\dot{\gamma}) = \eta_\infty + (\eta_0 - \eta_\infty)\,
    \big[1 + (\lambda\,\dot{\gamma})^a\big]^{(n-1)/a}
```

Setting ``a = 2`` recovers the standard **Carreau model**.
The parameter ``\lambda`` is the relaxation time (inverse of the
critical shear rate) and controls the onset of shear-thinning.

```julia
CarreauYasuda(eta_0, eta_inf, lambda, a, n)
```

## Cross model

An alternative to Carreau-Yasuda with a different functional form:
```math
\eta(\dot{\gamma}) = \eta_\infty
    + \frac{\eta_0 - \eta_\infty}{1 + (K\,\dot{\gamma})^m}
```

```julia
Cross(eta_0, eta_inf, K, m)
```

## Bingham plastic

A **yield-stress fluid** that behaves as a rigid solid below a
critical stress ``\tau_y`` and flows like a Newtonian fluid above it.
The ideal Bingham model has a discontinuity at ``\dot{\gamma} = 0``:
```math
\eta(\dot{\gamma}) = \frac{\tau_y}{\dot{\gamma}} + \mu_p
    \qquad (\dot{\gamma} > 0)
```

This is regularized using the **Papanastasiou (1987)** exponential
approximation, which is branchless and GPU-safe:
```math
\eta(\dot{\gamma}) = \tau_y\,
    \frac{1 - \exp(-m\,\dot{\gamma})}{\dot{\gamma}} + \mu_p
```

Larger ``m`` (regularization parameter) gives a closer approximation
to ideal Bingham behaviour.

```julia
Bingham(tau_y, mu_p; m_reg=1000.0)
```

## Herschel-Bulkley model

Combines yield stress with power-law behaviour above yielding:
```math
\eta(\dot{\gamma}) = \tau_y\,
    \frac{1 - \exp(-m\,\dot{\gamma})}{\dot{\gamma}}
    + K\,\dot{\gamma}^{\,n-1}
```

This generalizes both Bingham (``n = 1``) and power-law (``\tau_y = 0``).

```julia
HerschelBulkley(tau_y, K, n; m_reg=1000.0)
```

## Thermal coupling

For temperature-dependent rheology, Kraken provides two shift models
that multiply all rheological parameters by a thermal shift factor
``a_T(T)``:

**Arrhenius**:
```math
a_T = \exp\!\left[E_a\left(\frac{1}{T} - \frac{1}{T_\text{ref}}\right)\right]
```

**Williams-Landel-Ferry (WLF)**:
```math
\log a_T = -\frac{C_1\,(T - T_\text{ref})}{C_2 + T - T_\text{ref}}
```

Both are implemented as lightweight structs with zero overhead when
not used (``\texttt{IsothermalCoupling}`` is eliminated at compile time):

```julia
PowerLaw(K, n; thermal=ArrheniusCoupling(T_ref, E_a))
CarreauYasuda(eta_0, eta_inf, lambda, a, n; thermal=WLFCoupling(T_ref, C1, C2))
```

## Viscoelastic models

For fluids with memory (polymer melts, Boger fluids), Kraken provides
viscoelastic constitutive models that evolve a **conformation tensor**
``\mathbf{C}`` alongside the LBM distributions.

### Oldroyd-B

Linear viscoelastic model with a Newtonian solvent:
```math
\boldsymbol{\tau}_p + \lambda\,\overset{\nabla}{\boldsymbol{\tau}_p}
    = 2\,\eta_p\,\mathbf{S}
```

where ``\overset{\nabla}{\boldsymbol{\tau}_p}`` is the upper-convected
derivative and ``\lambda`` is the polymer relaxation time.

```julia
OldroydB(nu_s, nu_p, lambda)
```

### FENE-P

Finitely extensible nonlinear elastic model with Peterlin closure:
```math
\boldsymbol{\tau}_p + \lambda\,\overset{\nabla}{\boldsymbol{\tau}_p}
    = f(\mathrm{tr}\,\mathbf{C})\;2\,\eta_p\,\mathbf{S}
```

with the Peterlin function:
```math
f(\mathrm{tr}\,\mathbf{C}) = \frac{L^2}{L^2 - \mathrm{tr}\,\mathbf{C}}
```

where ``L`` is the maximum extensibility.  Reduces to Oldroyd-B as
``L \to \infty``.

```julia
FENEP(nu_s, nu_p, lambda, L_max)
```

### Log-conformation formulation

`LogConfFormulation()` is the default of the `OldroydB`, `FENEP` and
`Saramito` constructors; pass `formulation=StressFormulation()` to evolve
the stress directly.  See the next page for which driver uses which model.

At high Weissenberg numbers (``\mathrm{Wi} = \lambda\,\dot{\gamma} \gg 1``),
the conformation tensor can lose positive-definiteness, causing numerical
blow-up.  The **log-conformation** approach (Fattal & Kupferman, 2004)
evolves ``\boldsymbol{\Theta} = \log\mathbf{C}`` instead, which
guarantees ``\mathbf{C}`` remains positive-definite:

```julia
OldroydB(nu_s, nu_p, lambda; formulation=LogConfFormulation())
FENEP(nu_s, nu_p, lambda, L_max; formulation=LogConfFormulation())
```

## Implementation in Kraken

### Type hierarchy

The rheology module uses Julia's type system for zero-overhead dispatch:

```
AbstractRheology
├── GeneralizedNewtonian
│   ├── Newtonian
│   ├── PowerLaw
│   ├── CarreauYasuda
│   ├── Cross
│   ├── Bingham
│   └── HerschelBulkley
└── Viscoelastic
    ├── OldroydB
    ├── FENEP
    └── Saramito
```

All structs are lightweight (no arrays) and passed directly as GPU
kernel arguments.  Julia's JIT specializes kernels per concrete type,
so the dispatch is resolved at compile time with zero runtime overhead.

### Collision kernel

The rheological collision kernel computes the local viscosity and
relaxation rate at each node.  It accepts generalized Newtonian models
only; `tau_field` stores the relaxation time of the previous step.
Viscoelastic models use the separate drivers of the next page.
Variants with a Guo body force (`collide_rheology_guo_2d!`) and with a
temperature field (`collide_rheology_thermal_2d!`) also exist:

```julia
collide_rheology_2d!(f, is_solid, model::GeneralizedNewtonian, tau_field)
```

The algorithm at each node is:

1. Compute macroscopic ``\rho``, ``\mathbf{u}`` from ``f_q``
2. Compute equilibrium ``f_q^{\mathrm{eq}}``
3. Compute ``\dot{\gamma}`` from ``\Pi^{\mathrm{neq}}``
4. Evaluate ``\nu = \texttt{effective\_viscosity}(\texttt{model}, \dot{\gamma})``
5. Compute ``\omega = 1/(3\nu + 1/2)``
6. BGK collision: ``f_q^\star = f_q - \omega\,(f_q - f_q^{\mathrm{eq}})``

For thermal coupling, the variant `effective_viscosity_thermal` is used,
which applies the shift factor ``a_T`` before evaluating the model.

### Thermally coupled collision

When the flow populations ``f_q`` are coupled to a thermal
double-distribution field ``g_q`` (see
[thermal DDF](08_thermal_ddf.md)), the kernel
`collide_rheology_thermal_2d!(f, is_solid, rheology, tau_field, Fx, Fy, Temp)`
is called after streaming. At each fluid node:

1. The temperature is the zeroth moment of the thermal populations,
   ``T = \sum_q g_q``, stored in `Temp` by the streaming step.
2. The Boussinesq force ``F_y = \beta g\,(T - T_\text{ref})``
   (``F_x = 0``) enters the velocity as the Guo half-force term,
   ``\mathbf{u} = (\sum_q \mathbf{c}_q f_q + \mathbf{F}/2)/\rho``.
3. The shear rate ``\dot\gamma`` is evaluated from the non-equilibrium
   stress, using the relaxation time of the previous step.
4. `effective_viscosity_thermal(model, gamma_dot, T)` multiplies the
   model's parameters by ``a_T(T)``, then evaluates the model and applies
   the viscosity bounds of the model, if any.
5. ``\omega = 1/(3\nu + 1/2)`` and ``\tau = 1/\omega`` is stored in
   `tau_field` for the next step.
6. BGK collision with Guo forcing, using this ``\omega`` and the factor
   ``1 - \omega/2`` on the force term.

The driver `run_natural_convection_gnf_2d` assembles these steps for a
differentially heated cavity; see
[Generalised Newtonian natural convection](../users/benchmarks/gnf-natural-convection.md).
For a model without thermal coupling (``a_T = 1``), the viscosity still
depends on ``\dot\gamma`` only, and temperature acts through the buoyancy
force alone.

```julia
nothing  # suppress REPL output
```

