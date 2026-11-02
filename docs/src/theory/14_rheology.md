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

### D3Q19 TRT with LI-BB walls

`fused_trt_gnf_libb_guo_field_step_3d!` combines D3Q19 pull streaming,
LI-BB cut links, local GNF relaxation and Guo forcing. This 3D path supports
isothermal `Newtonian`, `PowerLaw`, `CarreauYasuda` and `Cross` models.
Model parameters are converted to the population element type before
launch; viscosity limits are those supplied by the constitutive model.
The forcing convention follows [guo2002discrete](@cite guo2002discrete),
with the even and odd Guo source terms weighted by their respective TRT
factors ``1-s_+/2`` and ``1-s_-/2``.

The equilibrium velocity and corrected non-equilibrium moment are
```math
\mathbf{u}^{*} = \frac{\sum_q f_q\mathbf{e}_q+\mathbf{F}/2}{\rho},
\qquad
\Pi_{\alpha\beta}^{*}
= \sum_q \bigl[f_q-f_q^{\mathrm{eq}}(\rho,\mathbf{u}^{*})\bigr]
         e_{q\alpha}e_{q\beta}
  + \frac{F_\alpha u_\beta^{*}+u_\alpha^{*}F_\beta}{2}.
```

The previous even relaxation time enters the strain evaluation:
```math
\tau_+^{\mathrm{old}} = \frac{1}{s_+^{\mathrm{old}}},
\qquad S_{\alpha\beta}
= -\frac{3\Pi_{\alpha\beta}^{*}}{2\rho\tau_+^{\mathrm{old}}},
\qquad
\dot{\gamma} =
\sqrt{2\left[S_{xx}^2+S_{yy}^2+S_{zz}^2+
       2(S_{xy}^2+S_{xz}^2+S_{yz}^2)\right]}.
```

The new rates are stored per cell and used for the current collision.
The channel driver keeps the TRT magic parameter fixed at ``\Lambda=3/16``:
```math
\nu = \nu(\dot{\gamma}),\qquad
s_+ = \frac{1}{3\nu+1/2},\qquad
s_- = \frac{1}{\Lambda/(3\nu)+1/2}.
```

`run_gnf_channel_3d` lowers planar walls at arbitrary cut fraction
``q_w\in(0,1]`` with gap ``H=N-1+2q_w``. Body-force channels support
any distinct flow and wall axes. The Zou-He option is restricted to
x-directed flow with zero body force. It copies each open face's relaxation
rates from the adjacent interior plane before rebuilding populations.
This imposes a zero-normal-gradient closure for the rates: populations on
an open face are incomplete before reconstruction, so their local strain
is unsuitable for determining those boundary rates.
Periodic dimensions include two ghost planes, refreshed for populations
and both rate fields; reported profiles exclude ghosts, solid planes and
Zou-He end faces.

CPU Float64 refinement on 2026-11-02 (macOS arm64, Julia 1.12.5) measured
the following relative velocity-L2 orders. These are the N=16,32,64 cases
in `test/analytical/gnf3d_channel_order.jl`; the independent reference
rescales constitutive parameters under diffusive refinement. Orders use
the actual gap ratio, ``p=\log(E_1/E_2)/\log(H_2/H_1)``.

| Law | ``q_w`` | ``p_{16\to32}`` | ``p_{32\to64}`` |
|:----|:-------:|---------------:|---------------:|
| Carreau, ``n=0.5`` | 0.5 | 2.000 | 2.000 |
| Carreau, ``n=0.5`` | 0.3 | 1.990 | 1.995 |
| Power law, ``n=0.7`` | 0.5 | 2.000 | 2.000 |
| Power law, ``n=1.5`` | 0.5 | 1.932 | 1.950 |

Every adjacent pair exceeds the fixed 1.8 gate, with no active power-law
clamp or smooth-stress companion. These measurements establish CPU
behaviour only; the GPU sweep to larger grids is scripted in
`benchmarks/validation/gnf3d/`.

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
With `cavity_powerlaw`, the reference viscosity ``\nu_0`` is the nominal
viscosity ``K(\alpha/H^2)^{n-1}`` of Turan et al. (2011), so Ra and Pr follow
their power-law definitions.
For a model without thermal coupling (``a_T = 1``), the viscosity still
depends on ``\dot\gamma`` only, and temperature acts through the buoyancy
force alone.

```julia
nothing  # suppress REPL output
```

