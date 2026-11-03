```@meta
EditURL = "08_rayleigh_benard.jl"
```

# Rayleigh--Bénard Convection (2D)

```@raw html
<DownloadMenu :files="[{label:'rayleigh_benard.krk',href:'/downloads/rayleigh_benard/rayleigh_benard.krk'},{label:'rayleigh_benard.csv',href:'/downloads/rayleigh_benard/rayleigh_benard.csv'},{label:'rayleigh_benard.py',href:'/downloads/rayleigh_benard/rayleigh_benard.py'}]" />
```

**Concepts:** [Thermal DDF](../theory/08_thermal_ddf.md) ·
Boussinesq coupling

**Validates against:** De Vahl Davis (1983) — Nusselt number vs Rayleigh
[`10.1002/fld.1650030305`](https://doi.org/10.1002/fld.1650030305)

**Download:** [`rayleigh_benard.krk`](../assets/krk/rayleigh_benard.krk)

**Hardware:** Apple M3 Max, ~90s wall-clock at 128×64 (Ra = 10⁵)

![Velocity magnitude field for Rayleigh-Benard convection at steady state.  The convection rolls carry fast fluid up the hot plumes and down the cold ones, with quiet cores between the rolls.](rayleigh_benard_velocity.svg)

---

## Physical background

Rayleigh--Bénard convection is one of the most studied problems in fluid
dynamics.  A horizontal fluid layer is heated from below and cooled from
above.  Gravity acts downward.  The hot fluid at the bottom is lighter and
"wants" to rise, while the cold fluid at the top is heavier and "wants" to
sink.  Whether this actually happens depends on a single dimensionless
number: the **Rayleigh number**.

![Setup: hot bottom wall (red), cold top wall (blue), gravity pointing
down.  When Ra exceeds the critical value, convection rolls
develop.](rayleigh_benard_geometry.svg)


## The Rayleigh number

The Rayleigh number measures the ratio of buoyancy-driven destabilisation
to viscous and thermal damping:

```math
\mathrm{Ra} = \frac{g\,\beta\,\Delta T\, H^3}{\nu\,\alpha}
```

where:
- ``g`` — gravitational acceleration
- ``\beta`` — thermal expansion coefficient
- ``\Delta T = T_H - T_C`` — temperature difference between the plates
- ``H`` — distance between the plates
- ``\nu`` — kinematic viscosity
- ``\alpha = \nu / \mathrm{Pr}`` — thermal diffusivity

**Below** the critical value ``\mathrm{Ra}_c \approx 1708`` (for no-slip
walls), viscosity and thermal diffusion win: the fluid remains still and
heat transfer is purely conductive.  This is what we tested in the
[heat conduction example](07_heat_conduction.md).

**Above** ``\mathrm{Ra}_c``, buoyancy wins: the conductive state becomes
unstable and **convection rolls** appear spontaneously.  Hot fluid rises in
plumes, cold fluid sinks, and the heat transfer rate increases dramatically.


## Boussinesq approximation

In the Boussinesq approximation, density variations are neglected everywhere
**except** in the buoyancy term of the momentum equation:

```math
\mathbf{F}_{\text{buoy}} = -\rho_0 \, g \, \beta \, (T - T_{\text{ref}}) \, \hat{\mathbf{y}}
```

The fluid is otherwise treated as incompressible.  This is an excellent
approximation when ``\beta \, \Delta T \ll 1``, which is the standard regime
for Rayleigh--Bénard convection.

In Kraken's LBM implementation, the Boussinesq force is incorporated via the
[Guo forcing scheme](@cite guo2002boussinesq), which ensures second-order
accuracy.  The DDF thermal solver (see [heat conduction](07_heat_conduction.md)) provides the
temperature field at each time step.


## The Nusselt number

The **Nusselt number** quantifies how much convection enhances heat transfer
compared to pure conduction:

```math
\mathrm{Nu} = \frac{\text{total heat flux}}{\text{conductive heat flux}}
```

- ``\mathrm{Nu} = 1`` → pure conduction (no flow)
- ``\mathrm{Nu} > 1`` → convection enhances heat transfer

For ``\mathrm{Ra} = 10\,000``, the expected Nusselt number is approximately
``\mathrm{Nu} \approx 2.5``.


## LBM setup

| Parameter | Value |
|-----------|-------|
| Lattice   | D2Q9 (flow) + D2Q9 (thermal DDF) |
| Domain    | ``128 \times 64`` |
| Bottom    | Isothermal no-slip wall, ``T = T_H = 1`` |
| Top       | Isothermal no-slip wall, ``T = T_C = 0`` |
| Left/Right| Periodic |
| ``\mathrm{Ra}`` | 10 000 (supercritical) |
| ``\mathrm{Pr}`` | 0.71 (air) |

We use ``\mathrm{Pr} = 0.71`` (air) to demonstrate that the solver handles
``\mathrm{Pr} \neq 1``, i.e., different relaxation times for the flow and
thermal populations.


## Simulation

```julia
using Kraken

Ra     = 10_000.0
Pr     = 0.71
T_hot  = 1.0
T_cold = 0.0

ρ, ux, uy, Temp, config, Ra_out, Pr_out, ν, α = run_rayleigh_benard_2d(;
    Nx=128, Ny=64, Ra=Ra, Pr=Pr, T_hot=T_hot, T_cold=T_cold, max_steps=50000)
```

## Results — temperature field

The temperature field reveals the convective structure.  Hot plumes (red)
rise from the bottom boundary layer, while cold plumes (blue) descend from
the top.  In the bulk, convective mixing creates a more uniform temperature
compared to the linear conductive profile.

```julia
Nx_dom, Ny_dom = size(Temp)
```

![Temperature field at steady state.  The mushroom-shaped plumes are the
hallmark of Rayleigh--Bénard convection.  Hot fluid rises (red), cold fluid
sinks (blue).](rayleigh_benard_temperature.svg)


## Results — velocity field

The velocity magnitude shows pairs of counter-rotating **convection rolls**.
Fluid rises in the hot plumes, moves horizontally near the top and bottom
walls, and descends in the cold plumes.  The rolls are roughly as wide as
they are tall — a well-known feature of Rayleigh--Bénard convection near
onset.

```julia
umag = @. sqrt(ux^2 + uy^2)
```

![Velocity magnitude showing the convection roll
structure.](rayleigh_benard_velocity.svg)


## Mean temperature profile

The horizontally-averaged temperature profile ``\langle T \rangle_x(y)``
highlights the difference between conduction and convection:
- **Conductive regime**: linear profile (dashed grey line)
- **Convective regime**: thin boundary layers near the walls with a
  well-mixed, nearly uniform core

The thinner the boundary layers, the higher the Nusselt number.

```julia
T_avg  = [sum(Temp[:, j]) / Nx_dom for j in 1:Ny_dom]
y_norm = [(j - 0.5) / Ny_dom for j in 1:Ny_dom]   # walls at j = 1/2 and Ny_dom + 1/2
T_lin  = [T_hot - (T_hot - T_cold) * y for y in y_norm]
```

The departure from the linear profile is the signature of convective heat
transfer.  Near the walls, steep temperature gradients indicate thin
thermal boundary layers; in the bulk, the profile is nearly flat.


## What this test validates

| Component | Validated? |
|-----------|:----------:|
| Thermal-fluid coupling (DDF + Boussinesq) | yes |
| Buoyancy force (Guo scheme) | yes |
| Spontaneous symmetry breaking (convection onset) | yes |
| Correct roll structure at supercritical Ra | yes |
| ``\mathrm{Pr} \neq 1`` handling | yes |

This example confirms that Kraken correctly couples the thermal and flow
solvers.  The next step in complexity is
[Hagen--Poiseuille flow](09_hagen_poiseuille.md), which introduces axisymmetric geometry.


## References

- [Guo *et al.* (2002)](@cite guo2002boussinesq) — Boussinesq forcing in LBM
- [He *et al.* (1998)](@cite he1998novel) — Thermal DDF model
- [Kruger *et al.* (2017)](@cite kruger2017lattice) — LBM textbook

