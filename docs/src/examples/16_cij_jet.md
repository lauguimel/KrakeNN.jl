```@meta
EditURL = "16_cij_jet.jl"
```

# CIJ Jet Breakup (Axisymmetric Validation)


## Problem Statement

A Continuous InkJet (CIJ) printhead ejects a liquid jet through a nozzle
and stimulates it with a periodic velocity perturbation.  The stimulation
triggers the **Rayleigh--Plateau instability** at a controlled wavelength,
causing the jet to break up into uniform droplets.

This is a validation case: we compare KrakeNN's axisymmetric two-phase
LBM against reference data from **Basilisk** (Popinet, 2015), a
well-validated Navier--Stokes VOF solver with adaptive mesh refinement.

The reference dataset comprises ~110,000 interface snapshots spanning
``\text{Re} \in [100, 1000]``, ``\text{We} = 600``, and 6 stimulation
amplitudes (Roche *et al.*, 2021).

### Dimensionless parameters

The problem is controlled by three dimensionless numbers:

```math
\text{Re} = \frac{U\, R_0}{\nu_l}, \qquad
\text{We} = \frac{\rho_l\, U^2\, R_0}{\sigma}, \qquad
\text{Oh} = \frac{\mu_l}{\sqrt{\rho_l\, \sigma\, R_0}} = \frac{\sqrt{\text{We}}}{\text{Re}}
```

where ``U`` is the mean inlet velocity, ``R_0`` the jet radius,
``\nu_l`` the liquid kinematic viscosity, and ``\sigma`` the surface
tension coefficient.

The pulsed inlet velocity is:
```math
u(t) = U \left(1 + \delta \sin\!\left(\frac{2\pi t}{T}\right)\right)
```
with period ``T = 7\,R_0/U`` (wavelength ``\lambda = 7\,R_0``),
corresponding to an optimal Rayleigh--Plateau wavenumber ``k\,R_0 \approx 0.9``.

### Basilisk reference setup

| Parameter | Value |
|:----------|:------|
| Geometry | Axisymmetric, domain ``512 \times 256`` (``z \times r``) |
| Reynolds | ``\text{Re} = 100``--``1000`` |
| Weber | ``\text{We} = 600`` |
| Density ratio | ``\rho_l / \rho_g = 1000`` |
| Viscosity ratio | ``\mu_l / \mu_g = 500`` |
| Stimulation | ``\delta \in \{0.01, 0.015, 0.02, 0.025, 0.03, 0.035\}`` |
| AMR | Level 15, ``\Delta x_\min \approx 0.0156\, R_0`` |
| Solver | Navier--Stokes VOF with wavelet adaptation |

## KrakeNN approach

### Two simulation models

KrakeNN provides **two drivers** for the CIJ jet:

| Driver | Model | Max ``\rho_l/\rho_g`` | Interface tracking |
|:-------|:------|:---------------------:|:-------------------|
| `run_cij_jet_axisym_2d` | MRT + VOF PLIC | ~10 | Geometric (PLIC) |
| `run_cij_jet_phasefield_2d` | Phase-field + pressure MRT | **1000** | Allen-Cahn |

The **phase-field driver** uses two D2Q9 distributions:
- ``f_q``: pressure-based MRT (modified equilibrium, ``\rho_\text{lbm} \approx 1``)
- ``g_q``: conservative Allen-Cahn for order parameter ``\varphi``

See the [Phase-Field Theory](@ref) page for the mathematical formulation.

### LBM parameter mapping

We map the dimensionless numbers to lattice units:
```math
\nu_l = \frac{u_\text{lb}\, R_0}{Re}, \qquad
\sigma = \frac{\rho_l\, u_\text{lb}^2\, R_0}{We}, \qquad
\tau = 3\,\nu_l + \tfrac{1}{2}
```

**Stability constraint**: MRT collision is used since BGK becomes unstable
when ``\tau`` approaches ``0.5``.  With ``R_0 = 40`` and
``u_\text{lb} = 0.04``, we obtain ``\tau = 0.524`` for ``\text{Re} = 200``.

### Simulation components (phase-field driver)

- **Streaming**: `stream_axisym_inlet_2d!` — specular reflection at axis
  (``j = 1``), wall at far field (``j = N_r``)
- **Inlet**: Zou-He velocity BC for ``f_q``, equilibrium BC for ``g_q``
- **Outlet**: Zou-He pressure for ``f_q``, zero-gradient extrapolation for ``g_q``
- **Interface**: conservative Allen-Cahn with antidiffusion flux
- **Surface tension**: ``\mathbf{F} = \mu\,\nabla\varphi`` from chemical potential
  (+ azimuthal correction ``\kappa/r\,\partial\varphi/\partial r``)
- **Collision**: pressure-based MRT with variable viscosity ``\nu(\varphi)``
- **Axisymmetric correction**: viscous term ``\nu/r \cdot \partial u_z / \partial r``

## Setup

### VOF-based driver (moderate density ratio)

```julia
using KrakeNN

Re = 200
We = 600
δ = 0.02
R0 = 40
u_lb = 0.04

ν_l = u_lb * R0 / Re
σ_lb = u_lb^2 * R0 / We
τ = 3ν_l + 0.5

println("LBM parameters:")
println("  ν = $ν_l, τ = $τ, σ = $σ_lb")
```

The VOF driver uses geometric interface tracking (PLIC) and is limited to
moderate density ratios:

```julia
result_vof = run_cij_jet_axisym_2d(;
    Re=Re, We=We, δ=δ,
    R0=R0, u_lb=u_lb,
    domain_ratio=40, nr_ratio=3,
    ρ_ratio=10.0, μ_ratio=10.0,
    max_steps=10_000, output_interval=2000,
    output_dir=joinpath(@__DIR__, "cij_vof_output"))

println("VOF driver — breakup: ", result_vof.breakup_detected)
```

### Phase-field driver (high density ratio)

The phase-field driver handles ``\rho_l/\rho_g = 1000`` using the
pressure-based formulation where distributions stay ``O(1)``:

```julia
result_pf = run_cij_jet_phasefield_2d(;
    Re=Re, We=We, δ=δ,
    R0=R0, u_lb=u_lb,
    domain_ratio=40, nr_ratio=3,
    ρ_ratio=1000.0, μ_ratio=10.0,
    W_pf=5.0, τ_g=0.6,
    max_steps=10_000, output_interval=2000,
    output_dir=joinpath(@__DIR__, "cij_pf_output"))

println("Phase-field driver — breakup: ", result_pf.breakup_detected)
```

## Load Basilisk reference

We load the Basilisk interface data for the same (Re, δ) at a matching
physical time.

```julia
basilisk_dir = "/Users/guillaume/Documents/Recherche/Rheodrop/data/numerical/ds_num"
t_phys = 10_000 * u_lb / R0

bas_file = find_basilisk_snapshot(basilisk_dir, Re, δ, 155.0; tol=1.0)
if bas_file !== nothing
    bas_contour = load_basilisk_interface_contour(bas_file)
    println("Basilisk interface: $(length(bas_contour)) points")
end
```

## Comparison

The key validation metrics are:

1. **Jet morphology**: wavelength of the perturbation, satellite drop
   formation pattern
2. **Breakup length**: axial distance from the nozzle to the first
   pinch-off point
3. **Drop size and spacing**: regularity of the main drops

With the phase-field driver at ``\rho_l/\rho_g = 1000``, KrakeNN
matches the Basilisk density ratio exactly, enabling direct quantitative
comparison of interface shapes and breakup dynamics.

```julia
nothing  # suppress REPL output
```

