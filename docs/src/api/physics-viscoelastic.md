# Viscoelastic Physics API

This page documents the legacy viscoelastic and rheology API present on this branch: constitutive model types, viscosity and strain-rate helpers, legacy polymer-stress kernels, and the cylinder driver. The log-conformation finite-volume solver lives on a separate branch and is merge-pending.

```@autodocs
Modules = [KrakeNN]
Pages = [
    "rheology/models.jl",
    "rheology/viscosity.jl",
    "rheology/strain_rate.jl",
    "rheology/linalg.jl",
    "kernels/collide_rheology_2d.jl",
    "kernels/viscoelastic_2d.jl",
    "drivers/viscoelastic.jl",
]
Order = [:constant, :type, :function]
```
