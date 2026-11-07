# Thermal Physics API

Thermal physics APIs provide the DDF temperature kernels, Boussinesq coupling, and natural-convection drivers for 2D and 3D workflows. Thermal refinement helpers are listed below for reference; Grid refinement (patch and conservative AMR) is under development and not validated; it is not part of the supported feature set.

```@autodocs
Modules = [KrakeNN]
Pages = [
    "kernels/thermal_2d.jl",
    "kernels/thermal_3d.jl",
    "kernels/fused_thermal_2d.jl",
    "drivers/thermal.jl",
    "refinement/thermal_refinement.jl",
    "refinement/refinement_3d.jl",
]
Order = [:constant, :type, :function]
```
