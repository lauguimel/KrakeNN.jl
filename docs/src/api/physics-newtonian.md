# Newtonian Physics API

Newtonian physics APIs include the standard BGK/MRT/TRT flow drivers, Li/Bouzidi boundary precomputation, Shan-Chen and VOF/phase-field multiphase helpers, species transport, and generalized-Newtonian model entry points.

```@autodocs
Modules = [KrakeNN]
Pages = [
    "drivers/basic.jl",
    "drivers/cylinder_libb.jl",
    "drivers/axisymmetric.jl",
    "drivers/multiphase.jl",
    "drivers/rheology.jl",
    "kernels/li_bb_2d.jl",
    "kernels/li_bb_2d_v2.jl",
    "kernels/li_bb_3d_v2.jl",
    "kernels/drag_gpu.jl",
    "kernels/species_2d.jl",
    "kernels/multiphase_2d.jl",
    "kernels/vof_2d.jl",
    "kernels/dualgrid_2d.jl",
    "kernels/phasefield_2d.jl",
    "kernels/pressure_vof_2d.jl",
    "kernels/smooth_vof_2d.jl",
    "kernels/ghost_fluid_2d.jl",
    "kernels/advect_prescribed_2d.jl",
    "kernels/collide_twophase_rheology_2d.jl",
    "rheology/models.jl",
    "rheology/viscosity.jl",
    "rheology/strain_rate.jl",
]
Order = [:constant, :type, :function]
```
