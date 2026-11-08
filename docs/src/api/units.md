# Units API

The `KrakeNN.Units` submodule converts nondimensional problem specifications into lattice-unit simulation plans, audits stability and boundary-condition compatibility, and exposes helper reports for driver integration.

```@autodocs
Modules = [KrakeNN.Units]
Pages = [
    "units/Units.jl",
    "units/audit_trail.jl",
    "units/physics_registry.jl",
    "units/lattice_units.jl",
    "units/stability_cone.jl",
    "units/stl_audit.jl",
    "units/bc_consistency.jl",
    "units/report.jl",
    "units/krk_binding.jl",
    "units/physics/newtonian.jl",
    "units/physics/viscoelastic.jl",
    "units/physics/thermal.jl",
    "units/physics/non_newt.jl",
    "units/physics/multiphase.jl",
    "units/physics/electromagn.jl",
]
Order = [:module, :constant, :type, :function]
```
