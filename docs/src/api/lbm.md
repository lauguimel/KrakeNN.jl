# LBM API

The LBM API documents the core lattice traits, equilibrium helpers, collision and streaming wrappers, macroscopic reconstruction, forcing, MRT/TRT variants, and runtime kernel-building utilities.

```@autodocs
Modules = [KrakeNN]
Pages = [
    "lattice/lattice.jl",
    "lattice/d2q9.jl",
    "lattice/d3q19.jl",
    "kernels/equilibrium_helpers.jl",
    "kernels/equilibrium_helpers_3d.jl",
    "kernels/collide_stream_2d.jl",
    "kernels/collide_stream_3d.jl",
    "kernels/stream_periodic_2d.jl",
    "kernels/collide_guo_2d.jl",
    "kernels/collide_guo_3d.jl",
    "kernels/macroscopic.jl",
    "kernels/collide_mrt_2d.jl",
    "kernels/fused_bgk_2d.jl",
    "kernels/fused_trt_2d.jl",
    "kernels/aa_bgk_2d.jl",
    "kernels/persistent_bgk_2d.jl",
    "kernels/dsl/lbm_spec.jl",
    "kernels/dsl/bricks.jl",
    "kernels/dsl/bricks_3d.jl",
    "kernels/dsl/lbm_builder.jl",
]
Order = [:constant, :type, :function]
```
