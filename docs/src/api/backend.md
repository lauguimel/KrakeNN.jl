# Backend API

Backend APIs document the KernelAbstractions-facing execution helpers, fused kernels, runtime kernel DSL, persistent-kernel entry points, and GPU route-packing utilities that support CPU/GPU portability.

```@autodocs
Modules = [KrakeNN]
Pages = [
    "kernels/dsl/lbm_spec.jl",
    "kernels/dsl/bricks.jl",
    "kernels/dsl/bricks_3d.jl",
    "kernels/dsl/lbm_builder.jl",
    "kernels/fused_bgk_2d.jl",
    "kernels/fused_trt_2d.jl",
    "kernels/fused_thermal_2d.jl",
    "kernels/aa_bgk_2d.jl",
    "kernels/persistent_bgk_2d.jl",
    "kernels/drag_gpu.jl",
    "refinement/conservative_tree_gpu_pack_2d.jl",
]
Order = [:constant, :type, :function]
```
