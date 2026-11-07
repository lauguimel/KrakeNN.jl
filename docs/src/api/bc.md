# Boundary Condition API

Boundary-condition APIs include direct Zou-He and bounce-back kernels, spatially varying inlet/outlet wrappers, modular rebuild specs, and cut-link precomputation for interpolated bounce-back boundaries.

On the east face of the 2D rebuild path (`apply_bc_rebuild_2d!`), `NeqExtrapolationPressure(ρ_out)` imposes density and a zero normal gradient of velocity in the raw-moment sense: it copies `Σ c f / Σ f` from the completed post-collision column `Nx-1`. The log-FV drivers use TRT Guo forcing with raw-moment velocity, without a `+F/(2ρ)` correction; the outlet applies no additional force at `Nx`. Like the Zou-He x-face rebuilds, it includes end rows next to `HalfwayBB` or `InterfaceBC` south/north faces and excludes an end row next to a Zou-He face. At half-way walls, the copied neighbour already includes wall bounce-back before collision, so no additional wall-link treatment is needed. Place the outlet several obstacle diameters downstream. Backflow is not supported.

Pressure outlets on the west, south, and north faces of the 2D STL `wall = libb` path retain textbook `ZouHePressure` and its regularized-rebuild warning; in the measured viscoelastic channel, the new east outlet's error changes are confined to the last few columns.

```@autodocs
Modules = [KrakeNN]
Pages = [
    "kernels/boundary_2d.jl",
    "kernels/boundary_3d.jl",
    "kernels/boundary_rebuild.jl",
    "kernels/boundary_spatial_2d.jl",
    "kernels/li_bb_2d.jl",
    "kernels/li_bb_2d_v2.jl",
    "kernels/li_bb_3d_v2.jl",
]
Order = [:constant, :type, :function]
```
