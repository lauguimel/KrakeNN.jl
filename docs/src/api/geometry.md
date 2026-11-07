# Geometry API

Geometry APIs cover structured curvilinear meshes, SLBM geometry transfer, multi-block topology and ghost exchange, Gmsh import, and STL-based obstacle preprocessing used by the LBM drivers.

```@autodocs
Modules = [KrakeNN]
Pages = [
    "curvilinear/mesh.jl",
    "curvilinear/generators.jl",
    "curvilinear/slbm.jl",
    "curvilinear/mesh_3d.jl",
    "curvilinear/slbm_3d.jl",
    "curvilinear/mesh_from_arrays.jl",
    "curvilinear/mesh_gmsh.jl",
    "multiblock/topology.jl",
    "multiblock/sanity.jl",
    "multiblock/state.jl",
    "multiblock/exchange.jl",
    "multiblock/wall_ghost.jl",
    "multiblock/mesh_gmsh_multiblock.jl",
    "multiblock/reorient.jl",
    "multiblock/mesh_extend.jl",
    "geometry/stl_reader.jl",
    "geometry/voxelizer.jl",
    "geometry/stl_cut_fraction.jl",
    "geometry/mask_apply.jl",
    "geometry/libb_precompute.jl",
    "geometry/descriptor.jl",
    "drivers/obstacle_3d.jl",
]
Order = [:constant, :type, :function]
```
