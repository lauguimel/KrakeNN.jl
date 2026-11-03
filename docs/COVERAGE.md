# Documentation Coverage Matrix

Last updated: 2026-04-03

## Theory Pages

| # | Module | Source kernel | Theory page | Status |
|---|--------|-------------|-------------|--------|
| 01 | LBM fundamentals | — | `theory/01_lbm_fundamentals` | Done |
| 02 | D2Q9 lattice | `lattice/d2q9.jl` | `theory/02_d2q9_lattice` | Done |
| 03 | BGK collision | `collide_stream_2d.jl` | `theory/03_bgk_collision` | Done |
| 04 | Streaming | `collide_stream_2d.jl` | `theory/04_streaming` | Done |
| 05 | Boundary conditions | `boundary_2d.jl` | `theory/05_boundary_conditions` | Done |
| 06 | 2D to 3D | `collide_stream_3d.jl` | `theory/06_from_2d_to_3d` | Done |
| 07 | Body forces (Guo) | `collide_guo_2d.jl` | `theory/07_body_forces` | Done |
| 08 | Thermal DDF | `thermal_2d.jl` | `theory/08_thermal_ddf` | Done |
| 09 | Axisymmetric | `simulation.jl` | `theory/09_axisymmetric` | Done |
| 10 | Limitations | — | `theory/10_limitations` | Done |
| 11 | Phase-field | `phasefield_2d.jl` | `theory/11_phasefield` | Done |
| 12 | MRT collision | `collide_mrt_2d.jl` | `theory/12_mrt` | Done |
| 13 | VOF-PLIC | `vof_2d.jl` | `theory/13_vof_plic` | Done |
| 14 | Rheology (GNF) | `rheology/*.jl` | `theory/14_rheology` | Done |
| 15 | Viscoelastic | `viscoelastic_2d.jl` | `theory/15_viscoelastic` | Done |
| 16 | Shan-Chen | `multiphase_2d.jl` | `theory/16_shan_chen` | Done |
| 17 | Species transport | `species_2d.jl` | `theory/17_species` | Done |
| 18 | Grid refinement | `refinement/*.jl` | `theory/18_grid_refinement` | Under development (not validated, unlisted) |
| 19 | Spatial BCs | `boundary_spatial_2d.jl` | `theory/19_spatial_bcs` | Done |

## Examples

| # | Example | Source kernel | Page | Status |
|---|---------|-------------|------|--------|
| 01 | Poiseuille 2D | `collide_guo_2d.jl` | `examples/01_poiseuille_2d` | Done |
| 02 | Couette 2D | `collide_stream_2d.jl` | `examples/02_couette_2d` | Done |
| 03 | Taylor-Green 2D | `collide_stream_2d.jl` | `examples/03_taylor_green_2d` | Done |
| 04 | Cavity 2D | `boundary_2d.jl` | `examples/04_cavity_2d` | Done |
| 05 | Cavity 3D | `boundary_3d.jl` | `examples/05_cavity_3d` | Done |
| 06 | Cylinder 2D | `boundary_2d.jl` | `examples/06_cylinder_2d` | Done |
| 07 | Heat conduction | `thermal_2d.jl` | `examples/07_heat_conduction` | Done |
| 08 | Rayleigh-Bénard | `thermal_2d.jl` | `examples/08_rayleigh_benard` | Done |
| 09 | Hagen-Poiseuille | axisym kernel | `examples/09_hagen_poiseuille` | Done |
| 10 | .krk config | `kraken_parser.jl` | `examples/10_krk_config` | Done |
| 11 | Zalesak disk | `advect_prescribed_2d.jl` | `examples/11_zalesak_disk` | Done |
| 12 | Reversed vortex | `advect_prescribed_2d.jl` | `examples/12_reversed_vortex` | Done |
| 13 | Capillary wave | `vof_2d.jl` | `examples/13_capillary_wave` | Done |
| 14 | Static droplet | `vof_2d.jl` | `examples/14_static_droplet` | Done |
| 15 | Rayleigh-Plateau | `vof_2d.jl` | `examples/15_rp_axisym` | Done |
| 16 | CIJ jet | `phasefield_2d.jl` | `examples/16_cij_jet` | Done |
| 17 | Rheology Poiseuille | `collide_rheology_2d.jl` | `examples/17_poiseuille_rheology` | Done |
| 18 | Shan-Chen spinodal | `multiphase_2d.jl` | `examples/18_shan_chen_spinodal` | Done |
| 19 | Species diffusion | `species_2d.jl` | `examples/19_species_diffusion` | Done |
| 20 | Grid refinement cavity | `refinement/*.jl` | `examples/20_grid_refinement_cavity` | Under development (not validated, unlisted) |
| 21 | Viscoelastic channel | `viscoelastic_2d.jl` | `examples/21_viscoelastic_channel` | Done |

## Benchmarks

| Benchmark | Page | Status |
|-----------|------|--------|
| MLUPs CPU/GPU | `benchmarks/mlups_cpu_gpu` | Done |
| Mesh convergence | `benchmarks/mesh_convergence` | Done |
| Comparison OpenFOAM | `benchmarks/comparison_openfoam` | Done |
| Rheology vs RheoTool | — | **Missing** |
| Multiphase vs Basilisk | — | **Missing** |
| Thermal convergence | — | **Missing** |
| GPU performance scaling | — | **Missing** |

## Test Coverage

| Kernel file | Test file(s) | Status |
|-------------|-------------|--------|
| `collide_stream_2d.jl` | test_lbm_basic, test_cavity | OK |
| `collide_stream_3d.jl` | test_cavity_3d | OK |
| `collide_guo_2d.jl` | test_poiseuille | OK |
| `collide_guo_3d.jl` | test_poiseuille_3d | OK |
| `collide_mrt_2d.jl` | test_mrt | OK |
| `collide_rheology_2d.jl` | test_rheology | OK |
| `collide_twophase_rheology_2d.jl` | test_twophase_rheology | OK |
| `thermal_2d.jl` | test_thermal | OK |
| `fused_thermal_2d.jl` | test_thermal (indirect) | Partial |
| `species_2d.jl` | test_species | OK |
| `multiphase_2d.jl` | test_multiphase | OK |
| `vof_2d.jl` | test_vof | OK |
| `phasefield_2d.jl` | test_phasefield | OK |
| `viscoelastic_2d.jl` | test_viscoelastic | OK |
| `boundary_spatial_2d.jl` | test_simulation_runner (indirect) | Partial |
| `advect_prescribed_2d.jl` | test_advection_prescribed | OK |
| `refinement_exchange_2d.jl` | test_refinement | OK |
| `dualgrid_2d.jl` | test_vof (partial) | Partial |
| `postprocess.jl` | test_postprocess | OK |
