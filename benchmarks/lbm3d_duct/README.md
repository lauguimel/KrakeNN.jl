# 3D LBM GNF duct with a voxel obstacle

`run_duct_gpu.jl` runs `run_gnf_duct_obstacle_3d` (D3Q19, TRT, Guo body force,
LI-BB walls, GNF viscosity) on a rectangular duct periodic in x, with an
optional solid obstacle, and exports the steady state as an independent
reference for a 3D finite-volume (FV) solver. Only a body force is imposed;
the flow rate is an output.

## Run

On a CUDA node (A100 or H100 on Aqua; this directory does not submit jobs):

```bash
export JULIA_CUDA_USE_COMPAT=false
DUCT_NX=256 DUCT_NY=128 DUCT_NZ=128 DUCT_LAW=carreau DUCT_F=1e-5 \
DUCT_STEPS=400000 DUCT_OUT=duct_256x128x128 \
julia --project benchmarks/lbm3d_duct/run_duct_gpu.jl
```

CPU check at a tiny size (seconds):

```bash
DUCT_BACKEND=cpu DUCT_NX=12 DUCT_NY=8 DUCT_NZ=8 DUCT_F=6e-5 DUCT_STEPS=3000 \
DUCT_OUT=tmp/duct_try julia --project benchmarks/lbm3d_duct/run_duct_gpu.jl
```

## Environment variables (defaults)

Grid `DUCT_NX/NY/NZ` (128/64/64); law `DUCT_LAW` = newtonian, carreau
(default), cross, powerlaw, with lattice parameters `DUCT_NU0` (0.1),
`DUCT_NUINF` (0.02), `DUCT_LAMBDA` (3000), `DUCT_A` (2), `DUCT_N` (0.5),
`DUCT_K` (0.1), `DUCT_NUMIN/NUMAX` (1e-3/1); force `DUCT_F` (1e-5); stopping
`DUCT_STEPS` (200000), `DUCT_TOL` (1e-9), `DUCT_CHECK_EVERY` (500); obstacle
`DUCT_OBSTACLE` = cuboid (default), sphere, none, with `DUCT_OBS_CENTER` and
`DUCT_OBS_SIZE` as fractions of the duct (cuboid: three side fractions; sphere:
first value = diameter fraction of min(Ly, Lz)); precision `DUCT_FT`
(Float64 or Float32); scaling `DUCT_H` (m) and `DUCT_DT` (s), default 1;
`DUCT_BACKEND` (cuda or cpu); `DUCT_OUT` (output directory).

## Output

Same layout as the 2D E5 export (`tmp/e5_export/<grid>/`): raw `ux.bin`,
`uy.bin`, `uz.bin`, `nu.bin`, `rho.bin`, `is_solid.bin` (1.0 = obstacle),
Float64 little-endian, column-major (x fastest), shape `(Nx, Ny, Nz)`, plus
`case.toml` (grid, law, obstacle, constants and scales, flow rate, mass-flux
spread, Mach, convergence). Read in Julia with
`reshape(reinterpret(Float64, read(path)), Nx, Ny, Nz)`.

## Units

The solver uses h = 1, dt = 1, rho = 1; the fields are in lattice units. A
physical value is the lattice value times the scale in `case.toml`: velocity
h/dt, kinematic viscosity h^2/dt, body force h/dt^2, shear rate 1/dt, flow rate
h^3/dt. To reproduce a physical law, give the lattice law `nu_lat =
nu_phys*dt/h^2` evaluated at `gamma_dot_lat = gamma_dot_phys*dt` (a Carreau
`lambda` becomes `lambda_phys/dt`). Walls are on the cell faces (q_w = 1/2):
the duct is Nx*h by Ny*h by Nz*h, cell (i,j,k) is centred at
((i-1/2)h, (j-1/2)h, (k-1/2)h). Obstacle voxels are a staircase with halfway
bounce-back. Keep the maximum lattice speed below about 0.05 (Mach =
`u_max*sqrt(3)` is in `case.toml`).

## Status

Verified on CPU only (tests in `test/analytical/gnf3d_duct_obstacle.jl`, the
tiny run above). No GPU run, no resolution study of the obstacle drag and no
comparison with an FV solution have been made.
