# Viscoelastic duct slice (Reiner-Rivlin stress, Psi2)

`slice_forward.jl` solves the secondary flow of a Giesekus-like fluid in a
square duct on one periodic slice (Nx = 1) of the 3D steady FV Newton solver,
with the algebraic stress `tau = 2 eta(gdot) D + 4 Psi2(gdot) D.D`
(`FVReinerRivlinViscosity`, `src/methods/inc_ns/fv_second_normal_3d.jl`).
Case, environment variables and outputs are listed in the script header.
The CPU tests are `test/analytical/incns_fv3d_reiner_rivlin_test.jl`.

`ve2_law_fitted_g.csv` holds the jointly fitted correction `c(log10 Wi)`
(column `jointD1_r0.0001_c`, one law for De 0.1-1.0) and the exact Giesekus
`g` on the same nodes; `VE_LAW=fitted_g` reads it.

Reference (RheoTool Giesekus, graded 200 x 200 mesh): max |u_xy| / U =
1.3e-3 at De 0.1 and 3.1e-3 at De 1.

## Aqua

Job folder `~/kraken_ve_slice` with `src/` a copy of this worktree (the
changes are uncommitted: copy the tree, not a `git archive` of a commit).
Every job: `-P UXJFSK0637`, private depot in the job folder, environment set
up inside the job (never on the login node), the Julia binary called
directly, `JULIA_CUDA_USE_COMPAT=false`, GPU host pinned to gpu0n009 or
gpu0n005 (gpu0n001 fails cuInit).

CPU test job (`test_rr.pbs`):

```bash
#!/bin/bash -l
#PBS -N rr_test
#PBS -P UXJFSK0637
#PBS -l select=1:ncpus=8:mem=32GB
#PBS -l walltime=02:00:00
cd ~/kraken_ve_slice
export JULIA_DEPOT_PATH=$HOME/kraken_ve_slice/depot
export JULIA_NUM_THREADS=8
JL=$(ls -d $HOME/.julia/juliaup/julia-1.11*/bin/julia | tail -1)
cd src
KRAKEN_ONLY=analytical/incns_fv3d_reiner_rivlin_test.jl \
  $JL --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()' > ../test_rr.log 2>&1
echo "TEST EXIT $?" >> ../test_rr.log
```

GPU driver job (`slice_gpu.pbs`, N = 100, De 0.1 then 0.3, 0.6, 1.0 by
continuation, both laws):

```bash
#!/bin/bash -l
#PBS -N rr_slice
#PBS -P UXJFSK0637
#PBS -l select=1:ncpus=4:ngpus=1:mem=32GB:gpu_id=H100:host=gpu0n009
#PBS -l walltime=01:30:00
cd ~/kraken_ve_slice
export JULIA_DEPOT_PATH=$HOME/kraken_ve_slice/depot
export JULIA_CUDA_USE_COMPAT=false
JL=$(ls -d $HOME/.julia/juliaup/julia-1.11*/bin/julia | tail -1)
mkdir -p env
cat > env/Project.toml <<'TOML'
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
KrakeNN = "b9c8ae56-6711-4442-a2ee-93cdf3c40864"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
TOML = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[compat]
CUDA = "6"
CUDSS = "1"
TOML
$JL --project=env -e 'using Pkg; Pkg.develop(path="src"); Pkg.instantiate(); Pkg.precompile(); using CUDA; CUDA.precompile_runtime()' > setup.log 2>&1
for law in giesekus_closed_form fitted_g; do
  VE_BACKEND=cuda VE_N=100 VE_DE=0.1,0.3,0.6,1.0 VE_LAW=$law VE_OUT=$HOME/kraken_ve_slice/results \
    $JL --project=env src/benchmarks/ve_duct/slice_forward.jl > slice_$law.log 2>&1
  echo "RUN $law EXIT $?" >> slice_$law.log
done
```

Smoke run on CPU (tiny grid): `VE_BACKEND=cpu VE_N=12 VE_DE=0.1 julia --project benchmarks/ve_duct/slice_forward.jl`.
Not run yet.
