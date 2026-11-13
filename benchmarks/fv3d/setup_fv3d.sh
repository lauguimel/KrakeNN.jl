#!/bin/bash -l
# Login-node setup of ~/kraken_fv3d (network available there, not on the compute
# node): private depot, env with KrakeNN dev'd by path from src_fv3d (a git
# archive of feat/fv-3d), instantiate + precompile, CUDA runtime precompiled.
# Seed the depot from a previous job folder to avoid re-downloading CUDA
# artifacts (optional):  cp -a ~/kraken_c4/depot ~/kraken_fv3d/depot
cd ~/kraken_fv3d || exit 2
export JULIA_DEPOT_PATH=$HOME/kraken_fv3d/depot
export JULIA_CUDA_USE_COMPAT=false
export PATH=$HOME/.juliaup/bin:$PATH
mkdir -p env
cat > env/Project.toml <<'TOML'
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
KrakeNN = "b9c8ae56-6711-4442-a2ee-93cdf3c40864"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
TOML = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[compat]
CUDA = "6"
CUDSS = "1"
TOML
date
julia --project=env -e 'using Pkg; Pkg.develop(path="src_fv3d"); Pkg.resolve(); Pkg.instantiate(); Pkg.precompile(); using CUDA, CUDSS, KrakeNN; println("versions CUDA ", pkgversion(CUDA), " CUDSS ", pkgversion(CUDSS), " KA ", pkgversion(KrakeNN.KernelAbstractions), " SMC ", pkgversion(KrakeNN.SparseMatrixColorings)); CUDA.precompile_runtime()'
echo "SETUP EXIT $?"
date
