#!/bin/bash -l
# Login-node setup of ~/kraken_c2 (network available there, not on the compute
# node): private depot, env with Kraken dev'd by path from src_c2 (a git
# archive of feat/fv-c2), instantiate + precompile, CUDA runtime precompiled.
# Seed the depot from a previous job folder to avoid re-downloading CUDA
# artifacts (optional):  cp -a ~/kraken_c1/depot ~/kraken_c2/depot
cd ~/kraken_c2 || exit 2
export JULIA_DEPOT_PATH=$HOME/kraken_c2/depot
export JULIA_CUDA_USE_COMPAT=false
export PATH=$HOME/.juliaup/bin:$PATH
mkdir -p env
cat > env/Project.toml <<'TOML'
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
KrakeNN = "b9c8ae56-6711-4442-a2ee-93cdf3c40864"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
CUDA = "6"
CUDSS = "1"
TOML
date
julia --project=env -e 'using Pkg; Pkg.develop(path="src_c2"); Pkg.resolve(); Pkg.instantiate(); Pkg.precompile(); using CUDA, CUDSS, ForwardDiff, KrakeNN; println("versions CUDA ", pkgversion(CUDA), " CUDSS ", pkgversion(CUDSS), " ForwardDiff ", pkgversion(ForwardDiff), " KA ", pkgversion(KrakeNN.KernelAbstractions), " SMC ", pkgversion(KrakeNN.SparseMatrixColorings)); CUDA.precompile_runtime()'
echo "SETUP EXIT $?"
date
