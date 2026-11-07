#!/bin/bash -l
# Prepared only: login-node setup of ~/kraken_c5 using the src_c5 snapshot.
# This script installs the environment; it is not run by the local session.
set -e
cd "$HOME/kraken_c5" || exit 2
export JULIA_DEPOT_PATH=$HOME/kraken_c5/depot
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
julia --startup-file=no --project=env -e 'using Pkg; Pkg.develop(path="src_c5"); Pkg.instantiate(); Pkg.precompile(); using CUDA, CUDSS, ForwardDiff, KrakeNN; println("versions CUDA ", pkgversion(CUDA), " CUDSS ", pkgversion(CUDSS), " ForwardDiff ", pkgversion(ForwardDiff), " KA ", pkgversion(KrakeNN.KernelAbstractions), " SMC ", pkgversion(KrakeNN.SparseMatrixColorings)); CUDA.precompile_runtime()'
echo "SETUP EXIT 0"
