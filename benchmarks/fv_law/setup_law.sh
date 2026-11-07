#!/bin/bash -l
# Prepared only; run manually on the login node after copying src_law.
# Creates an isolated environment. Does not modify src_law/Project.toml.
set -euo pipefail
WORKDIR=${WORKDIR:-$HOME/kraken_law}
cd "$WORKDIR"
export JULIA_DEPOT_PATH=$WORKDIR/depot
export JULIA_CUDA_USE_COMPAT=false
export PATH=$HOME/.juliaup/bin:$PATH
mkdir -p env
cat > env/Project.toml <<'TOML'
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
Kraken = "b9c8ae56-6711-4442-a2ee-93cdf3c40864"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
CUDA = "6"
CUDSS = "1"
TOML
julia --startup-file=no --project=env -e 'using Pkg; Pkg.develop(path="src_law"); Pkg.instantiate(); Pkg.precompile(); using CUDA, CUDSS, Kraken; println("CUDA ", pkgversion(CUDA), " CUDSS ", pkgversion(CUDSS), " Kraken ", pkgversion(Kraken)); CUDA.precompile_runtime()'
echo "SETUP EXIT 0"
