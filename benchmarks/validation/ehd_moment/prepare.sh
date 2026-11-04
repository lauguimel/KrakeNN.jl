#!/bin/bash
# Metadata/file preparation only; install dependencies separately before sealing.
set -euo pipefail
case_name=${1:?Usage: bash prepare.sh restart|scalar NEW_DIRECTORY [SOLVER_SHA]}
case "$case_name" in restart|scalar) ;; *) echo 'Invalid case' >&2; exit 2;; esac
package=$(cd -- "$(dirname -- "$0")" && pwd -P)
repo=$(git -C "$package" rev-parse --show-toplevel)
test -z "$(git -C "$repo" status --porcelain)" || { echo 'Use a clean committed checkout' >&2; exit 1; }
solver_sha=$(git -C "$repo" rev-parse "${3:-HEAD}^{commit}")
harness_sha=$(git -C "$repo" rev-parse HEAD)
(cd "$package" && sha256sum -c ORIGINAL_SHA256SUMS)
# mkdir without -p intentionally refuses to overwrite an existing run.
mkdir -- "${2:?New run directory required}"
run=$(cd -- "$2" && pwd -P)
git clone --quiet --no-hardlinks "$repo" "$run/source"
git -C "$run/source" checkout --quiet --detach "$solver_sha"
mkdir "$run/environment" "$run/raw" "$run/logs"
cp "$package/environment/Project.toml" "$package/environment/Manifest.toml" "$run/environment/"
cp "$package/$case_name/"*.jl "$package/$case_name/PROTOCOL.md" "$run/"
if [ "$case_name" = scalar ]; then
    cp -r "$package/scalar/reference" "$run/"
    cp "$package/scalar/ADMISSION_PROTOCOL.md" "$run/"
fi
cp "$package/run.sh" "$package/seal.sh" "$package/${case_name}.sub" "$run/"
printf '%s\n' "$solver_sha" >"$run/SOURCE_SHA"
printf '%s\n' "$harness_sha" >"$run/HARNESS_SHA"
printf '%s\n' "$case_name" >"$run/CASE"
echo "Prepared $run; instantiate pinned environment, then bash seal.sh before qsub."
