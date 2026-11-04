#!/bin/bash
set -euo pipefail
export JULIA_CUDA_USE_COMPAT=false JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_CPU_TARGET=generic
export JULIA_LOAD_PATH='@:@stdlib' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
JULIA=${JULIA:-julia}
integrity() {
    test "$(git -C source rev-parse HEAD)" = "$(cat SOURCE_SHA)" &&
    test -z "$(git -C source status --porcelain)" &&
    sha256sum --quiet -c inputs.sha256 &&
    (cd source && sha256sum --quiet -c ../source.sha256)
}
finish() {
    rc=$?; trap - EXIT; set +e
    integrity; irc=$?
    find raw -type f -print0 | sort -z | xargs -0 -r sha256sum >outputs.sha256
    printf 'Execution exit: %s; integrity exit: %s\n' "$rc" "$irc"
    if [ "$rc" -eq 0 ] && [ "$irc" -eq 0 ]; then exit 0; else exit 1; fi
}
integrity
# Refuse reusing a completed/partial output directory; never overwrite evidence.
test -z "$(find raw -mindepth 1 -print -quit)"
trap finish EXIT
date -u; nvidia-smi; "$JULIA" --version
args=(--startup-file=no -O0 --compiled-modules=existing --pkgimages=existing --project=environment)
case "$(cat CASE)" in
    restart)
        export JULIA_NUM_THREADS=1
        timeout --signal=TERM --kill-after=10s 480s "$JULIA" "${args[@]}" calibrate.jl calibration
        sha256sum -c raw/tolerances.sha256
        timeout --signal=TERM --kill-after=10s 600s "$JULIA" "${args[@]}" check.jl verification
        ;;
    scalar)
        export JULIA_NUM_THREADS=1 CASE_STAGE=single
        timeout --signal=TERM --kill-after=10s 300s "$JULIA" "${args[@]}" verify.jl
        export JULIA_NUM_THREADS=2 CASE_STAGE=main
        timeout --signal=TERM --kill-after=10s 600s "$JULIA" "${args[@]}" verify.jl
        ;;
    *) echo 'Invalid CASE' >&2; exit 2;;
esac
