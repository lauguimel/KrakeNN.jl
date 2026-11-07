# Per-T wrapper for Aqua PBS: one job = one T value (parallel sweep).
# Parameters via environment (qsub -v T_VAL=...,CYCLES=...): avoids the
# qsub -v comma-splitting and the #PBS ${} expansion limitations.
# Top-level loads, BEFORE KrakeNN (loaded by tc_sweep's include): the inner
# `@eval using` path is world-age fragile, and loading CUDSS after KrakeNN can
# leave KrakeNNCUDSSExt untriggered on Julia 1.12 (silent CPU fallback).
using CUDA, CUDSS

tval = get(ENV, "T_VAL", "190")
cycles = get(ENV, "CYCLES", "600000")
phi_scheme = get(ENV, "PHI_SCHEME", "lbm")
force_projection = get(ENV, "FORCE_PROJECTION", "none")
growth_window = get(ENV, "GROWTH_WINDOW", "0.4")

empty!(ARGS)
append!(ARGS, [
    "--ns-scheme=mrt",
    "--grid=197x321",
    "--T=$(tval)",
    "--cycles=$(cycles)",
    "--phi-scheme=$(phi_scheme)",
    "--force-projection=$(force_projection)",
    "--growth-window=$(growth_window)",
    "--gpu",
])
include(joinpath(@__DIR__, "..", "benchmarks", "ehd", "tc_sweep.jl"))
main(ARGS)
