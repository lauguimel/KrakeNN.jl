# Reproduce shape_geometry.csv, shape_fd_sweep.csv, shape_derivatives.csv
# and shape_orders.csv from the permanent fixed-topology shape tests.
# Run: julia --startup-file=no --project benchmarks/fv_c5/shape_derivative.jl
# Baseline: 2026-11-05, macOS arm64, Julia 1.12.5, CPU Float64 Stokes.
# Each testset is a short unit test (<30 seconds); no production GPU campaign.
using Test, KrakeNN
ENV["KRAKEN_WRITE_C5_SHAPE_EVIDENCE"] = "true"
ENV["KRAKEN_C5_SHAPE_GEOMETRY_ONLY"] = "false"
include(joinpath(@__DIR__,"../../test/analytical/incns_fv_immersed_shape.jl"))
