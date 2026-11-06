# Reproduce channel_mass.csv, channel_orders.csv and channel_flux_orders.csv
# from permanent short tests.
# Run: julia --startup-file=no --project benchmarks/fv_c5/channel_mass.jl
# Baseline: 2026-11-05, macOS arm64, Julia1.12.5, CPU Float64 Stokes.
# Each testset is bounded to short unit scale; no production GPU campaign.
using Test,Kraken
ENV["KRAKEN_WRITE_C5_CHANNEL_EVIDENCE"] = "true"
include(joinpath(@__DIR__,"../../test/analytical/incns_fv_immersed_channel.jl"))
