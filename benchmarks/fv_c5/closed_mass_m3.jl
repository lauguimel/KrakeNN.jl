# Regenerate the C5-M3 numerical evidence from the permanent tests, in the
# mandatory order: algebra/rank, discrete derivatives, closed-domain solves.
# Run: julia --startup-file=no --project benchmarks/fv_c5/closed_mass_m3.jl
# Each individual testset is a short CPU Float64 unit test (<30 seconds), not
# a production convergence campaign. No package installation or GPU job occurs.
# Measured baseline: 2026-11-05, macOS arm64, Julia 1.12.5, CPU Float64.
# CSV files are generated beside this script. Historical M1 is closed_mass.csv
# (regenerate with KRAKEN_WRITE_C5_MASS_EVIDENCE=true and the M1 test file).
using Test, Kraken
ENV["KRAKEN_WRITE_C5_M3_EVIDENCE"] = "true"
root = normpath(joinpath(@__DIR__,"..",".."))
for stage in ("algebra","adjoint","convergence")
    include(joinpath(root,"test","analytical","incns_fv_immersed_m3_"*stage*".jl"))
end
