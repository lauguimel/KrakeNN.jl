# Kraken.jl quickstart: the README shows this file verbatim and CI runs it on CPU.
# From a clone of the repository:
#
#     julia --project=. examples/quickstart.jl
using Kraken

# 1. Run a lid-driven cavity described by a .krk configuration file.
#    `max_steps` overrides the file's `Run 60000 steps` so the script finishes
#    in a few seconds on a laptop CPU.
krk_file = joinpath(@__DIR__, "cavity.krk")
result = run_simulation(krk_file; max_steps = 2000)
println("krk cavity : max |ux| = ", maximum(abs, result.ux))

# 2. The same kind of case built from Julia, without the .krk file.
config = LBMConfig(D2Q9(); Nx = 32, Ny = 32, ν = 0.1, u_lid = 0.1, max_steps = 100)
cavity = run_cavity_2d(config)
println("julia cavity: max |ux| = ", maximum(abs, cavity.ux))

# Sanity checks: the script fails (non-zero exit) if the flow blew up.
@assert all(isfinite, result.ux) && all(isfinite, result.uy)
@assert all(isfinite, cavity.ux) && all(isfinite, cavity.uy)
@assert maximum(abs, cavity.ux) > 0
println("quickstart OK")
