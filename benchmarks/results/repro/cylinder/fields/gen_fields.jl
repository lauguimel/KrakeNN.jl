# Generate 2D velocity fields for the flow-past-cylinder case (Re=20).
# Runs the examples/cylinder.krk case and writes ux/uy CSVs (Nx rows x Ny cols).
using KrakeNN
using DelimitedFiles

const HERE = @__DIR__
const KRK = joinpath(@__DIR__, "..", "..", "..", "..", "..", "examples", "cylinder.krk")

function main()
    @info "Running cylinder case" KRK
    r = run_simulation(KRK; max_steps=8000)
    println("keys(r) = ", keys(r))
    writedlm(joinpath(HERE, "field_ux.csv"), r.ux, ",")
    writedlm(joinpath(HERE, "field_uy.csv"), r.uy, ",")
    # Domain geometry for axes (L=10 x H=2.5, cylinder at cx=2.5 cy=1.25 R=0.5)
    open(joinpath(HERE, "geometry.csv"), "w") do io
        println(io, "Lx,Ly,cx,cy,R")
        println(io, "10.0,2.5,2.5,1.25,0.5")
    end
    @info "Wrote CSVs" size(r.ux)
end

main()
