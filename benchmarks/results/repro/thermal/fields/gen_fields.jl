# Generate 2D temperature + velocity fields for the differentially heated
# square cavity (natural convection, Ra=1e4). Writes CSVs (Nx rows x Ny cols).
using KrakeNN
using DelimitedFiles

const HERE = @__DIR__

function main()
    @info "Running natural convection cavity" Ra=1e4
    r = KrakeNN.run_natural_convection_2d(; N=128, Ra=1e4, Pr=0.71,
                                          max_steps=40000, FT=Float64)
    println("keys(r) = ", keys(r))
    println("Nu = ", r.Nu)
    writedlm(joinpath(HERE, "field_T.csv"),  r.Temp, ",")
    writedlm(joinpath(HERE, "field_ux.csv"), r.ux,   ",")
    writedlm(joinpath(HERE, "field_uy.csv"), r.uy,   ",")
    @info "Wrote CSVs" size(r.Temp)
end

main()
