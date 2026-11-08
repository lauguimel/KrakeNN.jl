# Generate 2D velocity + polymer-stress fields for the viscoelastic
# (Oldroyd-B) flow past a cylinder. Writes CSVs (Nx rows x Ny cols).
using KrakeNN
using DelimitedFiles

const HERE = @__DIR__

function main()
    @info "Running viscoelastic cylinder"
    # formulation=:logconf is the numerically-stable path (the :stress default
    # loses positive-definiteness and diverges). lambda set so Wi ~ 0.5 to grow
    # a visible viscoelastic wake.
    r = KrakeNN.run_viscoelastic_cylinder_2d(; Nx=240, Ny=80, radius=16,
                                             u_in=0.04, lambda=200.0,
                                             formulation=:logconf,
                                             max_steps=8000, avg_window=2000,
                                             FT=Float64)
    println("keys(r) = ", keys(r))
    println("Cd = ", r.Cd, "  Re = ", r.Re, "  Wi = ", r.Wi, "  beta = ", r.beta)
    writedlm(joinpath(HERE, "field_ux.csv"),       r.ux,       ",")
    writedlm(joinpath(HERE, "field_uy.csv"),       r.uy,       ",")
    writedlm(joinpath(HERE, "field_tau_p_xx.csv"), r.tau_p_xx, ",")
    writedlm(joinpath(HERE, "field_tau_p_yy.csv"), r.tau_p_yy, ",")
    # Domain geometry: cylinder at cx=Nx/4, cy=Ny/2, radius=16.
    open(joinpath(HERE, "geometry.csv"), "w") do io
        println(io, "Nx,Ny,cx,cy,R")
        println(io, "240,80,60,40,16")
    end
    @info "Wrote CSVs" size(r.ux)
end

main()
