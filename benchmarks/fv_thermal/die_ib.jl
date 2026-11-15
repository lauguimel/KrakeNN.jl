# Die prediction with smooth immersed walls: the learned-law comparison of
# die_learned.jl (truth E5ThermalLaw, identified network, fitted Carreau-WLF)
# with the tapered die walls given as immersed profile walls
# (fv_immersed_walls + FVProfileWall, quadratic ghost, M3 mass, Dirichlet wall
# temperature through the same ghost closure) instead of voxel staircases.
#
# Geometry (lattice units, h = 1): inlet gap G between y = m and y = m + G,
# solid margins of m = 4 cells below and above (walls keep >= 3h from the
# S/N sides), Ny = G + 2m, Nx = 7G/2 (= 7H). Centre y_c = Ny/2. Walls
# y = y_c -+ gap(x)/2 with gap = G for x <= 0.75G, linear to G/4 at x = 2.25G
# (wall slope 1/4), then constant: the outlet gap G/4 = 16 cells at G = 64
# (>= 3h contract). The die geometry, the physical units and the laws are those
# of die_learned.jl (H = G/2 = 4 mm, same Pe, Na, T_w, T_in, speeds).
#
# Inlet: Poiseuille of mean U between the inlet walls, T_in. Outlet: E side.
# Walls: no-slip, T_w; M3 with zero physical wall flux.
#
# Environment: as die_learned.jl (KRAKEN_DIE_BACKEND, KRAKEN_DIE_NN_SUMMARY,
# KRAKEN_DIE_WLF_SUMMARY, KRAKEN_DIE_LAWS, KRAKEN_DIE_SPEEDS, KRAKEN_DIE_GRIDS,
# KRAKEN_DIE_G, KRAKEN_DIE_TW_C, KRAKEN_DIE_OUT), plus KRAKEN_DIE_MARGIN (m, 4).
# Per run the same files as die_learned.jl (summary.toml, outlet.csv,
# cells.csv, *.bin with Nx x Ny, column-major), plus walls.csv (wall vertices,
# lattice units) and the wall heat in summary.toml.
module DieIB

include(joinpath(@__DIR__, "die_learned.jl"))
using .DieLearned
using .DieLearned.DiePredict
using KrakeNN, LinearAlgebra, Printf, Statistics, TOML
const DL = DieLearned
const DP = DieLearned.DiePredict
if DP.BACKEND == "cuda"
    using CUDA
end

const H_OUT = 0.25
const X_START = 0.75
const X_END = 2.25

"""Die walls: (lower, upper) FVProfileWall for inlet gap G, centre yc, length Nx."""
function die_walls(G, yc, Nx)
    xs = [0.0, X_START * G, X_END * G, Float64(Nx)]
    half = [G / 2, G / 2, H_OUT * G / 2, H_OUT * G / 2]
    return (FVProfileWall(xs, yc .- half; solid=:below), FVProfileWall(xs, yc .+ half; solid=:above))
end

"""Physical case of die_learned.jl on the immersed geometry."""
function ib_case(G, U_mm, Tw_C; margin=parse(Int, get(ENV, "KRAKEN_DIE_MARGIN", "4")))
    c = DL.physical_case(G, U_mm, Tw_C)
    Ny = G + 2margin; yc = Ny / 2
    walls = die_walls(G, yc, c.Nx)
    ib = fv_immersed_walls(c.Nx, Ny; walls, h=1.0, closure=:quadratic, mass=:M3, pressure=:linear)
    solid = ib.tag .== FV_CELL_SOLID
    return merge(c, (; Ny, margin, yc, walls, ib, solid))
end

u_inlet(c, y) = (s = (y - c.margin) / c.G; 0 <= s <= 1 ? 6c.U * s * (1 - s) : 0.0)

function build_problem(c, visc)
    return fv_steady_problem(c.Nx, c.Ny; h=1.0, nu=c.nu_ref, conv=true,
        bc=(W=:inlet, E=:outlet, S=:wall, N=:wall), immersed=c.ib, wall_flux=0.0,
        boundary_velocity=(side, x, y) -> (side === :W ? u_inlet(c, y) : 0.0, 0.0),
        viscosity=visc,
        energy=FVEnergy(; alpha=c.alpha, sigma=c.heating, left=c.theta_in,
                        bottom=c.theta_wall, top=c.theta_wall, solid=c.theta_wall))
end

function initial_state(c, prob)
    fluid = .!c.solid
    q = sum(u_inlet(c, j - 0.5) for j in 1:c.Ny)
    u = [fluid[i, j] ? q / max(count(view(fluid, i, :)), 1) : 0.0 for i in 1:c.Nx, j in 1:c.Ny]
    z = zeros(c.Nx, c.Ny)
    return fv_state(prob, u, z, z, c.theta_in)
end

"""sigma = 0 solve then the heating ladder (copy of DiePredict.solve_die with the
immersed initial state)."""
function solve_ib(c, prob; maxit=60)
    tl = DP.tools(); records = Dict{String,Any}[]
    U = tl.to(initial_state(c, prob)); saved = copy(U)
    p0 = KrakeNN._fv_heating_problem(prob, 0.0)
    s = fv_newton_solver(p0; tl.to, tl.linsolve)
    ok = false
    for dtau0 in (nothing, 1.0)
        copyto!(U, saved); mem0 = DP.gpu_used()
        stats, secs = DP.timed(() -> fv_newton!(s, U; maxit, dtau0))
        rec = DP.newton_record(stats, 0.0, secs, mem0, DP.gpu_used()); rec["dtau0"] = string(dtau0)
        push!(records, rec)
        @info "sigma = 0" converged = stats.converged iterations = stats.iterations dtau0
        ok = stats.converged
        ok && break
    end
    ok || return U, s, records, nothing
    reached = 0.0; step = c.heating; halvings = 0
    while c.heating > 0 && reached < c.heating
        target = min(c.heating, reached + step)
        copyto!(saved, U)
        st = fv_newton_solver(KrakeNN._fv_heating_problem(prob, target); tl.to, tl.linsolve)
        mem0 = DP.gpu_used()
        stats, secs = DP.timed(() -> fv_newton!(st, U; maxit))
        push!(records, DP.newton_record(stats, target, secs, mem0, DP.gpu_used()))
        @info "heating step" sigma = target converged = stats.converged iterations = stats.iterations
        if stats.converged
            reached = target; s = st
        else
            copyto!(U, saved); halvings += 1
            halvings > 5 && break
            step /= 2
        end
    end
    return U, s, records, reached
end

"""Fields, dp, mixing-cup outlet temperature and the discrete energy balance
(theta = T - T_ref units of the thermal row; immersed walls through
fv_immersed_wall_heat, heat entering counts positive)."""
function diagnostics(c, hp, Uh)
    u, v, p, T = fv_fields(hp, Uh)
    fluid = .!c.solid; Nx, Ny = c.Nx, c.Ny
    col_mean(A, i) = mean(A[i, j] for j in 1:Ny if fluid[i, j])
    dp = col_mean(p, 1) - col_mean(p, Nx)
    mass_in = sum(u_inlet(c, j - 0.5) for j in 1:Ny if fluid[1, j])
    out = [j for j in 1:Ny if fluid[Nx, j]]
    mass_out = sum(u[Nx, j] for j in out)
    T_cup = sum(u[Nx, j] * T[Nx, j] for j in out) / mass_out
    Tref = hp.energy.Tref; a = c.alpha
    adv_in = sum(u_inlet(c, j - 0.5) for j in 1:Ny if fluid[1, j]) * (c.theta_in - Tref)
    cond_in = sum(2a * (c.theta_in - T[1, j]) for j in 1:Ny if fluid[1, j])
    adv_out = sum(u[Nx, j] * (T[Nx, j] - Tref) for j in out)
    wall_heat = fv_immersed_wall_heat(Uh, hp)
    heat = 0.0
    qcut, wvel, closure = hp.immersed.qcut, hp.immersed.wvel, hp.immersed.closure
    for j in 1:Ny, i in 1:Nx
        fluid[i, j] || continue
        heat += KrakeNN._fv_energy_heating(Uh, hp.tag, hp.nu, hp.bvel, hp.grid, i, j,
                                           hp.visc, hp.energy.sigma, qcut, wvel, closure)
    end
    terms = Dict{String,Any}("advection_in" => adv_in, "conduction_in" => cond_in,
        "advection_out" => adv_out, "immersed_walls" => -sum(wall_heat), "viscous_heating" => heat)
    imbalance = adv_in + cond_in - adv_out - sum(wall_heat) + heat
    return (; u, v, p, T, dp, mass_in, mass_out, T_cup, wall_heat,
            balance=Dict{String,Any}("terms" => terms, "imbalance" => imbalance,
                "relative" => abs(imbalance) / maximum(abs, values(terms))),
            theta_min=minimum(T[fluid]), theta_max=maximum(T[fluid]))
end

function run_one(law, c, th; outdir)
    mkpath(outdir)
    prob = build_problem(c, DL.viscosity(law, c, th))
    (U, s, records, reached), secs = DP.timed(() -> solve_ib(c, prob))
    converged = reached !== nothing && reached == c.heating
    Uh = Array(U)
    hp = s.prob.nu isa Array ? s.prob : fv_adapt(Array, s.prob)
    d = diagnostics(c, hp, Uh)
    fluid = .!c.solid; Nx, Ny = c.Nx, c.Ny
    out = [j for j in 1:Ny if fluid[Nx, j]]
    umean_out = mean(d.u[Nx, j] for j in out)
    # outlet column: y measured from the physical lower outlet wall
    ylo = c.yc - H_OUT * c.G / 2; hout = H_OUT * c.G
    open(joinpath(outdir, "outlet.csv"), "w") do io
        println(io, "j,y_lat,y_over_hout,u_over_umean,theta_star,u_lat,T_K")
        for j in out
            @printf(io, "%d,%.6f,%.8f,%.12g,%.12g,%.12g,%.12g\n", j, j - 0.5, (j - 0.5 - ylo) / hout,
                    d.u[Nx, j] / umean_out, (d.T[Nx, j] - c.Tw) / (c.Tin - c.Tw), d.u[Nx, j], d.T[Nx, j])
        end
    end
    b = hp.immersed
    open(joinpath(outdir, "cells.csv"), "w") do io
        println(io, "i,j,gdot,T,diss_truth")
        for j in 1:Ny, i in 1:Nx
            fluid[i, j] || continue
            gP = KrakeNN._fv_cell_grad(Uh, hp.tag, hp.bvel, hp.grid, i, j, b.qcut, b.wvel, b.closure)
            ux, uy, vx, vy = gP[1], gP[2], gP[3], gP[4]
            sxy = (uy + vx) / 2
            g = c.s * sqrt(2 * (ux^2 + vy^2 + 2 * sxy^2))
            @printf(io, "%d,%d,%.8g,%.8g,%.8g\n", i, j, g, d.T[i, j], DL.truth_eta(g, d.T[i, j]) * g^2)
        end
    end
    open(joinpath(outdir, "walls.csv"), "w") do io
        println(io, "wall,x_lat,y_lat")
        for (k, w) in enumerate(c.walls), l in eachindex(w.x)
            @printf(io, "%d,%.12g,%.12g\n", k, w.x[l], w.y[l])
        end
    end
    for (name, A) in (("ux", d.u), ("uy", d.v), ("p", d.p), ("theta", d.T), ("is_solid", c.solid))
        DP.writebin(joinpath(outdir, name * ".bin"), A)
    end
    summary = Dict{String,Any}("law" => string(law), "G" => c.G, "Nx" => Nx, "Ny" => Ny,
        "margin" => c.margin, "geometry" => "immersed_profile_walls", "U_mm_s" => c.U_mm,
        "Tw_K" => c.Tw, "Tin_K" => c.Tin, "Pe" => c.Pe, "Na" => c.Na, "shear_scale" => c.s,
        "eta_ref" => c.eta_ref, "converged" => converged,
        "sigma_reached" => reached === nothing ? -1.0 : reached,
        "sigma_target" => c.heating, "total_seconds" => secs, "newton" => records,
        "dp_lat" => d.dp, "dp_star" => d.dp / c.dp_scale,
        "dp_Pa" => d.dp / c.dp_scale * c.eta_ref * c.U_mm * 1e-3 / DL.H_PHYS,
        "T_cup_out_K" => d.T_cup, "theta_cup_out" => (d.T_cup - c.Tw) / (c.Tin - c.Tw),
        "T_max_K" => d.theta_max, "T_min_K" => d.theta_min,
        "mass_balance_relative" => abs(d.mass_in - d.mass_out) / d.mass_in,
        "energy_balance_relative" => d.balance["relative"], "energy_balance" => d.balance,
        "wall_heat_theta_units" => collect(d.wall_heat), "umean_out_lat" => umean_out,
        "n_unknowns" => fv_nunknowns(prob), "m3_pairs" => KrakeNN._fv_npairs(prob.immersed))
    DP.BACKEND == "cuda" && (summary["device"] = CUDA.name(CUDA.device()))
    open(joinpath(outdir, "summary.toml"), "w") do io
        TOML.print(io, summary; sorted=true)
    end
    @info "run done" law c.G c.U_mm converged dp_star = summary["dp_star"] theta_cup = summary["theta_cup_out"] balance = d.balance["relative"] secs
    return summary
end

function main()
    root = get(ENV, "KRAKEN_DIE_OUT", joinpath(pwd(), "die_ib_out")); mkpath(root)
    th = Dict{Symbol,Vector{Float64}}(:truth => DL.E5_TRUTH)
    nnf = get(ENV, "KRAKEN_DIE_NN_SUMMARY", ""); isempty(nnf) || (th[:nn] = DL.read_theta(nnf))
    wf = get(ENV, "KRAKEN_DIE_WLF_SUMMARY", ""); isempty(wf) || (th[:carreau_wlf] = DL.read_theta(wf))
    laws = Symbol.(split(get(ENV, "KRAKEN_DIE_LAWS", join(string.(filter(k -> haskey(th, k), [:truth, :nn, :carreau_wlf])), ",")), ","))
    speeds = parse.(Float64, split(get(ENV, "KRAKEN_DIE_SPEEDS", "1,1.43,2.03,2.9,4.14,5.9,8.42,12"), ","))
    Tw_C = parse(Float64, get(ENV, "KRAKEN_DIE_TW_C", "175"))
    G0 = parse(Int, get(ENV, "KRAKEN_DIE_G", "64"))
    jobs = [(G0, k) for k in eachindex(speeds)]
    for p in split(get(ENV, "KRAKEN_DIE_GRIDS", ""), ","; keepempty=false)
        g, k = parse.(Int, split(p, ":")); push!(jobs, (g, k))
    end
    @info "die immersed-wall sweep" laws speeds Tw_C jobs
    rows = Dict{String,Any}[]
    for (G, k) in jobs, law in laws
        dir = joinpath(root, @sprintf("%s_G%d_U%02d", law, G, k))
        r = try
            c = ib_case(G, speeds[k], Tw_C)
            run_one(law, c, get(th, law, DL.E5_TRUTH); outdir=dir)
        catch err
            @error "run failed" law G k exception = (err, catch_backtrace())
            Dict{String,Any}("law" => string(law), "G" => G, "U_mm_s" => speeds[k], "converged" => false,
                             "dp_star" => NaN, "theta_cup_out" => NaN, "total_seconds" => NaN)
        end
        r["k"] = k; push!(rows, r)
        open(joinpath(root, "runs.csv"), "w") do io
            println(io, "law,G,k,U_mm_s,converged,dp_star,theta_cup_out,seconds")
            for x in rows
                @printf(io, "%s,%d,%d,%.6g,%s,%.12g,%.12g,%.3f\n", x["law"], x["G"], x["k"], x["U_mm_s"],
                        x["converged"], x["dp_star"], x["theta_cup_out"], x["total_seconds"])
            end
        end
        GC.gc(); DP.BACKEND == "cuda" && CUDA.reclaim()
    end
    println(read(joinpath(root, "runs.csv"), String))
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    DieIB.main()
end
