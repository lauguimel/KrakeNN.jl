# Die prediction with the E5 viscosity laws: the truth law of the heated
# six-cylinder cell (E5ThermalLaw at its reference parameters), the network
# identified on that cell (2 -> 16 -> 16 -> 1 tanh, eta = E0 exp(NN(ln gdot, T)))
# and the fitted Carreau-WLF library law, each solved with the steady FV Newton
# (flow + temperature) of die_predict.jl in the tapered die.
#
# Physical set-up (same material constants as the E5 cell): the die half inlet
# gap equals the cell height H = 4 mm (inlet gap 2H, outlet gap H/2, length 7H),
# k = 0.2 W/(m K), rho cp = 2e6 J/(m^3 K), walls and die body at T_w, melt enters
# at T_in = T_w + 1/beta (beta = 0.035 1/K, i.e. theta = 1 at the inlet).
# Temperatures are solved in kelvin (FVEnergy reference shift), so every law sees
# absolute T and the physical shear rate s*gdot_lat (s = 1/dt).
# Unit choices frozen for every law (as in the E5 cell): eta_ref = eta_truth(0, T_w)
# is the viscosity that maps to nu_ref.
#
# Environment:
#   KRAKEN_DIE_BACKEND       cpu | cuda
#   KRAKEN_DIE_NN_SUMMARY    summary.txt of the network fit (line "theta=...")
#   KRAKEN_DIE_WLF_SUMMARY   summary.txt of the Carreau-WLF fit (optional)
#   KRAKEN_DIE_LAWS          comma list among truth,nn,carreau_wlf (default all available)
#   KRAKEN_DIE_SPEEDS        comma list of mean inlet speeds in mm/s
#   KRAKEN_DIE_GRIDS         comma list G:speed_index for extra grids (e.g. "128:5")
#   KRAKEN_DIE_TW_C          wall temperature in degC (default 175)
#   KRAKEN_DIE_OUT           output root
# Per run (law, G, speed) a folder with summary.toml, outlet.csv (outlet column),
# cells.csv (per fluid cell: x, y, gdot [1/s], T [K], truth dissipation weight),
# ux.bin, uy.bin, theta.bin (absolute T), is_solid.bin; and runs.csv at the root.
module DieLearned

include(joinpath(@__DIR__, "die_predict.jl"))
using .DiePredict
using KrakeNN, LinearAlgebra, Printf, Statistics, TOML
import KrakeNN: law_ninputs, law_nparams, law_value, _fv_law_invariant, _fv_law_regular_at_rest,
                _fv_reference_nu, _fv_law_number_type
if DiePredict.BACKEND == "cuda"
    using CUDA
end

# ------------------------------------------------------------------ E5 constants
const H_PHYS = 4.0e-3          # m
const K_COND = 0.20            # W/(m K)
const RHO_CP = 2.0e6           # J/(m^3 K)
const ALPHA_PHYS = K_COND / RHO_CP
const BETA = 0.035             # 1/K
const E5_TRUTH = [2000.0, 0.4, 0.035, 0.30, 0.006, 1.5, 465.15, 2.5]

# ------------------------------------------------------------------ network law (copy of e5_fv/families.jl, forward only)
const NN_E0 = 1000.0
const NN_D = 1.0e-3
const NN_SHIFT = (2.0, 465.0)
const NN_SCALE = (2.5, 15.0)
const NN_H = 16

struct NNViscosityLaw{M<:TinyMLP,T} <: AbstractScalarLaw
    mlp::M
    nu_c::T
    nu_ref::T
    shear_scale::T
end
@inline _nn_inputs(law::NNViscosityLaw, g2, Tk) = (log(law.shear_scale^2 * g2 + NN_D^2) / 2, Tk)
@inline _nn_nu(law::NNViscosityLaw, g2, Tk, theta) =
    law.nu_c * exp(KrakeNN._mlp_value(law.mlp, _nn_inputs(law, g2, Tk), theta, Val(false)))
@inline law_ninputs(::NNViscosityLaw) = 2
@inline law_nparams(law::NNViscosityLaw) = law_nparams(law.mlp)
@inline law_value(law::NNViscosityLaw, x::Tuple, theta::AbstractVector) = _nn_nu(law, x[1] * x[1], x[2], theta)
@inline _fv_law_invariant(law::NNViscosityLaw, theta, I1, fields) = _nn_nu(law, 2 * I1, fields[1], theta)
_fv_law_regular_at_rest(::NNViscosityLaw, theta) = true
_fv_reference_nu(v::FVLawViscosity{<:NNViscosityLaw}, fallback) = v.law.nu_ref
@inline _fv_law_number_type(::Type{NNViscosityLaw{M,T}}) where {M,T} = promote_type(T, _fv_law_number_type(M))
nn_mlp() = TinyMLP{2,NN_H,2}(; shift = NN_SHIFT, scale = NN_SCALE)
nn_eta(theta, gdot, Tk) = NN_E0 * exp(KrakeNN._mlp_value(nn_mlp(), (log(gdot^2 + NN_D^2) / 2, Tk), theta, Val(false)))

# ------------------------------------------------------------------ Carreau-WLF library law (copy of e5_fv/families.jl)
const FAM_TREF = 483.15
struct CarreauWLFLaw{T} <: AbstractScalarLaw
    eta_ref::T
    nu_ref::T
    shear_scale::T
end
@inline function _wlf_eta(g2, Tk, theta, ss)
    @inbounds eta0, lam, n, C1, C2 = theta[1], theta[2], theta[3], theta[4], theta[5]
    x = Tk - FAM_TREF
    a = exp(-log(10.0) * C1 * x / (C2 + x))
    return eta0 * a * (1 + (lam * a * ss)^2 * g2)^((n - 1) / 2)
end
@inline law_ninputs(::CarreauWLFLaw) = 2
@inline law_nparams(::CarreauWLFLaw) = 5
@inline law_value(law::CarreauWLFLaw, x::Tuple, theta::AbstractVector) =
    (law.nu_ref / law.eta_ref) * _wlf_eta(x[1] * x[1], x[2], theta, law.shear_scale)
@inline _fv_law_invariant(law::CarreauWLFLaw, theta, I1, fields) =
    (law.nu_ref / law.eta_ref) * _wlf_eta(2 * I1, fields[1], theta, law.shear_scale)
_fv_law_regular_at_rest(::CarreauWLFLaw, theta) = true
_fv_reference_nu(v::FVLawViscosity{<:CarreauWLFLaw}, fallback) = v.law.nu_ref
@inline _fv_law_number_type(::Type{CarreauWLFLaw{T}}) where {T} = T

truth_eta(gdot, Tk) = KrakeNN._e5_dynamic_viscosity(gdot^2, Tk, E5_TRUTH, 1.0)

function read_theta(path)
    for l in eachline(path)
        startswith(l, "theta=") && return parse.(Float64, split(l[7:end], ","))
    end
    error("no theta= line in $path")
end

# ------------------------------------------------------------------ case
"""Die case at grid `G`, mean inlet speed `U_mm` (mm/s), wall `Tw_C` (degC)."""
function physical_case(G, U_mm, Tw_C)
    c0 = DiePredict.case_from_env(; G, profile = :linear)
    U = U_mm * 1e-3; Tw = Tw_C + 273.15; Tin = Tw + 1 / BETA
    dx = H_PHYS / c0.H; dt = dx * c0.U / U; s = 1 / dt
    Pe = U * H_PHYS / ALPHA_PHYS
    alpha = ALPHA_PHYS * dt / dx^2                     # = c0.U * c0.H / Pe
    eta_ref = truth_eta(0.0, Tw)
    heating = eta_ref * s / (c0.nu_ref * RHO_CP)       # kelvin units
    Na = BETA * eta_ref * U^2 / K_COND
    return merge(c0, (; Pe, Na, alpha, heating, theta_in = Tin, theta_wall = Tw, U_mm = Float64(U_mm),
                      Tw_C = Float64(Tw_C), Tw, Tin, s, dx, dt, eta_ref))
end

function viscosity(law, c, th)
    f = (FVStateTemperature(),)
    law === :truth && return FVLawViscosity(E5ThermalLaw(E5_TRUTH; T_wall = c.Tw, nu_ref = c.nu_ref,
                                                         shear_scale = c.s), E5_TRUTH, f; eps = 0.0)
    law === :nn && return FVLawViscosity(NNViscosityLaw(nn_mlp(), c.nu_ref * NN_E0 / c.eta_ref, c.nu_ref, c.s),
                                         th, f; eps = 0.0)
    law === :carreau_wlf && return FVLawViscosity(CarreauWLFLaw(c.eta_ref, c.nu_ref, c.s), th, f; eps = 0.0)
    error("unknown law $law")
end

# ------------------------------------------------------------------ one run
function run_one(law, c, th; outdir)
    mkpath(outdir)
    prob = DiePredict.build_problem(c, viscosity(law, c, th))
    (U, s, records, reached), secs = DiePredict.timed(() -> DiePredict.solve_die(c, prob))
    converged = reached !== nothing && reached == c.heating
    Uh = Array(U)
    hp = s.prob.nu isa Array ? s.prob : fv_adapt(Array, s.prob)
    d = DiePredict.diagnostics(c, hp, Uh)
    fluid = .!c.solid; Nx, Ny = c.Nx, c.Ny
    # outlet column
    out = [j for j in 1:Ny if fluid[Nx, j]]
    umean_out = mean(d.u[Nx, j] for j in out)
    open(joinpath(outdir, "outlet.csv"), "w") do io
        println(io, "j,y_lat,y_over_hout,u_over_umean,theta_star,u_lat,T_K")
        y0 = (first(out) - 1); hout = length(out)
        for j in out
            @printf(io, "%d,%.6f,%.8f,%.12g,%.12g,%.12g,%.12g\n", j, j - 0.5, (j - 0.5 - y0) / hout,
                    d.u[Nx, j] / umean_out, (d.T[Nx, j] - c.Tw) / (c.Tin - c.Tw), d.u[Nx, j], d.T[Nx, j])
        end
    end
    # per-cell shear rate and temperature (for the training-range check)
    open(joinpath(outdir, "cells.csv"), "w") do io
        println(io, "i,j,gdot,T,diss_truth")
        for j in 1:Ny, i in 1:Nx
            fluid[i, j] || continue
            gP = KrakeNN._fv_cell_grad(Uh, hp.tag, hp.bvel, hp.grid, i, j)
            ux, uy, vx, vy = gP[1], gP[2], gP[3], gP[4]
            sxy = (uy + vx) / 2
            g = c.s * sqrt(2 * (ux^2 + vy^2 + 2 * sxy^2))
            @printf(io, "%d,%d,%.8g,%.8g,%.8g\n", i, j, g, d.T[i, j], truth_eta(g, d.T[i, j]) * g^2)
        end
    end
    for (name, A) in (("ux", d.u), ("uy", d.v), ("p", d.p), ("theta", d.T), ("is_solid", c.solid))
        DiePredict.writebin(joinpath(outdir, name * ".bin"), A)
    end
    summary = Dict{String,Any}("law" => string(law), "G" => c.G, "U_mm_s" => c.U_mm, "Tw_K" => c.Tw,
        "Tin_K" => c.Tin, "Pe" => c.Pe, "Na" => c.Na, "shear_scale" => c.s, "eta_ref" => c.eta_ref,
        "converged" => converged, "sigma_reached" => reached === nothing ? -1.0 : reached,
        "sigma_target" => c.heating, "total_seconds" => secs, "newton" => records,
        "dp_lat" => d.dp, "dp_star" => d.dp / c.dp_scale,
        "dp_Pa" => d.dp / c.dp_scale * c.eta_ref * c.U_mm * 1e-3 / H_PHYS,
        "T_cup_out_K" => d.T_cup, "theta_cup_out" => (d.T_cup - c.Tw) / (c.Tin - c.Tw),
        "T_max_K" => d.theta_max, "T_min_K" => d.theta_min,
        "mass_balance_relative" => abs(d.mass_in - d.mass_out) / d.mass_in,
        "energy_balance_relative" => d.balance["relative"], "umean_out_lat" => umean_out)
    DiePredict.BACKEND == "cuda" && (summary["device"] = CUDA.name(CUDA.device()))
    open(joinpath(outdir, "summary.toml"), "w") do io
        TOML.print(io, summary; sorted = true)
    end
    @info "run done" law c.G c.U_mm converged dp_star = summary["dp_star"] theta_cup = summary["theta_cup_out"] secs
    return summary
end

function main()
    root = get(ENV, "KRAKEN_DIE_OUT", joinpath(pwd(), "die_learned_out")); mkpath(root)
    th = Dict{Symbol,Vector{Float64}}(:truth => E5_TRUTH)
    nnf = get(ENV, "KRAKEN_DIE_NN_SUMMARY", ""); isempty(nnf) || (th[:nn] = read_theta(nnf))
    wf = get(ENV, "KRAKEN_DIE_WLF_SUMMARY", ""); isempty(wf) || (th[:carreau_wlf] = read_theta(wf))
    haskey(th, :nn) && (length(th[:nn]) == law_nparams(nn_mlp()) || error("network has $(length(th[:nn])) weights"))
    laws = Symbol.(split(get(ENV, "KRAKEN_DIE_LAWS", join(string.(filter(k -> haskey(th, k), [:truth, :nn, :carreau_wlf])), ",")), ","))
    speeds = parse.(Float64, split(get(ENV, "KRAKEN_DIE_SPEEDS", "1,1.43,2.03,2.9,4.14,5.9,8.42,12"), ","))
    Tw_C = parse(Float64, get(ENV, "KRAKEN_DIE_TW_C", "175"))
    G0 = parse(Int, get(ENV, "KRAKEN_DIE_G", "64"))
    jobs = [(G0, k) for k in eachindex(speeds)]
    for p in split(get(ENV, "KRAKEN_DIE_GRIDS", ""), ","; keepempty = false)
        g, k = parse.(Int, split(p, ":")); push!(jobs, (g, k))
    end
    @info "die learned-law sweep" laws speeds Tw_C jobs
    rows = Dict{String,Any}[]
    for (G, k) in jobs, law in laws
        c = physical_case(G, speeds[k], Tw_C)
        dir = joinpath(root, @sprintf("%s_G%d_U%02d", law, G, k))
        r = try
            run_one(law, c, get(th, law, E5_TRUTH); outdir = dir)
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
        GC.gc(); DiePredict.BACKEND == "cuda" && CUDA.reclaim()
    end
    println(read(joinpath(root, "runs.csv"), String))
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    DieLearned.main()
end
