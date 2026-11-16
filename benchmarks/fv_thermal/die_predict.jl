# Die prediction driver: steady FV Newton (flow + temperature) in the tapered
# extrusion die of the LBM open-die driver (branch feat/gnf-open-steady,
# `run_gnf_open_steady`, docs/src/users/benchmarks/gnf-open-die.md), with a
# given viscosity law, writing fields on the same Nx x Ny cells as the LBM
# truth (cell (i,j) at (i-1/2, j-1/2), h = 1).
#
# Boundary conditions, matched to the truth:
#   W inlet: u = u_in[j] (Poiseuille of mean U across the gap G), v = 0, T = theta_in
#   E outlet: p = 0, zero normal gradient of velocity and temperature
#   S/N walls (y = 0, y = G): no-slip, theta_wall; solid die voxels: no-slip, theta_wall
#
# Environment:
#   KRAKEN_DIE_BACKEND  cpu | cuda                 (default cpu; cuda needs CUDA + CUDSS)
#   KRAKEN_DIE_G        nodes across the inlet gap (default 64; H = G/2, Nx = 7H)
#   KRAKEN_DIE_PROFILE  linear | tanh               (default linear)
#   KRAKEN_DIE_PE, KRAKEN_DIE_NA, KRAKEN_DIE_RE, KRAKEN_DIE_NUREF
#                       groups of the truth         (defaults 30, 0.05, 0.02, 0.3)
#   KRAKEN_DIE_THETA_IN, KRAKEN_DIE_THETA_WALL      (defaults 1.0, 0.0)
#   KRAKEN_DIE_EPS      shear-rate regularisation   (default 0.0, as the LBM closure)
#   KRAKEN_DIE_LAW_FILE optional Julia file defining `die_viscosity(c)` -> FV viscosity
#                       model (e.g. FVLawViscosity(learned_law, theta,
#                       (FVStateTemperature(),); eps)); default: the truth's
#                       Carreau-Yasuda + WLF law through FVGNFViscosity
#   KRAKEN_DIE_DTAU0    pseudo-transient start of the sigma = 0 solve (default: none,
#                       automatic retry with 1.0 if plain Newton fails)
#   KRAKEN_DIE_TRUTH    optional directory with the LBM fields ux.bin, uy.bin,
#                       theta.bin (raw Float64, Nx x Ny, column-major) and
#                       truth.toml (keys dp, mass_in); adds an FV - LBM comparison
#   KRAKEN_DIE_OUT      output directory (default tmp/die_predict/G<G>_<profile>_<backend>)
#
# Outputs: summary.toml (case, Newton statistics per heating step, dp,
# dp_tilde = dp/dp_scale, mass in/out, mixing-cup outlet temperature, energy
# balance terms, timings, GPU memory, optional comparison) and raw Float64
# fields ux.bin, uy.bin, p.bin, theta.bin, is_solid.bin (Nx x Ny), plus fields.csv.
#
# Run (repository root):
#   KRAKEN_DIE_BACKEND=cuda julia --project=. benchmarks/fv_thermal/die_predict.jl
module DiePredict

using KrakeNN, LinearAlgebra, Printf, Statistics, TOML

const BACKEND = lowercase(get(ENV, "KRAKEN_DIE_BACKEND", "cpu"))
BACKEND in ("cpu", "cuda") || error("KRAKEN_DIE_BACKEND must be cpu or cuda, got $BACKEND")
if BACKEND == "cuda"
    using CUDA, CUDSS      # top level, never inside a function (world age)
end

tools() = BACKEND == "cuda" ? (to = CUDA.CuArray, linsolve = CUDABackendTag()) :
                              (to = Array, linsolve = CPUBackendTag())
sync() = BACKEND == "cuda" && CUDA.synchronize()
gpu_used() = BACKEND == "cuda" ? (CUDA.synchronize(); Int(CUDA.total_memory() - CUDA.free_memory())) : 0
function timed(f)
    sync(); t0 = time_ns(); r = f(); sync()
    return r, (time_ns() - t0) / 1e9
end
envf(key, default) = parse(Float64, get(ENV, key, string(default)))

# ------------------------------------------------------------------- geometry
"""Copy of `gnf_tapered_die_mask` (feat/gnf-open-steady): node (i,j) at
(i-1/2, j-1/2) is solid when |y - G/2| >= gap(x)/2, lengths in inlet gaps G = Ny."""
function die_mask(Nx::Integer, Ny::Integer; profile::Symbol=:linear, h_out=0.25,
                  x_start=0.75, x_end=2.25, x_mid=1.625, width=0.3)
    profile in (:linear, :tanh) || throw(ArgumentError("profile must be :linear or :tanh"))
    G = Float64(Ny)
    solid = fill(false, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        x = (i - 0.5) / G; y = j - 0.5
        gap = if profile === :linear
            G * (1 + (h_out - 1) * clamp((x - x_start) / (x_end - x_start), 0.0, 1.0))
        else
            G * (h_out + (1 - h_out) * (1 - tanh((x - x_mid) / width)) / 2)
        end
        solid[i, j] = abs(y - G / 2) >= gap / 2
    end
    return solid
end

"""Copy of `gnf_open_thermal_groups` (feat/gnf-open-steady)."""
thermal_groups(; Pe, Na=0, H, U, nu_ref) =
    (; alpha=U * H / Pe, heating=Na * (U * H / Pe) / (nu_ref * U^2), Re=U * H / nu_ref,
       dp_scale=nu_ref * U / H)

function case_from_env(; G=parse(Int, get(ENV, "KRAKEN_DIE_G", "64")),
                       profile=Symbol(get(ENV, "KRAKEN_DIE_PROFILE", "linear")))
    iseven(G) || error("G must be even")
    H = G ÷ 2; Nx = 7H
    nu_ref = envf("KRAKEN_DIE_NUREF", 0.3); Re = envf("KRAKEN_DIE_RE", 0.02)
    U = Re * nu_ref / H
    groups = thermal_groups(; Pe=envf("KRAKEN_DIE_PE", 30.0), Na=envf("KRAKEN_DIE_NA", 0.05),
                            H, U, nu_ref)
    return (; G, H, Nx, Ny=G, profile, nu_ref, U, Re, Pe=envf("KRAKEN_DIE_PE", 30.0),
            Na=envf("KRAKEN_DIE_NA", 0.05), groups..., theta_in=envf("KRAKEN_DIE_THETA_IN", 1.0),
            theta_wall=envf("KRAKEN_DIE_THETA_WALL", 0.0), eps=envf("KRAKEN_DIE_EPS", 0.0),
            solid=die_mask(7H, G; profile))
end

"""The truth's example law: Carreau (n = 0.6, lambda U/H = 0.625) with a WLF shift in theta."""
default_law(c) = CarreauYasuda(c.nu_ref, 0.0, 0.625 * c.H / c.U, 2.0, 0.6;
                               thermal=WLFCoupling(1.0, 2.0, 5.0))
default_viscosity(c) = FVGNFViscosity(default_law(c), c.eps; temperature=FVStateTemperature())

u_inlet(c, y) = 6c.U * (y / c.G) * (1 - y / c.G)

function build_problem(c, visc)
    return fv_steady_problem(c.Nx, c.Ny; h=1.0, nu=c.nu_ref, conv=true,
        bc=(W=:inlet, E=:outlet, S=:wall, N=:wall), solid=c.solid,
        boundary_velocity=(side, x, y) -> (side === :W ? u_inlet(c, y) : 0.0, 0.0),
        viscosity=visc,
        energy=FVEnergy(; alpha=c.alpha, sigma=c.heating, left=c.theta_in,
                        bottom=c.theta_wall, top=c.theta_wall, solid=c.theta_wall))
end

function initial_state(c, prob)
    # Plug of the local gap with the inlet flux, theta_in in the fluid.
    fluid = .!c.solid
    q = sum(u_inlet(c, j - 0.5) for j in 1:c.Ny)
    u = [fluid[i, j] ? q / max(count(view(fluid, i, :)), 1) : 0.0 for i in 1:c.Nx, j in 1:c.Ny]
    z = zeros(c.Nx, c.Ny)
    return fv_state(prob, u, z, z, c.theta_in)
end

# ------------------------------------------------------------------- solve
function newton_record(stats, sigma, secs, mem0, mem1)
    return Dict{String,Any}("sigma" => sigma, "converged" => stats.converged,
        "iterations" => stats.iterations, "stagnated" => stats.stagnated,
        "gate" => string(get(stats, :gate, :none)),
        "initial_residual_inf" => isempty(stats.inf) ? NaN : first(stats.inf),
        "final_residual_inf" => isempty(stats.inf) ? NaN : last(stats.inf),
        "seconds" => secs, "gpu_used_bytes_before" => mem0, "gpu_used_bytes_after" => mem1)
end

"""sigma = 0 solve, then the heating ladder 0 -> c.heating with step halving
(at most 5 halvings per step). Returns (U on device, solver, records, sigma reached)."""
function solve_die(c, prob; maxit=60)
    tl = tools(); records = Dict{String,Any}[]
    U = tl.to(initial_state(c, prob)); saved = copy(U)
    dtau_env = get(ENV, "KRAKEN_DIE_DTAU0", "")
    attempts = isempty(dtau_env) ? (nothing, 1.0) : (parse(Float64, dtau_env),)
    p0 = KrakeNN._fv_heating_problem(prob, 0.0)
    s = fv_newton_solver(p0; tl.to, tl.linsolve)
    ok = false
    for dtau0 in attempts
        copyto!(U, saved); mem0 = gpu_used()
        stats, secs = timed(() -> fv_newton!(s, U; maxit, dtau0))
        rec = newton_record(stats, 0.0, secs, mem0, gpu_used()); rec["dtau0"] = string(dtau0)
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
        mem0 = gpu_used()
        stats, secs = timed(() -> fv_newton!(st, U; maxit))
        push!(records, newton_record(stats, target, secs, mem0, gpu_used()))
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

# ------------------------------------------------------------------- diagnostics
"""Fields and balances of the solved problem `hp` (host copy of the last converged solver's problem)."""
function diagnostics(c, hp, Uh)
    u, v, p, T = fv_fields(hp, Uh)
    fluid = .!c.solid; Nx, Ny = c.Nx, c.Ny
    col_mean(A, i) = mean(A[i, j] for j in 1:Ny if fluid[i, j])
    dp = col_mean(p, 1) - col_mean(p, Nx)
    mass_in = sum(u_inlet(c, j - 0.5) for j in 1:Ny if fluid[1, j])
    out = [j for j in 1:Ny if fluid[Nx, j]]
    mass_out = sum(u[Nx, j] for j in out)
    T_cup = sum(u[Nx, j] * T[Nx, j] for j in out) / mass_out
    # Energy balance in theta, every term recomputed from fields and BC data
    # (heat entering counts positive; h = 1).
    a = c.alpha
    adv_in = mass_in * c.theta_in
    cond_in = sum(2a * (c.theta_in - T[1, j]) for j in 1:Ny if fluid[1, j])
    adv_out = sum(u[Nx, j] * T[Nx, j] for j in out)
    walls = 0.0; heat = 0.0
    for j in 1:Ny, i in 1:Nx
        fluid[i, j] || continue
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ii = i + di; jj = j + dj
            if 1 <= jj <= Ny && 1 <= ii <= Nx
                fluid[ii, jj] || (walls += 2a * (c.theta_wall - T[i, j]))
            elseif jj < 1 || jj > Ny
                walls += 2a * (c.theta_wall - T[i, j])
            end
        end
        heat += KrakeNN._fv_energy_heating(Uh, hp.tag, hp.nu, hp.bvel, hp.grid, i, j,
                                           hp.visc, hp.energy.sigma)
    end
    terms = Dict{String,Any}("advection_in" => adv_in, "conduction_in" => cond_in,
        "advection_out" => adv_out, "walls_and_solid" => walls, "viscous_heating" => heat)
    imbalance = adv_in + cond_in - adv_out + walls + heat
    return (; u, v, p, T, dp, mass_in, mass_out, T_cup,
            balance=Dict{String,Any}("terms" => terms, "imbalance" => imbalance,
                "relative" => abs(imbalance) / maximum(abs, values(terms))),
            theta_min=minimum(T[fluid]), theta_max=maximum(T[fluid]),
            u_max=maximum(abs, u[fluid]))
end

function readbin(path, Nx, Ny)
    filesize(path) == 8Nx * Ny || error("$path has size $(filesize(path)), expected $(8Nx*Ny)")
    return reshape(copy(reinterpret(Float64, read(path))), Nx, Ny)
end
writebin(path, A) = write(path, Vector{Float64}(vec(Float64.(A))))

function compare(a, b, fluid)
    d = a[fluid] .- b[fluid]; scale = maximum(abs, b[fluid])
    return Dict{String,Any}("l2_over_max" => sqrt(mean(abs2, d)) / scale,
        "linf_over_max" => maximum(abs, d) / scale, "l2_rel_norm" => norm(d) / norm(b[fluid]))
end

function truth_comparison(c, d, dir)
    fluid = .!c.solid; out = Dict{String,Any}("truth_dir" => dir)
    rd(name) = readbin(joinpath(dir, name), c.Nx, c.Ny)
    isfile(joinpath(dir, "theta.bin")) && (out["theta"] = compare(d.T, rd("theta.bin"), fluid))
    isfile(joinpath(dir, "ux.bin")) && (out["ux"] = compare(d.u, rd("ux.bin"), fluid))
    if isfile(joinpath(dir, "uy.bin"))
        uy = rd("uy.bin"); ux = isfile(joinpath(dir, "ux.bin")) ? rd("ux.bin") : d.u
        out["uy_linf_over_max_ux"] = maximum(abs, d.v[fluid] .- uy[fluid]) / maximum(abs, ux[fluid])
    end
    if isfile(joinpath(dir, "truth.toml"))
        t = TOML.parsefile(joinpath(dir, "truth.toml"))
        haskey(t, "dp") && (out["dp_lbm"] = t["dp"]; out["dp_ratio_fv_over_lbm"] = d.dp / t["dp"])
        haskey(t, "mass_in") && (out["mass_in_lbm"] = t["mass_in"])
    end
    return out
end

function write_csv(path, cols, Nx, Ny)
    open(path, "w") do io
        println(io, "i,j,", join(first.(cols), ","))
        for j in 1:Ny, i in 1:Nx
            print(io, i, ",", j)
            for (_, A) in cols
                @printf(io, ",%.17g", A[i, j])
            end
            println(io)
        end
    end
end

# ------------------------------------------------------------------- main
"""Solve the die for case `c` with viscosity model `visc`; write into `outdir`."""
function run_case(c, visc; outdir, law_name="carreau_wlf", maxit=60)
    mkpath(outdir)
    prob = build_problem(c, visc)
    (U, s, records, reached), secs = timed(() -> solve_die(c, prob; maxit))
    summary = Dict{String,Any}("backend" => BACKEND, "julia" => string(VERSION), "law" => law_name,
        "case" => Dict{String,Any}("G" => c.G, "H" => c.H, "Nx" => c.Nx, "Ny" => c.Ny,
            "profile" => string(c.profile), "nu_ref" => c.nu_ref, "U" => c.U, "Re" => c.Re,
            "Pe" => c.Pe, "Na" => c.Na, "alpha" => c.alpha, "heating" => c.heating,
            "dp_scale" => c.dp_scale, "theta_in" => c.theta_in, "theta_wall" => c.theta_wall,
            "eps" => c.eps, "fluid_cells" => count(!, c.solid),
            "n_unknowns" => fv_nunknowns(prob)),
        "newton" => records, "total_seconds" => secs,
        "converged" => reached !== nothing && reached == c.heating,
        "sigma_reached" => reached === nothing ? -1.0 : reached)
    if BACKEND == "cuda"
        summary["device"] = CUDA.name(CUDA.device())
        summary["gpu_total_bytes"] = Int(CUDA.total_memory())
    end
    Uh = Array(U)
    d = diagnostics(c, s.prob.nu isa Array ? s.prob : fv_adapt(Array, s.prob), Uh)
    summary["dp"] = d.dp; summary["dp_tilde"] = d.dp / c.dp_scale
    summary["mass_in"] = d.mass_in; summary["mass_out"] = d.mass_out
    summary["mass_balance_relative"] = abs(d.mass_in - d.mass_out) / d.mass_in
    summary["theta_cup_outlet"] = d.T_cup
    summary["theta_min"] = d.theta_min; summary["theta_max"] = d.theta_max; summary["u_max"] = d.u_max
    summary["energy_balance"] = d.balance
    truth = get(ENV, "KRAKEN_DIE_TRUTH", "")
    isempty(truth) || (summary["comparison"] = truth_comparison(c, d, truth))
    for (name, A) in (("ux", d.u), ("uy", d.v), ("p", d.p), ("theta", d.T), ("is_solid", c.solid))
        writebin(joinpath(outdir, name * ".bin"), A)
    end
    write_csv(joinpath(outdir, "fields.csv"),
        ["fluid" => Float64.(.!c.solid), "ux" => d.u, "uy" => d.v, "p" => d.p, "theta" => d.T],
        c.Nx, c.Ny)
    open(joinpath(outdir, "summary.toml"), "w") do io
        TOML.print(io, summary; sorted=true)
    end
    @info "die summary written" outdir converged = summary["converged"] dp_tilde = summary["dp_tilde"]
    return summary, d
end

function main()
    c = case_from_env()
    root = normpath(joinpath(@__DIR__, "..", ".."))
    outdir = get(ENV, "KRAKEN_DIE_OUT",
                 joinpath(root, "tmp", "die_predict", "G$(c.G)_$(c.profile)_$(BACKEND)"))
    file = get(ENV, "KRAKEN_DIE_LAW_FILE", "")
    if isempty(file)
        return run_case(c, default_viscosity(c); outdir)
    end
    Base.include(@__MODULE__, abspath(file))
    # The file's bindings and methods are only visible in the latest world.
    f = Base.invokelatest(getglobal, @__MODULE__, :die_viscosity)
    visc = Base.invokelatest(f, c)
    return Base.invokelatest(run_case, c, visc; outdir, law_name=basename(file))
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    DiePredict.main()
end
