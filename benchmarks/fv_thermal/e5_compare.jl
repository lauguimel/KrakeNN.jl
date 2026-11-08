# T5 comparison driver: FV (KrakeNN fixed-point/Newton thermal) vs LBM on the E5
# six-cylinder periodic cell, at identical voxel mask and identical lattice
# coefficients (plan R6 ladder). Nothing here tunes the FV discretisation.
#
# Environment:
#   KRAKEN_E5_BACKEND  cpu | cuda            (default cpu; cuda needs CUDA + CUDSS)
#   KRAKEN_E5_GRID     300x100 | 600x200     (default 300x100)
#   KRAKEN_E5_EXPORT   export directory      (default tmp/e5_export/<grid>)
#   KRAKEN_E5_OUT      output directory      (default tmp/e5_compare_out/<grid>_<backend>)
#   KRAKEN_E5_STAGES   comma list of 1,2,3   (default 1,2,3)
#   KRAKEN_E5_FD       1 | 0                 (stage 3 central FD on eta0, beta; default 1)
#   KRAKEN_E5_INIT     zero | lbm            (initial flow guess of stages 1 and 3;
#                                             the converged solution does not depend on it)
#
# The summary ends with a [thresholds] table (limit, value, pass per item; pass only
# for grid 600x200 with stage 3 run, applies = false otherwise), see `thresholds` below.
#
# Run (repository root):  sh tmp/thermal_julia.sh --project=. benchmarks/fv_thermal/e5_compare.jl
# Metric conventions (all over fluid cells, FV minus LBM):
#   l2_over_max  = RMS(diff) / max|reference|        linf_over_max = max|diff| / max|reference|
#   l2_rel_norm  = ||diff||_2 / ||reference||_2
module E5Compare

using KrakeNN, LinearAlgebra, Printf, Statistics, TOML, SparseArrays

const BACKEND = lowercase(get(ENV, "KRAKEN_E5_BACKEND", "cpu"))
BACKEND in ("cpu", "cuda") || error("KRAKEN_E5_BACKEND must be cpu or cuda, got $BACKEND")
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

# ------------------------------------------------------------------- data
struct Case
    grid::String
    Nx::Int
    Ny::Int
    consts::Dict{String,Any}
    theta::Vector{Float64}
    ux::Matrix{Float64}; uy::Matrix{Float64}
    uf::Matrix{Float64}; vf::Matrix{Float64}
    T::Matrix{Float64}
    solid::BitMatrix
    nu::Matrix{Float64}       # nu_from_gradient_base
    I1::Matrix{Float64}       # I1_from_gradient_base
    nu_kin::Union{Nothing,Matrix{Float64}}   # nu.bin   (600x200 only)
    I1_kin::Union{Nothing,Matrix{Float64}}   # I1.bin   (600x200 only)
    gradient::Union{Nothing,Dict{String,Any}}
    raw_case::Dict{String,Any}
end

function readbin(path, Nx, Ny)
    filesize(path) == 8Nx * Ny || error("$path has size $(filesize(path)), expected $(8Nx*Ny)")
    return reshape(copy(reinterpret(Float64, read(path))), Nx, Ny)
end

function load_case(dir::AbstractString)
    raw = TOML.parsefile(joinpath(dir, "case.toml"))
    c = raw["constants"]; Nx = Int(c["Nx"]); Ny = Int(c["Ny"])
    Tuple(raw["format"]["shape"]) == (Nx, Ny) || error("case.toml shape differs from constants Nx,Ny")
    lp = raw["law_parameters"]
    theta = Float64[lp["eta0"], lp["lambda"], lp["beta"], lp["n0"], lp["slope"],
                    lp["amp"], lp["Tc"], lp["width"]]
    grd = isfile(joinpath(dir, "gradient.toml")) ? TOML.parsefile(joinpath(dir, "gradient.toml")) : nothing
    if grd !== nothing
        Float64.(grd["parameter_values"]) == theta ||
            error("gradient.toml parameter_values differ from case.toml law_parameters")
    end
    rd(name) = readbin(joinpath(dir, name), Nx, Ny)
    opt(name) = isfile(joinpath(dir, name)) ? rd(name) : nothing
    return Case(string(raw["grid"]), Nx, Ny, c, theta, rd("ux.bin"), rd("uy.bin"),
                rd("uf.bin"), rd("vf.bin"), rd("T.bin"), rd("is_solid.bin") .> 0.5,
                rd("nu_from_gradient_base.bin"), rd("I1_from_gradient_base.bin"),
                opt("nu.bin"), opt("I1.bin"), grd, raw)
end

cval(cs::Case, k) = Float64(cs.consts[k])
make_law(cs::Case) = E5ThermalLaw(cs.theta; T_wall = cval(cs, "T_wall"),
                                  nu_ref = cval(cs, "nuref"), shear_scale = cval(cs, "s"))
law_nu(law, theta, I1, T) = KrakeNN.law_value(law, (sqrt(2 * I1), T), theta)

# ------------------------------------------------------------------- metrics
function compare(a, b, fluid, scale = maximum(abs, b[fluid]))
    d = (a .- b)[fluid]; r = b[fluid]
    return Dict{String,Any}("l2_over_max" => sqrt(mean(abs2, d)) / scale,
                            "linf_over_max" => maximum(abs, d) / scale,
                            "l2_rel_norm" => norm(d) / max(norm(r), floatmin()),
                            "scale" => scale)
end
function percentiles(x)
    s = sort(vec(x)); n = length(s)
    q(p) = s[clamp(ceil(Int, p * n), 1, n)]
    return Dict{String,Any}("p50" => q(0.5), "p90" => q(0.9), "p99" => q(0.99),
                            "p999" => q(0.999), "max" => s[end])
end
mean_rise(T, fluid, Tw) = mean(T[fluid]) - Tw

# ------------------------------------------------------------------- export check
function check_export(cs::Case)
    Nx, Ny = cs.Nx, cs.Ny; fluid = .!cs.solid; Tw = cval(cs, "T_wall")
    out = Dict{String,Any}()
    out["n_fluid_cells"] = count(fluid)
    cs.gradient === nothing || (out["n_fluid_cells_gradient_toml"] = Int(cs.gradient["n_fluid_cells"]))
    out["nu_per_eta_times_eta_ref_minus_nuref"] = cval(cs, "nu_per_eta") * cval(cs, "eta_ref") - cval(cs, "nuref")
    law = make_law(cs)
    out["law_eta_ref_rel_diff_vs_case_toml"] = abs(law.eta_ref - cval(cs, "eta_ref")) / cval(cs, "eta_ref")
    out["law_nu_per_eta_rel_diff"] = abs(law.nu_ref / law.eta_ref - cval(cs, "nu_per_eta")) / cval(cs, "nu_per_eta")
    function lawcheck(nu, I1)
        nuL = [law_nu(law, cs.theta, I1[i, j], cs.T[i, j]) for i in 1:Nx, j in 1:Ny]
        rel = abs.(nuL .- nu)[fluid] ./ nu[fluid]
        return Dict{String,Any}("rel_diff" => percentiles(rel),
                                "nu_min" => minimum(nu[fluid]), "nu_max" => maximum(nu[fluid]))
    end
    out["law_vs_nu_from_gradient_base"] = lawcheck(cs.nu, cs.I1)
    cs.nu_kin === nothing || (out["law_vs_nu_kinetic"] = lawcheck(cs.nu_kin, cs.I1_kin))
    sol = cs.solid
    out["max_abs_in_solid"] = Dict{String,Any}(
        "ux" => maximum(abs, cs.ux[sol]; init = 0.0), "uy" => maximum(abs, cs.uy[sol]; init = 0.0),
        "nu" => maximum(abs, cs.nu[sol]; init = 0.0), "T_minus_Twall" => maximum(abs, cs.T[sol] .- Tw; init = 0.0),
        "uf" => maximum(abs, cs.uf[sol]; init = 0.0), "vf" => maximum(abs, cs.vf[sol]; init = 0.0))
    # faces: uf[i,j] between (i,j) and (i+1,j) periodic; vf[i,j] between (i,j) and (i,j+1)
    east_solid = sol .| circshift(sol, (-1, 0))
    north_solid = sol .| [j < Ny ? sol[i, j+1] : true for i in 1:Nx, j in 1:Ny]
    out["max_abs_uf_on_faces_touching_solid"] = maximum(abs, cs.uf[east_solid]; init = 0.0)
    out["max_abs_vf_on_faces_touching_solid"] = maximum(abs, cs.vf[north_solid]; init = 0.0)
    out["max_abs_vf_top_row"] = maximum(abs, cs.vf[:, Ny])
    uscale = maximum(abs, cs.uf)
    div = [cs.uf[i, j] - cs.uf[mod1(i - 1, Nx), j] + cs.vf[i, j] - (j > 1 ? cs.vf[i, j-1] : 0.0)
           for i in 1:Nx, j in 1:Ny]
    out["max_abs_face_divergence_over_max_uf"] = maximum(abs, div[fluid]) / uscale
    uavg = [(cs.uf[mod1(i - 1, Nx), j] + cs.uf[i, j]) / 2 for i in 1:Nx, j in 1:Ny]
    vavg = [((j > 1 ? cs.vf[i, j-1] : 0.0) + cs.vf[i, j]) / 2 for i in 1:Nx, j in 1:Ny]
    out["cell_vs_face_average"] = Dict{String,Any}(
        "ux_max_diff_over_max_ux" => maximum(abs, (uavg .- cs.ux)[fluid]) / maximum(abs, cs.ux),
        "uy_max_diff_over_max_ux" => maximum(abs, (vavg .- cs.uy)[fluid]) / maximum(abs, cs.ux),
        "note" => "export built by node average then projection: equality is not expected")
    m = mean_rise(cs.T, fluid, Tw)
    out["m_from_T"] = m
    cs.gradient === nothing || (out["m_gradient_toml"] = Float64(cs.gradient["m_K"]);
                                out["m_from_T_minus_gradient_toml"] = m - Float64(cs.gradient["m_K"]))
    S = cval(cs, "sigma") .* cs.nu .* 2 .* cs.I1
    out["integral_source_S"] = sum(S[fluid])
    return out
end

# ------------------------------------------------------------------- shared FV builders
const BC = (W = :periodic, E = :periodic, S = :wall, N = :wall)

function build_problem(cs::Case; visc, energy = nothing)
    Fx = cval(cs, "Fx")
    return fv_steady_problem(cs.Nx, cs.Ny; h = 1.0, nu = cval(cs, "nuref"), bc = BC,
        solid = cs.solid, source = (x, y) -> (Fx, 0.0, 0.0), conv = true,
        viscosity = visc, energy)
end

function init_state(cs::Case, prob; temperature = nothing)
    z = zeros(cs.Nx, cs.Ny)
    useL = lowercase(get(ENV, "KRAKEN_E5_INIT", "zero")) == "lbm"
    u = useL ? cs.ux : z; v = useL ? cs.uy : z
    U = temperature === nothing ? fv_state(prob, u, v, z) : fv_state(prob, u, v, z, temperature)
    KrakeNN._fv_initial_force!(U, prob)
    return U
end

function cell_fields(prob, Uh, law, theta, T)
    # FV cell strain invariant (Green-Gauss gradient, same as the heating term) and law viscosity
    g = prob.grid; Nx, Ny = g.Nx, g.Ny
    I1 = zeros(Nx, Ny); nu = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        prob.tag[i, j] == KrakeNN.FV_CELL_SOLID && continue
        gP = KrakeNN._fv_cell_grad(Uh, prob.tag, prob.bvel, g, i, j)
        sxy = (gP[2] + gP[3]) / 2
        I1[i, j] = gP[1]^2 + gP[4]^2 + 2sxy^2
        nu[i, j] = law_nu(law, theta, I1[i, j], T[i, j])
    end
    return I1, nu
end

host(prob) = prob.nu isa Array ? prob : fv_adapt(Array, prob)

function bottom_flux(T, fluid, Tw, alpha)
    # FV Dirichlet row of the bottom wall: 2 alpha (T - Tw) per fluid cell at j = 1
    return sum(2alpha * (T[i, 1] - Tw) for i in 1:size(T, 1) if fluid[i, 1]; init = 0.0)
end

# Stopping gate and round-off-floor diagnostics (thermal solves only).
function gate_fields!(res, stats)
    hasproperty(stats, :gate) || return res
    res["gate"] = string(stats.gate)
    f = stats.energy_floor
    f === nothing || (res["floor_max_abs"] = f.max_floor; res["floor_max_ratio_residual_over_floor"] = f.max_ratio)
    return res
end

function newton_summary(stats, nunk, nnz_, secs, mem0, mem1)
    return gate_fields!(Dict{String,Any}("converged" => stats.converged, "iterations" => stats.iterations,
        "final_residual_inf" => isempty(stats.inf) ? NaN : last(stats.inf),
        "initial_residual_inf" => isempty(stats.inf) ? NaN : first(stats.inf),
        "n_unknowns" => nunk, "nnz" => nnz_, "seconds" => secs,
        "gpu_used_bytes_before" => mem0, "gpu_used_bytes_after" => mem1,
        "gpu_peak_note" => "device used memory (total-free, includes pool cache) sampled before and after the solve"), stats)
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

# ------------------------------------------------------------------- stage 1
function stage1(cs::Case; outdir = nothing)
    tl = tools(); fluid = .!cs.solid; law = make_law(cs); Tw = cval(cs, "T_wall")
    visc = FVLawViscosity(law, copy(cs.theta), (copy(cs.T),); eps = 0.0)
    prob = build_problem(cs; visc)
    mem0 = gpu_used()
    (U, s), tsetup = timed() do
        U = tl.to(init_state(cs, prob))
        U, fv_newton_solver(prob; tl.to, tl.linsolve)
    end
    stats, tsolve = timed(() -> fv_newton!(s, U; maxit = 60, atol = 1e-10 * cval(cs, "Fx")))
    mem1 = gpu_used()
    Uh = Array(U); u, v, p = fv_fields(prob, Uh)
    I1, nu = cell_fields(host(prob), Uh, law, cs.theta, cs.T)
    res = newton_summary(stats, fv_nunknowns(prob), length(s.nzval), tsolve, mem0, mem1)
    res["setup_seconds"] = tsetup
    res["atol_used"] = 1e-10 * cval(cs, "Fx")
    res["ux"] = compare(u, cs.ux, fluid); res["uy"] = compare(v, cs.uy, fluid, maximum(abs, cs.ux[fluid]))
    res["mean_ux_fv"] = mean(u[fluid]); res["mean_ux_lbm"] = mean(cs.ux[fluid])
    res["flow_rate_ratio_fv_over_lbm"] = mean(u[fluid]) / mean(cs.ux[fluid])
    res["nu_vs_nu_from_gradient_base"] = compare(nu, cs.nu, fluid)
    res["I1_vs_I1_from_gradient_base"] = compare(I1, cs.I1, fluid)
    res["max_abs_uy_fv_over_max_ux"] = maximum(abs, v[fluid]) / maximum(abs, u[fluid])
    outdir === nothing || write_csv(joinpath(outdir, "stage1_fields.csv"),
        ["fluid" => Float64.(fluid), "ux_fv" => u, "ux_lbm" => cs.ux, "uy_fv" => v, "uy_lbm" => cs.uy,
         "nu_fv" => nu, "nu_lbm" => cs.nu, "I1_fv" => I1, "I1_lbm" => cs.I1, "p_fv" => p], cs.Nx, cs.Ny)
    return res
end

# ------------------------------------------------------------------- stage 2
"""Heat only. `S` is the prescribed source per area; `uf, vf` the prescribed face velocities."""
function solve_heat_only(cs::Case, S, uf, vf; Tinit = nothing)
    tl = tools(); Tw = cval(cs, "T_wall"); alpha = cval(cs, "alpha")
    energy = FVEnergy(alpha = alpha, sigma = 0.0, bottom = Tw, source = S, face_velocity = (uf, vf))
    prob = build_problem(cs; visc = nothing, energy)
    z = zeros(cs.Nx, cs.Ny)
    U = fv_state(prob, z, z, z, Tw)
    mem0 = gpu_used()
    sol, tsetup = timed(() -> fv_energy_solver(prob, U; tl.to, tl.linsolve))
    # The solver block is theta = T - T_ref (T_ref = prob.energy.Tref; here T_wall, so 0 at start).
    # Solid rows are storage identities (-> 0): start them at 0.
    Tref = Float64(prob.energy.Tref)
    T0 = tl.to(Float64.(vec(ifelse.(cs.solid, 0.0, Tw - Tref))))
    stats, tsolve = timed(() -> fv_energy_solve!(sol, T0))
    mem1 = gpu_used()
    Tflat = Array(T0) .+ Tref                  # back to absolute temperature
    T = copy(reshape(Tflat, cs.Nx, cs.Ny))    # cell ordering is i fastest, as the matrix
    T[cs.solid] .= Tw                          # report solids at T_wall as in the export
    return (; prob, sol, stats, T, tsetup, tsolve, mem0, mem1)
end

function stage2(cs::Case; outdir = nothing)
    fluid = .!cs.solid; Tw = cval(cs, "T_wall"); alpha = cval(cs, "alpha"); sigma = cval(cs, "sigma")
    S = sigma .* cs.nu .* 2 .* cs.I1
    r = solve_heat_only(cs, S, cs.uf, cs.vf)
    T = r.T
    res = Dict{String,Any}("converged" => r.stats.converged, "iterations" => r.stats.iterations,
        "final_scaled_residual" => last(r.stats.history),
        "min_scaled_residual" => minimum(r.stats.history),
        "last_increment_over_deltaT" => isempty(r.stats.temperature_increment) ? 0.0 :
            last(r.stats.temperature_increment) / r.sol.scales.deltaT,
        "gate_note" => "gate is (scaled residual <= 1e-10 AND increment <= 1e-10 deltaT) = tolerance, or (increment criterion AND every row <= C eps sum|terms|) = roundoff_floor; the state is theta = T - T_ref (T_ref = T_wall)",
        "n_unknowns" => length(T),
        "nnz" => length(r.sol.nzval), "seconds" => r.tsolve, "setup_seconds" => r.tsetup,
        "gpu_used_bytes_before" => r.mem0, "gpu_used_bytes_after" => r.mem1)
    gate_fields!(res, r.stats)
    res["dT"] = compare(T .- Tw, cs.T .- Tw, fluid)
    m_fv = mean_rise(T, fluid, Tw); m_lbm = mean_rise(cs.T, fluid, Tw)
    res["m_fv"] = m_fv; res["m_lbm"] = m_lbm; res["m_ratio_fv_over_lbm"] = m_fv / m_lbm
    res["dT_max_fv"] = maximum((T .- Tw)[fluid]); res["dT_max_lbm"] = maximum((cs.T .- Tw)[fluid])
    P = sum(S[fluid]); res["integral_source"] = P
    res["power_balance_fv"] = Dict{String,Any}("bottom_flux" => bottom_flux(T, fluid, Tw, alpha),
        "relative_imbalance" => (bottom_flux(T, fluid, Tw, alpha) - P) / P)
    res["power_balance_lbm_T_in_fv_operator"] = Dict{String,Any}(
        "bottom_flux" => bottom_flux(cs.T, fluid, Tw, alpha),
        "relative_imbalance" => (bottom_flux(cs.T, fluid, Tw, alpha) - P) / P)
    # operator equality: residual of the FV energy operator at the LBM temperature
    tl = tools()
    Tref = Float64(r.prob.energy.Tref)
    Tl = tl.to(Float64.(vec(ifelse.(cs.solid, 0.0, cs.T .- Tref))))   # theta block
    st0 = fv_energy_solve!(r.sol, Tl; maxit = 0)
    Rv = Array(r.sol.R)
    res["fv_operator_residual_at_lbm_T"] = Dict{String,Any}(
        "scaled_inf" => first(st0.history), "scale" => r.sol.scales.equations[4],
        "abs_inf" => maximum(abs, Rv), "abs_sum" => sum(Rv),
        "abs_inf_over_max_cell_source" => maximum(abs, Rv) / maximum(S))
    outdir === nothing || write_csv(joinpath(outdir, "stage2_fields.csv"),
        ["fluid" => Float64.(fluid), "T_fv" => T, "T_lbm" => cs.T, "S" => S], cs.Nx, cs.Ny)
    return res
end

# ------------------------------------------------------------------- stage 3
function stage3(cs::Case; outdir = nothing, fd::Bool = true)
    tl = tools(); fluid = .!cs.solid; law = make_law(cs); Tw = cval(cs, "T_wall")
    alpha = cval(cs, "alpha"); sigma = cval(cs, "sigma")
    visc = FVLawViscosity(law, copy(cs.theta), (FVStateTemperature(),); eps = 0.0)
    energy = FVEnergy(alpha = alpha, sigma = sigma, bottom = Tw)
    prob = build_problem(cs; visc, energy)
    z = zeros(cs.Nx, cs.Ny)
    U0 = lowercase(get(ENV, "KRAKEN_E5_INIT", "zero")) == "lbm" ? fv_state(prob, cs.ux, cs.uy, z, Tw) : nothing
    mem0 = gpu_used()
    cont, tcont = timed(() -> fv_thermal_continuation(prob; U0, tl.to, tl.linsolve))
    mem1 = gpu_used()
    res = Dict{String,Any}("converged" => cont.converged, "reason" => string(cont.reason),
        "last_sigma" => something(cont.last_sigma, NaN), "halvings" => cont.halvings,
        "continuation_steps" => length(cont.history), "continuation_attempts" => length(cont.attempts),
        "newton_iterations_per_step" => [h.stats.iterations for h in cont.history],
        "newton_iterations_total" => sum(h.stats.iterations for h in cont.history; init = 0),
        "sigma_per_step" => [h.sigma for h in cont.history],
        "gate_per_step" => string.(cont.gates),
        "n_unknowns" => fv_nunknowns(prob), "nnz" => length(cont.solver.nzval),
        "seconds_continuation" => tcont, "gpu_used_bytes_before" => mem0, "gpu_used_bytes_after" => mem1)
    res["attempt_reasons"] = [string(a.reason) for a in cont.attempts]
    res["attempt_sigmas"] = [a.sigma for a in cont.attempts]
    if !cont.converged
        a = first(cont.attempts)
        a.stats === nothing || (res["first_attempt_last_equation_inf_scaled"] = last(a.stats.equation_inf);
                                res["first_attempt_alpha_history"] = a.stats.alpha)
        return res
    end
    Uh = Array(cont.U); u, v, p, T = fv_fields(prob, Uh)
    hp = host(prob); I1, nu = cell_fields(hp, Uh, law, cs.theta, T)
    res["ux"] = compare(u, cs.ux, fluid); res["uy"] = compare(v, cs.uy, fluid, maximum(abs, cs.ux[fluid]))
    res["flow_rate_ratio_fv_over_lbm"] = mean(u[fluid]) / mean(cs.ux[fluid])
    res["dT"] = compare(T .- Tw, cs.T .- Tw, fluid)
    m_fv = mean_rise(T, fluid, Tw); m_lbm = mean_rise(cs.T, fluid, Tw)
    res["m_fv"] = m_fv; res["m_lbm"] = m_lbm; res["m_ratio_fv_over_lbm"] = m_fv / m_lbm
    res["dT_max_fv"] = maximum((T .- Tw)[fluid]); res["dT_max_lbm"] = maximum((cs.T .- Tw)[fluid])
    res["nu_vs_nu_from_gradient_base"] = compare(nu, cs.nu, fluid)
    res["I1_vs_I1_from_gradient_base"] = compare(I1, cs.I1, fluid)
    P = sum(KrakeNN._fv_energy_heating(Uh, hp.tag, hp.nu, hp.bvel, hp.grid, i, j, hp.visc, sigma)
            for j in 1:cs.Ny, i in 1:cs.Nx if fluid[i, j])
    Q = bottom_flux(T, fluid, Tw, alpha)
    res["power_balance_fv"] = Dict{String,Any}("integral_heating" => P, "bottom_flux" => Q,
                                                "relative_imbalance" => (Q - P) / P)
    Plbm = sum((sigma .* cs.nu .* 2 .* cs.I1)[fluid]); Qlbm = bottom_flux(cs.T, fluid, Tw, alpha)
    res["power_balance_lbm_T_in_fv_operator"] = Dict{String,Any}("integral_heating" => Plbm,
        "bottom_flux" => Qlbm, "relative_imbalance" => (Qlbm - Plbm) / Plbm)
    # --- adjoint (T4) of J = 1/2 m^2  =>  dJ/dtheta = m dm/dtheta
    map = FVLawParameters(8)
    obj = fv_mean_temperature(prob; T_wall = Tw, target = 0.0)
    r, tadj = timed(() -> fv_steady_gradient(cont.solver, cont.U, copy(cs.theta), obj; params = map))
    dm_fv = r.grad ./ m_fv
    res["adjoint_seconds"] = tadj; res["adjoint_residual_inf"] = r.residual_inf
    res["m_from_adjoint_objective"] = sqrt(2 * r.J)
    res["dm_dtheta_fv"] = dm_fv
    if cs.gradient !== nothing
        dm_l = Float64.(cs.gradient["dm_dtheta"])
        res["dm_dtheta_lbm"] = dm_l
        res["dm_dtheta_componentwise_rel_diff"] = abs.(dm_fv .- dm_l) ./ abs.(dm_l)
        res["dm_dtheta_sign_agree"] = sign.(dm_fv) .== sign.(dm_l)
        res["dm_dtheta_norm_rel_diff"] = norm(dm_fv .- dm_l) / norm(dm_l)
        res["parameter_names"] = String.(cs.gradient["parameter_names"])
    end
    if fd
        sfd = fv_newton_solver(prob; tl.to, tl.linsolve)
        function mval(theta)
            fv_set_params!(sfd, map, theta)
            V = copy(cont.U)
            st = fv_newton!(sfd, V; maxit = 30)
            for _ in 1:2
                fv_newton_finalize!(sfd, V); fv_residual!(sfd.R, V, sfd.prob)
                lin_solve!(sfd.dU, sfd.cache, sfd.R); V .-= sfd.dU
            end
            _, _, _, Tv = fv_fields(prob, Array(V))
            return mean_rise(Tv, fluid, Tw), st.converged
        end
        fdres = Dict{String,Any}()
        for (k, name) in ((1, "eta0"), (3, "beta"))
            h = 1e-3 * cs.theta[k]
            tp = copy(cs.theta); tp[k] += h; tm = copy(cs.theta); tm[k] -= h
            (mp, cp), (mm, cm) = mval(tp), mval(tm)
            fdv = (mp - mm) / 2h
            fdres[name] = Dict{String,Any}("m_plus" => mp, "m_minus" => mm, "converged" => cp && cm,
                "relative_step" => 1e-3, "dm_dtheta_fd" => fdv, "dm_dtheta_adjoint" => dm_fv[k],
                "relative_difference" => abs(fdv - dm_fv[k]) / abs(fdv))
            cs.gradient === nothing || (fdres[name]["lbm_fd"] = cs.gradient["finite_difference"][name]["dm_dtheta_fd"])
        end
        res["finite_difference"] = fdres
    end
    outdir === nothing || write_csv(joinpath(outdir, "stage3_fields.csv"),
        ["fluid" => Float64.(fluid), "ux_fv" => u, "ux_lbm" => cs.ux, "uy_fv" => v, "uy_lbm" => cs.uy,
         "T_fv" => T, "T_lbm" => cs.T, "nu_fv" => nu, "nu_lbm" => cs.nu, "I1_fv" => I1, "I1_lbm" => cs.I1,
         "p_fv" => p], cs.Nx, cs.Ny)
    return res
end

# ------------------------------------------------------------------- thresholds
# Acceptance thresholds for FV vs LBM on the E5 six-cylinder cell, full coupling
# (stage 3), grid 600x200. Decided 2026-11-08 from the job measurement of
# 2026-11-08 (Aqua H100, 600x200) with a margin; they do not apply to other grids
# (300x100 is reported for the record, it does not meet them). Not tuned on any
# later run.
const THRESHOLDS_GRID = "600x200"
const THRESHOLDS_DATE = "2026-11-08"
const THRESHOLDS_SOURCE = "job measurement 2026-11-08 (H100, 600x200, stage 3), with margin"
const THRESHOLD_M_RATIO = 0.02       # |m_FV / m_LBM - 1|
const THRESHOLD_FLOW_RATIO = 0.02    # |flow-rate ratio - 1|
const THRESHOLD_GRADIENT_NORM = 0.03 # ||dm/dtheta_FV - dm/dtheta_LBM|| / ||dm/dtheta_LBM||
const THRESHOLD_GRADIENT_COMPONENT = 0.06   # each component, relative
# plus: the 8 signs of dm/dtheta agree.

"""Threshold table for grid `grid`, evaluated on the stage-3 result `st3`.

The limits are always written. `value` and `pass` are written only when stage 3
converged and carries the LBM gradient (`evaluated = true`), and `pass` only when
the thresholds apply (grid 600x200); other grids get `applies = false`.
"""
function thresholds(grid::AbstractString, st3)
    keys3 = ("m_ratio_fv_over_lbm", "flow_rate_ratio_fv_over_lbm", "dm_dtheta_norm_rel_diff",
             "dm_dtheta_componentwise_rel_diff", "dm_dtheta_sign_agree")
    evaluated = st3 !== nothing && all(haskey.(Ref(st3), keys3))
    applies = grid == THRESHOLDS_GRID
    function item(value, limit)
        d = Dict{String,Any}("limit" => limit, "applies" => applies)
        evaluated && (d["value"] = value; applies && (d["pass"] = value <= limit))
        return d
    end
    g(k, f = identity) = evaluated ? f(st3[k]) : nothing
    t = Dict{String,Any}("applies" => applies, "evaluated" => evaluated,
        "grid_required" => THRESHOLDS_GRID, "grid" => grid,
        "date" => THRESHOLDS_DATE, "source" => THRESHOLDS_SOURCE,
        "m_ratio_deviation" => item(g("m_ratio_fv_over_lbm", x -> abs(x - 1)), THRESHOLD_M_RATIO),
        "flow_rate_deviation" => item(g("flow_rate_ratio_fv_over_lbm", x -> abs(x - 1)), THRESHOLD_FLOW_RATIO),
        "gradient_norm_rel_diff" => item(g("dm_dtheta_norm_rel_diff"), THRESHOLD_GRADIENT_NORM),
        "gradient_worst_component_rel_diff" => item(g("dm_dtheta_componentwise_rel_diff", maximum),
                                                    THRESHOLD_GRADIENT_COMPONENT))
    signs = Dict{String,Any}("limit" => evaluated ? length(st3["dm_dtheta_sign_agree"]) : 8, "applies" => applies)
    if evaluated
        signs["value"] = count(st3["dm_dtheta_sign_agree"])
        applies && (signs["pass"] = signs["value"] == signs["limit"])
    end
    t["gradient_signs_agree"] = signs
    if applies && evaluated
        items = ("m_ratio_deviation", "flow_rate_deviation", "gradient_norm_rel_diff",
                 "gradient_worst_component_rel_diff", "gradient_signs_agree")
        t["all_pass"] = all(t[k]["pass"] for k in items)
    end
    return t
end

# ------------------------------------------------------------------- main
function write_summary(path, summary)
    open(path, "w") do io
        TOML.print(io, summary; sorted = true)
    end
end

function main()
    grid = get(ENV, "KRAKEN_E5_GRID", "300x100")
    root = normpath(joinpath(@__DIR__, "..", ".."))
    indir = get(ENV, "KRAKEN_E5_EXPORT", joinpath(root, "tmp", "e5_export", grid))
    outdir = get(ENV, "KRAKEN_E5_OUT", joinpath(root, "tmp", "e5_compare_out", "$(grid)_$(BACKEND)"))
    stages = parse.(Int, split(get(ENV, "KRAKEN_E5_STAGES", "1,2,3"), ","))
    fd = get(ENV, "KRAKEN_E5_FD", "1") != "0"
    mkpath(outdir)
    cs = load_case(indir)
    cs.grid == grid || error("export grid $(cs.grid) differs from KRAKEN_E5_GRID=$grid")
    summary = Dict{String,Any}("grid" => grid, "backend" => BACKEND, "export_dir" => indir,
        "init" => get(ENV, "KRAKEN_E5_INIT", "zero"), "julia" => string(VERSION))
    if BACKEND == "cuda"
        summary["device"] = CUDA.name(CUDA.device())
        summary["gpu_total_bytes"] = Int(CUDA.total_memory())
    end
    path = joinpath(outdir, "summary.toml")
    summary["export_check"] = check_export(cs); write_summary(path, summary)
    @info "export check done" outdir
    for k in (1, 2, 3)
        k in stages || continue
        name = "stage$k"
        @info "running $name"
        summary[name] = k == 1 ? stage1(cs; outdir) : k == 2 ? stage2(cs; outdir) : stage3(cs; outdir, fd)
        write_summary(path, summary)
        @info "$name done" converged = get(summary[name], "converged", missing)
    end
    summary["thresholds"] = thresholds(grid, get(summary, "stage3", nothing))
    write_summary(path, summary)
    @info "summary written" path
    return summary
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    E5Compare.main()
end
