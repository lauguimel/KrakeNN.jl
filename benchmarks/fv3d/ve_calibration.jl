# Identifiability and calibration of a Giesekus fluid on the square-duct slice
# (plan "discrete viscoelastic adjoints", step F6). Slice 1 x N x N of
# fv3d_conformation_gate_setup.jl (ve_slice: h = 1/N, walls y and z, imposed
# mean axial velocity 1, so De = lambda), nu_s = nu_p = 1, alpha = FV3D_VE_CAL_ALPHA
# (0.4, the RheoTool value). Linear solver passed explicitly: direct (cuDSS on
# GPU, UMFPACK on CPU, fv3d_linsolve()); :auto is not used and no iterative
# refinement is added.
#
# Observations: the velocity at the cell centres of the section (x-normal stereo
# PIV plane at x = h/2), axial u and secondary (v, w), and the body force Fx.
# Data sets: (i) u, (ii) u + (v, w), (iii) u + (v, w) + Fx.
#
# Parts (FV3D_VE_CAL_PARTS, comma-separated):
# ident (F6.1), at each De of FV3D_VE_CAL_DE (0.1,1.0), N = FV3D_VE_CAL_N (48):
#   S = d(observations)/dq by tangent solves, q = (log lambda, log nu_p, log nu_s,
#   alpha/0.5) (full) and q = (log lambda, alpha/0.5) (reduced, nu fixed: the
#   columns 1 and 4 of the same S). (a) singular values of the unweighted S and
#   ratio smallest/largest per data set and problem; (b) Fisher F = S' W S with
#   W = 1/sigma^2, sigma_u = sigma_vw = sigma (1 % then 5 % of max |(v, w)|),
#   sigma_Fx = 1 % of Fx; standard deviations of lambda (relative) and alpha from
#   F^-1 converted from q by the Jacobian diag(lambda, nu_p, nu_s, 0.5).
#   Reported only (no gate).
# twin (F6.2), at each De, N = FV3D_VE_CAL_N: data from KrakeNN at (lambda = De,
#   alpha), data set FV3D_VE_CAL_TWIN_SET (3), weights of the 1 % scenario.
#   Levenberg-Marquardt on q = (log lambda, alpha/0.5) from (2 lambda, 0.1)
#   (state by continuation from rest at alpha = 0.1); a trial step whose Newton
#   solve fails, or alpha outside (0, 0.5), is rejected (damping x 4).
#   Gate (sigma = 0): |alpha^ - alpha| <= 1e-6 and |lambda^/lambda - 1| <= 1e-6.
#   Then FV3D_VE_CAL_DRAWS (10) noisy data sets per scenario (1 %, 5 %), each fit
#   started from the truth: bias, dispersion against the Laplace standard
#   deviation (from F at the truth) and coverage of +-2 Laplace sigma (reported).
# rheotool (F6.3): needs FV3D_VE_RHEOTOOL_DIR, the folder of CSV files written by
#   ve_rheotool_export.py (one per De, named De<De>.csv, e.g. De0.1.csv). Fit of
#   (lambda, alpha) to the velocities only (u, v, w on the 50 x 50 centres of the
#   lower-left quarter), KrakeNN interpolated there with the PIV operator, at
#   N = FV3D_VE_RHEOTOOL_N (48,100,200), equal weight on every component, the 50
#   samples of the row and of the column nearest the walls (0.005 from the wall,
#   inside the first half cell at N = 48 where the operator extrapolates by a
#   constant) given weight 0 at every N. Start from the nominal (De, 0.4).
#   Reported: lambda^, alpha^ per N, gap to (De, 0.4), differences between N, and
#   a Richardson estimate (order p from the three N when the differences are
#   monotone). The RheoTool fields are one instant (t = 30); their steadiness is
#   not shown.
#
#   FV3D_BACKEND=cuda julia --project=<env with CUDA, CUDSS> benchmarks/fv3d/ve_calibration.jl
#   FV3D_BACKEND=cpu FV3D_VE_CAL_N=8 FV3D_VE_CAL_DRAWS=2 julia --project benchmarks/fv3d/ve_calibration.jl  (smoke)
# Options: FV3D_VE_CAL_PARTS (ident,twin), FV3D_VE_CAL_DE, FV3D_VE_CAL_N,
# FV3D_VE_CAL_ALPHA, FV3D_VE_CAL_TWIN_SET, FV3D_VE_CAL_DRAWS, FV3D_VE_RHEOTOOL_DIR,
# FV3D_VE_RHEOTOOL_N, FV3D_OUT.
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
include(joinpath(@__DIR__, "fv3d_common.jl"))
include(joinpath(@__DIR__, "..", "..", "test", "analytical", "fv3d_conformation_gate_setup.jl"))

const SCRIPT = "ve_calibration"
const PARTS = Symbol.(split(get(ENV, "FV3D_VE_CAL_PARTS", "ident,twin"), ','; keepempty = false))
const DES = parse.(Float64, split(get(ENV, "FV3D_VE_CAL_DE", "0.1,1.0"), ','; keepempty = false))
const NCAL = parse(Int, get(ENV, "FV3D_VE_CAL_N", "48"))
const ALPHA = parse(Float64, get(ENV, "FV3D_VE_CAL_ALPHA", "0.4"))
const TWIN_SET = parse(Int, get(ENV, "FV3D_VE_CAL_TWIN_SET", "3"))
const NDRAWS = parse(Int, get(ENV, "FV3D_VE_CAL_DRAWS", "10"))
const RT_DIR = get(ENV, "FV3D_VE_RHEOTOOL_DIR", "")
const RT_NS = parse.(Int, split(get(ENV, "FV3D_VE_RHEOTOOL_N", "48,100,200"), ','; keepempty = false))
const NOISE = (0.01, 0.05)        # sigma_u = sigma_vw = NOISE * max |(v, w)|
const FX_NOISE = 0.01             # sigma_Fx = FX_NOISE * |Fx|
const GATE_TOL = 1e-6
const SETNAMES = ("i axial", "ii +secondary", "iii +Fx")

full_map() = FVRelaxationParameters(:lambda, :nu_p, :nu_s, :theta)   # theta = (lambda, nu_p, nu_s, alpha)
red_map() = FVRelaxationParameters(:lambda, :theta)                  # theta = (lambda, alpha)
θ_of_q(q) = [exp(q[1]), 0.5 * q[2]]
q_of_θ(θ) = [log(θ[1]), θ[2] / 0.5]
giesekus(lam, a) = FVConformation3D(; lambda = lam, nu_p = 1.0, alpha = a)
ramp_to(lam) = [filter(<(lam), [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]); lam]

# continuation from rest on solver s (or a new one) up to (lam, a); returns s, U
function slice_state(N, lam, a; s = nothing)
    to = fv3d_to()
    if s === nothing
        p = ve_slice(giesekus(ramp_to(lam)[1], a); N)
        s = fv_newton_solver(p; to, linsolve = fv3d_linsolve())
    end
    U = to(ve_rest(KrakeNN._fv_host_problem(s.prob)))
    for l in ramp_to(lam)
        fv_set_conformation!(s, giesekus(l, a))
        ve_solve!(s, U)
    end
    fv_set_params!(s, red_map(), [lam, a])
    ve_solve!(s, U)
    fv_check_conformation_domain(s.prob, U)
    return s, U
end

# ------------------------------------------------------------------ observations
struct Obs
    op::Any           # host PIV operator, x-normal stereo: samples (v, w, u)
    n::Int            # samples per component
    iF::Int
end
function Obs(ph, ys, zs)
    g = ph.grid
    plane = PIVPlane(; normal = :x, position = 0.5 * g.h, xs = ys, ys = zs, stereo = true)
    op = piv_operator(plane, zeros(g.Nx, g.Ny, g.Nz); h = (g.h, g.h, g.h))
    return Obs(op, length(ys) * length(zs), KrakeNN._fv_force_index(ph))
end
cell_obs(ph) = (c = ((1:ph.grid.Ny) .- 0.5) .* ph.grid.h; Obs(ph, c, c))

# observation vector [u; v; w; Fx] of a host state or tangent X (linear in X)
function observe(ph, o::Obs, X::Vector{Float64})
    u, v, w, _ = fv_fields(ph, X)
    sv, sw, su = piv_apply(o.op, (u, v, w))
    return [vec(su); vec(sv); vec(sw); X[o.iF]]
end
rows(o::Obs, set) = set == 1 ? (1:o.n) : set == 2 ? (1:3o.n) : (1:3o.n+1)

# columns d(observations)/d theta_k by tangent solves at (U, θ)
function sens(s, U, m, θ, ph, o)
    ws = fv_adjoint_workspace(s, m; chunk = 4)
    cols = map(eachindex(θ)) do k
        dU = fv_steady_tangent(s, U, θ, Float64.(eachindex(θ) .== k); params = m, ws, finalize = k == 1)
        observe(ph, o, Vector{Float64}(Array(dU)))
    end
    return reduce(hcat, cols)
end

function noise_sigmas(o::Obs, y, frac)
    vw = maximum(hypot.(y[o.n+1:2o.n], y[2o.n+1:3o.n]))
    return [fill(frac * vw, 3o.n); FX_NOISE * abs(y[end])]
end

# Laplace standard deviations (lambda relative, alpha) from F = A'A, A = S_q ./ sigma
function laplace_sd(A, idx_lam, idx_alpha)
    F = Symmetric(A' * A)
    ev = eigvals(F)
    minimum(ev) <= 1e-14 * maximum(ev) && return (NaN, NaN, maximum(ev) / max(minimum(ev), 1e-300))
    C = inv(F)
    return (sqrt(C[idx_lam, idx_lam]), 0.5 * sqrt(C[idx_alpha, idx_alpha]), maximum(ev) / minimum(ev))
end

# ------------------------------------------------------------------ F6.1
function run_ident(De)
    out = Dict{String,Any}()
    s, U = slice_state(NCAL, De, ALPHA)
    ph = KrakeNN._fv_host_problem(s.prob)
    o = cell_obs(ph)
    θf = [De, 1.0, 1.0, ALPHA]
    fv_set_params!(s, full_map(), θf); ve_solve!(s, U)
    y = observe(ph, o, Vector{Float64}(Array(U)))
    Sθ = sens(s, U, full_map(), θf, ph, o)
    Sq = Sθ .* [De 1.0 1.0 0.5]                    # d/dq = d/dθ * dθ/dq
    @printf("  ident De %.2f N %d: Fx %.6e  max|u| %.4f  max|(v,w)| %.3e\n", De, NCAL, y[end],
            maximum(abs, y[1:o.n]), maximum(hypot.(y[o.n+1:2o.n], y[2o.n+1:3o.n])))
    for set in 1:3, (pname, cols) in (("full", 1:4), ("reduced", [1, 4]))
        S = Sq[rows(o, set), cols]
        sv = svdvals(S)
        ratio = minimum(sv) / maximum(sv)
        @printf("  ident De %.2f %-13s %-7s singular values %s  ratio %.2e%s\n", De, SETNAMES[set], pname,
                join((@sprintf("%.3e", x) for x in sv), " "), ratio, ratio < 1e-6 ? "  (< 1e-6: not all identifiable)" : "")
        key = "De$(De)_set$(set)_$pname"
        out[key * "_sv"] = sv; out[key * "_ratio"] = ratio
        for frac in NOISE
            sig = noise_sigmas(o, y, frac)[rows(o, set)]
            sdl, sda, cond = laplace_sd(S ./ sig, 1, length(cols))
            @printf("  ident De %.2f %-13s %-7s noise %2.0f %%: sd(lambda)/lambda %.2e  sd(alpha) %.2e  cond(F) %.2e\n",
                    De, SETNAMES[set], pname, 100frac, sdl, sda, cond)
            out[key * "_noise$(frac)_sd_lambda_rel"] = sdl; out[key * "_noise$(frac)_sd_alpha"] = sda
        end
    end
    fv3d_release_all!(s)
    return out
end

# ------------------------------------------------------------------ fitting
# Levenberg-Marquardt on q = (log lambda, alpha/0.5); residual r = wsq .* (y(q) - yd)
# on the rows `r_idx`. Returns q, U, cost, iterations, history.
function lm_fit!(s, U0, q0, ph, o, yd, wsq, r_idx; maxit = 40)
    m = red_map()
    function evaluate(q, Ustart)
        0 < q[2] < 1 || return (false, Ustart, Float64[])
        fv_set_params!(s, m, θ_of_q(q))
        Ut = copy(Ustart)
        try
            ve_solve!(s, Ut)
            fv_check_conformation_domain(s.prob, Ut)
        catch err
            err isa InterruptException && rethrow()
            return (false, Ustart, Float64[])
        end
        return (true, Ut, wsq .* (observe(ph, o, Vector{Float64}(Array(Ut)))[r_idx] .- yd))
    end
    q = copy(q0)
    ok, U, r = evaluate(q, U0)
    ok || error("LM: Newton failed at the starting point")
    cost = 0.5 * sum(abs2, r)
    μ = 1e-3; it = 0; hist = [(0, θ_of_q(q)..., cost)]
    while it < maxit
        it += 1
        θ = θ_of_q(q)
        fv_set_params!(s, m, θ)
        A = wsq .* (sens(s, U, m, θ, ph, o)[r_idx, :] .* [θ[1] 0.5])
        g = A' * r; H = A' * A
        accepted = false; δ = zeros(2)
        for _ in 1:15
            δ = -((H + μ * Diagonal(diag(H))) \ g)
            okn, Un, rn = evaluate(q .+ δ, U)
            if okn && 0.5 * sum(abs2, rn) < cost
                q = q .+ δ; U = Un; r = rn; cost = 0.5 * sum(abs2, rn)
                μ = max(μ / 3, 1e-12); accepted = true
                break
            end
            μ *= 4
        end
        push!(hist, (it, θ_of_q(q)..., cost))
        # no decrease possible (round-off floor) or step below 1e-12 in q
        (!accepted || norm(δ, Inf) < 1e-12) && break
    end
    fv_set_params!(s, m, θ_of_q(q))
    return q, U, cost, it, hist
end

# ------------------------------------------------------------------ F6.2
function run_twin(De, fails)
    out = Dict{String,Any}()
    θt = [De, ALPHA]; qt = q_of_θ(θt)
    s, Ut = slice_state(NCAL, De, ALPHA)
    ph = KrakeNN._fv_host_problem(s.prob)
    o = cell_obs(ph)
    ri = rows(o, TWIN_SET)
    y = observe(ph, o, Vector{Float64}(Array(Ut)))
    yd = y[ri]
    wsq = 1 ./ noise_sigmas(o, y, NOISE[1])[ri]
    # noise-free fit from (2 lambda, 0.1)
    _, U0 = slice_state(NCAL, 2De, 0.1; s)
    t0 = time()
    q, _, cost, it, hist = lm_fit!(s, U0, q_of_θ([2De, 0.1]), ph, o, yd, wsq, ri)
    t = time() - t0
    θh = θ_of_q(q)
    el = abs(θh[1] / De - 1); ea = abs(θh[2] - ALPHA)
    for h in hist
        @printf("  twin De %.2f LM %2d  lambda %.12f  alpha %.12f  cost %.3e\n", De, h...)
    end
    okg = el <= GATE_TOL && ea <= GATE_TOL
    @printf("  twin De %.2f %s: lambda^/lambda - 1 = %.2e, alpha^ - alpha = %.2e (tol %.0e), %d LM iterations, %.1f s  %s\n",
            De, SETNAMES[TWIN_SET], θh[1] / De - 1, θh[2] - ALPHA, GATE_TOL, it, t, okg ? "ok" : "FAIL")
    okg || push!(fails, @sprintf("twin De %.2f: |lambda^/lambda-1| %.2e, |alpha^-alpha| %.2e > %.0e", De, el, ea, GATE_TOL))
    out["noise_free"] = Dict("lambda" => θh[1], "alpha" => θh[2], "lambda_relerr" => el, "alpha_err" => ea,
                             "iterations" => it, "seconds" => t, "cost" => cost)
    # noisy draws from the truth
    rng = MersenneTwister(2026)
    fv_set_params!(s, red_map(), θt); ve_solve!(s, Ut)
    St = sens(s, Ut, red_map(), θt, ph, o)[ri, :] .* [De 0.5]
    for frac in NOISE
        sig = noise_sigmas(o, y, frac)[ri]
        sdl, sda, _ = laplace_sd(St ./ sig, 1, 2)
        est = zeros(NDRAWS, 2); nfail = 0
        for d in 1:NDRAWS
            ydn = yd .+ sig .* randn(rng, length(yd))
            try
                qn, _, _, _, _ = lm_fit!(s, Ut, qt, ph, o, ydn, 1 ./ sig, ri)
                est[d, :] = θ_of_q(qn)
            catch err
                err isa InterruptException && rethrow()
                est[d, :] .= NaN; nfail += 1
            end
        end
        good = vec(all(isfinite, est; dims = 2))
        E = est[good, :]
        bl = sum(E[:, 1] ./ De .- 1) / size(E, 1); ba = sum(E[:, 2] .- ALPHA) / size(E, 1)
        dl = sqrt(sum(abs2, E[:, 1] ./ De .- 1 .- bl) / max(size(E, 1) - 1, 1))
        da = sqrt(sum(abs2, E[:, 2] .- ALPHA .- ba) / max(size(E, 1) - 1, 1))
        cl = count(abs.(log.(E[:, 1] ./ De)) .<= 2sdl) / size(E, 1)
        ca = count(abs.(E[:, 2] .- ALPHA) .<= 2sda) / size(E, 1)
        @printf("  twin De %.2f noise %2.0f %% (%d draws, %d failed): lambda bias %+.2e sd %.2e (Laplace %.2e) cover %.0f %%;  alpha bias %+.2e sd %.2e (Laplace %.2e) cover %.0f %%\n",
                De, 100frac, NDRAWS, nfail, bl, dl, sdl, 100cl, ba, da, sda, 100ca)
        out["noise$(frac)"] = Dict("lambda_bias_rel" => bl, "lambda_sd_rel" => dl, "lambda_laplace_rel" => sdl,
                                   "lambda_cover2" => cl, "alpha_bias" => ba, "alpha_sd" => da,
                                   "alpha_laplace" => sda, "alpha_cover2" => ca, "failed" => nfail)
    end
    fv3d_release_all!(s)
    return out
end

# ------------------------------------------------------------------ F6.3
function read_rheotool_csv(path)
    meta = Dict{String,Float64}(); rowsv = Vector{NTuple{5,Float64}}()
    for line in eachline(path)
        if startswith(line, "#")
            parts = split(strip(line[2:end]))
            length(parts) == 2 && (v = tryparse(Float64, parts[2])) !== nothing && (meta[parts[1]] = v)
        elseif !startswith(line, "y,")
            f = parse.(Float64, split(line, ','))
            push!(rowsv, (f[1], f[2], f[3], f[4], f[5]))
        end
    end
    ys = sort(unique(first.(rowsv))); zs = sort(unique(getindex.(rowsv, 2)))
    iy = Dict(v => i for (i, v) in enumerate(ys)); iz = Dict(v => i for (i, v) in enumerate(zs))
    u = fill(NaN, length(ys), length(zs)); v = copy(u); w = copy(u)
    for r in rowsv
        i, j = iy[r[1]], iz[r[2]]
        u[i, j] = r[3]; v[i, j] = r[4]; w[i, j] = r[5]
    end
    all(isfinite, u) || error("$path: incomplete grid")
    return (; ys, zs, u, v, w, meta)
end

# Richardson estimate from (h_k, x_k), h decreasing: order p from the three
# values when the two differences have the same sign, extrapolated value
function richardson(hs, xs)
    d1 = xs[1] - xs[2]; d2 = xs[2] - xs[3]
    (length(xs) == 3 && d1 * d2 > 0) || return (NaN, NaN)
    f(p) = (hs[1]^p - hs[2]^p) / (hs[2]^p - hs[3]^p) - d1 / d2
    lo, hi = 0.1, 8.0
    f(lo) * f(hi) > 0 && return (NaN, NaN)
    for _ in 1:200
        mid = (lo + hi) / 2
        f(lo) * f(mid) <= 0 ? (hi = mid) : (lo = mid)
    end
    p = (lo + hi) / 2
    return p, xs[3] - d2 * hs[3]^p / (hs[2]^p - hs[3]^p)
end

function run_rheotool(De)
    out = Dict{String,Any}()
    path = joinpath(RT_DIR, "De$(De).csv")
    isfile(path) || (println("  rheotool De $De: no file De$(De).csv in FV3D_VE_RHEOTOOL_DIR, skipped"); return out)
    d = read_rheotool_csv(path)
    nominal = (get(d.meta, "lambda", De), get(d.meta, "alpha", 0.4))
    n = length(d.ys) * length(d.zs)
    yd = [vec(d.u); vec(d.v); vec(d.w)]
    keep = [(y >= 0.5 / 48 && z >= 0.5 / 48) for y in d.ys, z in d.zs]       # drop the wall row and column
    wsq = repeat(Float64.(vec(keep)), 3)
    ri = 1:3n
    est = Dict{Int,Vector{Float64}}()
    for N in RT_NS
        t0 = time()
        begin
            s, U = slice_state(N, nominal[1], nominal[2])
            ph = KrakeNN._fv_host_problem(s.prob)
            o = Obs(ph, d.ys, d.zs)
            q, Uf, cost, it, _ = lm_fit!(s, U, q_of_θ(collect(nominal)), ph, o, yd, wsq, ri)
            θh = θ_of_q(q)
            ym = observe(ph, o, Vector{Float64}(Array(Uf)))
            kk = vec(keep)
            ra = norm((ym[1:n] .- yd[1:n])[kk]) / norm(yd[1:n][kk])
            rs = norm([(ym[n+1:2n] .- yd[n+1:2n])[kk]; (ym[2n+1:3n] .- yd[2n+1:3n])[kk]]) /
                 norm([yd[n+1:2n][kk]; yd[2n+1:3n][kk]])
            fv3d_release_all!(s)
        end
        t = time() - t0
        est[N] = θh
        @printf("  rheotool De %.2f N %3d: lambda^ %.6f (%+.3e rel. to %.3g)  alpha^ %.6f (%+.3e to %.3g)  misfit axial %.2e secondary %.2e  %d LM it, %.1f s\n",
                De, N, θh[1], θh[1] / nominal[1] - 1, nominal[1], θh[2], θh[2] - nominal[2], nominal[2], ra, rs, it, t)
        out["N$N"] = Dict("lambda" => θh[1], "alpha" => θh[2], "misfit_axial" => ra, "misfit_secondary" => rs,
                          "cost" => cost, "iterations" => it, "seconds" => t)
        flush(stdout)
    end
    Ns = sort(collect(keys(est)))
    for k in 2:length(Ns)
        a, b = est[Ns[k-1]], est[Ns[k]]
        @printf("  rheotool De %.2f N %d -> %d: d lambda^/lambda %+.3e  d alpha^ %+.3e\n", De, Ns[k-1], Ns[k],
                b[1] / a[1] - 1, b[2] - a[2])
    end
    if length(Ns) == 3
        hs = 1 ./ Ns
        for (i, name) in ((1, "lambda"), (2, "alpha"))
            p, x0 = richardson(hs, [est[N][i] for N in Ns])
            @printf("  rheotool De %.2f Richardson %s: order %.2f, h -> 0 value %.6f (gap to nominal %+.3e)\n",
                    De, name, p, x0, i == 1 ? x0 / nominal[1] - 1 : x0 - nominal[2])
            out["richardson_$(name)"] = Dict("order" => p, "value" => x0)
        end
    end
    return out
end

function main()
    fv3d_banner(SCRIPT)
    println("parts ", join(PARTS, ","), "  De ", join(DES, ","), "  N ", NCAL, "  alpha ", ALPHA,
            "  linear solver direct (", FV3D_GPU ? "cuDSS" : "UMFPACK", ")")
    data = Dict{String,Any}("N" => NCAL, "alpha" => ALPHA, "De" => DES)
    failures = String[]
    for part in PARTS
        part in (:ident, :twin, :rheotool) || error("unknown part $part (ident, twin, rheotool)")
        part === :rheotool && isempty(RT_DIR) && (println("  rheotool: FV3D_VE_RHEOTOOL_DIR not set, skipped"); continue)
        for De in DES
            try
                data["$(part)_De$(De)"] = part === :ident ? run_ident(De) :
                                          part === :twin ? run_twin(De, failures) : run_rheotool(De)
            catch err
                msg = first(sprint(showerror, err), 300)
                println("  $part De $De: ERROR ", msg)
                data["$(part)_De$(De)"] = Dict("error" => msg)
                push!(failures, "$part De $De: error")
            end
            fv3d_write(fv3d_out(SCRIPT), data)
            flush(stdout)
        end
    end
    return fv3d_verdict(SCRIPT, failures)
end

exit(main())
