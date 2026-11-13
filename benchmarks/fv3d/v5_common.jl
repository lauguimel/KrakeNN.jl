# Shared code of the V5 science case of the 3D steady FV track: recover a
# non-Newtonian viscosity law in 3D from 2D PIV planes (README.md, section V5).
#
# Included by every v5 stage script AFTER the script has loaded CUDA/CUDSS at
# top level (FV3D_BACKEND=cuda, the default). Guard in the stage scripts:
#     isdefined(@__MODULE__, :FV5_COMMON_LOADED) || include(joinpath(@__DIR__, "v5_common.jl"))
#
# Units. FV units: duct height Ly = 1 (h = 1/Ny), viscosity ν0, body force f.
# The LBM truth runs in lattice units (h_lat = dt_lat = 1) with the time step
# dt = ν_lat h² / ν0 (ν_lat = FV5_NU_LAT, lattice viscosity at rest): a lattice
# value times its scale gives the FV value (velocity h/dt, viscosity h²/dt,
# acceleration h/dt², shear rate 1/dt, Carreau λ: λ_lat = λ / dt). One LBM node
# per FV cell, walls on the FV cell faces (q_w = 1/2), same voxel obstacle with
# halfway bounce-back: LBM node (i, j, k) and FV cell (i, j, k) have the same
# centre, so the truth is injected cell by cell (no interpolation) and the
# same PIV operator is applied to the truth and to the FV state.
#
# Environment (stage a reads the case from it and writes it to
# <FV5_OUT>/truth/truth.toml; later stages read the case from that file):
#   FV3D_BACKEND     cuda (default) | cpu
#   FV5_OUT          output directory (default benchmarks/fv3d/results/v5_<backend>)
#   FV5_N            Ny = Nz = N, Nx = 2N (default 48 on cuda, 12 on cpu)
#   FV5_NX/NY/NZ     override one size
#   FV5_NU0 (0.1), FV5_NUINF_RATIO (0.1: ν∞ = 0.1 ν0), FV5_LAMBDA (5),
#   FV5_A (2), FV5_NINDEX (0.5): true Carreau–Yasuda law, FV units
#   FV5_UREF (0.2)   target mean velocity of the Newtonian duct (sets f)
#   FV5_TAU_MIN (0.56) smallest LBM relaxation time, reached at ν = ν∞; sets
#                    FV5_NU_LAT = (τ_min − 1/2)/(3 ν∞/ν0) (0.2), the lattice
#                    viscosity at rest (sets dt); FV5_NU_LAT overrides it
#   FV5_LINSOLVE     krylov (default: GMRES + P3 multigrid block SIMPLE) | direct
#   FV5_ETA (1e-6), FV5_ETA_T (1e-10)  GMRES tolerances (forward, transposed)
#   FV5_ATOL_REL (1e-9)  Newton ‖R‖∞ tolerance relative to f h³
include(joinpath(@__DIR__, "fv3d_common.jl"))
include(joinpath(@__DIR__, "..", "validation", "gnf_calibration", "bfgs.jl"))
using Random, Statistics
const FV5_COMMON_LOADED = true

fv5_env(name, default) = get(ENV, name, string(default))
fv5_int(name, default) = parse(Int, fv5_env(name, default))
fv5_float(name, default) = parse(Float64, fv5_env(name, default))
fv5_outdir() = abspath(get(ENV, "FV5_OUT", joinpath(@__DIR__, "results", "v5_" * FV3D_BACKEND)))

# ------------------------------------------------------------------ case
# The LBM is only stable while its relaxation time τ = 1/2 + 3 ν_lat stays
# clear of 1/2, and nothing in the GNF kernel bounds ν from below: the
# viscosity floor is the law's ν∞. The lattice viscosity at rest is therefore
# chosen from the floor, ν_lat = (τ_min − 1/2) / (3 ν∞/ν0), so that every cell
# has τ ≥ τ_min whatever its shear rate (FV5_TAU_MIN, default 0.56). Since dt
# grows with ν_lat, the lattice velocity u dt/h grows too: FV5_UREF (0.2) is
# kept low enough for Mach ≲ 0.3 at N = 12 (stage a checks it).
function fv5_config_from_env()
    N = fv5_int("FV5_N", FV3D_GPU ? 48 : 12)
    nu0 = fv5_float("FV5_NU0", 0.1)
    rinf = fv5_float("FV5_NUINF_RATIO", 0.1)
    taumin = fv5_float("FV5_TAU_MIN", 0.56)
    haskey(ENV, "FV5_NU_LAT") || rinf > 0 ||
        error("FV5_NUINF_RATIO = 0 leaves the LBM viscosity unbounded below; set FV5_NU_LAT explicitly")
    nu_lat = haskey(ENV, "FV5_NU_LAT") ? fv5_float("FV5_NU_LAT", 0.2) : (taumin - 0.5) / (3 * rinf)
    return Dict{String,Any}("Nx" => fv5_int("FV5_NX", 2N), "Ny" => fv5_int("FV5_NY", N),
                            "Nz" => fv5_int("FV5_NZ", N), "nu0" => nu0, "nuinf" => rinf * nu0,
                            "lambda" => fv5_float("FV5_LAMBDA", 5.0), "a" => fv5_float("FV5_A", 2.0),
                            "n" => fv5_float("FV5_NINDEX", 0.5), "uref" => fv5_float("FV5_UREF", 0.2),
                            "nu_lat" => nu_lat, "tau_min_target" => 0.5 + 3 * nu_lat * rinf)
end

"""Case NamedTuple from the stored dictionary (FV units + lattice scales)."""
function fv5_case(d::Dict)
    Nx, Ny, Nz = Int(d["Nx"]), Int(d["Ny"]), Int(d["Nz"])
    h = 1.0 / Ny
    nu0 = Float64(d["nu0"])
    # square-duct Poiseuille: u_mean = 0.0351 f H² / ν (H = Ly = 1)
    force = Float64(d["uref"]) * nu0 / 0.0351
    dt = Float64(d["nu_lat"]) * h^2 / nu0
    return (; Nx, Ny, Nz, h, nu0, nuinf = Float64(d["nuinf"]), lambda = Float64(d["lambda"]),
            a = Float64(d["a"]), n = Float64(d["n"]), uref = Float64(d["uref"]),
            nu_lat = Float64(d["nu_lat"]), force, dt,
            Lx = Nx * h, Ly = Ny * h, Lz = Nz * h, dict = d)
end

fv5_theta_true(c) = [c.nu0, c.nuinf, c.lambda, c.a, c.n]

function fv5_load_case(out = fv5_outdir())
    path = joinpath(out, "truth", "truth.toml")
    isfile(path) || error("no truth file $path: run stage a (v5a_truth_lbm.jl) with the same FV5_OUT first")
    t = TOML.parsefile(path)
    return fv5_case(t["case"]), t
end

"""Off-centre bar on the bottom wall z = 0 (8 × 16 × 12 cells at 96 × 48 × 48)."""
function fv5_obstacle(Nx, Ny, Nz)
    i0 = round(Int, Nx / 4) + 1; li = max(1, round(Int, Nx / 12))
    j0 = round(Int, 0.19 * Ny) + 1; lj = max(1, round(Int, Ny / 3))
    lk = max(1, round(Int, Nz / 4))
    solid = falses(Nx, Ny, Nz)
    solid[i0:(i0 + li - 1), j0:(j0 + lj - 1), 1:lk] .= true
    return solid, (; i = i0:(i0 + li - 1), j = j0:(j0 + lj - 1), k = 1:lk)
end

const FV5_BC = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)

fv5_carreau_visc(θ) = FVLawViscosity(ClassicalLaw(CarreauYasuda(θ...)), copy(θ), (), 0.0)
fv5_carreau_eta(θ, γ) = KrakeNN.law_value(ClassicalLaw(CarreauYasuda(θ...)), (γ,), θ)

"""Host FV problem of the duct with the given viscosity model, uniform x body force."""
function fv5_problem(c, visc)
    solid, _ = fv5_obstacle(c.Nx, c.Ny, c.Nz)
    f = c.force
    return fv_steady_problem(c.Nx, c.Ny, c.Nz; h = c.h, nu = c.nu0, bc = FV5_BC, solid,
                             viscosity = visc, source = (x, y, z) -> (f, 0.0, 0.0, 0.0))
end

function fv5_linsolve()
    kind = lowercase(fv5_env("FV5_LINSOLVE", "krylov"))
    kind == "direct" && return fv3d_linsolve()
    kind == "krylov" || error("FV5_LINSOLVE must be krylov or direct, got $kind")
    # FV5_PRECOND = p3l (default): block upper-triangular with the LSC Schur
    # approximation and 4 multigrid cycles per block (P3L, gate D2 passed at 64³);
    # p3: the earlier block SIMPLE with one cycle (P3), kept for comparison.
    pc = lowercase(fv5_env("FV5_PRECOND", "p3l"))
    pc in ("p3l", "p3") || error("FV5_PRECOND must be p3l or p3, got $pc")
    # same tag as before the helper: FV3D_P3_BASE (fv3d_common.jl) plus `extra`
    extra = pc == "p3l" ? FV3D_P3L_EXTRA : (; mg_cycles = 1)
    return fv3d_p3_tag(; rtol = fv5_float("FV5_ETA", 1e-6), rtol_transpose = fv5_float("FV5_ETA_T", 1e-10), extra)
end

fv5_atol(c) = fv5_float("FV5_ATOL_REL", 1e-9) * c.force * c.h^3
fv5_newton_ptc(c; dtau0 = 1.0, maxit = 80) =
    (; atol = fv5_atol(c), rtol = 1e-12, maxit, dtau0, dtau_max = 1e4)
fv5_newton_warm(c) = (; atol = fv5_atol(c), rtol = 1e-12, maxit = 30)

# ------------------------------------------------------------------ I/O
fv5_wbin(path, A) = (mkpath(dirname(path)); open(io -> write(io, Float64.(vec(A))), path, "w"); path)
fv5_rbin(path, dims...) = reshape(collect(reinterpret(Float64, read(path))), dims...)

function fv5_wtoml(path, data)
    mkpath(dirname(path))
    open(io -> TOML.print(io, data; sorted = true), path, "w")
    println("written ", path)
end

function fv5_load_truth(out, c)
    d = joinpath(out, "truth")
    r(name) = fv5_rbin(joinpath(d, name * ".bin"), c.Nx, c.Ny, c.Nz)
    return (; u = r("u"), v = r("v"), w = r("w"), nu = r("nu"), solid = r("is_solid") .!= 0)
end

# ------------------------------------------------------------------ PIV planes
"""
Plane list: three fitted planes (stereo z-plane through the obstacle mid-height,
stereo x-plane one obstacle length downstream, 2C y-plane through the wake) and
one held-out 2C z-plane above the obstacle. Vectors every 2h, window 2h × 2h,
Gaussian light sheet of standard deviation FV5_SHEET h (default 0.5: ±2σ = 2h).
"""
function fv5_planes(c)
    h = c.h
    _, ob = fv5_obstacle(c.Nx, c.Ny, c.Nz)
    grid(L) = collect(h:(2h):(L - h + 1e-12))
    w = (2h, 2h); th = fv5_float("FV5_SHEET", 0.5) * h
    zmid = length(ob.k) * h / 2
    xw = (last(ob.i) + length(ob.i)) * h
    yc = (first(ob.j) - 1 + length(ob.j) / 2) * h
    return [(; name = "z_mid", fit = true,
             plane = PIVPlane(; normal = :z, position = zmid, xs = grid(c.Lx), ys = grid(c.Ly),
                              window = w, thickness = th, stereo = true)),
            (; name = "x_wake", fit = true,
             plane = PIVPlane(; normal = :x, position = xw, xs = grid(c.Ly), ys = grid(c.Lz),
                              window = w, thickness = th, stereo = true)),
            (; name = "y_wake", fit = true,
             plane = PIVPlane(; normal = :y, position = yc, xs = grid(c.Lx), ys = grid(c.Lz),
                              window = w, thickness = th)),
            (; name = "z_upper", fit = false,
             plane = PIVPlane(; normal = :z, position = 0.6 * c.Lz, xs = grid(c.Lx), ys = grid(c.Ly),
                              window = w, thickness = th))]
end

fv5_axis(::PIVOperator{Ax}) where {Ax} = Ax
fv5_cell(Ax, a, b, n) = Ax == 1 ? (n, a, b) : Ax == 2 ? (a, n, b) : (a, b, n)

"""
Mask (1 = measured) of a host operator: a vector is dropped when any cell in
its footprint (window × light sheet) is solid. Also returns the fluid cells of
the footprint of the measured vectors (for the observed shear range).
"""
function fv5_mask_footprint(op, solid)
    Ax = fv5_axis(op)
    Wa, Wb, wn = op.Wa, op.Wb, op.wn
    nsa, nsb = size(Wa, 1), size(Wb, 1)
    na = [findall(!iszero, view(Wa, p, :)) for p in 1:nsa]
    nb = [findall(!iszero, view(Wb, q, :)) for q in 1:nsb]
    nn = findall(!iszero, wn)
    m = ones(nsa, nsb)
    cells = Set{NTuple{3,Int}}()
    for q in 1:nsb, p in 1:nsa
        hit = false
        for b in nb[q], a in na[p], n in nn
            solid[fv5_cell(Ax, a, b, n)...] && (hit = true; break)
        end
        if hit
            m[p, q] = 0
        else
            for b in nb[q], a in na[p], n in nn
                push!(cells, fv5_cell(Ax, a, b, n))
            end
        end
    end
    return m, cells
end

fv5_host_ops(c, planes) = [piv_operator(pl.plane, zeros(c.Nx, c.Ny, c.Nz); h = (c.h, c.h, c.h)) for pl in planes]

# ------------------------------------------------------------------ shear rate
# value and distance of the neighbour of fluid cell I along axis d, side s:
# x periodic; a wall or solid neighbour is the no-slip value 0 at h/2.
function fv5_nb(f, solid, I, d, s, h)
    J = collect(I); J[d] += s
    N = size(f, d)
    if d == 1
        J[1] = mod1(J[1], N)
    elseif !(1 <= J[d] <= N)
        return 0.0, h / 2
    end
    solid[J...] && return 0.0, h / 2
    return f[J...], h
end

"""γ̇ = sqrt(2 S:S) per fluid cell from cell-centred velocities (host, approximate near walls)."""
function fv5_shear_rate(u, v, w, solid, h)
    γ = zeros(size(u))
    comps = (u, v, w)
    G = zeros(3, 3)
    for I in CartesianIndices(u)
        solid[I] && continue
        for d in 1:3, cc in 1:3
            fp, dp = fv5_nb(comps[cc], solid, Tuple(I), d, 1, h)
            fm, dm = fv5_nb(comps[cc], solid, Tuple(I), d, -1, h)
            # one-sided weights for unequal distances (exact for a quadratic)
            f0 = comps[cc][I]
            G[cc, d] = (dm^2 * (fp - f0) + dp^2 * (f0 - fm)) / (dp * dm * (dp + dm))
        end
        S2 = 0.0
        for a in 1:3, b in 1:3
            S2 += ((G[a, b] + G[b, a]) / 2)^2
        end
        γ[I] = sqrt(2S2)
    end
    return γ
end

"""Fluid cells (index tuples) of the footprints of the measured vectors of the fitted planes."""
function fv5_footprint_cells(c, solid)
    planes = fv5_planes(c)
    ops = fv5_host_ops(c, planes)
    cells = Set{NTuple{3,Int}}()
    for (pl, op) in zip(planes, ops)
        pl.fit || continue
        _, cs = fv5_mask_footprint(op, solid)
        union!(cells, cs)
    end
    return [I for I in cells if !solid[I...]]
end

"""Observed shear range: 5–95 % quantiles of the truth γ̇ over the footprints of the fitted planes."""
function fv5_gamma_range(c, truth)
    γ = fv5_shear_rate(truth.u, truth.v, truth.w, truth.solid, c.h)
    vals = [γ[I...] for I in fv5_footprint_cells(c, truth.solid)]
    return (; lo = quantile(vals, 0.05), hi = quantile(vals, 0.95), min = minimum(vals),
            max = maximum(vals), median = median(vals), ncells = length(vals))
end

fv5_gamma_grid(gr; n = 64) = exp.(range(log(gr.lo), log(gr.hi); length = n))
"""Plot grid: the observed range and `pad` (one decade) on either side."""
fv5_plot_grid(gr; n = 121, pad = 10.0) = exp.(range(log(gr.lo / pad), log(gr.hi * pad); length = n))

function fv5_log_hist(vals, edges)
    cnt = zeros(Int, length(edges) - 1)
    below = 0; above = 0
    for v in vals
        if v < edges[1]
            below += 1
        elseif v >= edges[end]
            above += 1
        else
            cnt[searchsortedlast(edges, v)] += 1
        end
    end
    return cnt, below, above
end

"""
Histogram of γ̇ (log bins over the plot grid of `gr`) in the footprints of the
fitted planes and in all fluid cells, for host fields `(; u, v, w, solid)`.
"""
function fv5_gamma_hist(c, fields, gr; nbins = 48)
    γ = fv5_shear_rate(fields.u, fields.v, fields.w, fields.solid, c.h)
    fp = [γ[I...] for I in fv5_footprint_cells(c, fields.solid)]
    vf = γ[.!fields.solid]
    lo, hi = extrema(fv5_plot_grid(gr))
    edges = collect(exp.(range(log(lo), log(hi); length = nbins + 1)))
    cf, bf, af = fv5_log_hist(fp, edges)
    ca, ba, aa = fv5_log_hist(vf, edges)
    return (; edges, count_footprint = cf, count_fluid = ca, below_footprint = bf, above_footprint = af,
            below_fluid = ba, above_fluid = aa, n_footprint = length(fp), n_fluid = length(vf))
end

"""Relative error of a law η(γ̇) against the true law over the observed range."""
function fv5_law_error(eta, c, gr)
    θt = fv5_theta_true(c)
    γs = fv5_gamma_grid(gr)
    rel = [eta(γ) / fv5_carreau_eta(θt, γ) - 1 for γ in γs]
    return (; max = maximum(abs, rel), rms = sqrt(mean(rel .^ 2)), gammas = γs, rel)
end

# ------------------------------------------------------------------ PIV data
"""
Load the PIV data written by stage b (and the FV twin of stage c for the twin
variants). Returns the planes, host operators, masks, the data of `variant`
(noisy | clean | twin | twin_noisy), the clean LBM samples, σ_u, the flow-rate
target of the variant and σ_Q.
"""
function fv5_load_piv(out, c; variant = fv5_env("FV5_DATA", "noisy"))
    variant in ("noisy", "clean", "twin", "twin_noisy") ||
        error("FV5_DATA must be noisy, clean, twin or twin_noisy, got $variant")
    meta = TOML.parsefile(joinpath(out, "piv", "piv.toml"))
    planes = fv5_planes(c)
    ops = fv5_host_ops(c, planes)
    d = joinpath(out, "piv")
    masks = Matrix{Float64}[]; data = Any[]; clean = Any[]
    twin_meta = variant in ("twin", "twin_noisy") ? TOML.parsefile(joinpath(out, "c_floor.toml")) : nothing
    for (pl, op) in zip(planes, ops)
        nsa, nsb = size(op.Wa, 1), size(op.Wb, 1)
        nc = pl.plane.stereo ? 3 : 2
        r(tag, ci) = fv5_rbin(joinpath(d, "$(pl.name)_$(tag)_c$(ci).bin"), nsa, nsb)
        push!(masks, fv5_rbin(joinpath(d, "$(pl.name)_mask.bin"), nsa, nsb))
        cl = Tuple(r("clean", ci) for ci in 1:nc)
        no = Tuple(r("noisy", ci) for ci in 1:nc)
        push!(clean, cl)
        if variant == "noisy"
            push!(data, no)
        elseif variant == "clean"
            push!(data, cl)
        else
            tw = Tuple(r("twin", ci) for ci in 1:nc)
            push!(data, variant == "twin" ? tw : Tuple(tw[ci] .+ (no[ci] .- cl[ci]) for ci in 1:nc))
        end
    end
    Q = variant == "noisy" ? meta["Q_noisy"] : variant == "clean" ? meta["Q_clean"] :
        variant == "twin" ? twin_meta["Q_fv_true"] : twin_meta["Q_fv_true"] + (meta["Q_noisy"] - meta["Q_clean"])
    return (; planes, ops, masks, data, clean, sigma_u = meta["sigma_u"], Q, sigma_Q = meta["sigma_Q"],
            variant, meta)
end

"""Composite objective on the solver backend: fitted planes / σ_u² + (Q − Q_t)² / (2 σ_Q²)."""
function fv5_objective(p, c, P)
    to = fv3d_to()
    tmpl = to(zeros(c.Nx, c.Ny, c.Nz))
    fit = findall(pl -> pl.fit, P.planes)
    ops = Tuple(piv_operator(P.planes[i].plane, tmpl; h = (c.h, c.h, c.h)) for i in fit)
    data = Tuple(Tuple(Matrix{Float64}(x) for x in P.data[i]) for i in fit)
    mask = Tuple(P.masks[i] for i in fit)
    terms = ((fv_flow_rate(p, 1), 1 / P.sigma_Q^2, P.Q),)
    return FVPIVObjective(p, ops, data; mask, sigma_u = P.sigma_u, terms)
end

"""
Model discrepancy on the fitted observations, whitened and in the order of
`fv5_residuals`: FV planes at the true law (written by stage c as
piv/<plane>_twin_c<k>.bin) minus clean LBM planes, over σ_u; last entry the
flow rate (Q_fv_true − Q_clean)/σ_Q. It is the residual of the true law on
noise-free data, so ½‖r_model‖² equals J(true, clean) of stage c.
"""
function fv5_model_residual(out, P, floor)
    r = Float64[]
    for (i, (pl, op)) in enumerate(zip(P.planes, P.ops))
        pl.fit || continue
        nsa, nsb = size(op.Wa, 1), size(op.Wb, 1)
        m = P.masks[i] .> 0
        for ci in eachindex(P.clean[i])
            tw = fv5_rbin(joinpath(out, "piv", "$(pl.name)_twin_c$(ci).bin"), nsa, nsb)
            append!(r, vec((tw .- P.clean[i][ci])[m]) ./ P.sigma_u)
        end
    end
    push!(r, (floor["Q_fv_true"] - P.meta["Q_clean"]) / P.sigma_Q)
    return r
end

fv5_nobs(P) = sum(count(>(0), P.masks[i]) * length(P.data[i]) for i in eachindex(P.planes) if P.planes[i].fit)

"""Flow rate through slab i = 1 (fluid cells), as fv_flow_rate."""
fv5_flow_rate(u, solid, h) = h^2 * sum(u[1, j, k] for j in axes(u, 2), k in axes(u, 3) if !solid[1, j, k])

"""Per-plane misfit of host fields against data and against the clean LBM samples."""
function fv5_plane_metrics(P, u, v, w)
    rows = Dict{String,Any}()
    for (i, (pl, op)) in enumerate(zip(P.planes, P.ops))
        s = piv_apply(op, (u, v, w))
        m = P.masks[i] .> 0
        nc = length(s)
        res_d = vcat((vec((s[ci] .- P.data[i][ci])[m]) for ci in 1:nc)...)
        res_c = vcat((vec((s[ci] .- P.clean[i][ci])[m]) for ci in 1:nc)...)
        sig = vcat((vec(P.clean[i][ci][m]) for ci in 1:nc)...)
        rows[pl.name] = Dict("fit" => pl.fit, "nvec" => count(m), "ncomp" => nc,
                             "rms_vs_data_sigma" => sqrt(mean(res_d .^ 2)) / P.sigma_u,
                             "rms_vs_clean_sigma" => sqrt(mean(res_c .^ 2)) / P.sigma_u,
                             "rel_rms_vs_clean" => sqrt(mean(res_c .^ 2)) / sqrt(mean(sig .^ 2)))
    end
    return rows
end

function fv5_print_planes(tag, rows)
    for name in sort(collect(keys(rows)))
        r = rows[name]
        @printf("SUMMARY %s plane %-8s %s vectors %4d x %dC  rms(model-data)/σ_u %.3f  rms(model-LBM clean)/σ_u %.3f  rel rms vs LBM %.4f\n",
                tag, name, r["fit"] ? "fit " : "HELD", r["nvec"], r["ncomp"], r["rms_vs_data_sigma"],
                r["rms_vs_clean_sigma"], r["rel_rms_vs_clean"])
    end
    flush(stdout)
end

# ------------------------------------------------------------------ fit engine
"""
State of a law fit: one Newton solver kept for the whole fit (pattern,
colouring, Krylov/multigrid structures built once; `fv_set_params!` swaps the
law parameters), the device state `U` reused as warm start, the best state so
far as fallback, counters.
"""
mutable struct FV5Fit
    c::Any
    s::Any
    map::Any
    obj::Any
    ws::Any
    U::Any
    Ubest::Any
    fbest::Float64
    thetafun::Function       # optimisation variables x -> law parameters θ
    chain::Function          # (x, dJ/dθ) -> dJ/dx
    valid::Function          # θ -> Bool
    xlast::Vector{Float64}
    oklast::Bool
    first::Bool
    n_eval::Int
    n_fail::Int
    n_grad::Int
    newton_its::Int
    newton_attempts::Dict{String,Int}
    gmres_fwd::Int
    gmres_tr::Int
    gmres_unconverged::Int
    t_newton::Float64
    t_grad::Float64
    mem_peak::Int
    mem_base::Int
end

fv5_isK(s) = s.linsolve isa KrylovBackendTag && s.cache !== nothing
fv5_klen(s) = fv5_isK(s) ? length(lin_krylov_log(s.cache)) : 0

function fv5_kcount!(F, i0)
    fv5_isK(F.s) || return
    lg = lin_krylov_log(F.s.cache)
    for e in lg[(i0 + 1):end]
        e.transpose ? (F.gmres_tr += e.iters) : (F.gmres_fwd += e.iters)
        e.converged || (F.gmres_unconverged += 1)
    end
end

function fv5_fit_setup(c, P, visc, nθ; thetafun, chain, valid)
    p = fv5_problem(c, visc)
    to = fv3d_to()
    fv3d_free()
    m0 = fv3d_devused()
    t_build = @elapsed s = fv_newton_solver(p; to, linsolve = fv5_linsolve())
    obj = fv5_objective(p, c, P)
    n = fv_nunknowns(p)
    U = to(KrakeNN._fv_initial_force!(zeros(n), p))
    F = FV5Fit(c, s, FVLawParameters(nθ), obj, nothing, U, copy(U), Inf, thetafun, chain, valid,
               Float64[], false, true, 0, 0, 0, 0, Dict{String,Int}(), 0, 0, 0, 0.0, 0.0, 0, m0)
    @printf("   solver built in %.1f s: n = %d unknowns, nnz(J) = %d, linsolve %s\n", t_build, n,
            nnz(fv_jacobian_pattern(p)), typeof(s.linsolve))
    flush(stdout)
    return F
end

"""Newton at the current parameters: warm start, then the best state with PTC, then rest with PTC."""
function fv5_solve!(F)
    c = F.c
    U0 = copy(F.U)
    attempts = F.first ? ("rest_ptc", "rest_ptc_small") : ("warm", "best_ptc", "rest_ptc", "rest_ptc_small")
    i0 = fv5_klen(F.s)
    its = 0
    for a in attempts
        a == "warm" && copyto!(F.U, U0)
        a == "best_ptc" && copyto!(F.U, F.Ubest)
        startswith(a, "rest") && fill!(F.U, 0)
        kw = a == "warm" ? fv5_newton_warm(c) :
             a == "rest_ptc_small" ? fv5_newton_ptc(c; dtau0 = 0.1, maxit = 160) : fv5_newton_ptc(c)
        i0 = fv5_klen(F.s)
        st = fv_newton!(F.s, F.U; kw...)
        fv3d_sync()
        its += st.iterations
        fv5_kcount!(F, i0)
        if st.converged
            F.newton_attempts[a] = get(F.newton_attempts, a, 0) + 1
            F.first = false
            return true, its
        end
        @printf("   newton attempt %s failed after %d iterations (inf %.2e)\n", a, st.iterations,
                isempty(st.inf) ? NaN : last(st.inf))
        flush(stdout)
    end
    copyto!(F.U, U0)
    F.newton_attempts["failed"] = get(F.newton_attempts, "failed", 0) + 1
    return false, its
end

"""Objective at x (Inf when x is invalid or Newton fails); leaves U converged at x."""
function fv5_fval!(F, x)
    θ = F.thetafun(x)
    F.n_eval += 1
    F.xlast = copy(x); F.oklast = false
    if !F.valid(θ)
        F.n_fail += 1
        return Inf
    end
    fv_set_params!(F.s, F.map, θ)
    t = @elapsed ok, its = fv5_solve!(F)
    F.t_newton += t
    F.newton_its += its
    F.mem_peak = max(F.mem_peak, fv3d_devused())
    if !ok
        F.n_fail += 1
        return Inf
    end
    J, _ = fv_objective_gradient!(F.obj, F.U, F.s.prob)
    J = Float64(J)
    F.oklast = isfinite(J)
    if J < F.fbest
        F.fbest = J
        copyto!(F.Ubest, F.U)
    end
    @printf("   eval %3d  J = %.8e  newton it %2d  %.1f s\n", F.n_eval, J, its, t)
    flush(stdout)
    return J
end

"""Adjoint gradient dJ/dx at x (re-solves if the last evaluation was elsewhere)."""
function fv5_fgrad!(F, x)
    (F.oklast && F.xlast == x) || (isfinite(fv5_fval!(F, x)) || error("gradient requested at a failed point"))
    θ = F.thetafun(x)
    F.ws === nothing && (F.ws = fv_adjoint_workspace(F.s, F.map))
    i0 = fv5_klen(F.s)
    t = @elapsed (r = fv_steady_gradient(F.s, F.U, θ, F.obj; params = F.map, ws = F.ws); fv3d_sync())
    fv5_kcount!(F, i0)
    F.t_grad += t
    F.n_grad += 1
    F.mem_peak = max(F.mem_peak, fv3d_devused())
    return F.chain(x, Float64.(Array(r.grad)))
end

"""Whitened residual vector at the current state (host): fitted planes / σ_u, flow rate / σ_Q."""
function fv5_residuals(F, P)
    c = F.c
    u, v, w, _ = fv_fields(F.s.prob, F.U)
    solid, _ = fv5_obstacle(c.Nx, c.Ny, c.Nz)
    r = Float64[]
    for (i, (pl, op)) in enumerate(zip(P.planes, P.ops))
        pl.fit || continue
        s = piv_apply(op, (u, v, w))
        m = P.masks[i] .> 0
        for ci in eachindex(s)
            append!(r, vec((s[ci] .- P.data[i][ci])[m]) ./ P.sigma_u)
        end
    end
    push!(r, (fv5_flow_rate(u, solid, c.h) - P.Q) / P.sigma_Q)
    return r
end

"""Run gc_bfgs with per-iteration SUMMARY-free progress lines and a progress file for FV5_RESUME."""
function fv5_bfgs(F, x0, progress_path; maxiter, max_step, gtol_rel = 1e-6, ftol = 1e-10, names = nothing)
    t0 = time()
    hist = Dict{String,Any}[]
    cb = rec -> begin
        @printf("ITER %3d  J = %.8e  |g|inf = %.3e  step %.3e  alpha %.3g  backtracks %d  evals %d  t %.0f s  x = %s\n",
                rec.iter, rec.f, rec.gnorm, rec.step, rec.alpha, rec.backtracks, F.n_eval, time() - t0,
                string(round.(rec.x; sigdigits = 6)))
        flush(stdout)
        push!(hist, Dict("iter" => rec.iter, "f" => rec.f, "gnorm" => rec.gnorm, "x" => rec.x,
                         "t" => time() - t0))
        fv5_wtoml(progress_path, Dict("history" => hist, "x_last" => rec.x,
                                      "names" => names === nothing ? String[] : collect(names)))
    end
    fval = x -> fv5_fval!(F, x)
    fgrad = x -> fv5_fgrad!(F, x)
    res = gc_bfgs(fval, fgrad, x0; maxiter, gtol_rel, ftol, xtol = 1e-9, max_step, callback = cb)
    # leave the state at the optimum
    (F.oklast && F.xlast == res.x) || fv5_fval!(F, res.x)
    return res, time() - t0
end

"""
Start point: x0, or the last x of the progress file when FV5_RESUME=1 (`legacy`:
older file name read when the first is missing, e.g. the progress file of a run
made before the file names carried the data variant).
"""
function fv5_start(x0, progress_path; legacy = nothing)
    fv5_env("FV5_RESUME", "0") == "1" || return x0
    path = isfile(progress_path) ? progress_path : ((legacy !== nothing && isfile(legacy)) ? legacy : nothing)
    if path === nothing
        println("   FV5_RESUME=1 but no progress file ", progress_path, ": starting from x0")
        return x0
    end
    x = Float64.(TOML.parsefile(path)["x_last"])
    length(x) == length(x0) || error("progress file $path has $(length(x)) variables, expected $(length(x0))")
    println("   resuming from ", path)
    return x
end

# ------------------------------------------------------------------ figure data
"""Plain CSV: header line, one row per index, floats as %.10e, integers and Bools as integers."""
function fv5_wcsv(path, names, cols)
    mkpath(dirname(path))
    n = length(first(cols))
    all(length(cl) == n for cl in cols) || error("fv5_wcsv: columns of unequal length for $path")
    length(names) == length(cols) || error("fv5_wcsv: $(length(names)) names for $(length(cols)) columns")
    fmt(v) = (v isa Integer || v isa Bool) ? string(Int(v)) : @sprintf("%.10e", v)
    open(path, "w") do io
        println(io, join(names, ','))
        for i in 1:n
            println(io, join((fmt(cl[i]) for cl in cols), ','))
        end
    end
    println("written ", path)
    return path
end

"""Case scales for the figure index (FV units; η in the scripts is the kinematic viscosity ν)."""
fv5_fig_case(c, P) = Dict{String,Any}("nu0" => c.nu0, "nuinf" => c.nuinf, "lambda" => c.lambda, "a" => c.a,
                                      "n" => c.n, "Lx" => c.Lx, "Ly" => c.Ly, "Lz" => c.Lz, "h" => c.h,
                                      "sigma_u" => P.sigma_u, "u_max" => P.meta["u_max"],
                                      "U_bulk" => P.meta["Q_clean"] / (c.Ly * c.Lz))

const FV5_AXN = ("x", "y", "z")

"""
Write, for every plane, the model samples of host fields (u, v, w), the data of
the variant, the residual model − data and the clean LBM samples to
<out>/figure/<prefix>_<plane>_{model,data,residual,clean}_c<k>.bin (masked
vectors set to 0; the mask is piv/<plane>_mask.bin). Returns the index entries.
"""
function fv5_write_planes(out, prefix, P, u, v, w)
    entries = Dict{String,Any}[]
    for (i, (pl, op)) in enumerate(zip(P.planes, P.ops))
        Ax = fv5_axis(op)
        s = piv_apply(op, (u, v, w))
        m = P.masks[i]
        inplane = Ax == 1 ? (2, 3) : Ax == 2 ? (1, 3) : (1, 2)
        comps = (inplane..., Ax)[1:length(s)]
        files = Dict{String,Vector{String}}("model" => String[], "data" => String[], "residual" => String[],
                                            "clean" => String[])
        for ci in eachindex(s)
            for (key, A) in (("model", s[ci]), ("data", P.data[i][ci]), ("residual", s[ci] .- P.data[i][ci]),
                             ("clean", P.clean[i][ci]))
                path = joinpath(out, "figure", "$(prefix)_$(pl.name)_$(key)_c$(ci).bin")
                fv5_wbin(path, A .* m)
                push!(files[key], relpath(path, out))
            end
        end
        push!(entries, Dict{String,Any}("name" => pl.name, "fit" => pl.fit, "normal" => String(pl.plane.normal),
                                        "position" => pl.plane.position, "stereo" => pl.plane.stereo,
                                        "shape" => [size(m, 1), size(m, 2)],
                                        "axes" => [FV5_AXN[inplane[1]], FV5_AXN[inplane[2]]],
                                        "centres_a" => collect(Float64, pl.plane.xs),
                                        "centres_b" => collect(Float64, pl.plane.ys),
                                        "components" => [FV5_AXN[k] for k in comps],
                                        "mask" => "piv/$(pl.name)_mask.bin",
                                        "layout" => "Float64 little-endian, column-major, shape[1] fastest",
                                        files...))
    end
    return entries
end

function fv5_counters(F)
    return Dict{String,Any}("evals" => F.n_eval, "failed_evals" => F.n_fail, "gradients" => F.n_grad,
                            "newton_iterations" => F.newton_its, "gmres_forward_iterations" => F.gmres_fwd,
                            "gmres_transposed_iterations" => F.gmres_tr,
                            "gmres_unconverged_solves" => F.gmres_unconverged,
                            "newton_attempts" => F.newton_attempts, "t_newton" => F.t_newton,
                            "t_gradient" => F.t_grad, "gpu_mem_peak_GiB" => F.mem_peak / 2^30,
                            "gpu_mem_before_GiB" => F.mem_base / 2^30)
end

fv5_counter_line(F) = @sprintf("evals %d (failed %d) gradients %d  newton it %d  gmres fwd/transp it %d/%d (unconverged solves %d)  t newton/gradient %.0f/%.0f s  GPU mem peak %.2f GiB (before %.2f)",
                               F.n_eval, F.n_fail, F.n_grad, F.newton_its, F.gmres_fwd, F.gmres_tr,
                               F.gmres_unconverged, F.t_newton, F.t_grad, F.mem_peak / 2^30, F.mem_base / 2^30)
