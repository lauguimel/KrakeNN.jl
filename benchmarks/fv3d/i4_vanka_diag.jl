# Diagnostic of P4 (monolithic Vanka multigrid, `precond = :vanka`) on the two
# drives of the D2 harness (i4_krylov_measure.jl): why the body-force case
# ("source") fails where the imposed-flow-rate case passes. Not a gate: it
# prints DIAG lines and writes a TOML, exit code 0 unless the script errors.
#
# Same duct as the harness (periodic-x square duct, off-centre obstacle on the
# bottom wall, Carreau–Yasuda, Re 50). Cases: flowrate, source (force × 1) and
# source07 (force × FV3D_DIAG_LOWRE_SCALE, default 0.7: Re near the flow-rate
# case; run only at N in FV3D_DIAG_LOWRE_NS).
#
# For each (case, N):
#   (0) converged state from the DIRECT branch (cuDSS Newton, PTC as in the
#       harness), so that P4 is judged on the same J whatever its own Newton
#       does. Optional (FV3D_DIAG_P4NEWTON=1): P4 Newton from rest as well
#       (convergence, GMRES per step, state vs direct): H4.
#   (1) base P4 (constant prolongation, harness defaults), at that state:
#       - hierarchy, smallest Vanka pivot ratio per level, forward and
#         transposed;
#       - per level: smoother-only factor (10 homogeneous sweeps) and cycle
#         factor from that level down (10 homogeneous cycles): H2;
#       - coarsest level (dense, n <= FV3D_DIAG_DENSE_MAX): singular values,
#         eigenvalues (count with Re < 0, min |λ|), structure of the smallest
#         singular vector: H1;
#       - fine level: dominant mode of the stationary iteration I − M⁻¹J
#         (power iteration, 30 steps) and its structure (field fractions,
#         x-invariance, energy next to walls/obstacle): H1/H2/H3;
#       - one-application residual and 8-step stationary factor on the
#         uniform x-force direction c vs the harness rhs: H3;
#       - GMRES panel, forward (rtol FV3D_ETA) and transposed (FV3D_ETA_T):
#         GMRES(50) with the relres history at the restarts, GMRES-DR(50, k)
#         (k = FV3D_DIAG_DEFLATE, default 10), GMRES(200), and the harmonic
#         Ritz values of J M⁻¹ after two cycles of GMRES(50): H1/H3/H5;
#       - flowrate only: the same GMRES(50) on the LOCAL block alone (border
#         row and column removed, i.e. the flow-rate matrix driven like the
#         source case): H3.
#   (2) variants on the same J (FV3D_DIAG_VARIANTS, default
#       omega05,scale15,coarse20k,sweeps4): GMRES(50) forward/transposed with
#       history, stationary factor.
#
# Hypotheses and what decides them (see the DIAG lines):
#   H1 near-singular / indefinite coarse Galerkin operator (R J P of the full
#      Newton Jacobian; the zeroth-order Newton convection terms grow ×8 per
#      level, the constant-P Galerkin viscous terms ×4): σ_min/σ_max of the
#      coarsest level orders of magnitude smaller in source than in flowrate,
#      GMRES(50) relres plateau (floor) rather than slow decay, isolated tiny
#      Ritz values, GMRES-DR converging; coarse20k changes the picture.
#   H2 additive Vanka smoother or cycle unstable on some level in the source
#      state (thinner fluid, larger Newton terms): smoother or cycle factor
#      >= 1 on a level, omega05 / sweeps4 change the GMRES count a lot,
#      source07 passes while source fails.
#   H3 the flow-rate border deflates the mean-flow mode (a bordered
#      eigenvalue μ ≈ 0 on the force direction becomes ±sqrt(ρ)): the
#      flowrate local-block-only GMRES fails like source; the dominant mode
#      is x-invariant u; one-application residual on c >> on random rhs.
#   H4 the harness failure comes from the P4 Newton path (state), not from
#      the converged J: P4 passes here at the direct state.
#   H5 restart stagnation with a few outlying eigenvalues: GMRES(200) or
#      GMRES-DR converge in far fewer iterations than GMRES(50).
#
#   FV3D_BACKEND=cuda FV3D_NS=32,48 julia --project=<env> benchmarks/fv3d/i4_vanka_diag.jl
#   FV3D_BACKEND=cpu  FV3D_NS=6     julia --project benchmarks/fv3d/i4_vanka_diag.jl     (smoke)
# Knobs: FV3D_DIAG_CASES (flowrate,source,source07), FV3D_DIAG_MAXITER (1000),
# FV3D_DIAG_VARIANTS, FV3D_DIAG_DEFLATE (10), FV3D_DIAG_DENSE_MAX (3000),
# FV3D_DIAG_P4NEWTON (0), FV3D_DIAG_P4NEWTON_MAXIT (20), FV3D_VANKA_PROLONG
# (constant), FV3D_VANKA_OMEGA (0.7), FV3D_MG_* as in the harness.
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
include(joinpath(@__DIR__, "fv3d_common.jl"))
using Statistics, Random

const K = KrakeNN
const ETA = parse(Float64, get(ENV, "FV3D_ETA", "1e-6"))
const ETA_T = parse(Float64, get(ENV, "FV3D_ETA_T", "1e-10"))
const DMAXIT = parse(Int, get(ENV, "FV3D_DIAG_MAXITER", "1000"))
const DEFL = parse(Int, get(ENV, "FV3D_DIAG_DEFLATE", "10"))
const DENSE_MAX = parse(Int, get(ENV, "FV3D_DIAG_DENSE_MAX", "3000"))
const P4NEWTON = get(ENV, "FV3D_DIAG_P4NEWTON", "0") == "1"
const P4NEWTON_MAXIT = parse(Int, get(ENV, "FV3D_DIAG_P4NEWTON_MAXIT", "20"))
const CASES = split(get(ENV, "FV3D_DIAG_CASES", "flowrate,source,source07"), ','; keepempty = false)
const LOWRE_SCALE = parse(Float64, get(ENV, "FV3D_DIAG_LOWRE_SCALE", "0.7"))
const LOWRE_NS = fv3d_ints("FV3D_DIAG_LOWRE_NS", FV3D_GPU ? "32" : "6")
const VARIANTS = split(get(ENV, "FV3D_DIAG_VARIANTS", "omega05,scale15,coarse20k,sweeps4"), ','; keepempty = false)
const NU0 = 0.02
const THETA = [NU0, 0.002, 1.0, 2.0, 0.5]
const UREF = 1.0
const NEWTON = (; atol = 1e-10, rtol = 1e-12, maxit = 80, dtau0 = 1.0, dtau_max = 1e4)
const SOURCE_DTAU = parse(Float64, get(ENV, "FV3D_SOURCE_DTAU", "1"))
const BASE = (; rtol = ETA, rtol_transpose = ETA_T, restart = 50, maxiter = DMAXIT, strict = false,
              precond = :vanka,
              mg_cycle = Symbol(get(ENV, "FV3D_MG_CYCLE", "W")),
              mg_sweeps = parse(Int, get(ENV, "FV3D_MG_SWEEPS", "2")),
              mg_cycles = parse(Int, get(ENV, "FV3D_MG_CYCLES", "1")),
              mg_coarse = parse(Int, get(ENV, "FV3D_MG_COARSE", "2000")),
              mg_scale = parse(Float64, get(ENV, "FV3D_MG_SCALE", "1")),
              vanka_omega = parse(Float64, get(ENV, "FV3D_VANKA_OMEGA", "0.7")),
              vanka_prolong = Symbol(get(ENV, "FV3D_VANKA_PROLONG", "constant")))
const VARIANT_DEFS = Dict("base" => (;), "omega05" => (; vanka_omega = 0.5), "omega09" => (; vanka_omega = 0.9),
                          "scale15" => (; mg_scale = 1.5), "coarse20k" => (; mg_coarse = 20000),
                          "sweeps4" => (; mg_sweeps = 4), "vcycle" => (; mg_cycle = :V),
                          "linear" => (; vanka_prolong = :linear))
for v in VARIANTS
    haskey(VARIANT_DEFS, v) || error("FV3D_DIAG_VARIANTS: unknown variant $v ($(join(keys(VARIANT_DEFS), ", ")))")
end
tagof(extra = (;)) = KrylovBackendTag(; merge(BASE, extra)...)

function duct(N, case)
    bc = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)
    solid = falses(N, N, N)
    solid[(2N ÷ 5 + 1):(3N ÷ 5), 1:max(1, N ÷ 3), (N ÷ 4 + 1):(N ÷ 2)] .= true
    visc = FVLawViscosity(ClassicalLaw(CarreauYasuda(THETA...)), copy(THETA), (), 0.0)
    if case == "flowrate"
        return fv_steady_problem(N, N, N; h = 1 / N, nu = NU0, bc, solid, viscosity = visc,
                                 source = (x, y, z) -> (0.0, 0.0, 0.0, 0.0), u_mean_target = UREF)
    end
    f = (case == "source07" ? LOWRE_SCALE : 1.0) * UREF * NU0 / 0.0351
    return fv_steady_problem(N, N, N; h = 1 / N, nu = NU0, bc, solid, viscosity = visc,
                             source = (x, y, z) -> (f, 0.0, 0.0, 0.0))
end
newton_opts(case) = case == "flowrate" ? NEWTON : merge(NEWTON, (; dtau0 = SOURCE_DTAU))

function mean_u(p, U)
    u = fv_fields(p, U)[1]
    fluid = Array(p.tag) .!= K.FV_CELL_SOLID
    return sum(u[fluid]) / count(fluid)
end

# --- device memory release (as in i4_krylov_measure.jl) ---
function release!(x, seen = Base.IdSet{Any}(), depth = 0)
    (depth > 12 || x === nothing || isbits(x) || x isa Module || x isa DataType ||
     x isa Function || x isa String || x isa Symbol) && return
    x in seen && return
    push!(seen, x)
    if FV3D_GPU && x isa CUDA.CuArray
        CUDA.unsafe_free!(x)
        return
    end
    if x isa Array
        isbitstype(eltype(x)) || foreach(e -> release!(e, seen, depth + 1), x)
        return
    end
    for f in fieldnames(typeof(x))
        isdefined(x, f) && release!(getfield(x, f), seen, depth + 1)
    end
    FV3D_GPU && ismutable(x) && parentmodule(typeof(x)) === CUDSS && finalize(x)
    return
end
function release_all!(s)
    try
        s === nothing || release!(s)
    catch err
        println("NOTE release!: ", first(sprint(showerror, err), 200))
    end
    GC.gc(true); GC.gc(true)
    fv3d_free()
    return nothing
end

# --- structure of a vector on the cell grid ---
# fine level: interleaved (nf fields per cell); fractions of the energy per
# field, x-invariance per velocity field and pressure (energy of the x-mean),
# energy in cells next to a wall or a solid cell, location of the largest entry.
function near_mask(dims, solid)
    Nx, Ny, Nz = dims
    S = reshape(solid, dims)
    m = falses(dims)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        S[i, j, k] && continue
        nb = ((mod1(i + 1, Nx), j, k), (mod1(i - 1, Nx), j, k), (i, j + 1, k), (i, j - 1, k), (i, j, k + 1), (i, j, k - 1))
        m[i, j, k] = j == 1 || j == Ny || k == 1 || k == Nz ||
                     any(c -> 1 <= c[2] <= Ny && 1 <= c[3] <= Nz && S[c...], nb)
    end
    return vec(m)
end
function fine_structure(x::Vector{Float64}, nf, dims, solid, near)
    nc = prod(dims)
    X = reshape(x[1:nf*nc], nf, nc)
    E = max(sum(abs2, X), eps())
    ff = [sum(abs2, view(X, f, :)) / E for f in 1:min(nf, 4)]
    xinv = map(1:min(nf, 4)) do f
        F = reshape(X[f, :], dims)
        m = mean(F; dims = 1)
        dims[1] * sum(abs2, m) / max(sum(abs2, F), eps())
    end
    nearfrac = sum(abs2, X[:, near]) / E
    solfrac = sum(abs2, X[:, solid]) / E
    a = argmax(abs.(X))
    c = a[2] - 1
    loc = (a[1], c % dims[1] + 1, (c ÷ dims[1]) % dims[2] + 1, c ÷ (dims[1] * dims[2]) + 1)
    return (; field_frac = ff, x_invariance = xinv, near_frac = nearfrac, solid_frac = solfrac, argmax = loc)
end
fmt(v) = join((@sprintf("%.3f", x) for x in v), "/")
fmtfs(s) = @sprintf("energy u/v/w/p %s  x-invariance u/v/w/p %s  next to wall/solid %.3f  in solid %.3f  max at (field %d, i %d, j %d, k %d)",
                    fmt(s.field_frac), fmt(s.x_invariance), s.near_frac, s.solid_frac,
                    s.argmax[1], s.argmax[2], s.argmax[3], s.argmax[4])

# coarse level nodes (comp, cls, ijk), rebuilt with the same aggregation as the hierarchy
function level_nodes(nf, geom, nlevels)
    nloc = nf * prod(geom.dims)
    nd = K._mg_nodes(Int32[(i - 1) ÷ nf + 1 for i in 1:nloc], Int32[(i - 1) % nf + 1 for i in 1:nloc], geom)
    out = Any[nd]
    for _ in 2:nlevels
        _, nd, _ = K._mg_aggregate(nd)
        push!(out, nd)
    end
    return out
end
function coarse_structure(x::Vector{Float64}, nd)
    E = max(sum(abs2, x), eps())
    ff = [sum(abs2, x[nd.comp .== f]) / E for f in 1:4]
    sol = sum(abs2, x[nd.cls .!= 0]) / E
    a = argmax(abs.(x))
    return @sprintf("energy u/v/w/p %s  solid-class %.3f  max at (field %d, block %d,%d,%d)",
                    fmt(ff), sol, nd.comp[a], nd.ijk[1, a], nd.ijk[2, a], nd.ijk[3, a])
end

# --- GMRES with history on one side of a Krylov cache ---
function gmres_side(side, b, tol; restart = 50, deflate = 0, maxiter = DMAXIT, ritz = false)
    x = fill!(similar(b), 0.0)
    t = @elapsed info = gnf_gmres!(x, K._KrylovOp(side), b; tol, restart, maxiter, deflate, orth = :cgs2, ritz)
    fv3d_sync()
    return info, t
end
function hist(rc)
    idx = unique(filter(i -> i <= length(rc), vcat(1, 2, 3, 5, 9, 17, 33, length(rc))))
    return join((@sprintf("c%d:%.1e", i - 1, rc[i]) for i in idx), " ")
end
report_gmres(lab, info, t) =
    @printf("DIAG %s  it=%d conv=%s relres=%.2e (%.1f s)  history at restart cycles %s  last-cycle reduction %.3f\n",
            lab, info.iters, info.converged, info.relres, t, hist(info.relres_cycles),
            length(info.relres_cycles) > 1 ? info.relres_cycles[end] / info.relres_cycles[end-1] : NaN)

# power iteration on I − M⁻¹J restricted to the local block (extras zeroed)
function dominant_mode(side, n, nloc; steps = 30)
    Random.seed!(7)
    to = fv3d_to()
    x = to(randn(n)); nloc < n && fill!(view(x, (nloc+1):n), 0.0)
    x ./= norm(x)
    t = similar(x); z = similar(x)
    ratios = Float64[]
    for _ in 1:steps
        K._krylov_matvec!(t, side, x)
        nloc < n && fill!(view(t, (nloc+1):n), 0.0)
        K._krylov_apply!(z, side, t)
        x .-= z
        nloc < n && fill!(view(x, (nloc+1):n), 0.0)
        r = norm(x); push!(ratios, r); x ./= max(r, eps())
    end
    fv3d_sync()
    return ratios, Vector{Float64}(Array(x))
end

# per level: smoother-only and cycle-from-this-level factors (homogeneous problem)
function level_factors(vk)
    be = K._vanka_backend(vk)
    nl = length(vk.levels)
    out = NamedTuple[]
    for l in 1:(nl-1)
        L = vk.levels[l]
        Random.seed!(11 + l)
        res = Float64[]
        for mode in (:smooth, :cycle)
            copyto!(L.x, randn(L.n)); fill!(L.b, 0.0)
            r0 = norm(L.x); rs = Float64[]
            for _ in 1:10
                mode === :smooth ? K._vanka_smooth!(be, L, 1, false) : K._vanka_cycle!(be, vk, l, false)
                r1 = norm(L.x); push!(rs, r1 / max(r0, 1e-300)); r0 = r1
            end
            push!(res, rs[end], (prod(rs))^(1 / 10))
        end
        prmin = minimum(vk.pr_host[l])
        push!(out, (; level = l, n = L.n, smooth_tail = res[1], smooth_mean = res[2], cycle_tail = res[3],
                    cycle_mean = res[4], pivot_min = prmin))
    end
    fill!(vk.levels[1].x, 0.0)
    return out
end

function coarsest_dense(vk, nodes)
    nl = length(vk.levels)
    L = vk.levels[nl]
    L.n > DENSE_MAX && return @sprintf("coarsest n = %d > FV3D_DIAG_DENSE_MAX, skipped", L.n)
    length(nodes[nl].comp) == L.n ||
        return @sprintf("coarsest n = %d but %d rebuilt nodes: structure skipped", L.n, length(nodes[nl].comp))
    A = Matrix(K._vanka_level_matrix(vk, nl))
    F = svd(A)
    λ = eigvals(A)
    nneg = count(z -> real(z) < 0, λ)
    s = sort(F.S)
    return @sprintf("coarsest n=%d  sigma min %.2e  2nd %.2e  max %.2e  cond %.2e  eig: %d of %d with Re<0, min|λ| %.2e, max|λ| %.2e  smallest singular vector: %s",
                    L.n, s[1], s[min(2, end)], s[end], s[end] / s[1], nneg, length(λ), minimum(abs, λ),
                    maximum(abs, λ), coarse_structure(F.V[:, end], nodes[nl]))
end

function analyse(cache, J, case, N, label, Ub, nodes_cache; full::Bool)
    f = cache.factor
    n = J.n; nloc = J.nlocal; nf = J.nfields
    geom = J.geom
    to = fv3d_to()
    b = to(collect(range(-1.0, 1.0; length = n)) .* (1 .+ 0.1 .* sin.(1:n)))
    pre = "$case N=$N $label"
    sides = (("fwd", f.fwd, ETA), ("transp", K._krylov_tside!(f), ETA_T))
    res = Dict{String,Any}()
    for (sname, side, tol) in sides
        vk = side.simple
        if full
            info = vk.info.vanka
            sname == "fwd" && @printf("DIAG %s hierarchy levels %s nnz %s patches %s complexity %.2f\n", pre,
                                      string(info.sizes), string(info.nnz), string(info.patches), info.operator_complexity)
            haskey(nodes_cache, length(vk.levels)) || (nodes_cache[length(vk.levels)] = level_nodes(nf, geom, length(vk.levels)))
            nodes = nodes_cache[length(vk.levels)]
            for lf in level_factors(vk)
                @printf("DIAG %s %s level %d (n %d): pivot min %.1e  smoother factor tail %.3f mean %.3f  cycle-from-level factor tail %.3f mean %.3f\n",
                        pre, sname, lf.level, lf.n, lf.pivot_min, lf.smooth_tail, lf.smooth_mean, lf.cycle_tail, lf.cycle_mean)
            end
            println("DIAG ", pre, " ", sname, " ", coarsest_dense(vk, nodes))
            rat, xm = dominant_mode(side, n, nloc)
            @printf("DIAG %s %s dominant mode of I-M^-1 J: growth tail %.3f (steps 25-30 mean %.3f)  %s\n", pre, sname,
                    rat[end], mean(rat[25:30]), fmtfs(fine_structure(xm, nf, geom.dims, geom.solid, near_mask(geom.dims, geom.solid))))
            res["$(sname)_mode_growth"] = rat[end]
        end
        # force direction c (uniform x force on fluid cells) vs the harness rhs
        if sname == "fwd"
            ch = zeros(n)
            for c in 1:prod(geom.dims)
                geom.solid[c] || (ch[nf*(c-1)+1] = 1.0)
            end
            cdev = to(ch); z = similar(cdev); t = similar(cdev)
            for (vname, v) in (("rhs", b), ("force c", cdev))
                vl = copy(v); nloc < n && fill!(view(vl, (nloc+1):n), 0.0)
                K._krylov_apply!(z, side, vl); K._krylov_matvec!(t, side, z)
                t .-= vl; nloc < n && fill!(view(t, (nloc+1):n), 0.0)
                st = lin_krylov_stationary(cache, vl; iters = 8, local_block = true)
                @printf("DIAG %s fwd on %-7s: one-application residual %.3e  stationary factor (8 steps) mean %.3f tail %.3f\n",
                        pre, vname, norm(t) / norm(vl), st.factor, st.ratios[end])
            end
        else
            st = lin_krylov_stationary(cache, b; iters = 8, local_block = true, transpose = true)
            @printf("DIAG %s transp on rhs    : stationary factor (8 steps) mean %.3f tail %.3f\n", pre, st.factor, st.ratios[end])
        end
        info, t = gmres_side(side, b, tol)
        report_gmres("$pre $sname GMRES(50)", info, t)
        res["$(sname)_gmres50_iters"] = info.iters; res["$(sname)_gmres50_conv"] = info.converged
        res["$(sname)_gmres50_relres"] = info.relres
        if full
            info, t = gmres_side(side, b, tol; deflate = DEFL)
            report_gmres("$pre $sname GMRES-DR(50,$DEFL)", info, t)
            res["$(sname)_dr_iters"] = info.iters; res["$(sname)_dr_conv"] = info.converged
            info, t = gmres_side(side, b, tol; restart = 200)
            report_gmres("$pre $sname GMRES(200)", info, t)
            res["$(sname)_gmres200_iters"] = info.iters; res["$(sname)_gmres200_conv"] = info.converged
            info, _ = gmres_side(side, b, 1e-30; maxiter = 100, ritz = true)
            θ = sort(info.ritz; by = abs)
            @printf("DIAG %s %s harmonic Ritz values of J M^-1 (cycle 2): %d of %d with Re<0; smallest |θ|: %s; largest |θ| %.2e\n",
                    pre, sname, count(z -> real(z) < 0, θ), length(θ),
                    join((@sprintf("%.2e%+.2ei", real(z), imag(z)) for z in θ[1:min(6, end)]), ", "),
                    isempty(θ) ? NaN : abs(θ[end]))
        end
        flush(stdout)
    end
    return res
end

fv3d_banner("i4_vanka_diag")
@printf("   P4 base: prolong %s omega %.2f cycle %s sweeps %d coarse %d scale %.2f  maxiter %d  variants %s  deflate %d\n",
        BASE.vanka_prolong, BASE.vanka_omega, BASE.mg_cycle, BASE.mg_sweeps, BASE.mg_coarse, BASE.mg_scale,
        DMAXIT, join(VARIANTS, ","), DEFL)
results = Dict{String,Any}("backend" => FV3D_BACKEND, "base" => Dict(String(k) => string(v) for (k, v) in pairs(BASE)))
NS = fv3d_ints("FV3D_NS", FV3D_GPU ? "32,48" : "6")
for case in CASES, N in NS
    case == "source07" && !(N in LOWRE_NS) && continue
    key = "$(case)_N$N"
    row = Dict{String,Any}()
    t0 = time()
    try
        p = duct(N, case)
        to = fv3d_to()
        # (0) direct state
        sD = fv_newton_solver(p; to, linsolve = fv3d_linsolve())
        U = to(K._fv_initial_force!(zeros(fv_nunknowns(p)), p))
        tD = @elapsed (stD = fv_newton!(sD, U; newton_opts(case)...); fv3d_sync())
        Ub = Vector{Float64}(Array(U))
        Re = mean_u(p, Ub) / NU0
        @printf("DIAG %s direct Newton conv=%s it=%d %.1f s  Re=%.1f  (%d unknowns)\n", key, stD.converged,
                stD.iterations, tD, Re, length(Ub))
        row["direct_converged"] = stD.converged; row["Re"] = Re
        release_all!(sD); sD = nothing
        flush(stdout)
        # (1) base P4 at the direct state
        sK = fv_newton_solver(p; to, linsolve = tagof())
        fv_newton_finalize!(sK, to(Ub))
        nodes_cache = Dict{Int,Any}()
        merge!(row, Dict("base_" * k => v for (k, v) in analyse(sK.cache, sK.J, case, N, "base", Ub, nodes_cache; full = true)))
        # flowrate: same matrix without the border (local block alone), driven like the source case
        if case == "flowrate"
            J = sK.J
            hrp = Array(J.rowptr); hcv = Array(J.colval); hv = Array(J.nzval)
            nl = J.nlocal
            lrp = Int32[1]; lcv = Int32[]; lv = Float64[]
            for i in 1:nl
                for q in hrp[i]:(hrp[i+1]-1)
                    c = hcv[q]
                    c <= nl && (push!(lcv, c); push!(lv, hv[q]))
                end
                push!(lrp, length(lcv) + 1)
            end
            AL = krylov_csr_matrix(to, lrp, lcv, nl; nfields = J.nfields, pslot = J.pslot, nlocal = nl,
                                   geometry = (; dims = J.geom.dims, solid = J.geom.solid))
            copyto!(AL.nzval, lv)
            cL = lin_factorize(AL; backend = tagof())
            bL = to(collect(range(-1.0, 1.0; length = nl)) .* (1 .+ 0.1 .* sin.(1:nl)))
            for (sname, side, tol) in (("fwd", cL.factor.fwd, ETA), ("transp", K._krylov_tside!(cL.factor), ETA_T))
                info, t = gmres_side(side, bL, tol)
                report_gmres("$key local block only (no border) $sname GMRES(50)", info, t)
                row["local_only_$(sname)_iters"] = info.iters; row["local_only_$(sname)_conv"] = info.converged
            end
            release_all!(cL); cL = nothing; AL = nothing      # cL holds AL: freed with it
        end
        # (2) variants on the same J
        for v in VARIANTS
            cV = lin_factorize(sK.J; backend = tagof(VARIANT_DEFS[v]))
            merge!(row, Dict("$(v)_" * k => x for (k, x) in analyse(cV, sK.J, case, N, v, Ub, nodes_cache; full = false)))
            # not release!: the cache holds sK.J (shared values); the GC frees the hierarchy
            cV = nothing; GC.gc(true); fv3d_free()
        end
        release_all!(sK); sK = nothing
        # optional: P4 Newton from rest (the harness path)
        if P4NEWTON
            s4 = fv_newton_solver(p; to, linsolve = tagof())
            U4 = to(K._fv_initial_force!(zeros(fv_nunknowns(p)), p))
            t4 = @elapsed (st4 = fv_newton!(s4, U4; merge(newton_opts(case), (; maxit = P4NEWTON_MAXIT))...); fv3d_sync())
            U4h = Vector{Float64}(Array(U4))
            @printf("DIAG %s P4 Newton from rest conv=%s it=%d %.1f s  state vs direct %.1e  gmres per step %s\n", key,
                    st4.converged, st4.iterations, t4, norm(U4h - Ub) / norm(Ub),
                    string([e.iters for e in lin_krylov_log(s4.cache)]))
            row["p4newton_converged"] = st4.converged
            release_all!(s4); s4 = nothing
        end
    catch err
        println("ERROR ", key, ": ", first(sprint(showerror, err, catch_backtrace()), 1500))
        row["error"] = first(sprint(showerror, err), 300)
        release_all!(nothing)
    end
    @printf("DIAG %s done in %.0f s\n", key, time() - t0)
    flush(stdout)
    results[key] = row
    release_all!(nothing)
end
fv3d_write(fv3d_out("i4_vanka_diag"), results)
