# I5 diagnostic of the block preconditioners of branch K (input of D2): where
# do the GMRES iterations of P3 come from, the inner solvers (multigrid on A and
# on S̃) or the outer block approximation (SIMPLE / triangular, Schur
# approximation)? Shared measuring tool: every configuration is a set of
# `KrylovBackendTag` options given on the command line, so any preconditioner
# reachable through the tag can be plugged in.
#
# Case: the flow-rate duct of i4_krylov_measure.jl (periodic-x square duct with
# an off-centre obstacle, Carreau–Yasuda, Re 50, imposed mean velocity), or the
# body-force duct ("source", plain PTC Newton, no force continuation). For each
# N in FV3D_NS the converged state is computed once (branch D for
# N <= FV3D_DIRECT_MAX, else P3 with default options), the Jacobian J is
# assembled once at that state (no PTC shift), and every configuration builds
# its own preconditioner on a copy of J and measures:
#   (a) each multigrid hierarchy of the configuration (velocity block A, Schur
#       matrix S̃) as a stationary iteration x ← x + M (b − A x): residual
#       reduction factor per cycle, from every starting level l (level l's
#       matrix solved by the cycle restricted to levels l…end), mean over the
#       last half of FV3D_STAT_CYCLES cycles. ρ < 1 and flat in l and N =
#       convergent, h-independent cycle; ρ ≥ 1 = the cycle amplifies some
#       modes (then more cycles per application make the preconditioner worse);
#   (b) GMRES on A alone, right-preconditioned by the configured velocity-block
#       solver (ILU(0), multigrid, exact LU): inner-solver quality on A;
#   (c) GMRES on S̃ alone, right-preconditioned by the configured Schur solver:
#       inner-solver quality on S̃;
#   (e) when the velocity block is exact (`ablock=lu`): GMRES on the TRUE Schur
#       complement S = C − D A⁻¹ G (applied with the exact A⁻¹), right-
#       preconditioned by the configured Schur approximation Ŝ⁻¹: quality of
#       the Schur approximation alone (SIMPLE, SIMPLEC, LSC, PCD, shift);
#   (d) GMRES on J with the full block preconditioner, forward (η = FV3D_ETA)
#       and transposed (FV3D_ETA_T, if FV3D_TRANSPOSE=1). For N <= FV3D_EXACT_MAX
#       every block configuration is also run with exact inner solves
#       (`ablock=lu, schur=lu`, label "<label>+exact"): its count is the
#       quality of the block approximation itself, the difference with the
#       inexact run the cost of the inner solves.
# Every (b)–(e) GMRES uses restart FV3D_RESTART, at most FV3D_MAXITER
# iterations, tolerance FV3D_ETA, a fixed random right-hand side.
#
# Configurations: FV3D_TAGS = "label:opt=val,opt=val;label2:…", options are
# KrylovBackendTag fields (Symbol values with or without ':'), e.g.
#   FV3D_TAGS="P3:precond=mg;P3-lsc:precond=mg,block=upper,schur_approx=lsc"
# FV3D_TAG_BASE = options applied to every configuration (overridden by the
# configuration's own), e.g. "mg_sweeps=4". `schur_qscale=volume` (h³) or
# `=stokes` (6 ν0 h) is resolved per N (with `schur_q=scalar`). Default FV3D_TAGS: P3 (SIMPLE),
# P3-upper, P3-simplec, P3-shift4, P3-lsc (upper), P3-pcd (upper).
#
#   FV3D_BACKEND=cuda FV3D_NS=16,24,32 julia --project=<env> benchmarks/fv3d/i5_mg_diagnose.jl
#   FV3D_BACKEND=cpu  FV3D_NS=6,8      julia --project=<env> benchmarks/fv3d/i5_mg_diagnose.jl   (smoke)
# Other knobs: FV3D_CASES (flowrate), FV3D_ETA (1e-6), FV3D_ETA_T (1e-10),
# FV3D_RESTART (50), FV3D_MAXITER (2000), FV3D_STAT_CYCLES (20),
# FV3D_DIRECT_MAX (32 GPU / 8 CPU), FV3D_EXACT (1), FV3D_EXACT_MAX (32 GPU /
# 8 CPU), FV3D_TRANSPOSE (1), FV3D_PARTS (abcde: the measurements to run).
# Output: SUMMARY lines (one per measurement, then iterations vs N per
# configuration) and a TOML file (fv3d_out). No gate: exit code 0 unless a
# run raised.
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
include(joinpath(@__DIR__, "fv3d_common.jl"))
using Random

const K = KrakeNN
const ETA = parse(Float64, get(ENV, "FV3D_ETA", "1e-6"))
const ETA_T = parse(Float64, get(ENV, "FV3D_ETA_T", "1e-10"))
const RESTART = parse(Int, get(ENV, "FV3D_RESTART", "50"))
const MAXITER = parse(Int, get(ENV, "FV3D_MAXITER", "2000"))
const STAT_CYCLES = parse(Int, get(ENV, "FV3D_STAT_CYCLES", "20"))
const DIRECT_MAX = parse(Int, get(ENV, "FV3D_DIRECT_MAX", FV3D_GPU ? "32" : "8"))
const EXACT = get(ENV, "FV3D_EXACT", "1") == "1"
const EXACT_MAX = parse(Int, get(ENV, "FV3D_EXACT_MAX", FV3D_GPU ? "32" : "8"))
const TRANSPOSE = get(ENV, "FV3D_TRANSPOSE", "1") == "1"
const PARTS = lowercase(get(ENV, "FV3D_PARTS", "abcde"))
const CASES = split(get(ENV, "FV3D_CASES", "flowrate"), ','; keepempty = false)
const NU0 = 0.02
const THETA = [NU0, 0.002, 1.0, 2.0, 0.5]
const UREF = 1.0
const NEWTON = (; atol = 1e-10, rtol = 1e-12, maxit = 80, dtau0 = 1.0, dtau_max = 1e4)
const DEFAULT_TAGS = "P3:precond=mg;P3-upper:precond=mg,block=upper;" *
                     "P3-simplec:precond=mg,schur_approx=simplec;P3-shift4:precond=mg,schur_shift=4;" *
                     "P3-lsc:precond=mg,block=upper,schur_approx=lsc;P3-pcd:precond=mg,block=upper,schur_approx=pcd"

# ---- configurations from the environment ----

parse_opts(str) = fv3d_tag_opts(str)                 # fv3d_common.jl

function parse_configs(str)
    cfgs = Tuple{String,Dict{Symbol,Any}}[]
    for item in split(str, ';'; keepempty = false)
        lo = split(strip(item), ':'; limit = 2)
        push!(cfgs, (String(strip(lo[1])), length(lo) == 2 ? parse_opts(lo[2]) : Dict{Symbol,Any}()))
    end
    return cfgs
end

const BASE = parse_opts(get(ENV, "FV3D_TAG_BASE", ""))
const CONFIGS = parse_configs(get(ENV, "FV3D_TAGS", DEFAULT_TAGS))

function make_tag(opts, N)
    # precond = :ilu0 unless the configuration names one (the tag default,
    # :auto, would be P4 on these 3D Jacobians)
    d = Dict{Symbol,Any}(:rtol => ETA, :rtol_transpose => ETA_T, :restart => RESTART,
                         :maxiter => MAXITER, :strict => false, :precond => :ilu0)
    merge!(d, BASE, opts)
    return KrylovBackendTag(; fv3d_tag_resolve(d, N, NU0)...)
end

isblock(tag) = tag.precond in (:simple, :mg)
isexact(tag) = isblock(tag) && K._krylov_ablock_kind(tag) === :lu && K._krylov_schur_kind(tag) === :lu

# ---- problem and state ----

function duct(N, case)
    bc = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)
    solid = falses(N, N, N)
    solid[(2N ÷ 5 + 1):(3N ÷ 5), 1:max(1, N ÷ 3), (N ÷ 4 + 1):(N ÷ 2)] .= true
    visc = FVLawViscosity(ClassicalLaw(CarreauYasuda(THETA...)), copy(THETA), (), 0.0)
    if case == "flowrate"
        return fv_steady_problem(N, N, N; h = 1 / N, nu = NU0, bc, solid, viscosity = visc,
                                 source = (x, y, z) -> (0.0, 0.0, 0.0, 0.0), u_mean_target = UREF)
    end
    f = UREF * NU0 / 0.0351
    return fv_steady_problem(N, N, N; h = 1 / N, nu = NU0, bc, solid, viscosity = visc,
                             source = (x, y, z) -> (f, 0.0, 0.0, 0.0))
end

# Free the device memory of a finished object (every CuArray reached from it,
# every cuDSS object finalized), as in i4_krylov_measure.jl.
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

function converged_state(p, N)
    to = fv3d_to()
    lin = N <= DIRECT_MAX ? fv3d_linsolve() :
          KrylovBackendTag(; rtol = ETA, restart = RESTART, maxiter = MAXITER, strict = false, precond = :mg)
    s = fv_newton_solver(p; to, linsolve = lin)
    U = to(K._fv_initial_force!(zeros(fv_nunknowns(p)), p))
    t = @elapsed (st = fv_newton!(s, U; NEWTON...); fv3d_sync())
    Uh = Vector{Float64}(Array(U))
    release_all!(s)
    return Uh, st, t, N <= DIRECT_MAX ? "D" : "P3"
end

# ---- measurements ----

randdev(rng, n) = fv3d_to()(randn(rng, n))

# (a) stationary multigrid iteration from level l: residual reduction per cycle
function mg_stationary(mg, rng)
    be = K._mg_backend(mg)
    out = NamedTuple[]
    for l in 1:length(mg.levels)-1
        L = mg.levels[l]
        copyto!(L.b, randdev(rng, L.n))
        fill!(L.x, 0.0)
        r = Float64[norm(L.b)]
        for c in 1:STAT_CYCLES
            K._mg_cycle!(be, mg, l, c == 1)
            K._mg_resid!(be, L)
            push!(r, norm(L.r))
            (!isfinite(r[end]) || r[end] > 1e12 * r[1]) && break
        end
        nc = length(r) - 1
        h = max(1, nc ÷ 2)
        rho_tail = (r[end] / r[end-h])^(1 / h)
        rho_first = r[2] / r[1]
        push!(out, (; level = l, n = L.n, rho_first, rho_tail, relres = r[end] / r[1], cycles = nc))
    end
    return out
end

# GMRES on an operator, right-preconditioned: iterations, relres, seconds
function gmres_count(apply!, b)
    x = similar(b); fill!(x, 0.0)
    t = @elapsed info = K.gnf_gmres!(x, apply!, b; tol = ETA, restart = RESTART, maxiter = MAXITER, orth = :cgs2)
    return (; iters = info.iters, relres = info.relres, converged = info.converged, seconds = t)
end

function measure(J0, tag, label, case, N, rng, row)
    Jc = KrylovCSRMatrix(J0.n, copy(J0.rowptr), copy(J0.colval), copy(J0.nzval), J0.border,
                         J0.nfields, J0.pslot, J0.nlocal, J0.geom)
    cache = nothing
    try
        t_f = @elapsed (cache = lin_factorize(Jc; backend = tag); fv3d_sync())
        sm = cache.factor.fwd.simple
        res = Dict{String,Any}("t_factorize" => t_f)
        head = @sprintf("%-16s %-8s N=%3d", label, case, N)
        if sm !== nothing
            blk = lin_krylov_blocks(cache)
            res["ablock"] = String(blk.ablock); res["schur"] = String(blk.schur)
            res["block"] = String(blk.block); res["schur_approx"] = String(blk.schur_approx)
            @printf("SUMMARY info %s  ablock %s  schur solver %s  block %s  schur_approx %s (q %s, shift %.3g)  A levels %s  S levels %s\n",
                    head, blk.ablock, blk.schur, blk.block, blk.schur_approx, blk.schur_q, blk.schur_shift,
                    blk.mg_A === nothing ? "-" : string(blk.mg_A.sizes),
                    blk.mg_S === nothing ? "-" : string(blk.mg_S.sizes))
            # (a) stationary multigrid
            if occursin('a', PARTS)
                for (name, mg) in (("A", sm.Ablk), ("S", sm.schur))
                    mg isa K._KrylovMG || continue
                    st = mg_stationary(mg, rng)
                    res["a_$(name)_rho_tail"] = [x.rho_tail for x in st]
                    res["a_$(name)_rho_first"] = [x.rho_first for x in st]
                    for x in st
                        @printf("SUMMARY a %s  %s hierarchy  start level %d (n=%d)  rho first %.3f  rho tail %.3f  relres after %d cycles %.2e\n",
                                head, name, x.level, x.n, x.rho_first, x.rho_tail, x.cycles, x.relres)
                    end
                end
            end
            # (b) GMRES on A with the velocity-block solver
            if occursin('b', PARTS)
                Arp, Acv, Avals = K._simple_ablk_csr(sm.Ablk)
                zA = similar(sm.vu); vA = similar(sm.vu)      # GMRES passes views: copy first
                opA = (w, v) -> (vA .= v; K._simple_ablk_solve!(zA, sm.Ablk, vA); K._bs_sync(zA);
                                 K._bs_spmv!(w, Arp, Acv, Avals, zA))
                g = gmres_count(opA, randdev(rng, sm.nu))
                res["b_iters"] = g.iters; res["b_converged"] = g.converged
                @printf("SUMMARY b %s  GMRES on A (solver %s): it=%d conv=%s relres %.1e  %.2f s\n",
                        head, res["ablock"], g.iters, g.converged, g.relres, g.seconds)
            end
            # (c) GMRES on S̃ with the Schur solver
            if occursin('c', PARTS)
                zS = similar(sm.vp); vS = similar(sm.vp)
                opS = (w, v) -> (vS .= v; K._bs_ssolve!(zS, sm, vS); K._bs_spmv!(w, sm.Srp, sm.Scv, sm.Sval, zS))
                g = gmres_count(opS, randdev(rng, sm.np))
                res["c_iters"] = g.iters; res["c_converged"] = g.converged
                @printf("SUMMARY c %s  GMRES on S~ (solver %s): it=%d conv=%s relres %.1e  %.2f s\n",
                        head, res["schur"], g.iters, g.converged, g.relres, g.seconds)
            end
            # (e) GMRES on the true Schur complement, preconditioned by Ŝ⁻¹ (exact A only)
            if occursin('e', PARTS) && blk.ablock === :lu
                tu1 = similar(sm.vu); tu2 = similar(sm.vu); tp = similar(sm.vp); zP = similar(sm.vp)
                vT = similar(sm.vp)
                opT = (w, v) -> (vT .= v; K._bs_schur_solve!(zP, sm, vT); K._bs_schur_true!(w, sm, zP, tu1, tu2, tp))
                g = gmres_count(opT, randdev(rng, sm.np))
                res["e_iters"] = g.iters; res["e_converged"] = g.converged
                @printf("SUMMARY e %s  GMRES on S = C - D A^-1 G with S^-1 ~ %s%s: it=%d conv=%s relres %.1e  %.2f s\n",
                        head, res["schur_approx"], blk.schur === :lu ? " (exact S~ solves)" : " (inexact S~ solves)",
                        g.iters, g.converged, g.relres, g.seconds)
            end
        end
        # (d) full preconditioner on J
        if occursin('d', PARTS)
            b = randdev(rng, J0.n)
            x = similar(b)
            lin_solve!(x, cache, b); fv3d_sync()
            f = last(lin_krylov_log(cache))
            res["d_forward_iters"] = f.iters; res["d_forward_converged"] = f.converged
            line = @sprintf("SUMMARY d %s  GMRES on J: forward it=%d conv=%s relres %.1e %.2f s",
                            head, f.iters, f.converged, f.relres, f.seconds)
            if TRANSPOSE
                lin_solve!(x, cache, b; transpose = true); fv3d_sync()
                tr = last(lin_krylov_log(cache))
                res["d_transpose_iters"] = tr.iters; res["d_transpose_converged"] = tr.converged
                line *= @sprintf("  transposed it=%d conv=%s relres %.1e %.2f s", tr.iters, tr.converged,
                                 tr.relres, tr.seconds)
            end
            println(line, @sprintf("  factorize %.2f s", t_f))
        end
        flush(stdout)
        row[label] = res
        return res
    catch err
        if fv3d_isoom(err)
            println("NOTE ", label, " ", case, " N=", N, ": out of device memory: ", first(sprint(showerror, err), 300))
            row[label] = Dict{String,Any}("oom" => true)
            return nothing
        end
        println("ERROR ", label, " ", case, " N=", N, ": ", first(sprint(showerror, err), 800))
        flush(stdout)
        rethrow()
    finally
        release_all!(cache)
        cache = nothing
    end
end

# ---- driver ----

fv3d_banner("i5_mg_diagnose")
@printf("   eta %.0e  eta_T %.0e  restart %d  maxiter %d  stationary cycles %d  parts %s  exact inner solves up to N=%d (%s)  base options %s\n",
        ETA, ETA_T, RESTART, MAXITER, STAT_CYCLES, PARTS, EXACT_MAX, EXACT ? "on" : "off", string(BASE))
for (label, opts) in CONFIGS
    println("   config ", label, ": ", string(merge(BASE, opts)))
end
flush(stdout)
results = Dict{String,Any}("backend" => FV3D_BACKEND, "eta" => ETA, "restart" => RESTART, "maxiter" => MAXITER,
                           "configs" => Dict(l => Dict(String(k) => string(v) for (k, v) in o) for (l, o) in CONFIGS))
errors = String[]
NS = fv3d_ints("FV3D_NS", FV3D_GPU ? "16,24,32" : "6,8")
dcount = Dict{String,Dict{Int,Int}}()
for case in CASES
    for N in NS
        key = "$(case)_N$N"
        p = duct(N, case)
        Uh, st, t, how = converged_state(p, N)
        @printf("SUMMARY state %-8s N=%3d  n=%d  Newton (%s) converged=%s it=%d  last residual %.1e  %.1f s\n",
                case, N, fv_nunknowns(p), how, st.converged, st.iterations,
                isempty(st.inf) ? NaN : last(st.inf), t)
        flush(stdout)
        to = fv3d_to()
        sK = fv_newton_solver(p; to, linsolve = KrylovBackendTag(; precond = :mg))
        fv_jacobian!(sK, to(Uh))
        fv3d_sync()
        J0 = sK.J
        row = Dict{String,Any}("N" => N, "unknowns" => fv_nunknowns(p), "newton_converged" => st.converged)
        rng = MersenneTwister(97)
        for (label, opts) in CONFIGS
            tag = make_tag(opts, N)
            runs = [(label, tag)]
            if EXACT && N <= EXACT_MAX && isblock(tag) && !isexact(tag)
                push!(runs, (label * "+exact", make_tag(merge(opts, Dict{Symbol,Any}(:ablock => :lu, :schur => :lu)), N)))
            end
            for (lab, tg) in runs
                try
                    r = measure(J0, tg, lab, case, N, rng, row)
                    if r !== nothing && haskey(r, "d_forward_iters")
                        get!(dcount, "$case $lab", Dict{Int,Int}())[N] = r["d_forward_iters"]
                    end
                catch err
                    push!(errors, "$key $lab")
                end
            end
        end
        results[key] = row
        release_all!(sK)
        sK = nothing
    end
end
for k in sort(collect(keys(dcount)))
    it = dcount[k]
    @printf("SUMMARY d vs N  %-28s forward GMRES it: %s\n", k,
            join(("N=$N: $(it[N])" for N in sort(collect(keys(it)))), ", "))
end
results["errors"] = errors
fv3d_write(fv3d_out("i5_mg_diagnose"), results)
isempty(errors) || println("ERRORS: ", join(errors, "; "))
exit(isempty(errors) ? 0 : 1)
