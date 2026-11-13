# Increment I4 measurement (plan "solveur FV stationnaire 3D", branch K, input
# of decision D2): restarted GMRES on the assembled Jacobian (`KrylovBackendTag`)
# with preconditioner P1 (ILU(0) of J) and P2 (block SIMPLE: ILU(0) of the
# velocity block, S̃ = C − D diag(A)⁻¹ G by cuDSS LU) side by side, against the
# direct cuDSS branch D.
#
# Case: periodic-x square duct (H = 1, h = 1/N) with an off-centre obstacle on
# the bottom wall, Carreau–Yasuda law (ν0 = 0.02, ν∞ = 0.002, λ = 1, a = 2,
# n = 0.5), Re = U H / ν0 = 50, two drives:
#   "flowrate": imposed mean velocity U = 1 (u_mean_target, flow-rate border
#               unknown in the Jacobian);
#   "source"  : uniform body force sized for U ≈ 1 on the Newtonian duct
#               (f = U ν0 / (0.0351 H²), times FV3D_SOURCE_SCALE, default 1);
#               the actual Re is reported (60–78: the shear-thinning law
#               raises U above 1). Pseudo-time step: fv_newton! shifts the
#               momentum rows by h^3/Δτ (cell volume, the residual's scale), so
#               dtau0 = FV3D_SOURCE_DTAU (default 1) is a physical step in 2D
#               and 3D. This case uses it with full force first,
#               and, should a solve still fail,
#               continuation in the force (every branch, D included): stages
#               f_k = s_k f with s rising from FV3D_SOURCE_RAMP (default 1,
#               i.e. full force first) to 1, each a Newton + PTC solve (at
#               most FV3D_STAGE_MAXIT iterations, default 40; the last stage
#               the full budget) from the previous state; a failed stage is
#               retried with half the increment, down to 1/64; a stage is
#               accepted when it converges, or stagnates with an inf residual
#               <= 1e-8 (intermediate stages only). The stage list is reported.
# For each case, N in FV3D_NS (default 24,32,48 on GPU, 6,8 on CPU) and each
# preconditioner in FV3D_PRECONDS (default ilu0,simple; simple_ilu = P2 with an
# ILU(0) of S̃ instead of its LU; mg = P3, block SIMPLE with one multigrid cycle
# for the velocity block and one for S̃; mg_lu = P3s, P3 with the LU of S̃;
# lsc = P3L, block upper triangular with the LSC Schur approximation, multigrid
# for A and S̃; lsc_lu = P3Ls, P3L with the LU of S̃; see
# src/solve/linear_solve_block_schur.jl):
#   (1) Newton from rest with pseudo-transient continuation on branch K
#       (forward GMRES tolerance FV3D_ETA, default 1e-6); GMRES iterations of
#       every Newton step;
#   (2) at the converged state, re-assembled J: one forward solve at η and one
#       transposed solve at FV3D_ETA_T (default 1e-10): iterations, time,
#       device bytes of the call; pivot report; P2 block sizes;
#   (3) adjoint gradient w.r.t. the five Carreau parameters;
#   (4) device memory in use after the build, the Newton solve and the adjoint,
#       relative to the memory in use before the build (every earlier solver is
#       released first: its device arrays freed and cuDSS objects finalized);
#   (5) for N <= FV3D_DIRECT_MAX (default 32): branch D (cuDSS) once per
#       (case, N), state and gradient compared with each K run.
# D2 gates (on the preconditioners in FV3D_GATE, default mg when mg is in
# FV3D_PRECONDS, else simple; the others are reported only): every K Newton
# solve and every transposed solve converge; GMRES iterations at the largest N
# <= 200 and growing at most linearly in N (for P1/P2 the forward solve of
# step (2); for P3/P3s/P3L/P3Ls the largest of that count and of every
# Newton-step solve: the step-(2) solve has a synthetic O(1) right-hand side,
# also in the continuity rows, and at 48³–64³ with P3 it needs about twice the
# worst Newton step, so it usually sets the gated count; both are printed);
# device memory <= FV3D_MEM_GB (default 9) for N <= FV3D_MEM_NMAX
# (default 48; every N for P1/P2); K vs D: state and gradient within 1e-6
# relative (1e-8 for P3/P3s/P3L/P3Ls). For those one line
# "D2 <label> <case>: PASS|FAIL (...)" per case sums the gate up. A NOTE line
# flags every solve whose GMRES count exceeds the restart length.
#
#   FV3D_BACKEND=cuda FV3D_NS=24,32,48 julia --project=<env> benchmarks/fv3d/i4_krylov_measure.jl
#   FV3D_BACKEND=cpu  FV3D_NS=6,8      julia --project benchmarks/fv3d/i4_krylov_measure.jl   (smoke)
# Other knobs: FV3D_RESTART (50), FV3D_MAXITER (2000), FV3D_TRISOLVE
# (persistent | simple), FV3D_CASES (flowrate,source). P3 knobs (see
# KrylovBackendTag): FV3D_MG_CYCLE (W | V), FV3D_MG_SMOOTHER (l1jacobi |
# jacobi | ilu0), FV3D_MG_SWEEPS (2), FV3D_MG_CYCLES (1), FV3D_MG_COARSE
# (2000), FV3D_MG_OMEGA (0 = smoother default), FV3D_MG_SCALE (1).
# FV3D_TAG_EXTRA = any KrylovBackendTag options "name=value,…", applied last to
# every K preconditioner (they override the label's and the FV3D_MG_* ones), e.g.
#   FV3D_PRECONDS=lsc FV3D_TAG_EXTRA="mg_cycles=4"
#   FV3D_PRECONDS=mg FV3D_TAG_EXTRA="block=upper,schur_approx=lsc,mg_cycles=4"
#   FV3D_TAG_EXTRA="schur_q=scalar,schur_qscale=volume"   (Q = h³ I, per N)
# Use FV3D_OUT to keep the TOML file of each variant.
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
include(joinpath(@__DIR__, "fv3d_common.jl"))
using Statistics

const ETA = parse(Float64, get(ENV, "FV3D_ETA", "1e-6"))
const ETA_T = parse(Float64, get(ENV, "FV3D_ETA_T", "1e-10"))
const RESTART = parse(Int, get(ENV, "FV3D_RESTART", "50"))
const MAXITER = parse(Int, get(ENV, "FV3D_MAXITER", "2000"))
const TRISOLVE = Symbol(get(ENV, "FV3D_TRISOLVE", "persistent"))
const DIRECT_MAX = parse(Int, get(ENV, "FV3D_DIRECT_MAX", FV3D_GPU ? "32" : "8"))
const MEM_GB = parse(Float64, get(ENV, "FV3D_MEM_GB", "9"))
const CASES = split(get(ENV, "FV3D_CASES", "flowrate,source"), ','; keepempty = false)
const PRECONDS = split(get(ENV, "FV3D_PRECONDS", "ilu0,simple"), ','; keepempty = false)
const GATED_DEFAULT = filter(in(PRECONDS), ["mg", "lsc", "lsc_lu"])
const GATED = split(get(ENV, "FV3D_GATE", isempty(GATED_DEFAULT) ? "simple" : join(GATED_DEFAULT, ",")), ',';
                    keepempty = false)
const MEM_NMAX = parse(Int, get(ENV, "FV3D_MEM_NMAX", "48"))
const NU0 = 0.02
const THETA = [NU0, 0.002, 1.0, 2.0, 0.5]
const UREF = 1.0                                  # Re = UREF * H / NU0 = 50
const NEWTON = (; atol = 1e-10, rtol = 1e-12, maxit = 80, dtau0 = 1.0, dtau_max = 1e4)
const SOURCE_SCALE = parse(Float64, get(ENV, "FV3D_SOURCE_SCALE", "1"))
const SOURCE_RAMP = parse(Float64, get(ENV, "FV3D_SOURCE_RAMP", "1"))
const SOURCE_DTAU = parse(Float64, get(ENV, "FV3D_SOURCE_DTAU", "1"))
const STAGE_MAXIT = parse(Int, get(ENV, "FV3D_STAGE_MAXIT", "40"))
const MG = (; mg_cycle = Symbol(get(ENV, "FV3D_MG_CYCLE", "W")),
            mg_smoother = Symbol(get(ENV, "FV3D_MG_SMOOTHER", "l1jacobi")),
            mg_sweeps = parse(Int, get(ENV, "FV3D_MG_SWEEPS", "2")),
            mg_cycles = parse(Int, get(ENV, "FV3D_MG_CYCLES", "1")),
            mg_coarse = parse(Int, get(ENV, "FV3D_MG_COARSE", "2000")),
            mg_omega = parse(Float64, get(ENV, "FV3D_MG_OMEGA", "0")),
            mg_scale = parse(Float64, get(ENV, "FV3D_MG_SCALE", "1")))

const EXTRA = fv3d_tag_opts(get(ENV, "FV3D_TAG_EXTRA", ""))   # fv3d_common.jl

const PLABEL = Dict("ilu0" => "P1", "simple" => "P2", "simple_ilu" => "P2i", "mg" => "P3", "mg_lu" => "P3s",
                    "lsc" => "P3L", "lsc_lu" => "P3Ls")
for pc in PRECONDS
    haskey(PLABEL, pc) ||
        error("FV3D_PRECONDS: unknown preconditioner $pc (ilu0, simple, simple_ilu, mg, mg_lu, lsc, lsc_lu)")
end
ismg(pc) = pc in ("mg", "mg_lu", "lsc", "lsc_lu")

function ktag(pc, N)
    d = Dict{Symbol,Any}(:rtol => ETA, :rtol_transpose => ETA_T, :restart => RESTART,
                         :maxiter => MAXITER, :trisolve => TRISOLVE, :strict => false,
                         :precond => pc == "ilu0" ? :ilu0 : ismg(pc) ? :mg : :simple,
                         :schur => pc == "simple_ilu" ? :ilu0 : pc in ("mg", "lsc") ? :mg : :lu)
    merge!(d, Dict{Symbol,Any}(pairs(MG)))
    pc in ("lsc", "lsc_lu") && merge!(d, Dict{Symbol,Any}(:block => :upper, :schur_approx => :lsc))
    merge!(d, EXTRA)
    return KrylovBackendTag(; fv3d_tag_resolve(d, N, NU0)...)
end

function duct(N, case)
    bc = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)
    solid = falses(N, N, N)
    solid[(2N ÷ 5 + 1):(3N ÷ 5), 1:max(1, N ÷ 3), (N ÷ 4 + 1):(N ÷ 2)] .= true
    visc = FVLawViscosity(ClassicalLaw(CarreauYasuda(THETA...)), copy(THETA), (), 0.0)
    if case == "flowrate"
        return fv_steady_problem(N, N, N; h = 1 / N, nu = NU0, bc, solid, viscosity = visc,
                                 source = (x, y, z) -> (0.0, 0.0, 0.0, 0.0), u_mean_target = UREF)
    else
        f = SOURCE_SCALE * UREF * NU0 / 0.0351
        return fv_steady_problem(N, N, N; h = 1 / N, nu = NU0, bc, solid, viscosity = visc,
                                 source = (x, y, z) -> (f, 0.0, 0.0, 0.0))
    end
end

objective(p, case) = case == "flowrate" ? fv_flow_force_objective(p) : fv_flow_rate(p, 1)

function mean_u(p, U)
    u = fv_fields(p, U)[1]
    fluid = Array(p.tag) .!= KrakeNN.FV_CELL_SOLID
    return sum(u[fluid]) / count(fluid)
end

# device-memory estimate of P1 (bytes): J values + column indices, ILU values,
# transposed copy (values, ILU values, indices, permutation), assembler triplet
# indices, GMRES basis, ~10 work vectors. P2 replaces the two full ILU value
# arrays by the velocity-block copies (~3/4 of them, each side) plus the Schur
# LU (not estimated: reported by the measured memory).
krylov_estimate(n, nz) = nz * (8 + 4 + 8 + 8 + 8 + 4 + 4 + 8) + n * 8 * (RESTART + 1) + 10n * 8

# Free the device memory held by a finished solver now: every CuArray reached
# from it is freed, and every cuDSS object (matrix / config / data, whose
# memory is outside the CUDA.jl pool) finalized, children first. Nothing else
# is finalized (no stream or context can be reached by mistake). The solver is
# unusable afterwards; host memory is left to the GC.
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

# Continuation in the body force (case "source"): stages s f, s -> 1, each a
# Newton + PTC solve from the previous state; a failed stage is undone and
# retried with half the increment. Returns the status of the last solve with
# the full force (or of the last attempt); s.prob.src always ends at the full
# force (the adjoint sees the real problem).
function source_continuation!(s, U, stages)
    src = copy(s.prob.src)
    newton = merge(NEWTON, (; dtau0 = SOURCE_DTAU))
    frac = 0.0; step = SOURCE_RAMP
    st = nothing
    try
        while true
            f = min(1.0, frac + step)
            s.prob.src .= f .* src
            U0 = copy(U)
            st = fv_newton!(s, U; merge(newton, (; maxit = f < 1 ? STAGE_MAXIT : NEWTON.maxit))...)
            ok = st.converged || (f < 1 && st.stagnated && !isempty(st.inf) && last(st.inf) <= 1e-8)
            push!(stages, (f, st.iterations, ok))
            @printf("   source continuation: force fraction %.4f  newton it=%d  %s\n", f, st.iterations,
                    ok ? "accepted" : "rejected")
            flush(stdout)
            if ok
                frac = f
                frac >= 1 && break
                step = min(2step, 1 - frac)
            else
                copyto!(U, U0)
                step /= 2
                step < 1 / 64 && break
            end
        end
    finally
        copyto!(s.prob.src, src)
    end
    return st
end

function run_branch(p, case, linsolve)
    to = fv3d_to()
    fv3d_free()
    m0 = fv3d_devused()
    t_build = @elapsed s = fv_newton_solver(p; to, linsolve)
    m1 = fv3d_devused()
    U = to(KrakeNN._fv_initial_force!(zeros(fv_nunknowns(p)), p))
    stages = Tuple{Float64,Int,Bool}[]            # (force fraction, Newton iterations, accepted)
    t_newton = @elapsed begin
        st = case == "source" ? source_continuation!(s, U, stages) : fv_newton!(s, U; NEWTON...)
        fv3d_sync()
    end
    m2 = fv3d_devused()
    isK = linsolve isa KrylovBackendTag
    newton_log = isK ? copy(lin_krylov_log(s.cache)) : NamedTuple[]
    # (2) one forward and one transposed solve at the converged state
    fv_newton_finalize!(s, U)
    n = fv_nunknowns(p)
    b = to(collect(range(-1.0, 1.0; length = n)) .* (1 .+ 0.1 .* sin.(1:n)))
    x = similar(b)
    i0 = isK ? length(lin_krylov_log(s.cache)) : 0
    # timed call first, then the same call again for the device bytes (GPU only)
    t_fwd = @elapsed (lin_solve!(x, s.cache, b); fv3d_sync())
    bytes_fwd = fv3d_devalloc(() -> lin_solve!(x, s.cache, b))
    t_tr = @elapsed (lin_solve!(x, s.cache, b; transpose = true); fv3d_sync())
    bytes_tr = fv3d_devalloc(() -> lin_solve!(x, s.cache, b; transpose = true))
    # (3) adjoint gradient w.r.t. the Carreau parameters
    obj = objective(p, case)
    t_grad = @elapsed (r = fv_steady_gradient(s, U, THETA, obj; params = FVLawParameters(5)); fv3d_sync())
    m3 = fv3d_devused()
    Uh = Vector{Float64}(Array(U))
    out = Dict{String,Any}("converged" => st.converged,
                           "newton_iterations" => isempty(stages) ? st.iterations : sum(x[2] for x in stages),
                           "stages" => [collect(x) for x in stages],
                           "t_build" => t_build, "t_newton" => t_newton, "t_gradient" => t_grad,
                           "mem_before_GiB" => m0 / 2^30,
                           "mem_build_GiB" => (m1 - m0) / 2^30, "mem_solve_GiB" => (m2 - m0) / 2^30,
                           "mem_adjoint_GiB" => (m3 - m0) / 2^30, "mem_used_GiB" => m3 / 2^30,
                           "grad" => Vector{Float64}(r.grad), "residual_inf" => r.residual_inf,
                           "Re" => mean_u(p, Uh) * 1.0 / NU0,
                           "t_forward_solve" => t_fwd, "t_transpose_solve" => t_tr,
                           "bytes_forward_solve" => bytes_fwd, "bytes_transpose_solve" => bytes_tr)
    if isK
        lg = lin_krylov_log(s.cache)
        fwd = lg[i0+1]                                # (2), timed calls
        tr = lg[findfirst(e -> e.transpose, lg[i0+1:end]) + i0]
        adj = lg[end]                                 # (3), the adjoint solve
        piv = lin_krylov_pivot(s.cache)
        out["newton_gmres_iters"] = [e.iters for e in newton_log]
        out["newton_gmres_seconds"] = [e.seconds for e in newton_log]
        out["forward_iters"] = fwd.iters; out["forward_relres"] = fwd.relres
        out["forward_converged"] = fwd.converged; out["forward_seconds"] = fwd.seconds
        out["transpose_iters"] = tr.iters; out["transpose_relres"] = tr.relres
        out["transpose_converged"] = tr.converged; out["transpose_seconds"] = tr.seconds
        out["adjoint_iters"] = adj.iters; out["adjoint_converged"] = adj.converged
        out["pivot_min_ratio"] = piv.min_ratio; out["pivot_small"] = piv.n_small
        out["nnz"] = nnz(s.J)
        blk = lin_krylov_blocks(s.cache)
        blk === nothing || (out["nnz_schur"] = blk.nnz_S; out["nnz_velocity_block"] = blk.nnz_A)
        if blk !== nothing && blk.mg_A !== nothing
            out["mg_A_sizes"] = blk.mg_A.sizes; out["mg_A_complexity"] = blk.mg_A.operator_complexity
        end
        if blk !== nothing && blk.mg_S !== nothing
            out["mg_S_sizes"] = blk.mg_S.sizes; out["mg_S_complexity"] = blk.mg_S.operator_complexity
        end
    end
    release_all!(s)
    s = nothing
    return out, Uh
end

fv3d_banner("i4_krylov_measure")
@printf("   eta %.0e  eta_T %.0e  restart %d  maxiter %d  trisolve %s  preconditioners %s (gated: %s)  direct up to N=%d  memory gate %.1f GB  multigrid %s  extra tag options %s\n",
        ETA, ETA_T, RESTART, MAXITER, TRISOLVE, join(PRECONDS, ","), join(GATED, ","), DIRECT_MAX, MEM_GB,
        string(MG), string(EXTRA))
failures = String[]
notes = String[]
results = Dict{String,Any}("backend" => FV3D_BACKEND, "eta" => ETA, "eta_T" => ETA_T,
                           "restart" => RESTART, "maxiter" => MAXITER, "trisolve" => String(TRISOLVE),
                           "preconds" => collect(String.(PRECONDS)),
                           "tag_extra" => Dict(String(k) => string(v) for (k, v) in EXTRA),
                           "multigrid" => Dict(String(k) => string(v) for (k, v) in pairs(MG)))
NS = fv3d_ints("FV3D_NS", FV3D_GPU ? "24,32,48" : "6,8")
for case in CASES
    iters = Dict{String,Dict{Int,Int}}(pc => Dict{Int,Int}() for pc in PRECONDS)
    iters_newton = Dict{String,Dict{Int,Int}}(pc => Dict{Int,Int}() for pc in PRECONDS)  # worst Newton step
    iters_fwd = Dict{String,Dict{Int,Int}}(pc => Dict{Int,Int}() for pc in PRECONDS)     # step (2), synthetic rhs
    d2 = Dict{String,Vector{String}}(pc => String[] for pc in PRECONDS)   # P3 gate failures per case
    for N in NS
        key = "$(case)_N$N"
        p = duct(N, case)
        n = fv_nunknowns(p)
        nz = nnz(fv_jacobian_pattern(p))
        est = krylov_estimate(n, nz)
        @printf("   %s: n = %d  nnz = %d  P1 device estimate %.2f GiB\n", key, n, nz, est / 2^30)
        flush(stdout)
        row = Dict{String,Any}("N" => N, "unknowns" => n, "estimate_P1_GiB" => est / 2^30)
        UK = Dict{String,Any}()
        for pc in PRECONDS
            L = PLABEL[pc]
            fail(msg) = (push!(pc in GATED ? failures : notes, "$key $L $msg"); push!(d2[pc], "$key $msg"))
            try
                k, UKp = run_branch(p, case, ktag(pc, N))
                UK[pc] = (UKp, k["grad"])
                merge!(row, Dict("$(L)_" * a => v for (a, v) in k))
                # P3: the gate counts every Newton-step solve as well
                nmax = isempty(k["newton_gmres_iters"]) ? 0 : maximum(k["newton_gmres_iters"])
                iters_newton[pc][N] = nmax
                iters_fwd[pc][N] = k["forward_iters"]
                iters[pc][N] = ismg(pc) ? max(k["forward_iters"], nmax) : k["forward_iters"]
                over = [nm for (nm, c) in (("newton-step", nmax), ("forward", k["forward_iters"]),
                                           ("transposed", k["transpose_iters"]), ("adjoint", k["adjoint_iters"]))
                        if c > RESTART]
                isempty(over) || println("NOTE ", key, " ", L, ": GMRES restarted (count > restart ", RESTART,
                                         ") in the ", join(over, ", "), " solve(s)")
                isempty(k["stages"]) ||
                    @printf("SUMMARY %-3s %-8s N=%3d  force continuation stages (fraction, newton it, accepted) %s\n",
                            L, case, N, string([Tuple(x) for x in k["stages"]]))
                haskey(k, "mg_A_sizes") &&
                    @printf("SUMMARY %-3s %-8s N=%3d  multigrid A levels %s (operator complexity %.2f)%s\n",
                            L, case, N, string(k["mg_A_sizes"]), k["mg_A_complexity"],
                            haskey(k, "mg_S_sizes") ? @sprintf("  S levels %s (operator complexity %.2f)",
                                                               string(k["mg_S_sizes"]), k["mg_S_complexity"]) : "")
                @printf("SUMMARY %-3s %-8s N=%3d  Re=%.1f  newton it=%d conv=%s %.1f s  gmres per newton step %s (max %d)  fwd it (synthetic rhs)=%d (%.2f s, %d B)  transp it=%d conv=%s (%.2f s, %d B)  adjoint it=%d %.1f s  pivot min %.1e small %d  mem build/solve/adjoint %.2f/%.2f/%.2f GiB (before %.2f)%s\n",
                        L, case, N, k["Re"], k["newton_iterations"], k["converged"], k["t_newton"],
                        string(k["newton_gmres_iters"]), nmax, k["forward_iters"], k["forward_seconds"],
                        k["bytes_forward_solve"], k["transpose_iters"], k["transpose_converged"],
                        k["transpose_seconds"], k["bytes_transpose_solve"], k["adjoint_iters"],
                        k["t_gradient"], k["pivot_min_ratio"], k["pivot_small"], k["mem_build_GiB"],
                        k["mem_solve_GiB"], k["mem_adjoint_GiB"], k["mem_before_GiB"],
                        haskey(k, "nnz_schur") ? @sprintf("  nnz J %d, A %d, S %d", k["nnz"],
                                                          k["nnz_velocity_block"], k["nnz_schur"]) : "")
                flush(stdout)
                k["converged"] || fail("Newton not converged")
                k["transpose_converged"] && k["adjoint_converged"] || fail("transposed solve not converged")
                k["forward_converged"] || fail("forward solve not converged")
                FV3D_GPU && (!ismg(pc) || N <= MEM_NMAX) && k["mem_adjoint_GiB"] * 2^30 > MEM_GB * 1e9 &&
                    fail("memory above $(MEM_GB) GB")
            catch err
                if fv3d_isoom(err)
                    println("NOTE ", key, " ", L, ": out of device memory: ", first(sprint(showerror, err), 300))
                    row["$(L)_oom"] = true
                    fail("out of device memory")
                else
                    println("ERROR ", key, " ", L, ": ", first(sprint(showerror, err), 600))
                    fail("error")
                end
                flush(stdout)
                release_all!(nothing)
            end
        end
        if N <= DIRECT_MAX && !isempty(UK)
            try
                d, UD = run_branch(p, case, fv3d_linsolve())
                merge!(row, Dict("D_" * a => v for (a, v) in d))
                @printf("SUMMARY D   %-8s N=%3d  newton it=%d %.1f s  adjoint %.2f s  mem build/solve/adjoint %.2f/%.2f/%.2f GiB\n",
                        case, N, d["newton_iterations"], d["t_newton"], d["t_gradient"],
                        d["mem_build_GiB"], d["mem_solve_GiB"], d["mem_adjoint_GiB"])
                for (pc, (Uk, gk)) in UK
                    L = PLABEL[pc]
                    eU = norm(Uk - UD) / norm(UD)
                    eg = norm(gk - d["grad"]) / norm(d["grad"])
                    row["$(L)vsD_state_rel"] = eU; row["$(L)vsD_grad_rel"] = eg
                    @printf("SUMMARY %-3s vs D %-8s N=%3d  state %.1e  gradient %.1e\n", L, case, N, eU, eg)
                    dst = pc in GATED ? failures : notes
                    tolKD = ismg(pc) ? 1e-8 : 1e-6
                    eU <= tolKD || (push!(dst, "$key $L vs D state $(eU)"); push!(d2[pc], "$key vs D state $(eU)"))
                    eg <= tolKD || (push!(dst, "$key $L vs D gradient $(eg)"); push!(d2[pc], "$key vs D gradient $(eg)"))
                end
                flush(stdout)
            catch err
                if fv3d_isoom(err)
                    println("NOTE ", key, " D: out of device memory (branch D does not fit)")
                    row["D_oom"] = true
                else
                    println("ERROR ", key, " D: ", first(sprint(showerror, err), 600))
                    push!(failures, "$key D error")
                end
                flush(stdout)
                release_all!(nothing)
            end
        end
        results[key] = row
        release_all!(nothing)
    end
    for pc in PRECONDS
        it = iters[pc]
        isempty(it) && continue
        L = PLABEL[pc]
        dst = pc in GATED ? failures : notes
        Nmin, Nmax = minimum(keys(it)), maximum(keys(it))
        if it[Nmax] > 200
            push!(dst, "$case $L forward GMRES iterations $(it[Nmax]) > 200 at N=$Nmax")
            push!(d2[pc], "GMRES iterations $(it[Nmax]) > 200 at N=$Nmax")
        end
        if Nmax > Nmin && it[Nmax] / max(it[Nmin], 1) > Nmax / Nmin
            push!(dst, "$case $L GMRES iterations grow faster than N ($(it[Nmin]) -> $(it[Nmax]))")
            push!(d2[pc], "growth faster than N ($(it[Nmin]) -> $(it[Nmax]))")
        end
        @printf("SUMMARY %-3s %s %s GMRES iterations vs N: %s\n", L, case,
                ismg(pc) ? "max(forward, Newton-step) [gated]" : "forward [gated]",
                join(("N=$N: $(it[N])" for N in sort(collect(keys(it)))), ", "))
        @printf("SUMMARY %-3s %s worst Newton step / synthetic-rhs forward vs N: %s\n", L, case,
                join(("N=$N: $(iters_newton[pc][N]) / $(iters_fwd[pc][N])" for N in sort(collect(keys(it)))), ", "))
    end
    for pc in PRECONDS
        ismg(pc) || continue
        L = PLABEL[pc]
        Ns = sort(collect(keys(iters[pc])))
        ok = isempty(d2[pc]) && !isempty(Ns) && length(Ns) == length(NS)
        @printf("D2 %s %s: %s (N %s; %s)\n", L, case, ok ? "PASS" : "FAIL", join(NS, ","),
                isempty(d2[pc]) ? (length(Ns) == length(NS) ? "all gates met" : "some N did not run") :
                join(d2[pc], "; "))
    end
end
for nt in notes
    println("NOTE (not gated) ", nt)
end
results["failures"] = failures
results["notes"] = notes
fv3d_write(fv3d_out("i4_krylov_measure"), results)
exit(fv3d_verdict("i4_krylov_measure", failures))
