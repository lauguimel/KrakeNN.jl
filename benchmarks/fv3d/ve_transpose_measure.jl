# Transposed solve of the 3D FV conformation adjoint (plan "discrete
# viscoelastic adjoints", step F3): J^T lambda = dJ/dU at 10 fields per cell
# (u, v, w, p, 6 conformation components) with the options of the plan's F3
# table, used as they are (no preconditioner is modified here):
#   cudss    cuDSS direct, transposed solve of the LU (CUDABackendTag; the reference)
#   auto     GMRES, KrylovBackendTag(precond = :auto): resolves to ILU(0) at 10 fields
#   lsc      GMRES + block upper-triangular LSC (precond = :mg, block = :upper,
#            schur_approx = :lsc, mg_cycles = 4, as :auto with energy)
#   vanka    GMRES + coupled Vanka multigrid (precond = :vanka, vanka_prolong =
#            :constant, as :auto without conformation); the conformation unknowns
#            are covered only by the damped Jacobi part of the smoother
# Earlier forward check (Giesekus duct 4 x 8 x 8, De 0.1 -> 0.3): the three GMRES
# options converge in the Newton solve, within 1e-14 of the direct solve.
#
# Case: square duct N^3 (h = 1/N, periodic x, walls y and z; optional rib of
# solid cells on the bottom wall, FV3D_VE_SOLID=1, the plan's F5.2 type of
# case), imposed mean axial velocity 1, Giesekus alpha = 0.3, nu_s = nu_p = 1,
# De continuation FV3D_VE_DE (default 0.1,0.2,0.3) from rest. The state is
# solved once per N with cuDSS (GMRES + block LSC if cuDSS runs out of memory).
# Right-hand side g = dJ/dU of the test objective of the derivative gate
# (stereo PIV plane, Fx and N1 terms, test/analytical/fv3d_conformation_gate_setup.jl)
# with the data from the state of the previous De step.
#
# Per option, each in a fresh solver (the previous one freed, pool trimmed):
# build, re-assembly + factorisation / preconditioner set-up at the state
# (fv_newton_finalize!), one forward solve of g at rtol FV3D_ETA (1e-6, the
# Newton tolerance) and two transposed solves at rtol_transpose = 1e-10 (the
# first includes the lazy set-up of the transposed side), GMRES at most
# FV3D_VE_MAXITER (500, plan F3.2) iterations, restart 50, non-strict.
# Reported: iterations and GMRES relative residual (forward, transposed), times,
# true residual |J^T lambda - g| / |g| (J assembled once per N, host SpMV),
# |lambda - lambda_cuDSS| / |lambda_cuDSS| (2-norm), and the device memory of
# the whole option run (build -> last solve), both measured:
#   pool_peak    high-water mark of the CUDA.jl memory pool (MEMPOOL_ATTR_USED_MEM_HIGH,
#                reset at the start) above the pool use at the start;
#   device_peak  largest device memory in use (cuMemGetInfo) above the start,
#                sampled every 2 ms on a second thread (run with julia -t 2;
#                with one thread only the end of each phase is sampled, said in
#                the output); includes memory allocated outside the pool (cuDSS).
# Measurements only, no gate. F3.2 decision line: options whose transposed solve
# converges in <= FV3D_VE_MAXITER iterations at each N; none = separate mission.
#
#   FV3D_BACKEND=cuda julia -t 2 --project=<env with CUDA, CUDSS> benchmarks/fv3d/ve_transpose_measure.jl
#   FV3D_BACKEND=cpu  FV3D_NS=6 julia --project benchmarks/fv3d/ve_transpose_measure.jl   (smoke: UMFPACK)
# Options: FV3D_NS (default 16,24,32), FV3D_VE_LARGE=1 (adds 48), FV3D_VE_OPTIONS
# (default cudss,auto,lsc,vanka), FV3D_VE_DE, FV3D_VE_SOLID, FV3D_VE_MAXITER,
# FV3D_ETA, FV3D_VE_WARMUP (warm-up size, default 8, 0 = none), FV3D_OUT.
if lowercase(get(ENV, "FV3D_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
include(joinpath(@__DIR__, "fv3d_common.jl"))
include(joinpath(@__DIR__, "..", "..", "test", "analytical", "fv3d_conformation_gate_setup.jl"))

const SCRIPT = "ve_transpose_measure"
const NS = [fv3d_ints("FV3D_NS", "16,24,32"); get(ENV, "FV3D_VE_LARGE", "0") == "1" ? [48] : Int[]]
const OPTIONS = Symbol.(split(get(ENV, "FV3D_VE_OPTIONS", "cudss,auto,lsc,vanka"), ','; keepempty = false))
const DES = fv3d_floats("FV3D_VE_DE", "0.1,0.2,0.3")
length(DES) >= 2 || error("FV3D_VE_DE needs at least two values (the data come from the last but one)")
const SOLID = get(ENV, "FV3D_VE_SOLID", "0") == "1"
const MAXITER = parse(Int, get(ENV, "FV3D_VE_MAXITER", "500"))
const ETA = parse(Float64, get(ENV, "FV3D_ETA", "1e-6"))
const ETA_T = 1e-10
const WARMUP = parse(Int, get(ENV, "FV3D_VE_WARMUP", "8"))
const STATE_NEWTON = (; atol = 1e-11, rtol = 1e-13, maxit = 40)

# ------------------------------------------------------------------ options
const KRYLOV_EXTRA = Dict{Symbol,NamedTuple}(
    :auto => (; precond = :auto),
    :lsc => (; precond = :mg, block = :upper, schur_approx = :lsc, mg_cycles = 4),
    :vanka => (; precond = :vanka, vanka_prolong = :constant))
krylov_tag(extra; maxiter = MAXITER) =
    KrylovBackendTag(; rtol = ETA, rtol_transpose = ETA_T, restart = 50, maxiter, strict = false, extra...)
option_tag(o::Symbol) = o === :cudss ? fv3d_linsolve() :
                        haskey(KRYLOV_EXTRA, o) ? krylov_tag(KRYLOV_EXTRA[o]) :
                        error("unknown option $o (cudss, auto, lsc, vanka)")
for o in OPTIONS
    option_tag(o)
end

# ------------------------------------------------------------------ case
function duct_problem(N, De)
    solid = falses(N, N, N)
    SOLID && (solid[(3N ÷ 8 + 1):(5N ÷ 8), 1:max(1, N ÷ 4), :] .= true)
    conf = FVConformation3D(; lambda = De, nu_p = 1.0, alpha = VE_ALPHA)
    return fv_steady_problem(N, N, N; h = 1 / N, nu = 1.0, bc = VE_BC, conformation = conf,
                             solid = SOLID ? solid : nothing,
                             source = (x, y, z) -> (50.0, 0.0, 0.0, 0.0), u_mean_target = 1.0)
end
rest_state(p) = (z = zeros(p.grid.Nx, p.grid.Ny, p.grid.Nz); fv_state(p, z, z, z, z))

# Free every device array and cuDSS object reached from a finished solver (as in
# i4_krylov_measure.jl), then trim the pool.
function ve_release!(x, seen = Base.IdSet{Any}(), depth = 0)
    (depth > 12 || x === nothing || isbits(x) || x isa Module || x isa DataType ||
     x isa Function || x isa String || x isa Symbol) && return
    x in seen && return
    push!(seen, x)
    if FV3D_GPU && x isa CUDA.CuArray
        CUDA.unsafe_free!(x)
        return
    end
    if x isa Array
        isbitstype(eltype(x)) || foreach(e -> ve_release!(e, seen, depth + 1), x)
        return
    end
    for f in fieldnames(typeof(x))
        isdefined(x, f) && ve_release!(getfield(x, f), seen, depth + 1)
    end
    FV3D_GPU && ismutable(x) && parentmodule(typeof(x)) === CUDSS && finalize(x)
    return
end
function ve_release_all!(s)
    try
        s === nothing || ve_release!(s)
    catch err
        println("NOTE release: ", first(sprint(showerror, err), 200))
    end
    GC.gc(true); GC.gc(true)
    fv3d_free()
    return nothing
end

# State at the last De of DES with linear solver `tag`; host data state = the
# state at the last but one De.
function solve_state(N, tag)
    to = fv3d_to()
    p = duct_problem(N, first(DES))
    s = fv_newton_solver(p; to, linsolve = tag)
    U = to(rest_state(p))
    Ud = nothing; its = Int[]
    for (q, De) in enumerate(DES)
        fv_set_conformation!(s, FVConformation3D(; lambda = De, nu_p = 1.0, alpha = VE_ALPHA))
        st = fv_newton!(s, U; STATE_NEWTON...)
        st.converged || last(st.inf) <= 1e-9 ||
            error(@sprintf("Newton did not converge at De = %g (residual %.2e)", De, last(st.inf)))
        push!(its, length(st.inf) - 1)
        q == length(DES) - 1 && (Ud = Vector{Float64}(Array(U)))
    end
    fv_check_conformation_domain(s.prob, U)
    return s, U, Ud, its
end

# ------------------------------------------------------------------ memory
pool_attr(a) = try
    Int(CUDA.attribute(UInt64, CUDA.memory_pool(CUDA.device()), a))
catch
    -1
end
pool_used() = FV3D_GPU ? pool_attr(CUDA.MEMPOOL_ATTR_USED_MEM_CURRENT) : -1
pool_high() = FV3D_GPU ? pool_attr(CUDA.MEMPOOL_ATTR_USED_MEM_HIGH) : -1
function pool_reset_high!()
    FV3D_GPU || return
    try
        CUDA.attribute!(CUDA.memory_pool(CUDA.device()), CUDA.MEMPOOL_ATTR_USED_MEM_HIGH, UInt64(0))
    catch err
        println("NOTE pool high-water reset failed: ", first(sprint(showerror, err), 120))
    end
end

mutable struct MemWindow
    dev0::Int
    pool0::Int
    peak::Threads.Atomic{Int}
    stop::Threads.Atomic{Bool}
    task::Union{Nothing,Task}
end
function mem_begin()
    fv3d_free()
    dev0 = fv3d_devused()
    pool_reset_high!()
    w = MemWindow(dev0, pool_used(), Threads.Atomic{Int}(dev0), Threads.Atomic{Bool}(false), nothing)
    if FV3D_GPU && Threads.nthreads() > 1
        ctx = CUDA.context()
        w.task = Threads.@spawn begin
            CUDA.context!(ctx)
            while !w.stop[]
                Threads.atomic_max!(w.peak, fv3d_devused())
                sleep(0.002)
            end
        end
    end
    return w
end
mem_mark!(w::MemWindow) = (fv3d_sync(); Threads.atomic_max!(w.peak, fv3d_devused()); nothing)
function mem_end!(w::MemWindow)
    mem_mark!(w)
    w.stop[] = true
    w.task === nothing || wait(w.task)
    high = pool_high()
    return (; device_peak = FV3D_GPU ? w.peak[] - w.dev0 : -1,
            pool_peak = high < 0 || w.pool0 < 0 ? -1 : high - w.pool0,
            device_after = FV3D_GPU ? fv3d_devused() - w.dev0 : -1,
            sampled = w.task !== nothing)
end

# ------------------------------------------------------------------ one option
function run_option(o::Symbol, ph, U, gh)
    to = fv3d_to(); tag = option_tag(o)
    s = nothing
    w = mem_begin()
    try
        t_build = @elapsed s = fv_newton_solver(ph; to, linsolve = tag)
        mem_mark!(w)
        t_setup = @elapsed (fv_newton_finalize!(s, U); fv3d_sync())
        mem_mark!(w)
        isK = tag isa KrylovBackendTag
        b = to(copy(gh)); x = similar(b)
        i0 = isK ? length(lin_krylov_log(s.cache)) : 0
        t_fwd = @elapsed (lin_solve!(x, s.cache, b); fv3d_sync())
        xf = Vector{Float64}(Array(x))
        mem_mark!(w)
        t_tr1 = @elapsed (lin_solve!(x, s.cache, b; transpose = true); fv3d_sync())
        mem_mark!(w)
        t_tr = @elapsed (lin_solve!(x, s.cache, b; transpose = true); fv3d_sync())
        lam = Vector{Float64}(Array(x))
        mem = mem_end!(w)
        out = Dict{String,Any}("status" => "ok", "t_build" => t_build, "t_setup" => t_setup,
                               "t_forward" => t_fwd, "t_transpose_first" => t_tr1, "t_transpose" => t_tr,
                               "pool_peak_bytes" => mem.pool_peak, "device_peak_bytes" => mem.device_peak,
                               "device_after_bytes" => mem.device_after, "device_sampled" => mem.sampled)
        if isK
            lg = lin_krylov_log(s.cache)[i0+1:end]
            fw = lg[findfirst(e -> !e.transpose, lg)]
            tr = lg[findlast(e -> e.transpose, lg)]
            out["precond"] = try String(s.cache.factor.tag.precond) catch; "?" end
            out["forward_iters"] = fw.iters; out["forward_relres"] = fw.relres
            out["forward_converged"] = fw.converged
            out["transpose_iters"] = tr.iters; out["transpose_relres"] = tr.relres
            out["transpose_converged"] = tr.converged
        else
            out["precond"] = "direct"
        end
        ve_release_all!(s); s = nothing
        return out, xf, lam
    catch err
        mem = mem_end!(w)
        ve_release_all!(s); s = nothing
        fv3d_isoom(err) || rethrow()
        println("   ", o, ": out of device memory")
        return Dict{String,Any}("status" => "oom", "device_peak_bytes" => mem.device_peak,
                                "pool_peak_bytes" => mem.pool_peak), nothing, nothing
    end
end

# J assembled once at the state (device Krylov solver, no factorisation), as host
# CSC of J^T (the CSR arrays of J read as CSC); used for the true residuals.
function host_jt(ph, U)
    s = fv_newton_solver(ph; to = fv3d_to(), linsolve = krylov_tag((; precond = :ilu0)))
    KrakeNN.fv_jacobian!(s, U)
    fv3d_sync()
    A = s.J
    JT = SparseMatrixCSC(A.n, A.n, Int.(Array(A.rowptr)), Int.(Array(A.colval)),
                         Vector{Float64}(Array(A.nzval)))
    ve_release_all!(s)
    return JT
end

relres(JT, lam, g) = norm(JT * lam - g) / norm(g)
reldiff(a, b) = norm(a - b) / norm(b)

function measure(N; quiet = false)
    to = fv3d_to()
    res = Dict{String,Any}()
    # state: cuDSS, GMRES + block LSC if cuDSS runs out of memory
    s, U, Ud, its, state_solver = nothing, nothing, nothing, Int[], "cudss"
    try
        s, U, Ud, its = solve_state(N, fv3d_linsolve())
    catch err
        fv3d_isoom(err) || rethrow()
        ve_release_all!(s)
        println("   N=$N state: cuDSS out of device memory, state with GMRES + block LSC")
        state_solver = "lsc"
        s, U, Ud, its = solve_state(N, krylov_tag(KRYLOV_EXTRA[:lsc]; maxiter = 2000))
    end
    ph = KrakeNN._fv_host_problem(s.prob)
    obj, _ = ve_objective(s.prob, Ud; to)
    gh = Vector{Float64}(Array(fv_objective_gradient!(obj, U, s.prob)[2]))
    ve_release_all!(s); s = nothing; obj = nothing
    JT = host_jt(ph, U)
    n = size(JT, 1)
    res["n"] = n; res["nnz"] = nnz(JT); res["state_solver"] = state_solver
    res["state_newton_iters"] = its
    quiet || @printf("N=%d  %d^3 cells, %d unknowns, nnz %d, De %s, state by %s (Newton its %s)\n",
                     N, N, n, nnz(JT), join(DES, "->"), state_solver, join(its, ","))
    lam_ref = nothing; x_ref = nothing
    for o in OPTIONS
        out, xf, lam = run_option(o, ph, U, gh)
        if lam !== nothing
            out["true_transpose_relres"] = relres(JT, lam, gh)
            if o === :cudss
                lam_ref, x_ref = lam, xf
            elseif lam_ref !== nothing
                out["lambda_reldiff_cudss"] = reldiff(lam, lam_ref)
                out["forward_reldiff_cudss"] = reldiff(xf, x_ref)
            end
        end
        res[String(o)] = out
        quiet || print_row(o, out)
    end
    ve_release_all!(nothing)
    return res
end

gb(b) = b < 0 ? "    -" : @sprintf("%5.2f", b / 2^30)
function print_row(o, d)
    if d["status"] != "ok"
        @printf("   %-6s %s  (device peak %s GB, pool peak %s GB)\n", o, d["status"],
                gb(get(d, "device_peak_bytes", -1)), gb(get(d, "pool_peak_bytes", -1)))
    else
        it(k) = haskey(d, k) ? @sprintf("%4d", d[k]) : "   -"
        rr(k) = haskey(d, k) ? @sprintf("%.1e", d[k]) : "      -"
        cv = haskey(d, "transpose_converged") ? (d["transpose_converged"] ? "yes" : "NO ") : "  -"
        @printf("   %-6s %-6s fwd it %s rr %s | T it %s rr %s conv %s true %.1e dlam %s | setup %7.3f s  T1 %7.3f s  T %7.3f s | peak dev %s GB pool %s GB%s\n",
                o, d["precond"], it("forward_iters"), rr("forward_relres"), it("transpose_iters"),
                rr("transpose_relres"), cv, d["true_transpose_relres"], rr("lambda_reldiff_cudss"),
                d["t_setup"], d["t_transpose_first"], d["t_transpose"], gb(d["device_peak_bytes"]),
                gb(d["pool_peak_bytes"]), d["device_sampled"] ? "" : " (end of phases only)")
    end
    flush(stdout)
end

function main()
    fv3d_banner(SCRIPT)
    println("options ", join(OPTIONS, ","), "  N ", join(NS, ","), "  De ", join(DES, "->"),
            "  solid rib ", SOLID, "  rtol forward ", ETA, "  rtol_transpose ", ETA_T,
            "  GMRES maxiter ", MAXITER)
    FV3D_GPU && Threads.nthreads() == 1 &&
        println("NOTE one thread: device peak sampled at the end of each phase only (run with julia -t 2)")
    if WARMUP > 0
        println("warm-up N=$WARMUP (compilation, not reported)"); flush(stdout)
        measure(WARMUP; quiet = true)
    end
    data = Dict{String,Any}("options" => String.(OPTIONS), "De" => DES, "solid" => SOLID,
                            "rtol" => ETA, "rtol_transpose" => ETA_T, "maxiter" => MAXITER)
    ok = Dict{Int,Vector{Symbol}}()
    for N in NS
        r = measure(N)
        data["N$N"] = r
        ok[N] = [o for o in OPTIONS if o !== :cudss && get(r[String(o)], "status", "") == "ok" &&
                 get(r[String(o)], "transpose_converged", false)]
        fv3d_write(fv3d_out(SCRIPT), data)
    end
    println("\nF3.2 transposed GMRES converged in <= $MAXITER iterations at rtol_transpose = $ETA_T:")
    for N in NS
        println("   N=$N: ", isempty(ok[N]) ? "none" : join(ok[N], ", "))
    end
    any(N -> isempty(ok[N]), NS) &&
        println("   at least one N without a converging GMRES option: plan F3.2 says separate mission, Guillaume decides")
    return 0
end

exit(main())
