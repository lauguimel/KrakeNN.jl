# Shared helpers of the 3D steady FV GPU scripts (plan "solveur FV stationnaire
# 3D", increment I1). Included at top level by each script AFTER the script has
# loaded CUDA/CUDSS at top level (never `@eval using CUDA` inside a function).
#
# Environment:
#   FV3D_BACKEND  cuda (default) | cpu   (cpu: UMFPACK, for tiny smoke runs)
#   FV3D_NS       comma-separated grid sizes (script specific default)
#   FV3D_OUT      TOML result file (default results/<script>_<backend>.toml)
using KrakeNN, LinearAlgebra, Printf, TOML
using KrakeNN.SparseArrays

const FV3D_BACKEND = lowercase(get(ENV, "FV3D_BACKEND", "cuda"))
FV3D_BACKEND in ("cpu", "cuda") || error("FV3D_BACKEND must be cpu or cuda, got $FV3D_BACKEND")
const FV3D_GPU = FV3D_BACKEND == "cuda"

# CUDA.@allocated is expanded only once CUDA is loaded (top-level eval, not in a function)
FV3D_GPU ? @eval(fv3d_devalloc(f) = CUDA.@allocated f()) : (fv3d_devalloc(f) = -1)
fv3d_to() = FV3D_GPU ? CUDA.CuArray : Array
fv3d_linsolve() = FV3D_GPU ? CUDABackendTag() : CPUBackendTag()
fv3d_sync() = FV3D_GPU && CUDA.synchronize()
# release solver factorisations kept alive by dead references (GC, then give the pool back)
FV3D_GPU ? @eval(fv3d_free() = (GC.gc(); CUDA.reclaim(); nothing)) : (fv3d_free() = (GC.gc(); nothing))
fv3d_isoom(err) = occursin("out of gpu memory", lowercase(sprint(showerror, err))) || err isa OutOfMemoryError
fv3d_devused() = FV3D_GPU ? Int(CUDA.total_memory() - CUDA.free_memory()) : 0

fv3d_ints(name, default) = parse.(Int, split(get(ENV, name, default), ','; keepempty = false))
fv3d_floats(name, default) = parse.(Float64, split(get(ENV, name, default), ','; keepempty = false))

# KrylovBackendTag options from "name=value,name=value" (FV3D_TAGS, FV3D_TAG_BASE
# of i5; FV3D_TAG_EXTRA of i4). Symbol values with or without ':'.
# `schur_qscale` also takes `volume` (h³, cell volume of the unit duct, h = 1/N)
# or `stokes` (6 ν0 h, the Stokes diagonal of A), resolved per N by
# fv3d_tag_resolve.
function fv3d_tag_opts(str)
    d = Dict{Symbol,Any}()
    for item in split(str, ','; keepempty = false)
        kv = split(strip(item), '='; limit = 2)
        length(kv) == 2 || error("option '$item': expected name=value")
        f = Symbol(strip(kv[1])); v = strip(kv[2])
        hasfield(KrylovBackendTag, f) || error("unknown KrylovBackendTag option '$f'")
        T = fieldtype(KrylovBackendTag, f)
        d[f] = f === :schur_qscale && v in ("volume", "stokes") ? Symbol(v) :
               T === Symbol ? Symbol(lstrip(v, ':')) : T === Bool ? parse(Bool, v) : parse(T, v)
    end
    return d
end

function fv3d_tag_resolve(d, N, nu0)
    q = get(d, :schur_qscale, 0.0)
    q isa Symbol || return d
    return merge(d, Dict{Symbol,Any}(:schur_qscale => q === :volume ? (1 / N)^3 : 6 * nu0 / N))
end

function fv3d_banner(script)
    println("== ", script, "  backend=", FV3D_BACKEND, "  KrakeNN ", pkgversion(KrakeNN),
            "  threads ", Threads.nthreads())
    FV3D_GPU && println("   GPU ", CUDA.name(CUDA.device()), "  CUDA.jl ", pkgversion(CUDA),
                        "  CUDSS.jl ", pkgversion(CUDSS))
    flush(stdout)
end

"""
Branch K tag of the multigrid block preconditioners: GMRES (restart 50, at most
2000 iterations, non-strict: an unconverged solve is logged, not raised) with
P3 multigrid blocks; `extra` defaults to P3L (block upper triangular, LSC
Schur approximation, 4 multigrid cycles per block). Shared by V5
(`fv5_linsolve`, options `FV5_PRECOND=p3l` / `p3`; the V5 default is P4) and
the I3 scripts (`FV3D_LINSOLVE=krylov`, which keep P3L: their GPU receipts
were run with it).
"""
const FV3D_P3_BASE = (; restart = 50, maxiter = 2000, strict = false, trisolve = :persistent,
                      precond = :mg, schur = :mg, mg_cycle = :W, mg_smoother = :l1jacobi,
                      mg_sweeps = 2, mg_coarse = 2000, mg_omega = 0.0, mg_scale = 1.0)
const FV3D_P3L_EXTRA = (; block = :upper, schur_approx = :lsc, mg_cycles = 4)
fv3d_p3_tag(; rtol = 1e-6, rtol_transpose = 1e-10, extra = FV3D_P3L_EXTRA) =
    KrylovBackendTag(; rtol, rtol_transpose, FV3D_P3_BASE..., extra...)

# Linear solver of the Newton steps chosen by FV3D_LINSOLVE (I3 scripts):
# direct (default: cuDSS on GPU, UMFPACK on CPU) | krylov (GMRES + P3L).
# FV3D_DIRECT_MAX (24): with krylov, grids N <= it are also solved with direct.
function fv3d_linsolve_kind()
    k = lowercase(get(ENV, "FV3D_LINSOLVE", "direct"))
    k in ("direct", "krylov") || error("FV3D_LINSOLVE must be direct or krylov, got $k")
    return k
end
fv3d_linsolve_env() = fv3d_linsolve_kind() == "krylov" ? fv3d_p3_tag() : fv3d_linsolve()
fv3d_direct_max() = parse(Int, get(ENV, "FV3D_DIRECT_MAX", "24"))

"""
Newton solve of a host problem on the selected backend; returns host U.
With a Krylov tag, `krylov` holds the GMRES iterations of every Newton step
(`iters`, one forward solve per step), the number of unconverged solves and
the largest final relative residual; `nothing` for a direct solver.
"""
function fv3d_solve(p; U0 = nothing, atol = 1e-10, rtol = 1e-12, maxit = 40, linsolve = fv3d_linsolve(),
                    newton...)
    to = fv3d_to()
    t_setup = @elapsed s = fv_newton_solver(p; to, linsolve)
    U = to(U0 === nothing ? KrakeNN._fv_initial_force!(zeros(fv_nunknowns(p)), p) : Vector{Float64}(U0))
    t = @elapsed begin
        st = fv_newton!(s, U; atol, rtol, maxit, newton...)
        fv3d_sync()
    end
    krylov = nothing
    if linsolve isa KrylovBackendTag
        lg = s.cache === nothing ? NamedTuple[] : lin_krylov_log(s.cache)
        krylov = (; iters = Int[e.iters for e in lg], unconverged = count(e -> !e.converged, lg),
                  relres_max = isempty(lg) ? 0.0 : maximum(e.relres for e in lg))
    end
    return (; U = Vector{Float64}(Array(U)), stats = st, solver = s, seconds = t, t_setup, krylov)
end

"""
`fv3d_solve` with the solver of FV3D_LINSOLVE. Device memory is read right
after that solve (`mem`). With krylov and `N <= fv3d_direct_max()`, the
problem is solved again with the direct solver (same arguments) once the
Krylov solver is released: `kd` = (; diff = ‖U_K − U_D‖∞ / ‖U_D‖∞,
converged, gate 1e-6), else `nothing`. The returned result drops the solver.
"""
function fv3d_solve_selected(p, N; newton...)
    ls = fv3d_linsolve_env()
    r = fv3d_solve(p; linsolve = ls, newton...)
    mem = fv3d_devused()
    r = merge(r, (; solver = nothing))
    kd = nothing
    if ls isa KrylovBackendTag && N <= fv3d_direct_max()
        fv3d_free()
        rd = fv3d_solve(p; linsolve = fv3d_linsolve(), newton...)
        diff = norm(r.U - rd.U, Inf) / norm(rd.U, Inf)
        kd = (; diff, converged = rd.stats.converged, gate = 1e-6, pass = rd.stats.converged && diff <= 1e-6)
        rd = nothing
        fv3d_free()
    end
    return r, mem, kd
end

"""Print the branch K line of a solve; push failures (unconverged GMRES, K vs D); TOML entries."""
function fv3d_report_krylov!(failures, label, r, kd)
    out = Dict{String,Any}()
    if r.krylov !== nothing
        k = r.krylov
        @printf("   %s gmres per newton step [%s] (max %d)  unconverged solves %d  max relres %.2e\n", label,
                join(k.iters, ","), isempty(k.iters) ? 0 : maximum(k.iters), k.unconverged, k.relres_max)
        k.unconverged == 0 || push!(failures, "$label: $(k.unconverged) GMRES solve(s) not converged")
        out["gmres_iters"] = k.iters
        out["gmres_unconverged"] = k.unconverged
        out["gmres_relres_max"] = k.relres_max
    end
    if kd !== nothing
        @printf("   %s K vs D: state rel diff %.2e (gate %.0e), direct converged %s  %s\n", label, kd.diff,
                kd.gate, kd.converged, kd.pass ? "ok" : "ABOVE GATE")
        kd.converged || push!(failures, "$label: direct reference solve not converged")
        kd.diff <= kd.gate || push!(failures, @sprintf("%s: K vs D state difference %.2e > %.0e", label, kd.diff, kd.gate))
        out["kd_diff"] = kd.diff
    end
    flush(stdout)
    return out
end

fv3d_order(e1, e2, N1, N2) = log(e1 / e2) / log(N2 / N1)

function fv3d_out(script)
    default = joinpath(@__DIR__, "results", "$(script)_$(FV3D_BACKEND).toml")
    return get(ENV, "FV3D_OUT", default)
end

function fv3d_write(path, data)
    mkpath(dirname(path))
    open(io -> TOML.print(io, data; sorted = true), path, "w")
    println("results written to ", path)
end

"""One PASS/FAIL line against the plan's gates; returns the process exit code."""
function fv3d_verdict(script, failures)
    if isempty(failures)
        println("PASS ", script)
        return 0
    end
    println("FAIL ", script, ": ", join(failures, "; "))
    return 1
end
