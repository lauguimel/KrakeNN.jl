# Viscoelastic secondary flow in a square duct on one periodic slice (Nx = 1,
# four walls) of the 3D steady FV Newton solver, with the conformation tensor c
# as six extra unknowns per cell (FVConformation3D, plan
# .orchestrator/fv-conformation-plan.md, rungs 3-4).
#
# Case (RheoTool Giesekus reference, kraken-research/hidden-rheology/ve-duct):
# duct side H = 1, mean axial velocity U = 1 imposed (global force unknown),
# rho = 1, eta_s = eta_p = 1 (beta = 0.5, eta_0 = 2, Re = rho U H / eta_0 = 0.5),
# Giesekus alpha = 0.4, lambda = De (De = lambda U / H with U the MEAN axial
# velocity and H the full side: the RheoTool cases set lambda = De with
# U_mean = 1.000 and side 1). Uniform N x N grid, kinematic units (rho = 1).
#
# Continuation in De: the values of VE_DE are reached in order, each from the
# previous solution with the same solver (one pattern, one symbolic analysis,
# fv_set_conformation!). A failed Newton solve is retried once with
# pseudo-transient damping of the momentum rows (dtau0 = VE_DTAU_RETRY); if that
# fails too, the De step is halved (state restored), down to VE_DE_MIN_STEP.
#
# Environment:
#   VE_BACKEND       cuda (default) | cpu
#   VE_N             cells per side (default 100)
#   VE_DE            comma-separated De values (default 0.1,0.3,0.6,1.0)
#   VE_ALPHA         Giesekus mobility (default 0.4; 0 = Oldroyd-B)
#   VE_BETA          eta_s / eta_0 (default 0.5; eta_s = 2 beta, eta_p = 2 (1 - beta))
#   VE_ATOL, VE_RTOL, VE_MAXIT   Newton stopping (defaults 1e-11, 1e-13, 40)
#   VE_DTAU_RETRY    pseudo-step of the damped retry (default h^2 / eta_0)
#   VE_DE_MIN_STEP   smallest De sub-step (default 0.0125)
#   VE_OUT           output folder (default benchmarks/ve_duct/results)
#   VE_CSV           1 (default) writes one CSV of cell fields per De, 0 skips it
#
# Output: <VE_OUT>/slice_conformation_N<N>_<backend>.toml, one table per
# requested De: converged, Newton iterations (total and per sub-step), the De
# sub-steps actually solved, final residual, seconds, U_mean, max|u_xy|/U and
# where, RMS (L2) secondary speed / U, Fx, smallest eigenvalue of c, symmetry
# defect of the secondary field, device bytes in use after the solve (includes
# the factors) and host max RSS, RheoTool target and relative deviation; and
# one CSV per De with the cell fields (j, k, y, z, u, v, w, uxy, c_xx..c_zz).
# Exit code 1 if a requested De is not reached.
#
#   VE_BACKEND=cuda VE_N=100 julia --project=<env> benchmarks/ve_duct/slice_conformation.jl
#   VE_BACKEND=cpu  VE_N=12 VE_DE=0.1 julia --project benchmarks/ve_duct/slice_conformation.jl   (smoke)
#
# Aqua (job folder ~/kraken_ve_conf, src/ = copy of this worktree), GPU job body:
#   export JULIA_DEPOT_PATH=$HOME/kraken_ve_conf/depot JULIA_CUDA_USE_COMPAT=false
#   (env/Project.toml with CUDA 6, CUDSS 1, KrakeNN developed from src, LinearAlgebra, Printf, TOML)
#   VE_BACKEND=cuda VE_N=100 VE_DE=0.1,0.3,0.6,1.0 VE_OUT=$HOME/kraken_ve_conf/results \
#     $JL --project=env src/benchmarks/ve_duct/slice_conformation.jl > slice_conf.log 2>&1
if lowercase(get(ENV, "VE_BACKEND", "cuda")) == "cuda"
    using CUDA
    CUDA.functional() || error("CUDA not functional")
    using CUDSS
end
using KrakeNN, LinearAlgebra, Printf, TOML

const BACKEND = lowercase(get(ENV, "VE_BACKEND", "cuda"))
BACKEND in ("cpu", "cuda") || error("VE_BACKEND must be cpu or cuda, got $BACKEND")
const GPU = BACKEND == "cuda"
to_device() = GPU ? CUDA.CuArray : Array
linsolve() = GPU ? CUDABackendTag() : CPUBackendTag()
GPU ? @eval(devsync() = CUDA.synchronize()) : (devsync() = nothing)
GPU ? @eval(devused() = Int(CUDA.total_memory() - CUDA.free_memory())) : (devused() = 0)
GPU ? @eval(devtotal() = Int(CUDA.total_memory())) : (devtotal() = 0)

const N = parse(Int, get(ENV, "VE_N", "100"))
const DES = parse.(Float64, split(get(ENV, "VE_DE", "0.1,0.3,0.6,1.0"), ','; keepempty = false))
issorted(DES) && all(>(0), DES) || error("VE_DE must be positive and increasing")
const ALPHA = parse(Float64, get(ENV, "VE_ALPHA", "0.4"))
const BETA = parse(Float64, get(ENV, "VE_BETA", "0.5"))
const ETA_S = 2 * BETA
const ETA_P = 2 * (1 - BETA)
const ATOL = parse(Float64, get(ENV, "VE_ATOL", "1e-11"))
const RTOL = parse(Float64, get(ENV, "VE_RTOL", "1e-13"))
const MAXIT = parse(Int, get(ENV, "VE_MAXIT", "40"))
const DTAU_RETRY = parse(Float64, get(ENV, "VE_DTAU_RETRY", string((1 / N)^2 / (ETA_S + ETA_P))))
const DE_MIN_STEP = parse(Float64, get(ENV, "VE_DE_MIN_STEP", "0.0125"))
const OUT = get(ENV, "VE_OUT", joinpath(@__DIR__, "results"))
const WRITE_CSV = get(ENV, "VE_CSV", "1") != "0"
# RheoTool Giesekus (alpha 0.4, beta 0.5), graded 200 x 200 mesh: max |u_xy| / U_mean
const RHEOTOOL = Dict(0.1 => 1.294e-3, 0.2 => 2.88e-3, 0.3 => 3.53e-3, 0.6 => 3.61e-3, 1.0 => 3.088e-3)
const BC = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)

conformation(de) = FVConformation3D(; lambda = de, nu_p = ETA_P, alpha = ALPHA)

function build_problem(de)
    F0 = 28.45 * (ETA_S + ETA_P)          # Newtonian square-duct body force for U = 1
    return fv_steady_problem(1, N, N; h = 1 / N, nu = ETA_S, bc = BC, conformation = conformation(de),
                             source = (x, y, z) -> (F0, 0.0, 0.0, 0.0), u_mean_target = 1.0)
end

accepted(st) = st.converged || (st.stagnated && last(st.inf) <= 1e3 * ATOL)

# one Newton solve at De, plain then damped; U is left at the last iterate
function solve_at!(s, U, de, Usave)
    fv_set_conformation!(s, conformation(de))
    t = @elapsed begin
        st = fv_newton!(s, U; atol = ATOL, rtol = RTOL, maxit = MAXIT)
        devsync()
    end
    damped = false
    ok = accepted(st) && all(isfinite, st.inf)
    if !ok
        @printf("  De=%g plain Newton failed (it %d, res %.2e); damped retry dtau0=%.3g\n",
                de, st.iterations, last(st.inf), DTAU_RETRY)
        U .= Usave
        t += @elapsed begin
            st2 = fv_newton!(s, U; atol = ATOL, rtol = RTOL, maxit = 2 * MAXIT, dtau0 = DTAU_RETRY)
            devsync()
        end
        st = merge(st2, (; iterations = st.iterations + st2.iterations))
        damped = true
        ok = accepted(st2) && all(isfinite, st2.inf)
    end
    return ok, st, t, damped
end

# reach `target` from the converged state U at `from` (0 = rest); halves the step on failure
function continue_to!(s, U, from, target)
    subs = Dict{String,Any}[]
    Usave = similar(U)
    stack = [target]
    cur = from
    iters = 0; secs = 0.0; last_st = nothing
    while !isempty(stack)
        de = stack[end]
        Usave .= U
        ok, st, t, damped = solve_at!(s, U, de, Usave)
        iters += st.iterations; secs += t
        push!(subs, Dict{String,Any}("De" => de, "ok" => ok, "iterations" => st.iterations,
                                     "damped" => damped, "residual_inf" => last(st.inf), "seconds" => t))
        @printf("  sub-step De=%g  %s  it=%d  res=%.2e  %.2f s%s\n", de, ok ? "ok" : "FAILED",
                st.iterations, last(st.inf), t, damped ? "  (damped)" : "")
        flush(stdout)
        if ok
            cur = de; pop!(stack); last_st = st
        else
            U .= Usave                         # back to the last converged state (at De = cur)
            de - cur <= DE_MIN_STEP && return false, cur, iters, secs, subs, st
            push!(stack, (cur + de) / 2)
        end
    end
    return true, cur, iters, secs, subs, last_st
end

function fields_out(p, Uh)
    u, v, w, _ = fv_fields(p, Uh)
    u = u[1, :, :]; v = v[1, :, :]; w = w[1, :, :]
    c = fv_conformation_fields(p, Uh)
    c = map(a -> a[1, :, :], c)
    return u, v, w, sqrt.(v .^ 2 .+ w .^ 2), c
end

function min_eig(c)
    m = Inf
    for k in 1:N, j in 1:N
        C = Symmetric([c.xx[j, k] c.xy[j, k] c.xz[j, k]; c.xy[j, k] c.yy[j, k] c.yz[j, k];
                       c.xz[j, k] c.yz[j, k] c.zz[j, k]])
        m = min(m, eigmin(C))
    end
    return m
end

# mirror y -> 1 - y, z -> 1 - z and diagonal y <-> z defects of (v, w), relative to max |u_xy|
function symmetry_defect(v, w, mx)
    M = N + 1
    d = 0.0
    for k in 1:N, j in 1:N
        d = max(d, abs(v[j, k] + v[M - j, k]), abs(v[j, k] - v[j, M - k]),
                abs(w[j, k] - w[M - j, k]), abs(w[j, k] + w[j, M - k]), abs(v[j, k] - w[k, j]))
    end
    return d / max(mx, floatmin())
end

function main()
    mkpath(OUT)
    println("== slice_conformation  backend=$BACKEND  N=$N  alpha=$ALPHA  beta=$BETA  De=$(join(DES, ","))  KrakeNN ",
            pkgversion(KrakeNN), "  threads ", Threads.nthreads())
    GPU && println("   GPU ", CUDA.name(CUDA.device()), "  memory ", round(devtotal() / 2^30; digits = 2),
                   " GiB  CUDA.jl ", pkgversion(CUDA), "  CUDSS.jl ", pkgversion(CUDSS))
    flush(stdout)
    p = build_problem(DES[1])
    t_setup = @elapsed s = fv_newton_solver(p; to = to_device(), linsolve = linsolve())
    mem_setup = devused()
    z = zeros(1, N, N)
    U = to_device()(fv_state(p, z, z, z, z))
    summary = Dict{String,Any}("backend" => BACKEND, "N" => N, "model" => ALPHA == 0 ? "Oldroyd-B" : "Giesekus",
                               "alpha" => ALPHA, "beta" => BETA, "eta_s" => ETA_S, "eta_p" => ETA_P,
                               "Re" => 1 / (ETA_S + ETA_P), "De_definition" => "lambda U_mean / H, H = side",
                               "unknowns" => fv_nunknowns(p), "nnz_jacobian" => length(s.nzval),
                               "t_setup_s" => t_setup, "t_pattern_s" => s.t_pattern,
                               "t_coloring_s" => s.t_coloring, "device_bytes_total" => devtotal(),
                               "device_bytes_after_setup" => mem_setup, "atol" => ATOL, "rtol" => RTOL,
                               "dtau_retry" => DTAU_RETRY)
    fails = String[]
    cur = 0.0
    t_total = @elapsed for de in DES
        ok, cur, iters, secs, subs, st = continue_to!(s, U, cur, de)
        tag = @sprintf("De%g", de)
        if !ok
            push!(fails, "De=$de not reached (last converged De $cur)")
            summary[tag] = Dict{String,Any}("De" => de, "converged" => false, "iterations" => iters,
                                            "seconds" => secs, "substeps" => subs, "last_converged_De" => cur)
            @printf("SUMMARY De=%g  NOT REACHED (last converged De %g)\n", de, cur)
            break
        end
        Uh = Vector{Float64}(Array(U))
        u, v, w, uxy, c = fields_out(p, Uh)
        Um = sum(u) / length(u)
        mx, imx = findmax(uxy)
        rms = sqrt(sum(abs2, uxy) / length(uxy))
        lmin = min_eig(c)
        sym = symmetry_defect(v, w, mx)
        row = Dict{String,Any}("De" => de, "converged" => st.converged, "stagnated" => st.stagnated,
                               "iterations" => iters, "substeps" => subs,
                               "substep_De" => [x["De"] for x in subs if x["ok"]],
                               "residual_inf" => last(st.inf), "seconds" => secs, "U_mean" => Um,
                               "max_uxy_over_U" => mx / Um,
                               "max_uxy_at" => [(imx[1] - 0.5) / N, (imx[2] - 0.5) / N],
                               "rms_uxy_over_U" => rms / Um, "Fx" => Uh[end], "min_eig_c" => lmin,
                               "symmetry_defect_rel" => sym, "n_factorize" => s.n_factorize,
                               "device_bytes_after_solve" => devused(), "host_maxrss_bytes" => Int(Sys.maxrss()))
        if WRITE_CSV
            csv = joinpath(OUT, "slice_conformation_N$(N)_$(tag)_$(BACKEND).csv")
            open(csv, "w") do io
                println(io, "j,k,y,z,u,v,w,uxy,cxx,cxy,cxz,cyy,cyz,czz")
                for k in 1:N, j in 1:N
                    @printf(io, "%d,%d,%.8g,%.8g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                            j, k, (j - 0.5) / N, (k - 0.5) / N, u[j, k], v[j, k], w[j, k], uxy[j, k],
                            c.xx[j, k], c.xy[j, k], c.xz[j, k], c.yy[j, k], c.yz[j, k], c.zz[j, k])
                end
            end
            row["fields_csv"] = csv
        end
        ref = ALPHA == 0.4 && BETA == 0.5 ? get(RHEOTOOL, de, nothing) : nothing
        ref === nothing || (row["rheotool_max_uxy_over_U"] = ref; row["rel_dev_rheotool"] = mx / Um / ref - 1)
        summary[tag] = row
        @printf("SUMMARY De=%g  max|u_xy|/U=%.4e  rms=%.4e  iters=%d  substeps=%d  res=%.2e  min eig c=%.3e  sym=%.1e  %.2f s  dev %.2f GiB%s\n",
                de, mx / Um, rms / Um, iters, length(subs), last(st.inf), lmin, sym, secs, devused() / 2^30,
                ref === nothing ? "" : @sprintf("  RheoTool %.3e (%+.1f %%)", ref, 100 * (mx / Um / ref - 1)))
        lmin > 0 || push!(fails, "De=$de: c not positive definite (min eigenvalue $lmin)")
        flush(stdout)
    end
    summary["t_total_s"] = t_total
    toml = joinpath(OUT, "slice_conformation_N$(N)_$(BACKEND).toml")
    open(io -> TOML.print(io, summary), toml, "w")
    println("wrote ", toml)
    isempty(fails) ? println("PASS slice_conformation") : println("FAIL slice_conformation: ", join(fails, "; "))
    return isempty(fails)
end

main() || exit(1)
