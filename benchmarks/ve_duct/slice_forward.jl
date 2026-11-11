# Viscoelastic secondary flow in a square duct on one periodic slice
# (Nx = 1, four walls) of the 3D steady FV Newton solver, with the algebraic
# Reiner-Rivlin stress tau = 2 eta(gdot) D + 4 Psi2(gdot) D.D
# (FVReinerRivlinViscosity).
#
# Case (matches the RheoTool Giesekus reference): duct side H = 1, mean axial
# velocity U = 1 imposed (global force unknown), rho = 1, eta_s = eta_p0 = 1
# (beta = 0.5, eta_0 = 2, Re = rho U H / eta_0 = 0.5), Giesekus alpha = 0.4,
# lambda = De (De = lambda U / H). Uniform N x N grid.
#
# Laws (VE_LAW):
#   giesekus_closed_form  eta and Psi2 of the Giesekus model in steady shear;
#   fitted_g              eta as above, Psi2 = -eta_p0 lambda g_G(Wi) c(log10 Wi),
#                         c piecewise linear on the nodes of VE_LAW_CSV
#                         (column log10Wi and column VE_LAW_COLUMN), constant
#                         outside the node range.
#
# Environment:
#   VE_BACKEND     cuda (default) | cpu
#   VE_N           cells per side (default 100)
#   VE_DE          comma-separated De values (default 0.1); solved in the given
#                  order, each from the previous solution (continuation) with
#                  the same solver (one symbolic analysis)
#   VE_LAW         giesekus_closed_form (default) | fitted_g
#   VE_LAW_CSV     default benchmarks/ve_duct/ve2_law_fitted_g.csv
#   VE_LAW_COLUMN  default jointD1_r0.0001_c
#   VE_ALPHA       default 0.4;  VE_BETA default 0.5 (eta_s = 2 beta, eta_p0 = 2 (1 - beta))
#   VE_EPS         shear-rate regularisation of both laws (default 1e-4)
#   VE_DTAU0       pseudo-time step of the first Newton iteration (default: none,
#                  plain Newton with line search)
#   VE_ATOL, VE_RTOL, VE_MAXIT   Newton stopping (defaults 1e-11, 1e-13, 60)
#   VE_OUT         output folder (default benchmarks/ve_duct/results)
#
# Output: <VE_OUT>/slice_<law>_N<N>_<backend>.toml (one table per De: max|u_xy|/U,
# RMS secondary speed / U, Newton iterations, residual, times, memory, RheoTool
# target where available) and one CSV per De with the cell fields
# (j, k, y, z, u, v, w, uxy). Exit code 1 if a solve does not converge.
#
#   VE_BACKEND=cuda VE_N=100 VE_DE=0.1 julia --project=<env> benchmarks/ve_duct/slice_forward.jl
#   VE_BACKEND=cpu  VE_N=12  VE_DE=0.1 julia --project benchmarks/ve_duct/slice_forward.jl   (smoke)
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

const N = parse(Int, get(ENV, "VE_N", "100"))
const DES = parse.(Float64, split(get(ENV, "VE_DE", "0.1"), ','; keepempty = false))
const LAW = get(ENV, "VE_LAW", "giesekus_closed_form")
LAW in ("giesekus_closed_form", "fitted_g") || error("VE_LAW must be giesekus_closed_form or fitted_g")
const ALPHA = parse(Float64, get(ENV, "VE_ALPHA", "0.4"))
const BETA = parse(Float64, get(ENV, "VE_BETA", "0.5"))
const ETA_S = 2 * BETA
const ETA_P = 2 * (1 - BETA)
const EPS = parse(Float64, get(ENV, "VE_EPS", "1e-4"))
const DTAU0 = haskey(ENV, "VE_DTAU0") ? parse(Float64, ENV["VE_DTAU0"]) : nothing
const ATOL = parse(Float64, get(ENV, "VE_ATOL", "1e-11"))
const RTOL = parse(Float64, get(ENV, "VE_RTOL", "1e-13"))
const MAXIT = parse(Int, get(ENV, "VE_MAXIT", "60"))
const OUT = get(ENV, "VE_OUT", joinpath(@__DIR__, "results"))
const CSV_PATH = get(ENV, "VE_LAW_CSV", joinpath(@__DIR__, "ve2_law_fitted_g.csv"))
const CSV_COLUMN = get(ENV, "VE_LAW_COLUMN", "jointD1_r0.0001_c")
# RheoTool Giesekus, graded 200 x 200 mesh: max |u_xy| / U
const RHEOTOOL = Dict(0.1 => 1.3e-3, 1.0 => 3.1e-3)

function read_law_csv(path, column)
    lines = filter(!isempty, strip.(readlines(path)))
    head = strip.(split(lines[1], ','))
    ix = findfirst(==("log10Wi"), head); ic = findfirst(==(column), head)
    ix === nothing && error("$path has no log10Wi column")
    ic === nothing && error("$path has no column $column (columns: $(join(head, ", ")))")
    rows = [parse.(Float64, split(l, ',')) for l in lines[2:end]]
    return [r[ix] for r in rows], [r[ic] for r in rows]
end

eta_theta(de) = [ETA_S, ETA_P, de, ALPHA]
function psi_law()
    LAW == "giesekus_closed_form" && return GiesekusPsi2Law(), Float64[]
    nodes, c = read_law_csv(CSV_PATH, CSV_COLUMN)
    return TabulatedGiesekusPsi2Law(nodes), c
end
const PSI_LAW, PSI_C = psi_law()
psi_theta(de) = [ETA_P; de; ALPHA; PSI_C]
theta(de) = [eta_theta(de); psi_theta(de)]

function build_problem(de)
    visc = FVReinerRivlinViscosity(FVLawViscosity(GiesekusViscosityLaw(), eta_theta(de); eps = EPS),
                                   FVLawViscosity(PSI_LAW, psi_theta(de); eps = EPS))
    F0 = 28.45 * (ETA_S + ETA_P)          # Newtonian square-duct body force for U = 1
    return fv_steady_problem(1, N, N; h = 1 / N,
                             bc = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall),
                             viscosity = visc, source = (x, y, z) -> (F0, 0.0, 0.0, 0.0), u_mean_target = 1.0)
end

function fields_out(p, Uh)
    u, v, w, _ = fv_fields(p, Uh)
    u = u[1, :, :]; v = v[1, :, :]; w = w[1, :, :]
    uxy = sqrt.(v .^ 2 .+ w .^ 2)
    return u, v, w, uxy
end

function main()
    mkpath(OUT)
    println("== slice_forward  backend=$BACKEND  N=$N  law=$LAW  De=$(join(DES, ","))  KrakeNN ",
            pkgversion(KrakeNN), "  threads ", Threads.nthreads())
    GPU && println("   GPU ", CUDA.name(CUDA.device()), "  CUDA.jl ", pkgversion(CUDA), "  CUDSS.jl ", pkgversion(CUDSS))
    flush(stdout)
    p = build_problem(DES[1])
    map = FVReinerRivlinParameters(p.visc)
    t_setup = @elapsed s = fv_newton_solver(p; to = to_device(), linsolve = linsolve())
    mem_setup = devused()
    U = to_device()(KrakeNN._fv_initial_force!(zeros(fv_nunknowns(p)), p))
    summary = Dict{String,Any}("backend" => BACKEND, "N" => N, "law" => LAW, "alpha" => ALPHA, "beta" => BETA,
                               "eta_s" => ETA_S, "eta_p0" => ETA_P, "eps" => EPS, "Re" => 1 / (ETA_S + ETA_P),
                               "unknowns" => fv_nunknowns(p), "t_setup_s" => t_setup,
                               "device_bytes_after_setup" => mem_setup)
    LAW == "fitted_g" && (summary["law_csv"] = CSV_PATH; summary["law_column"] = CSV_COLUMN;
                          summary["law_c"] = PSI_C)
    fails = String[]
    for de in DES
        fv_set_params!(s, map, theta(de))
        newton = DTAU0 === nothing ? (;) : (; dtau0 = DTAU0)
        t = @elapsed begin
            st = fv_newton!(s, U; atol = ATOL, rtol = RTOL, maxit = MAXIT, newton...)
            devsync()
        end
        ok = st.converged || (st.stagnated && last(st.inf) <= 1e3 * ATOL)
        ok || push!(fails, "De=$de not converged (inf residual $(last(st.inf)))")
        Uh = Vector{Float64}(Array(U))
        u, v, w, uxy = fields_out(p, Uh)
        Um = sum(u) / length(u)
        mx, imx = findmax(uxy)
        rms = sqrt(sum(abs2, uxy) / length(uxy))
        tag = @sprintf("De%g", de)
        csv = joinpath(OUT, "slice_$(LAW)_N$(N)_$(tag)_$(BACKEND).csv")
        open(csv, "w") do io
            println(io, "j,k,y,z,u,v,w,uxy")
            for k in 1:N, j in 1:N
                @printf(io, "%d,%d,%.8g,%.8g,%.12g,%.12g,%.12g,%.12g\n", j, k, (j - 0.5) / N, (k - 0.5) / N,
                        u[j, k], v[j, k], w[j, k], uxy[j, k])
            end
        end
        row = Dict{String,Any}("De" => de, "converged" => st.converged, "stagnated" => st.stagnated,
                               "iterations" => st.iterations, "residual_inf" => last(st.inf),
                               "seconds" => t, "U_mean" => Um, "max_uxy_over_U" => mx / Um,
                               "max_uxy_at" => [(imx[1] - 0.5) / N, (imx[2] - 0.5) / N],
                               "rms_uxy_over_U" => rms / Um, "Fx" => Uh[end],
                               "device_bytes_after_solve" => devused(), "host_maxrss_bytes" => Int(Sys.maxrss()),
                               "fields_csv" => csv)
        ref = get(RHEOTOOL, de, nothing)
        ref === nothing || (row["rheotool_max_uxy_over_U"] = ref; row["rel_dev_rheotool"] = mx / Um / ref - 1)
        summary[tag] = row
        @printf("SUMMARY De=%g  max|u_xy|/U=%.4e  rms=%.4e  iters=%d  res=%.2e  %.2f s%s\n", de, mx / Um,
                rms / Um, st.iterations, last(st.inf), t,
                ref === nothing ? "" : @sprintf("  RheoTool %.2e (%+.1f %%)", ref, 100 * (mx / Um / ref - 1)))
        flush(stdout)
    end
    toml = joinpath(OUT, "slice_$(LAW)_N$(N)_$(BACKEND).toml")
    open(io -> TOML.print(io, summary), toml, "w")
    println("wrote ", toml)
    isempty(fails) ? println("PASS slice_forward") : println("FAIL slice_forward: ", join(fails, "; "))
    return isempty(fails)
end

main() || exit(1)
