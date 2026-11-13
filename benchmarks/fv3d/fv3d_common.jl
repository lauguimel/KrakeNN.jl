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

"""Newton solve of a host problem on the selected backend; returns host U."""
function fv3d_solve(p; U0 = nothing, atol = 1e-10, rtol = 1e-12, maxit = 40, newton...)
    to = fv3d_to()
    t_setup = @elapsed s = fv_newton_solver(p; to, linsolve = fv3d_linsolve())
    U = to(U0 === nothing ? KrakeNN._fv_initial_force!(zeros(fv_nunknowns(p)), p) : Vector{Float64}(U0))
    t = @elapsed begin
        st = fv_newton!(s, U; atol, rtol, maxit, newton...)
        fv3d_sync()
    end
    return (; U = Vector{Float64}(Array(U)), stats = st, solver = s, seconds = t, t_setup)
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
