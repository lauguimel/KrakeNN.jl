# 3D natural-convection mesh-ladder sweep — backs the docs claim on
# docs/src/users/benchmarks/thermal-natural-convection.md §"3D — cubic cavity".
#
# Produces (appends to) a CSV with the SAME schema as the bundled
# benchmarks/results/repro/thermal/kraken_natconv_3d_results.csv:
#
#     Ra,N,backend,steps,Nu_kraken,Nu_fusegi,Nu_tric,err_vs_tric_pct
#
# The matrix it runs is the union needed to back ALL three claimed numbers:
#   * N = 48,64,96,128 at Ra=1e5            -> the mesh ladder (+13.8/+9.9/+6.3/...%)
#   * Ra = 1e3,1e4,1e5 at N=96              -> the per-Ra tuple (+1.45/+3.36/+6.29%)
#   * Ra = 1e4,1e5 at N=96 in F32 AND F64   -> the F32 ==/!= F64 cross-check (<0.04%)
#
# Driver: run_natural_convection_3d (src/drivers/thermal.jl), hot west / cold
# east cube, D3Q19 flow + D3Q19 temperature, Boussinesq buoyancy. Nu is the
# hot-wall surface-gradient Nusselt number returned by the driver.
#
# Reference Nusselt numbers (same as the bundled CSV / docs page):
#   Tric et al. (2000), spectral : 1.070 / 2.054 / 4.337  at Ra=1e3/1e4/1e5
#   Fusegi et al. (1991), FVM     : 1.085 / 2.10  / 4.361  at Ra=1e3/1e4/1e5
# err_vs_tric_pct is computed against the Tric reference.
#
# Configuration is via env vars so the same file drives the local smoke and
# the Aqua run:
#   KRK_NC3D_BACKEND  : cpu | metal | cuda | auto   (default auto)
#   KRK_NC3D_FT       : float32 | float64           (default float64)
#   KRK_NC3D_NLIST    : comma N list, e.g. "48,64,96,128"
#   KRK_NC3D_RALIST   : comma Ra list, e.g. "1e3,1e4,1e5"
#   KRK_NC3D_STEPS    : explicit step count (overrides the auto formula)
#   KRK_NC3D_STEPMULT : steps = round(KRK_NC3D_STEPMULT * N^2) (default 18)
#   KRK_NC3D_OUT      : output CSV path (appended; header written if new)
#
# Run from the repo root:  julia --project=. benchmarks/natconv_3d_meshladder_sweep.jl

using KrakeNN
using KernelAbstractions
using Printf

# --- reference Nusselt numbers, keyed by Ra label -----------------------------
const NU_TRIC   = Dict("1e3" => 1.070, "1e4" => 2.054, "1e5" => 4.337)
const NU_FUSEGI = Dict("1e3" => 1.085, "1e4" => 2.10,  "1e5" => 4.361)

# --- backend resolution (mirrors benchmarks/amr_d_backend_complex_benchmark) ---
_load_cuda_module() = Base.require(Base.PkgId(
    Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"))
_load_metal_module() = Base.require(Base.PkgId(
    Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962"), "Metal"))

function resolve_backend(name::AbstractString)
    name = lowercase(name)
    if name == "cpu"
        return KernelAbstractions.CPU(), "cpu"
    elseif name == "cuda"
        cuda = _load_cuda_module()
        Base.invokelatest(cuda.functional) || error("CUDA is not functional")
        return Base.invokelatest(cuda.CUDABackend), "cuda"
    elseif name == "metal"
        metal = _load_metal_module()
        Base.invokelatest(metal.functional) || error("Metal is not functional")
        return Base.invokelatest(metal.MetalBackend), "metal"
    elseif name == "auto"
        try
            cuda = _load_cuda_module()
            Base.invokelatest(cuda.functional) &&
                return Base.invokelatest(cuda.CUDABackend), "cuda"
        catch
        end
        try
            metal = _load_metal_module()
            Base.invokelatest(metal.functional) &&
                return Base.invokelatest(metal.MetalBackend), "metal"
        catch
        end
        return KernelAbstractions.CPU(), "cpu"
    end
    throw(ArgumentError("KRK_NC3D_BACKEND must be cpu, metal, cuda, or auto"))
end

# dict key, e.g. 1.0e5 -> "1e5"
ra_key(Ra) = "1e" * string(round(Int, log10(Ra)))

# CSV Ra label matching the bundled file's exponent form, e.g. 1.0e3 -> "1e+03"
ra_csv(Ra) = @sprintf("1e+%02d", round(Int, log10(Ra)))

# backend column label, e.g. cuda + Float32 -> "cuda_f32"
backend_label(name, FT) = string(name, "_", FT === Float32 ? "f32" : "f64")

# steps-to-steady: ~18*N^2 by default (160k at N=96 reproduces the bundled row),
# bumped for the convection-dominated Ra=1e5 case where the boundary layer is
# thinnest and the transient longest.
function steps_for(N::Int, Ra::Float64, stepmult::Float64, override::Int)
    override > 0 && return override
    base = round(Int, stepmult * N^2)
    Ra >= 5e4 && (base = round(Int, 1.5 * base))
    return base
end

function parse_list(env, default)
    s = get(ENV, env, default)
    return [strip(x) for x in split(s, ",") if !isempty(strip(x))]
end

function main()
    backend_name = lowercase(get(ENV, "KRK_NC3D_BACKEND", "auto"))
    ft_name      = lowercase(get(ENV, "KRK_NC3D_FT", "float64"))
    FT           = ft_name in ("float32", "f32") ? Float32 : Float64
    stepmult     = parse(Float64, get(ENV, "KRK_NC3D_STEPMULT", "18"))
    step_override = parse(Int, get(ENV, "KRK_NC3D_STEPS", "0"))
    Pr           = 0.71

    Nlist  = [parse(Int, x) for x in parse_list("KRK_NC3D_NLIST", "48,64,96,128")]
    Ralist = [parse(Float64, x) for x in parse_list("KRK_NC3D_RALIST", "1e3,1e4,1e5")]

    out = get(ENV, "KRK_NC3D_OUT",
        joinpath(@__DIR__, "results", "repro", "thermal",
                 "kraken_natconv_3d_sweep.csv"))
    mkpath(dirname(out))

    backend, bname = resolve_backend(backend_name)
    @info "natconv 3D sweep" backend=bname FT=FT Nlist=Nlist Ralist=Ralist out=out

    new_file = !isfile(out)
    open(out, "a") do io
        if new_file
            println(io, "Ra,N,backend,steps,Nu_kraken,Nu_fusegi,Nu_tric,err_vs_tric_pct")
        end
        for Ra in Ralist
            rkey = ra_key(Ra)
            rlab = ra_csv(Ra)
            haskey(NU_TRIC, rkey) || (@warn "no reference Nu for Ra=$Ra ($rkey); skipping"; continue)
            nu_tric   = NU_TRIC[rkey]
            nu_fusegi = NU_FUSEGI[rkey]
            for N in Nlist
                steps = steps_for(N, Ra, stepmult, step_override)
                blabel = backend_label(bname, FT)
                @printf("\n=== Ra=%s  N=%d  %s  steps=%d ===\n", rlab, N, blabel, steps)
                t0 = time()
                r = run_natural_convection_3d(; N=N, Ra=Ra, Pr=Pr,
                        max_steps=steps, backend=backend, FT=FT)
                dt = time() - t0
                Nu = Float64(r.Nu)
                err = abs(Nu - nu_tric) / nu_tric * 100
                finite = isfinite(Nu)
                @printf("  Nu=%.5f  Tric=%.4f  Fusegi=%.4f  err_vs_tric=%+.3f%%  (%.1fs, %s)\n",
                        Nu, nu_tric, nu_fusegi, err, dt, finite ? "finite" : "NON-FINITE")
                # CSV row: Ra label, N, backend, steps, Nu, Nu_fusegi, Nu_tric, err%
                @printf(io, "%s,%d,%s,%d,%.5f,%.4f,%.4f,%.3f\n",
                        rlab, N, blabel, steps, Nu, nu_fusegi, nu_tric, err)
                flush(io)
            end
        end
    end
    @info "natconv 3D sweep complete" out=out
end

main()
