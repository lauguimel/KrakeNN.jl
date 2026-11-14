# Step C0 of .orchestrator/3d-kernel-plan.md: why does the 3D D3Q19 GNF step
# (fused_trt_gnf_libb_guo_field_step_3d!, src/kernels/li_bb_3d_v2.jl:414, called by
# _gnf3d_periodic_step! in src/ad/gnf3d_adjoint.jl) reach only ~0.28 of copy bandwidth?
#
# For each row (law x precision) on CUDABackend, at N (padded (N+2)^3 lattice):
#   (a) register count, local (spill) / shared / constant memory, max threads per block,
#       occupancy and the workgroup size KernelAbstractions picks, for the step kernel
#       (rebuilt through the same calls as CUDACore.CUDAKernels.launch_tuple) and for every
#       other kernel compiled since the precision's setup (ghost-wrap broadcasts included),
#       read from CUDACore._kernel_instances; SASS local loads/stores (LDL/STL) counted;
#   (b) 200 timed full steps (CUDA.@elapsed) -> MLUPS, plus 200 kernel-only calls;
#   (c) 3 full steps inside an NVTX push/pop range named after the row (default domain),
#       for `ncu --nvtx --nvtx-include "<row>/"`.
# Case construction copied from Kraken.jl-perf benchmarks/throughput/throughput.jl
# (tp_closure, bench_gnf3d, _tp_gnf3d_law!), so numbers compare with the gnf3d rows.
#
# Every result line starts with "PF " and is a list of key=value fields (no spaces in values).
#
# Settings (environment):
#   KRAKEN_PF_N          lattice size N (default 192)
#   KRAKEN_PF_ROWS       rows, comma separated, from cy_f64, net_f64, cy_f32, net_f32, netc_f32
#                        (cy = carreau_yasuda, net = densenet, netc = densenet_f64consts)
#                        default cy_f64,net_f64,cy_f32,net_f32
#   KRAKEN_PF_MODE       full (default) | ncu (warm-up + NVTX ranges only; for the ncu run)
#   KRAKEN_PF_CASE       box (default, as throughput.jl: all-fluid periodic box)
#                        | channel (solid planes y=1 and y=N, periodic x and z)
#   KRAKEN_PF_STEPS      timed steps (default 200)
#   KRAKEN_PF_WARMUP     warm-up steps (default 5)
#   KRAKEN_PF_NVTX_STEPS steps inside the NVTX range (default 3)
#   KRAKEN_PF_SETUP_N    N for the SETUP_TIME line, Float64 (default 256; none to skip)
#   KRAKEN_PF_COPY_GBPS  measured copy bandwidth (A100 1383, H100 3029) for frac2q (optional)
#   KRAKEN_PF_OUT        folder for the SASS dumps (default ./results)
#   KRAKEN_PF_TAG        suffix of the SASS file names (default manual)
# Step C1/C6/N1 sweep (kind=sweep lines; kernel-only timing of every combination):
#   KRAKEN_PF_VARIANTS   collision variants, from reference, lean (default reference)
#   KRAKEN_PF_WG         workgroup sizes, entries separated by ';' or '/', each "a,b,c" or
#                        "axbxc" or auto (default auto = KernelAbstractions/CUDA choice);
#                        for qsub -v use the comma-free form, e.g. auto/256x1x1/128x2x1/64x4x1/32x4x2/32x8x1
#   KRAKEN_PF_MAXREGS    register caps, 0 = none (default 0). Passed as `maxregs` to
#                        fused_trt_gnf_libb_guo_field_step_3d! (src launch path,
#                        KrakeNN.lbm_launch! + ext/KrakeNNCUDAExt.jl). Production now caps
#                        by default (maxregs = :auto: 128 Float64, 96 Float32); every
#                        launch of this script passes its cap explicitly, 0 = uncapped,
#                        so time_step/time_kernel/equal lines stay uncapped as before.
#   KRAKEN_PF_EQ         1 (default): one step from the same state with each variant (and,
#                        for by-value rows, with the device-array law), compared with the
#                        reference (kind=equal lines: count of differing values, max |diff|)
# Lists (ROWS, VARIANTS, MAXREGS) accept ',' or '/' as separator (qsub -v splits on ',').
# Rows cyv_* / netv_* = the same law with its parameters held by value
# (KrakeNN.lbm_law_by_value, plan row N1).

using KrakeNN
using CUDA                       # top level only, never inside a function
using Printf, Random

const KA = KrakeNN.KernelAbstractions

_pf_env(key, default) = (v = strip(get(ENV, key, "")); isempty(v) ? String(default) : String(v))
_pf_list(s) = [String(strip(x)) for x in split(s, r"[,/]") if !isempty(strip(x))]
_pf_wg(x) = lowercase(x) == "auto" ? nothing :
    Tuple(parse(Int, strip(t)) for t in split(x, r"[,x]") if !isempty(strip(t)))
const WGS = [_pf_wg(String(strip(x))) for x in split(_pf_env("KRAKEN_PF_WG", "auto"), r"[;/]")
             if !isempty(strip(x))]

const N = parse(Int, _pf_env("KRAKEN_PF_N", "192"))
const ROWS = _pf_list(_pf_env("KRAKEN_PF_ROWS", "cy_f64,net_f64,cy_f32,net_f32"))
const MODE = _pf_env("KRAKEN_PF_MODE", "full")
const CASE = _pf_env("KRAKEN_PF_CASE", "box")
const STEPS = parse(Int, _pf_env("KRAKEN_PF_STEPS", "200"))
const WARMUP = parse(Int, _pf_env("KRAKEN_PF_WARMUP", "5"))
const NVTX_STEPS = parse(Int, _pf_env("KRAKEN_PF_NVTX_STEPS", "3"))
const SETUP_N = lowercase(_pf_env("KRAKEN_PF_SETUP_N", "256"))
const COPY_GBPS = parse(Float64, _pf_env("KRAKEN_PF_COPY_GBPS", "NaN"))
const OUT = _pf_env("KRAKEN_PF_OUT", joinpath(pwd(), "results"))
const TAG = _pf_env("KRAKEN_PF_TAG", "manual")
const VARIANTS = Symbol.(_pf_list(_pf_env("KRAKEN_PF_VARIANTS", "reference")))
const MAXREGS = parse.(Int, _pf_list(_pf_env("KRAKEN_PF_MAXREGS", "0")))
const CHECK_EQ = _pf_env("KRAKEN_PF_EQ", "1") == "1"

MODE in ("full", "ncu") || error("KRAKEN_PF_MODE must be full or ncu, got $MODE")
CASE in ("box", "channel") || error("KRAKEN_PF_CASE must be box or channel, got $CASE")
const ROW_DEFS = Dict("cy_f64" => (Float64, "carreau_yasuda"), "net_f64" => (Float64, "densenet"),
                      "cy_f32" => (Float32, "carreau_yasuda"), "net_f32" => (Float32, "densenet"),
                      "netc_f32" => (Float32, "densenet_f64consts"),
                      "cyv_f64" => (Float64, "carreau_yasuda"), "netv_f64" => (Float64, "densenet"),
                      "cyv_f32" => (Float32, "carreau_yasuda"), "netv_f32" => (Float32, "densenet"))
row_byvalue(row) = startswith(row, "cyv_") || startswith(row, "netv_")
all(v -> v in (:reference, :lean), VARIANTS) ||
    error("KRAKEN_PF_VARIANTS: reference or lean, got $VARIANTS")
for r in ROWS
    haskey(ROW_DEFS, r) || error("KRAKEN_PF_ROWS: unknown row $r (known: $(join(sort(collect(keys(ROW_DEFS))), ',')))")
end

CUDA.functional() || error("CUDA.functional() is false: refusing to run a GPU profile")
const BACKEND = CUDABackend()

# CUDA.jl 6 splits into CUDACore (compiler, launch) and CUDATools (profiler, which loads
# NVTX.jl). Both are reached through CUDA, so no dependency is added to any Project.toml.
const CC = isdefined(CUDA, :CUDACore) ? CUDA.CUDACore : CUDA
const NVTXM = if isdefined(CUDA, :CUDATools) && isdefined(CUDA.CUDATools, :NVTX)
    CUDA.CUDATools.NVTX
elseif isdefined(CUDA, :NVTX)
    CUDA.NVTX
else
    nothing
end

pf(fields...) = (println("PF ", join(fields, " ")); flush(stdout))

# Short error field for the PF line (no spaces, cut) plus, once per distinct message, the
# full error and backtrace on stderr, tagged with the same err_id.
const _PF_ERRS = Dict{String,Int}()
function pf_err(err, bt=nothing; n=200)
    msg = sprint(showerror, err)
    id = get!(_PF_ERRS, msg) do
        k = length(_PF_ERRS) + 1
        println(stderr, "PF_ERROR err_id=$k full message:")
        bt === nothing ? showerror(stderr, err) : showerror(stderr, err, bt)
        println(stderr)
        flush(stderr)
        k
    end
    return "err_id=$id error=$(replace(first(msg, n), r"\s" => ""))"
end
_v(x) = replace(string(x), ' ' => "")
_tup(t) = join(t, "x")

pf("info", "gpu=$(_v(CUDA.name(CUDA.device())))", "julia=$(VERSION)",
   "cuda_jl=$(pkgversion(CUDA))", "ka=$(pkgversion(KA))",
   "N=$N", "case=$CASE", "mode=$MODE", "rows=$(join(ROWS, ','))",
   "nvtx=$(NVTXM === nothing ? "missing" : "found")",
   "nvtx_active=$(NVTXM !== nothing && NVTXM.isactive())",
   "cc_module=$(nameof(CC))", "variants=$(join(VARIANTS, ','))",
   "wg=$(join([w === nothing ? "auto" : _tup(w) for w in WGS], ';'))",
   "maxregs=$(join(MAXREGS, ','))")

# ------------------------------------------------------------------ laws (throughput.jl tp_closure)

function pf_closure(::Type{T}, law::String, gamma_scale::Real) where {T}
    if law == "carreau_yasuda"
        lw = ClassicalLaw(CarreauYasuda(0.17, 0.0034, 20.0, 2.0, 0.7))
        c = LBMLawViscosity(lw, CuArray(law_init(lw)); eps=0.0)
    elseif law == "densenet" || law == "densenet_f64consts"
        S = law == "densenet" ? T : Float64
        net = DenseNet(1, (8, 8); activation=:tanh, output=:exp, shift=(zero(S),), scale=(S(5),))
        theta = law_init(net; rng=MersenneTwister(3)) .* 10
        theta[end] = log(0.1)
        c = LBMLawViscosity(net, CuArray(theta); eps=1e-3, nu_scale=1.0,
                            gamma_scale=Float64(gamma_scale))
    else
        error("unknown law $law")
    end
    return KrakeNN.gnf_convert_closure(T, c)
end
pf_closure(row::String) = (cl = pf_closure(ROW_DEFS[row][1], ROW_DEFS[row][2], N^2);
                           row_byvalue(row) ? KrakeNN.lbm_law_by_value(cl) : cl)

function pf_solid(n)
    s = fill(false, n, n, n)
    if CASE == "channel"
        s[:, 1, :] .= true
        s[:, n, :] .= true
    end
    return s
end

# Same arguments as bench_gnf3d: F = (1/N^3, 0, 0), nu_init 0.1, Λ 3/16, no f_init/tau_init.
pf_setup(::Type{T}, n) where {T} =
    KrakeNN._gnf3d_periodic_setup(pf_solid(n), pf_closure(T, "carreau_yasuda", n^2),
                                  (1.0 / n^3, 0.0, 0.0), 0.1, 3 / 16, nothing, nothing,
                                  BACKEND, T)

mutable struct PingPong{A}
    a::A
    b::A
end
function swap!(p::PingPong)
    a = p.a
    p.a = p.b
    p.b = a
    return nothing
end

# Full step (kernel + ghost wrap) with the FIRST variant and workgroup of the lists, no
# register cap. Every launch here passes variant, workgroup (nothing = KA automatic)
# and cap explicitly, so the production defaults (lean, CUDA F64 32x4x2, cap :auto)
# do not change what this script measures.
full_step!(s, p) = (KrakeNN._gnf3d_periodic_step!(s, p.b, p.a; variant=VARIANTS[1],
                                                  workgroupsize=WGS[1], maxregs=0); swap!(p))

# Kernel alone, as _gnf3d_periodic_step! calls it (throughput.jl gnf3d_kernel row).
# maxregs = 0 (default here): no register cap; > 0: that cap (CUDA launch of the ext).
function kernel_step!(s, p; variant::Symbol=:reference, workgroupsize=nothing, model=s.model,
                      maxregs::Int=0)
    nd = s.dims
    fused_trt_gnf_libb_guo_field_step_3d!(p.b, p.a, s.rho, s.ux, s.uy, s.uz, s.is_solid,
        s.q_wall, s.uw, s.uw, s.uw, s.Fx, s.Fy, s.Fz, s.sp, s.sm, nd[1], nd[2], nd[3], model;
        Λ=s.Λ, variant, workgroupsize, maxregs)
    swap!(p)
end

# Arguments of the step kernel, in the order fused_trt_gnf_libb_guo_field_step_3d! passes them.
function step_args(s, p, model)
    ET = eltype(p.a)
    nd = s.dims
    return (p.b, s.rho, s.ux, s.uy, s.uz, p.a, s.is_solid,
            s.q_wall, s.uw, s.uw, s.uw, nd[1], nd[2], nd[3],
            s.Fx, s.Fy, s.Fz, KrakeNN._gnf3d_rheology(ET, model), s.sp, s.sm, ET(s.Λ))
end

step_raw_kernel(variant) =
    KrakeNN.lbm_raw_kernel(BACKEND, KrakeNN._gnf3d_step_spec(false, variant))

# Compiled kernel and launch shape of one (variant, workgroup, register cap) combination,
# exactly as KrakeNN.lbm_launch! launches it: KrakeNN.lbm_cuda_launch_config
# (ext/KrakeNNCUDAExt.jl, the steps of CUDACore.CUDAKernels.launch_tuple plus the compiler
# option maxregs; maxregs = 0 = the plain KernelAbstractions launch).
# World age: the step kernel is defined by Core.eval inside build_lbm_kernel
# (src/kernels/dsl/lbm_builder.jl) while main() runs, so its method is newer than the
# world main() runs in; kernel_compile looks the method up in the current world and
# fails with a MethodError. Every direct compile/reflection of the kernel here goes
# through Base.invokelatest, as the src launch does.
step_config_latest(rk, args, ndr, wg, maxregs) =
    Base.invokelatest(KrakeNN.lbm_cuda_launch_config, rk, args, ndr, wg, maxregs)

# ------------------------------------------------------------------ introspection

kernel_label(k) = _v(first(string(k.f), 60))

function report_hostkernel(row, kind, k; threads_used=nothing, extra=String[])
    fun = k.fun
    regs = CC.registers(k)
    mem = CC.memory(k)
    mt = CC.maxthreads(k)
    cfg = CC.launch_configuration(fun)
    th = threads_used === nothing ? cfg.threads : threads_used
    occ = try
        @sprintf("%.3f", CC.occupancy(fun, th))
    catch err
        "err"
    end
    act = try
        string(CC.active_blocks(fun, th))
    catch err
        "err"
    end
    pf("row=$row", "kind=$kind", "name=$(kernel_label(k))", "regs=$regs",
       "local_B=$(getfield(mem, :local))", "shared_B=$(mem.shared)", "const_B=$(mem.constant)",
       "maxthreads=$mt", "suggest_threads=$(cfg.threads)", "suggest_blocks=$(cfg.blocks)",
       "threads=$th", "occupancy=$occ", "active_blocks_per_sm=$act", extra...)
end

_wgs(wg) = wg === nothing ? "auto" : _tup(wg)

# Registers, spills, occupancy and launch shape of the step kernel for one
# (variant, workgroup, register cap) combination, compiled as it is launched
# (step_config_latest).
# The raw kernel comes from KrakeNN.lbm_raw_kernel (the closure-field search of the C0
# version failed: "no KA.Kernel captured", results/pf_plain_h100_28089422.txt).
function introspect_step(row, s, p; variant::Symbol=VARIANTS[1], wg=WGS[1], maxregs::Int=0,
                         kind="step_kernel", model=s.model)
    try
        c = step_config_latest(step_raw_kernel(variant), step_args(s, p, model), s.dims, wg, maxregs)
        live = prod(c.ndrange) / (c.blocks * prod(c.workgroupsize))
        report_hostkernel(row, kind, c.kernel; threads_used=prod(c.workgroupsize),
                          extra=["variant=$variant", "wg_req=$(_wgs(wg))", "maxregs=$maxregs",
                                 "workgroup=$(_tup(c.workgroupsize))", "grid_blocks=$(c.blocks)",
                                 "ndrange=$(_tup(c.ndrange))", @sprintf("live_lane_frac=%.3f", live)])
        return c.kernel
    catch err
        pf("row=$row", "kind=$kind", "variant=$variant", "wg_req=$(_wgs(wg))", "maxregs=$maxregs",
           pf_err(err, catch_backtrace()))
        return nothing
    end
end

kernel_cache() = isdefined(CC, :_kernel_instances) ? getfield(CC, :_kernel_instances) : nothing
cache_keys() = (c = kernel_cache(); c === nothing ? Set{Any}() : Set{Any}(keys(c)))

# Ghost wrap = `selectdim(a, d, 1) .= selectdim(a, d, n-1)` (src/drivers/gnf_3d.jl:46):
# a broadcast kernel writing a SubArray with an identity Broadcasted. Heuristic, printed as such.
function kernel_kind(k)
    nm = lowercase(string(k.f))
    tt = string(typeof(k).parameters[2])
    occursin("lbm_gen_kernel", nm) && return "step_kernel"
    occursin("broadcast", nm) && occursin("SubArray", tt) && occursin("identity", tt) &&
        return "wrap_broadcast?"
    occursin("broadcast", nm) && return "other_broadcast"
    return "other"
end

function count_sass(row, idx, k)
    MODE == "full" || return String[]
    try
        tt = typeof(k).parameters[2]
        path = joinpath(OUT, "sass_$(row)_$(idx)_$TAG.txt")
        open(path, "w") do io
            Base.invokelatest(CUDA.code_sass, io, k.f, tt)   # world age, see step_config_latest
        end
        txt = read(path, String)
        ldl = count(r"\bLDL\b", txt)
        stl = count(r"\bSTL\b", txt)
        ldg = count(r"\bLDG\b", txt)
        stg = count(r"\bSTG\b", txt)
        return ["sass_LDL=$ldl", "sass_STL=$stl", "sass_LDG=$ldg", "sass_STG=$stg",
                "sass_file=$(basename(path))"]
    catch err
        return ["sass=error", pf_err(err, catch_backtrace(); n=120)]
    end
end

function introspect_new(row, before)
    c = kernel_cache()
    if c === nothing
        pf("row=$row", "kind=cache", "error=no_CUDACore._kernel_instances")
        return
    end
    idx = 0
    for (key, k) in collect(c)
        key in before && continue
        idx += 1
        try
            report_hostkernel(row, kernel_kind(k), k; extra=vcat(["cache_idx=$idx"], count_sass(row, idx, k)))
        catch err
            pf("row=$row", "kind=cache", "cache_idx=$idx", pf_err(err, catch_backtrace(); n=160))
        end
    end
end

# ------------------------------------------------------------------ timing and NVTX

function time_steps(f!, n)
    CUDA.synchronize()
    t = CUDA.@elapsed begin
        for _ in 1:n
            f!()
        end
    end
    return Float64(t)
end

function nvtx_range(row, f!, n)
    CUDA.synchronize()
    NVTXM === nothing || NVTXM.range_push(; message=row)
    try
        for _ in 1:n
            f!()
        end
        CUDA.synchronize()
    finally
        NVTXM === nothing || NVTXM.range_pop()
    end
    pf("row=$row", "kind=nvtx", "range=$row/", "steps=$n",
       "launches_expected=$(n * 19)", "active=$(NVTXM !== nothing && NVTXM.isactive())")
end

finite_ok(s, p) = all(isfinite, p.a) && all(isfinite, s.sp) && all(isfinite, s.ux)

# One kernel step from the same state (populations, rates) per configuration; outputs
# compared with the first one (reference variant; device-array law for by-value rows).
function equality_check(row, s, p)
    T, law = ROW_DEFS[row]
    configs = Any[("reference", :reference, s.model)]
    for v in VARIANTS
        v === :reference || push!(configs, (string(v), v, s.model))
    end
    if row_byvalue(row)
        arr = KrakeNN._gnf3d_rheology(T, pf_closure(T, law, N^2))
        configs[1] = ("reference_array_theta", :reference, arr)
        push!(configs, ("reference_by_value", :reference, s.model))
        for v in VARIANTS
            v === :reference || push!(configs, ("$(v)_by_value", v, s.model))
        end
    end
    sp0, sm0 = copy(s.sp), copy(s.sm)
    outs = Any[]
    for (label, v, model) in configs
        copyto!(s.sp, sp0)
        copyto!(s.sm, sm0)
        kernel_step!(s, p; variant=v, model)
        swap!(p)                     # p.a stays the common input state
        push!(outs, (label, map(copy, (p.b, s.sp, s.sm, s.rho, s.ux, s.uy, s.uz))))
    end
    CUDA.synchronize()
    names = ("f", "sp", "sm", "rho", "ux", "uy", "uz")
    base_label, ref = outs[1]
    for (label, o) in outs[2:end], (nm, a, b) in zip(names, ref, o)
        ndiff = count(.!isequal.(a, b))
        md = Float64(maximum(abs.(a .- b)))
        pf("row=$row", "kind=equal", "base=$base_label", "cmp=$label", "array=$nm",
           "n_diff=$ndiff", @sprintf("max_abs_diff=%.3e", md),
           @sprintf("max_abs_base=%.3e", Float64(maximum(abs, a))))
    end
    copyto!(s.sp, sp0)
    copyto!(s.sm, sm0)
    return nothing
end

# Kernel-only timing of every (variant, workgroup, register cap) combination.
function sweep(row, s, p, T, law)
    cells = (N + 2)^3
    b2q = 38 * sizeof(T)
    for v in VARIANTS, wg in WGS, mr in MAXREGS
        hk = introspect_step(row, s, p; variant=v, wg, maxregs=mr, kind="sweep_kernel")
        hk === nothing && continue
        f! = () -> kernel_step!(s, p; variant=v, workgroupsize=wg, maxregs=mr)
        t = try
            f!()                     # first launch outside the timing
            time_steps(f!, STEPS)
        catch err
            pf("row=$row", "kind=sweep", "variant=$v", "wg=$(_wgs(wg))", "maxregs=$mr",
               pf_err(err, catch_backtrace()))
            continue
        end
        ml = cells * STEPS / t / 1e6
        f2q = isnan(COPY_GBPS) ? "NaN" : @sprintf("%.3f", ml * b2q / 1e3 / COPY_GBPS)
        pf("row=$row", "kind=sweep", "T=$T", "law=$law", "variant=$v", "wg=$(_wgs(wg))",
           "maxregs=$mr", "regs=$(CC.registers(hk))", "local_B=$(getfield(CC.memory(hk), :local))",
           "N=$N", "steps=$STEPS", @sprintf("seconds=%.4f", t), @sprintf("MLUPS=%.1f", ml),
           "frac2q=$f2q", "finite=$(finite_ok(s, p))")
    end
end

function run_row(row, s0, p, before_setup)
    T, law = ROW_DEFS[row]
    cl = pf_closure(row)
    s = merge(s0, (; model=KrakeNN._gnf3d_rheology(T, cl)))
    for _ in 1:WARMUP
        full_step!(s, p)
    end
    MODE == "full" && CHECK_EQ && equality_check(row, s, p)
    CUDA.synchronize()
    pf("row=$row", "kind=warmup", "T=$T", "law=$law", "steps=$WARMUP", "finite=$(finite_ok(s, p))")

    # (a) introspection
    introspect_step(row, s, p)
    MODE == "full" && introspect_new(row, before_setup)

    # (b) timing
    if MODE == "full"
        cells = (N + 2)^3
        useful = N^3
        t_full = time_steps(() -> full_step!(s, p), STEPS)
        t_kern = time_steps(() -> kernel_step!(s, p; variant=VARIANTS[1], workgroupsize=WGS[1]), STEPS)
        mlups(t, c) = c * STEPS / t / 1e6
        b2q = 38 * sizeof(T)
        f2q(t) = isnan(COPY_GBPS) ? "NaN" : @sprintf("%.3f", mlups(t, cells) * b2q / 1e3 / COPY_GBPS)
        for (kind, t) in (("time_step", t_full), ("time_kernel", t_kern))
            # Both use the FIRST variant and workgroup of the lists, never a register cap.
            pf("row=$row", "kind=$kind", "T=$T", "law=$law", "variant=$(VARIANTS[1])",
               "wg=$(_wgs(WGS[1]))", "maxregs=0", "N=$N", "steps=$STEPS",
               @sprintf("seconds=%.4f", t), @sprintf("ns_per_cell=%.4f", t / STEPS / cells * 1e9),
               @sprintf("MLUPS=%.1f", mlups(t, cells)), @sprintf("MLUPS_useful=%.1f", mlups(t, useful)),
               "bytes_2q=$b2q", "frac2q=$(f2q(t))")
        end
        pf("row=$row", "kind=wrap_share", @sprintf("frac=%.3f", 1 - t_kern / t_full),
           "finite=$(finite_ok(s, p))")
        sweep(row, s, p, T, law)
    end

    # (c) NVTX range (ncu --nvtx-include "<row>/")
    nvtx_range(row, () -> full_step!(s, p), NVTX_STEPS)
    return nothing
end

function main()
    mkpath(OUT)
    for T in (Float64, Float32)
        rows = [r for r in ROWS if ROW_DEFS[r][1] === T]
        isempty(rows) && continue
        before = cache_keys()
        t0 = time()
        s0 = pf_setup(T, N)
        CUDA.synchronize()
        pf("setup", "T=$T", "N=$N", @sprintf("seconds=%.2f", time() - t0))
        p = PingPong(s0.f_in, s0.f_out)
        for row in rows
            run_row(row, s0, p, before)
            before = cache_keys()   # next rows: only kernels compiled for them
        end
        s0 = nothing
        p = nothing
        GC.gc(true)
        CUDA.reclaim()
    end

    # Setup time of the backend-built _gnf3d_periodic_setup (commit 6d8d92a51), Float64.
    if SETUP_N != "none"
        n = parse(Int, SETUP_N)
        ts = Float64[]
        for _ in 1:2
            t0 = time()
            s = pf_setup(Float64, n)
            CUDA.synchronize()
            push!(ts, time() - t0)
            s = nothing
            GC.gc(true)
            CUDA.reclaim()
        end
        pf("SETUP_TIME", "N=$n", "T=Float64", "case=$CASE", @sprintf("first_s=%.2f", ts[1]),
           @sprintf("second_s=%.2f", ts[2]))
    end
    pf("done")
end

main()
