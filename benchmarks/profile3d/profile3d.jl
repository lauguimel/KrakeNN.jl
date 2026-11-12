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

using KrakeNN
using CUDA                       # top level only, never inside a function
using Printf, Random

const KA = KrakeNN.KernelAbstractions

_pf_env(key, default) = (v = strip(get(ENV, key, "")); isempty(v) ? String(default) : String(v))
_pf_list(s) = [String(strip(x)) for x in split(s, ',') if !isempty(strip(x))]

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

MODE in ("full", "ncu") || error("KRAKEN_PF_MODE must be full or ncu, got $MODE")
CASE in ("box", "channel") || error("KRAKEN_PF_CASE must be box or channel, got $CASE")
const ROW_DEFS = Dict("cy_f64" => (Float64, "carreau_yasuda"), "net_f64" => (Float64, "densenet"),
                      "cy_f32" => (Float32, "carreau_yasuda"), "net_f32" => (Float32, "densenet"),
                      "netc_f32" => (Float32, "densenet_f64consts"))
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
_v(x) = replace(string(x), ' ' => "")
_tup(t) = join(t, "x")

pf("info", "gpu=$(_v(CUDA.name(CUDA.device())))", "julia=$(VERSION)",
   "cuda_jl=$(pkgversion(CUDA))", "ka=$(pkgversion(KA))",
   "N=$N", "case=$CASE", "mode=$MODE", "rows=$(join(ROWS, ','))",
   "nvtx=$(NVTXM === nothing ? "missing" : "found")",
   "nvtx_active=$(NVTXM !== nothing && NVTXM.isactive())",
   "cc_module=$(nameof(CC))")

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

full_step!(s, p) = (KrakeNN._gnf3d_periodic_step!(s, p.b, p.a); swap!(p))

# Kernel alone, as _gnf3d_periodic_step! calls it (throughput.jl gnf3d_kernel row).
function kernel_step!(s, p)
    nd = s.dims
    fused_trt_gnf_libb_guo_field_step_3d!(p.b, p.a, s.rho, s.ux, s.uy, s.uz, s.is_solid,
        s.q_wall, s.uw, s.uw, s.uw, s.Fx, s.Fy, s.Fz, s.sp, s.sm, nd[1], nd[2], nd[3], s.model;
        Λ=s.Λ)
    swap!(p)
end

# ------------------------------------------------------------------ introspection

# The raw KernelAbstractions kernel is captured by the closure that build_lbm_kernel caches
# (src/kernels/dsl/lbm_builder.jl:104): find the captured field that is a KA.Kernel.
function raw_ka_kernel(wrapped)
    for n in fieldnames(typeof(wrapped))
        v = getfield(wrapped, n)
        v isa Core.Box && (v = v.contents)
        v isa KA.Kernel && return v
    end
    error("no KA.Kernel captured in $(typeof(wrapped))")
end

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

# Same steps as CUDACore.CUDAKernels.launch_tuple (CUDA.jl 6.4): launch_config, mkcontext,
# kernel_call, kernel_compile (cached, so this returns the HostKernel the launches use),
# launch_configuration(max_threads=prod(ndrange)), threads_to_workgroupsize.
function introspect_step(row, s, p)
    try
        ET = eltype(p.a)
        model = KrakeNN._gnf3d_rheology(ET, s.model)
        spec = KrakeNN._TRT_GNF_LIBB_GUO_FIELD_SPEC_3D
        rk = raw_ka_kernel(KrakeNN.build_lbm_kernel(BACKEND, spec))
        nd = s.dims
        args = (p.b, s.rho, s.ux, s.uy, s.uz, p.a, s.is_solid,
                s.q_wall, s.uw, s.uw, s.uw, nd[1], nd[2], nd[3],
                s.Fx, s.Fy, s.Fz, model, s.sp, s.sm, ET(s.Λ))
        ndrange = (nd[1], nd[2], nd[3])
        ndr, _, iterspace, _ = KA.launch_config(rk, ndrange, nothing)
        ctx = KA.mkcontext(rk, ndr, iterspace)
        call = CC.kernel_call(rk.f, (ctx, args...))
        bk = KA.backend(rk)
        hk = CC.kernel_compile(call; always_inline=bk.always_inline, fastmath=bk.fastmath,
                               maxthreads=nothing)
        cfg = CC.launch_configuration(hk.fun; max_threads=prod(ndr))
        threads = bk.prefer_blocks ? cld(prod(ndr), max(cld(prod(ndr), min(prod(ndr), cfg.threads)), cfg.blocks)) :
                                     cfg.threads
        wg = CC.CUDAKernels.threads_to_workgroupsize(threads, ndr)
        _, iters2 = KA.partition(rk, ndr, wg)
        blocks = length(KA.blocks(iters2))
        live = prod(ndr) / (blocks * prod(wg))
        report_hostkernel(row, "step_kernel", hk; threads_used=prod(wg),
                          extra=["workgroup=$(_tup(wg))", "grid_blocks=$blocks",
                                 "ndrange=$(_tup(ndr))", @sprintf("live_lane_frac=%.3f", live),
                                 "prefer_blocks=$(bk.prefer_blocks)"])
        return hk
    catch err
        pf("row=$row", "kind=step_kernel", "error=$(_v(first(sprint(showerror, err), 200)))")
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
            CUDA.code_sass(io, k.f, tt)
        end
        txt = read(path, String)
        ldl = count(r"\bLDL\b", txt)
        stl = count(r"\bSTL\b", txt)
        ldg = count(r"\bLDG\b", txt)
        stg = count(r"\bSTG\b", txt)
        return ["sass_LDL=$ldl", "sass_STL=$stl", "sass_LDG=$ldg", "sass_STG=$stg",
                "sass_file=$(basename(path))"]
    catch err
        return ["sass=error:$(_v(first(sprint(showerror, err), 120)))"]
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
            pf("row=$row", "kind=cache", "cache_idx=$idx", "error=$(_v(first(sprint(showerror, err), 160)))")
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

function run_row(row, s0, p, before_setup)
    T, law = ROW_DEFS[row]
    cl = pf_closure(T, law, N^2)
    s = merge(s0, (; model=KrakeNN._gnf3d_rheology(T, cl)))
    for _ in 1:WARMUP
        full_step!(s, p)
    end
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
        t_kern = time_steps(() -> kernel_step!(s, p), STEPS)
        mlups(t, c) = c * STEPS / t / 1e6
        b2q = 38 * sizeof(T)
        f2q(t) = isnan(COPY_GBPS) ? "NaN" : @sprintf("%.3f", mlups(t, cells) * b2q / 1e3 / COPY_GBPS)
        for (kind, t) in (("time_step", t_full), ("time_kernel", t_kern))
            pf("row=$row", "kind=$kind", "T=$T", "law=$law", "N=$N", "steps=$STEPS",
               @sprintf("seconds=%.4f", t), @sprintf("ns_per_cell=%.4f", t / STEPS / cells * 1e9),
               @sprintf("MLUPS=%.1f", mlups(t, cells)), @sprintf("MLUPS_useful=%.1f", mlups(t, useful)),
               "bytes_2q=$b2q", "frac2q=$(f2q(t))")
        end
        pf("row=$row", "kind=wrap_share", @sprintf("frac=%.3f", 1 - t_kern / t_full),
           "finite=$(finite_ok(s, p))")
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
