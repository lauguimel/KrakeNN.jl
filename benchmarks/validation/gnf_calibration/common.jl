# Shared helpers of `calibrate_carreau.jl` and `cost_bench.jl`
# (benchmarks/validation/gnf_calibration). Not part of the package API.
#
# Case: the periodic obstacle channel of `test/ad/gnf_adjoint_test.jl` and of
# `benchmarks/validation/gnf_adjoint/run_gnf_adjoint_gpu.jl`: domain 8 × 4 units,
# m lattice nodes per unit (grid 8m × 4m), periodic in x, half-way bounce-back
# walls in y, a disk of radius 1.5 centred on each wall at x = 4 (a half-disk on
# each wall), uniform body force along x, forward solved by
# `run_gnf_periodic_steady` (fused step, distance stop rule).
#
# Reduced units: length L = m nodes, kinematic viscosity ν0 (GC_NU0 in lattice
# units), time L²/ν0. Reduced force G* = G_lat m³/ν0², reduced velocity
# u* = u_lat m/ν0. Carreau–Yasuda law of the code (`CarreauYasuda(eta_0, eta_inf,
# lambda, a, n)`): ν = η∞ + (η0 − η∞)(1 + (λγ̇)^a)^((n−1)/a), with η0 = ν0 fixed,
# η∞* = η∞/η0 and k* = λ ν0/m² (λ in reduced time units).

using LinearAlgebra
using Printf
using Random
using KernelAbstractions
using KrakeNN

const GC_NU0 = 0.17
const GC_GSTAR_REF = 5.0                # G* of the adjoint check before its scale factor
const GC_GSCALES = (0.1, 0.2, 0.3)      # the S = 3 forcings (scale factors of G*), shared by both scripts
const GC_FWD_TOL = 1e-10                # forward stop: distance estimate (accuracy check, level 1)
const GC_ADJ_TOL = 1e-8                 # adjoint relative residual (accuracy check and calibration)
const GC_ZERO_SHEAR = 1e-12             # I1 threshold counted by `gnf_linearise`
const GC_REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

# ---------------------------------------------------------------- case ---

"""
    gc_case(m) -> NamedTuple

Obstacle channel with m nodes per unit length (grid 8m × 4m).
"""
function gc_case(m::Integer)
    m >= 2 || throw(ArgumentError("m must be ≥ 2, got $m"))
    Nx, Ny = 8m, 4m
    solid = [(((i - 0.5) / m - 4)^2 + ((j - 0.5) / m)^2 < 1.5^2) ||
             (((i - 0.5) / m - 4)^2 + ((j - 0.5) / m - 4)^2 < 1.5^2) for i in 1:Nx, j in 1:Ny]
    nodes = findall(vec(.!solid))
    return (; m=Int(m), Nx, Ny, solid, nodes, nfluid=length(nodes), nu0=GC_NU0,
            uscale=GC_NU0 / m)
end

"Lattice body force for a scale factor of the reference reduced force G* = 5."
gc_force(case, gscale) = gscale * GC_GSTAR_REF * case.nu0^2 / case.m^3

"θ = (ln η∞*, ln k*, n, a) → closure in lattice units (the adjoint check's map)."
gc_closure4(case) = θ -> CarreauYasuda(case.nu0, exp(θ[1]) * case.nu0,
                                       exp(θ[2]) * case.m^2 / case.nu0, θ[4], θ[3])

"θ = (ln η∞*, ln k*, n) with a fixed (Carreau for a = 2) → closure in lattice units."
gc_closure3(case; a=2.0) = θ -> CarreauYasuda(case.nu0, exp(θ[1]) * case.nu0,
                                              exp(θ[2]) * case.m^2 / case.nu0, a, θ[3])

"Grid string `NXxNY` (Nx = 2 Ny, Ny multiple of 4) → m."
function gc_parse_grid(s::AbstractString)
    parts = split(lowercase(strip(s)), 'x')
    length(parts) == 2 || error("grid must look like 32x16, got \"$s\"")
    Nx, Ny = parse(Int, parts[1]), parse(Int, parts[2])
    (Nx == 2Ny && Ny % 4 == 0 && Ny >= 8) ||
        error("grid must be (8m)x(4m) with m ≥ 2 (16x8, 32x16, 64x32, ...), got \"$s\"")
    return Ny ÷ 4
end

gc_max_steps(case) = max(200_000, 2000 * case.m^2)

"""
    gc_forward(case, closure, g; tol, f_init, tau_init, backend, T, max_steps)

Steady forward on the case (as in the adjoint check: fused step, checks every 20
steps, distance estimate fitted on 20 checks, stop at `dist_est < tol`).
"""
function gc_forward(case, closure, g; tol=GC_FWD_TOL, f_init=nothing, tau_init=nothing,
                    backend=KernelAbstractions.CPU(), T::Type=Float64,
                    max_steps=gc_max_steps(case))
    return run_gnf_periodic_steady(; is_solid=case.solid, closure=closure, Fx=g, backend=backend,
                                   T=T, tol=tol, check_every=20, fit_checks=20, fused=true,
                                   f_init=f_init, tau_init=tau_init, max_steps=max_steps)
end

"""
    gc_obs(case, ux, uy, uref) -> GNFVelocityObs

Both velocity components at every fluid node; targets ux/uref, uy/uref
(lattice velocities of the data), scale 1/uref, so that J = ½ Σ ((u − u_data)/uref)².
"""
function gc_obs(case, ux, uy, uref)
    M = 2 * case.nfluid
    node = zeros(Int, 1, M); w = ones(1, M); comp = zeros(Int, M); tgt = zeros(M)
    for (k, nd) in enumerate(case.nodes)
        node[1, 2k-1] = nd; comp[2k-1] = 1; tgt[2k-1] = ux[nd] / uref
        node[1, 2k] = nd; comp[2k] = 2; tgt[2k] = uy[nd] / uref
    end
    return GNFVelocityObs(node, w, comp, tgt, 1 / uref)
end

"Maximum fluid speed of a host velocity field (lattice units)."
gc_umax(case, ux, uy) = maximum(sqrt(ux[nd]^2 + uy[nd]^2) for nd in case.nodes)

"Number of fluid nodes on the zero-shear branch (I1 < 1e-12, as counted by `gnf_linearise`)."
function gc_zero_shear(case, I1)
    I1h = Array(I1)
    return count(nd -> I1h[nd] < GC_ZERO_SHEAR, case.nodes)
end

"""
    gc_gradient(state, θ, obs, closure_of, g, case; backend, T, solver, tol)

Adjoint gradient by `gnf_steady_gradient` at a converged forward `state`.
"""
function gc_gradient(state, θ, obs, closure_of, g, case; backend=KernelAbstractions.CPU(),
                     T::Type=Float64, solver::Symbol=:kstep, tol=GC_ADJ_TOL)
    kw = solver === :gmres ? (; restart=1000, maxiter=20_000) : (;)
    return gnf_steady_gradient(state.f, state.tau, θ, obs; is_solid=case.solid,
                               closure_of=closure_of, Fx=g, backend=backend, T=T,
                               solver=solver, tol=tol, kw...)
end

"Number of operator products of one adjoint solve (K-step: VJPs; GMRES: iterations)."
gc_nvjp(info) = hasproperty(info, :n_vjp) ? info.n_vjp :
                hasproperty(info, :n_apply) ? info.n_apply : info.iters

# ------------------------------------------------------- command line ---

"""
    gc_options(args, defaults) -> Dict

`--key=value` arguments over a dictionary of defaults (every key must exist).
"""
function gc_options(args, defaults::Dict{String,String})
    opts = copy(defaults)
    for arg in args
        parts = split(arg, '='; limit=2)
        (length(parts) == 2 && startswith(parts[1], "--")) || error("expected --key=value, got \"$arg\"")
        key = String(parts[1][3:end])
        haskey(opts, key) || error("unknown option --$key (known: $(join(sort(collect(keys(opts))), ", ")))")
        opts[key] = String(parts[2])
    end
    return opts
end

"Comma list of integers and ranges `a:b`, e.g. `1,3:5` → [1, 3, 4, 5]."
function gc_parse_ints(s::AbstractString)
    out = Int[]
    for part in split(s, ',')
        part = strip(part)
        isempty(part) && continue
        if occursin(':', part)
            a, b = parse.(Int, split(part, ':'))
            append!(out, a:b)
        else
            push!(out, parse(Int, part))
        end
    end
    return out
end

gc_parse_list(s::AbstractString) = [String(strip(x)) for x in split(s, ',') if !isempty(strip(x))]

function gc_precision(s::AbstractString)
    s == "Float64" && return Float64
    s == "Float32" && return Float32
    error("--precision must be Float64 or Float32, got \"$s\"")
end

function gc_solver(s::AbstractString)
    s in ("kstep", "gmres") || error("solver must be kstep or gmres, got \"$s\"")
    return Symbol(s)
end

"Backend named by the last `--backend=` argument, else `default`, is `cuda`."
function gc_wants_cuda(args, default::AbstractString)
    name = default
    for arg in args
        startswith(arg, "--backend=") && (name = arg[length("--backend=")+1:end])
    end
    return name == "cuda"
end

"""
    gc_backend(name) -> KernelAbstractions backend

`cpu` or `cuda`. For `cuda`, CUDA must have been loaded at top level before the
call (the scripts do it before `main` runs) and `CUDA.functional()` must be true.
"""
function gc_backend(name::AbstractString)
    name == "cpu" && return KernelAbstractions.CPU()
    name == "cuda" || error("--backend must be cpu or cuda, got \"$name\"")
    isdefined(@__MODULE__, :CUDA) ||
        error("CUDA is not loaded: run the script as a program with --backend=cuda")
    CUDA.functional() || error("CUDA.functional() is false: no usable NVIDIA GPU and driver here")
    CUDA.allowscalar(false)
    return CUDA.CUDABackend()
end

# ---------------------------------------------------------- provenance ---

function gc_git(args...)
    try
        out = chomp(read(pipeline(`git -C $GC_REPO_ROOT $args`; stderr=devnull), String))
        return isempty(out) && args[1] != "status" ? nothing : out
    catch
        return nothing
    end
end

function gc_pkg_version(name::AbstractString)
    for (id, mod) in Base.loaded_modules
        id.name == name || continue
        v = pkgversion(mod)
        return v === nothing ? "unknown" : string(v)
    end
    return "not loaded"
end

"""
    gc_provenance(; allow_unknown_commit=false) -> NamedTuple

Commit of the code that runs: `git rev-parse HEAD` in the repository holding the
script; if that fails (a copy without `.git`), the `KRAKEN_GIT_COMMIT`
environment variable; otherwise an error, unless `allow_unknown_commit`.
Also the number and names of modified tracked files (`git status --porcelain`).
"""
function gc_provenance(; allow_unknown_commit::Bool=false)
    commit = gc_git("rev-parse", "HEAD")
    source = "git rev-parse HEAD"
    if commit === nothing
        env = strip(get(ENV, "KRAKEN_GIT_COMMIT", ""))
        if !isempty(env)
            commit = String(env); source = "KRAKEN_GIT_COMMIT environment variable"
        elseif allow_unknown_commit
            commit = "unknown"; source = "none (--allow-unknown-commit=yes)"
        else
            error("cannot determine the git commit of the code: run from a git checkout, " *
                  "or set KRAKEN_GIT_COMMIT to the commit the code was copied from, " *
                  "or pass --allow-unknown-commit=yes")
        end
    end
    status = gc_git("status", "--porcelain", "--untracked-files=no")
    dirty = status === nothing ? String[] : [String(l) for l in split(status, '\n') if !isempty(l)]
    return (; commit, source, branch=something(gc_git("rev-parse", "--abbrev-ref", "HEAD"), "unknown"),
            dirty_known=status !== nothing, dirty)
end

"Print the provenance header (stdout, i.e. the log)."
function gc_print_provenance(prov, script; backend_name, T, extra=Pair{String,Any}[])
    println("Provenance: script=", script)
    println("git_commit=", prov.commit, " (source: ", prov.source, ") branch=", prov.branch)
    if prov.dirty_known
        println("git_modified_tracked_files=", length(prov.dirty))
        for l in first(prov.dirty, 20)
            println("  ", l)
        end
        length(prov.dirty) > 20 && println("  ... (", length(prov.dirty) - 20, " more)")
    else
        println("git_modified_tracked_files=unknown")
    end
    println("Julia=", VERSION, " os=", Sys.KERNEL, " arch=", Sys.ARCH, " cpu=", Sys.CPU_NAME,
            " Julia_threads=", Threads.nthreads(), " BLAS_threads=", BLAS.get_num_threads())
    println("KrakeNN=", gc_pkg_version("KrakeNN"), " KernelAbstractions=", gc_pkg_version("KernelAbstractions"),
            " CUDA.jl=", gc_pkg_version("CUDA"), " ForwardDiff=", gc_pkg_version("ForwardDiff"))
    println("backend=", backend_name, " precision=", T)
    if backend_name == "cuda"
        println("device=", CUDA.name(CUDA.device()), " CUDA_runtime=", CUDA.runtime_version(),
                " CUDA_driver=", CUDA.driver_version())
    else
        println("device=CPU CUDA_runtime=not applicable")
    end
    for (k, v) in extra
        println(k, "=", v)
    end
    println("start_local=", Libc.strftime("%Y-%m-%dT%H:%M:%S%z", time()))
    return nothing
end

gc_stamp() = Libc.strftime("%Y%m%d_%H%M%S", time())

# ------------------------------------------------------------- output ---

"""
    gc_transcript(f, path)

Run `f()` with stdout and stderr copied to `path` and to the console.
"""
function gc_transcript(f, path)
    console = stdout
    open(path, "w") do log
        pipe = Pipe()
        Base.link_pipe!(pipe; reader_supports_async=true, writer_supports_async=true)
        copier = @async while !eof(pipe)
            bytes = readavailable(pipe)
            write(log, bytes); flush(log)
            write(console, bytes); flush(console)
        end
        try
            redirect_stdout(pipe) do
                redirect_stderr(pipe) do
                    f()
                end
            end
        finally
            close(pipe.in)
            wait(copier)
            close(pipe.out)
        end
    end
end

gc_csvfield(x::AbstractString) = '"' * replace(x, '"' => "\"\"") * '"'
gc_csvfield(x::Symbol) = gc_csvfield(string(x))
gc_csvfield(x::Bool) = string(x)
gc_csvfield(x::Integer) = string(x)
gc_csvfield(x::AbstractFloat) = string(Float64(x))
gc_csvfield(::Nothing) = ""
gc_csvfield(x) = gc_csvfield(string(x))

function gc_csvrow(io, fields)
    println(io, join((gc_csvfield(x) for x in fields), ','))
    flush(io)
    return nothing
end

"Path relative to the working directory (keeps absolute home paths out of logs)."
gc_relpath(p) = relpath(abspath(p), pwd())

"Wall time of `f()` with the backend synchronised before and after."
function gc_timed(f, backend)
    KernelAbstractions.synchronize(backend)
    t0 = time()
    out = f()
    KernelAbstractions.synchronize(backend)
    return out, time() - t0
end

function gc_median(x)
    isempty(x) && return NaN
    s = sort(collect(Float64, x)); n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end
