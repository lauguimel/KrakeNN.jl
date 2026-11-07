# Grid-convergence comparison of run_natural_convection_gnf_2d with Turan et al.
# (2011), doi:10.1016/j.jnnfm.2011.06.003. See README.md beside this file.
#
#   julia --project benchmarks/validation/natconv_gnf/run_turan_comparison.jl            # CUDA, N = 64,128,256
#   julia --project ... run_turan_comparison.jl --backend=cpu --grids=64 --cases=RA3_N06  # CPU check
#   julia --project ... run_turan_comparison.jl --smoke                                  # CPU, N = 16, 200 steps
using KrakeNN
using KernelAbstractions: CPU

# Paper Nu values are figure readings (journal pages, reading uncertainty).
# nu0 only sets the lattice time step; n = 1.8 needs a small nu0 because its
# viscosity is largest where the shear is largest (see README.md). Measured on
# CPU at N = 64, Ra = 1e4: nu0 = 0.3 and 0.1 put 84 % and 22 % of the nodes at
# nu_max; nu0 = 0.05 puts 0.1 % there.
const NU0_N18 = 0.05
const TURAN_CASES = (
    (; id="RA3_N06", n=0.6, Ra=1e3, Pr=10.0, Nu_paper=2.07,
       source="Fig. 9a p.1059 (read +-0.07)", nu0=0.3),
    (; id="RA3_N10", n=1.0, Ra=1e3, Pr=10.0, Nu_paper=1.14,
       source="Fig. 9b p.1059 (read +-0.02)", nu0=0.3),
    (; id="RA3_N18", n=1.8, Ra=1e3, Pr=10.0, Nu_paper=1.01,
       source="Fig. 9c p.1059 (read +-0.005)", nu0=NU0_N18),
    (; id="RA4_N06", n=0.6, Ra=1e4, Pr=10.0, Nu_paper=5.40,
       source="Fig. 10 p.1060 (read +-0.02)", nu0=0.3),
    (; id="RA4_N10", n=1.0, Ra=1e4, Pr=10.0, Nu_paper=2.29,
       source="Figs. 10-11 pp.1060-1061 (read +-0.015)", nu0=0.3),
    (; id="RA4_N18", n=1.8, Ra=1e4, Pr=10.0, Nu_paper=1.10,
       source="Fig. 11 p.1061 (read +-0.01)", nu0=NU0_N18),
)

function parse_turan_options(args)
    smoke = false
    backend = "cuda"
    grids = [64, 128, 256]
    cases = [case.id for case in TURAN_CASES]
    output_dir = joinpath(@__DIR__, "results")
    for arg in args
        if arg == "--smoke"
            smoke = true
        elseif startswith(arg, "--backend=")
            backend = split(arg, '='; limit=2)[2]
            backend in ("cuda", "cpu") || throw(ArgumentError("--backend must be cuda or cpu"))
        elseif startswith(arg, "--output-dir=")
            output_dir = split(arg, '='; limit=2)[2]
            isempty(output_dir) && throw(ArgumentError("--output-dir needs a path"))
        elseif startswith(arg, "--grids=")
            grids = parse.(Int, split(split(arg, '='; limit=2)[2], ','))
            isempty(grids) && throw(ArgumentError("--grids needs at least one grid"))
            all(N -> N >= 3, grids) || throw(ArgumentError("all grids must be >= 3"))
        elseif startswith(arg, "--cases=")
            cases = String.(split(split(arg, '='; limit=2)[2], ','))
            known = [case.id for case in TURAN_CASES]
            all(in(known), cases) || throw(ArgumentError("--cases must be among $(join(known, ','))"))
        else
            throw(ArgumentError("unknown option: $arg"))
        end
    end
    if smoke
        backend = "cpu"
        grids = [16]
        output_dir = normpath(joinpath(@__DIR__, "..", "..", "..", "tmp", "m1c", "turan_smoke"))
    end
    return (; smoke, backend, grids=sort!(unique(grids)), cases, output_dir=abspath(output_dir))
end

function turan_vmax_mid(uy, N, alpha)
    # j is vertical: average the two central horizontal planes for even N.
    j_lo, j_hi = (N + 1) ÷ 2, N ÷ 2 + 1
    return maximum(i -> abs((uy[i, j_lo] + uy[i, j_hi]) / 2), axes(uy, 1)) * N / alpha
end

function source_revision()
    try
        return readchomp(`git -C $(@__DIR__) rev-parse --short HEAD`)
    catch
        return "unknown"
    end
end

function run_turan_comparison(options; backend, device)
    mkpath(options.output_dir)
    csv_path = joinpath(options.output_dir, "turan_comparison.csv")
    log_path = joinpath(options.output_dir, "turan_comparison.log")
    hardware = "device=$device, Julia=$VERSION, precision=Float64, " *
               "source=$(source_revision()), thermal walls half-way (H = N)"
    selected = filter(case -> case.id in options.cases, collect(TURAN_CASES))
    failed = false
    open(csv_path, "w") do csv
        open(log_path, "w") do log
            println(log, hardware)
            flush(log)
            println(csv, "case,n,Ra,Pr,N,Nu_kraken,Nu_paper,rel_err,steps,converged,Nu_drift," *
                         "nu0,clamped_fraction,Vmax_mid,nu_field_min,nu_field_max,wall_s,Nu_paper_source")
            flush(csv)
            @info "Turan comparison hardware" hardware
            for N in options.grids, case in selected
                (; n, Ra, Pr, nu0) = case
                alpha = nu0 / Pr
                t_diff = N^2 / alpha
                check_every = ceil(Int, 0.01 * t_diff)
                min_steps = ceil(Int, 0.1 * t_diff)
                max_steps = ceil(Int, t_diff)
                if options.smoke
                    max_steps = min(max_steps, 200)
                    check_every = min(check_every, 100)
                end
                rheology = cavity_powerlaw(n; nu0, alpha, H=N, nu_min=1e-3, nu_max=2.0)
                started = time()
                result = run_natural_convection_gnf_2d(; N, Ra, Pr, nu0, rheology,
                    backend, FT=Float64, max_steps, check_every, min_steps,
                    convergence_rtol=1e-6)
                wall_s = time() - started
                rel_err = (result.Nu - case.Nu_paper) / case.Nu_paper
                Vmax_mid = turan_vmax_mid(result.uy, N, alpha)
                nu_field_min, nu_field_max = extrema(result.nu_field)
                source = '"' * replace(case.source, "\"" => "\"\"") * '"'
                row = (case.id, n, Ra, Pr, N, result.Nu, case.Nu_paper, rel_err,
                       result.steps, result.converged, result.Nu_drift, nu0,
                       result.clamped_fraction, Vmax_mid, nu_field_min, nu_field_max,
                       wall_s, source)
                println(csv, join(row, ','))
                flush(csv)
                println(log, join(row, ','))
                flush(log)
                @info "Turan comparison row" case=case.id N Nu=result.Nu steps=result.steps converged=result.converged clamped=result.clamped_fraction wall_s
                failed |= !isfinite(result.Nu) || (!options.smoke && !result.converged)
            end
        end
    end
    return failed ? 1 : 0
end

# Load CUDA at top level, in its own top-level block, before any function uses
# it: `using CUDA` inside a function hits a world-age error on first use.
if abspath(PROGRAM_FILE) == @__FILE__
    const TURAN_OPTIONS = parse_turan_options(ARGS)
    if TURAN_OPTIONS.backend == "cuda"
        using CUDA
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if TURAN_OPTIONS.backend == "cuda"
        CUDA.functional() || error("CUDA.functional() is false: a functional CUDA device is required")
        backend = CUDA.CUDABackend()
        device = CUDA.name(CUDA.device())
    else
        backend = CPU()
        device = "CPU ($(Sys.cpu_info()[1].model), $(Threads.nthreads()) threads)"
    end
    exit(run_turan_comparison(TURAN_OPTIONS; backend, device))
end
