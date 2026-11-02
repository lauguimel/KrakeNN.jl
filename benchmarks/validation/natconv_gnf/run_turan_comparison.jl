using Kraken
using KernelAbstractions: CPU

# Turan et al. (2011), doi:10.1016/j.jnnfm.2011.06.003; figure readings.
const TURAN_CASES = (
    (; id="RA3_N06", n=0.6, Ra=1e3, Pr=10.0, Nu_paper=2.07,
       source="Fig. 9a p.1059 (read +-0.07)", nu0=0.1),
    (; id="RA4_N06", n=0.6, Ra=1e4, Pr=10.0, Nu_paper=5.40,
       source="Fig. 10 p.1060 (read +-0.02)", nu0=0.1),
    (; id="RA4_N08", n=0.8, Ra=1e4, Pr=10.0, Nu_paper=3.38,
       source="Fig. 10 p.1060 (read +-0.02)", nu0=0.1),
    (; id="RA4_N10", n=1.0, Ra=1e4, Pr=10.0, Nu_paper=2.29,
       source="Figs. 10-11 pp.1060-1061 (read +-0.015)", nu0=0.1),
    (; id="RA4_N18", n=1.8, Ra=1e4, Pr=10.0, Nu_paper=1.10,
       source="Fig. 11 p.1061 (read +-0.01)", nu0=0.02),
)

function parse_turan_options(args)
    smoke = false
    grids = [64, 128, 256]
    output_dir = joinpath(@__DIR__, "results")
    for arg in args
        if arg == "--smoke"
            smoke = true
        elseif startswith(arg, "--output-dir=")
            output_dir = split(arg, '='; limit=2)[2]
            isempty(output_dir) && throw(ArgumentError("--output-dir needs a path"))
        elseif startswith(arg, "--grids=")
            grids = parse.(Int, split(split(arg, '='; limit=2)[2], ','))
            isempty(grids) && throw(ArgumentError("--grids needs at least one grid"))
            all(N -> N >= 3, grids) || throw(ArgumentError("all grids must be >= 3"))
        else
            throw(ArgumentError("unknown option: $arg"))
        end
    end
    if smoke
        grids = [16]
        output_dir = normpath(joinpath(@__DIR__, "..", "..", "..", "tmp", "m1c", "turan_smoke"))
    end
    return (; smoke, grids=sort!(unique(grids)), output_dir=abspath(output_dir))
end

function turan_vmax_mid(uy, N, alpha)
    # j is vertical: average the two central horizontal planes for even N.
    j_lo, j_hi = (N + 1) ÷ 2, N ÷ 2 + 1
    return maximum(i -> abs((uy[i, j_lo] + uy[i, j_hi]) / 2), axes(uy, 1)) * N / alpha
end

function run_turan_comparison(options; backend, device)
    mkpath(options.output_dir)
    csv_path = joinpath(options.output_dir, "turan_comparison.csv")
    log_path = joinpath(options.output_dir, "turan_comparison.log")
    hardware = "device=$device, Julia=$VERSION, precision=Float64"
    failed = false
    open(csv_path, "w") do csv
        open(log_path, "w") do log
            println(log, hardware)
            flush(log)
            println(csv, "case,n,Ra,Pr,N,Nu_kraken,Nu_paper,rel_err,steps,converged," *
                         "nu0,clamped_fraction,Vmax_mid,nu_field_min,nu_field_max,wall_s,Nu_paper_source")
            flush(csv)
            @info "Turan comparison hardware" hardware
            for N in options.grids, case in TURAN_CASES
                (; n, Ra, Pr, nu0) = case
                alpha = nu0 / Pr
                t_diff = N^2 / alpha
                check_every = ceil(Int, 0.01 * t_diff)
                min_steps = ceil(Int, 0.1 * t_diff)
                max_steps = ceil(Int, 1.5 * t_diff)
                options.smoke && (max_steps = min(max_steps, 200))
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
                       result.steps, result.converged, nu0, result.clamped_fraction,
                       Vmax_mid, nu_field_min, nu_field_max, wall_s, source)
                println(csv, join(row, ','))
                flush(csv)
                println(log, join(row, ','))
                flush(log)
                @info "Turan comparison row" case=case.id N Nu=result.Nu steps=result.steps converged=result.converged
                failed |= !isfinite(result.Nu) || (!options.smoke && !result.converged)
            end
        end
    end
    return failed ? 1 : 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    options = parse_turan_options(ARGS)
    if !options.smoke
        using CUDA
        CUDA.functional() || error("A functional CUDA device is required")
    end
    backend = options.smoke ? CPU() : CUDA.CUDABackend()
    device = options.smoke ? "CPU smoke" : CUDA.name(CUDA.device())
    exit(run_turan_comparison(options; backend, device))
end
