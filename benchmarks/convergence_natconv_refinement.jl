# Natural convection cavity: showcase refinement cost vs accuracy
#
# Reference: De Vahl Davis (1983), Ra=1e3 → Nu=1.118.
# Nu is computed from the fine temperature patch at the hot wall, so this
# benchmark bypasses the Filippova–Hänel force-rescaling issue that
# currently blocks cylinder Cd on refined grids.
#
# Shows:
#   - Uniform grid: Nu converges ~O(1) with N (half-way BB thermal wall).
#   - Wall-refined grid: a thin patch (wall_fraction·L × L) at each
#     thermal wall at ratio r=2 gives the same accuracy as a globally
#     uniform grid of ≈ N·r cells, at a fraction of the total cell·step
#     cost — because the bulk still runs on the coarse grid.
#
# Usage:
#   julia --project benchmarks/convergence_natconv_refinement.jl

using KrakeNN
using KernelAbstractions
using Printf
using Dates

const NU_REF = 1.118    # De Vahl Davis (1983), Ra=1e3

# Steps to reach steady state — convection time scale ≈ N² (α ≈ 0.07)
nsteps(N) = max(5000, round(Int, 8 * N^2))

"""
    run_uniform_natconv(N; Ra, Pr, backend, FT)
Returns (N, Nu, err_pct, cells, steps, walltime, cell_steps).
"""
function run_uniform_natconv(N::Int; Ra=1e3, Pr=0.71,
                             backend=KernelAbstractions.CPU(), FT=Float64)
    steps = nsteps(N)
    t0 = time()
    r = run_natural_convection_2d(; N=N, Ra=Ra, Pr=Pr, Rc=1.0,
                                    max_steps=steps, backend=backend, FT=FT)
    dt = time() - t0
    err = abs(r.Nu - NU_REF) / NU_REF * 100
    cells = N * N
    return (; N, Nu=r.Nu, err, cells, steps, walltime=dt,
              cell_steps=cells * steps)
end

"""
    run_refined_natconv(N_base; ratio, wall_fraction, Ra, Pr, backend, FT)

Refined cavity with **two symmetric patches**: one at the hot (west)
wall and one at the cold (east) wall. Each patch has physical width
`wall_fraction · L` and full height `L`, and is subdivided by `ratio`
in each direction. The bulk of the cavity runs on the coarse grid.

Effective near-wall resolution is `N·ratio`. Total cell count is
N² (coarse) + 2·(wall_fraction·N·ratio) · (N·ratio) fine cells.

Cost metric (`cell_steps`): coarse_cells · n_coarse_steps
                          + 2 · fine_cells · ratio · n_coarse_steps
                            (each coarse step triggers `ratio` fine sub-steps).
"""
function run_refined_natconv(N_base::Int; ratio=2, wall_fraction=0.2,
                             Ra=1e3, Pr=0.71,
                             backend=KernelAbstractions.CPU(), FT=Float64)
    steps = nsteps(N_base)
    t0 = time()
    r = run_natural_convection_refined_2d(; N=N_base, Ra=Ra, Pr=Pr, Rc=1.0,
                                            max_steps=steps,
                                            wall_fraction=wall_fraction,
                                            ratio=ratio,
                                            backend=backend, FT=FT)
    dt = time() - t0
    err = abs(r.Nu - NU_REF) / NU_REF * 100

    coarse_cells = N_base * N_base
    fine_nx = round(Int, wall_fraction * N_base) * ratio
    fine_ny = N_base * ratio
    fine_cells = fine_nx * fine_ny
    total_cells = coarse_cells + 2 * fine_cells    # two wall patches
    cost = (coarse_cells + 2 * ratio * fine_cells) * steps

    return (; N=N_base, ratio, wall_fraction, Nu=r.Nu, err,
              coarse_cells, fine_cells=2*fine_cells, cells=total_cells,
              steps, walltime=dt, cell_steps=cost)
end

# ---------------------------------------------------------------------
function main(; Ns_uniform=[32, 64, 96, 128],
                Ns_refined=[32, 48, 64],
                ratios=[2],
                wall_fractions=[0.15, 0.25],
                Ra=1e3, Pr=0.71,
                backend=KernelAbstractions.CPU(), FT=Float64,
                csv_path=nothing)
    println("\n" * "="^86)
    @printf("Natural-convection cavity — Ra=%.0e, Pr=%.2f, Nu_ref=%.3f (De Vahl Davis)\n",
            Ra, Pr, NU_REF)
    println("="^86)

    uni = []
    println("\n-- Uniform --")
    @printf("%6s %10s %10s %6s %10s %12s\n",
            "N", "cells", "Nu", "err%", "walltime", "cell·steps")
    for N in Ns_uniform
        r = run_uniform_natconv(N; Ra=Ra, Pr=Pr, backend=backend, FT=FT)
        push!(uni, r)
        @printf("%6d %10d %10.4f %5.2f%% %9.1fs %12.2e\n",
                r.N, r.cells, r.Nu, r.err, r.walltime, r.cell_steps)
    end

    ref = []
    println("\n-- Refined (two wall patches, ratio=r, wall_fraction=wf) --")
    @printf("%6s %4s %6s %10s %10s %10s %6s %10s %12s\n",
            "N_b", "r", "wf", "coarse", "fine", "Nu", "err%",
            "walltime", "cell·steps")
    for N in Ns_refined, r in ratios, wf in wall_fractions
        res = run_refined_natconv(N; ratio=r, wall_fraction=wf,
                                  Ra=Ra, Pr=Pr, backend=backend, FT=FT)
        push!(ref, res)
        @printf("%6d %4d %6.2f %10d %10d %10.4f %5.2f%% %9.1fs %12.2e\n",
                res.N, res.ratio, res.wall_fraction,
                res.coarse_cells, res.fine_cells,
                res.Nu, res.err, res.walltime, res.cell_steps)
    end

    # Cost-accuracy summary: for each target accuracy, pick cheapest variant.
    println("\n-- Cost to reach err < 5% --")
    all_runs = Any[(row=r, kind=:uniform) for r in uni]
    for r in ref
        push!(all_runs, (row=r, kind=:refined))
    end
    sorted = sort(all_runs, by = x -> x.row.cell_steps)
    for x in sorted
        if x.row.err < 5.0
            kind = x.kind == :uniform ? "uniform" :
                   @sprintf("refined r=%d wf=%.2f", x.row.ratio, x.row.wall_fraction)
            @printf("  N=%d %s  Nu=%.4f err=%.2f%% cost=%.2e\n",
                    x.row.N, kind, x.row.Nu, x.row.err, x.row.cell_steps)
        end
    end

    if csv_path !== nothing
        mkpath(dirname(csv_path))
        open(csv_path, "w") do io
            println(io, "mode,N_base,ratio,wall_fraction,coarse_cells,fine_cells,total_cells,steps,cell_steps,Nu,err_pct,walltime_s")
            for r in uni
                println(io, "uniform,$(r.N),1,0,$(r.cells),0,$(r.cells),$(r.steps),$(r.cell_steps),$(r.Nu),$(r.err),$(r.walltime)")
            end
            for r in ref
                println(io, "refined,$(r.N),$(r.ratio),$(r.wall_fraction),$(r.coarse_cells),$(r.fine_cells),$(r.cells),$(r.steps),$(r.cell_steps),$(r.Nu),$(r.err),$(r.walltime)")
            end
        end
        println("\nCSV: $csv_path")
    end
    return (uniform=uni, refined=ref)
end

if abspath(PROGRAM_FILE) == @__FILE__
    ts = Dates.format(now(), "yyyymmdd_HHMMSS")
    csv = joinpath(@__DIR__, "results", "convergence_natconv_refinement_$ts.csv")
    main(csv_path=csv)
end
