# Cylinder Cd convergence: uniform grid vs patch refinement around the cylinder
#
# Reference: Schäfer & Turek (1996), 2D-1 steady benchmark
#   H = 0.41, D = 0.1, U_mean = 0.2, Re = U·D/ν = 20, Cd_ref = 5.57–5.59
#
# STATUS (v0.1.0):
#   - The uniform sweep works and shows order-1 convergence with cell count.
#   - The refined sweep via MEA on the *fine* patch requires Filippova–Hänel
#     force rescaling, which is an open research item (see Lagrava et al.
#     2012). MEA on the coarse grid after restriction is also inconsistent
#     because the coarse collision still treats cylinder cells as solid.
#   - Therefore the refined runs in this file are currently stability-only
#     (same as convergence_refinement.jl). The natural-convection refined
#     benchmark (Nu from fine temperature patch, `convergence_refinement.jl`
#     Part 1) is the recommended refinement showcase for v0.1.0.
#
# TODO v0.2.0:
#   - Implement FH-rescaled MEA on fine patches (Lagrava 2012), or
#   - Add pressure+viscous stress surface integration (resolution-agnostic).

using KrakeNN
using KernelAbstractions
using Printf
using Dates

const CD_REF = 5.58   # Schäfer & Turek (1996)

# ---------------------------------------------------------------------
# Uniform-grid convergence
# ---------------------------------------------------------------------
"""
    run_uniform(Ny; u_in, avg_window) → (Cd, err%, cells, walltime, cell_steps)
"""
function run_uniform(Ny::Int; u_in::Float64=0.05, avg_window_frac::Float64=0.2,
                     backend=KernelAbstractions.CPU(), FT::Type=Float64)
    Nx = 4*Ny
    R = round(Int, 0.125*Ny)      # D/H = 0.25 → D = Ny/4 → R = Ny/8
    D = 2R
    ν = u_in * D / 20.0           # Re = 20
    steps = max(10_000, round(Int, 12 * Ny^2))
    avg = max(500, round(Int, avg_window_frac * steps))

    t0 = time()
    r = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R, u_in=u_in, ν=ν,
                         max_steps=steps, avg_window=avg,
                         backend=backend, T=FT)
    dt = time() - t0

    err = abs(r.Cd - CD_REF) / CD_REF * 100
    cells = Nx * Ny
    cell_steps = cells * steps
    return (; N=Ny, Cd=r.Cd, err=err, cells, steps, walltime=dt, cell_steps)
end

# ---------------------------------------------------------------------
# Refined-grid convergence (patch around cylinder)
# ---------------------------------------------------------------------
"""
    run_refined(Ny; ratio, u_in, patch_radius_R) → same named tuple + ratio

A rectangular patch of size `patch_radius_R × R_phys` around the cylinder
(in base-grid coords) is refined by factor `ratio`. MEA drag is evaluated
on the fine patch; D_fine = D·ratio in the non-dimensionalisation.
"""
function run_refined(Ny::Int; ratio::Int=2, u_in::Float64=0.05,
                     patch_margin_R::Float64=3.0,
                     avg_window_frac::Float64=0.2,
                     backend=KernelAbstractions.CPU(), FT::Type=Float64)
    Nx = 4*Ny
    R = round(Int, 0.125*Ny)
    D = 2R
    ν = u_in * D / 20.0
    steps = max(10_000, round(Int, 12 * Ny^2))
    avg = max(500, round(Int, avg_window_frac * steps))
    cx = 0.2*Nx; cy = 0.5*Ny

    # Base grid: cylinder obstacle on coarse
    state, config = initialize_cylinder_2d(; Nx=Nx, Ny=Ny,
                                             cx=round(Int, cx), cy=round(Int, cy),
                                             radius=R, u_in=u_in, ν=ν,
                                             backend=backend, T=FT)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    ω = FT(KrakeNN.omega(config))

    # Refinement patch: axis-aligned rectangle around cylinder
    margin = patch_margin_R * Float64(R)
    xmin = max(0.0, cx - margin)
    ymin = max(0.0, cy - margin)
    xmax = min(Float64(Nx), cx + 2*margin)    # extend downstream for wake
    ymax = min(Float64(Ny), cy + margin)

    patch = KrakeNN.create_patch("cyl_patch", 1, ratio,
                                (xmin, ymin, xmax, ymax),
                                Nx, Ny, 1.0, Float64(ω), FT; backend=backend)
    domain = KrakeNN.create_refined_domain(Nx, Ny, 1.0, Float64(ω), [patch])

    # Rasterise cylinder on the fine patch.
    # Fine node if_ ∈ [1, Nx_p] with Nx_p = Nx_inner + 2·ng; interior is
    # if_ ∈ [ng+1, ng+Nx_inner]. A coarse cell at index i_c spans physical
    # x ∈ [i_c-1, i_c]; its `ratio` fine sub-cells have centers at
    #     x_f(if_) = (i_c_start - 1) + (if_ - ng - 0.5) · dx_f.
    patch_is_solid_cpu = zeros(Bool, patch.Nx, patch.Ny)
    ng = patch.n_ghost
    i0c = first(patch.parent_i_range); j0c = first(patch.parent_j_range)
    dx_f = patch.dx
    for jf in 1:patch.Ny, if_ in 1:patch.Nx
        xf = (i0c - 1) + (if_ - ng - 0.5) * dx_f
        yf = (j0c - 1) + (jf - ng - 0.5) * dx_f
        if (xf - cx)^2 + (yf - cy)^2 <= R^2
            patch_is_solid_cpu[if_, jf] = true
        end
    end
    copyto!(patch.is_solid, patch_is_solid_cpu)

    # Initialise patch from coarse (already uniform u_in equilibrium)
    KrakeNN.compute_macroscopic_2d!(ρ, ux, uy, f_in)
    KrakeNN.prolongate_f_rescaled_full_2d!(
        patch.f_in, f_in, ρ, ux, uy,
        patch.ratio, patch.Nx_inner, patch.Ny_inner,
        patch.n_ghost, first(patch.parent_i_range), first(patch.parent_j_range),
        Nx, Ny, Float64(ω), Float64(patch.omega))
    copyto!(patch.f_out, patch.f_in)
    KrakeNN.compute_macroscopic_2d!(patch.rho, patch.ux, patch.uy, patch.f_in)

    # Closures: coarse BCs (Zou-He inlet, pressure outlet), patch uses coarse
    stream_fn! = KrakeNN.stream_2d!
    bc_base_fn = f -> begin
        KrakeNN.apply_zou_he_west_2d!(f, u_in, Nx, Ny)
        KrakeNN.apply_zou_he_pressure_east_2d!(f, Nx, Ny)
    end
    collide_fn = (f, is_s) -> KrakeNN.collide_2d!(f, is_s, ω)
    macro_fn = KrakeNN.compute_macroscopic_2d!

    # Coarse-grid MEA: the restriction step at the end of each coarse call
    # pushes fine-patch data back into `f_in` over the cylinder region, so
    # the coarse MEA reflects the refined physics without FH force rescaling.
    Fx_sum = Ref(0.0); Fy_sum = Ref(0.0); n_avg = Ref(0)
    coarse_diag = (f_pre, f_post, is_s, nxc, nyc) -> begin
        d = compute_drag_mea_2d(f_pre, f_post, is_s, nxc, nyc)
        Fx_sum[] += d.Fx
        Fy_sum[] += d.Fy
        n_avg[] += 1
    end

    t0 = time()
    for step in 1:steps
        diag_this_step = step > steps - avg
        cdf = diag_this_step ? coarse_diag : nothing
        f_in, f_out = KrakeNN.advance_refined_step!(
            domain, f_in, f_out, ρ, ux, uy, is_solid;
            stream_fn=stream_fn!, collide_fn=collide_fn, macro_fn=macro_fn,
            bc_base_fn=bc_base_fn, coarse_diag_fn=cdf)
    end
    dt = time() - t0

    # Coarse diameter (coarse cells) — same normalisation as uniform case.
    Fx_avg = Fx_sum[] / max(n_avg[], 1)
    Cd = 2.0 * Fx_avg / (1.0 * u_in^2 * Float64(D))
    err = abs(Cd - CD_REF) / CD_REF * 100

    # Cost: coarse cells + fine cells·ratio (one fine step per coarse sub-step)
    coarse_cells = Nx * Ny
    fine_cells = patch.Nx_inner * patch.Ny_inner
    # each coarse step does 1 coarse update + ratio fine updates on the patch
    cost = (coarse_cells + ratio * fine_cells) * steps
    total_cells = coarse_cells + fine_cells

    return (; N=Ny, ratio, Cd, err,
              cells=total_cells, coarse_cells, fine_cells,
              steps, walltime=dt, cell_steps=cost)
end

# ---------------------------------------------------------------------
# Sweep + pretty table
# ---------------------------------------------------------------------
function main(; Ns_uniform=[40, 80, 160], Ns_refined=[40, 80], ratios=[2, 4],
               u_in=0.05, backend=KernelAbstractions.CPU(), FT=Float64,
               csv_path=nothing)
    println("\n" * "="^82)
    println("Cylinder Cd convergence at Re=20 (Schäfer-Turek, Cd_ref=$(CD_REF))")
    println("="^82)

    uni = typeof(run_uniform(40; u_in=0.0))[]   # placeholder type
    ref = typeof(run_refined(40; u_in=0.0, ratio=2))[]
    uni = Any[]; ref = Any[]   # reset — keep Any for heterogeneous named tuples

    println("\n-- Uniform --")
    @printf("%8s %8s %10s %8s %10s %14s\n",
            "N", "Nx", "cells", "Cd", "err%", "walltime_s")
    for N in Ns_uniform
        r = run_uniform(N; u_in=u_in, backend=backend, FT=FT)
        push!(uni, r)
        @printf("%8d %8d %10d %8.4f %9.2f%% %14.2f\n",
                r.N, 4r.N, r.cells, r.Cd, r.err, r.walltime)
    end

    println("\n-- Refined (patch around cylinder) --")
    @printf("%8s %6s %8s %8s %8s %10s %8s %10s %14s\n",
            "N_base", "ratio", "coarse", "fine", "total", "cell×steps",
            "Cd", "err%", "walltime_s")
    for N in Ns_refined, r in ratios
        res = run_refined(N; ratio=r, u_in=u_in, backend=backend, FT=FT)
        push!(ref, res)
        @printf("%8d %6d %8d %8d %8d %10.2e %8.4f %9.2f%% %14.2f\n",
                res.N, res.ratio, res.coarse_cells, res.fine_cells,
                res.cells, res.cell_steps, res.Cd, res.err, res.walltime)
    end

    if csv_path !== nothing
        open(csv_path, "w") do io
            println(io, "mode,N_base,ratio,coarse_cells,fine_cells,total_cells,steps,cell_steps,Cd,err_pct,walltime_s")
            for r in uni
                println(io, "uniform,$(r.N),1,$(r.cells),0,$(r.cells),$(r.steps),$(r.cell_steps),$(r.Cd),$(r.err),$(r.walltime)")
            end
            for r in ref
                println(io, "refined,$(r.N),$(r.ratio),$(r.coarse_cells),$(r.fine_cells),$(r.cells),$(r.steps),$(r.cell_steps),$(r.Cd),$(r.err),$(r.walltime)")
            end
        end
        println("\nCSV: $csv_path")
    end

    return (uniform=uni, refined=ref)
end

if abspath(PROGRAM_FILE) == @__FILE__
    ts = Dates.format(now(), "yyyymmdd_HHMMSS")
    csv = joinpath(@__DIR__, "results", "convergence_cylinder_refinement_$ts.csv")
    main(csv_path=csv)
end
