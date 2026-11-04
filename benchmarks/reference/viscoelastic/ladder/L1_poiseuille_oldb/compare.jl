#!/usr/bin/env julia
# L1 — planar Poiseuille Oldroyd-B (full pipeline), comparator
#
# Invocation:
#   julia --project=. bench/viscoelastic_validation/L1_poiseuille_oldb/compare.jl
#
# Reads `results/L1_run_latest.json` produced by `run.jl`, applies the
# thresholds defined in `reference.json`, prints a per-quantity PASS/FAIL
# table, and exits 0 (all PASS) or 1 (any FAIL).
#
# Threshold tweaks must go into `reference.json`, not here.

using JSON3
using Printf

const HERE = @__DIR__

struct CheckResult
    name::String
    value::Float64
    threshold::Float64
    pass::Bool
    note::String
end

function check_lt(name, value, threshold; note="")
    return CheckResult(name, value, threshold, value < threshold, note)
end

function check_gt(name, value, threshold; note="")
    return CheckResult(name, value, threshold, value > threshold, note)
end

function check_bool(name, value::Bool; note="")
    return CheckResult(name, value ? 1.0 : 0.0, 1.0, value, note)
end

function load_json(path)
    return JSON3.read(read(path, String))
end

"""
    hwbb_aware_wall_gamma_dot(u_profile)

Recompute the analytic shear rate `gamma_dot` on a Kraken-faithful 3-point
one-sided stencil at the wall-adjacent cells `j=1` and `j=Ny`, then a
standard central difference everywhere else. This matches the BC-aware
stencil applied inside `src/fvfd/operators_2d.jl::_fvfd_solid_bc_derivative_y_2d`
when the south/north BC is `:wall` (FVFD_BC_WALL, polymer_wall_extrap=Val(:quadratic)),
i.e. `(-3*u[1] + 4*u[2] - u[3])/(2*dy)` at `j=1` and mirror at `j=Ny`.

The previous reference used a central difference with a Dirichlet
ghost u(j=0)=0, giving `u[2]/2` at j=1. That ghost lives at the wall
location (j=0.5) only in a half-cell sense; the resulting CD then samples
the gradient at the *midpoint* j=0.5+1 = 1.5 (NOT at the cell center j=1),
so it undershoots the true cell-center gradient by ~26 % for a parabolic
profile. Kraken's stencil samples the cell-center gradient at j=1
correctly, producing the apparent mismatch flagged in M39 RESULTS_20260523.md.
"""
function hwbb_aware_wall_gamma_dot(u_profile)
    Ny = length(u_profile)
    gamma_dot = zeros(Float64, Ny)
    # interior: standard central difference (matches Kraken interior stencil)
    for j in 2:(Ny - 1)
        gamma_dot[j] = (u_profile[j + 1] - u_profile[j - 1]) / 2.0
    end
    # wall cells: 3-point quadratic one-sided (Kraken-faithful)
    gamma_dot[1] = (-3.0 * u_profile[1] + 4.0 * u_profile[2] - u_profile[3]) / 2.0
    gamma_dot[Ny] = (3.0 * u_profile[Ny] - 4.0 * u_profile[Ny - 1] + u_profile[Ny - 2]) / 2.0
    return gamma_dot
end

"""
    recompute_wall_metrics_hwbb(payload, refj)

Given the dumped run JSON, recompute the wall-adjacent comparison
metrics (`tau_xy_wall_relative`, `tau_xx_wall_relative`) using the
HWBB-aware analytic gamma_dot stencil instead of the central-difference
stencil baked into `run.jl::analytic_oldroydb_poiseuille`. Returns a
`NamedTuple` of overrides applied on top of `payload.diagnostics`.

Stencil provenance: see `hwbb_aware_wall_gamma_dot` docstring.
"""
function recompute_wall_metrics_hwbb(payload)
    prof = payload.profiles
    u_a = Float64.(collect(prof.u_analytic))
    tau_xy_k = Float64.(collect(prof.tau_xy_kraken))
    tau_xx_k = Float64.(collect(prof.tau_xx_kraken))

    # Wall-stencil-corrected analytic
    gd_corr = hwbb_aware_wall_gamma_dot(u_a)
    cfg = payload.config
    nu_p = Float64(cfg.nu_p)
    lambda = Float64(cfg.lambda)
    tau_xy_a_corr = nu_p .* gd_corr
    tau_xx_a_corr = 2.0 * lambda * nu_p .* gd_corr .^ 2

    Ny = length(u_a)
    tau_xy_wall_rel = max(
        abs(tau_xy_k[1] - tau_xy_a_corr[1]) / max(abs(tau_xy_a_corr[1]), eps()),
        abs(tau_xy_k[Ny] - tau_xy_a_corr[Ny]) / max(abs(tau_xy_a_corr[Ny]), eps()),
    )
    tau_xx_wall_rel = max(
        abs(tau_xx_k[1] - tau_xx_a_corr[1]) / max(abs(tau_xx_a_corr[1]), eps()),
        abs(tau_xx_k[Ny] - tau_xx_a_corr[Ny]) / max(abs(tau_xx_a_corr[Ny]), eps()),
    )
    return (
        tau_xy_wall_relative = tau_xy_wall_rel,
        tau_xx_wall_relative = tau_xx_wall_rel,
        tau_xy_wall_pair_analytic_corr = (tau_xy_a_corr[1], tau_xy_a_corr[Ny]),
        tau_xx_wall_pair_analytic_corr = (tau_xx_a_corr[1], tau_xx_a_corr[Ny]),
        gamma_dot_wall_corr = (gd_corr[1], gd_corr[Ny]),
    )
end

function format_value(v::Real)
    return @sprintf("%.6e", v)
end

format_value(v::Bool) = v ? "true" : "false"
format_value(v) = string(v)

function print_table(checks)
    println()
    println("L1 Poiseuille Oldroyd-B — PASS/FAIL report")
    println("==========================================")
    println(@sprintf("  %-40s  %12s  %12s  %-6s  %s",
                     "quantity", "value", "threshold", "verdict", "note"))
    println(@sprintf("  %-40s  %12s  %12s  %-6s  %s",
                     "─"^40, "─"^12, "─"^12, "─"^6, "─"^30))
    for c in checks
        verdict = c.pass ? "PASS" : "FAIL"
        println(@sprintf("  %-40s  %12s  %12s  %-6s  %s",
            c.name,
            format_value(c.value),
            format_value(c.threshold),
            verdict,
            c.note,
        ))
    end
    n_pass = count(c -> c.pass, checks)
    n_fail = length(checks) - n_pass
    println()
    println(@sprintf("Overall: %d/%d PASS, %d FAIL", n_pass, length(checks), n_fail))
    return n_fail == 0
end

function main()
    results_dir = joinpath(HERE, "results")
    latest = joinpath(results_dir, "L1_run_latest.json")
    if !isfile(latest)
        @error "no run output found; please run run.jl first" latest
        exit(2)
    end
    payload = load_json(latest)
    refj = load_json(joinpath(HERE, "reference.json"))
    diag = payload.diagnostics
    tol = refj.tolerance_pass_fail

    # M40 Phase A: recompute wall metrics with HWBB-aware (Kraken-faithful)
    # one-sided gamma_dot stencil at j=1, j=Ny. See compare.jl docstring on
    # `hwbb_aware_wall_gamma_dot`. The dumped wall metrics in `diag` used
    # the old central-difference stencil and undershoot by ~26 % (tau_xy)
    # / 84 % (tau_xx) at the wall row.
    overrides = recompute_wall_metrics_hwbb(payload)

    tau_xy_wall_rel_used = overrides.tau_xy_wall_relative
    tau_xx_wall_rel_used = overrides.tau_xx_wall_relative

    println(@sprintf("[M40] HWBB-aware wall stencil applied:"))
    println(@sprintf("  gamma_dot_wall_corr  = (%.6e, %.6e)",
        overrides.gamma_dot_wall_corr[1], overrides.gamma_dot_wall_corr[2]))
    println(@sprintf("  tau_xy_wall_a_corr   = (%.6e, %.6e)   (was %.6e)",
        overrides.tau_xy_wall_pair_analytic_corr[1],
        overrides.tau_xy_wall_pair_analytic_corr[2],
        Float64(diag.tau_xy_wall_pair_analytic[1])))
    println(@sprintf("  tau_xx_wall_a_corr   = (%.6e, %.6e)   (was %.6e)",
        overrides.tau_xx_wall_pair_analytic_corr[1],
        overrides.tau_xx_wall_pair_analytic_corr[2],
        Float64(diag.tau_xx_wall_pair_analytic[1])))
    println(@sprintf("  tau_xy_wall_rel: %.6e (pre-fix %.6e)",
        tau_xy_wall_rel_used, Float64(diag.tau_xy_wall_relative)))
    println(@sprintf("  tau_xx_wall_rel: %.6e (pre-fix %.6e)",
        tau_xx_wall_rel_used, Float64(diag.tau_xx_wall_relative)))

    checks = CheckResult[
        check_lt(
            "u_centerline_relative",
            Float64(diag.u_centerline_relative),
            Float64(tol.u_centerline_relL2_max);
            note="vs Bird-AH §3.4 peak",
        ),
        check_lt(
            "u_profile_interior_relL2",
            Float64(diag.u_profile_interior_relL2),
            Float64(tol.u_centerline_relL2_max);
            note="profile relL2 vs analytic",
        ),
        check_lt(
            "tau_xy_wall_relative",
            tau_xy_wall_rel_used,
            Float64(tol.tau_xy_wall_relL2_max);
            note="HWBB-aware stencil at j=1, j=Ny",
        ),
        check_lt(
            "tau_xy_profile_interior_relL2",
            Float64(diag.tau_xy_profile_interior_relL2),
            Float64(tol.tau_xy_wall_relL2_max);
            note="interior profile relL2",
        ),
        check_lt(
            "tau_xx_wall_relative",
            tau_xx_wall_rel_used,
            Float64(tol.tau_xx_wall_relL2_max);
            note="HWBB-aware stencil at j=1, j=Ny",
        ),
        check_lt(
            "tau_xx_profile_interior_relL2",
            Float64(diag.tau_xx_profile_interior_relL2),
            Float64(tol.tau_xx_wall_relL2_max);
            note="interior profile relL2",
        ),
        check_lt(
            "rho_max_abs_deviation",
            Float64(diag.rho_max_abs_deviation),
            Float64(tol.rho_max_abs_deviation);
            note="incompressibility / LBM density drift",
        ),
        check_lt(
            "uy_interior_max_abs",
            Float64(diag.uy_interior_max_abs),
            Float64(tol.uy_interior_max_abs);
            note="should be ~0 (1D flow)",
        ),
        check_gt(
            "min_eig_C",
            Float64(diag.min_eig_C),
            Float64(tol.min_eig_C_min);
            note="conformation SPD",
        ),
        check_bool(
            "no_nan_no_inf",
            Bool(diag.no_nan_no_inf);
            note="sentinel",
        ),
    ]

    ok = print_table(checks)
    println()
    if ok
        println("L1: PASS")
        exit(0)
    else
        println("L1: FAIL — see per-quantity verdicts above; consult EXPECTED.md for diagnosis")
        exit(1)
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
