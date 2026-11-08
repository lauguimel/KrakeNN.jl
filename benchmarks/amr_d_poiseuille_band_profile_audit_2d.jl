#!/usr/bin/env julia

using CairoMakie
using KrakeNN
using Printf

const OUTDIR = get(ENV, "KRK_AMR_D_POISEUILLE_AUDIT_OUTDIR",
                   joinpath(dirname(@__DIR__), "benchmarks", "results",
                            "figures", "amr_d_poiseuille_band_profile_audit_2d"))
const STEPS = parse(Int, get(ENV, "KRK_AMR_D_POISEUILLE_AUDIT_STEPS", "5000"))
const NX = parse(Int, get(ENV, "KRK_AMR_D_POISEUILLE_AUDIT_NX", "24"))
const NY = parse(Int, get(ENV, "KRK_AMR_D_POISEUILLE_AUDIT_NY", "16"))
const FX = parse(Float64, get(ENV, "KRK_AMR_D_POISEUILLE_AUDIT_FX", "5e-5"))

mkpath(OUTDIR)

function _leaf_fields(result; force_x=FX, volume=0.25)
    leaf = zeros(Float64, 2 * size(result.coarse_F, 1),
                 2 * size(result.coarse_F, 2), 9)
    composite_to_leaf_F_2d!(leaf, result.coarse_F, result.patch)
    ux = zeros(Float64, size(leaf, 1), size(leaf, 2))
    rho = similar(ux)
    @inbounds for j in axes(leaf, 2), i in axes(leaf, 1)
        cell = @view leaf[i, j, :]
        rho_ij = mass_F(cell) / volume
        mx, _ = momentum_F(cell)
        rho[i, j] = rho_ij
        ux[i, j] = (mx / volume + force_x / 2) / rho_ij
    end
    return (; ux, rho)
end

function _row_mean(A)
    out = zeros(Float64, size(A, 2))
    @inbounds for j in axes(A, 2)
        out[j] = sum(@view A[:, j]) / size(A, 1)
    end
    return out
end

function _run_case(label::Symbol, patch_i, patch_j)
    route = run_conservative_tree_poiseuille_route_native_2d(
        ; Nx=NX, Ny=NY, patch_i_range=patch_i, patch_j_range=patch_j,
        Fx=FX, steps=STEPS)
    cart = run_conservative_tree_poiseuille_macroflow_2d(
        ; Nx=NX, Ny=NY, patch_i_range=1:NX, patch_j_range=1:NY,
        Fx=FX, steps=STEPS)
    route_fields = _leaf_fields(route)
    cart_fields = _leaf_fields(cart)
    return (; label, route, cart, route_fields, cart_fields)
end

function _plot_case(case)
    y = collect(1:length(case.route.ux_profile))
    route_rho = _row_mean(case.route_fields.rho)
    cart_rho = _row_mean(case.cart_fields.rho)
    du = case.route.ux_profile .- case.cart.ux_profile
    drho = route_rho .- cart_rho

    fig = Figure(size=(1450, 900), fontsize=16)
    ax1 = Axis(fig[1, 1], title="$(case.label) ux profile",
               xlabel="ux", ylabel="y leaf")
    lines!(ax1, case.route.analytic_ux_profile, y;
           label="analytic", color=:black, linewidth=3)
    lines!(ax1, case.cart.ux_profile, y;
           label="leaf Cartesian", color=:dodgerblue4, linewidth=2.5)
    lines!(ax1, case.route.ux_profile, y;
           label="AMR-D route", color=:orangered, linewidth=2.5)
    axislegend(ax1, position=:rb)

    ax2 = Axis(fig[1, 2], title="$(case.label) rho profile",
               xlabel="rho", ylabel="y leaf")
    lines!(ax2, fill(1.0, length(y)), y;
           label="rho=1", color=:black, linestyle=:dash, linewidth=2)
    lines!(ax2, cart_rho, y;
           label="leaf Cartesian", color=:dodgerblue4, linewidth=2.5)
    lines!(ax2, route_rho, y;
           label="AMR-D route", color=:orangered, linewidth=2.5)
    axislegend(ax2, position=:rb)

    ax3 = Axis(fig[2, 1], title="$(case.label) ux route - Cartesian",
               xlabel="delta ux", ylabel="y leaf")
    lines!(ax3, du, y; color=:purple4, linewidth=2.5)
    vlines!(ax3, [0.0]; color=:black, linestyle=:dash)

    ax4 = Axis(fig[2, 2], title="$(case.label) rho route - Cartesian",
               xlabel="delta rho", ylabel="y leaf")
    lines!(ax4, drho, y; color=:purple4, linewidth=2.5)
    vlines!(ax4, [0.0]; color=:black, linestyle=:dash)

    path = joinpath(OUTDIR, "$(case.label)_profiles.png")
    save(path, fig)
    return path
end

function _write_summary(cases, paths)
    csv_path = joinpath(OUTDIR, "summary.csv")
    open(csv_path, "w") do io
        println(io, "case,steps,route_mean_ux,cart_mean_ux,analytic_mean_ux,route_max_ux,cart_max_ux,analytic_max_ux,route_l2,cart_l2,route_linf,cart_linf,route_mass_rel,plot")
        for (case, path) in zip(cases, paths)
            route = case.route
            cart = case.cart
            rel = abs(route.mass_drift) / max(abs(route.mass_initial), eps(Float64))
            @printf(io, "%s,%d,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%s\n",
                    String(case.label), STEPS,
                    sum(route.ux_profile) / length(route.ux_profile),
                    sum(cart.ux_profile) / length(cart.ux_profile),
                    sum(route.analytic_ux_profile) / length(route.analytic_ux_profile),
                    maximum(route.ux_profile), maximum(cart.ux_profile),
                    maximum(route.analytic_ux_profile),
                    route.l2_error, cart.l2_error,
                    route.linf_error, cart.linf_error, rel, path)
        end
    end

    md_path = joinpath(OUTDIR, "summary.md")
    open(md_path, "w") do io
        println(io, "# AMR-D Poiseuille Band Profile Audit")
        println(io)
        println(io, "Steps: `$(STEPS)`")
        println(io)
        println(io, "| case | route mean ux | cart mean ux | analytic mean ux | route L2 | cart L2 | route Linf | cart Linf | mass rel |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for case in cases
            route = case.route
            cart = case.cart
            rel = abs(route.mass_drift) / max(abs(route.mass_initial), eps(Float64))
            @printf(io, "| %s | %.6e | %.6e | %.6e | %.3e | %.3e | %.3e | %.3e | %.3e |\n",
                    String(case.label),
                    sum(route.ux_profile) / length(route.ux_profile),
                    sum(cart.ux_profile) / length(cart.ux_profile),
                    sum(route.analytic_ux_profile) / length(route.analytic_ux_profile),
                    route.l2_error, cart.l2_error,
                    route.linf_error, cart.linf_error, rel)
        end
        println(io)
        println(io, "Plots:")
        for path in paths
            println(io, "- `$(path)`")
        end
    end
    return csv_path, md_path
end

function main()
    cases = (
        _run_case(:xband, 11:14, 1:NY),
        _run_case(:yband, 1:NX, 7:10),
    )
    paths = [_plot_case(case) for case in cases]
    csv_path, md_path = _write_summary(cases, paths)
    println("wrote ", csv_path)
    println("wrote ", md_path)
    return cases
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
