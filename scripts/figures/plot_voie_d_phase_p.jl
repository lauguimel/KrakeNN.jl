#!/usr/bin/env julia --project=.

using CairoMakie
using KrakeNN
using Printf

const FIGDIR = joinpath("paper", "figures")
const DATADIR = joinpath("paper", "data")
mkpath(FIGDIR)
mkpath(DATADIR)

function fluid_mass(F, is_solid)
    total = 0.0
    for q in 1:9, j in axes(F, 2), i in axes(F, 1)
        is_solid[i, j] && continue
        total += F[i, j, q]
    end
    return total
end

function fluid_mean_velocity(F, is_solid; volume=0.25, force_x=0.0, force_y=0.0)
    ux_sum = 0.0
    uy_sum = 0.0
    n_fluid = 0
    for j in axes(F, 2), i in axes(F, 1)
        is_solid[i, j] && continue
        cell = @view F[i, j, :]
        rho = mass_F(cell) / volume
        p = momentum_F(cell)
        ux_sum += (p[1] / volume + force_x / 2) / rho
        uy_sum += (p[2] / volume + force_y / 2) / rho
        n_fluid += 1
    end
    return ux_sum / n_fluid, uy_sum / n_fluid
end

function velocity_fields(F, is_solid; volume=0.25, force_x=0.0, force_y=0.0)
    ux = fill(NaN, size(F, 1), size(F, 2))
    uy = fill(NaN, size(F, 1), size(F, 2))
    speed = fill(NaN, size(F, 1), size(F, 2))
    rho = fill(NaN, size(F, 1), size(F, 2))
    for j in axes(F, 2), i in axes(F, 1)
        is_solid[i, j] && continue
        cell = @view F[i, j, :]
        rho_ij = mass_F(cell) / volume
        p = momentum_F(cell)
        ux_ij = (p[1] / volume + force_x / 2) / rho_ij
        uy_ij = (p[2] / volume + force_y / 2) / rho_ij
        rho[i, j] = rho_ij
        ux[i, j] = ux_ij
        uy[i, j] = uy_ij
        speed[i, j] = sqrt(ux_ij^2 + uy_ij^2)
    end
    return (; ux, uy, speed, rho)
end

function patch_leaf_map(Nx, Ny, patch)
    m = zeros(Float64, 2 * Nx, 2 * Ny)
    for J in patch.parent_j_range, I in patch.parent_i_range
        m[2 * I - 1:2 * I, 2 * J - 1:2 * J] .= 1.0
    end
    return m
end

function rect_lines_xy!(ax, xmin, xmax, ymin, ymax; color=:black, linewidth=1)
    lines!(ax, [xmin, xmax, xmax, xmin, xmin],
              [ymin, ymin, ymax, ymax, ymin];
           color=color, linewidth=linewidth)
end

function rect_lines!(ax, i_range, j_range; color=:black, linewidth=2)
    rect_lines_xy!(ax, first(i_range) - 0.5, last(i_range) + 0.5,
                   first(j_range) - 0.5, last(j_range) + 0.5;
                   color=color, linewidth=linewidth)
end

function draw_active_mesh!(ax, Nx, Ny, patch)
    pi = patch.parent_i_range
    pj = patch.parent_j_range
    for J in 1:Ny, I in 1:Nx
        if first(pi) <= I <= last(pi) && first(pj) <= J <= last(pj)
            for jf in (2 * J - 1):(2 * J), ifine in (2 * I - 1):(2 * I)
                rect_lines_xy!(ax, ifine - 0.5, ifine + 0.5,
                               jf - 0.5, jf + 0.5;
                               color=(:dodgerblue4, 0.75), linewidth=0.7)
            end
        else
            rect_lines_xy!(ax, 2 * I - 1.5, 2 * I + 0.5,
                           2 * J - 1.5, 2 * J + 0.5;
                           color=(:gray20, 0.55), linewidth=0.8)
        end
    end
    rect_lines!(ax, 2 * first(pi) - 1:2 * last(pi),
                2 * first(pj) - 1:2 * last(pj);
                color=:black, linewidth=3)
    return ax
end

function patch_leaf_ranges(patch)
    return (2 * first(patch.parent_i_range) - 1):(2 * last(patch.parent_i_range)),
           (2 * first(patch.parent_j_range) - 1):(2 * last(patch.parent_j_range))
end

function run_poiseuille_band(; name, Nx, Ny, patch_i_range, patch_j_range,
                             Fx=5e-5, omega=1.0, steps=5000)
    result = run_conservative_tree_poiseuille_macroflow_2d(
        ; Nx=Nx, Ny=Ny, patch_i_range=patch_i_range,
        patch_j_range=patch_j_range, Fx=Fx, omega=omega, steps=steps)
    leaf = zeros(Float64, 2 * Nx, 2 * Ny, 9)
    composite_to_leaf_F_2d!(leaf, result.coarse_F, result.patch)
    is_solid = falses(size(leaf, 1), size(leaf, 2))
    fields = velocity_fields(leaf, is_solid; volume=0.25, force_x=Fx)
    return (; name, result, leaf, fields)
end

function run_couette_case(; Nx=24, Ny=16, patch_i_range=9:16,
                          patch_j_range=5:12, U=0.04, omega=1.0,
                          steps=3000)
    result = run_conservative_tree_couette_macroflow_2d(
        ; Nx=Nx, Ny=Ny, patch_i_range=patch_i_range,
        patch_j_range=patch_j_range, U=U, omega=omega, steps=steps)
    leaf = zeros(Float64, 2 * Nx, 2 * Ny, 9)
    composite_to_leaf_F_2d!(leaf, result.coarse_F, result.patch)
    is_solid = falses(size(leaf, 1), size(leaf, 2))
    fields = velocity_fields(leaf, is_solid; volume=0.25)
    return (; name="Couette central patch", result, leaf, fields)
end

function run_bfs_case(; Nx=28, Ny=14, patch_i_range=1:12,
                      patch_j_range=1:8, step_i_leaf=16,
                      step_height_leaf=8, u_in=0.03, omega=1.0,
                      steps=800)
    result = run_conservative_tree_bfs_macroflow_2d(
        ; Nx=Nx, Ny=Ny, patch_i_range=patch_i_range,
        patch_j_range=patch_j_range, step_i_leaf=step_i_leaf,
        step_height_leaf=step_height_leaf, u_in=u_in, omega=omega,
        steps=steps)
    leaf = zeros(Float64, 2 * Nx, 2 * Ny, 9)
    composite_to_leaf_F_2d!(leaf, result.coarse_F, result.patch)
    fields = velocity_fields(leaf, result.is_solid_leaf; volume=0.25)
    return (; result, leaf, fields, u_in, omega)
end

function run_cartesian_bfs(is_solid; u_in=0.03, steps=800, omega=1.0)
    F = zeros(Float64, size(is_solid, 1), size(is_solid, 2), 9)
    Fnext = similar(F)
    fill_equilibrium_integrated_D2Q9!(F, 0.25, 1.0, u_in, 0.0)
    mass_initial = fluid_mass(F, is_solid)

    for _ in 1:steps
        stream_bounceback_xy_solid_F_2d!(Fnext, F, is_solid)
        apply_zou_he_west_F_2d!(Fnext, u_in, 0.25, is_solid)
        apply_zou_he_pressure_east_F_2d!(Fnext, 0.25, is_solid; rho_out=1.0)
        collide_BGK_integrated_D2Q9!(Fnext, is_solid, 0.25, omega)
        F, Fnext = Fnext, F
    end

    fields = velocity_fields(F, is_solid; volume=0.25)
    ux_mean, uy_mean = fluid_mean_velocity(F, is_solid; volume=0.25)
    return (; F, fields, ux_mean, uy_mean, mass_initial,
            mass_final=fluid_mass(F, is_solid), steps)
end

function run_refined_square_drag(; Nx=24, Ny=14, patch_i_range=9:16,
                                 patch_j_range=4:11,
                                 obstacle_i_range=21:28,
                                 obstacle_j_range=11:18,
                                 Fx=2e-5, Fy=0.0, steps=1200,
                                 avg_window=300, omega=1.0)
    coarse = zeros(Float64, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range)
    leaf = zeros(Float64, 2 * Nx, 2 * Ny, 9)
    leaf_next = similar(leaf)
    is_solid = square_solid_mask_leaf_2d(2 * Nx, 2 * Ny,
                                         obstacle_i_range, obstacle_j_range)

    fill_equilibrium_integrated_D2Q9!(coarse, 1.0, 1.0, 0.0, 0.0)
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, 0.25, 1.0, 0.0, 0.0)
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = fluid_mass(leaf, is_solid)

    Fx_sum = 0.0
    Fy_sum = 0.0
    n_avg = 0
    for step in 1:steps
        composite_to_leaf_F_2d!(leaf, coarse, patch)
        collide_Guo_integrated_D2Q9!(leaf, is_solid, 0.25, omega, Fx, Fy)
        stream_periodic_x_wall_y_solid_F_2d!(leaf_next, leaf, is_solid)
        if step > steps - avg_window
            drag = compute_drag_mea_solid_F_2d(leaf, leaf_next, is_solid)
            Fx_sum += drag.Fx
            Fy_sum += drag.Fy
            n_avg += 1
        end
        leaf_to_composite_F_2d!(coarse_next, patch_next, leaf_next)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    fields = velocity_fields(leaf, is_solid; volume=0.25, force_x=Fx, force_y=Fy)
    ux_mean, uy_mean = fluid_mean_velocity(leaf, is_solid;
                                           volume=0.25, force_x=Fx, force_y=Fy)
    return (; coarse, patch, leaf, fields, is_solid, obstacle_i_range,
            obstacle_j_range, Fx_drag=Fx_sum / n_avg, Fy_drag=Fy_sum / n_avg,
            ux_mean, uy_mean, mass_initial,
            mass_final=fluid_mass(leaf, is_solid), Fx, Fy, steps, avg_window)
end

function run_coarse_square_drag(; Nx=24, Ny=14, obstacle_i_range=11:14,
                                obstacle_j_range=6:9, Fx=2e-5, Fy=0.0,
                                steps=1200, avg_window=300, omega=1.0)
    F = zeros(Float64, Nx, Ny, 9)
    Fnext = similar(F)
    is_solid = square_solid_mask_leaf_2d(Nx, Ny, obstacle_i_range, obstacle_j_range)
    fill_equilibrium_integrated_D2Q9!(F, 1.0, 1.0, 0.0, 0.0)
    mass_initial = fluid_mass(F, is_solid)

    Fx_sum = 0.0
    Fy_sum = 0.0
    n_avg = 0
    for step in 1:steps
        collide_Guo_integrated_D2Q9!(F, is_solid, 1.0, omega, Fx, Fy)
        stream_periodic_x_wall_y_solid_F_2d!(Fnext, F, is_solid)
        if step > steps - avg_window
            drag = compute_drag_mea_solid_F_2d(F, Fnext, is_solid)
            Fx_sum += drag.Fx
            Fy_sum += drag.Fy
            n_avg += 1
        end
        F, Fnext = Fnext, F
    end

    fields = velocity_fields(F, is_solid; volume=1.0, force_x=Fx, force_y=Fy)
    ux_mean, uy_mean = fluid_mean_velocity(F, is_solid;
                                           volume=1.0, force_x=Fx, force_y=Fy)
    return (; F, fields, is_solid, obstacle_i_range, obstacle_j_range,
            Fx_drag=Fx_sum / n_avg, Fy_drag=Fy_sum / n_avg, ux_mean, uy_mean,
            mass_initial, mass_final=fluid_mass(F, is_solid),
            Fx, Fy, steps, avg_window)
end

function plot_poiseuille_bands(vertical, horizontal)
    fig = Figure(size=(1500, 900), fontsize=18)
    for (row, case) in enumerate((vertical, horizontal))
        result = case.result
        mesh = patch_leaf_map(size(result.coarse_F, 1), size(result.coarse_F, 2),
                              result.patch)
        pi_leaf, pj_leaf = patch_leaf_ranges(result.patch)

        axm = Axis(fig[row, 1], title="$(case.name): active mesh",
                   aspect=DataAspect(), xlabel="x leaf", ylabel="y leaf")
        heatmap!(axm, 1:size(mesh, 1), 1:size(mesh, 2), mesh;
                 colormap=[:gray96, :lightskyblue1])
        draw_active_mesh!(axm, size(result.coarse_F, 1), size(result.coarse_F, 2),
                          result.patch)

        axu = Axis(fig[row, 2],
                   title=@sprintf("ux, L2=%.3e, Linf=%.3e",
                                  result.l2_error, result.linf_error),
                   aspect=DataAspect(), xlabel="x leaf", ylabel="y leaf")
        hm = heatmap!(axu, 1:size(case.fields.ux, 1), 1:size(case.fields.ux, 2),
                      case.fields.ux; colormap=:viridis)
        rect_lines!(axu, pi_leaf, pj_leaf; color=:white, linewidth=3)
        Colorbar(fig[row, 3], hm, label="ux")

        axp = Axis(fig[row, 4], title="profile vs analytic",
                   xlabel="ux", ylabel="y leaf")
        y = collect(1:length(result.ux_profile))
        lines!(axp, result.analytic_ux_profile, y; label="analytic",
               color=:black, linewidth=3)
        lines!(axp, result.ux_profile, y; label="voie D",
               color=:orangered, linewidth=3)
        axislegend(axp, position=:rb)
    end
    save(joinpath(FIGDIR, "voie_d_poiseuille_bands.pdf"), fig)
    save(joinpath(FIGDIR, "voie_d_poiseuille_bands.png"), fig; px_per_unit=2)
end

function plot_square_obstacle(refined, coarse)
    fig = Figure(size=(1500, 950), fontsize=18)
    pi_leaf, pj_leaf = patch_leaf_ranges(refined.patch)

    ax1 = Axis(fig[1, 1], title="refined square obstacle: speed",
               aspect=DataAspect(), xlabel="x leaf", ylabel="y leaf")
    hm1 = heatmap!(ax1, 1:size(refined.fields.speed, 1),
                   1:size(refined.fields.speed, 2), refined.fields.speed;
                   colormap=:viridis, nan_color=:black)
    rect_lines!(ax1, pi_leaf, pj_leaf; color=:white, linewidth=3)
    rect_lines!(ax1, refined.obstacle_i_range, refined.obstacle_j_range;
                color=:red, linewidth=3)
    Colorbar(fig[1, 2], hm1, label="|u|")

    ax2 = Axis(fig[1, 3], title="coarse Cartesian square: speed",
               aspect=DataAspect(), xlabel="x coarse", ylabel="y coarse")
    hm2 = heatmap!(ax2, 1:size(coarse.fields.speed, 1),
                   1:size(coarse.fields.speed, 2), coarse.fields.speed;
                   colormap=:viridis, nan_color=:black)
    rect_lines!(ax2, coarse.obstacle_i_range, coarse.obstacle_j_range;
                color=:red, linewidth=3)
    Colorbar(fig[1, 4], hm2, label="|u|")

    mesh = patch_leaf_map(size(refined.coarse, 1), size(refined.coarse, 2),
                          refined.patch)
    obstacle = Float64.(refined.is_solid)
    ax3 = Axis(fig[2, 1], title="refined mesh and obstacle",
               aspect=DataAspect(), xlabel="x leaf", ylabel="y leaf")
    heatmap!(ax3, 1:size(mesh, 1), 1:size(mesh, 2), mesh;
             colormap=[:gray92, :dodgerblue3])
    heatmap!(ax3, 1:size(obstacle, 1), 1:size(obstacle, 2), obstacle;
             colormap=(:Reds, 0.65))
    draw_active_mesh!(ax3, size(refined.coarse, 1), size(refined.coarse, 2),
                      refined.patch)
    rect_lines!(ax3, refined.obstacle_i_range, refined.obstacle_j_range;
                color=:red, linewidth=3)

    ax4 = Axis(fig[2, 3], title="centerline ux",
               xlabel="x / L", ylabel="ux")
    j_ref = cld(size(refined.fields.ux, 2), 2)
    j_coarse = cld(size(coarse.fields.ux, 2), 2)
    x_ref = collect(1:size(refined.fields.ux, 1)) ./ size(refined.fields.ux, 1)
    x_coarse = collect(1:size(coarse.fields.ux, 1)) ./ size(coarse.fields.ux, 1)
    lines!(ax4, x_ref, refined.fields.ux[:, j_ref]; label="refined",
           color=:orangered, linewidth=3)
    lines!(ax4, x_coarse, coarse.fields.ux[:, j_coarse]; label="coarse",
           color=:black, linewidth=3)
    axislegend(ax4, position=:rb)

    ax5 = Axis(fig[2, 4], title="drag Fx", ylabel="Fx")
    barplot!(ax5, [1, 2], [refined.Fx_drag, coarse.Fx_drag];
             color=[:orangered, :gray35])
    ax5.xticks = ([1, 2], ["refined", "coarse"])

    Label(fig[3, 1:4],
          @sprintf("refined Fx=%.6e, coarse Fx=%.6e, ratio=%.3f | refined ux=%.6e, coarse ux=%.6e",
                   refined.Fx_drag, coarse.Fx_drag,
                   refined.Fx_drag / coarse.Fx_drag,
                   refined.ux_mean, coarse.ux_mean),
          tellwidth=false)
    save(joinpath(FIGDIR, "voie_d_square_obstacle.pdf"), fig)
    save(joinpath(FIGDIR, "voie_d_square_obstacle.png"), fig; px_per_unit=2)
end

function plot_couette_and_bfs(couette, bfs, bfs_cart)
    fig = Figure(size=(1500, 950), fontsize=18)

    result = couette.result
    mesh = patch_leaf_map(size(result.coarse_F, 1), size(result.coarse_F, 2),
                          result.patch)
    pi_leaf, pj_leaf = patch_leaf_ranges(result.patch)
    ax1 = Axis(fig[1, 1], title="Couette: active mesh",
               aspect=DataAspect(), xlabel="x leaf", ylabel="y leaf")
    heatmap!(ax1, 1:size(mesh, 1), 1:size(mesh, 2), mesh;
             colormap=[:gray96, :lightskyblue1])
    draw_active_mesh!(ax1, size(result.coarse_F, 1), size(result.coarse_F, 2),
                      result.patch)

    ax2 = Axis(fig[1, 2], title=@sprintf("Couette ux, L2=%.3e",
                                         result.l2_error),
               aspect=DataAspect(), xlabel="x leaf", ylabel="y leaf")
    hm2 = heatmap!(ax2, 1:size(couette.fields.ux, 1), 1:size(couette.fields.ux, 2),
                   couette.fields.ux; colormap=:viridis)
    rect_lines!(ax2, pi_leaf, pj_leaf; color=:white, linewidth=3)
    Colorbar(fig[1, 3], hm2, label="ux")

    ax3 = Axis(fig[1, 4], title="Couette profile",
               xlabel="ux", ylabel="y leaf")
    y = collect(1:length(result.ux_profile))
    lines!(ax3, result.analytic_ux_profile, y; label="analytic",
           color=:black, linewidth=3)
    lines!(ax3, result.ux_profile, y; label="voie D",
           color=:orangered, linewidth=3)
    axislegend(ax3, position=:rb)

    bfs_result = bfs.result
    bfs_mesh = patch_leaf_map(size(bfs_result.coarse_F, 1),
                              size(bfs_result.coarse_F, 2),
                              bfs_result.patch)
    pi_leaf, pj_leaf = patch_leaf_ranges(bfs_result.patch)

    ax4 = Axis(fig[2, 1], title="BFS: refined speed",
               aspect=DataAspect(), xlabel="x leaf", ylabel="y leaf")
    hm4 = heatmap!(ax4, 1:size(bfs.fields.speed, 1), 1:size(bfs.fields.speed, 2),
                   bfs.fields.speed; colormap=:viridis, nan_color=:black)
    heatmap!(ax4, 1:size(bfs_result.is_solid_leaf, 1),
             1:size(bfs_result.is_solid_leaf, 2), Float64.(bfs_result.is_solid_leaf);
             colormap=(:Reds, 0.55))
    rect_lines!(ax4, pi_leaf, pj_leaf; color=:white, linewidth=3)
    Colorbar(fig[2, 2], hm4, label="|u|")

    ax5 = Axis(fig[2, 3], title="BFS: Cartesian leaf speed",
               aspect=DataAspect(), xlabel="x leaf", ylabel="y leaf")
    hm5 = heatmap!(ax5, 1:size(bfs_cart.fields.speed, 1),
                   1:size(bfs_cart.fields.speed, 2), bfs_cart.fields.speed;
                   colormap=:viridis, nan_color=:black)
    heatmap!(ax5, 1:size(bfs_result.is_solid_leaf, 1),
             1:size(bfs_result.is_solid_leaf, 2), Float64.(bfs_result.is_solid_leaf);
             colormap=(:Reds, 0.55))
    Colorbar(fig[2, 4], hm5, label="|u|")

    ax6 = Axis(fig[3, 1:4], title="BFS centerline ux",
               xlabel="x / L", ylabel="ux")
    j_ref = cld(size(bfs.fields.ux, 2), 2)
    x_ref = collect(1:size(bfs.fields.ux, 1)) ./ size(bfs.fields.ux, 1)
    lines!(ax6, x_ref, bfs.fields.ux[:, j_ref]; label="voie D",
           color=:orangered, linewidth=3)
    lines!(ax6, x_ref, bfs_cart.fields.ux[:, j_ref]; label="Cartesian leaf",
           color=:black, linewidth=3)
    axislegend(ax6, position=:rb)

    save(joinpath(FIGDIR, "voie_d_couette_bfs.pdf"), fig)
    save(joinpath(FIGDIR, "voie_d_couette_bfs.png"), fig; px_per_unit=2)
end

function write_summary(path, couette, vertical, horizontal, refined_square,
                       coarse_square, bfs, bfs_cart)
    open(path, "w") do io
        println(io, "# Voie D Phase P Summary")
        println(io)
        println(io, "Scope: fixed ratio-2 patch refinement for D2Q9, validated on Couette, Poiseuille bands, square obstacle drag, and backward-facing-step (BFS/VFS) smoke comparison.")
        println(io)
        println(io, "## Couette")
        println(io)
        println(io, "| case | steps | mass drift | L2 error | Linf error |")
        println(io, "|---|---:|---:|---:|---:|")
        r = couette.result
        @printf(io, "| %s | %d | %.6e | %.6e | %.6e |\n",
                couette.name, r.steps, r.mass_drift, r.l2_error, r.linf_error)
        println(io)
        println(io, "## Poiseuille Bands")
        println(io)
        println(io, "| case | steps | mass drift | L2 error | Linf error |")
        println(io, "|---|---:|---:|---:|---:|")
        for case in (vertical, horizontal)
            r = case.result
            @printf(io, "| %s | %d | %.6e | %.6e | %.6e |\n",
                    case.name, r.steps, r.mass_drift, r.l2_error, r.linf_error)
        end
        println(io)
        println(io, "## Square Obstacle")
        println(io)
        println(io, "| case | steps | avg window | mass drift | Fx | Fy | ux mean |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|")
        @printf(io, "| refined patch | %d | %d | %.6e | %.6e | %.6e | %.6e |\n",
                refined_square.steps, refined_square.avg_window,
                refined_square.mass_final - refined_square.mass_initial,
                refined_square.Fx_drag, refined_square.Fy_drag, refined_square.ux_mean)
        @printf(io, "| coarse Cartesian | %d | %d | %.6e | %.6e | %.6e | %.6e |\n",
                coarse_square.steps, coarse_square.avg_window,
                coarse_square.mass_final - coarse_square.mass_initial,
                coarse_square.Fx_drag, coarse_square.Fy_drag, coarse_square.ux_mean)
        @printf(io, "\nFx ratio refined/coarse: %.6f\n",
                refined_square.Fx_drag / coarse_square.Fx_drag)
        println(io)
        println(io, "## Backward-Facing Step")
        println(io)
        println(io, "| case | steps | mass final | ux mean | uy mean |")
        println(io, "|---|---:|---:|---:|---:|")
        r = bfs.result
        @printf(io, "| voie D | %d | %.6e | %.6e | %.6e |\n",
                r.steps, r.mass_final, r.ux_mean, r.uy_mean)
        @printf(io, "| Cartesian leaf | %d | %.6e | %.6e | %.6e |\n",
                bfs_cart.steps, bfs_cart.mass_final, bfs_cart.ux_mean, bfs_cart.uy_mean)
        @printf(io, "\nBFS ux mean delta: %.6e\n", r.ux_mean - bfs_cart.ux_mean)
        @printf(io, "BFS uy mean delta: %.6e\n", r.uy_mean - bfs_cart.uy_mean)
        println(io)
        println(io, "## Acceptance Gates")
        println(io)
        println(io, "| metric | gate |")
        println(io, "|---|---:|")
        println(io, "| Couette mass drift | `< 1e-8` |")
        println(io, "| Couette L2 / Linf | `< 1e-3` / `< 2e-3` |")
        println(io, "| Poiseuille band mass drift | `< 1e-8` |")
        println(io, "| Poiseuille band L2 / Linf | `< 2e-3` / `< 3e-3` |")
        println(io, "| Square obstacle mass drift | `< 1e-8` |")
        println(io, "| Square obstacle drag ratio | `0.85 < Fx_refined/Fx_coarse < 1.15` |")
        println(io, "| Square obstacle lift ratio | `< 1e-10` |")
        println(io, "| BFS/VFS mean velocity deltas | `|dux| < 5e-4`, `|duy| < 5e-5` |")
        println(io, "| BFS/VFS mass-final delta | `< 1.0` |")
        println(io)
        println(io, "Non-claims: no dynamic AMR, no native dx-local streaming, no subcycling, no cylinder validation in Phase P.")
    end
end

function assert_phase_p_gate(couette, vertical, horizontal, refined_square,
                             coarse_square, bfs, bfs_cart)
    c = couette.result
    @assert abs(c.mass_drift) < 1e-8
    @assert c.l2_error < 1e-3
    @assert c.linf_error < 2e-3

    for case in (vertical, horizontal)
        r = case.result
        @assert abs(r.mass_drift) < 1e-8
        @assert r.l2_error < 2e-3
        @assert r.linf_error < 3e-3
    end

    refined_mass_drift = refined_square.mass_final - refined_square.mass_initial
    coarse_mass_drift = coarse_square.mass_final - coarse_square.mass_initial
    drag_ratio = refined_square.Fx_drag / coarse_square.Fx_drag
    @assert abs(refined_mass_drift) < 1e-8
    @assert abs(coarse_mass_drift) < 1e-8
    @assert refined_square.Fx_drag > 0
    @assert coarse_square.Fx_drag > 0
    @assert abs(refined_square.Fy_drag / refined_square.Fx_drag) < 1e-10
    @assert abs(coarse_square.Fy_drag / coarse_square.Fx_drag) < 1e-10
    @assert 0.85 < drag_ratio < 1.15

    b = bfs.result
    @assert b.flow == :bfs
    @assert abs(b.ux_mean - bfs_cart.ux_mean) < 5e-4
    @assert abs(b.uy_mean - bfs_cart.uy_mean) < 5e-5
    @assert abs(b.mass_final - bfs_cart.mass_final) < 1.0
    return true
end

couette = run_couette_case()
vertical = run_poiseuille_band(; name="vertical band x=L/2",
                               Nx=24, Ny=16,
                               patch_i_range=11:14, patch_j_range=1:16)
horizontal = run_poiseuille_band(; name="horizontal band y=L/2",
                                 Nx=24, Ny=16,
                                 patch_i_range=1:24, patch_j_range=7:10)
refined_square = run_refined_square_drag()
coarse_square = run_coarse_square_drag()
bfs = run_bfs_case()
bfs_cart = run_cartesian_bfs(bfs.result.is_solid_leaf;
                             u_in=bfs.u_in, omega=bfs.omega,
                             steps=bfs.result.steps)

assert_phase_p_gate(couette, vertical, horizontal, refined_square,
                    coarse_square, bfs, bfs_cart)
plot_poiseuille_bands(vertical, horizontal)
plot_square_obstacle(refined_square, coarse_square)
plot_couette_and_bfs(couette, bfs, bfs_cart)
summary_path = joinpath(DATADIR, "voie_d_phase_p_summary.md")
write_summary(summary_path, couette, vertical, horizontal, refined_square,
              coarse_square, bfs, bfs_cart)

@info "wrote" path=joinpath(FIGDIR, "voie_d_poiseuille_bands.pdf")
@info "wrote" path=joinpath(FIGDIR, "voie_d_square_obstacle.pdf")
@info "wrote" path=joinpath(FIGDIR, "voie_d_couette_bfs.pdf")
@info "wrote" path=summary_path
