# Adiabatic walls of the thermal double-distribution solver (v0.7). The walls
# with no fixed temperature reflect the thermal populations specularly: a
# population whose source lies across such a wall is the mirror image
# (wall-normal velocity component flipped) of the one leaving the same wall
# row towards it. Zero normal heat flux, tangential flux kept. Before v0.7 the
# diagonal populations were bounced back, which reversed the wall-parallel
# flux on the wall rows: in pure conduction the profile was bent near the
# adiabatic walls (Nu - 1 = -1.1e-2 at N = 16 with Nu averaged over all rows).
#
# CPU Float64. Runtime: about a minute, most of it the N = 64 cavity.

using Test
using Kraken

@testset "Thermal adiabatic walls: specular reflection" begin
    lattice_c(lat) = lat isa D2Q9 ?
        (Kraken.velocities_x(lat), Kraken.velocities_y(lat)) :
        (Kraken.velocities_x(lat), Kraken.velocities_y(lat), Kraken.velocities_z(lat))

    # Reference pull for a box heated along x: a source across the x walls
    # gives the node's own opposite population (bounced value, turned into the
    # Dirichlet rule by the wall kernels); across any other wall the crossed
    # velocity components are mirrored and the source stays in the wall layer.
    function reference_pull(g_in, lat)
        dims = size(g_in)[1:end-1]
        D = length(dims)
        c = lattice_c(lat)
        opp = Kraken.opposite(lat)
        Q = size(g_in, D + 1)
        slot = Dict(ntuple(d -> c[d][q], D) => q for q in 1:Q)
        g_out = similar(g_in)
        for q in 1:Q, I in CartesianIndices(dims)
            src = ntuple(d -> I[d] - c[d][q], D)
            if !(1 <= src[1] <= dims[1])
                g_out[I, q] = g_in[I, opp[q]]
                continue
            end
            out = ntuple(d -> !(1 <= src[d] <= dims[d]), D)
            s = ntuple(d -> out[d] ? I[d] : src[d], D)
            cm = ntuple(d -> out[d] ? -c[d][q] : c[d][q], D)
            g_out[I, q] = g_in[CartesianIndex(s), slot[cm]]
        end
        return g_out
    end

    @testset "thermal stream kernels follow the mirror rule on every slot" begin
        g2 = rand(7, 5, 9)
        out2 = similar(g2)
        Kraken.stream_thermal_cavity_2d!(out2, g2, 7, 5)
        @test out2 == reference_pull(g2, D2Q9())
        # The fused cavity kernels pull through the same function.
        @test all(Kraken.stream_pull_thermal_node(g2, i, j, 7, 5) == Tuple(out2[i, j, :])
                  for i in 1:7, j in 1:5)
        g3 = rand(5, 4, 6, 19)
        out3 = similar(g3)
        Kraken.stream_thermal_cavity_3d!(out3, g3, 5, 4, 6)
        @test out3 == reference_pull(g3, D3Q19())
    end

    @testset "pure conduction: linear profile on every row" begin
        # Hot west and cold east half-way walls (x = 0.5 and x = N + 0.5, node
        # i at x = i), Ra = 0: the steady profile is T_i = 1 - (i - 0.5)/N on
        # every row, rows 1 and N included, and Nu = 1 on every row. Run until
        # the initial perturbation has decayed; what is left is round-off. In
        # the fused 2D kernels the round-off floor grows as N^2 (6.9e-14 at
        # N = 16, 2.5e-13 at N = 32, the same after 45000 and 90000 steps);
        # the separate stream/wall/collision kernels (Rayleigh-Benard driver,
        # 3D) stay near 3e-15. Gates are 4 to 7 times the measured floor
        # (2026-11-04, Apple-silicon Mac, CPU Float64).
        linear(N) = [1 - (i - 0.5) / N for i in 1:N, j in 1:N]
        rowdev(Temp, lin) = maximum(abs.(Temp .- lin); dims=1)   # one value per row

        for (N, steps, tol_T, tol_Nu) in ((16, 15000, 3e-13, 2e-12),
                                          (32, 45000, 1e-12, 5e-12))
            r = run_natural_convection_2d(; N, Ra=0.0, Pr=0.71, max_steps=steps)
            dev = rowdev(r.Temp, linear(N))
            rows = Kraken._hot_wall_nusselt_rows_2d(r.Temp, N, 1.0, 1.0)
            @test maximum(dev) < tol_T          # measured 6.9e-14 / 2.5e-13
            @test maximum(abs.(rows .- 1)) < tol_Nu  # measured 3.2e-13 / 1.2e-12
            @test abs(r.Nu - 1) < tol_Nu
            @info "Cavity conduction N=$N" max_dev=maximum(dev) row1=dev[1] rowN=dev[end] Nu_minus_1=r.Nu - 1
        end

        # Generalised-Newtonian driver, same thermal kernel (power law at rest).
        N = 16
        scales = gnf_reference_scales(; Ra=0.0, Pr=0.71, nu0=0.05, H=Float64(N),
                                      T_hot=1.0, T_cold=0.0)
        model = cavity_powerlaw(0.7; nu0=0.05, alpha=scales.alpha, H=N)
        rg = run_natural_convection_gnf_2d(; N, Ra=0.0, Pr=0.71, rheology=model,
                                           nu0=0.05, max_steps=15000)
        @test maximum(abs.(rg.Temp .- linear(N))) < 3e-13   # measured 6.9e-14
        @test abs(rg.Nu - 1) < 2e-12                        # measured 3.2e-13

        # Rayleigh-Benard driver, :horizontal orientation (the conduction
        # fallback of a west/east `.krk` case), rectangular box.
        rb = run_rayleigh_benard_2d(; Nx=16, Ny=8, Ra=0.0, Pr=0.71, max_steps=15000,
                                    orientation=:horizontal)
        lin_rb = [1 - (i - 0.5) / 16 for i in 1:16, j in 1:8]
        @test maximum(abs.(rb.Temp .- lin_rb)) < 1e-14      # measured 2.0e-15

        # 3D cube: adiabatic on the four faces normal to y and z.
        for (N, steps) in ((8, 5000), (12, 10000))
            r3 = run_natural_convection_3d(; N, Ra=0.0, Pr=0.71, max_steps=steps)
            lin3 = [1 - (i - 0.5) / N for i in 1:N, j in 1:N, k in 1:N]
            @test maximum(abs.(r3.Temp .- lin3)) < 2e-14    # measured 1.4e-15 / 3.1e-15
            @test abs(r3.Nu - 1) < 3e-13                    # measured 2.9e-14 / 5.2e-14
        end
    end

    @testset "AD thermal step equals the fused production kernel" begin
        # src/ad/ad_thermal_step.jl is the CPU copy that Enzyme differentiates;
        # with the half-way geometry (q = 1/2 on both heated walls) it must
        # give the fields of fused_natconv_step!, flow and thermal populations.
        N = 10
        p = Kraken.ad_natconv_params(; N=N, Ra=1e4, Pr=0.71)
        geom = Kraken.ad_cavity_wall_geometry(N, N, Float64(N); q_hot=0.5)
        wts = Kraken.weights(D2Q9())
        f = zeros(N, N, 9)
        g = zeros(N, N, 9)
        for j in 1:N, i in 1:N
            ux = 0.02 * sin(2pi * i / N) * cos(pi * j / N)
            uy = -0.015 * cos(pi * i / N) * sin(2pi * j / N)
            for q in 1:9
                f[i, j, q] = Kraken.feq_2d(Val(q), 1.0 + 0.01 * cos(i + 2j), ux, uy,
                                           ux^2 + uy^2) + 1e-4 * sin(3i + j + q)
                g[i, j, q] = wts[q] * (1 - (i - 0.5) / N + 0.05 * cos(2i + 5j + q))
            end
        end
        w = vcat(vec(f), vec(g))
        w_out = similar(w)
        f_out, g_out, Temp = similar(f), similar(g), zeros(N, N)
        worst = 0.0
        for step in 1:200
            Kraken.fused_natconv_step!(f_out, f, g_out, g, Temp, N, N, p.omega_f,
                                       p.omega_T, p.beta_g, p.T_ref, p.T_hot, p.T_cold)
            Kraken.ad_thermal_cut_step!(w_out, w, geom.q_wall, geom.q_wall, p)
            d = maximum(abs.(w_out .- vcat(vec(f_out), vec(g_out))))
            step == 1 && @test d < 1e-15            # measured 1.4e-17
            worst = max(worst, d)
            f, f_out = f_out, f
            g, g_out = g_out, g
            w, w_out = w_out, w
        end
        @test worst < 1e-13                         # measured 8.3e-16
    end

    @testset "Ra = 1e3 cavity: second-order hot-wall and cavity Nusselt" begin
        # De Vahl Davis cavity, Pr = 0.71. Hot-wall Nu: the all-rows mean
        # returned by the driver. Cavity Nu: 1 + H <u (T - T_cold)> / (alpha ΔT),
        # the mean x heat flux over the cavity (the conduction part of the
        # face-centred flux telescopes to 1). Measured 2026-11-04, CPU Float64
        # (each N run to steady Nu; doubling the steps moves the 7th digit):
        #   N        16        32        64
        #   wall  1.113794  1.116775  1.117533   order 1.97
        #   cavity 1.112804 1.116618  1.117511   order 2.09
        # With the former bounce-back walls the wall-row flux error was first
        # order. Gate: observed order >= 1.8.
        function cavity_point(N, steps)
            r = run_natural_convection_2d(; N, Ra=1e3, Pr=0.71, max_steps=steps)
            conv = sum(r.ux .* r.Temp) / N^2 * N / r.α
            return r.Nu, 1 + conv
        end
        pts = [cavity_point(16, 4000), cavity_point(32, 15000), cavity_point(64, 40000)]
        order(v) = log2((v[2] - v[1]) / (v[3] - v[2]))
        p_wall = order(first.(pts))
        p_cav = order(last.(pts))
        @test p_wall >= 1.8
        @test p_cav >= 1.8
        @info "Ra=1e3 orders" wall=first.(pts) cavity=last.(pts) p_wall p_cav
    end
end
