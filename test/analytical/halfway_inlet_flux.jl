# Discrete flow rate of the parabolic inlets between half-way walls (fix for
# v0.7). The channel walls sit half a spacing outside the first and last open
# rows, so the parabola must vanish there, and its row sum (the discrete flow
# rate) must equal H * u_mean exactly. Before the fix the zeros were put on
# rows 1 and H, which made the delivered flow rate low by 1/(H - 1). The Zou-He
# rebuild must then impose the profile on rows 1 and H too, and in 3D on the
# edge rows and planes next to the half-way walls (`halfway_walls`, last four
# testsets).
#
# Gates are analytical identities (round-off only) unless stated, CPU Float64.
# Runtime: about 25 s (3D part about 14 s).

using Test
using Random
using Kraken

@testset "Inlet flow rate (half-way walls)" begin
    rel(a, b) = abs(a - b) / abs(b)
    # Quadratic through rows k0, k0+1, k0+2 of `u`, evaluated at the real
    # row coordinate `k` (Lagrange form). The profile is a quadratic in the
    # row index, so this extrapolates it exactly up to round-off.
    function quad_at(u, k0, k)
        a, b, c = u[k0], u[k0 + 1], u[k0 + 2]
        t = k - k0
        return a * (t - 1) * (t - 2) / 2 - b * t * (t - 2) + c * t * (t - 1) / 2
    end

    @testset "half-way Poiseuille profile: flux and wall zeros" begin
        for H in (2, 3, 5, 8, 20, 33, 80, 101, 200, 401), u_mean in (1e-3, 0.02, 1.0)
            u = Kraken._halfway_poiseuille_profile(Float64, H, u_mean)
            @test length(u) == H
            # Discrete flow rate equals H * u_mean (gate 1e-12, round-off).
            @test rel(sum(u), H * u_mean) <= 1e-12
            # Symmetric and positive on every row (no zero on the end rows).
            @test all(>(0), u)
            @test maximum(abs.(u .- reverse(u))) <= 1e-15 * u_mean * H
            if H >= 3
                # Zero on both half-way walls, k = 1/2 and k = H + 1/2.
                @test abs(quad_at(u, 1, 0.5)) <= 1e-12 * u_mean
                @test abs(quad_at(u, H - 2, H + 0.5)) <= 1e-12 * u_mean
            end
        end
        # Float32 output: same values rounded once.
        u32 = Kraken._halfway_poiseuille_profile(Float32, 40, 0.02)
        u64 = Kraken._halfway_poiseuille_profile(Float64, 40, 0.02)
        @test eltype(u32) === Float32
        @test u32 == Float32.(u64)
        @test rel(Float64(sum(u32)), 40 * 0.02) <= 1e-6
        @test_throws ArgumentError Kraken._halfway_poiseuille_profile(Float64, 0, 1.0)
    end

    @testset "cylinder LI-BB inlet (ad_inlet_profile, same helper as run_cylinder_libb_2d)" begin
        u_in = 0.04
        for Ny in (16, 41, 80, 161)
            u = Kraken.ad_inlet_profile(Ny, u_in, :parabolic)
            # u_ref = (2/3) u_in is the mean used in Cd = 2 Fx / (u_ref^2 D).
            @test rel(sum(u), Ny * (2 / 3) * u_in) <= 1e-12
            @test u == Kraken._halfway_poiseuille_profile(Float64, Ny, (2 / 3) * u_in)
            @test abs(quad_at(u, 1, 0.5)) <= 1e-12 * u_in
            @test abs(quad_at(u, Ny - 2, Ny + 0.5)) <= 1e-12 * u_in
        end
        @test Kraken.ad_inlet_profile(10, u_in, :uniform) == fill(u_in, 10)
    end

    @testset "parabolic_face_profile_2d on step and obstacle channels" begin
        u_mean = 0.01
        geoms = (
            backward_facing_step_geometry_2d(; H_in=8, L_up=2, L_down=3),
            backward_facing_step_geometry_2d(; H_in=17, L_up=2, L_down=3),
            contraction_step_geometry_2d(; H_out=6, β_c=4, L_up=2, L_down=3),
            contraction_step_geometry_2d(; H_out=10, β_c=2, L_up=2, L_down=3),
            square_obstacle_channel_geometry_2d(; H=24, side=6, L_up=2, L_down=3),
            Kraken._logfv_cylinder_channel_geometry_2d(; radius=6),
        )
        for geom in geoms, face in (:west, :east)
            open = face === :west ? geom.inlet_open : geom.outlet_open
            i_face = face === :west ? 1 : geom.Nx
            H = length(open)
            u = parabolic_face_profile_2d(geom; face=face, mean_velocity=u_mean)
            @test length(u) == geom.Ny
            @test rel(sum(u[open]), H * u_mean) <= 1e-12
            @test all(iszero, u[setdiff(1:geom.Ny, open)])
            @test all(>(0), u[open])
            @test abs(quad_at(u, first(open), first(open) - 0.5)) <= 1e-12 * u_mean
            @test abs(quad_at(u, last(open) - 2, last(open) + 0.5)) <= 1e-12 * u_mean
            # Premise of the construction: the opening is bounded by half-way
            # walls (q = 1/2 on the links leaving its end rows, south = 5,
            # north = 3 in the D2Q9 ordering), either the domain edge or a
            # solid neighbour. The cylinder channel leaves its domain-edge
            # walls to the kernel's half-way bounce-back, outside q_wall.
            if geom.name !== :cylinder
                @test geom.q_wall[i_face, first(open), 5] == 0.5
                @test geom.q_wall[i_face, last(open), 3] == 0.5
            end
        end
        # Float32 path converts the Float64 values once.
        g = backward_facing_step_geometry_2d(; H_in=8, L_up=2, L_down=3)
        u32 = parabolic_face_profile_2d(g; mean_velocity=0.01, FT=Float32)
        @test eltype(u32) === Float32
        @test u32 == Float32.(parabolic_face_profile_2d(g; mean_velocity=0.01))
    end

    @testset "AD viscoelastic channel profile" begin
        for Ny in (24, 31, 80)
            u = Kraken.ad_ve_poiseuille_profile(Ny, 2e-4)
            @test rel(sum(u), Ny * 2e-4) <= 1e-12
            g = square_obstacle_channel_geometry_2d(; H=Ny, side=iseven(Ny) ? 6 : 5,
                                                    L_up=2, L_down=3)
            @test u == parabolic_face_profile_2d(g; mean_velocity=2e-4)
        end
    end

    # The profile only delivers its flow rate if the Zou-He rebuild imposes it on
    # every row, including rows 1 and Ny next to the half-way walls. Before the
    # fix those two rows were skipped and kept the bulk-kernel values.
    @testset "Zou-He x-face rebuild sets rows 1..Ny (half-way walls)" begin
        # One rebuild on a state at equilibrium with the profile. f_out starts as
        # NaN, so a row the rebuild skips is caught. nu = 1/6 gives s_plus = 1:
        # the rebuild collision then conserves mass as well as momentum, so the
        # imposed velocity or density is recovered to round-off (gate 1e-12).
        Nx, Ny, nu = 7, 12, 1 / 6
        s_p, s_m = trt_rates(nu; Λ=3 / 16)
        dens(f, i, j) = sum(f[i, j, q] for q in 1:9)
        velx(f, i, j) = (f[i, j, 2] - f[i, j, 4] + f[i, j, 6] - f[i, j, 7] -
                         f[i, j, 8] + f[i, j, 9]) / dens(f, i, j)
        vely(f, i, j) = (f[i, j, 3] - f[i, j, 5] + f[i, j, 6] + f[i, j, 7] -
                         f[i, j, 8] - f[i, j, 9]) / dens(f, i, j)
        for prof in (Kraken._halfway_poiseuille_profile(Float64, Ny, 0.03), fill(0.03, Ny))
            f_in = [Kraken.equilibrium(D2Q9(), 1.0, prof[j], 0.0, q)
                    for i in 1:Nx, j in 1:Ny, q in 1:9]
            # (spec, face column, :u imposes prof / :rho imposes 1) for each x-face
            # kernel: velocity and pressure, west and east, and the east kernels
            # that use west-pointing populations (reoriented blocks).
            cases = (
                (BCSpec2D(; west=ZouHeVelocity(prof), east=ZouHePressure(1.0)),
                 ((1, :u), (Nx, :rho))),
                (BCSpec2D(; west=ZouHePressure(1.0), east=ZouHeVelocity(prof)),
                 ((1, :rho), (Nx, :u))),
                (BCSpec2D(; east=ZouHeVelocity(prof, :west)), ((Nx, :u),)),
                (BCSpec2D(; east=ZouHePressure(1.0, :west)), ((Nx, :rho),)),
            )
            for (spec, checks) in cases, rates in (:uniform, :field)
                f_out = fill(NaN, Nx, Ny, 9)
                if rates === :uniform
                    apply_bc_rebuild_2d!(f_out, f_in, spec, nu, Nx, Ny)
                else
                    apply_bc_rebuild_2d!(f_out, f_in, spec, nu, Nx, Ny;
                                         sp_field=fill(s_p, Nx, Ny),
                                         sm_field=fill(s_m, Nx, Ny))
                end
                for (i, what) in checks
                    err = if what === :u
                        maximum(abs(velx(f_out, i, j) - prof[j]) for j in 1:Ny)
                    else
                        maximum(abs(dens(f_out, i, j) - 1) for j in 1:Ny)
                    end
                    @test err <= 1e-12
                    @test maximum(abs(vely(f_out, i, j)) for j in 1:Ny) <= 1e-12
                end
            end
        end
    end

    # Delivered flow rate in a steady flow: run_cylinder_libb_2d with a cylinder
    # too small to cover a node (R = 0.45) far downstream, velocity flow rate
    # Sum_j u_x on column i = 3 over Ny u_ref. Measured 2026-11-04 (CPU Float64):
    # parabolic inlet -0.89 / -0.32 / -0.081 % at Ny = 24 / 40 / 80 before the
    # fix, +0.032 / +0.012 / +0.0029 % after. Gate 5e-4 at every Ny, and the
    # residual must shrink as Ny grows.
    @testset "steady delivered flow rate, run_cylinder_libb_2d" begin
        function flow_rate_error(inlet, Ny, steps)
            r = run_cylinder_libb_2d(; Nx=60, Ny=Ny, cx=45, cy=Ny ÷ 2, radius=0.45,
                                     u_in=0.04, ν=0.1, max_steps=steps, avg_window=1,
                                     inlet=inlet)
            return sum(r.ux[3, :]) / (Ny * r.u_ref) - 1
        end
        steps = Dict(24 => 12_000, 40 => 16_000, 80 => 24_000)
        e = Dict{Int,Float64}()
        for Ny in (24, 40, 80)
            e[Ny] = flow_rate_error(:parabolic, Ny, steps[Ny])
            # Within the gate, and steady: the value moves by less than 1e-6
            # over the last 4000 steps.
            drift = abs(flow_rate_error(:parabolic, Ny, steps[Ny] - 4000) - e[Ny])
            @test abs(e[Ny]) < 5e-4 && drift < 1e-6
        end
        @test abs(e[80]) < abs(e[40]) < abs(e[24]) < 5e-4

        # Uniform inlet: the rebuild imposes u_in on every row (testset above),
        # yet the velocity flow rate on column 3 is +0.43 / +0.26 / +0.13 % at
        # Ny = 24 / 40 / 80, about +0.10/Ny (it was -7.9 / -4.7 / -2.4 % before
        # the fix). A uniform velocity against a no-slip wall is singular at the
        # corner: the density on the two corner nodes of the inlet column is
        # about 6 % above the outlet value at Ny = 40 and relaxes over the next
        # cells, so Sum_j u_x exceeds the target by a first-order amount. Gates:
        # excess below 0.15/Ny at each Ny, and first-order decrease from 40 to 80.
        eu = Dict(Ny => flow_rate_error(:uniform, Ny, steps[Ny]) for Ny in (24, 40, 80))
        for Ny in (24, 40, 80)
            @test abs(eu[Ny]) < 0.15 / Ny
        end
        @test 0 < eu[80] < 0.6 * eu[40]
    end

    # 3D: the x-face rebuild covers the edge rows and planes of the lateral
    # faces named in `halfway_walls` (default none, the interior j = 2..Ny-1,
    # k = 2..Nz-1 only).
    @testset "3D Zou-He x-face rebuild with halfway_walls" begin
        Nx, Ny, Nz, nu = 5, 6, 7, 0.1
        s_p, s_m = trt_rates(nu; Λ=3 / 16)
        cx = Kraken.velocities_x(D3Q19())
        cy = Kraken.velocities_y(D3Q19())
        cz = Kraken.velocities_z(D3Q19())
        opp = [findfirst(p -> (cx[p], cy[p], cz[p]) == (-cx[q], -cy[q], -cz[q]), 1:19)
               for q in 1:19]
        all4 = (:south, :north, :bottom, :top)
        dens(f, i, j, k) = sum(f[i, j, k, q] for q in 1:19)
        velx(f, i, j, k) = sum(cx[q] * f[i, j, k, q] for q in 1:19) / dens(f, i, j, k)
        rebuild!(f_out, f_in, spec, walls, rates) = rates === :uniform ?
            apply_bc_rebuild_3d!(f_out, f_in, spec, nu, Nx, Ny, Nz; halfway_walls=walls) :
            apply_bc_rebuild_3d!(f_out, f_in, spec, nu, Nx, Ny, Nz; halfway_walls=walls,
                                 sp_field=fill(s_p, Nx, Ny, Nz),
                                 sm_field=fill(s_m, Nx, Ny, Nz))

        # (a) One rebuild of a state at equilibrium with the inlet profile, all
        # four lateral faces walls. f_out starts as NaN, so a skipped node is
        # caught. Gate 1e-12 (round-off): the Zou-He closure sets the density
        # and x-momentum exactly and the TRT collision conserves both, so every
        # west node, edges and corners included, carries u_x = profile and
        # every east node rho = 1.
        sy = Kraken._halfway_poiseuille_profile(Float64, Ny, 2 / 3)
        sz = Kraken._halfway_poiseuille_profile(Float64, Nz, 2 / 3)
        py = Kraken._halfway_poiseuille_profile(Float64, Ny, (2 / 3) * 0.04)
        for prof in ([0.04 * sy[j] * sz[k] for j in 1:Ny, k in 1:Nz],   # :parabolic
                     [py[j] for j in 1:Ny, k in 1:Nz]),                 # :parabolic_y
            rates in (:uniform, :field)
            f_in = [Kraken.equilibrium(D3Q19(), 1.0, prof[j, k], 0.0, 0.0, q)
                    for i in 1:Nx, j in 1:Ny, k in 1:Nz, q in 1:19]
            f_out = fill(NaN, Nx, Ny, Nz, 19)
            spec = BCSpec3D(; west=ZouHeVelocity(prof), east=ZouHePressure(1.0))
            rebuild!(f_out, f_in, spec, all4, rates)
            @test maximum(abs(velx(f_out, 1, j, k) - prof[j, k])
                          for j in 1:Ny, k in 1:Nz) <= 1e-12
            @test maximum(abs(dens(f_out, Nx, j, k) - 1) for j in 1:Ny, k in 1:Nz) <= 1e-12
        end

        # (b) Which populations the rim reads, on a random state. Reference: the
        # default rebuild (interior only) on a copy padded with one ghost layer
        # beyond each listed wall, where ghost population q holds the opposite
        # population of the node it streams into: the pull-scheme form of
        # half-way bounce-back used by PullHalfwayBB_3D. Same arithmetic on the
        # same values, so the gate is round-off; nodes outside the listed walls
        # must stay NaN in both.
        function ghost_reference(f_in, prof, walls, rates)
            ly, hy = Int(:south in walls), Int(:north in walls)
            lz, hz = Int(:bottom in walls), Int(:top in walls)
            Nyp, Nzp = Ny + ly + hy, Nz + lz + hz
            real_j, real_k = (1 + ly):(Ny + ly), (1 + lz):(Nz + lz)
            g = fill(NaN, Nx, Nyp, Nzp, 19)
            g[:, real_j, real_k, :] .= f_in
            for kp in 1:Nzp, jp in 1:Nyp, i in 1:Nx, q in 1:19
                (jp in real_j && kp in real_k) && continue
                it, jt, kt = i + cx[q], jp - ly + cy[q], kp - lz + cz[q]
                (1 <= it <= Nx && 1 <= jt <= Ny && 1 <= kt <= Nz) || continue
                g[i, jp, kp, q] = f_in[it, jt, kt, opp[q]]
            end
            profp = zeros(Nyp, Nzp)
            profp[real_j, real_k] .= prof
            gspec = BCSpec3D(; west=ZouHeVelocity(profp), east=ZouHePressure(1.0))
            out = fill(NaN, Nx, Nyp, Nzp, 19)
            if rates === :uniform
                apply_bc_rebuild_3d!(out, g, gspec, nu, Nx, Nyp, Nzp)
            else
                apply_bc_rebuild_3d!(out, g, gspec, nu, Nx, Nyp, Nzp;
                                     sp_field=fill(s_p, Nx, Nyp, Nzp),
                                     sm_field=fill(s_m, Nx, Nyp, Nzp))
            end
            return out[:, real_j, real_k, :]
        end
        same(a, b) = isnan.(a) == isnan.(b) &&
                     maximum(abs, filter(!isnan, a .- b); init=0.0) <= 1e-14
        rng = Random.Xoshiro(115)
        f_in = 0.02 .+ 0.08 .* rand(rng, Nx, Ny, Nz, 19)
        prof = 0.03 .* rand(rng, Ny, Nz)
        spec = BCSpec3D(; west=ZouHeVelocity(prof), east=ZouHePressure(1.0))
        for walls in (all4, (:north, :bottom)), rates in (:uniform, :field)
            f_out = fill(NaN, Nx, Ny, Nz, 19)
            rebuild!(f_out, f_in, spec, walls, rates)
            @test same(f_out, ghost_reference(f_in, prof, walls, rates))
        end

        @test_throws ArgumentError apply_bc_rebuild_3d!(similar(f_in), f_in, spec, nu,
                                                        Nx, Ny, Nz; halfway_walls=(:west,))
        @test_throws ArgumentError apply_bc_rebuild_3d!(similar(f_in), f_in, spec, nu,
                                                        Nx, Ny, Nz; halfway_walls=all4,
                                                        apply_transverse=true)
    end

    # Delivered flow rate in a steady 3D duct: run_sphere_libb_3d with a sphere
    # of radius 0.3 centred between nodes, which covers no node and cuts no link
    # (empty duct), Nx = 24, nu = 0.1, u_in = 0.04, N = Ny = Nz. Velocity flow
    # rate Sum u_x on the plane i = 3 over N^2 u_ref. Measured 2026-11-04 (CPU
    # Float64) at N = 8 / 16, before the fix (profile with its zeros on the
    # edge nodes, rim not rebuilt) and after:
    #   :parabolic    -24.8 / -12.5 %  ->  +0.47 / +0.12 %
    #   :parabolic_y  -33.7 / -17.4 %  ->  +1.25 / +0.56 %
    # The value moves by less than 1.1e-6 between 1500 and 2000 steps.
    @testset "steady delivered flow rate, run_sphere_libb_3d" begin
        function flow_rate_error_3d(inlet, N, steps)
            r = run_sphere_libb_3d(; Nx=24, Ny=N, Nz=N, cx=18.5, cy=N / 2 - 0.5,
                                   cz=N / 2 - 0.5, radius=0.3, u_in=0.04, ν=0.1,
                                   max_steps=steps, avg_window=1, inlet=inlet)
            return sum(r.ux[3, :, :]) / (N^2 * r.u_ref) - 1
        end
        e = Dict{Tuple{Symbol,Int},Float64}()
        for inlet in (:parabolic, :parabolic_y), N in (8, 16)
            e[(inlet, N)] = flow_rate_error_3d(inlet, N, 2000)
            # Steady: the value moves by less than 1e-5 (under 1 % of the
            # smallest residual gated below) over the last 500 steps.
            drift = abs(flow_rate_error_3d(inlet, N, 1500) - e[(inlet, N)])
            gate = inlet === :parabolic ? 0.5 / N^2 : 0.15 / N
            @test 0 < e[(inlet, N)] < gate && drift < 1e-5
        end
        # :parabolic vanishes on all four walls, so the residual is a
        # second-order discretisation error (measured 0.30/N^2): gate 0.5/N^2
        # above, and a decrease by more than 2.8x from N = 8 to 16 (4x expected,
        # 3.9x measured).
        @test 0 < e[(:parabolic, 16)] < 0.35 * e[(:parabolic, 8)]
        # :parabolic_y is uniform in z, so u_x is not zero against the bottom and
        # top walls: the corner singularity of the 2D uniform inlet above, a
        # first-order excess (measured 0.10/N and 0.09/N). Gate 0.15/N above and
        # a first-order decrease (2.2x measured).
        @test 0 < e[(:parabolic_y, 16)] < 0.6 * e[(:parabolic_y, 8)]
    end
end
