using Test
using Kraken

@testset "Thermal DDF" begin

    @testset "Heat conduction 1D" begin
        # Pure conduction (no flow): linear temperature profile at steady state
        # Hot wall (j=1) = 1.0, cold wall (j=Ny) = 0.0, periodic x
        Nx, Ny = 4, 32
        α = 0.1  # thermal diffusivity
        ω_T = Float64(1.0 / (3.0 * α + 0.5))

        # Initialize thermal populations to uniform T=0.5
        g_in  = zeros(Float64, Nx, Ny, 9)
        g_out = zeros(Float64, Nx, Ny, 9)
        Temp  = zeros(Float64, Nx, Ny)
        ux    = zeros(Float64, Nx, Ny)  # no flow
        uy    = zeros(Float64, Nx, Ny)

        w = Kraken.weights(D2Q9())
        for j in 1:Ny, i in 1:Nx, q in 1:9
            g_in[i, j, q] = w[q] * 0.5
        end
        copy!(g_out, g_in)
        is_solid = zeros(Bool, Nx, Ny)

        for step in 1:5000
            Kraken.stream_periodic_x_wall_y_2d!(g_out, g_in, Nx, Ny)
            Kraken.apply_fixed_temp_south_2d!(g_out, 1.0, Nx)
            Kraken.apply_fixed_temp_north_2d!(g_out, 0.0, Nx, Ny)
            Kraken.collide_thermal_2d!(g_out, ux, uy, ω_T)
            g_in, g_out = g_out, g_in
        end

        Kraken.compute_temperature_2d!(Temp, g_in)
        T_profile = Temp[2, :]

        # Analytical: linear from T_hot at wall (y=0.5) to T_cold at wall (y=Ny+0.5)
        T_analytical = [1.0 - (j - 0.5) / Ny for j in 1:Ny]

        err = maximum(abs.(T_profile .- T_analytical))
        # Measured 2026-11-03 (CPU Float64): 1.3e-9 with half-way walls; the
        # former on-node walls gave 1.6e-2 against this half-way profile.
        @test err < 1e-7
        @info "Heat conduction: max error = $(err)"
    end

    @testset "Half-way Dirichlet walls: pure conduction to round-off" begin
        # Each pair of opposite Dirichlet faces is run alone, with the other
        # axes periodic, so no adiabatic wall enters. The walls sit half a
        # spacing outside the first and last nodes along the wall axis
        # (x = 0.5 and x = N + 0.5 with node i at x = i), so the steady profile
        # is T_i = T_a + (T_b - T_a) (i - 0.5) / N and the second-order wall
        # gradient (-8 T_a + 9 T_1 - T_2) / 3 gives Nu = 1, both to round-off.

        # Pull streaming on CPU arrays: periodic on every axis except
        # `wall_axis`, where a population that would come from outside the box
        # is the node's own opposite population (the bounce-back convention of
        # the library streaming kernels).
        function _stream_one_wall_axis!(g_out, g_in, c, opp, wall_axis)
            dims = size(g_in)[1:end-1]
            D = Val(ndims(g_in) - 1)
            for q in axes(g_in, ndims(g_in)), I in CartesianIndices(dims)
                src = ntuple(d -> I[d] - c[d][q], D)
                if 1 <= src[wall_axis] <= dims[wall_axis]
                    s = ntuple(d -> d == wall_axis ? src[d] : mod1(src[d], dims[d]), D)
                    g_out[I, q] = g_in[CartesianIndex(s), q]
                else
                    g_out[I, q] = g_in[I, opp[q]]
                end
            end
            return g_out
        end

        T_a, T_b = 0.8, 0.2
        ω_T = 1.0 / (3.0 * 0.1 + 0.5)   # α = 0.1
        N_wall, N_side, steps = 8, 3, 3000

        function _run_pair(lattice, wall_axis, apply_a!, apply_b!)
            D = lattice isa D2Q9 ? 2 : 3
            dims = ntuple(d -> d == wall_axis ? N_wall : N_side, D)
            Q = lattice isa D2Q9 ? 9 : 19
            c = D == 2 ? (Kraken.velocities_x(lattice), Kraken.velocities_y(lattice)) :
                         (Kraken.velocities_x(lattice), Kraken.velocities_y(lattice),
                          Kraken.velocities_z(lattice))
            opp = Kraken.opposite(lattice)
            w = Kraken.weights(lattice)
            g_in = zeros(Float64, dims..., Q)
            for q in 1:Q, I in CartesianIndices(dims)
                g_in[I, q] = w[q] * 0.5
            end
            g_out = copy(g_in)
            u = ntuple(_ -> zeros(Float64, dims...), D)
            Temp = zeros(Float64, dims...)
            for step in 1:steps
                _stream_one_wall_axis!(g_out, g_in, c, opp, wall_axis)
                apply_a!(g_out)
                apply_b!(g_out)
                if D == 2
                    Kraken.collide_thermal_2d!(g_out, u..., ω_T)
                else
                    Kraken.collide_thermal_3d!(g_out, u..., ω_T)
                end
                g_in, g_out = g_out, g_in
            end
            D == 2 ? Kraken.compute_temperature_2d!(Temp, g_in) :
                     Kraken.compute_temperature_3d!(Temp, g_in)
            return Temp
        end

        cases = [
            ("2D west/east", D2Q9(), 1,
             g -> apply_fixed_temp_west_2d!(g, T_a, N_side),
             g -> apply_fixed_temp_east_2d!(g, T_b, N_wall, N_side)),
            ("2D south/north", D2Q9(), 2,
             g -> apply_fixed_temp_south_2d!(g, T_a, N_side),
             g -> apply_fixed_temp_north_2d!(g, T_b, N_side, N_wall)),
            ("3D west/east", D3Q19(), 1,
             g -> apply_fixed_temp_west_3d!(g, T_a, N_side, N_side),
             g -> apply_fixed_temp_east_3d!(g, T_b, N_wall, N_side, N_side)),
            ("3D south/north", D3Q19(), 2,
             g -> apply_fixed_temp_south_3d!(g, T_a, N_side, N_side),
             g -> apply_fixed_temp_north_3d!(g, T_b, N_side, N_wall, N_side)),
            ("3D bottom/top", D3Q19(), 3,
             g -> apply_fixed_temp_bottom_3d!(g, T_a, N_side, N_side),
             g -> apply_fixed_temp_top_3d!(g, T_b, N_side, N_side, N_wall)),
        ]
        for (name, lattice, wall_axis, apply_a!, apply_b!) in cases
            Temp = _run_pair(lattice, wall_axis, apply_a!, apply_b!)
            idx = i -> CartesianIndex(ntuple(d -> d == wall_axis ? i : 1, ndims(Temp)))
            T_lin = i -> T_a + (T_b - T_a) * (i - 0.5) / N_wall
            dev = maximum(abs(Temp[I] - T_lin(I[wall_axis])) for I in CartesianIndices(Temp))
            T1, T2 = Temp[idx(1)], Temp[idx(2)]
            Nu = -N_wall * (-8 * T_a + 9 * T1 - T2) / 3 / (T_a - T_b)
            @testset "$name" begin
                @test dev < 1e-12
                @test abs(Nu - 1) < 1e-10
                # Linear extrapolation of the first two nodes to x = 0.5 is
                # the wall value.
                @test abs(1.5 * T1 - 0.5 * T2 - T_a) < 1e-12
            end
        end
    end

    @testset "Adjacent Dirichlet faces: corner and edge slots flipped once" begin
        # Reference built from the lattice velocities alone: a slot whose link
        # comes from outside the box through one or more Dirichlet faces gets
        # g = -g0 + 2 w_q T_link, with T_link the mean of the temperatures of
        # the Dirichlet faces it crosses; every other slot is unchanged.
        function _expected_fixed_temp(g0, lattice, temps)
            D = ndims(g0) - 1
            dims = size(g0)[1:D]
            c = D == 2 ? (Kraken.velocities_x(lattice), Kraken.velocities_y(lattice)) :
                         (Kraken.velocities_x(lattice), Kraken.velocities_y(lattice),
                          Kraken.velocities_z(lattice))
            w = Kraken.weights(lattice)
            faces = ((:west, 1, true), (:east, 1, false), (:south, 2, true),
                     (:north, 2, false), (:bottom, 3, true), (:top, 3, false))
            g = copy(g0)
            for q in axes(g0, D + 1), I in CartesianIndices(dims)
                T_crossed = Float64[]
                for (face, d, low) in faces
                    d <= D || continue
                    T_face = get(temps, face, nothing)
                    T_face === nothing && continue
                    src = I[d] - c[d][q]
                    (low ? src < 1 : src > dims[d]) && push!(T_crossed, T_face)
                end
                isempty(T_crossed) && continue
                g[I, q] = -g0[I, q] + 2 * w[q] * sum(T_crossed) / length(T_crossed)
            end
            return g
        end

        @testset "2D kernels against the reference" begin
            Nx, Ny = 5, 4
            g0 = rand(Float64, Nx, Ny, 9)
            per_face = Dict(
                :west  => (g, t) -> apply_fixed_temp_west_2d!(g, t, Ny),
                :east  => (g, t) -> apply_fixed_temp_east_2d!(g, t, Nx, Ny),
                :south => (g, t) -> apply_fixed_temp_south_2d!(g, t, Nx),
                :north => (g, t) -> apply_fixed_temp_north_2d!(g, t, Nx, Ny))
            for (face, t) in ((:west, 0.9), (:east, 0.1), (:south, 0.7), (:north, 0.3))
                g_one = copy(g0)
                per_face[face](g_one, t)
                g_all = copy(g0)
                apply_fixed_temp_2d!(g_all, Nx, Ny; NamedTuple{(face,)}((t,))...)
                @test g_all == g_one
                @test maximum(abs.(g_all .- _expected_fixed_temp(g0, D2Q9(), Dict(face => t)))) < 1e-15
            end
            for temps in ((; west=0.9, east=0.1, south=0.7, north=0.3),
                          (; west=0.9, south=0.7),
                          (; east=0.1, north=0.3),
                          (; west=0.9, north=0.3, east=0.1))
                g = copy(g0)
                apply_fixed_temp_2d!(g, Nx, Ny; temps...)
                g_ref = _expected_fixed_temp(g0, D2Q9(), Dict(pairs(temps)))
                @test maximum(abs.(g .- g_ref)) < 1e-15
            end
            # The four corner slots with all faces Dirichlet: mean temperature.
            g = copy(g0)
            apply_fixed_temp_2d!(g, Nx, Ny; west=0.9, east=0.1, south=0.7, north=0.3)
            @test g[1, 1, 6]   ≈ -g0[1, 1, 6]   + (0.9 + 0.7) / 36 atol=1e-15
            @test g[1, Ny, 9]  ≈ -g0[1, Ny, 9]  + (0.9 + 0.3) / 36 atol=1e-15
            @test g[Nx, 1, 7]  ≈ -g0[Nx, 1, 7]  + (0.1 + 0.7) / 36 atol=1e-15
            @test g[Nx, Ny, 8] ≈ -g0[Nx, Ny, 8] + (0.1 + 0.3) / 36 atol=1e-15
        end

        @testset "3D kernels against the reference" begin
            Nx, Ny, Nz = 4, 5, 3
            g0 = rand(Float64, Nx, Ny, Nz, 19)
            per_face = Dict(
                :west   => (g, t) -> apply_fixed_temp_west_3d!(g, t, Ny, Nz),
                :east   => (g, t) -> apply_fixed_temp_east_3d!(g, t, Nx, Ny, Nz),
                :south  => (g, t) -> apply_fixed_temp_south_3d!(g, t, Nx, Nz),
                :north  => (g, t) -> apply_fixed_temp_north_3d!(g, t, Nx, Ny, Nz),
                :bottom => (g, t) -> apply_fixed_temp_bottom_3d!(g, t, Nx, Ny),
                :top    => (g, t) -> apply_fixed_temp_top_3d!(g, t, Nx, Ny, Nz))
            all_temps = (; west=0.9, east=0.1, south=0.7, north=0.3, bottom=0.6, top=0.2)
            for face in keys(all_temps)
                t = all_temps[face]
                g_one = copy(g0)
                per_face[face](g_one, t)
                g_all = copy(g0)
                apply_fixed_temp_3d!(g_all, Nx, Ny, Nz; NamedTuple{(face,)}((t,))...)
                @test g_all == g_one
                @test maximum(abs.(g_all .- _expected_fixed_temp(g0, D3Q19(), Dict(face => t)))) < 1e-15
            end
            for temps in (all_temps,
                          (; west=0.9, south=0.7, top=0.2),
                          (; east=0.1, north=0.3, bottom=0.6),
                          (; south=0.7, bottom=0.6, top=0.2))
                g = copy(g0)
                apply_fixed_temp_3d!(g, Nx, Ny, Nz; temps...)
                g_ref = _expected_fixed_temp(g0, D3Q19(), Dict(pairs(temps)))
                @test maximum(abs.(g .- g_ref)) < 1e-15
            end
        end

        T_w, T_c = 0.8, 0.2
        ω_T = 1.0 / (3.0 * 0.1 + 0.5)   # α = 0.1

        @testset "2D square, hot west and south, cold east and north" begin
            # Pure conduction (buoyancy off, u = 0) with the library streaming
            # kernel. Swapping x and y maps the boundary conditions onto
            # themselves, so the steady field is symmetric about the diagonal
            # and, by the maximum principle, lies between the wall temperatures.
            N = 12
            w = Kraken.weights(D2Q9())
            g_in = zeros(Float64, N, N, 9)
            for q in 1:9
                g_in[:, :, q] .= w[q] * 0.5
            end
            g_out = copy(g_in)
            u0 = zeros(Float64, N, N)
            Temp = zeros(Float64, N, N)
            Temp_prev = zeros(Float64, N, N)
            for step in 1:6000
                stream_2d!(g_out, g_in, N, N)
                apply_fixed_temp_2d!(g_out, N, N; west=T_w, south=T_w, east=T_c, north=T_c)
                collide_thermal_2d!(g_out, u0, u0, ω_T)
                g_in, g_out = g_out, g_in
            end
            compute_temperature_2d!(Temp_prev, g_in)
            stream_2d!(g_out, g_in, N, N)
            apply_fixed_temp_2d!(g_out, N, N; west=T_w, south=T_w, east=T_c, north=T_c)
            collide_thermal_2d!(g_out, u0, u0, ω_T)
            compute_temperature_2d!(Temp, g_out)
            @test maximum(abs.(Temp .- Temp_prev)) < 1e-13          # steady
            @test maximum(abs.(Temp .- permutedims(Temp))) < 1e-13  # diagonal symmetry
            @test minimum(Temp) >= T_c - 1e-13
            @test maximum(Temp) <= T_w + 1e-13
            @test Temp[1, N] ≈ 0.5 * (T_w + T_c) atol=1e-13           # off-diagonal corners
        end

        @testset "3D cube, hot west, south and bottom, cold east, north and top" begin
            # Each swap of two axes maps the boundary conditions onto
            # themselves; the twelve edges are all shared by two Dirichlet faces.
            N = 6
            w = Kraken.weights(D3Q19())
            g_in = zeros(Float64, N, N, N, 19)
            for q in 1:19
                g_in[:, :, :, q] .= w[q] * 0.5
            end
            g_out = copy(g_in)
            u0 = zeros(Float64, N, N, N)
            Temp = zeros(Float64, N, N, N)
            Temp_prev = zeros(Float64, N, N, N)
            bc!(g) = apply_fixed_temp_3d!(g, N, N, N; west=T_w, south=T_w, bottom=T_w,
                                          east=T_c, north=T_c, top=T_c)
            for step in 1:3000
                stream_3d!(g_out, g_in, N, N, N)
                bc!(g_out)
                collide_thermal_3d!(g_out, u0, u0, u0, ω_T)
                g_in, g_out = g_out, g_in
            end
            compute_temperature_3d!(Temp_prev, g_in)
            stream_3d!(g_out, g_in, N, N, N)
            bc!(g_out)
            collide_thermal_3d!(g_out, u0, u0, u0, ω_T)
            compute_temperature_3d!(Temp, g_out)
            @test maximum(abs.(Temp .- Temp_prev)) < 1e-13
            @test maximum(abs.(Temp .- permutedims(Temp, (2, 1, 3)))) < 1e-13
            @test maximum(abs.(Temp .- permutedims(Temp, (3, 2, 1)))) < 1e-13
            @test maximum(abs.(Temp .- permutedims(Temp, (1, 3, 2)))) < 1e-13
            @test minimum(Temp) >= T_c - 1e-13
            @test maximum(Temp) <= T_w + 1e-13
        end
    end

    @testset "Pure conduction in the cavity drivers (Ra = 0)" begin
        # Hot west and cold east half-way walls, adiabatic bounce-back on the
        # other walls. The profile is not linear to round-off here: bounce-back
        # of the diagonal populations at an adiabatic wall does not carry the
        # wall-parallel gradient exactly, which bends the profile near the
        # adiabatic walls (row 1 deviates by 2e-3 at N = 16, 1e-3 at N = 32).
        # That floor is second order in the bulk. Measured 2026-11-03, CPU
        # Float64; the former on-node walls (H = N - 1 profile, 3-point Nu)
        # gave Nu - 1 = 7.2e-2 (2D, N = 16) and 1.07e-1 (3D, N = 12), and a
        # mid-row deviation of 3.1e-2 from the half-way profile.
        r16 = run_natural_convection_2d(; N=16, Ra=0.0, Pr=0.71, max_steps=8000)
        r32 = run_natural_convection_2d(; N=32, Ra=0.0, Pr=0.71, max_steps=20000)
        middev(r, N) = maximum(abs(r.Temp[i, N ÷ 2] - (1 - (i - 0.5) / N)) for i in 1:N)
        d16, d32 = middev(r16, 16), middev(r32, 32)
        @test abs(r16.Nu - 1) < 1e-3     # measured 3.1e-4
        @test abs(r32.Nu - 1) < 1e-4     # measured 4.9e-6
        @test d16 < 1e-4                 # measured 5.2e-5
        @test d16 / d32 > 3.5            # measured 4.04: second order
        @info "Cavity pure conduction: Nu-1 = $(r16.Nu - 1) (N=16), $(r32.Nu - 1) (N=32); " *
              "mid-row deviation $(d16), $(d32)"

        r3 = run_natural_convection_3d(; N=12, Ra=0.0, Pr=0.71, max_steps=6000)
        d3 = maximum(abs(r3.Temp[i, 6, 6] - (1 - (i - 0.5) / 12)) for i in 1:12)
        @test abs(r3.Nu - 1) < 3e-3      # measured 1.40e-3
        @test d3 < 4e-4                  # measured 1.78e-4
        @info "Box pure conduction: Nu-1 = $(r3.Nu - 1), mid-line deviation $(d3) (N=12)"
    end

    @testset "Rayleigh-Bénard onset" begin
        # Below critical Ra (~1708): no convection, linear T profile
        # Above critical Ra: convection rolls develop
        result_sub = run_rayleigh_benard_2d(; Nx=64, Ny=16, Ra=1000.0, max_steps=10000)
        result_sup = run_rayleigh_benard_2d(; Nx=64, Ny=16, Ra=5000.0, max_steps=20000)

        # Sub-critical: velocity should remain very small
        max_uy_sub = maximum(abs.(result_sub.uy))

        # Super-critical: vertical velocity should develop (convection)
        max_uy_sup = maximum(abs.(result_sup.uy))

        @test max_uy_sup > max_uy_sub * 5  # convection much stronger above Ra_c
        @test !any(isnan, result_sub.Temp)
        @test !any(isnan, result_sup.Temp)

        @info "RB sub-critical (Ra=1000): max|uy| = $(round(max_uy_sub, sigdigits=3))"
        @info "RB super-critical (Ra=5000): max|uy| = $(round(max_uy_sup, sigdigits=3))"
    end

    @testset "ν(T) Arrhenius kernel" begin
        # Verify the Arrhenius viscosity kernel runs without NaN/Inf
        # Use a simple channel with temperature gradient
        Nx, Ny = 4, 32
        ν_ref = 0.05; T_ref = 1.0; A_arr = 0.5

        config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν_ref, u_lid=0.0, max_steps=0)
        state = initialize_2d(config)
        f_in, f_out = state.f_in, state.f_out

        # Setup temperature field: linear 1.2 (bottom) to 0.8 (top)
        Temp = zeros(Float64, Nx, Ny)
        for j in 1:Ny
            Temp[:, j] .= 1.2 - 0.4 * (j - 1) / (Ny - 1)
        end

        for step in 1:500
            Kraken.stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
            collide_boussinesq_vt_2d!(f_out, Temp, state.is_solid,
                                      ν_ref, T_ref, A_arr, 0.0)
            f_in, f_out = f_out, f_in
        end

        Kraken.compute_macroscopic_2d!(state.ρ, state.ux, state.uy, f_in)
        @test !any(isnan, Array(state.ρ))
        @test all(isfinite, Array(state.ux))
        @info "ν(T) Arrhenius kernel: stable after 500 steps"
    end

    @testset "Natural convection cavity (Rc=1)" begin
        # De Vahl Davis benchmark: Ra=1e3, Pr=0.71, Nu ≈ 1.118 (cavity average; hot-wall Nu0 = 1.117)
        result = run_natural_convection_2d(; N=64, Ra=1e3, Pr=0.71, Rc=1.0,
                                             max_steps=30000)

        @test !any(isnan, result.Temp)
        @test !any(isnan, result.ux)
        @test !any(isnan, result.uy)

        Nu_ref = 1.118  # De Vahl Davis (1983)
        rel_err = abs(result.Nu - Nu_ref) / Nu_ref
        # Measured 2026-11-03 (CPU Float64): Nu = 1.11851, +0.045 %. The former
        # on-node walls gave 1.14226 (+2.17 %).
        @test rel_err < 0.005
        @info "Natural convection Ra=1e3: Nu = $(round(result.Nu, digits=4)) " *
              "(ref = $Nu_ref, error = $(round(100*rel_err, digits=2))%)"
    end

    @testset "Modified Arrhenius kernel (Rc>1)" begin
        # Verify stability of modified Arrhenius kernel with Rc=10
        result = run_natural_convection_2d(; N=32, Ra=1e3, Pr=0.71, Rc=10.0,
                                             max_steps=10000)

        @test !any(isnan, result.Temp)
        @test !any(isnan, result.ux)
        @test result.Nu > 1.0  # should be > conduction limit
        @info "Natural convection Ra=1e3, Rc=10: Nu = $(round(result.Nu, digits=4))"
    end
end
