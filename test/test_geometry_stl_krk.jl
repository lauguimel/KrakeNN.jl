# KRK-GEO — end-to-end test for the Cartesian immersed-boundary STL path.
#
# An `Obstacle … stl(file=…)` declared in a .krk file must voxelize into the
# same is_solid mask as the analytical `Obstacle … { (x-cx)^2+(y-cy)^2 <= R^2 }`,
# and `run_simulation` must complete and reproduce the analytical-cylinder flow.
#
# This test also guards the M-GEO-2 regression fix: `run_simulation` references
# `setup.mesh`, so it fails with `FieldError: no field mesh` if the `mesh` field
# is ever dropped from `SimulationSetup`.

using KrakeNN
using Test

@testset "Geometry: STL obstacle .krk (Cartesian immersed boundary)" begin
    root        = normpath(joinpath(@__DIR__, ".."))
    example_dir = joinpath(root, "examples", "geometry_stl")
    stl_case    = joinpath(example_dir, "cylinder_stl_flow.krk")
    anal_case   = joinpath(example_dir, "cylinder_analytic_flow.krk")
    stl_file    = joinpath(example_dir, "cylinder.stl")
    output_dir  = joinpath(root, "output")

    @test isfile(stl_file)

    mask_of(setup) = begin
        dom = setup.domain
        dx = dom.Lx / dom.Nx
        dy = dom.Ly / dom.Ny
        m  = falses(dom.Nx, dom.Ny)
        KrakeNN._apply_geometry!(m, setup, dx, dy)
        (m, dx, dy)
    end

    # .krk stl(file=…) paths are repo-root-relative; resolve from there.
    cd(root) do
        stl_setup  = load_kraken(stl_case)
        anal_setup = load_kraken(anal_case)

        stl_mask, dx, dy   = mask_of(stl_setup)
        anal_mask, dx2, dy2 = mask_of(anal_setup)
        @test (dx, dy) == (dx2, dy2)

        # --- GATE-A: voxelized STL mask matches the analytical disc (rim band only)
        dom   = anal_setup.domain
        total = dom.Nx * dom.Ny
        disagree = stl_mask .!= anal_mask
        ndis  = sum(disagree)
        cx, cy = anal_setup.user_vars[:cx], anal_setup.user_vars[:cy]
        R      = anal_setup.user_vars[:R]
        maxrim = 0.0
        for j in 1:dom.Ny, i in 1:dom.Nx
            disagree[i, j] || continue
            x = (i - 0.5) * dx
            y = (j - 0.5) * dy
            maxrim = max(maxrim, abs(hypot(x - cx, y - cy) - R))
        end
        @test 100.0 * ndis / total <= 2.0            # ≤2% of cells disagree
        @test maxrim <= 1.5 * dx + 100 * eps(Float64) # …and only in the rim band

        # --- GATE-B: STL run completes, writes VTK, and reproduces the flow
        pvd_stl = joinpath(output_dir, stl_setup.name * ".pvd")
        isfile(pvd_stl) && rm(pvd_stl; force=true)   # ensure freshness

        stl_res  = run_simulation(stl_setup)
        anal_res = run_simulation(anal_setup)

        @test isfile(pvd_stl)                        # end-to-end: VTK written

        common = .!(stl_mask .| anal_mask)
        @test any(common)
        sp_stl = maximum(sqrt.(stl_res.ux[common] .^ 2 .+ stl_res.uy[common] .^ 2))
        sp_an  = maximum(sqrt.(anal_res.ux[common] .^ 2 .+ anal_res.uy[common] .^ 2))
        denom  = max(abs(sp_stl), abs(sp_an), eps(Float64))
        @test 100.0 * abs(sp_stl - sp_an) / denom <= 2.0
    end
end

@testset "Geometry: STL obstacle .krk LI-BB selector" begin
    root        = normpath(joinpath(@__DIR__, ".."))
    example_dir = joinpath(root, "examples", "geometry_stl")
    libb_case   = joinpath(example_dir, "cylinder_stl_libb.krk")
    stl_file    = joinpath(example_dir, "cylinder.stl")

    @test isfile(stl_file)

    cd(root) do
        setup = load_kraken(libb_case)
        region = only(filter(r -> r.stl !== nothing, setup.regions))
        @test region.bc_type === :libb

        dom = setup.domain
        dx = dom.Lx / dom.Nx
        dy = dom.Ly / dom.Ny
        mask = falses(dom.Nx, dom.Ny)
        KrakeNN._apply_geometry!(mask, setup, dx, dy)

        q_wall = KrakeNN._precompute_stl_libb_q_wall_2d(mask, setup, dx, dy, Float64)
        @test any(q -> q > 0 && abs(q - 0.5) > 1e-8, q_wall)

        res = run_simulation(setup)
        @test !any(isnan, res.ρ)
        @test !any(isnan, res.ux)
    end
end

@testset "Geometry: STL .krk LI-BB matches analytical-cylinder LI-BB driver" begin
    # Correctness guard: an STL cylinder run with `wall = libb` must reproduce
    # the validated analytical-cylinder LI-BB driver (run_cylinder_libb_2d) to
    # ≤3% (they differ only in rim faceting of q_wall), and differ measurably
    # from plain halfway-BB — proving LI-BB is genuinely exercised, not a
    # silent fallback. Mirrors the M-GEO-3a canary.
    include(joinpath(@__DIR__, "..", "examples", "geometry_stl", "make_cylinder_stl.jl"))

    Nx, Ny = 160, 80
    radius = 10.0
    cx, cy = Nx / 4, Ny / 2
    u_in   = 0.04
    ν      = u_in * (2radius) / 20.0   # Re = 20
    steps  = 1200

    vel_l2 = (a, b, fluid) -> begin
        n = count(fluid)
        du2 = sum((a.ux[fluid] .- b.ux[fluid]) .^ 2 .+ (a.uy[fluid] .- b.uy[fluid]) .^ 2) / n
        ref2 = sum(b.ux[fluid] .^ 2 .+ b.uy[fluid] .^ 2) / n
        sqrt(du2) / max(sqrt(ref2), eps(Float64))
    end
    solid_mask = setup -> begin
        dom = setup.domain
        m = falses(dom.Nx, dom.Ny)
        KrakeNN._apply_geometry!(m, setup, dom.Lx / dom.Nx, dom.Ly / dom.Ny)
        m
    end
    write_case = (path, stl_path, wall_libb) -> begin
        wall = wall_libb ? ", wall = libb" : ""
        name = wall_libb ? "geo_stl_libb_test" : "geo_stl_halfway_test"
        open(path, "w") do io
            write(io, """
Simulation $name D2Q9
Define U = $u_in
Domain L = $(Float64(Nx)) x $(Float64(Ny))  N = $Nx x $Ny
Physics nu = $ν
Obstacle cyl stl(file = "$stl_path", z_slice = 0.5$wall)
Boundary west  velocity(ux = U, uy = 0)
Boundary east  pressure(rho = 1.0)
Boundary south wall
Boundary north wall
Initial { ux = U uy = 0 rho = 1 }
Run $steps steps
""")
        end
    end

    mktempdir() do dir
        stl_path  = joinpath(dir, "cyl.stl")
        libb_case = joinpath(dir, "libb.krk")
        half_case = joinpath(dir, "half.krk")
        write_cylinder_stl(stl_path; radius=radius, cx=cx, cy=cy, z0=0.0, z1=1.0, segments=384)
        write_case(libb_case, stl_path, true)
        write_case(half_case, stl_path, false)

        driver   = run_cylinder_libb_2d(; Nx=Nx, Ny=Ny, radius=radius, cx=cx, cy=cy,
                                          u_in=u_in, ν=ν, inlet=:uniform,
                                          max_steps=steps, avg_window=200, T=Float64)
        stl_libb = run_simulation(load_kraken(libb_case); T=Float64)
        stl_half = run_simulation(load_kraken(half_case); T=Float64)

        fluid = .!(driver.is_solid .| solid_mask(stl_libb.setup))
        @test vel_l2(stl_libb, driver, fluid) <= 0.03                       # matches validated driver

        fluid_lh = .!(solid_mask(stl_libb.setup) .| solid_mask(stl_half.setup))
        @test vel_l2(stl_libb, stl_half, fluid_lh) >= 5e-4                  # LI-BB ≠ halfway-BB
    end
end
