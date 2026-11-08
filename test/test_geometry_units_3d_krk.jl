using KrakeNN
using Test

@testset "Geometry: 3D STL physical Units block lowers to raw LU mask" begin
    root = normpath(joinpath(@__DIR__, ".."))
    example_dir = joinpath(root, "examples", "geometry_stl")
    mm_case = joinpath(example_dir, "sphere_stl_3d_mm.krk")
    lu_case = joinpath(example_dir, "sphere_stl_3d_lu.krk")

    mask_of(setup) = begin
        dom = setup.domain
        mask = falses(dom.Nx, dom.Ny, dom.Nz)
        KrakeNN._apply_geometry_3d!(mask, setup, dom.Lx / dom.Nx)
        mask
    end

    scaled_mesh(region) = begin
        stl = region.stl
        mesh = KrakeNN.read_stl(stl.file)
        if stl.scale != 1.0 || stl.translate != (0.0, 0.0, 0.0)
            mesh = KrakeNN.transform_mesh(mesh; scale=stl.scale,
                                         translate=stl.translate)
        end
        mesh
    end

    cd(root) do
        setup_mm = load_kraken(mm_case)
        setup_lu = load_kraken(lu_case)

        region_mm = only(filter(r -> r.stl !== nothing, setup_mm.regions))
        region_lu = only(filter(r -> r.stl !== nothing, setup_lu.regions))
        @test setup_mm.units !== nothing
        @test setup_lu.units === nothing
        @test region_mm.stl.scale ≈ region_lu.stl.scale atol=1e-12 rtol=0
        @test setup_mm.physics.params[:nu] ≈ setup_lu.physics.params[:nu] atol=1e-12 rtol=0

        ux_mm = only(filter(b -> b.face === :west, setup_mm.boundaries)).values[:ux]
        ux_lu = only(filter(b -> b.face === :west, setup_lu.boundaries)).values[:ux]
        @test evaluate(ux_mm) ≈ evaluate(ux_lu) atol=1e-12 rtol=0

        mask_mm = mask_of(setup_mm)
        mask_lu = mask_of(setup_lu)
        @test count(mask_mm) == count(mask_lu)
        @test mask_mm == mask_lu

        sphere_mesh = scaled_mesh(region_mm)
        kappa = KrakeNN.stl_kappa_max(sphere_mesh)
        @test kappa ≈ 0.25 atol=1e-7 rtol=0

        Lu, Ld = KrakeNN.obstacle_extents_in_R(mask_mm, 4)
        @test Lu != 15.0
        @test Lu ≈ 5.0 atol=0.25 rtol=0
        @test Ld ≈ 15.0 atol=0.25 rtol=0

        geom_nt = (type=:obstacle,
                   blockage=Float64(count(mask_mm)) / Float64(length(mask_mm)),
                   L_up=Lu,
                   L_down=Ld,
                   q_wall_dist=KrakeNN.halfway_wall_distances(mask_mm),
                   kappa_max=0.25,
                   stl_hash=UInt64(0))
        bc_nt = (inlet=:velocity_uniform,
                 outlet=:zou_he_pressure,
                 north_wall=:halfwayBB,
                 south_wall=:halfwayBB,
                 wall_bc=:bouzidi_fl)
        plan = KrakeNN.Units.compile(; physics=:newtonian, geometry=geom_nt,
                                    bc=bc_nt, Re=1.0, R_LU=4,
                                    dx_real=0.05, scaling=:acoustic)
        @test :curvature_underresolved in [w.code for w in plan.warnings]
        @test plan.geometry.kappa_max ≈ 0.25
        @test plan.geometry.L_up ≈ Lu
        @test plan.geometry.L_down ≈ Ld
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    println("PASS M-GEO-5")
end
