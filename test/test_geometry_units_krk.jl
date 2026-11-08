using KrakeNN
using Test

@testset "Geometry: STL physical Units block lowers to raw LU at parse time" begin
    root = normpath(joinpath(@__DIR__, ".."))
    example_dir = joinpath(root, "examples", "geometry_stl")
    mm_case = joinpath(example_dir, "cylinder_stl_mm.krk")
    lu_case = joinpath(example_dir, "cylinder_stl_lu.krk")

    mask_of(setup) = begin
        dom = setup.domain
        mask = falses(dom.Nx, dom.Ny)
        KrakeNN._apply_geometry!(mask, setup, dom.Lx / dom.Nx, dom.Ly / dom.Ny)
        mask
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

        result_mm = run_simulation(setup_mm)
        result_lu = run_simulation(setup_lu)
        field_linf = max(maximum(abs.(result_mm.ux .- result_lu.ux)),
                         maximum(abs.(result_mm.uy .- result_lu.uy)))
        @test field_linf <= 1e-8
        @test !any(isnan, result_lu.ux)
        @test !any(isnan, result_lu.uy)
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    println("PASS M-GEO-4")
end
