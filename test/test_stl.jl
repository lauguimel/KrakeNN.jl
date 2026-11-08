using Test
using KrakeNN

# --- Helper: generate a simple STL cube for testing ---

"""Write a binary STL file for a unit cube [0,1]^3."""
function _write_cube_stl(filename::String)
    # 12 triangles for a cube
    verts = [
        # Front face (z=1)
        ((0,0,1), (1,0,1), (1,1,1)),
        ((0,0,1), (1,1,1), (0,1,1)),
        # Back face (z=0)
        ((0,0,0), (1,1,0), (1,0,0)),
        ((0,0,0), (0,1,0), (1,1,0)),
        # Right face (x=1)
        ((1,0,0), (1,1,0), (1,1,1)),
        ((1,0,0), (1,1,1), (1,0,1)),
        # Left face (x=0)
        ((0,0,0), (0,1,1), (0,1,0)),
        ((0,0,0), (0,0,1), (0,1,1)),
        # Top face (y=1)
        ((0,1,0), (0,1,1), (1,1,1)),
        ((0,1,0), (1,1,1), (1,1,0)),
        # Bottom face (y=0)
        ((0,0,0), (1,0,0), (1,0,1)),
        ((0,0,0), (1,0,1), (0,0,1)),
    ]

    open(filename, "w") do io
        # 80-byte header
        write(io, zeros(UInt8, 80))
        # Triangle count
        write(io, UInt32(length(verts)))
        for (v1, v2, v3) in verts
            # Normal (0,0,0) — reader doesn't use it for voxelization
            write(io, Float32(0), Float32(0), Float32(0))
            # Vertices
            for v in (v1, v2, v3)
                write(io, Float32(v[1]), Float32(v[2]), Float32(v[3]))
            end
            # Attribute byte count
            write(io, UInt16(0))
        end
    end
end

"""Write an ASCII STL file for a unit cube [0,1]^3."""
function _write_cube_stl_ascii(filename::String)
    verts = [
        ((0,0,1), (1,0,1), (1,1,1)),
        ((0,0,1), (1,1,1), (0,1,1)),
        ((0,0,0), (1,1,0), (1,0,0)),
        ((0,0,0), (0,1,0), (1,1,0)),
    ]

    open(filename, "w") do io
        println(io, "solid cube")
        for (v1, v2, v3) in verts
            println(io, "  facet normal 0 0 0")
            println(io, "    outer loop")
            println(io, "      vertex $(v1[1]) $(v1[2]) $(v1[3])")
            println(io, "      vertex $(v2[1]) $(v2[2]) $(v2[3])")
            println(io, "      vertex $(v3[1]) $(v3[2]) $(v3[3])")
            println(io, "    endloop")
            println(io, "  endfacet")
        end
        println(io, "endsolid cube")
    end
end

"""Write binary STL for a cylinder (axis along z, radius R, height H, centered at cx,cy)."""
function _write_cylinder_stl(filename::String; R=0.5, H=1.0, cx=0.5, cy=0.5,
                              N=32)
    open(filename, "w") do io
        write(io, zeros(UInt8, 80))
        ntri = 2 * N  # lateral surface only (no caps needed for 2D slice test)
        write(io, UInt32(ntri))

        for k in 1:N
            θ1 = 2π * (k - 1) / N
            θ2 = 2π * k / N
            x1, y1 = cx + R * cos(θ1), cy + R * sin(θ1)
            x2, y2 = cx + R * cos(θ2), cy + R * sin(θ2)

            # Bottom triangle
            write(io, Float32(0), Float32(0), Float32(0))  # normal
            write(io, Float32(x1), Float32(y1), Float32(0))
            write(io, Float32(x2), Float32(y2), Float32(0))
            write(io, Float32(x2), Float32(y2), Float32(H))
            write(io, UInt16(0))

            # Top triangle
            write(io, Float32(0), Float32(0), Float32(0))
            write(io, Float32(x1), Float32(y1), Float32(0))
            write(io, Float32(x2), Float32(y2), Float32(H))
            write(io, Float32(x1), Float32(y1), Float32(H))
            write(io, UInt16(0))
        end
    end
end

@testset "STL Support" begin

    @testset "Binary STL reader — cube" begin
        f = tempname() * ".stl"
        _write_cube_stl(f)
        mesh = read_stl(f)

        @test length(mesh.triangles) == 12
        @test mesh.bbox_min == (0.0, 0.0, 0.0)
        @test mesh.bbox_max == (1.0, 1.0, 1.0)

        # Vertices should be Float64
        @test mesh.triangles[1].v1 isa NTuple{3, Float64}
        rm(f)
    end

    @testset "ASCII STL reader" begin
        f = tempname() * ".stl"
        _write_cube_stl_ascii(f)
        mesh = read_stl(f)

        @test length(mesh.triangles) == 4
        @test mesh.bbox_min[1] ≈ 0.0
        @test mesh.bbox_max[1] ≈ 1.0
        rm(f)
    end

    @testset "Transform mesh" begin
        f = tempname() * ".stl"
        _write_cube_stl(f)
        mesh = read_stl(f)

        scaled = transform_mesh(mesh; scale=2.0, translate=(1.0, 0.0, 0.0))
        @test scaled.bbox_min == (1.0, 0.0, 0.0)
        @test scaled.bbox_max == (3.0, 2.0, 2.0)
        rm(f)
    end

    @testset "Voxelize 3D — cube" begin
        f = tempname() * ".stl"
        _write_cube_stl(f)
        mesh = read_stl(f)

        # Grid: 10x10x10 covering [0, 1]^3
        N = 10
        dx = 1.0 / N
        solid = voxelize_3d(mesh, N, N, N, dx, dx, dx)

        # Center node (5,5,5) at (0.45, 0.45, 0.45) should be inside
        @test solid[5, 5, 5] == true
        # Corner-ish node (1,1,1) at (0.05, 0.05, 0.05) should be inside
        @test solid[1, 1, 1] == true

        # Most interior nodes should be solid
        interior_count = sum(solid)
        @test interior_count > 0.7 * N^3  # cube fills most of the grid
        rm(f)
    end

    @testset "Voxelize 2D — cylinder cross-section" begin
        f = tempname() * ".stl"
        R = 0.3
        cx, cy = 0.5, 0.5
        _write_cylinder_stl(f; R=R, cx=cx, cy=cy, N=64)
        mesh = read_stl(f)

        # Grid: 20x20 covering [0, 1]^2
        N = 20
        dx = 1.0 / N
        solid = voxelize_2d(mesh, N, N, dx, dx; z_slice=0.5)

        # Center (10, 10) at (0.475, 0.475) should be inside
        @test solid[10, 10] == true
        # Far corner (1, 1) at (0.025, 0.025) should be outside
        @test solid[1, 1] == false

        # Check approximate area: π*R² ≈ 0.283, grid area = 1.0
        # Expected solid fraction ≈ 0.283
        solid_frac = sum(solid) / N^2
        @test 0.15 < solid_frac < 0.45

        rm(f)
    end

    @testset "Parser: Obstacle stl(...)" begin
        f = tempname() * ".stl"
        _write_cube_stl(f)

        setup = parse_kraken("""
            Simulation stl_test D2Q9
            Domain L = 2 x 2  N = 20 x 20
            Physics nu = 0.1

            Obstacle body stl(file = "$f")

            Boundary south wall
            Boundary north wall
            Boundary east wall
            Boundary west wall

            Run 100 steps
        """)

        @test length(setup.regions) == 1
        @test setup.regions[1].stl !== nothing
        @test setup.regions[1].stl.file == f
        @test setup.regions[1].stl.scale ≈ 1.0
        @test setup.regions[1].condition === nothing
        rm(f)
    end

    @testset "Parser: STL with transform" begin
        setup = parse_kraken("""
            Simulation stl_test D2Q9
            Domain L = 2 x 2  N = 20 x 20
            Physics nu = 0.1

            Obstacle body stl(file = "mesh.stl", scale = 0.01, translate = [1.0, 0.5, 0.0], z_slice = 0.5)

            Boundary south wall
            Boundary north wall
            Boundary east wall
            Boundary west wall

            Run 100 steps
        """)

        stl = setup.regions[1].stl
        @test stl.scale ≈ 0.01
        @test stl.translate == (1.0, 0.5, 0.0)
        @test stl.z_slice ≈ 0.5
    end

    @testset "End-to-end: run_simulation with STL obstacle" begin
        f = tempname() * ".stl"
        R = 0.3
        _write_cylinder_stl(f; R=R, cx=1.0, cy=0.5, N=32)

        setup = parse_kraken("""
            Simulation stl_e2e D2Q9
            Domain L = 2 x 1  N = 40 x 20
            Physics nu = 0.1

            Obstacle cylinder stl(file = "$f", z_slice = 0.5)

            Boundary west  velocity(ux = 0.05, uy = 0)
            Boundary east  pressure(rho = 1.0)
            Boundary south wall
            Boundary north wall

            Run 500 steps
        """)

        result = run_simulation(setup)
        @test !any(isnan, result.ρ)
        @test !any(isnan, result.ux)

        @info "STL end-to-end: simulation stable after 500 steps"
        rm(f)
    end
end
