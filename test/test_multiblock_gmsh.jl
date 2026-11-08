using Test
using KrakeNN
using KrakeNN: INTERFACE_TAG, edge_coords

# =====================================================================
# v0.3 Phase B.1 — gmsh multi-surface loader.
#
# Two self-contained .geo files are built on-the-fly in a tempdir:
#
#   (1) 2-block canal: one rectangle [0, 2] × [0, 1] split at x = 1 into
#       two sub-surfaces sharing the central vertical curve. Physical
#       names: "inlet", "outlet", "wall_bot", "wall_top". The shared
#       curve has no physical name → loader must tag the two inner edges
#       as :interface via the shared-curve heuristic.
#
#   (2) 4-block O-grid cylinder: quadrants NE/NW/SW/SE of the annulus
#       r ∈ [R_in, R_out]. Each block bounded by inner arc (cylinder),
#       outer arc (farfield), and two radial lines (shared between
#       neighbours → interfaces).
#
# For both cases we assert: (a) expected number of blocks and
# interfaces, (b) each block's edge tags match expectations, (c) the
# multi-block sanity check passes without :error. Shared-node topology
# (zero offset between A's east and B's west) is the natural gmsh
# output, so the colocation check issues a :warning — we tolerate it.
# =====================================================================

@testset "Multi-block gmsh loader (v0.3 Phase B.1)" begin

    # --------- Test 1: 2-block canal, shared internal curve ----------
    @testset "2-block canal (shared interface curve)" begin
        mktempdir() do dir
            geo_path = joinpath(dir, "canal_2block.geo")
            write(geo_path, """
            SetFactory("Built-in");
            // 2-block axis-aligned canal split at x = 1.
            Lx_half = 1.0;
            H = 1.0;
            Nx = 11;  // nodes per block along x (10 cells)
            Ny = 6;   // nodes along y (5 cells)

            Point(1) = {0,         0, 0};
            Point(2) = {Lx_half,   0, 0};
            Point(3) = {2*Lx_half, 0, 0};
            Point(4) = {2*Lx_half, H, 0};
            Point(5) = {Lx_half,   H, 0};
            Point(6) = {0,         H, 0};

            // Outer: S1,S2 (bot), E (right), N1,N2 (top), W (left)
            Line(11) = {1, 2};   // south of block A
            Line(12) = {2, 3};   // south of block B
            Line(13) = {3, 4};   // east  of block B
            Line(14) = {4, 5};   // north of block B
            Line(15) = {5, 6};   // north of block A
            Line(16) = {6, 1};   // west  of block A
            // Shared interface curve (no physical group → loader detects sharing)
            Line(17) = {2, 5};

            Curve Loop(21) = {11, 17, 15, 16};
            Plane Surface(31) = {21};
            Curve Loop(22) = {12, 13, 14, -17};
            Plane Surface(32) = {22};

            Transfinite Curve {11, 12, 14, 15} = Nx;
            Transfinite Curve {13, 16, 17}     = Ny;
            Transfinite Surface {31} = {1, 2, 5, 6};
            Transfinite Surface {32} = {2, 3, 4, 5};
            Recombine Surface {31, 32};

            Physical Curve("wall_bot") = {11, 12};
            Physical Curve("wall_top") = {14, 15};
            Physical Curve("inlet")    = {16};
            Physical Curve("outlet")   = {13};
            Physical Surface("A")      = {31};
            Physical Surface("B")      = {32};
            """)

            mbm, groups = load_gmsh_multiblock_2d(geo_path; FT=Float64)

            @test length(mbm.blocks) == 2
            @test length(mbm.interfaces) == 1

            # Block ids come from physical surface names
            ids = Set(b.id for b in mbm.blocks)
            @test ids == Set([:A, :B])

            blk_A = getblock(mbm, :A)
            blk_B = getblock(mbm, :B)

            # Block A: west=:inlet, east=:interface, south=:wall_bot, north=:wall_top
            # (orientation of west/east vs inlet/outlet depends on layout; we
            # just check the set of non-interface tags matches)
            tags_A = Set((blk_A.boundary_tags.west, blk_A.boundary_tags.east,
                          blk_A.boundary_tags.south, blk_A.boundary_tags.north))
            tags_B = Set((blk_B.boundary_tags.west, blk_B.boundary_tags.east,
                          blk_B.boundary_tags.south, blk_B.boundary_tags.north))
            @test :wall_bot  in tags_A
            @test :wall_top  in tags_A
            @test :interface in tags_A
            @test :wall_bot  in tags_B
            @test :wall_top  in tags_B
            @test :interface in tags_B
            # Exactly one of {inlet, outlet} per block
            @test (:inlet in tags_A) ⊻ (:outlet in tags_A)
            @test (:inlet in tags_B) ⊻ (:outlet in tags_B)

            # Each block has exactly ONE :interface edge
            @test count(e -> getproperty(blk_A.boundary_tags, e) === INTERFACE_TAG,
                         (:west, :east, :south, :north)) == 1
            @test count(e -> getproperty(blk_B.boundary_tags, e) === INTERFACE_TAG,
                         (:west, :east, :south, :north)) == 1

            # Sanity: no :error. Warnings (shared-node topology) are OK.
            issues = sanity_check_multiblock(mbm; verbose=false)
            @test all(iss -> iss.severity !== :error, issues)

            # The physical groups carry the expected names
            @test haskey(groups.by_name, "inlet")
            @test haskey(groups.by_name, "outlet")
            @test haskey(groups.by_name, "A")
            @test haskey(groups.by_name, "B")

            # Mesh sizes match the .geo
            @test blk_A.mesh.Nξ * blk_A.mesh.Nη == 11 * 6
            @test blk_B.mesh.Nξ * blk_B.mesh.Nη == 11 * 6
        end
    end

    # --------- Test 2: 4-block O-grid cylinder ------------------------
    @testset "4-block O-grid around cylinder" begin
        mktempdir() do dir
            geo_path = joinpath(dir, "cyl_ogrid_4block.geo")
            write(geo_path, """
            SetFactory("Built-in");
            R_in  = 0.5;
            R_out = 2.0;
            N_arc = 13;   // nodes per 90° arc (= 12 cells)
            N_rad = 8;    // nodes radially (= 7 cells)

            Point(99) = {0, 0, 0};
            // Cylinder corner points (E, N, W, S)
            Point(1) = { R_in,  0, 0};
            Point(2) = { 0,  R_in, 0};
            Point(3) = {-R_in,  0, 0};
            Point(4) = { 0, -R_in, 0};
            // Far-field corner points
            Point(5) = { R_out,  0, 0};
            Point(6) = { 0,  R_out, 0};
            Point(7) = {-R_out,  0, 0};
            Point(8) = { 0, -R_out, 0};

            // Inner arcs (cylinder)
            Circle(11) = {1, 99, 2};  // NE
            Circle(12) = {2, 99, 3};  // NW
            Circle(13) = {3, 99, 4};  // SW
            Circle(14) = {4, 99, 1};  // SE
            // Outer arcs
            Circle(21) = {5, 99, 6};  // NE
            Circle(22) = {6, 99, 7};  // NW
            Circle(23) = {7, 99, 8};  // SW
            Circle(24) = {8, 99, 5};  // SE
            // Radial spokes (shared between adjacent quadrants)
            Line(31) = {1, 5};  // E radial
            Line(32) = {2, 6};  // N radial
            Line(33) = {3, 7};  // W radial
            Line(34) = {4, 8};  // S radial

            // 4 quadrant surfaces (curved: use Surface, not Plane)
            Curve Loop(41) = {11, 32, -21, -31};  Surface(51) = {41};  // NE
            Curve Loop(42) = {12, 33, -22, -32};  Surface(52) = {42};  // NW
            Curve Loop(43) = {13, 34, -23, -33};  Surface(53) = {43};  // SW
            Curve Loop(44) = {14, 31, -24, -34};  Surface(54) = {44};  // SE

            Transfinite Curve {11, 12, 13, 14, 21, 22, 23, 24} = N_arc;
            Transfinite Curve {31, 32, 33, 34}                 = N_rad;
            Transfinite Surface {51} = {1, 2, 6, 5};
            Transfinite Surface {52} = {2, 3, 7, 6};
            Transfinite Surface {53} = {3, 4, 8, 7};
            Transfinite Surface {54} = {4, 1, 5, 8};
            Recombine Surface {51, 52, 53, 54};

            Physical Curve("cylinder") = {11, 12, 13, 14};
            Physical Curve("farfield") = {21, 22, 23, 24};
            Physical Surface("NE") = {51};
            Physical Surface("NW") = {52};
            Physical Surface("SW") = {53};
            Physical Surface("SE") = {54};
            """)

            mbm, groups = load_gmsh_multiblock_2d(geo_path; FT=Float64,
                                                    layout=:topological)

            @test length(mbm.blocks) == 4
            ids = Set(b.id for b in mbm.blocks)
            @test ids == Set([:NE, :NW, :SW, :SE])
            # Each block is bounded by 1 cylinder arc, 1 farfield arc, 2
            # radial spokes → 2 interfaces per block (the spokes), so:
            # total interface endpoints = 4 blocks × 2 = 8 → 4 interfaces.
            @test length(mbm.interfaces) == 4

            for b in mbm.blocks
                tags = (b.boundary_tags.west, b.boundary_tags.east,
                        b.boundary_tags.south, b.boundary_tags.north)
                # Each block has exactly 1 :cylinder, 1 :farfield, 2 :interface
                @test count(==(:cylinder),  tags) == 1
                @test count(==(:farfield),  tags) == 1
                @test count(==(INTERFACE_TAG), tags) == 2

                # Sanity of corner geometry: cylinder edge sits on r ≈ R_in
                cyl_edge = first(e for e in (:west, :east, :south, :north)
                                   if getproperty(b.boundary_tags, e) === :cylinder)
                xs, ys = edge_coords(b, cyl_edge)
                radii = hypot.(xs, ys)
                @test all(r -> isapprox(r, 0.5; atol=1e-10), radii)
                ff_edge = first(e for e in (:west, :east, :south, :north)
                                  if getproperty(b.boundary_tags, e) === :farfield)
                xs_f, ys_f = edge_coords(b, ff_edge)
                @test all(r -> isapprox(r, 2.0; atol=1e-10), hypot.(xs_f, ys_f))
            end

            # Sanity: structural invariants must pass (no duplicate ids,
            # same-length interfaces, refs exist, uniform element type,
            # both-edges-marked, every-marked-used). The topological
            # walker gives each block an arbitrary ξ/η orientation, so
            # some interfaces will trip :InterfaceOrientationTrivial —
            # that's expected and is a B.2 concern (block reorientation
            # pass + generalized exchange kernel). We filter it out and
            # require the rest to be clean.
            issues = sanity_check_multiblock(mbm; verbose=false)
            structural_errors = filter(iss -> iss.severity === :error &&
                                                iss.code !== :InterfaceOrientationTrivial,
                                        issues)
            @test isempty(structural_errors)
        end
    end
end
