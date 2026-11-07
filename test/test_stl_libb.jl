using Test
using KrakeNN

# ==========================================================================
# STL → LI-BB cut-fraction precomputation.
#
# Verifies that `precompute_q_wall_from_stl_{2,3}d` produce a q_wall
# array consistent with the voxelised is_solid mask (every fluid-cell
# link pointing into a solid neighbour has q_w = 0.5; others are 0).
# Also spot-checks that using the STL-derived (q_wall, is_solid) to
# run `fused_trt_libb_v2_step!` gives a stable, physical flow.
# ==========================================================================

# STL helpers, duplicated from test_stl.jl for standalone usage.
function _write_cylinder_stl_libb(filename::String; R=0.5, H=1.0,
                                   cx=0.5, cy=0.5, N=64)
    open(filename, "w") do io
        write(io, zeros(UInt8, 80))
        write(io, UInt32(2N))
        for k in 1:N
            θ1 = 2π * (k - 1) / N
            θ2 = 2π * k / N
            x1, y1 = cx + R * cos(θ1), cy + R * sin(θ1)
            x2, y2 = cx + R * cos(θ2), cy + R * sin(θ2)
            write(io, Float32(0), Float32(0), Float32(0))
            write(io, Float32(x1), Float32(y1), Float32(0))
            write(io, Float32(x2), Float32(y2), Float32(0))
            write(io, Float32(x2), Float32(y2), Float32(H))
            write(io, UInt16(0))
            write(io, Float32(0), Float32(0), Float32(0))
            write(io, Float32(x1), Float32(y1), Float32(0))
            write(io, Float32(x2), Float32(y2), Float32(H))
            write(io, Float32(x1), Float32(y1), Float32(H))
            write(io, UInt16(0))
        end
    end
end

function _write_cube_stl_libb(filename::String)
    verts = [
        ((0,0,1), (1,0,1), (1,1,1)), ((0,0,1), (1,1,1), (0,1,1)),
        ((0,0,0), (1,1,0), (1,0,0)), ((0,0,0), (0,1,0), (1,1,0)),
        ((1,0,0), (1,1,0), (1,1,1)), ((1,0,0), (1,1,1), (1,0,1)),
        ((0,0,0), (0,1,1), (0,1,0)), ((0,0,0), (0,0,1), (0,1,1)),
        ((0,1,0), (0,1,1), (1,1,1)), ((0,1,0), (1,1,1), (1,1,0)),
        ((0,0,0), (1,0,0), (1,0,1)), ((0,0,0), (1,0,1), (0,0,1)),
    ]
    open(filename, "w") do io
        write(io, zeros(UInt8, 80))
        write(io, UInt32(length(verts)))
        for (v1, v2, v3) in verts
            write(io, Float32(0), Float32(0), Float32(0))
            for v in (v1, v2, v3)
                write(io, Float32(v[1]), Float32(v[2]), Float32(v[3]))
            end
            write(io, UInt16(0))
        end
    end
end

@testset "STL → LI-BB q_wall" begin

    @testset "2D: cylinder STL matches analytic cylinder (halfway BB)" begin
        f = tempname() * ".stl"
        R = 0.3; cx = 0.5; cy = 0.5
        _write_cylinder_stl_libb(f; R=R, cx=cx, cy=cy, N=128)
        mesh = read_stl(f)

        # Grid 40x40 on [0, 1]^2
        N = 40
        dx = 1.0 / N
        qw, is_solid = precompute_q_wall_from_stl_2d(mesh, N, N, dx, dx;
                                                      z_slice = 0.5)

        @test size(qw) == (N, N, 9)
        @test size(is_solid) == (N, N)

        # Solid fraction ~ π R² = 0.283
        @test 0.2 < sum(is_solid) / N^2 < 0.4

        # Every fluid cell with a flagged link must have that link's
        # neighbour actually inside the solid mask, and q_w ∈ (0, 1].
        cxs = velocities_x(D2Q9())
        cys = velocities_y(D2Q9())
        for j in 1:N, i in 1:N
            is_solid[i, j] && continue
            for q in 2:9
                if qw[i, j, q] > 0
                    ni = i + Int(cxs[q]); nj = j + Int(cys[q])
                    @test 1 <= ni <= N && 1 <= nj <= N
                    @test is_solid[ni, nj]
                    @test 0 < qw[i, j, q] <= 1
                end
            end
        end

        # Sub-cell q_w distribution: values should span (0, 1], not
        # be constant 0.5.
        cuts = filter(>(0), vec(qw))
        @test length(cuts) > 20
        @test minimum(cuts) < 0.4   # some clearly < 0.5
        @test maximum(cuts) > 0.6   # some clearly > 0.5

        # Ring of fluid cells around the cylinder must have at least
        # one flagged link each.
        n_flagged_ring = 0
        for j in 1:N, i in 1:N
            is_solid[i, j] && continue
            if any(qw[i, j, q] > 0 for q in 2:9)
                n_flagged_ring += 1
            end
        end
        @test n_flagged_ring > 20   # perimeter of ~2πR / dx cells

        rm(f)
    end

    @testset "3D: cube STL, halfway-BB on all boundary faces" begin
        f = tempname() * ".stl"
        _write_cube_stl_libb(f)
        mesh = read_stl(f)

        # Grid 20x20x20 covering [0, 1]^3 — the cube is the FULL domain
        # except 1 layer of ghost on each face. Shift the grid to leave
        # a fluid border: grid [-0.2, 1.2]^3, cube [0, 1]^3 in middle.
        N = 20
        L = 1.4
        dx = L / N
        # Shift grid origin by -0.2 so first cell center is at (-0.15, ...)
        # Actually: voxelize places cell center i at ((i-0.5)*dx). With L=1.4
        # and N=20 → dx=0.07, cell 1 at 0.035, cell 20 at 1.365. The cube
        # [0, 1] covers approximately cells 1..14.
        qw, is_solid = precompute_q_wall_from_stl_3d(mesh, N, N, N,
                                                      dx, dx, dx)

        @test size(qw) == (N, N, N, 19)
        @test size(is_solid) == (N, N, N)

        # Cube occupies a substantial solid fraction
        @test sum(is_solid) > 0.3 * N^3

        # Every flagged link must point to a solid neighbour.
        cxs = velocities_x(D3Q19())
        cys = velocities_y(D3Q19())
        czs = velocities_z(D3Q19())
        for k in 1:N, j in 1:N, i in 1:N
            is_solid[i, j, k] && continue
            for q in 2:19
                if qw[i, j, k, q] > 0
                    ni = i + Int(cxs[q]); nj = j + Int(cys[q]); nk = k + Int(czs[q])
                    @test 1 <= ni <= N && 1 <= nj <= N && 1 <= nk <= N
                    @test is_solid[ni, nj, nk]
                    @test qw[i, j, k, q] == 0.5
                end
            end
        end

        rm(f)
    end

    @testset "3D: cube STL sub-cell q_w via Möller-Trumbore" begin
        # The cube [0,1]³ has axis-aligned faces. Using a grid that's
        # NOT cell-centre-aligned with the faces, sub-cell ray casts
        # must give q_w ∈ (0, 1) ≠ 0.5 on the cells adjacent to the
        # faces (since fluid-node / wall-face midpoints are offset).
        # This exercises the Möller-Trumbore path and confirms that
        # the computed fractions are bounded and correct.
        f = tempname() * ".stl"
        _write_cube_stl_libb(f)
        mesh = read_stl(f)
        N = 20
        L = 1.4; dx = L / N
        qw_h, is_s_h = precompute_q_wall_from_stl_3d(mesh, N, N, N, dx, dx, dx;
                                                       sub_cell=false)
        qw_s, is_s_s = precompute_q_wall_from_stl_3d(mesh, N, N, N, dx, dx, dx;
                                                       sub_cell=true)
        @test is_s_h == is_s_s
        nh = count(x -> x > 0, qw_h)
        @test nh > 100
        @test all(qw_h[qw_h .> 0] .== 0.5)
        ns = count(x -> x > 0, qw_s)
        @test ns > 0
        q_w_vals = qw_s[qw_s .> 0]
        @test all(0 .< q_w_vals .<= 1)
        # The cube [0,1]³ on a grid with dx=0.07 offset by half a cell
        # (cell centres at 0.035 + (i-1)·0.07) places the first fluid
        # cell centre at 1.015 and the cube wall at 1.0. Axial q_w =
        # 0.015 / 0.07 ≈ 0.214 — a specific sub-cell value ≠ 0.5. All
        # axial cut-link q_w should cluster around this ratio.
        @test all(q_w_vals .< 0.5)
        @test abs(maximum(q_w_vals) - 0.015/0.07) < 0.02
        rm(f)
    end

    @testset "STL-driven LI-BB 2D step runs stable (no NaNs)" begin
        f = tempname() * ".stl"
        _write_cylinder_stl_libb(f; R=0.15, cx=0.3, cy=0.5, N=64)
        mesh = read_stl(f)

        Nx, Ny = 80, 40
        L = 1.0; dx = L / Ny   # choose dy = dx, Nx covers 2·L
        qw, is_solid = precompute_q_wall_from_stl_2d(mesh, Nx, Ny, dx, dx;
                                                      z_slice = 0.5)
        # Quick step — verify integrates without NaN over a few hundred
        # steps with inlet/outlet BCs.
        uw_x = zeros(Float64, Nx, Ny, 9)
        uw_y = zeros(Float64, Nx, Ny, 9)
        f_in = zeros(Float64, Nx, Ny, 9)
        for j in 1:Ny, i in 1:Nx, q in 1:9
            f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, 0.01, 0.0, q)
        end
        f_out = similar(f_in)
        ρ = ones(Nx, Ny); ux = zeros(Nx, Ny); uy = zeros(Nx, Ny)

        for _ in 1:500
            fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                     qw, uw_x, uw_y, Nx, Ny, 0.05)
            # Equilibrium inlet
            for j in 1:Ny, q in 1:9
                f_out[1, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, 0.01, 0.0, q)
            end
            # Neumann outlet
            for j in 1:Ny, q in 1:9
                f_out[Nx, j, q] = f_out[Nx-1, j, q]
            end
            f_in, f_out = f_out, f_in
        end

        @test !any(isnan, ρ)
        @test !any(isnan, ux)
        @test maximum(ρ) < 2.0 && minimum(ρ) > 0.5
        rm(f)
    end

end
