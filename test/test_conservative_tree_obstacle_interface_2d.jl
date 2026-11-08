using Test
using Printf
using KrakeNN

# Phase A diagnostic: obstacle + interface coarse/fine
#
# Goal: localize the cause of the cylinder/square Cd ratio ~1.86x between
# AMR route-native and the leaf oracle, by isolating rest-state and 1-step
# transport differences near solid + interface configurations.
#
# Constraint reminder: an active coarse cell cannot have partially solid
# leaf children (`_check_route_solid_mask_layout`). The 4 leaf children of
# any active coarse cell must be all solid or all fluid.

const _VC = 1.0
const _VF = 0.25
const _OMEGA_REST = 1.0  # at u=0, BGK with omega=1 is identity on equilibrium

# ---------- helpers reused across canaries ----------

function _composite_to_leaf(coarse, patch)
    leaf = zeros(eltype(coarse), 2 * size(coarse, 1), 2 * size(coarse, 2), 9)
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    return leaf
end

function _make_state_2d(Nx::Int, Ny::Int,
                        patch_i_range::UnitRange{Int},
                        patch_j_range::UnitRange{Int};
                        rho::Float64=1.0, u::Float64=0.0,
                        coarse_route_mode::Symbol=:leaf_equivalent)
    coarse = zeros(Float64, Nx, Ny, 9)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range)
    topology = create_conservative_tree_topology_2d(
        Nx, Ny, patch; coarse_route_mode=coarse_route_mode)
    fill_equilibrium_integrated_D2Q9!(coarse, _VC, rho, u, 0.0)
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, _VF, rho, u, 0.0)
    return coarse, patch, topology
end

"""
    _step_route_native!(coarse, coarse_next, patch, patch_next, topology, is_solid;
                       Fx=0.0, Fy=0.0, omega=_OMEGA_REST)

Run one route-native (collide_Guo_solid + stream_routes_periodic_x_wall_y_solid)
step. Returns the post-step (coarse, patch) tuple after pointer swap.
"""
function _step_route_native!(coarse, coarse_next, patch, patch_next, topology,
                             is_solid; Fx=0.0, Fy=0.0, omega=_OMEGA_REST)
    collide_Guo_composite_solid_F_2d!(
        coarse, patch, topology, is_solid,
        _VC, _VF, omega, omega, Fx, Fy)
    stream_composite_routes_periodic_x_wall_y_solid_F_2d!(
        coarse_next, patch_next, coarse, patch, topology, is_solid)
    return coarse_next, patch_next
end

"""
    _step_oracle_leaf!(leaf, leaf_post, coarse, patch, coarse_next, patch_next, is_solid;
                      Fx=0.0, Fy=0.0, omega=_OMEGA_REST)

Run one oracle leaf step: composite_to_leaf, collide_Guo on leaf, leaf
periodic_x_wall_y solid stream, leaf_to_composite. Mirrors the path used by
`run_conservative_tree_cylinder_macroflow_2d`.
"""
function _step_oracle_leaf!(leaf, leaf_post, coarse, patch, coarse_next, patch_next,
                            is_solid; Fx=0.0, Fy=0.0, omega=_OMEGA_REST)
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    collide_Guo_integrated_D2Q9!(leaf, is_solid, _VF, omega, Fx, Fy)
    stream_periodic_x_wall_y_solid_F_2d!(leaf_post, leaf, is_solid)
    leaf_to_composite_F_2d!(coarse_next, patch_next, leaf_post)
    return coarse_next, patch_next
end

"""
    _leaf_velocity_field(coarse, patch, is_solid)

Return (rho, ux, uy) leaf-grid arrays computed from the composite state,
with rho=ux=uy=NaN inside solid cells (so they are skipped by ‖·‖_∞).
"""
function _leaf_velocity_field(coarse, patch, is_solid)
    leaf = _composite_to_leaf(coarse, patch)
    Nx, Ny, _ = size(leaf)
    rho = fill(NaN, Nx, Ny)
    ux = fill(NaN, Nx, Ny)
    uy = fill(NaN, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        F = @view leaf[i, j, :]
        m = mass_F(F)
        if m > 0
            rho[i, j] = m / _VF
            mom = momentum_F(F)
            ux[i, j] = mom[1] / m
            uy[i, j] = mom[2] / m
        end
    end
    return rho, ux, uy, leaf
end

"""
    diff_route_vs_oracle_one_step(Nx, Ny, patch_i_range, patch_j_range, is_solid;
                                  steps=1, rho=1.0, u=0.0, Fx=0.0, Fy=0.0,
                                  coarse_route_mode=:leaf_equivalent)

Run `steps` of both route-native and oracle leaf paths from identical
equilibrium init. Return a NamedTuple with:
- max_abs   : maximum |F_route - F_oracle| over (i,j,q) (leaf-grid)
- at        : (i, j, q) of max_abs
- max_du    : max |u_route - u_oracle| over fluid cells
- at_u      : (i, j) of max_du
- rho_route, ux_route, uy_route
- rho_oracle, ux_oracle, uy_oracle
- leaf_route, leaf_oracle
"""
function diff_route_vs_oracle_one_step(Nx::Int, Ny::Int,
                                       patch_i_range::UnitRange{Int},
                                       patch_j_range::UnitRange{Int},
                                       is_solid::AbstractArray{Bool,2};
                                       steps::Int=1, rho::Float64=1.0,
                                       u::Float64=0.0,
                                       Fx::Float64=0.0, Fy::Float64=0.0,
                                       omega::Float64=_OMEGA_REST,
                                       coarse_route_mode::Symbol=:leaf_equivalent)
    # route-native path
    cR, pR, topo = _make_state_2d(Nx, Ny, patch_i_range, patch_j_range;
                                  rho=rho, u=u,
                                  coarse_route_mode=coarse_route_mode)
    cR_next = similar(cR)
    pR_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range)
    for _ in 1:steps
        cR, pR = _step_route_native!(cR, cR_next, pR, pR_next, topo, is_solid;
                                     Fx=Fx, Fy=Fy, omega=omega)
        cR_next = similar(cR)
        pR_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range)
    end

    # oracle leaf path
    cO, pO, _ = _make_state_2d(Nx, Ny, patch_i_range, patch_j_range;
                               rho=rho, u=u,
                               coarse_route_mode=coarse_route_mode)
    cO_next = similar(cO)
    pO_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range)
    leafA = zeros(Float64, 2 * Nx, 2 * Ny, 9)
    leafB = similar(leafA)
    for _ in 1:steps
        cO, pO = _step_oracle_leaf!(leafA, leafB, cO, pO, cO_next, pO_next,
                                    is_solid; Fx=Fx, Fy=Fy, omega=omega)
        cO_next = similar(cO)
        pO_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range)
    end

    rhoR, uxR, uyR, leafR = _leaf_velocity_field(cR, pR, is_solid)
    rhoO, uxO, uyO, leafO = _leaf_velocity_field(cO, pO, is_solid)

    diff = leafR .- leafO
    abs_diff = abs.(diff)
    max_abs, lin = findmax(abs_diff)
    imax, jmax, qmax = Tuple(CartesianIndices(abs_diff)[lin])

    diff_u = sqrt.((uxR .- uxO).^2 .+ (uyR .- uyO).^2)
    # NaN at solid → replace with -1 to skip in findmax
    diff_u_safe = map(x -> isnan(x) ? -1.0 : x, diff_u)
    max_du, linu = findmax(diff_u_safe)
    iu, ju = Tuple(CartesianIndices(diff_u_safe)[linu])

    return (
        max_abs=max_abs, at=(imax, jmax, qmax),
        max_du=max_du, at_u=(iu, ju),
        rho_route=rhoR, ux_route=uxR, uy_route=uyR,
        rho_oracle=rhoO, ux_oracle=uxO, uy_oracle=uyO,
        leaf_route=leafR, leaf_oracle=leafO,
    )
end

# ---------- canaries: rest-state with obstacle + interface ----------

@testset "Conservative tree obstacle + interface 2D (Phase A)" begin

    # Common rest-state spec: ρ=1, u=0, periodic_x + wall_y, run 20 steps,
    # expect mass conservation to roundoff and zero velocity to roundoff.

    @testset "rest: obstacle entirely inside fine patch" begin
        Nx, Ny = 12, 8
        patch_i_range = 4:9
        patch_j_range = 3:6
        # Patch leaf range: i in 7:18, j in 5:12.
        # Obstacle on fine cells inside patch: leaf 11:14 × 9:10 (4×2 fine).
        is_solid = falses(2 * Nx, 2 * Ny)
        is_solid[11:14, 9:10] .= true

        coarse, patch, topology = _make_state_2d(
            Nx, Ny, patch_i_range, patch_j_range)
        coarse_next = similar(coarse)
        patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range)

        # constraint must be satisfied
        @test_nowarn KrakeNN._check_route_solid_mask_layout(
            topology, coarse, patch, is_solid)

        leaf0 = _composite_to_leaf(coarse, patch)
        m_initial = sum(@view leaf0[.!is_solid, :])

        for _ in 1:20
            coarse, patch = _step_route_native!(
                coarse, coarse_next, patch, patch_next, topology, is_solid)
            coarse_next = similar(coarse)
            patch_next = create_conservative_tree_patch_2d(
                patch_i_range, patch_j_range)
        end

        leafN = _composite_to_leaf(coarse, patch)
        m_final = sum(@view leafN[.!is_solid, :])

        rho, ux, uy, _ = _leaf_velocity_field(coarse, patch, is_solid)
        max_u = maximum(filter(!isnan, sqrt.(ux.^2 .+ uy.^2)))
        max_drho = maximum(filter(!isnan, abs.(rho .- 1.0)))

        @test abs(m_final - m_initial) / m_initial < 1e-12
        @test max_u < 1e-12
        @test max_drho < 1e-12
    end

    @testset "rest: obstacle entirely coarse, far from interface" begin
        Nx, Ny = 12, 8
        patch_i_range = 4:7
        patch_j_range = 3:6
        # Obstacle is one full coarse cell (10, 3) → leaf 19:20 × 5:6.
        # Coarse cell (10, 3) has all 4 leaf children solid ✓.
        # Patch is far away (parent_i_range=4:7).
        is_solid = falses(2 * Nx, 2 * Ny)
        is_solid[19:20, 5:6] .= true

        coarse, patch, topology = _make_state_2d(
            Nx, Ny, patch_i_range, patch_j_range)
        coarse_next = similar(coarse)
        patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range)

        @test_nowarn KrakeNN._check_route_solid_mask_layout(
            topology, coarse, patch, is_solid)

        leaf0 = _composite_to_leaf(coarse, patch)
        m_initial = sum(@view leaf0[.!is_solid, :])

        for _ in 1:20
            coarse, patch = _step_route_native!(
                coarse, coarse_next, patch, patch_next, topology, is_solid)
            coarse_next = similar(coarse)
            patch_next = create_conservative_tree_patch_2d(
                patch_i_range, patch_j_range)
        end

        leafN = _composite_to_leaf(coarse, patch)
        m_final = sum(@view leafN[.!is_solid, :])

        rho, ux, uy, _ = _leaf_velocity_field(coarse, patch, is_solid)
        max_u = maximum(filter(!isnan, sqrt.(ux.^2 .+ uy.^2)))

        @test abs(m_final - m_initial) / m_initial < 1e-12
        @test max_u < 1e-12
    end

    @testset "rest: obstacle straddling interface (coarse cell + fine cells)" begin
        Nx, Ny = 12, 8
        patch_i_range = 4:9
        patch_j_range = 3:6
        # Patch leaf i in 7:18, j in 5:12.
        # Obstacle: full coarse cell (3, 3) — leaf 5:6 × 5:6 (4 children solid)
        # PLUS fine cells immediately east of patch boundary at leaf 7:8 × 5:6.
        # Both pieces are at j=5:6.
        # On fine side: leaf 7:8 × 5:6 = inside patch (i_parent=4, j_parent=3),
        # they are 4 fine cells, can be partially solid (no constraint inside patch).
        is_solid = falses(2 * Nx, 2 * Ny)
        is_solid[5:6, 5:6] .= true   # full coarse cell (3, 3)
        is_solid[7:8, 5:6] .= true   # 4 fine cells inside patch, west edge

        coarse, patch, topology = _make_state_2d(
            Nx, Ny, patch_i_range, patch_j_range)
        coarse_next = similar(coarse)
        patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range)

        @test_nowarn KrakeNN._check_route_solid_mask_layout(
            topology, coarse, patch, is_solid)

        leaf0 = _composite_to_leaf(coarse, patch)
        m_initial = sum(@view leaf0[.!is_solid, :])

        for _ in 1:20
            coarse, patch = _step_route_native!(
                coarse, coarse_next, patch, patch_next, topology, is_solid)
            coarse_next = similar(coarse)
            patch_next = create_conservative_tree_patch_2d(
                patch_i_range, patch_j_range)
        end

        leafN = _composite_to_leaf(coarse, patch)
        m_final = sum(@view leafN[.!is_solid, :])

        rho, ux, uy, _ = _leaf_velocity_field(coarse, patch, is_solid)
        max_u = maximum(filter(!isnan, sqrt.(ux.^2 .+ uy.^2)))

        @test abs(m_final - m_initial) / m_initial < 1e-12
        @test max_u < 1e-12
    end

    @testset "rest: obstacle adjacent to interface (1 fine cell into patch)" begin
        Nx, Ny = 12, 8
        patch_i_range = 4:9
        patch_j_range = 3:6
        # Obstacle = 1 column of 4 fine cells at the very west edge of patch:
        # leaf 7 × 5:8. Adjacent to coarse cell (3, 3..4) which is FLUID.
        is_solid = falses(2 * Nx, 2 * Ny)
        is_solid[7, 5:8] .= true

        coarse, patch, topology = _make_state_2d(
            Nx, Ny, patch_i_range, patch_j_range)
        coarse_next = similar(coarse)
        patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range)

        @test_nowarn KrakeNN._check_route_solid_mask_layout(
            topology, coarse, patch, is_solid)

        leaf0 = _composite_to_leaf(coarse, patch)
        m_initial = sum(@view leaf0[.!is_solid, :])

        for _ in 1:20
            coarse, patch = _step_route_native!(
                coarse, coarse_next, patch, patch_next, topology, is_solid)
            coarse_next = similar(coarse)
            patch_next = create_conservative_tree_patch_2d(
                patch_i_range, patch_j_range)
        end

        leafN = _composite_to_leaf(coarse, patch)
        m_final = sum(@view leafN[.!is_solid, :])

        rho, ux, uy, _ = _leaf_velocity_field(coarse, patch, is_solid)
        max_u = maximum(filter(!isnan, sqrt.(ux.^2 .+ uy.^2)))

        @test abs(m_final - m_initial) / m_initial < 1e-12
        @test max_u < 1e-12
    end

    # ---------- A.3: 1/2/5/20-step diff route-native vs oracle leaf ----------
    # These are not strict pass/fail tests; they print where the divergence
    # appears and how it grows. They drive the B-1 vs B-2 decision.

    @testset "diff route vs oracle: 1/2/5/20 step localisation (rest)" begin
        Nx, Ny = 12, 8
        patch_i_range = 4:9
        patch_j_range = 3:6
        # Use the most revealing case: straddling obstacle.
        is_solid = falses(2 * Nx, 2 * Ny)
        is_solid[5:6, 5:6] .= true
        is_solid[7:8, 5:6] .= true

        println("\n=== Phase A.3 diff route-native vs oracle leaf (straddling obstacle, rest) ===")
        for steps in (1, 2, 5, 20)
            r = diff_route_vs_oracle_one_step(
                Nx, Ny, patch_i_range, patch_j_range, is_solid;
                steps=steps, u=0.0)
            println(@sprintf("  steps=%2d  max|ΔF|=%.3e at (i=%d,j=%d,q=%d)  max|Δu|=%.3e at (i=%d,j=%d)",
                             steps, r.max_abs, r.at[1], r.at[2], r.at[3],
                             r.max_du, r.at_u[1], r.at_u[2]))
        end
        # Sanity assertions: the diff exists or it doesn't, both informative.
        @test true
    end

    @testset "diff route vs oracle: 1/2/5/20 step localisation (uniform u=0.01)" begin
        Nx, Ny = 12, 8
        patch_i_range = 4:9
        patch_j_range = 3:6
        # Uniform inflow makes the transport divergence visible at step 1.
        is_solid = falses(2 * Nx, 2 * Ny)
        is_solid[5:6, 5:6] .= true
        is_solid[7:8, 5:6] .= true

        println("\n=== Phase A.3 diff route-native vs oracle leaf (straddling obstacle, u=0.01) ===")
        for steps in (1, 2, 5, 20)
            r = diff_route_vs_oracle_one_step(
                Nx, Ny, patch_i_range, patch_j_range, is_solid;
                steps=steps, u=0.01)
            println(@sprintf("  steps=%2d  max|ΔF|=%.3e at (i=%d,j=%d,q=%d)  max|Δu|=%.3e at (i=%d,j=%d)",
                             steps, r.max_abs, r.at[1], r.at[2], r.at[3],
                             r.max_du, r.at_u[1], r.at_u[2]))
        end
        @test true
    end

    @testset "diff route vs oracle: obstacle inside patch, u=0.01" begin
        Nx, Ny = 12, 8
        patch_i_range = 4:9
        patch_j_range = 3:6
        is_solid = falses(2 * Nx, 2 * Ny)
        is_solid[11:14, 9:10] .= true

        println("\n=== Phase A.3 diff route-native vs oracle leaf (obstacle in fine, u=0.01) ===")
        for steps in (1, 2, 5, 20)
            r = diff_route_vs_oracle_one_step(
                Nx, Ny, patch_i_range, patch_j_range, is_solid;
                steps=steps, u=0.01)
            println(@sprintf("  steps=%2d  max|ΔF|=%.3e at (i=%d,j=%d,q=%d)  max|Δu|=%.3e at (i=%d,j=%d)",
                             steps, r.max_abs, r.at[1], r.at[2], r.at[3],
                             r.max_du, r.at_u[1], r.at_u[2]))
        end
        @test true
    end

    @testset "diff route vs oracle: cylinder regime with Fx forcing" begin
        # Reproduce the cylinder Cd test geometry (Nx=24, Ny=14, patch 8:17 x 4:11,
        # cylinder of radius 3 at center) and watch how route-native diverges
        # from the leaf oracle under Fx forcing. This is the regime that yields
        # the ~1.86x Cd ratio in benchmarks.
        Nx, Ny = 24, 14
        patch_i_range = 8:17
        patch_j_range = 4:11
        cx_leaf = (2 * Nx + 1) / 2
        cy_leaf = (2 * Ny + 1) / 2
        is_solid = cylinder_solid_mask_leaf_2d(2 * Nx, 2 * Ny, cx_leaf, cy_leaf, 3.0)
        Fx = 2e-5
        omega = 1.0

        println("\n=== Phase A.3 diff route-native vs oracle leaf (cylinder, Fx=2e-5) ===")
        for steps in (1, 5, 20, 100)
            r = diff_route_vs_oracle_one_step(
                Nx, Ny, patch_i_range, patch_j_range, is_solid;
                steps=steps, u=0.0, Fx=Fx, Fy=0.0, omega=omega)
            println(@sprintf("  steps=%3d  max|ΔF|=%.3e at (i=%d,j=%d,q=%d)  max|Δu|=%.3e at (i=%d,j=%d)",
                             steps, r.max_abs, r.at[1], r.at[2], r.at[3],
                             r.max_du, r.at_u[1], r.at_u[2]))
        end
        @test true
    end

    # ---------- A.4: MEA drag isolation ----------

    @testset "A.4: MEA drag is identical when given identical Fpre/Fpost" begin
        # Build a non-trivial leaf field with a cylinder obstacle.
        Nx, Ny = 24, 14
        cx_leaf = (2 * Nx + 1) / 2
        cy_leaf = (2 * Ny + 1) / 2
        is_solid = cylinder_solid_mask_leaf_2d(2 * Nx, 2 * Ny,
                                               cx_leaf, cy_leaf, 3.0)
        leaf = zeros(Float64, 2 * Nx, 2 * Ny, 9)
        fill_equilibrium_integrated_D2Q9!(leaf, _VF, 1.0, 0.02, 0.0)
        leaf_post = copy(leaf)
        # Apply a single oracle leaf streaming step
        KrakeNN.stream_periodic_x_wall_y_solid_F_2d!(leaf_post, leaf, is_solid)

        # Both calls operate on the SAME leaf arrays, so they MUST agree.
        d1 = compute_drag_mea_solid_F_2d(leaf, leaf_post, is_solid)
        d2 = compute_drag_mea_solid_F_2d(leaf, leaf_post, is_solid)
        @test d1.Fx == d2.Fx
        @test d1.Fy == d2.Fy

        # Sanity: drag is finite and non-trivial.
        @test isfinite(d1.Fx)
        @test isfinite(d1.Fy)
        @test abs(d1.Fx) > 0
    end
end
