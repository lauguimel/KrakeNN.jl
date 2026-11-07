using Test
using KrakeNN, KernelAbstractions
using KrakeNN: LBMSpec, PullSLBM, SolidInert, ApplyLiBBPrePhase,
              Moments, CollideTRTDirect, WriteMoments,
              build_lbm_kernel, spec_args

# ===========================================================================
# Schäfer-Turek 2D-1 (Re=20) on a SLBM stretched box with LI-BB.
#
# Modular composition:
#   streaming  = PullSLBM (DSL brick)
#   solid      = SolidInert (DSL brick)
#   BC pre     = ApplyLiBBPrePhase (DSL brick)
#   moments    = Moments (DSL brick)
#   collision  = CollideTRTDirect (DSL brick, Λ=3/16)
#   write      = WriteMoments (DSL brick)
#   inlet/out  = BCSpec2D + apply_bc_rebuild_2d! (modular BC system)
#   drag       = compute_drag_libb_mei_2d (existing MEA)
#
# All components are reused from the existing modular infrastructure.
# Only the streaming brick changes: PullSLBM instead of PullHalfwayBB.
# ===========================================================================

function run_slbm_cylinder_stretched(;
        D_lu::Int=20,
        Re::Float64=20.0,
        u_max::Float64=0.1,
        aspect::Float64=2.2/0.41,
        blockage::Float64=0.1/0.41,
        x_stretch::Float64=1.5,
        y_stretch::Float64=1.5,
        steps::Int=50_000,
        drag_avg_window::Int=5_000)

    # --- Physical-to-lattice ---
    D_phys = 0.1
    H_phys = D_phys / blockage
    L_phys = H_phys * aspect
    Ny = round(Int, D_lu / blockage)
    Nx = round(Int, Ny * aspect)
    dx_phys = H_phys / (Ny - 1)
    u_mean = 2 * u_max / 3
    ν = u_mean * D_lu / Re
    cx_phys = 0.2; cy_phys = 0.2
    cx_lu = cx_phys / dx_phys + 1; cy_lu = cy_phys / dx_phys + 1
    R_phys = D_phys / 2

    # --- Stretched mesh ---
    mesh = stretched_box_mesh(; x_min=0.0, x_max=L_phys,
                                y_min=0.0, y_max=H_phys,
                                Nx=Nx, Ny=Ny,
                                x_stretch=x_stretch, y_stretch=y_stretch,
                                x_stretch_dir=:both, y_stretch_dir=:both,
                                FT=Float64)
    geom = build_slbm_geometry(mesh)

    # --- Solid mask + q_wall in physical space ---
    Nξ, Nη = mesh.Nξ, mesh.Nη
    is_solid = zeros(Bool, Nξ, Nη)
    R² = R_phys^2
    for j in 1:Nη, i in 1:Nξ
        dx = mesh.X[i, j] - cx_phys
        dy = mesh.Y[i, j] - cy_phys
        if dx * dx + dy * dy ≤ R²
            is_solid[i, j] = true
        end
    end

    q_wall, uw_link_x, uw_link_y = precompute_q_wall_slbm_cylinder_2d(
        mesh, is_solid, cx_phys, cy_phys, R_phys; omega_inner=0.0)

    # --- DSL spec: SLBM + LI-BB V2 + TRT ---
    spec = LBMSpec(PullSLBM(), SolidInert(), ApplyLiBBPrePhase(),
                   Moments(), CollideTRTDirect(), WriteMoments())
    kernel! = build_lbm_kernel(CPU(), spec)
    s_plus, s_minus = trt_rates(ν)

    # --- Modular BCs: parabolic inlet (west), pressure outlet (east) ---
    u_inlet = zeros(Nη)
    for j in 1:Nη
        y = mesh.Y[1, j]
        u_inlet[j] = 4 * u_max * y * (H_phys - y) / H_phys^2
    end
    bc = BCSpec2D(west=ZouHeVelocity(u_inlet), east=ZouHePressure(1.0))

    # --- Allocate ---
    f_in = zeros(Float64, Nξ, Nη, 9)
    f_out = similar(f_in)
    ρ = ones(Nξ, Nη); ux = zeros(Nξ, Nη); uy = zeros(Nξ, Nη)
    for j in 1:Nη, i in 1:Nξ, q in 1:9
        f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
    end

    # --- Time loop ---
    Cd_sum = 0.0; n_avg = 0
    for step in 1:steps
        # 1. Fused SLBM + LI-BB + TRT step (DSL-compiled kernel)
        kernel!(f_out, ρ, ux, uy, f_in, is_solid, q_wall, uw_link_x, uw_link_y,
                geom.i_dep, geom.j_dep, Nξ, Nη, s_plus, s_minus,
                geom.periodic_ξ, geom.periodic_η;
                ndrange=(Nξ, Nη))

        # 2. Modular BC rebuild (inlet + outlet)
        apply_bc_rebuild_2d!(f_out, f_in, bc, ν, Nξ, Nη)

        f_in, f_out = f_out, f_in

        # 3. Drag computation (existing MEA on cut links)
        if step > steps - drag_avg_window
            Fx, Fy = compute_drag_libb_mei_2d(f_in, q_wall, uw_link_x, uw_link_y,
                                               Nξ, Nη)
            Cd = 2 * Fx / (ρ[1, Nη÷2] * u_mean^2 * D_lu)
            Cd_sum += Cd; n_avg += 1
        end
    end

    Cd_avg = Cd_sum / max(n_avg, 1)
    n_solid = sum(is_solid)
    n_cut = sum(q_wall .> 0)
    return (; Cd=Cd_avg, ν, Nx=Nξ, Ny=Nη, D_lu, n_solid, n_cut, mesh)
end

@testset "SLBM Schäfer-Turek 2D-1 (Re=20)" begin

    @testset "Stretched box D=$D" for D in [20]
        result = run_slbm_cylinder_stretched(; D_lu=D, steps=30_000,
                                               drag_avg_window=5_000)
        Cd_ref = 5.58
        err = abs(result.Cd - Cd_ref) / Cd_ref
        @info "SLBM ST 2D-1 D=$(result.D_lu): Cd=$(round(result.Cd, digits=3)), " *
              "err=$(round(100*err, digits=1))%, " *
              "grid=$(result.Nx)×$(result.Ny), " *
              "solid=$(result.n_solid), cut=$(result.n_cut)"

        @test all(isfinite.([result.Cd]))
        @test result.n_cut > 0
        @test result.Cd > 0
    end
end
