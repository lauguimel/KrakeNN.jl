using Test
using KrakeNN
using KernelAbstractions

# ==========================================================================
# Kernel DSL — smoke + oracle tests.
#
# 1. Skeleton sanity: construct a spec, check arg ordering, run a trivial
#    pull-only kernel and compare against a hand-written loop.
# 2. Oracle fused_bgk: build a DSL kernel with the same brick sequence
#    as `fused_bgk_step_kernel!` and assert bit-exact equality.
# 3. Oracle fused_trt: idem for `fused_trt_step_kernel!`.
# 4. Oracle fused_trt_libb (buggy): idem for `fused_trt_libb_step_kernel!`.
#    Reproducing the bug proves the DSL captures the current semantics.
# 5. Fix: spec TRT_LIBB_V2 (solid-inert + recompute-moments); the planar
#    Couette canary must pass.
# ==========================================================================

function _init_equilibrium_2d(Nx, Ny, FT; ρ=1.0, ux=0.0, uy=0.0)
    f = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f[i, j, q] = KrakeNN.equilibrium(D2Q9(), FT(ρ), FT(ux), FT(uy), q)
    end
    return f
end

@testset "Kernel DSL" begin

    # ------------------------------------------------------------
    # 1. Skeleton sanity
    # ------------------------------------------------------------
    @testset "skeleton — pull + write bit-exact" begin
        spec = KrakeNN.LBMSpec(KrakeNN.PullHalfwayBB(), KrakeNN.WriteF())
        @test KrakeNN.spec_args(spec) == [:f_out, :f_in, :Nx, :Ny]

        Nx, Ny = 6, 5
        f_in = _init_equilibrium_2d(Nx, Ny, Float64; ρ=1.0, ux=0.01)
        f_out_dsl = similar(f_in)
        f_out_ref = similar(f_in)

        kernel! = KrakeNN.build_lbm_kernel(CPU(), spec)
        kernel!(f_out_dsl, f_in, Nx, Ny; ndrange=(Nx, Ny))
        KernelAbstractions.synchronize(CPU())

        for j in 1:Ny, i in 1:Nx
            f_out_ref[i, j, 1] = f_in[i, j, 1]
            f_out_ref[i, j, 2] = i > 1  ? f_in[i-1, j,   2] : f_in[i, j, 4]
            f_out_ref[i, j, 3] = j > 1  ? f_in[i,   j-1, 3] : f_in[i, j, 5]
            f_out_ref[i, j, 4] = i < Nx ? f_in[i+1, j,   4] : f_in[i, j, 2]
            f_out_ref[i, j, 5] = j < Ny ? f_in[i,   j+1, 5] : f_in[i, j, 3]
            f_out_ref[i, j, 6] = (i>1  && j>1)  ? f_in[i-1,j-1,6] : f_in[i,j,8]
            f_out_ref[i, j, 7] = (i<Nx && j>1)  ? f_in[i+1,j-1,7] : f_in[i,j,9]
            f_out_ref[i, j, 8] = (i<Nx && j<Ny) ? f_in[i+1,j+1,8] : f_in[i,j,6]
            f_out_ref[i, j, 9] = (i>1  && j<Ny) ? f_in[i-1,j+1,9] : f_in[i,j,7]
        end
        @test all(f_out_dsl .== f_out_ref)
    end

    @testset "cache: same spec returns identical compiled kernel" begin
        spec = KrakeNN.LBMSpec(KrakeNN.PullHalfwayBB(), KrakeNN.WriteF())
        k1 = KrakeNN.build_lbm_kernel(CPU(), spec)
        k2 = KrakeNN.build_lbm_kernel(CPU(), spec)
        @test k1 === k2
    end

    # ------------------------------------------------------------
    # Shared setup for oracle tests
    # ------------------------------------------------------------
    # A small domain with a solid blob in the middle, non-trivial
    # initial state (equilibrium at shifted velocity). Run one step,
    # compare DSL kernel output to hand-written fused kernel output.
    Nx, Ny = 8, 8
    FT = Float64

    function make_test_state()
        f_in = _init_equilibrium_2d(Nx, Ny, FT; ρ=1.0, ux=0.02, uy=-0.01)
        is_solid = zeros(Bool, Nx, Ny)
        is_solid[4:5, 4:5] .= true
        return f_in, is_solid
    end

    # ------------------------------------------------------------
    # 2. Oracle: fused_bgk
    # ------------------------------------------------------------
    @testset "oracle fused_bgk — bit-exact" begin
        f_in, is_solid = make_test_state()
        ν = 0.1; ω = FT(1.0 / (3ν + 0.5))

        # Hand-written reference
        f_ref  = similar(f_in); ρ_ref = ones(FT, Nx, Ny)
        ux_ref = zeros(FT, Nx, Ny); uy_ref = zeros(FT, Nx, Ny)
        fused_bgk_step!(f_ref, f_in, ρ_ref, ux_ref, uy_ref,
                        is_solid, Nx, Ny, ω)
        KernelAbstractions.synchronize(CPU())

        # DSL
        spec = KrakeNN.LBMSpec(KrakeNN.PullHalfwayBB(), KrakeNN.SolidSwapBB(),
                              KrakeNN.Moments(), KrakeNN.CollideBGKDirect(),
                              KrakeNN.WriteMoments())
        f_dsl  = similar(f_in); ρ_dsl = ones(FT, Nx, Ny)
        ux_dsl = zeros(FT, Nx, Ny); uy_dsl = zeros(FT, Nx, Ny)
        kernel! = KrakeNN.build_lbm_kernel(CPU(), spec)
        # spec_args canonical order:
        #   f_out, ρ_out, ux_out, uy_out, f_in, is_solid, Nx, Ny, ω
        kernel!(f_dsl, ρ_dsl, ux_dsl, uy_dsl, f_in, is_solid,
                Nx, Ny, ω; ndrange=(Nx, Ny))
        KernelAbstractions.synchronize(CPU())

        @test KrakeNN.spec_args(spec) ==
            [:f_out, :ρ_out, :ux_out, :uy_out, :f_in, :is_solid, :Nx, :Ny, :ω]
        @test all(f_dsl .== f_ref)
        @test all(ρ_dsl .== ρ_ref)
        @test all(ux_dsl .== ux_ref)
        @test all(uy_dsl .== uy_ref)
    end

    # ------------------------------------------------------------
    # 3. Oracle: fused_trt
    # ------------------------------------------------------------
    @testset "oracle fused_trt — bit-exact" begin
        f_in, is_solid = make_test_state()
        ν = 0.1
        s_plus, s_minus = KrakeNN.trt_rates(ν)
        sp = FT(s_plus); sm = FT(s_minus)

        f_ref  = similar(f_in); ρ_ref = ones(FT, Nx, Ny)
        ux_ref = zeros(FT, Nx, Ny); uy_ref = zeros(FT, Nx, Ny)
        fused_trt_step!(f_ref, f_in, ρ_ref, ux_ref, uy_ref,
                        is_solid, Nx, Ny, ν)
        KernelAbstractions.synchronize(CPU())

        spec = KrakeNN.LBMSpec(KrakeNN.PullHalfwayBB(), KrakeNN.SolidSwapBB(),
                              KrakeNN.Moments(), KrakeNN.CollideTRTDirect(),
                              KrakeNN.WriteMoments())
        f_dsl  = similar(f_in); ρ_dsl = ones(FT, Nx, Ny)
        ux_dsl = zeros(FT, Nx, Ny); uy_dsl = zeros(FT, Nx, Ny)
        kernel! = KrakeNN.build_lbm_kernel(CPU(), spec)
        kernel!(f_dsl, ρ_dsl, ux_dsl, uy_dsl, f_in, is_solid,
                Nx, Ny, sp, sm; ndrange=(Nx, Ny))
        KernelAbstractions.synchronize(CPU())

        @test all(f_dsl .== f_ref)
        @test all(ρ_dsl .== ρ_ref)
        @test all(ux_dsl .== ux_ref)
        @test all(uy_dsl .== uy_ref)
    end

    # ------------------------------------------------------------
    # 4. Oracle: fused_trt_libb (BUGGY — reproduces the double-BC error)
    # ------------------------------------------------------------
    @testset "oracle fused_trt_libb (buggy) — bit-exact" begin
        f_in, is_solid = make_test_state()
        ν = 0.1
        # Put a fake cut link so ApplyLiBB is exercised: fluid cell (6,6)
        # thinks it has a wall on the east at q_w = 0.3 (fake, just to
        # exercise the overwrite path).
        q_wall  = zeros(FT, Nx, Ny, 9)
        uw_x    = zeros(FT, Nx, Ny, 9)
        uw_y    = zeros(FT, Nx, Ny, 9)
        q_wall[6, 6, 2] = FT(0.3)   # east link out of (6,6) cut at 0.3
        uw_x[6, 6, 2]   = FT(0.01)
        s_plus, s_minus = KrakeNN.trt_rates(ν)
        sp = FT(s_plus); sm = FT(s_minus)

        f_ref  = similar(f_in); ρ_ref = ones(FT, Nx, Ny)
        ux_ref = zeros(FT, Nx, Ny); uy_ref = zeros(FT, Nx, Ny)
        fused_trt_libb_step!(f_ref, f_in, ρ_ref, ux_ref, uy_ref,
                              is_solid, q_wall, uw_x, uw_y, Nx, Ny, ν)
        KernelAbstractions.synchronize(CPU())

        spec = KrakeNN.LBMSpec(KrakeNN.PullHalfwayBB(), KrakeNN.SolidSwapBB(),
                              KrakeNN.Moments(), KrakeNN.CollideTRT(),
                              KrakeNN.ApplyLiBB(),
                              KrakeNN.WriteFLiBB(), KrakeNN.WriteMoments())
        f_dsl  = similar(f_in); ρ_dsl = ones(FT, Nx, Ny)
        ux_dsl = zeros(FT, Nx, Ny); uy_dsl = zeros(FT, Nx, Ny)
        kernel! = KrakeNN.build_lbm_kernel(CPU(), spec)
        kernel!(f_dsl, ρ_dsl, ux_dsl, uy_dsl, f_in, is_solid,
                q_wall, uw_x, uw_y, Nx, Ny, sp, sm; ndrange=(Nx, Ny))
        KernelAbstractions.synchronize(CPU())

        @test KrakeNN.spec_args(spec) ==
            [:f_out, :ρ_out, :ux_out, :uy_out,
             :f_in, :is_solid, :q_wall, :uw_link_x, :uw_link_y,
             :Nx, :Ny, :s_plus, :s_minus]
        @test all(f_dsl .== f_ref)
        @test all(ρ_dsl .== ρ_ref)
        @test all(ux_dsl .== ux_ref)
        @test all(uy_dsl .== uy_ref)
    end

end
