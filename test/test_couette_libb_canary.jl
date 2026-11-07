using Test
using KrakeNN

# ==========================================================================
# Planar Couette — LI-BB + TRT canary
#
# Smoking-gun test for the double-BC bug described in
# docs/design/libb_refactor_plan.md.
#
# Setup
# -----
# Domain Nx × Ny lattice. Rows j=1 and j=Ny are flagged as solid. Rows
# j=2..Ny-1 are fluid. Links from the fluid row j=2 into the south solid
# (q=5, 8, 9) carry q_wall=0.5 (halfway cut). Links from the fluid row
# j=Ny-1 into the north solid (q=3, 6, 7) carry q_wall=0.5. The north
# wall moves at (u_top, 0); the south wall is stationary.
#
# Columns i=1 and i=Nx are ghost columns: after each step we copy
# f_out[1,:,:] ← f_out[Nx-1,:,:] and f_out[Nx,:,:] ← f_out[2,:,:] so
# that the kernel's default halfway-BB fallback at the x boundaries is
# overridden by periodicity. Sampling happens in the interior column.
#
# With q_w = 0.5 uniformly, Bouzidi's formula reduces to halfway BB +
# Ladd's moving-wall correction. The steady-state velocity profile is
# exactly linear:
#
#     u_x(y) = u_top · (y − y_bot) / (y_top − y_bot)
#
# with y_bot = 0.5 and y_top = Ny − 1.5 (the wall sits halfway between
# solid row j=1 / fluid row j=2 and between fluid row j=Ny-1 / solid
# row j=Ny). So for fluid row j (j=2..Ny-1):
#
#     u_ana(j) = u_top · (j − 1.5) / (Ny − 2)
#
# Expected behaviour
# ------------------
# CORRECT kernel: L2(u_x, u_ana) < 0.1 %, u_y ~ machine-eps, u_x > 0
# everywhere in the gap.
#
# CURRENT kernel (bug — 2026-04-15): u_x stays non-negative and u_y
# stays ≈ 0 (so the profile is 1D, as expected from the symmetry), but
# the linear profile is badly CORRUPTED in amplitude: L2_rel ≈ 47 %,
# Linf_rel ≈ 55 %. The profile is flattened — u_x at the top wall only
# reaches ~45 % of u_top instead of saturating near 100 %. This is the
# signature of the double-BC bug: part of the wall-driven momentum is
# silently cancelled by the second application of bounce-back.
#
# (The plan's original description of "u_x NEGATIVE in the bottom half"
# came from a different setup — the corruption mode here is amplitude
# loss, not sign reversal. Same root cause.)
# ==========================================================================

function planar_couette_libb_setup(Nx::Int, Ny::Int, u_top::Real;
                                    FT::Type{<:AbstractFloat}=Float64,
                                    q_w::Real=0.5)
    is_solid = zeros(Bool, Nx, Ny)
    is_solid[:, 1]  .= true
    is_solid[:, Ny] .= true
    q_wall = zeros(FT, Nx, Ny, 9)
    uw_x   = zeros(FT, Nx, Ny, 9)
    uw_y   = zeros(FT, Nx, Ny, 9)
    # Row j=2 (fluid, bottom-adjacent): south-pointing links cross wall
    for i in 1:Nx
        q_wall[i, 2, 5] = FT(q_w)
        q_wall[i, 2, 8] = FT(q_w)
        q_wall[i, 2, 9] = FT(q_w)
    end
    # Row j=Ny-1 (fluid, top-adjacent): north-pointing links cross wall
    for i in 1:Nx
        q_wall[i, Ny-1, 3] = FT(q_w)
        q_wall[i, Ny-1, 6] = FT(q_w)
        q_wall[i, Ny-1, 7] = FT(q_w)
        uw_x[i, Ny-1, 3] = FT(u_top)
        uw_x[i, Ny-1, 6] = FT(u_top)
        uw_x[i, Ny-1, 7] = FT(u_top)
    end
    return is_solid, q_wall, uw_x, uw_y
end

# Copy interior column Nx-1 into ghost column 1, and interior column 2
# into ghost column Nx. Call after every step to fake x-periodicity.
function wrap_periodic_x!(f::AbstractArray{T,3}) where {T}
    Nx, Ny, Q = size(f)
    @inbounds for q in 1:Q, j in 1:Ny
        f[1,  j, q] = f[Nx-1, j, q]
        f[Nx, j, q] = f[2,    j, q]
    end
    return f
end

function run_planar_couette_libb(; Nx::Int=8, Ny::Int=33,
                                    ν::Real=0.1, u_top::Real=0.01,
                                    steps::Int=5000,
                                    q_w::Real=0.5,
                                    stepper! = fused_trt_libb_step!)
    is_solid, qw, uw_x, uw_y = planar_couette_libb_setup(Nx, Ny, u_top; q_w=q_w)
    f_in = zeros(Float64, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_in[i, j, q] = KrakeNN.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
    end
    wrap_periodic_x!(f_in)
    f_out = similar(f_in)
    ρ  = ones(Nx, Ny); ux = zeros(Nx, Ny); uy = zeros(Nx, Ny)
    for _ in 1:steps
        stepper!(f_out, f_in, ρ, ux, uy, is_solid,
                 qw, uw_x, uw_y, Nx, Ny, ν)
        wrap_periodic_x!(f_out)
        f_in, f_out = f_out, f_in
    end
    return (; Nx, Ny, u_top, ρ, ux, uy, is_solid)
end

@testset "Planar Couette — LI-BB canary (documents libb double-BC bug)" begin

    out = run_planar_couette_libb(; Nx=8, Ny=33,
                                    ν=0.1, u_top=0.01, steps=5000)
    Nx, Ny, u_top = out.Nx, out.Ny, out.u_top

    H = Float64(Ny - 2)
    u_ana = [u_top * (j - 1.5) / H for j in 2:Ny-1]

    # Sample interior column (away from ghost columns)
    i_mid = Nx ÷ 2
    u_num = out.ux[i_mid, 2:Ny-1]
    uy_num = out.uy[i_mid, 2:Ny-1]

    errs = u_num .- u_ana
    L2_rel = sqrt(sum(errs .^ 2) / sum(u_ana .^ 2))
    Linf_rel = maximum(abs.(errs)) / u_top
    u_min_bottom_half = minimum(u_num[1:length(u_num)÷2])

    @info "Planar Couette LI-BB canary" Nx Ny steps=5000 L2_rel Linf_rel u_min_bottom_half

    # Hard invariants that hold regardless of bug
    @test all(isfinite.(out.ux))
    @test all(isfinite.(out.uy))
    # Sign of u_x not corrupted in this setup (see header notes)
    @test u_min_bottom_half ≥ 0
    # 1D invariance preserved (no spurious uy)
    @test maximum(abs.(uy_num)) / u_top < 1e-6

    # Bug signature: amplitude corruption. These must FLIP to @test
    # once the refactor lands.
    @test_broken L2_rel < 1e-3
    @test_broken Linf_rel < 2e-3

end

@testset "Planar Couette — LI-BB V2 (DSL refactor, Ginzburg-exact)" begin

    # V2 spec:
    #   PullHalfwayBB → SolidInert | ApplyHalfwayBBPrePhase →
    #                   Moments → CollideTRTDirect → WriteMoments
    #
    # The fix: apply the halfway-BB correction ONCE, pre-collision, via
    # substitution on the pulled populations:
    # - SolidInert puts rest-equilibrium w_q on solid cells (bounces
    #   done locally on the fluid cell via substitution, not via
    #   stored swap at the solid, removing the first bounce-back path).
    # - ApplyHalfwayBBPrePhase replaces each junk pulled pop fp_{q̄}
    #   (sourced from a solid neighbour) with f_in[i,j,q] + δ_{q̄} —
    #   a lag-1 halfway-BB estimate + Ladd moving-wall correction.
    # - CollideTRTDirect then collides on properly-reconstructed pops,
    #   and its post-collision output IS the correctly-bounced pop.
    #   NO post-collision LI-BB overwrite is needed (applying it was
    #   the second bounce-back in the legacy kernel).
    #
    # For q_w = 0.5 this reproduces Ginzburg's halfway-BB + TRT Λ=3/16
    # exactness: L2_rel < 1e-4 at any resolution on CPU Float64. On
    # Metal GPU Float32, error is limited by FP32 accumulation in
    # long-running simulations (observed drift from 6e-4 at Ny=33 to
    # ~8e-3 at Ny=129 with 300k steps, all well within the 0.01 gate).
    #
    # Legacy kernel on this same setup: L2 = 47 %.
    # V2: L2 ≈ 1.7e-5 (CPU Float64, Ny=33, 5000 steps).

    out = run_planar_couette_libb(; Nx=8, Ny=33,
                                    ν=0.1, u_top=0.01, steps=20_000,
                                    stepper! = fused_trt_libb_v2_step!)
    Nx, Ny, u_top = out.Nx, out.Ny, out.u_top

    H = Float64(Ny - 2)
    u_ana = [u_top * (j - 1.5) / H for j in 2:Ny-1]

    i_mid = Nx ÷ 2
    u_num = out.ux[i_mid, 2:Ny-1]
    uy_num = out.uy[i_mid, 2:Ny-1]
    errs = u_num .- u_ana
    L2_rel = sqrt(sum(errs .^ 2) / sum(u_ana .^ 2))
    Linf_rel = maximum(abs.(errs)) / u_top
    u_min_bottom_half = minimum(u_num[1:length(u_num)÷2])

    @info "Planar Couette LI-BB V2" Nx Ny steps=5000 L2_rel Linf_rel u_min_bottom_half

    @test all(isfinite.(out.ux))
    @test all(isfinite.(out.uy))
    @test u_min_bottom_half ≥ 0
    @test maximum(abs.(uy_num)) / u_top < 1e-6
    # Ginzburg-exact gate: halfway-BB + TRT Λ=3/16 on Couette should
    # recover the linear profile to machine-precision-like accuracy
    # (well below 0.1 % L2).
    @test L2_rel < 1e-4
    @test Linf_rel < 1e-4

end

@testset "Planar Couette — LI-BB V2 off-center (q_w ≠ 0.5)" begin

    # Exercises the full Bouzidi interpolation (ApplyLiBBPrePhase brick,
    # both q_w ≤ 0.5 and q_w > 0.5 branches of `_libb_branch`). The
    # wall position is shifted from the halfway midpoint.
    #
    # For q_w uniform on both walls, the wall-to-fluid distance is:
    #   - bottom wall at y = 1 − q_w   (fluid row j=2 sits at y=1)
    #   - top wall at y = (Ny − 2) + q_w
    # Gap H = (Ny − 2 + q_w) − (1 − q_w) = (Ny − 3) + 2·q_w.
    # Analytical u_x(y) = u_top · (y − y_bot) / H.
    # At fluid row j (y = j − 1): u_ana(j) = u_top · (j − 2 + q_w) / H.

    for q_w in (0.3, 0.7)
        Nx, Ny = 8, 33
        u_top = 0.01; ν = 0.1
        out = run_planar_couette_libb(; Nx=Nx, Ny=Ny, ν=ν, u_top=u_top,
                                       steps=10_000, q_w=q_w,
                                       stepper! = fused_trt_libb_v2_step!)

        H = Float64(Ny - 3) + 2 * q_w
        u_ana = [u_top * (j - 2 + q_w) / H for j in 2:Ny-1]

        i_mid = Nx ÷ 2
        u_num = out.ux[i_mid, 2:Ny-1]
        uy_num = out.uy[i_mid, 2:Ny-1]
        errs = u_num .- u_ana
        L2_rel = sqrt(sum(errs .^ 2) / sum(u_ana .^ 2))
        Linf_rel = maximum(abs.(errs)) / u_top

        @info "Planar Couette V2 off-center" q_w Nx Ny L2_rel Linf_rel

        @test all(isfinite.(out.ux))
        @test all(isfinite.(out.uy))
        @test maximum(abs.(uy_num)) / u_top < 1e-4
        # Full Bouzidi + TRT Λ=3/16 on planar Couette is Ginzburg-exact
        # at ALL q_w (once the factor-2 δ scaling bug in _libb_branch
        # for q_w > 0.5 is fixed): L2 reaches machine precision at any
        # off-center wall position.
        @test L2_rel < 1e-4
        @test Linf_rel < 1e-4
    end

end
