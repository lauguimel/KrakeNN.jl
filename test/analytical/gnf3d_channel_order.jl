using Test
using Kraken
using KernelAbstractions

isdefined(@__MODULE__, :_gnf3d_reference) || include("gnf3d_reference.jl")

# GNF3D planar Poiseuille, CPU Float64, diffusive refinement N=16,32,64.
# The exact stress balance and integrated profiles are in gnf3d_reference.jl.
# A 1.8 minimum order on EACH adjacent pair is the prescribed second-order
# acceptance gate. Halfway and off-lattice Carreau walls are both covered.
#
# Accuracy gates are frozen at twice the measured N=64 relative L2 errors,
# measured 2026-11-02, macOS arm64 CPU, Julia 1.12.5, base c99f6f889:
# Carreau q=.5: 2.8960996811013933e-4; q=.3: 4.281689183901035e-4;
# power-law n=.7: 1.7139586652480804e-4; n=1.5: 2.0247376668795252e-4.
# Both power-law profiles pass the same 1.8 gate despite their centreline
# regularity: observed orders (.7)=(2.00024,2.00005), (1.5)=(1.93225,1.94996).
# No smooth-stress companion or relaxed power-law order gate is needed.
#
# Steady-state tolerance was checked separately by doubling the slowest N=64
# Carreau q=.3 case from 56200 to 112400 steps: the L2 error changed by
# 5.926659171938566e-5 relative (<1%). This diagnostic is not repeated here.

@testset "GNF3D analytical channel order" begin
    cases = ((:carreau, 0.5, 5.792199362202787e-4),
             (:carreau, 0.3, 8.563378367802070e-4),
             (:power_thin, 0.5, 3.427917330496161e-4),
             (:power_thick, 0.5, 4.049475333759050e-4))
    for (law, q_w, finest_limit) in cases
        @testset "$law q_w=$q_w" begin
            errors, nu_errors, gaps = Float64[], Float64[], Float64[]
            for N in (16, 32, 64)
                ref = _gnf3d_reference(law, N, q_w)
                # One rest start checks convergence independently of a good
                # velocity guess; other cases start from the exact profile.
                rest_start = law === :carreau && q_w == 0.5 && N == 16
                seconds = @elapsed out = Kraken.run_gnf_channel_3d(;
                    rheology=ref.rheology, N, q_w, F=ref.F,
                    nu_init=ref.nu, u_init=rest_start ? 0.0 : ref.u,
                    max_steps=60_000, tol_ss=1e-9, check_every=200,
                    backend=KernelAbstractions.CPU(), FT=Float64)
                error_u = _gnf3d_rel_l2(out.u, ref.u)
                error_nu = _gnf3d_rel_l2(out.nu, ref.nu)
                push!(errors, error_u); push!(nu_errors, error_nu); push!(gaps, ref.H)
                @info "GNF3D channel accuracy" law q_w N H=ref.H error_u error_nu steps=out.steps seconds
                @test out.converged
                @test out.positions == ref.positions
                for array in (out.fields.f, out.fields.rho, out.fields.ux,
                              out.fields.uy, out.fields.uz, out.fields.sp, out.fields.sm)
                    @test all(isfinite, array)
                end
                # Exact one-dimensional flow has no transverse velocity.
                # 1e-10 relative allows Float64 accumulation roundoff only.
                scale = maximum(abs, ref.u)
                @test maximum(abs, view(out.fields.uy, out.interior...)) <= 1e-10 * scale
                @test maximum(abs, view(out.fields.uz, out.interior...)) <= 1e-10 * scale
                fluid_nu = ((inv.(out.fields.sp) .- 0.5) ./ 3)[.!out.fields.is_solid]
                if law === :carreau
                    @test _gnf3d_inside_bounds(fluid_nu, ref.rheology.eta_inf, ref.rheology.eta_0)
                else
                    # Check EVERY fluid node (including periodic ghosts), not
                    # only the averaged profile: no model clamp may be active.
                    @test _gnf3d_inside_bounds(fluid_nu, ref.rheology.nu_min, ref.rheology.nu_max)
                end
            end
            orders = [log(errors[i] / errors[i+1]) / log(gaps[i+1] / gaps[i]) for i in 1:2]
            @info "GNF3D observed orders" law q_w orders errors nu_errors
            @test all(orders .>= 1.8)
            @test all(diff(nu_errors) .< 0)
            @test errors[end] <= finest_limit
        end
    end
end
