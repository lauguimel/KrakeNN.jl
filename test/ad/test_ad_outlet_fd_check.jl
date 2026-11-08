# AD mirror of the east extrapolation pressure outlet (issue #118).
#
# `ad_apply_zou_he_rebuild!` / `ad_apply_zou_he_rebuild_nufield!`
# (src/ad/ad_step.jl) and the fused rebuild of `ad_ve_coupled_step!`
# (src/ad/ad_ve_step.jl) mirror `NeqExtrapolationPressure`: the outlet column
# is rebuilt, without collision, from the post-collision column Nx-1. Checked
# on the 28 x 12 cut-cylinder channel, from a perturbed Poiseuille state:
#   (a) mirror parity: 200 steps of `ad_step_nufield!` (uniform rates) and of
#       `ad_step!` against production (`fused_trt_libb_v2_step!` +
#       `rebuild_inlet_outlet_libb_2d!`), gate 1e-14 absolute (measured
#       5.6e-16; the bulk mirror itself is a round-off mirror);
#   (b) one-step directional derivative of `ad_step_nufield!` with
#       non-uniform per-row rates: Enzyme forward JVP against a central finite
#       difference (step 1e-5) in the direction (df, dnu), once with df on
#       column Nx-1 only (the column the outlet reads) and once everywhere.
#       Gate 1e-7 relative, on the whole output and on the outlet column
#       (measured <= 5.5e-10: FD truncation and round-off);
#   (c) reverse mode in the same directions: <w, J d> == <J^T w, d>, gate
#       1e-12 relative (measured <= 4.3e-14);
#   (d) VE mirror: after 50 steps of `ad_ve_coupled_step!` on the fast
#       viscoelastic cut-cylinder case, its outlet column equals the
#       production kernel applied to its own column Nx-1, bit for bit.
# Measured on CPU, 2026-11-05, Apple M3 Max, Julia 1.12.5. Enzyme-driven:
# runs in the AD tier.

module KrakenADOutletFDCheck

using Test
using KrakeNN
using Enzyme
using Random
using LinearAlgebra
using KernelAbstractions: CPU

const K = KrakeNN

function _case()
    Nx, Ny = 28, 12
    q_wall, is_solid = K.precompute_q_wall_cylinder(Nx, Ny, 9, 6, 2.5; FT=Float64)
    u_in = 0.04
    u_prof = [4u_in * (j - 1) * (Ny - j) / (Ny - 1)^2 for j in 1:Ny]
    rng = MersenneTwister(3)
    f = zeros(Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        r = 1 + 1e-3 * randn(rng)
        a = u_prof[j] + 1e-3 * randn(rng)
        b = 1e-3 * randn(rng)
        for q in 1:9
            f[i, j, q] = K.feq_2d(Val(q), r, a, b, a * a + b * b)
        end
    end
    return (; Nx, Ny, q_wall, is_solid, u_prof, f, rho_out=1.0, rng)
end

function _parity(c; nsteps=200, ν=0.05)
    (; Nx, Ny, q_wall, is_solid, u_prof, rho_out) = c
    s_plus, s_minus = K.trt_rates(ν; Λ=3/16)
    nuu = fill(ν, Ny)
    ρ = ones(Nx, Ny); ux = zeros(Nx, Ny); uy = zeros(Nx, Ny)
    uw = zeros(Nx, Ny, 9)
    fa = copy(c.f); fb = similar(fa)
    ga = copy(c.f); gb = similar(ga)
    pa = copy(c.f); pb = similar(pa)
    for _ in 1:nsteps
        K.ad_step_nufield!(fb, fa, q_wall, is_solid, u_prof, rho_out, nuu, Nx, Ny)
        K.ad_step!(gb, ga, q_wall, is_solid, u_prof, rho_out, s_plus, s_minus, Nx, Ny)
        fused_trt_libb_v2_step!(pb, pa, ρ, ux, uy, is_solid, q_wall, uw, uw, Nx, Ny, ν)
        rebuild_inlet_outlet_libb_2d!(pb, pa, u_prof, rho_out, ν, Nx, Ny)
        fa, fb = fb, fa; ga, gb = gb, ga; pa, pb = pb, pa
    end
    return (; f=pa, nufield=maximum(abs.(fa .- pa)), uniform=maximum(abs.(ga .- pa)))
end

function _step(c, fin, nu)
    out = zeros(c.Nx, c.Ny, 9)
    K.ad_step_nufield!(out, fin, c.q_wall, c.is_solid, c.u_prof, c.rho_out, nu, c.Nx, c.Ny)
    return out
end

function _directional(c, f0, nuf, df, dnu; ε=1e-5)
    (; Nx, Ny) = c
    out = zeros(Nx, Ny, 9); dout = zeros(Nx, Ny, 9)
    Enzyme.autodiff(Enzyme.Forward, K.ad_step_nufield!, Enzyme.Const,
                    Enzyme.Duplicated(out, dout), Enzyme.Duplicated(copy(f0), copy(df)),
                    Enzyme.Const(c.q_wall), Enzyme.Const(c.is_solid),
                    Enzyme.Const(c.u_prof), Enzyme.Const(c.rho_out),
                    Enzyme.Duplicated(copy(nuf), copy(dnu)),
                    Enzyme.Const(Nx), Enzyme.Const(Ny))
    fd = (_step(c, f0 .+ ε .* df, nuf .+ ε .* dnu) .-
          _step(c, f0 .- ε .* df, nuf .- ε .* dnu)) ./ (2ε)
    rel_all = norm(fd .- dout) / norm(fd)
    rel_out = norm(fd[Nx, :, :] .- dout[Nx, :, :]) / norm(fd[Nx, :, :])

    w = randn(c.rng, Nx, Ny, 9)
    out2 = zeros(Nx, Ny, 9)
    gf = zeros(Nx, Ny, 9); gnu = zeros(Ny)
    Enzyme.autodiff(Enzyme.Reverse, K.ad_step_nufield!, Enzyme.Const,
                    Enzyme.Duplicated(out2, copy(w)), Enzyme.Duplicated(copy(f0), gf),
                    Enzyme.Const(c.q_wall), Enzyme.Const(c.is_solid),
                    Enzyme.Const(c.u_prof), Enzyme.Const(c.rho_out),
                    Enzyme.Duplicated(copy(nuf), gnu),
                    Enzyme.Const(Nx), Enzyme.Const(Ny))
    lhs = dot(w, dout); rhs = dot(gf, df) + dot(gnu, dnu)
    return (; rel_all, rel_out, rev_rel=abs(lhs - rhs) / abs(lhs),
            outlet_norm=norm(fd[Nx, :, :]))
end

function _ve_outlet_parity(; nsteps=50)
    Nx, Ny = 24, 24
    lambda = 0.5; nu_p = 0.02; nu_s = 0.08; Fx = 2e-4
    s_plus, s_minus = K.ad_ve_trt_rates(nu_s)
    p = K.ADVECoupledParams(Nx, Ny, lambda, 0.05, 4, nu_p / lambda, nu_s, Fx,
                            s_plus, s_minus)
    geom = K.ad_ve_build_geom(Nx, Ny, 12.35, 11.65, 5.13; samples=16, u_mean=Fx)
    w = K.ad_ve_initial_state(geom.g, Nx, Ny, 0.05)
    w_out = similar(w)
    for _ in 1:nsteps
        K.ad_ve_coupled_step!(w_out, w, geom.g, geom.q_wall, p, geom.u_profile, 1.0, nothing)
        w, w_out = w_out, w
    end
    n = Nx * Ny
    f = reshape(w[1:9n], Nx, Ny, 9)
    g = copy(f)
    K._apply_bc_2d_east_neq_extrap!(CPU(), g, NeqExtrapolationPressure(1.0), Nx, Ny,
                                    HalfwayBB(), HalfwayBB())
    return (; diff=maximum(abs.(g .- f)), outlet_norm=maximum(abs.(f[Nx, :, :])))
end

@testset "AD mirror of the extrapolation outlet (#118)" begin
    c = _case()
    @testset "(a) mirror parity, 200 steps" begin
        r = _parity(c)
        @test r.nufield <= 1e-14
        @test r.uniform <= 1e-14
        @info "AD outlet mirror parity" r.nufield r.uniform
    end

    f0 = _parity(c).f
    nuf = [0.02 + 0.08 * rand(c.rng) for _ in 1:c.Ny]
    for (name, colonly) in (("df on column Nx-1", true), ("df everywhere", false))
        @testset "(b,c) $name + dnu, non-uniform rates" begin
            df = zeros(c.Nx, c.Ny, 9)
            if colonly
                df[c.Nx-1, :, :] .= 1e-2 .* randn(c.rng, c.Ny, 9)
            else
                df .= 1e-2 .* randn(c.rng, c.Nx, c.Ny, 9)
            end
            dnu = 1e-2 .* randn(c.rng, c.Ny)
            r = _directional(c, f0, nuf, df, dnu)
            @test r.outlet_norm > 1e-3        # the direction moves the outlet
            @test r.rel_all <= 1e-7
            @test r.rel_out <= 1e-7
            @test r.rev_rel <= 1e-12
            @info "AD outlet directional check" name r.rel_all r.rel_out r.rev_rel r.outlet_norm
        end
    end

    @testset "(d) VE mirror outlet == production kernel" begin
        r = _ve_outlet_parity()
        @test r.outlet_norm > 0
        @test r.diff == 0
        @info "AD VE outlet parity" r.diff
    end
end

end # module
