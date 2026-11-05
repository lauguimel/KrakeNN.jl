# Regularized TRT collision at 2D Zou-He boundary nodes (issue #118).
#
# `_trt_collide_local` (src/bc/specs.jl) is the local collision that the 2D
# Zou-He rebuild (src/bc/rebuild_2d.jl) applies once the unknown populations of
# a boundary node are closed. It keeps only the stress part of the
# non-equilibrium populations (regularization):
#
#     Pi_ab    = sum_i c_ia c_ib (f_i - f_i^eq),
#     f_neq_i  = w_i / (2 c_s^4) (c_ia c_ib - c_s^2 delta_ab) Pi_ab,  c_s^2 = 1/3,
#
# then relaxes it with the TRT rates. f_neq_i is even in c_i, so the TRT
# relaxation reduces to one factor: F_i = f_i^eq + (1 - s_plus) f_neq_i.
# Analytical identities, checked on random pre-collision states:
#   (a) reconstruction (s_plus = s_minus = 0): sum f_neq = 0, sum c f_neq = 0
#       and sum c c f_neq = Pi, all three components;
#   (b) any rates: sum F = sum f and sum c F = sum c f (mass and momentum), and
#       sum c c (F - f^eq) = (1 - s_plus) Pi.
#
# Frozen gates: every residual <= 1e-13 in Float64 (absolute; populations are
# O(0.1) and Pi is O(1e-2) here), scaled by eps(Float32)/eps(Float64) in
# Float32. Origin: the identities above. Measured with the fix (CPU, 2026-11-05,
# Apple M3 Max, Julia 1.12.5): <= 4.4e-16 in Float64, <= 2.4e-7 in Float32.
# Measured with the v0.6.0 weights (normal-stress parts -S/9, (2 Pxx - Pyy)/18
# and S/72 with S = Pxx + Pyy, instead of -2S/3, (2 Pxx - Pyy)/6 and S/12) on
# the same states: mass of f_neq 1.3e-3, Pi error 1.2e-2, mass change per
# collision 9.5e-4, Pi error after relaxation 8.5e-3 (momentum was exact).
#
# The two automatic-differentiation copies of this collision
# (`ad_trt_regularized_local` in src/ad/ad_step.jl, `ad_ve_trt_collide_local`
# in src/ad/ad_ve_ops.jl) must satisfy the same identities and agree with the
# production function to round-off (gate 1e-14 absolute, Float64).
#
# The kernel-DSL brick `CollideRegularizedTRTLocal` (src/kernels/dsl/bricks.jl,
# used by `slbm_reg_libb_step_local_2d!`) carried the same weights. On a
# uniform Cartesian mesh the semi-Lagrangian pull is an exact pull, so the
# brick output at an interior node must equal `_trt_collide_local` of the
# pulled populations with that node's rates (gate 1e-14 absolute, Float64).
# Measured with the fix: 2.8e-17 (1.0e-2 with the v0.6.0 brick weights).
#
# CPU only: the defect is algebraic (weights inside an @inline function shared
# by the CPU and GPU kernels).

using Test
using Random
using Kraken

const _REGTRT_GATE = 1e-13
const _REGTRT_PARITY_GATE = 1e-14
const _REGTRT_CX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const _REGTRT_CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)

_regtrt_m1(f, c) = sum(c[q] * f[q] for q in 1:9)
_regtrt_m2(f, a, b) = sum(a[q] * b[q] * f[q] for q in 1:9)

function _regtrt_feq(ρ, ux, uy)
    usq = ux * ux + uy * uy
    return ntuple(q -> Kraken.feq_2d(Val(q), ρ, ux, uy, usq), 9)
end

# Equilibrium at a random (ρ, u) plus a random perturbation of every population
# (all moments, non-hydrodynamic ones included).
function _regtrt_state(rng, ::Type{T}) where {T}
    ρ = T(1 + 0.1 * (rand(rng) - 0.5))
    ux = T(0.1 * (rand(rng) - 0.5))
    uy = T(0.1 * (rand(rng) - 0.5))
    fe = _regtrt_feq(ρ, ux, uy)
    return ntuple(q -> fe[q] + T(1e-2 * (rand(rng) - 0.5)), 9)
end

# Worst residuals of the identities (a) and (b) over n random states.
function _regtrt_residuals(collide, ::Type{T}; n=500, seed=118) where {T}
    rng = MersenneTwister(seed)
    cx, cy = _REGTRT_CX, _REGTRT_CY
    r = Dict(:neq_mass => 0.0, :neq_mom => 0.0, :neq_pi => 0.0,
             :mass => 0.0, :mom => 0.0, :pi => 0.0)
    for _ in 1:n
        f = _regtrt_state(rng, T)
        ρ = sum(f); jx = _regtrt_m1(f, cx); jy = _regtrt_m1(f, cy)
        fe = _regtrt_feq(ρ, jx / ρ, jy / ρ)
        d = f .- fe
        Π = (_regtrt_m2(d, cx, cx), _regtrt_m2(d, cy, cy), _regtrt_m2(d, cx, cy))

        neq = collide(f..., zero(T), zero(T)) .- fe
        r[:neq_mass] = max(r[:neq_mass], abs(sum(neq)))
        r[:neq_mom] = max(r[:neq_mom], abs(_regtrt_m1(neq, cx)), abs(_regtrt_m1(neq, cy)))
        r[:neq_pi] = max(r[:neq_pi], abs(_regtrt_m2(neq, cx, cx) - Π[1]),
                         abs(_regtrt_m2(neq, cy, cy) - Π[2]),
                         abs(_regtrt_m2(neq, cx, cy) - Π[3]))

        s_p = T(0.2 + 1.7 * rand(rng)); s_m = T(0.2 + 1.7 * rand(rng))
        F = collide(f..., s_p, s_m)
        r[:mass] = max(r[:mass], abs(sum(F) - ρ))
        r[:mom] = max(r[:mom], abs(_regtrt_m1(F, cx) - jx), abs(_regtrt_m1(F, cy) - jy))
        dF = F .- fe
        k = one(T) - s_p
        r[:pi] = max(r[:pi], abs(_regtrt_m2(dF, cx, cx) - k * Π[1]),
                     abs(_regtrt_m2(dF, cy, cy) - k * Π[2]),
                     abs(_regtrt_m2(dF, cx, cy) - k * Π[3]))
    end
    return r
end

@testset "Regularized TRT at 2D Zou-He nodes: moments (#118)" begin
    for T in (Float64, Float32)
        gate = _REGTRT_GATE * eps(T) / eps(Float64)
        @testset "_trt_collide_local $T" begin
            r = _regtrt_residuals(Kraken._trt_collide_local, T)
            @test r[:neq_mass] <= gate
            @test r[:neq_mom] <= gate
            @test r[:neq_pi] <= gate
            @test r[:mass] <= gate
            @test r[:mom] <= gate
            @test r[:pi] <= gate
            @info "regularized TRT moments" T residuals = r
        end
    end

    for (name, collide) in (("ad_trt_regularized_local", Kraken.ad_trt_regularized_local),
                            ("ad_ve_trt_collide_local", Kraken.ad_ve_trt_collide_local))
        @testset "$name Float64" begin
            r = _regtrt_residuals(collide, Float64)
            @test r[:neq_mass] <= _REGTRT_GATE
            @test r[:neq_mom] <= _REGTRT_GATE
            @test r[:neq_pi] <= _REGTRT_GATE
            @test r[:mass] <= _REGTRT_GATE
            @test r[:mom] <= _REGTRT_GATE
            @test r[:pi] <= _REGTRT_GATE

            rng = MersenneTwister(1180)
            worst = 0.0
            for _ in 1:500
                f = _regtrt_state(rng, Float64)
                s_p = 0.2 + 1.7 * rand(rng); s_m = 0.2 + 1.7 * rand(rng)
                Fp = Kraken._trt_collide_local(f..., s_p, s_m)
                Fa = collide(f..., s_p, s_m)
                worst = max(worst, maximum(abs.(Fa .- Fp)))
            end
            @test worst <= _REGTRT_PARITY_GATE
        end
    end
end

@testset "Regularized TRT brick == _trt_collide_local (#118)" begin
    Nx, Ny = 8, 7
    mesh = stretched_box_mesh(; x_min=0.0, x_max=Float64(Nx - 1),
                                y_min=0.0, y_max=Float64(Ny - 1), Nx=Nx, Ny=Ny)
    geom = build_slbm_geometry(mesh)
    rng = MersenneTwister(1181)
    f_in = zeros(Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        f = _regtrt_state(rng, Float64)
        for q in 1:9
            f_in[i, j, q] = f[q]
        end
    end
    sp = 0.2 .+ 1.7 .* rand(rng, Nx, Ny); sm = 0.2 .+ 1.7 .* rand(rng, Nx, Ny)
    f_out = zeros(Nx, Ny, 9)
    ρ = zeros(Nx, Ny); ux = zeros(Nx, Ny); uy = zeros(Nx, Ny)
    is_solid = falses(Nx, Ny)
    q_wall = zeros(Nx, Ny, 9); uw_x = zeros(Nx, Ny, 9); uw_y = zeros(Nx, Ny, 9)
    slbm_reg_libb_step_local_2d!(f_out, f_in, ρ, ux, uy, is_solid, q_wall,
                                 uw_x, uw_y, geom, sp, sm)
    worst = 0.0
    for j in 2:Ny-1, i in 2:Nx-1
        fp = ntuple(q -> f_in[i - _REGTRT_CX[q], j - _REGTRT_CY[q], q], 9)
        F = Kraken._trt_collide_local(fp..., sp[i, j], sm[i, j])
        worst = max(worst, maximum(abs(f_out[i, j, q] - F[q]) for q in 1:9))
    end
    @test worst <= _REGTRT_PARITY_GATE
    @info "regularized TRT brick parity" worst
end
