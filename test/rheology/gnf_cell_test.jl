using Test
using KernelAbstractions
using Kraken

# Steady GNF forward:
#   T1  per-cell split/fused paths == collide_rheology_guo_2d_kernel! bit for bit
#   T1f Float32: split and fused paths identical (closure converted to T)
#   T1b ForwardDiff local Jacobian of gnf_cell_update == central differences
#   T1z ForwardDiff local Jacobian finite on a cell with S = 0 (closure in I1)
#   T2  Carreau-Yasuda Poiseuille vs 1D integration
#   T2c shear-rate estimator vs the analytic profile, and size of the Guo
#       forcing correction of Π^neq (left out of the production estimator)
#   T3  CPU/GPU parity (skipped without a functional CUDA device)

const _GNF_GPU_OK = (try
    @eval Main using CUDA
    @eval Main CUDA.functional()
catch
    false
end) === true

# ---------------------------------------------------------------- T1 case ---
# 32×16 channel, one half-disk obstacle on each wall, non-uniform body force,
# initial shear flow strong enough to spread τ (shear-thinning exercised).
function _gnf_t1_case()
    Nx, Ny = 32, 16
    solid = fill(false, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        x = i - 0.5; y = j - 0.5
        if (x - 16)^2 + y^2 < 25 || (x - 16)^2 + (y - 16)^2 < 16
            solid[i, j] = true
        end
    end
    Fx = [2e-4 * (1 + 0.3 * sinpi(2i / Nx)) for i in 1:Nx, j in 1:Ny]
    Fy = [5e-5 * cospi(2j / Ny) for i in 1:Nx, j in 1:Ny]
    closure = CarreauYasuda(0.2, 0.004, 2000.0, 2.0, 0.6)
    f0 = zeros(Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        y = (j - 0.5) / Ny
        ux = 0.02 * sinpi(y) + 0.005 * cospi(2i / Nx)
        uy = 0.003 * sinpi(2i / Nx) * sinpi(y)
        f0[i, j, :] .= gnf_feq_cell(1.0, ux, uy)
    end
    return (; Nx, Ny, solid, Fx, Fy, closure, f0, tau0 = 3 * 0.2 + 0.5)
end

function _gnf_dev(backend, a)
    d = KernelAbstractions.allocate(backend, eltype(a), size(a)...)
    copyto!(d, a)
    return d
end

# path = :production (stream + collide_rheology_guo_2d!), :split or :fused (gnf_step!).
# Returns (f, τ, I1, ν) on the host; I1 and ν stay 0 on the production path.
function _gnf_t1_run(case, backend, path::Symbol; nsteps::Int = 200,
                     T::Type = Float64, closure = case.closure)
    Nx, Ny = case.Nx, case.Ny
    f_in = _gnf_dev(backend, T.(case.f0))
    f_out = _gnf_dev(backend, T.(case.f0))
    tau = KernelAbstractions.allocate(backend, T, Nx, Ny)
    fill!(tau, T(case.tau0))
    solid = _gnf_dev(backend, case.solid)
    Fx = _gnf_dev(backend, T.(case.Fx))
    Fy = _gnf_dev(backend, T.(case.Fy))
    i1 = KernelAbstractions.zeros(backend, T, Nx, Ny)
    nu = KernelAbstractions.zeros(backend, T, Nx, Ny)
    for _ in 1:nsteps
        if path === :production
            stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
            collide_rheology_guo_2d!(f_out, solid, closure, tau, Fx, Fy)
        else
            gnf_step!(f_out, f_in, tau, i1, nu, solid, closure, Fx, Fy;
                      fused = path === :fused)
        end
        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)
    return Array(f_in), Array(tau), Array(i1), Array(nu)
end

# ------------------------------------------------------- 1D CY reference ---
# Plane Poiseuille, ρ = 1: ν(γ̇) γ̇ = G s at distance s from the centreline,
# u(s) = ∫_s^{h} γ̇(σ) dσ with h the half-width.
function _gnf_cy_shear_rate(cl, G, s)
    target = G * s
    target == 0 && return 0.0
    lo, hi = 0.0, target / cl.eta_inf
    for _ in 1:80
        mid = (lo + hi) / 2
        effective_viscosity(cl, mid) * mid < target ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

const _GL5_X = (0.0, -0.5384693101056831, 0.5384693101056831,
                -0.9061798459386640, 0.9061798459386640)
const _GL5_W = (0.5688888888888889, 0.4786286704993665, 0.4786286704993665,
                0.2369268850561891, 0.2369268850561891)

function _gnf_cy_velocity(cl, G, s, h; panels::Int = 400)
    s >= h && return 0.0
    acc = 0.0
    dl = (h - s) / panels
    for p in 1:panels
        a = s + (p - 1) * dl
        c = a + dl / 2
        for (x, w) in zip(_GL5_X, _GL5_W)
            acc += w * (dl / 2) * _gnf_cy_shear_rate(cl, G, c + x * dl / 2)
        end
    end
    return acc
end

# Local copy of the production shear-rate estimator with an optional Guo
# forcing correction, S = -3/(2ρτ) [Π^neq + ½(F u + u F)] (hypothesis H14).
function _gnf_shear_rate_local(fp, Fx, Fy, τ; corrected::Bool)
    ρ, ux, uy = gnf_moments_cell(fp, Fx, Fy)
    fe = gnf_feq_cell(ρ, ux, uy)
    n = fp .- fe
    Pxx = n[2] + n[4] + n[6] + n[7] + n[8] + n[9]
    Pyy = n[3] + n[5] + n[6] + n[7] + n[8] + n[9]
    Pxy = n[6] - n[7] + n[8] - n[9]
    if corrected
        Pxx += Fx * ux
        Pyy += Fy * uy
        Pxy += (Fx * uy + Fy * ux) / 2
    end
    c = -3 / (2 * ρ * τ)
    Sxx = c * Pxx; Syy = c * Pyy; Sxy = c * Pxy
    return sqrt(2 * (Sxx^2 + Syy^2 + 2 * Sxy^2))
end

# T2 reference Carreau–Yasuda parameters (η∞* = 0.02, k* = 5, n = 0.7, a = 2) at
# H* = 1 on m = 64 nodes, ν0 = 0.34 (lattice), G* = 10.
# Reduced units: length L = m lattice nodes, viscosity ν0, time L²/ν0;
# η∞* = η∞/η0, λ_lat = k* m²/ν0,lat, G* = G_lat m³/ν0,lat².
function _gnf_t2_case()
    m, ν0 = 64, 0.34
    G = 10 * ν0^2 / m^3
    k = 5 * m^2 / ν0
    cl = CarreauYasuda(ν0, 0.02 * ν0, k, 2.0, 0.7)
    return (; Nx = 4, Ny = m, ν0, G, cl)
end

@testset "GNF steady forward (gnf_cell)" begin

    @testset "T1 bit-for-bit parity with collide_rheology_guo_2d_kernel!" begin
        case = _gnf_t1_case()
        fP, tP, _, _ = _gnf_t1_run(case, CPU(), :production)
        fS, tS, iS, nS = _gnf_t1_run(case, CPU(), :split)
        fF, tF, iF, nF = _gnf_t1_run(case, CPU(), :fused)
        fl = .!case.solid
        dfS = maximum(abs.(fS .- fP)); dtS = maximum(abs.(tS .- tP))
        dfF = maximum(abs.(fF .- fP)); dtF = maximum(abs.(tF .- tP))
        spread = maximum(tP[fl]) - minimum(tP[fl])
        @info "T1 parity (200 steps, 32×16)" dfS dtS dfF dtF tau_min = minimum(tP[fl]) tau_max = maximum(tP[fl])
        @test all(isfinite, fP)
        @test dfS == 0
        @test dtS == 0
        @test dfF == 0
        @test dtF == 0
        @test spread > 0.05                          # shear-thinning really active
        @test all(tS[case.solid] .== case.tau0)      # τ never written at solid nodes
        @test iS == iF                               # split and fused see the same I1, ν
        @test nS == nF
        @test all(iS[case.solid] .== 0) && all(nS[case.solid] .== 0)
    end

    @testset "T1f Float32: split and fused paths identical" begin
        case = _gnf_t1_case()
        # Kernel level, closure converted to Float32.
        cl32 = Kraken.gnf_convert_closure(Float32, case.closure)
        fS, tS, iS, nS = _gnf_t1_run(case, CPU(), :split; T = Float32, closure = cl32)
        fF, tF, iF, nF = _gnf_t1_run(case, CPU(), :fused; T = Float32, closure = cl32)
        # Control (not a test): the same run with the Float64 closure.
        fS64, _, _, _ = _gnf_t1_run(case, CPU(), :split; T = Float32)
        fF64, _, _, _ = _gnf_t1_run(case, CPU(), :fused; T = Float32)
        @info "T1f Float32, unconverted Float64 closure (control)" max_split_vs_fused = maximum(abs.(fS64 .- fF64))
        @test cl32 isa CarreauYasuda{Float32}
        @test eltype(fS) == Float32
        @test all(isfinite, fS)
        @test fS == fF
        @test tS == tF
        @test iS == iF && nS == nF
        # Driver level: Float64 closure, T = Float32, converted by the driver.
        kw = (; is_solid = case.solid, closure = case.closure, Fx = case.Fx, Fy = case.Fy,
              T = Float32, f_init = case.f0, tau_init = case.tau0, tol = 0.0,
              max_steps = 200, check_every = 50)
        rS = run_gnf_periodic_steady(; kw..., fused = false)
        rF = run_gnf_periodic_steady(; kw..., fused = true)
        @test rS.steps == rF.steps == 200
        @test eltype(rS.f) == Float32
        @test Array(rS.f) == Array(rF.f)
        @test Array(rS.tau) == Array(rF.tau)
        @test Array(rS.f) == fS                      # driver == kernel-level run
    end

    @testset "T1b ForwardDiff local block of gnf_cell_update" begin
        cl = CarreauYasuda(0.2, 0.004, 2000.0, 2.0, 0.6)
        Fx, Fy = 1e-4, -3e-5
        fe = gnf_feq_cell(1.0, 0.03, -0.01)
        x0 = [collect(fe) .+ 1e-3 .* sin.(1:9); 0.9]
        g(x) = (r = gnf_cell_update(Kraken.SVector{9}(ntuple(k -> x[k], 9)), x[10], Fx, Fy, cl);
                vcat(collect(r[1]), r[2]))
        J = Kraken.ForwardDiff.jacobian(g, x0)
        h = 1e-6
        JFD = similar(J)
        for k in 1:10
            e = zeros(10); e[k] = h
            JFD[:, k] = (g(x0 .+ e) .- g(x0 .- e)) ./ (2h)
        end
        err = maximum(abs.(J .- JFD)) / maximum(abs.(J))
        @info "T1b ForwardDiff vs central FD (10×10 block)" err
        @test size(J) == (10, 10)
        @test all(isfinite, J)
        @test err <= 1e-6
    end

    @testset "T1z ForwardDiff block finite on a cell with S = 0" begin
        cl = CarreauYasuda(0.2, 0.004, 2000.0, 2.0, 0.6)
        τp = 0.9
        # Cell at rest at equilibrium, no force: f_q = feq_q(1, 0, 0) for q ≥ 2,
        # so Π^neq = 0 and I1 = 0 exactly (f1 does not enter Π^neq). f1 is moved
        # by a few ulps, if needed, until Σ f = 1 exactly in floating point.
        fe = gnf_feq_cell(1.0, 0.0, 0.0)
        cell(a) = Kraken.SVector{9}(a, fe[2], fe[3], fe[4], fe[5], fe[6], fe[7], fe[8], fe[9])
        f1 = fe[1]; found = false
        for k in 0:64
            c = fe[1] + (iseven(k) ? 1 : -1) * (k ÷ 2) * eps(fe[1])
            if gnf_cell_update(cell(c), τp, 0.0, 0.0, cl)[3] == 0
                f1 = c; found = true
                break
            end
        end
        @test found
        x0 = [collect(cell(f1)); τp]
        g(x) = (r = gnf_cell_update(Kraken.SVector{9}(ntuple(k -> x[k], 9)), x[10], 0.0, 0.0, cl);
                vcat(collect(r[1]), r[2], r[3], r[4]))     # fo, τ_new, I1, ν
        J = Kraken.ForwardDiff.jacobian(g, x0)
        # h = 1e-8, not 1e-6 as in T1b: around S = 0, ν − ν(0) ≈ κ I1 with
        # κ = (η0 − η∞)(n − 1)λ² ≈ −3e5 here, so the O(h²) truncation term of
        # the central difference is ≈ 3e7 h² (measured 2026-10-19: 3.3e-5 at
        # h = 1e-6, 3.3e-7 at 1e-7, 1e-8 at 1e-8, where round-off takes over).
        h = 1e-8
        JFD = similar(J)
        for k in 1:10
            e = zeros(10); e[k] = h
            JFD[:, k] = (g(x0 .+ e) .- g(x0 .- e)) ./ (2h)
        end
        err = maximum(abs.(J .- JFD)) / maximum(abs.(J))
        # Control (not a test): the closure applied to γ̇ = √(2 I1) instead of I1.
        function gold(x)
            fp = Kraken.SVector{9}(ntuple(k -> x[k], 9))
            ρ, ux, uy = gnf_moments_cell(fp, 0.0, 0.0)
            ν = effective_viscosity(cl, gnf_strain_rate_cell(fp, ρ, ux, uy, x[10]))
            r = gnf_collide_cell(fp, 0.0, 0.0, ν)
            return vcat(collect(r[1]), r[2])
        end
        @info "T1z S = 0 block" I1 = g(x0)[11] err nan_with_closure_in_gamma_dot = count(isnan, Kraken.ForwardDiff.jacobian(gold, x0))
        @test size(J) == (12, 10)
        @test all(isfinite, J)
        # At S = 0, ∂I1 = 2 S:∂S = 0, so I1, ν and τ_new have zero derivatives.
        @test all(iszero, J[11, :]) && all(iszero, J[12, :]) && all(iszero, J[10, :])
        @test err <= 1e-6
    end

    c2 = _gnf_t2_case()
    t2 = @elapsed res = run_gnf_periodic_steady(; is_solid = fill(false, c2.Nx, c2.Ny),
                                                closure = c2.cl, Fx = c2.G, tol = 1e-10,
                                                max_steps = 400_000, check_every = 500)
    yp = [(j - 0.5) - c2.Ny / 2 for j in 1:c2.Ny]      # distance from the centreline
    hw = c2.Ny / 2

    @testset "T2 Carreau-Yasuda Poiseuille vs 1D integration" begin
        uref = [_gnf_cy_velocity(c2.cl, c2.G, abs(y), hw) for y in yp]
        u = vec(sum(res.ux; dims = 1)) ./ c2.Nx
        err = maximum(abs.(u .- uref)) / maximum(uref)
        ctr = c2.Ny ÷ 2
        @info "T2 CY Poiseuille" steps = res.steps wall = t2 r_u = res.r_u r_u2 = res.r_u2 dist_est = res.dist_est contraction = res.contraction u_max = maximum(u) u_ref_max = maximum(uref) err err_centre = abs(u[ctr] - uref[ctr]) / uref[ctr] tau_min = res.tau_min tau_max = res.tau_max
        @test res.converged
        @test res.finite
        @test err <= 2e-3
        @test maximum(abs.(res.ux .- res.ux[1:1, :])) <= 1e-12   # x-invariant
        @test maximum(abs.(res.uy)) <= 1e-12
        @test res.tau_max - res.tau_min > 0.3                   # strong thinning (kγ̇_w ~ 1e2)
    end

    @testset "T2c shear-rate estimator and Guo forcing correction (H14)" begin
        gref = [_gnf_cy_shear_rate(c2.cl, c2.G, abs(y)) for y in yp]
        gd = Array(res.gamma_dot)
        gl = vec(sum(gd; dims = 1)) ./ c2.Nx
        rel = abs.(gl .- gref) ./ gref
        interior = 4:(c2.Ny - 3)
        wall = setdiff(1:c2.Ny, interior)
        @info "T2c γ̇ estimator" max_rel_interior = maximum(rel[interior]) rel_near_wall = string(round.(rel[wall]; sigdigits = 3)) kgd_wall = c2.cl.lambda * gref[1]
        @test maximum(rel[interior]) <= 1e-3

        # Arrived populations of the next step, τ of the last step.
        Nx, Ny = c2.Nx, c2.Ny
        fa = similar(res.f)
        stream_periodic_x_wall_y_2d!(fa, res.f, Nx, Ny)
        Fxa = fill(c2.G, Nx, Ny); Fya = zeros(Nx, Ny)
        solid = fill(false, Nx, Ny)
        gk = zeros(Nx, Ny)
        gnf_strain_rate_2d!(gk, fa, solid, res.tau, Fxa, Fya)
        g0 = zeros(Nx, Ny); gc = zeros(Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            fp = Kraken.SVector{9}(ntuple(q -> fa[i, j, q], 9))
            g0[i, j] = _gnf_shear_rate_local(fp, c2.G, 0.0, res.tau[i, j]; corrected = false)
            gc[i, j] = _gnf_shear_rate_local(fp, c2.G, 0.0, res.tau[i, j]; corrected = true)
        end
        copy_err = maximum(abs.(g0 .- gk) ./ gk)
        ν0 = [effective_viscosity(c2.cl, g) for g in g0]
        νc = [effective_viscosity(c2.cl, g) for g in gc]
        dnu = maximum(abs.(νc .- ν0) ./ ν0)
        dg_rel = maximum(abs.(gc .- g0) ./ g0)
        dg_max = maximum(abs.(gc .- g0)) / maximum(g0)
        @info "T2c Guo forcing correction (H14)" copy_err dnu_over_nu = dnu dgamma_pointwise = dg_rel dgamma_over_max = dg_max
        @test copy_err <= 1e-12
        @test dnu <= 1e-4
    end

    @testset "T3 CPU/GPU parity" begin
        if !_GNF_GPU_OK
            @info "Skipping GNF CPU/GPU parity (no functional CUDA device)"
            @test_skip "T3 needs CUDA.functional()"
        else
            gpu = Main.CUDA.CUDABackend()
            case = _gnf_t1_case()
            for path in (:production, :split, :fused)
                fc, tc, _, _ = _gnf_t1_run(case, CPU(), path)
                fg, tg, _, _ = _gnf_t1_run(case, gpu, path)
                df = maximum(abs.(fg .- fc)); dt = maximum(abs.(tg .- tc))
                @info "T3 parity $(path) (200 steps)" df dt
                @test df <= 1e-12
                @test dt <= 1e-12
            end
            kw = (; is_solid = fill(false, c2.Nx, c2.Ny), closure = c2.cl, Fx = c2.G,
                  tol = 0.0, max_steps = 5000, check_every = 1000)
            rc = run_gnf_periodic_steady(; kw..., backend = CPU())
            rg = run_gnf_periodic_steady(; kw..., backend = gpu)
            du = maximum(abs.(rg.ux .- rc.ux)) / maximum(abs.(rc.ux))
            dr = abs(rg.r_u - rc.r_u) / rc.r_u
            @info "T3 driver parity (Poiseuille, 5000 steps)" du dr
            @test rg.steps == rc.steps == 5000
            @test du <= 1e-10
            @test dr <= 1e-6
        end
    end
end
