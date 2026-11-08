using Test
using LinearAlgebra
using Printf
using Random
using KernelAbstractions
using KrakeNN

# Steady GNF adjoint, CPU
# KernelAbstractions backend (the kernels are the GPU ones):
#   S    transposed stream: S Sᵀ = Sᵀ S = I, ⟨S a, b⟩ = ⟨a, Sᵀ b⟩
#   T4b  16×8: mass gauge Aᵀc = c, active/inactive decoupling, reduced operator
#        assembled column by column (one singular value of I − A_red at zero,
#        bordered system regular), eigenvalue of A_red closest to −1
#   T4c  32×16: ⟨P Aᵀ P λ, v⟩ = ⟨λ, P A P v⟩
#   T4d  32×16: JVP vs central differences of the real step (gnf_step!)
#   Tb   ∂J/∂w vs central differences of the objective
#   T5   32×16: adjoint gradient vs central differences of J on the 4 CY
#        parameters (ln η∞, ln k, n, a), forward distance × adjoint tolerance × h
#   T6   truncated adjoint series vs GMRES
#   G    CPU/GPU parity of JVP/VJP and of the adjoint solve (skipped without CUDA)
#
# Case: the periodic obstacle channel (domain 8 × 4, disks of radius 1.5 at (4, 0) and
# (4, 4)) at m = 4 (32×16) and m = 2 (16×8) nodes per unit length, ν0,lat = 0.17.
# Reference Carreau–Yasuda parameters (η∞* = 0.02, k* = 5, n = 0.7, a = 2).
# Reduced units: length L = m lattice nodes, kinematic viscosity ν0, time L²/ν0;
# η∞* = η∞/η0, λ_lat = k* m²/ν0,lat, u* = u_lat m/ν0,lat,
# G* = G L³/ν0² = G_lat m³/ν0,lat², I1* = I1_lat m⁴/ν0,lat².
# The reference body force G* = 5 is scaled by _GA_GSCALE: at G* = 5 the lattice
# velocity at m = 4 would be
# ≈ 0.34 (Mach 0.6); the scaled force keeps Ma ≲ 0.1 while the viscosity still
# varies over the domain by more than a factor 2 (both printed by the "case"
# testset; G scale 0.1 gave ν/ν0 ∈ [0.59, 1], hence 0.3).
#
# Forward distances. The two-step estimator of `run_gnf_periodic_steady` gives
# the first level (dist_est ≤ 1e-10); on 32×16 it then saturates at the
# Float64 floor (measured 2026-10-20: 200 000 more steps left dist_est at
# 2e-10, the fitted ρ drifting to 1 on round-off). The finer levels are
# therefore reached by K more steps each, K = ln(100)/(−ln ρ₁) with ρ₁ the
# contraction fitted at the first level (prediction 1e-12, 1e-14), and the
# true distance of each level is measured by one Newton step on the active space,
# (I − P A P) δ = P (G(w) − w) with the mass border (`_ga_newton_dist`).

const _GA_GPU_OK = (try
    @eval Main using CUDA
    @eval Main CUDA.functional()
catch
    false
end) === true

const _GA_NU0 = 0.17
const _GA_GSCALE = 0.3
const _GA_THETA_REF = [log(0.02), log(5.0), 0.7, 2.0]   # (ln η∞, ln k, n, a)

function _ga_case(m::Int; nu0=_GA_NU0, gscale=_GA_GSCALE)
    Nx, Ny = 8m, 4m
    solid = [(((i - 0.5) / m - 4)^2 + ((j - 0.5) / m)^2 < 1.5^2) ||
             (((i - 0.5) / m - 4)^2 + ((j - 0.5) / m - 4)^2 < 1.5^2) for i in 1:Nx, j in 1:Ny]
    g = gscale * 5 * nu0^2 / m^3
    closure_of = θ -> CarreauYasuda(nu0, exp(θ[1]) * nu0, exp(θ[2]) * m^2 / nu0, θ[4], θ[3])
    return (; m, Nx, Ny, solid, g, nu0, uref=nu0 / m, closure_of)
end

function _ga_forward(case, θ; tol, f_init=nothing, tau_init=nothing, max_steps=200_000)
    return run_gnf_periodic_steady(; is_solid=case.solid, closure=case.closure_of(θ), Fx=case.g,
                                   tol=tol, check_every=20, fit_checks=20, fused=true,
                                   f_init=f_init, tau_init=tau_init, max_steps=max_steps)
end

# Forward to dist_est ≤ tol1, then `nextra` further levels of K steps each
# (one trajectory). K from the fitted contraction of the first level, unless given.
function _ga_forward_chain(case, θ; tol1=1e-10, nextra=2, K=nothing, f_init=nothing,
                           tau_init=nothing)
    r = _ga_forward(case, θ; tol=tol1, f_init=f_init, tau_init=tau_init)
    Kc = K === nothing ? clamp(ceil(Int, log(100) / -log(r.contraction)), 200, 20_000) : K
    out = [(; f=copy(r.f), tau=copy(r.tau), dist=r.dist_est, steps=r.steps, rho=r.contraction,
            converged=r.converged, ux=r.ux, uy=r.uy)]
    for _ in 1:nextra
        r = run_gnf_periodic_steady(; is_solid=case.solid, closure=case.closure_of(θ),
                                    Fx=case.g, tol=0.0, check_every=20, fit_checks=20,
                                    fused=true, f_init=r.f, tau_init=r.tau, max_steps=Kc)
        push!(out, (; f=copy(r.f), tau=copy(r.tau), dist=r.dist_est, steps=r.steps,
                    rho=r.contraction, converged=true, ux=r.ux, uy=r.uy))
    end
    return out, Kc
end

# True relative distance of u to the fixed point, from one Newton step on the
# active space: (I − P A P) δ + ĉ s = P (G(w) − w), ĉᵀ δ = 0 (mass border).
function _ga_newton_dist(case, st, θ)
    cl = case.closure_of(θ)
    lin = gnf_linearise(st.f, st.tau, case.solid, cl, case.g)
    op = GNFStepOperator(lin)
    n = op.n
    w = gnf_pack_state(st.f, st.tau)
    r = (gnf_step_state(w, case.solid, cl, fill(case.g, case.Nx, case.Ny),
                        zeros(case.Nx, case.Ny)) .- w) .* lin.active
    tmp = zeros(n)
    function apply!(y, x)
        xn = view(x, 1:n)
        gnf_jvp!(tmp, op, xn)
        view(y, 1:n) .= xn .- tmp .+ op.chat .* x[n+1]
        y[n+1] = dot(op.chat, xn)
        return y
    end
    x = zeros(n + 1)
    info = gnf_gmres!(x, apply!, vcat(r, 0.0); tol=1e-8, restart=500, maxiter=5000)
    f1, _ = gnf_unpack_state(w .+ x[1:n], case.Nx, case.Ny)
    _, ux0, uy0 = gnf_post_moments(st.f, case.solid, case.g)
    _, ux1, uy1 = gnf_post_moments(f1, case.solid, case.g)
    d = sqrt(sum(abs2, ux1 .- ux0) + sum(abs2, uy1 .- uy0)) / sqrt(sum(abs2, ux0) + sum(abs2, uy0))
    return d, info.iters
end

# Node-wise observation of both velocity components at every fluid node,
# u* = u_lat/u_ref, targets 0.8 u*_ref (so that ∇J ≠ 0).
function _ga_obs(case, ux, uy)
    nodes = findall(vec(.!case.solid))
    M = 2 * length(nodes)
    node = zeros(Int, 1, M); w = ones(1, M); comp = zeros(Int, M); tgt = zeros(M)
    for (k, nd) in enumerate(nodes)
        node[1, 2k-1] = nd; comp[2k-1] = 1; tgt[2k-1] = 0.8 * ux[nd] / case.uref
        node[1, 2k] = nd; comp[2k] = 2; tgt[2k] = 0.8 * uy[nd] / case.uref
    end
    return GNFVelocityObs(node, w, comp, tgt, 1 / case.uref)
end

_ga_J(case, obs, st) = (m = gnf_post_moments(st.f, case.solid, case.g);
                        gnf_obs_objective(obs, m[2], m[3]))

_ga_relerr(a, b) = norm(a .- b) / max(norm(b), 1e-300)

@testset "GNF steady adjoint" begin
    t_file = time()
    cpu = KernelAbstractions.CPU()
    rng = MersenneTwister(20261020)

    # ------------------------------------------------------------------ S ---
    @testset "S transposed stream" begin
        Nx, Ny = 7, 5
        a = rand(rng, Nx, Ny, 9); b = rand(rng, Nx, Ny, 9)
        Sa = similar(a); STa = similar(a); t1 = similar(a); t2 = similar(a)
        stream_periodic_x_wall_y_2d!(Sa, a, Nx, Ny)
        stream_periodic_x_wall_y_transpose_2d!(STa, a, Nx, Ny)
        stream_periodic_x_wall_y_transpose_2d!(t1, Sa, Nx, Ny)
        stream_periodic_x_wall_y_2d!(t2, STa, Nx, Ny)
        @test t1 == a
        @test t2 == a
        STb = similar(b); stream_periodic_x_wall_y_transpose_2d!(STb, b, Nx, Ny)
        @test abs(dot(Sa, b) - dot(a, STb)) <= 1e-14 * abs(dot(Sa, b))
    end

    # --------------------------------------------------------------- case ---
    c4 = _ga_case(4)
    θ0 = _GA_THETA_REF
    tcase = time()
    base, Kchain = _ga_forward_chain(c4, θ0)
    ndist = [_ga_newton_dist(c4, st, θ0) for st in base]
    @testset "case 32×16 (forward)" begin
        st = base[end]
        fl = .!c4.solid
        umax = maximum(sqrt.(st.ux[fl] .^ 2 .+ st.uy[fl] .^ 2))
        lin = gnf_linearise(st.f, st.tau, c4.solid, c4.closure_of(θ0), c4.g)
        @printf("  case m=4 32x16: g_lat=%.3e  u_lat,max=%.4f  Ma=%.3f  nu/nu0 in [%.3f, %.3f]  small-I1 nodes=%d (I1_min=%.2e)\n",
                c4.g, umax, sqrt(3) * umax, minimum(lin.nu[fl]) / c4.nu0, maximum(lin.nu[fl]) / c4.nu0,
                lin.n_small_I1, lin.I1_min)
        @printf("  forward levels: steps %s (K = %d, rho_1 = %.6f) ; dist_est %s ; Newton distance %s (GMRES it %s)\n",
                join([s.steps for s in base], "/"), Kchain, base[1].rho,
                join([@sprintf("%.1e", s.dist) for s in base], "/"),
                join([@sprintf("%.1e", d[1]) for d in ndist], "/"), join([d[2] for d in ndist], "/"))
        @test base[1].converged
        @test ndist[1][1] <= 1e-9 && ndist[end][1] <= ndist[1][1]
        @test sqrt(3) * umax < 0.15
        @test maximum(lin.nu[fl]) / minimum(lin.nu[fl]) > 2     # viscosity varies
        @printf("  case wall %.1f s\n", time() - tcase)
    end
    st0 = base[end]
    obs = _ga_obs(c4, st0.ux, st0.uy)

    # ---------------------------------------------------------------- T4b ---
    @testset "T4b gauge, decoupling, reduced operator (16×8)" begin
        t4b = time()
        c2 = _ga_case(2)
        s2 = _ga_forward_chain(c2, θ0; nextra=1)[1][end]
        t_fw = time() - t4b
        lin = gnf_linearise(s2.f, s2.tau, c2.solid, c2.closure_of(θ0), c2.g)
        op = GNFStepOperator(lin; backend=cpu)
        n = op.n; N = c2.Nx * c2.Ny
        act = lin.active
        c = Float64.(act); c[9N+1:10N] .= 0
        # (i) Aᵀ c = c on the active space
        y = similar(c); gnf_vjp!(y, op, c)
        e_gauge = maximum(abs.(y .- c))
        @printf("  T4b(i) max|P A^T P c - c| = %.2e  (n=%d, active=%d, fluid nodes=%d)\n",
                e_gauge, n, lin.n_active, lin.n_fluid)
        @test e_gauge <= 1e-14
        # (ii) decoupling, both directions, both operators (unprojected)
        vin = randn(rng, n) .* .!act; vact = randn(rng, n) .* act
        out = similar(vin)
        gnf_jvp!(out, op, vin; project=false); leak1 = maximum(abs.(out[act]))
        gnf_vjp!(out, op, vin; project=false); leak2 = maximum(abs.(out[act]))
        gnf_jvp!(out, op, vact; project=false); leak3 = maximum(abs.(out[.!act]))
        gnf_vjp!(out, op, vact; project=false); leak4 = maximum(abs.(out[.!act]))
        @printf("  T4b(ii) leaks inactive->active A %.1e A^T %.1e ; active->inactive A %.1e A^T %.1e\n",
                leak1, leak2, leak3, leak4)
        @test leak1 == 0 && leak2 == 0 && leak3 == 0 && leak4 == 0
        # (iii) reduced operator, column by column
        ia = findall(act); na = length(ia)
        Ared = zeros(na, na); AredT = zeros(na, na)
        e = zeros(n); col = zeros(n)
        for (k, idx) in enumerate(ia)
            e[idx] = 1
            gnf_jvp!(col, op, e); Ared[:, k] .= col[ia]
            gnf_vjp!(col, op, e); AredT[:, k] .= col[ia]
            e[idx] = 0
        end
        e_tr = maximum(abs.(AredT .- transpose(Ared)))
        t_as = time() - t4b - t_fw
        σ = svdvals(I - Ared)
        nzero = count(σ .< 1e-10 * σ[1])
        ĉa = c[ia] ./ norm(c[ia])
        Aug = [I - transpose(Ared) ĉa; transpose(ĉa) 0.0]
        σa = svdvals(Aug)
        @printf("  T4b(iii) n_active=%d  |A_jvp^T - A_vjp|=%.1e  sigma(I-A_red): max %.3e, smallest %.3e, %.3e ; #<1e-10 sigma_max: %d ; bordered sigma_min/sigma_max = %.3e\n",
                na, e_tr, σ[1], σ[end], σ[end-1], nzero, σa[end] / σa[1])
        @test e_tr <= 1e-14
        @test nzero == 1
        @test σa[end] > 1e-6 * σa[1]
        # (iv) eigenvalue of A_red closest to −1, and spectral radius off the mass mode
        μ = eigvals(Ared)
        t_la = time() - t4b - t_fw - t_as
        k1 = argmin(abs.(μ .+ 1))
        imass = argmin(abs.(μ .- 1))
        μrest = μ[setdiff(1:na, imass)]
        kdom = argmax(abs.(μrest))
        @printf("  T4b(iv) eigenvalue closest to -1: %s (|mu+1| = %.3e) ; mass eigenvalue %s ; dominant other |mu| = %.6f (mu = %s)\n",
                string(round(μ[k1]; sigdigits=8)), abs(μ[k1] + 1), string(round(μ[imass]; sigdigits=14)),
                abs(μrest[kdom]), string(round(μrest[kdom]; sigdigits=8)))
        @test abs(μ[k1] + 1) >= 1e-8
        @printf("  T4b wall %.1f s (forward %.1f, assembly %.1f, dense SVD+eig %.1f ; BLAS threads %d, Julia threads %d)\n",
                time() - t4b, t_fw, t_as, t_la, LinearAlgebra.BLAS.get_num_threads(), Threads.nthreads())
    end

    # ---------------------------------------------------------------- T4c ---
    lin0 = gnf_linearise(st0.f, st0.tau, c4.solid, c4.closure_of(θ0), c4.g)
    op0 = GNFStepOperator(lin0; backend=cpu)
    n0 = op0.n
    act0 = Float64.(lin0.active)
    @testset "T4c adjointness <A^T l, v> = <l, A v> (32×16)" begin
        worst = 0.0
        for _ in 1:5
            λ = randn(rng, n0) .* act0; v = randn(rng, n0) .* act0
            ATl = similar(λ); Av = similar(v)
            gnf_vjp!(ATl, op0, λ); gnf_jvp!(Av, op0, v)
            a = dot(ATl, v); b = dot(λ, Av)
            worst = max(worst, abs(a - b) / max(abs(a), abs(b)))
        end
        @printf("  T4c worst relative mismatch over 5 pairs: %.2e\n", worst)
        @test worst <= 1e-12
    end

    # ---------------------------------------------------------------- T4d ---
    @testset "T4d JVP vs central differences of the real step (32×16)" begin
        w0 = gnf_pack_state(st0.f, st0.tau)
        cl = c4.closure_of(θ0)
        Fx = fill(c4.g, c4.Nx, c4.Ny); Fy = zeros(c4.Nx, c4.Ny)
        hs = (1e-3, 1e-4, 1e-5, 1e-6)
        worst_best = 0.0; slopes_ok = true
        for d in 1:5
            v = randn(rng, n0) .* act0 .* abs.(w0)
            Av = similar(v); gnf_jvp!(Av, op0, v)
            errs = Float64[]
            for h in hs
                gp = gnf_step_state(w0 .+ h .* v, c4.solid, cl, Fx, Fy)
                gm = gnf_step_state(w0 .- h .* v, c4.solid, cl, Fx, Fy)
                fd = (gp .- gm) ./ (2h) .* act0
                push!(errs, _ga_relerr(fd, Av))
            end
            worst_best = max(worst_best, minimum(errs))
            # h² regime between the two largest h (above the round-off floor)
            ratio = errs[1] / errs[2]
            slopes_ok &= (errs[2] < 1e-9) || (30 <= ratio <= 300)
            @printf("  T4d dir %d: rel err at h = 1e-3..1e-6: %s (ratio h1/h2 = %.1f)\n", d,
                    join([@sprintf("%.2e", e) for e in errs], ", "), ratio)
        end
        @test worst_best <= 1e-6
        @test slopes_ok
    end

    # ----------------------------------------------------------------- Tb ---
    @testset "Tb dJ/dw vs central differences of J" begin
        J0, bh = gnf_obs_rhs(obs, lin0)
        @test J0 ≈ _ga_J(c4, obs, st0) rtol = 1e-14
        worst = 0.0
        for _ in 1:3
            v = randn(rng, n0) .* act0 .* abs.(gnf_pack_state(st0.f, st0.tau))
            h = 1e-6
            fp, tp = gnf_unpack_state(gnf_pack_state(st0.f, st0.tau) .+ h .* v, c4.Nx, c4.Ny)
            fm, tm = gnf_unpack_state(gnf_pack_state(st0.f, st0.tau) .- h .* v, c4.Nx, c4.Ny)
            Jp = _ga_J(c4, obs, (; f=fp)); Jm = _ga_J(c4, obs, (; f=fm))
            worst = max(worst, abs((Jp - Jm) / (2h) - dot(bh, v)) / abs(dot(bh, v)))
        end
        @printf("  Tb dJ/dw worst rel err %.2e\n", worst)
        @test worst <= 1e-7
    end

    # ----------------------------------------------------------------- T5 ---
    @testset "T5 adjoint gradient vs finite differences (32×16, reference Carreau–Yasuda parameters)" begin
        t5 = time()
        dists = (1e-10, 1e-12, 1e-14)     # level 1 estimated, levels 2-3 by K steps
        tols = (1e-4, 1e-6, 1e-8)
        hs = (1e-2, 1e-3, 1e-4, 1e-5)
        names = ("ln eta_inf", "ln k", "n", "a")
        # adjoint gradients at every (distance, tolerance)
        adj = Dict{Tuple{Int,Int},Vector{Float64}}()
        iters = Dict{Tuple{Int,Int},Int}()
        for (id, st) in enumerate(base)
            lin = gnf_linearise(st.f, st.tau, c4.solid, c4.closure_of(θ0), c4.g)
            op = GNFStepOperator(lin)
            _, bh = gnf_obs_rhs(obs, lin)
            for (it, tol) in enumerate(tols)
                λ, _, info = gnf_adjoint_solve(op, bh; tol=tol, restart=1000, maxiter=20_000)
                adj[(id, it)] = gnf_parameter_gradient(lin, λ, θ0, c4.closure_of)[1]
                iters[(id, it)] = info.iters
            end
        end
        # finite differences: one forward chain per perturbed θ, J at each distance
        fd = Dict{Tuple{Int,Int,Int},Float64}()    # (dist, param, h) → DF
        for p in 1:4, (ih, h) in enumerate(hs)
            δ = h * abs(θ0[p])
            θp = copy(θ0); θp[p] += δ
            θm = copy(θ0); θm[p] -= δ
            cp, _ = _ga_forward_chain(c4, θp; K=Kchain, f_init=base[1].f, tau_init=base[1].tau)
            cm, _ = _ga_forward_chain(c4, θm; K=Kchain, f_init=base[1].f, tau_init=base[1].tau)
            for id in eachindex(dists)
                fd[(id, p, ih)] = (_ga_J(c4, obs, cp[id]) - _ga_J(c4, obs, cm[id])) / (2δ)
            end
        end
        @printf("  T5 forward levels: Newton distance %s ; adjoint GMRES iterations (dist x tol): %s\n",
                join([@sprintf("%.1e", d[1]) for d in ndist], "/"),
                join([string(iters[(id, it)]) for id in 1:3 for it in 1:3], " "))
        for id in eachindex(dists), it in eachindex(tols)
            g = adj[(id, it)]
            line = @sprintf("  T5 dist %.0e tol %.0e:", dists[id], tols[it])
            for p in 1:4
                rel = [abs(g[p] - fd[(id, p, ih)]) / abs(fd[(id, p, ih)]) for ih in eachindex(hs)]
                line *= @sprintf("  %s %.4e [%s]", names[p], g[p],
                                 join([@sprintf("%.1e", x) for x in rel], " "))
            end
            println(line)
        end
        # criterion at (finest distance, tol 1e-8): best h per parameter
        g = adj[(3, 3)]
        atol = 1e-3 * norm(g) / sqrt(4)
        for p in 1:4
            errs = [abs(g[p] - fd[(3, p, ih)]) for ih in eachindex(hs)]
            ib = argmin(errs)
            ok = errs[ib] <= atol + 1e-3 * abs(fd[(3, p, ib)])
            @printf("  T5 criterion %s: adj %.6e  DF(h=%.0e) %.6e  |diff| %.2e  bound %.2e  %s\n",
                    names[p], g[p], hs[ib], fd[(3, p, ib)], errs[ib],
                    atol + 1e-3 * abs(fd[(3, p, ib)]), ok ? "ok" : "FAIL")
            @test ok
        end
        @printf("  T5 wall %.1f s\n", time() - t5)
    end

    # ----------------------------------------------------------------- T6 ---
    @testset "T6 truncated adjoint series vs GMRES" begin
        t6 = time()
        J0, bh = gnf_obs_rhs(obs, lin0)
        λ, η, info = gnf_adjoint_solve(op0, bh; tol=1e-12, restart=1000, maxiter=20_000)
        gref, _ = gnf_parameter_gradient(lin0, λ, θ0, c4.closure_of)
        ser = gnf_adjoint_series(op0, bh, (1000, 4000, 16000))
        @printf("  T6 GMRES tol 1e-12: %d it, relres %.1e, |c^T l| = %.1e\n", info.iters,
                info.relres, abs(dot(op0.chat, λ)))
        for N in (1000, 4000, 16000)
            gN, _ = gnf_parameter_gradient(lin0, ser[N], θ0, c4.closure_of)
            @printf("  T6 N=%5d: |l_N - l|/|l| = %.3e   |grad_N - grad|/|grad| = %.3e\n", N,
                    _ga_relerr(ser[N], λ), _ga_relerr(gN, gref))
        end
        @test info.converged
        @test all(isfinite, ser[16000])
        @printf("  T6 wall %.1f s\n", time() - t6)
    end

    # ------------------------------------------------------------------ C ---
    # Generic-cotangent entry point: the observation path
    # (validated by Tb/T5) is the special case g_u = dJ/du of gnf_steady_adjoint.
    @testset "C generic cotangent entry point (32×16)" begin
        tc = time()
        J0, bh = gnf_obs_rhs(obs, lin0)
        J1, gux, guy = gnf_obs_grad_u(obs, lin0)
        @test J1 == J0
        @test gnf_velocity_rhs(lin0, gux, guy) == bh
        gu3 = cat(gux, guy; dims=3)
        @test KrakeNN._gnf_split_gu(gu3) == (gux, guy)
        @test KrakeNN._gnf_split_gu((; ux=gux, uy=guy)) == (gux, guy)
        gsx = copy(gux); gsy = copy(guy)
        gsx[c4.solid] .= 1.0; gsy[c4.solid] .= -1.0
        @test gnf_velocity_rhs(lin0, gsx, gsy) == bh          # g_u ignored at solid nodes
        kw = (; tol=1e-10, restart=1000, maxiter=20_000)
        s, st = gnf_steady_adjoint(lin0, op0, gu3; m=c4.m, nu0_lat=c4.nu0, solver=:gmres, kw...)
        λ, _, info = gnf_adjoint_solve(op0, bh; kw...)
        gref, sref = gnf_parameter_gradient(lin0, λ, θ0, c4.closure_of)
        @test st.converged && info.converged
        @test st.iters == info.iters
        @test size(s) == (c4.Nx, c4.Ny)
        @test isapprox(s, sref; rtol=1e-13)
        @test all(iszero, s[c4.solid])
        @test st.I1star ≈ lin0.I1 .* (c4.m^4 / c4.nu0^2)
        r = gnf_steady_gradient(st0.f, st0.tau, θ0, obs; is_solid=c4.solid,
                                closure_of=c4.closure_of, Fx=c4.g, solver=:gmres, kw...)
        @test r.J == J0
        @test isapprox(r.grad, gref; rtol=1e-13)
        @printf("  C: %d it, relres %.1e, |s - s_ref|/|s_ref| = %.1e, wall %.1f s\n", st.iters,
                st.relres, _ga_relerr(s, sref), time() - tc)
    end

    # ------------------------------------------------------------ solvers ---
    # Fixed-point iteration with the two-step stop and GMRES
    # on the K-step system, against the bordered GMRES solution; K-step forms of
    # T4c (transposition) and T4d (composition of K real steps).
    @testset "Fixed-point and K-step adjoint solvers (32×16)" begin
        tv = time()
        _, bh = gnf_obs_rhs(obs, lin0)
        λr, _, ir = gnf_adjoint_solve(op0, bh; tol=1e-12, restart=1000, maxiter=20_000)
        gr, sr = gnf_parameter_gradient(lin0, λr, θ0, c4.closure_of)
        # T4c-K: <M^K l, v> = <l, (M^T)^K v>, M = Q P A^T P
        Kc = 101
        worst_c = 0.0
        for _ in 1:3
            l = randn(rng, n0) .* act0; v = randn(rng, n0) .* act0
            Ml = similar(l); Mv = similar(v)
            gnf_adjoint_power!(Ml, op0, l, Kc); gnf_tangent_power!(Mv, op0, v, Kc)
            worst_c = max(worst_c, abs(dot(Ml, v) - dot(l, Mv)) / (norm(Ml) * norm(v)))
        end
        @printf("  T4c-K (K = %d): worst |<M^K l, v> - <l, (M^T)^K v>| / (|M^K l| |v|) = %.2e\n", Kc, worst_c)
        @test worst_c <= 1e-12
        # T4d-K: (P A P)^K v vs central differences of K real steps
        Kd = 11
        w0 = gnf_pack_state(st0.f, st0.tau)
        cl = c4.closure_of(θ0)
        Fx = fill(c4.g, c4.Nx, c4.Ny); Fy = zeros(c4.Nx, c4.Ny)
        stepK = w -> (for _ in 1:Kd; w = gnf_step_state(w, c4.solid, cl, Fx, Fy); end; w)
        worst_d = 0.0
        for d in 1:2
            v = randn(rng, n0) .* act0 .* abs.(w0)
            u = copy(v); t = similar(v)
            for _ in 1:Kd
                gnf_jvp!(t, op0, u); u, t = t, u
            end
            errs = Float64[]
            for h in (1e-4, 1e-5, 1e-6)
                fd = (stepK(w0 .+ h .* v) .- stepK(w0 .- h .* v)) ./ (2h) .* act0
                push!(errs, _ga_relerr(fd, u))
            end
            worst_d = max(worst_d, minimum(errs))
            @printf("  T4d-K (K = %d) dir %d: rel err at h = 1e-4..1e-6: %s\n", Kd, d,
                    join([@sprintf("%.2e", e) for e in errs], ", "))
        end
        @test worst_d <= 1e-6
        # Fixed point, two-step stop, mean of the last two iterates
        λv, _, iv = gnf_adjoint_fixed_point(op0, bh; tol=1e-11, check_every=100)
        gv, _ = gnf_parameter_gradient(lin0, λv, θ0, c4.closure_of)
        @printf("  (v): %d it, contraction %.6f, dist_est %.1e, relres %.1e, |l - l_ref|/|l_ref| = %.1e, grad %.1e\n",
                iv.iters, iv.contraction, iv.dist_est, iv.relres, _ga_relerr(λv, λr), _ga_relerr(gv, gr))
        @test iv.converged
        @test _ga_relerr(λv, λr) <= 1e-9
        @test _ga_relerr(gv, gr) <= 1e-9
        # GMRES on (I - M^K) l = sum_{j<K} M^j b', K odd
        λk, _, ik = gnf_adjoint_kstep_solve(op0, bh; K=101, tol=1e-10, restart=30)
        @printf("  (vi) K = 101: %d it (%d VJP), relres %.1e (K-step %.1e, %d refine rounds), |l - l_ref|/|l_ref| = %.1e\n",
                ik.iters, ik.n_vjp, ik.relres, ik.relres_kstep, ik.rounds, _ga_relerr(λk, λr))
        @test ik.converged
        @test ik.relres <= 1e-10
        @test _ga_relerr(λk, λr) <= 1e-9
        @test_throws ArgumentError gnf_adjoint_kstep_solve(op0, bh; K=100)
        # through the generic entry point
        _, gux, guy = gnf_obs_grad_u(obs, lin0)
        sk, stk = gnf_steady_adjoint(lin0, op0, (gux, guy); solver=:kstep, K=101, tol=1e-10, restart=30)
        @test stk.converged
        @test _ga_relerr(sk, sr) <= 1e-9
        # E: end to end with the default solver (K-step, K = 1001): g_u at solid
        # nodes must not change s or the adjoint state (C only checks the RHS)
        @testset "E g_u at solid nodes, full default solve" begin
            te = time()
            gsx = copy(gux); gsy = copy(guy)
            gsx[c4.solid] .= 1.0; gsy[c4.solid] .= -1.0
            s1, st1 = gnf_steady_adjoint(lin0, op0, (gsx, gsy); tol=1e-10)
            nsol = count(c4.solid)
            gsx[c4.solid] .= randn(rng, nsol); gsy[c4.solid] .= randn(rng, nsol)
            s2, st2 = gnf_steady_adjoint(lin0, op0, (gsx, gsy); tol=1e-10)
            e_s = _ga_relerr(s2, s1); e_l = _ga_relerr(st2.λ, st1.λ)
            @printf("  E: default solver K = %d, %d it, relres %.1e, |s1 - s_ref|/|s_ref| = %.1e, |s2 - s1|/|s1| = %.1e, |l2 - l1|/|l1| = %.1e, wall %.1f s\n",
                    st1.info.K, st1.iters, st1.relres, _ga_relerr(s1, sr), e_s, e_l, time() - te)
            @test st1.info.K == 1001
            @test st1.converged && st2.converged
            @test st1.relres <= 1e-10
            @test _ga_relerr(s1, sr) <= 1e-9
            @test all(iszero, s1[c4.solid])
            @test e_s <= 1e-14
            @test e_l <= 1e-14
        end
        @printf("  V/VI wall %.1f s\n", time() - tv)
    end

    # ------------------------------------------------------------------ G ---
    @testset "G CPU/GPU parity" begin
        if !_GA_GPU_OK
            @test_skip "GPU parity needs CUDA.functional()"
        else
            gpu = Main.CUDA.CUDABackend()
            opg = GNFStepOperator(lin0; backend=gpu)
            v = randn(rng, n0) .* act0
            vg = Main.CUDA.CuArray(v)
            yc = similar(v); yg = similar(vg)
            gnf_jvp!(yc, op0, v); gnf_jvp!(yg, opg, vg)
            e1 = _ga_relerr(Array(yg), yc)
            gnf_vjp!(yc, op0, v); gnf_vjp!(yg, opg, vg)
            e2 = _ga_relerr(Array(yg), yc)
            J0, bh = gnf_obs_rhs(obs, lin0)
            λc, _, ic = gnf_adjoint_solve(op0, bh; tol=1e-10)
            λg, _, ig = gnf_adjoint_solve(opg, Main.CUDA.CuArray(bh); tol=1e-10)
            e3 = _ga_relerr(Array(λg), λc)
            @printf("  G parity: JVP %.1e  VJP %.1e  adjoint solve %.1e (it CPU %d, GPU %d)\n",
                    e1, e2, e3, ic.iters, ig.iters)
            @test e1 <= 1e-14 && e2 <= 1e-14
            @test e3 <= 1e-8
        end
    end
    @printf("  gnf_adjoint_test wall (incl. compilation) %.1f s\n", time() - t_file)
end
