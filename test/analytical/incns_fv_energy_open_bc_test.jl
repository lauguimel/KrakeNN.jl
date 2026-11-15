# Open thermal boundaries of the FV energy row, CPU Float64: inlet with an
# imposed temperature, zero-gradient outlet, Dirichlet voxel solids.
# Thresholds frozen 2026-11-11 before any run:
# - Graetz local Nusselt within 2 % of the 30-mode parallel-plate series for
#   x in [H, L - H]: the same gate and the same series as test O2 of the LBM
#   open-die driver (branch feat/gnf-open-steady, test/drivers/
#   gnf_open_steady_test.jl, which measured 0.96 % at 384 x 64, Pe = 200).
# - global energy balance |sum| <= 1e-9 * max|term| and mass balance <= 1e-10:
#   both are discrete identities at a converged state (Newton plus two explicit
#   corrections), the margin covers the summation of 100-odd rows.
# - constant temperature: thermal row = T * continuity row to 1e-12, Jacobian
#   dual vs coloured <= 1e-12*scale and the central-FD gate, adjoint
#   contraction <= 1e-12, adjoint residual <= 1e-10, full-resolve FD best
#   relative <= 1e-6: copied unchanged from incns_fv_energy_test.jl (T2) and
#   incns_fv_thermal_adjoint_test.jl (T4).
# Run: KRAKEN_ONLY=analytical/incns_fv_energy_open_bc_test.jl julia --project=. -e 'using Pkg; Pkg.test()'
using Test, KrakeNN, ForwardDiff, LinearAlgebra, Random
using KrakeNN.SparseArrays

const _FVOB_BC = (W=:inlet, E=:outlet, S=:wall, N=:wall)
_fvob_zero(x, y) = 0.0
# Poiseuille profile of mean Um across a gap G (y measured from the bottom wall).
_fvob_parabola(y, G, Um) = 6Um * (y / G) * (1 - y / G)

# ------------------------------------------------------------ Graetz series ---
# Copied from test O2 of the LBM open-die driver (feat/gnf-open-steady):
# Y'' + mu (1 - eta^2) Y = 0, Y(0) = 1, Y'(0) = 0, RK4 on [0, 1].
function _fvob_shoot(mu; n=4000)
    h = 1.0 / n
    eta = collect(range(0.0, 1.0; length=n + 1))
    Y = zeros(n + 1); P = zeros(n + 1)
    Y[1] = 1.0
    rhs(e, y, p) = (p, -mu * (1 - e^2) * y)
    for k in 1:n
        e = eta[k]; y = Y[k]; p = P[k]
        k1 = rhs(e, y, p)
        k2 = rhs(e + h / 2, y + h / 2 * k1[1], p + h / 2 * k1[2])
        k3 = rhs(e + h / 2, y + h / 2 * k2[1], p + h / 2 * k2[2])
        k4 = rhs(e + h, y + h * k3[1], p + h * k3[2])
        Y[k + 1] = y + h / 6 * (k1[1] + 2k2[1] + 2k3[1] + k4[1])
        P[k + 1] = p + h / 6 * (k1[2] + 2k2[2] + 2k3[2] + k4[2])
    end
    return eta, Y, P
end
_fvob_simpson(eta, g) = (h = eta[2] - eta[1]; n = length(eta) - 1;
    h / 3 * (g[1] + g[end] + 4sum(g[2:2:n]) + 2sum(g[3:2:n-1])))
function _fvob_graetz_modes(nmodes)
    modes = NamedTuple[]
    for n in 0:nmodes-1
        lo = 4n + 1.0; hi = 4n + 2.4
        Ylo = _fvob_shoot(lo^2)[2][end]
        for _ in 1:60
            mid = (lo + hi) / 2
            Ym = _fvob_shoot(mid^2)[2][end]
            if sign(Ym) == sign(Ylo)
                lo = mid; Ylo = Ym
            else
                hi = mid
            end
        end
        lambda = (lo + hi) / 2
        eta, Y, P = _fvob_shoot(lambda^2)
        w = 1 .- eta .^ 2
        I1 = _fvob_simpson(eta, w .* Y)
        I2 = _fvob_simpson(eta, w .* Y .^ 2)
        push!(modes, (; mu=lambda^2, C=I1 / I2, dY1=P[end], IY=I1))
    end
    return modes
end
# Local Nusselt number (hydraulic diameter 2G) and mixing-cup temperature
# at x from the inlet; b = G/2.
function _fvob_graetz_nu(modes, x, Umean, b, alpha)
    s_grad = 0.0; s_bulk = 0.0
    for m in modes
        e = exp(-m.mu * alpha * x / (1.5 * Umean * b^2))
        s_grad += m.C * m.dY1 * e
        s_bulk += m.C * m.IY * e
    end
    thb = 1.5 * s_bulk
    return 4 * abs(s_grad) / thb, thb
end
# Numerical local Nu exactly as test O2: second-order wall gradient, bulk
# temperature weighted by the cell x velocity, h = 1.
function _fvob_nusselt_errors(T, u, modes, Umean, alpha)
    Nx, G = size(T)
    errors = Float64[]
    for i in 1:Nx
        x = i - 0.5
        (G <= x <= Nx - G) || continue
        gs = (9T[i, 1] - T[i, 2]) / 3
        gn = (9T[i, G] - T[i, G - 1]) / 3
        thb = sum(u[i, :] .* T[i, :]) / sum(u[i, :])
        nu_num = 2G * (gs + gn) / 2 / thb
        nu_ser, _ = _fvob_graetz_nu(modes, x, Umean, G / 2, alpha)
        push!(errors, abs(nu_num - nu_ser) / nu_ser)
    end
    return errors
end

# ------------------------------------------------------------ shared helpers ---
function _fvob_dense_jacobian(p, U)
    D = ForwardDiff.Dual{KrakeNN.FVJacobianTag,Float64,4}
    X = D.(U); J = zeros(length(U), length(U)); R = similar(X)
    for col in eachindex(U)
        X[col] = D(U[col], ForwardDiff.Partials((1.0, 0.0, 0.0, 0.0)))
        fv_residual!(R, X, p)
        for row in eachindex(U)
            J[row, col] = ForwardDiff.partials(R[row])[1]
        end
        X[col] = D(U[col])
    end
    return J
end

# Rhie-Chow flux of every unique interior face (upwind decisions); `axis` 1 = x, 3 = y.
function _fvob_fluxes(p, U)
    g = p.grid; flux = Float64[]; axis = Int[]
    visc = KrakeNN._fv_bind_viscosity(p.visc, U, g)
    for j in 1:g.Ny, i in 1:g.Nx
        p.tag[i, j] == FV_CELL_SOLID && continue
        gp = KrakeNN._fv_cell_grad(U, p.tag, p.bvel, g, i, j)
        for d in (1, 3)
            kind, ii, jj = KrakeNN._fv_face_kind(g, p.tag, i, j, d)
            kind == KrakeNN._FV_F_INT || continue
            gn = KrakeNN._fv_cell_grad(U, p.tag, p.bvel, g, ii, jj)
            up, vp, pp = ntuple(c -> U[fv_index(g, i, j, c)], 3)
            un, vn, pn = ntuple(c -> U[fv_index(g, ii, jj, c)], 3)
            grad = d == 1 ? ((un - up) / g.h, (gp[2] + gn[2]) / 2, (vn - vp) / g.h, (gp[4] + gn[4]) / 2) :
                ((gp[1] + gn[1]) / 2, (un - up) / g.h, (gp[3] + gn[3]) / 2, (vn - vp) / g.h)
            nu = KrakeNN._fv_face_nu(visc, p.nu[i, j], p.nu[ii, jj], grad..., i, j, ii, jj)
            push!(flux, KrakeNN._fv_interior_mass_flux(g.h, g.conv, up, vp, pp, un, vn, pn,
                gp, gn, nu, d == 1, 1))
            push!(axis, d)
        end
    end
    return flux, axis
end

# Newton (production gate) then two explicit corrections; returns the
# remaining correction and residual (as _fvt4_solve! of the T4 file).
function _fvob_solve!(s, U)
    stats = fv_newton!(s, U; maxit=30)
    @test stats.converged
    for _ in 1:2
        fv_newton_finalize!(s, U)
        fv_residual!(s.R, U, s.prob)
        lin_solve!(s.dU, s.cache, s.R)
        U .-= s.dU
    end
    fv_newton_finalize!(s, U)
    fv_residual!(s.R, U, s.prob)
    lin_solve!(s.dU, s.cache, s.R)
    return copy(s.dU), norm(s.R, Inf)
end

# Small open channel of unit height: parabolic inlet (mean 0.5) on W, outlet
# on E, Dirichlet bottom wall, adiabatic top wall, a 2 x 3 solid block.
const _FVOB_SOLID = let s = falses(16, 8); s[6:7, 3:5] .= true; s end
function _fvob_channel(; model=:gnf, scheme=:linear_upwind, sigma=0.4, Tin=1.0, Tw=0.0, Ts=0.5)
    opts = model === :gnf ? (; rheology=CarreauYasuda(0.3, 0.05, 1.2, 2.0, 0.7), gnf_eps=0.02) :
        (; nu=0.3)
    return fv_steady_problem(16, 8; bc=_FVOB_BC, solid=_FVOB_SOLID, conv=true,
        boundary_velocity=(side, x, y) -> (side === :W ? _fvob_parabola(y, 1.0, 0.5) : 0.0, 0.0),
        opts..., energy=FVEnergy(; alpha=0.05, sigma, left=Tin, bottom=Tw, solid=Ts, scheme))
end

# Every boundary term of the energy balance, recomputed from the fields and
# the BC data only (heat entering the domain counts positive).
function _fvob_balance(p, U; Tin, Tw, Ts, alpha, sigma)
    g = p.grid; h = g.h
    u, v, _, T = fv_fields(p, U)
    fluid = p.tag .!= FV_CELL_SOLID
    adv_in = 0.0; cond_in = 0.0; adv_out = 0.0; wall = 0.0; solid = 0.0; heat = 0.0
    mass_in = 0.0; mass_out = 0.0; inflow_positive = true
    for j in 1:g.Ny
        if fluid[1, j]
            F = h * p.bvel[1, j, 2]            # inlet x velocity at the W face
            inflow_positive &= F > 0
            adv_in += F * Tin; mass_in += F
            cond_in += 2alpha * (Tin - T[1, j])
        end
        if fluid[g.Nx, j]
            F = h * u[g.Nx, j]
            adv_out += F * T[g.Nx, j]; mass_out += F
        end
    end
    for i in 1:g.Nx
        fluid[i, 1] && (wall += 2alpha * (Tw - T[i, 1]))
    end
    for j in 1:g.Ny, i in 1:g.Nx
        fluid[i, j] || continue
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ii = i + di; jj = j + dj
            (1 <= ii <= g.Nx && 1 <= jj <= g.Ny) || continue
            fluid[ii, jj] || (solid += 2alpha * (Ts - T[i, j]))
        end
        heat += KrakeNN._fv_energy_heating(U, p.tag, p.nu, p.bvel, g, i, j, p.visc, sigma)
    end
    terms = (; adv_in, cond_in, adv_out, wall, solid, heat)
    imbalance = adv_in + cond_in - adv_out + wall + solid + heat
    return (; terms, imbalance, mass_in, mass_out, inflow_positive)
end

# Die driver, loaded at top level (world age) for its smoke testset.
ENV["KRAKEN_DIE_BACKEND"] = "cpu"
include(joinpath(@__DIR__, "..", "..", "benchmarks", "fv_thermal", "die_predict.jl"))

@testset "FV energy open boundaries (CPU Float64)" begin
    @testset "configuration and rejections" begin
        @test_throws ArgumentError fv_steady_problem(4, 4; bc=_FVOB_BC,
            energy=FVEnergy(; alpha=0.2, bottom=1.0))                  # inlet without temperature
        @test_throws ArgumentError fv_steady_problem(4, 4; bc=_FVOB_BC,
            energy=FVEnergy(; alpha=0.2, left=1.0, right=0.0))         # outlet with a temperature
        @test_throws ArgumentError FVEnergy(; alpha=0.2, solid=Inf)
        @test_throws ArgumentError FVEnergy(; alpha=0.2, solid=:hot)
        # Face velocities serve one outlet per axis (shared wrap slot).
        two = (W=:outlet, E=:outlet, S=:wall, N=:wall)
        @test_throws ArgumentError fv_steady_problem(4, 4; bc=two,
            energy=FVEnergy(; alpha=0.2, bottom=1.0, face_velocity=(zeros(4, 4), zeros(4, 4))))
        @test fv_steady_problem(4, 4; bc=two, energy=FVEnergy(; alpha=0.2, bottom=1.0)).energy isa
            KrakeNN._FVEnergyData
        # Right pocket behind a solid column: outlet and adiabatic walls only.
        solid = falses(8, 4); solid[4, :] .= true
        @test_throws ArgumentError fv_steady_problem(8, 4; bc=_FVOB_BC, solid,
            energy=FVEnergy(; alpha=0.2, left=1.0))
        p = fv_steady_problem(8, 4; bc=_FVOB_BC, solid,
            energy=FVEnergy(; alpha=0.2, left=1.0, solid=0.5))
        @test p.energy.solid_dirichlet && p.energy.solid == 0.5
        # Default T_ref: mean of the prescribed side temperatures and the solid one.
        p = fv_steady_problem(8, 4; bc=_FVOB_BC, solid,
            energy=FVEnergy(; alpha=0.2, left=1.0, bottom=0.0, solid=0.5))
        @test p.energy.Tref ≈ 0.5
        # Historical problems are unchanged: adiabatic solids, same T_ref.
        q = fv_steady_problem(4, 4; bc=(W=:periodic, E=:periodic, S=:wall, N=:wall),
            energy=FVEnergy(; alpha=0.2, bottom=1.0, top=3.0))
        @test !q.energy.solid_dirichlet && q.energy.Tref == 2.0
    end

    @testset "constant temperature: thermal row = T x continuity row" begin
        rng = MersenneTwister(1911)
        for scheme in (:linear_upwind, :upwind), Ts in (:adiabatic, 2.5)
            p = fv_steady_problem(16, 8; bc=_FVOB_BC, solid=_FVOB_SOLID,
                boundary_velocity=(side, x, y) -> (side === :W ? _fvob_parabola(y, 1.0, 0.5) : 0.0, 0.0),
                energy=FVEnergy(; alpha=0.2, left=2.5, bottom=2.5, top=2.5, solid=Ts, scheme, T_ref=0.0))
            U = randn(rng, fv_nunknowns(p)); U[4:4:end] .= 2.5
            U[fv_index(p.grid, 16, 1, 1)] = 0.5; U[fv_index(p.grid, 16, 2, 1)] = -0.5
            tags = copy(p.tag); tags[tags .== FV_CELL_PINNED] .= FV_CELL_FLUID
            R = fv_residual(U, p); differences = Float64[]; outflow = Float64[]
            for j in 1:p.grid.Ny, i in 1:p.grid.Nx
                p.tag[i, j] == FV_CELL_SOLID && continue
                rc = KrakeNN.fv_cell_residual(U, tags, p.nu, p.bvel, p.src, p.grid, i, j, p.visc)[3]
                push!(differences, abs(R[fv_index(p.grid, i, j, 4)] - 2.5rc))
                i == p.grid.Nx && push!(outflow, U[fv_index(p.grid, i, j, 1)])
            end
            @test maximum(differences) <= 1e-12
            @test minimum(outflow) < 0 < maximum(outflow)   # reversed outlet faces are exercised
            @info "open BC constant-temperature conservation" scheme Ts max_difference=maximum(differences)
        end
    end

    @testset "every row against the written BC formulas (prescribed face velocities)" begin
        # Upwind, h = 0.5, T_ref = 0: inlet W (u_in, T_in), outlet E (both flow
        # directions), Dirichlet S wall, adiabatic N wall, Dirichlet solid block.
        Nx, Ny, h, alpha, Tin, Tb, Ts = 6, 4, 0.5, 0.3, 2.0, 1.0, 1.7
        solid = falses(Nx, Ny); solid[3, 2:3] .= true
        uin = [0.7 + 0.1j for j in 1:Ny]
        uf = [0.4 + 0.05i - 0.08j for i in 1:Nx, j in 1:Ny]; uf[Nx, 1] = -0.3
        vf = [0.03i - 0.02j + 0.01 for i in 1:Nx, j in 1:Ny]
        T = [0.1i + 0.37j + 0.05i * j for i in 1:Nx, j in 1:Ny]
        p = fv_steady_problem(Nx, Ny; h, bc=_FVOB_BC, solid,
            boundary_velocity=(side, x, y) -> (side === :W ? uin[round(Int, y / h + 0.5)] : 0.0, 0.0),
            energy=FVEnergy(; alpha, left=Tin, bottom=Tb, solid=Ts, scheme=:upwind, T_ref=0.0,
                            face_velocity=(uf, vf)))
        R = fv_residual(fv_state(p, _fvob_zero, _fvob_zero, _fvob_zero, T), p)
        up(F, a, b) = max(F, 0) * a + min(F, 0) * b
        dirichlet(TP, Tw) = 2alpha * (TP - Tw)
        worst = 0.0
        for j in 1:Ny, i in 1:Nx
            solid[i, j] && continue
            TP = T[i, j]; r = 0.0
            # E: outlet F*T_P (either sign, no diffusion) / solid / interior
            if i == Nx
                r += h * uf[Nx, j] * TP
            elseif solid[i + 1, j]
                r += dirichlet(TP, Ts)
            else
                r += up(h * uf[i, j], TP, T[i + 1, j]) + alpha * (TP - T[i + 1, j])
            end
            # W: inlet (upwind with T_in, half-cell Dirichlet) / solid / interior
            if i == 1
                r += up(-h * uin[j], TP, Tin) + dirichlet(TP, Tin)
            elseif solid[i - 1, j]
                r += dirichlet(TP, Ts)
            else
                r += up(-h * uf[i - 1, j], TP, T[i - 1, j]) + alpha * (TP - T[i - 1, j])
            end
            # N: adiabatic no-slip wall / solid / interior
            if j < Ny
                if solid[i, j + 1]
                    r += dirichlet(TP, Ts)
                else
                    r += up(h * vf[i, j], TP, T[i, j + 1]) + alpha * (TP - T[i, j + 1])
                end
            end
            # S: Dirichlet no-slip wall / solid / interior
            if j == 1
                r += dirichlet(TP, Tb)
            elseif solid[i, j - 1]
                r += dirichlet(TP, Ts)
            else
                r += up(-h * vf[i, j - 1], TP, T[i, j - 1]) + alpha * (TP - T[i, j - 1])
            end
            worst = max(worst, abs(R[fv_index(p.grid, i, j, 4)] - r))
        end
        @test worst <= 1e-13
        @info "open BC row formulas" max_difference=worst
    end

    @testset "Graetz problem: local Nusselt vs the parallel-plate series" begin
        modes = _fvob_graetz_modes(30)
        @test abs(8 / 3 * modes[1].mu - 7.5407) < 2e-4
        G = 64; Nx = 6G; Um = 1.0; Pe = 200.0; alpha = Um * G / Pe   # h = 1, Pe = U G / alpha
        inlet = (side, x, y) -> (side === :W ? _fvob_parabola(y, G, Um) : 0.0, 0.0)
        energy_kw = (; alpha, left=1.0, bottom=0.0, top=0.0)
        # (a) energy only, exact divergence-free face velocities (the truth's setting).
        uf = [_fvob_parabola(j - 0.5, G, Um) for i in 1:Nx, j in 1:G]; vf = zeros(Nx, G)
        p = fv_steady_problem(Nx, G; h=1.0, bc=_FVOB_BC, boundary_velocity=inlet,
            energy=FVEnergy(; energy_kw..., face_velocity=(uf, vf)))
        r = fv_energy_solve(p, fv_state(p, _fvob_zero, _fvob_zero, _fvob_zero, 0.5))
        @test r.converged
        errors = _fvob_nusselt_errors(r.T, uf, modes, Um, alpha)
        @test maximum(errors) < 0.02
        @info "open BC Graetz (face velocities)" max_error=maximum(errors) mean_error=sum(errors)/length(errors)
        # (b) coupled Stokes Newton: Rhie-Chow fluxes, outlet p = 0.
        p = fv_steady_problem(Nx, G; h=1.0, nu=1.0, conv=false, bc=_FVOB_BC,
            boundary_velocity=inlet, energy=FVEnergy(; energy_kw...))
        s = fv_newton_solver(p)
        U = fv_state(p, (x, y) -> _fvob_parabola(y, G, Um), _fvob_zero, _fvob_zero, 0.5)
        stats = fv_newton!(s, U; maxit=30)
        @test stats.converged
        u, _, _, T = fv_fields(p, U)
        errors = _fvob_nusselt_errors(T, u, modes, Um, alpha)
        @test maximum(errors) < 0.02
        @info "open BC Graetz (coupled Stokes)" iterations=stats.iterations max_error=maximum(errors) mean_error=sum(errors)/length(errors)
    end

    @testset "global energy balance with heating, inlet, outlet, solid" begin
        for model in (:newtonian, :gnf), scheme in (:linear_upwind, :upwind)
            Tin, Tw, Ts, alpha, sigma = 1.0, 0.0, 0.5, 0.05, 0.4
            p = _fvob_channel(; model, scheme, sigma, Tin, Tw, Ts)
            s = fv_newton_solver(p)
            U = fv_state(p, (x, y) -> _fvob_parabola(y, 1.0, 0.5), _fvob_zero, _fvob_zero, Tin)
            _fvob_solve!(s, U)
            b = _fvob_balance(p, U; Tin, Tw, Ts, alpha, sigma)
            scale = maximum(abs, values(b.terms))
            @test b.inflow_positive
            @test abs(b.imbalance) <= 1e-9 * scale
            @test abs(b.mass_in - b.mass_out) <= 1e-10 * b.mass_in
            @test b.terms.heat > 0 && b.terms.solid != 0 && b.terms.wall < 0
            @info "open BC energy balance" model scheme imbalance=b.imbalance scale terms=b.terms
        end
    end

    @testset "coloured / dense dual Jacobian and central FD (open sides, Dirichlet solid)" begin
        rng = MersenneTwister(52011)
        solid = falses(9, 7); solid[4:5, 3:4] .= true
        law = LearnedViscosity(TinyMLP{2,2,1}(); nu_ref=0.3, monotone=false)
        theta = [0.3, -0.2, 0.4, -0.3, 0.1, -0.15, 0.2, -0.15, 0.04]
        field = [0.13i + 0.09j + 0.03sin(i * j) for i in 1:9, j in 1:7]
        wlf = CarreauYasuda(0.3, 0.0, 1.2, 2.0, 0.6; thermal=WLFCoupling(1.0, 2.0, 5.0))
        models = ((:newtonian, (; nu=0.3)),
            (:gnf, (; rheology=CarreauYasuda(0.3, 0.05, 1.2, 2.0, 0.7), gnf_eps=0.02)),
            (:law, (; viscosity=FVLawViscosity(law, theta, (field,); eps=0.02))),
            (:thermal_gnf, (; viscosity=FVGNFViscosity(wlf, 0.02; temperature=FVStateTemperature()))))
        for (model, options) in models, scheme in (:linear_upwind, :upwind)
            p = fv_steady_problem(9, 7; solid, bc=_FVOB_BC, options...,
                boundary_velocity=(side, x, y) -> (side === :W ? 0.8 + 0.3sin(7y) : 0.0, side === :W ? 0.1 : 0.0),
                energy=FVEnergy(; alpha=0.2, sigma=0.3, left=2.4, bottom=1.6, solid=2.1, scheme))
            base = 0.3randn(rng, fv_nunknowns(p)); base[3:4:end] .*= 0.1
            base[4:4:end] .+= 0.1
            S = Matrix(fv_jacobian_pattern(p)); solver = fv_newton_solver(p)
            for direction in (1.0, -1.0)
                U = copy(base); U[1:4:end] .*= direction; U[2:4:end] .*= direction
                U[3:4:end] .*= direction
                flux, axis = _fvob_fluxes(p, U)
                for d in (1, 3)
                    @test minimum(flux[axis .== d]) < 0 < maximum(flux[axis .== d])
                end
                Jref = _fvob_dense_jacobian(p, U); J = Matrix(fv_jacobian!(solver, U))
                scale = max(1.0, maximum(abs, Jref))
                missing = findall((.!iszero.(Jref)) .& .!S)
                @test isempty(missing)
                @test maximum(abs, J - Jref) <= 1e-12 * scale
                v = randn(rng, length(U)); v ./= norm(v, Inf)
                delta = Float64[]
                for h in (1e-4, 1e-5, 1e-6, 1e-7)
                    @test sign.(first(_fvob_fluxes(p, U + h * v))) == sign.(flux)
                    @test sign.(first(_fvob_fluxes(p, U - h * v))) == sign.(flux)
                    fd = (fv_residual(U + h * v, p) - fv_residual(U - h * v, p)) / (2h)
                    push!(delta, norm(J * v - fd, Inf))
                end
                @test any(delta .<= 2e-8 .+ 2e-7 * norm(J * v, Inf))
                @info "open BC thermal Jacobian" model scheme direction missing=length(missing) max_dual_difference=maximum(abs, J - Jref) fd_absolute=Tuple(delta)
            end
        end
    end

    @testset "thermal adjoint vs full-resolve central FD (open channel)" begin
        Tw = 468.15
        steps = (1e-3, 3e-4, 1e-4)
        rel(a, b) = norm(a - b) / max(norm(a), norm(b), eps(Float64))
        e5theta = [2000.0, 0.4, 0.035, 0.30, 0.006, 1.5, 465.15, 2.5]
        e5 = E5ThermalLaw(e5theta; T_wall=Tw, nu_ref=0.3, shear_scale=2.0)
        learned = LearnedViscosity(TinyMLP{2,2,1}(; shift=(0.0, Tw), scale=(1.0, 2.0)); nu_ref=0.3, monotone=false)
        ltheta = [0.3, -0.2, 0.4, -0.3, 0.1, -0.15, 0.2, -0.15, 0.04]
        solid = falses(6, 5); solid[3, 3] = true
        for (kind, law, theta) in ((:e5, e5, e5theta), (:learned, learned, ltheta))
            p = fv_steady_problem(6, 5; bc=_FVOB_BC, solid, conv=true,
                boundary_velocity=(side, x, y) -> (side === :W ? _fvob_parabola(y, 1.0, 0.4) : 0.0, 0.0),
                source=(x, y) -> (0.0, 0.07sin(2x + 3y), 0.0),
                energy=FVEnergy(; alpha=0.08, sigma=0.6, left=Tw + 0.3, bottom=Tw, top=Tw, solid=Tw + 0.1),
                viscosity=FVLawViscosity(law, theta, (FVStateTemperature(),); eps=0.02))
            s = fv_newton_solver(p)
            U = fv_state(p, (x, y) -> _fvob_parabola(y, 1.0, 0.4), _fvob_zero, _fvob_zero, Tw)
            _fvob_solve!(s, U)
            map = FVLawParameters(length(theta)); ws = fv_adjoint_workspace(s, map)
            objectives = FVObjective[fv_mean_temperature(p; T_wall=Tw, target=0.25),
                fv_temperature_l2(p, Tw + 0.25; weights=0.7)]
            gradients = Vector{Float64}[]
            for obj in objectives
                r = fv_steady_gradient(s, U, theta, obj; params=map, ws)
                d = fv_steady_gradient(s, U, theta, obj; params=map, ws, law_gradient=:duals, finalize=false)
                @test rel(r.grad, d.grad) <= 1e-12
                @test norm(transpose(s.J) * r.λ - r.gU) / max(norm(r.gU), eps()) <= 1e-10
                push!(gradients, r.grad)
            end
            sfd = fv_newton_solver(p)
            baseflux = first(_fvob_fluxes(p, U))
            rng = MersenneTwister(7319)
            for k in 1:2
                direction = normalize(randn(rng, length(theta)))
                kind === :e5 && (direction .*= [2000.0, 0.4, 0.035, 0.30, 0.006, 1.5, 2.5, 2.5])
                exact = [dot(gr, direction) for gr in gradients]
                relative = zeros(length(objectives), length(steps)); absolute = similar(relative)
                switches = 0
                for (m, h) in enumerate(steps)
                    values_ = zeros(length(objectives), 2)
                    for (side, sgn) in enumerate((-1.0, 1.0))
                        fv_set_params!(sfd, map, theta .+ sgn * h .* direction)
                        V = copy(U); _fvob_solve!(sfd, V)
                        switches += count(sign.(first(_fvob_fluxes(sfd.prob, V))) .!= sign.(baseflux))
                        for (j, obj) in enumerate(objectives)
                            values_[j, side] = fv_objective(obj, V, sfd.prob)
                        end
                    end
                    for j in eachindex(objectives)
                        fd = (values_[j, 2] - values_[j, 1]) / (2h)
                        absolute[j, m] = abs(fd - exact[j])
                        relative[j, m] = absolute[j, m] / max(abs(exact[j]), eps(Float64))
                    end
                end
                fv_set_params!(sfd, map, theta)
                @test switches == 0
                for j in eachindex(objectives)
                    best = argmin(view(relative, j, :))
                    @test relative[j, best] <= 1e-6
                    @test absolute[j, best] <= 1e-9 + 1e-6 * abs(exact[j])
                    @info "open BC adjoint FD" kind direction=k objective=j exact=exact[j] relative=Tuple(relative[j, :]) minimum_flux=minimum(abs, baseflux)
                end
            end
        end
    end

    @testset "die driver smoke (benchmarks/fv_thermal/die_predict.jl, G = 8, CPU)" begin
        # Runs only; the FV - LBM comparison needs the GPU grid (G = 64).
        c = DiePredict.case_from_env(; G=8, profile=:linear)
        @test !any(c.solid[1, :]) && count(!, c.solid[c.Nx, :]) == 2
        mktempdir() do dir
            summary, _ = DiePredict.run_case(c, DiePredict.default_viscosity(c); outdir=dir)
            @test summary["converged"]
            @test summary["mass_balance_relative"] <= 1e-8
            @test summary["energy_balance"]["relative"] <= 1e-6
            @test 0 <= summary["theta_cup_outlet"] <= 1.1
            @test all(isfile, joinpath.(dir, ["summary.toml", "fields.csv", "ux.bin", "uy.bin",
                                             "p.bin", "theta.bin", "is_solid.bin"]))
            @info "die driver smoke" dp_tilde=summary["dp_tilde"] theta_cup=summary["theta_cup_outlet"] balance=summary["energy_balance"]["relative"]
        end
    end
end
