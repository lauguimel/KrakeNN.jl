# Conformation tensor in the steady 3D FV Newton solver (plan
# .orchestrator/fv-conformation-plan.md, rung CONF-1), CPU only, Float64 and BigFloat:
#   (a) pointwise law: (c - I)^2 against the matrix product, Oldroyd-B is c - I,
#       dual derivative against central differences, frame indifference,
#       Oldroyd-B simple-shear fixed point;
#   (b) pointwise Giesekus simple shear (alpha = 0.4, Wi = 0.1 ... 2) against the
#       closed-form viscometric functions (ported from
#       kraken-research/hidden-rheology/ve-duct/giesekus_viscometric.py), BigFloat and Float64;
#   (c) default path: conformation = nothing keeps 4 fields; nu_p = 0 gives the
#       Newtonian flow rows exactly; lowering errors;
#   (d) coloured Jacobian against dense ForwardDiff (walls, periodic, solid voxel,
#       moving tangential wall, Giesekus, imposed mean velocity);
#   (e) Oldroyd-B plane Poiseuille (Wi = 1 and 0.5): u equals the Newtonian
#       eta_0 = eta_s + eta_p discrete solution, second order against the parabola;
#       c_xy, c_xx, N1, N2 exact at the cell centres; odd-even indicator;
#   (f) same channel rotated (walls on x, flow along z);
#   (g) Giesekus plane Couette, continuation in Wi: eta_p, Psi1, Psi2 against the
#       closed form, u linear;
#   (h) square-duct slice 12 x 12 (Nx = 1 periodic, imposed mean velocity),
#       Giesekus De 0.1: converged, mirror and diagonal symmetries, eight-vortex
#       sense of the RheoTool solution, c positive definite; fv_set_conformation!
#       to De 0.2 (bit-identical residual, no new analysis); Oldroyd-B: N2 = 0 and
#       no secondary flow.
using Test
using LinearAlgebra
using Random
using ForwardDiff
using KrakeNN
using KrakeNN.SparseArrays

# ------------------------------------------------------------------ helpers
_fvct_mat(c) = [c[1] c[2] c[3]; c[2] c[4] c[5]; c[3] c[5] c[6]]
_fvct_tup(M) = (M[1, 1], M[1, 2], M[1, 3], M[2, 2], M[2, 3], M[3, 3])
_fvct_Ltup(L) = Tuple(vec(permutedims(L)))          # entry 3(a-1)+b = L[a, b]
_fvct_shearL(gd) = (zero(gd), gd, zero(gd), zero(gd), zero(gd), zero(gd), zero(gd), zero(gd), zero(gd))

# Giesekus steady simple shear, Wi = lambda gdot (giesekus_viscometric.py):
# returns (eta_p / eta_p0, Psi1 / (eta_p0 lambda), Psi2 / (eta_p0 lambda)).
function _fvct_giesekus_closed(alpha, Wi)
    x = 16 * alpha * (1 - alpha) * Wi^2
    s = sqrt(1 + x) + 1
    chi = sqrt(2 / s)
    g = 16 * alpha * (1 - alpha) / (s^2 * (1 + chi) * (1 + (1 - 2alpha) * chi))
    f = g * Wi^2
    return ((1 - f)^2 / (1 + (1 - 2alpha) * f), 2g * (1 - alpha * f) / (alpha * (1 - f)), -g)
end
# the same functions read off a conformation tuple
_fvct_viscometric(c, Wi) = (c[2] / Wi, (c[1] - c[4]) / Wi^2, (c[4] - c[6]) / Wi^2)

# pointwise Newton on the homogeneous steady residual
function _fvct_pointwise_newton(law, theta, lambda, L, c0; tol, maxit = 60)
    c = collect(c0)
    f(x) = collect(fv_conformation_source(law, theta, lambda, L, Tuple(x)))
    for it in 1:maxit
        r = f(c)
        maximum(abs, r) <= tol && return c, it, true
        c .-= ForwardDiff.jacobian(f, c) \ r
    end
    return c, maxit, maximum(abs, f(c)) <= tol
end

_fvct_dense(p, U) = ForwardDiff.jacobian((R, X) -> fv_residual!(R, X, p), zeros(length(U)), U)

_fvct_zero(p) = zeros(p.grid.Nx, p.grid.Ny, p.grid.Nz)
_fvct_rest(p) = (z = _fvct_zero(p); fv_state(p, z, z, z, z))

function _fvct_solve!(p, U; maxit = 20)
    s = fv_newton_solver(p)
    st = fv_newton!(s, U; atol = 0.0, rtol = 1e-10, maxit)
    return st
end

const _FVCT_YWALLS = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :periodic, T = :periodic)
const _FVCT_XWALLS = (W = :wall, E = :wall, S = :periodic, N = :periodic, B = :periodic, T = :periodic)
const _FVCT_DUCT = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)

# a second relaxation-law type (only to check that a law change is refused)
struct _FVCTOtherLaw <: AbstractRelaxationLaw end
KrakeNN.relaxation_nparams(::_FVCTOtherLaw) = 2
KrakeNN.relaxation_coefficients(::_FVCTOtherLaw, theta, I1, I2, I3) = (theta[1], theta[2])

# plane channel of width H = 1, body force F; walls on y (flow x) or on x (flow z)
function _fvct_channel(Ny; lambda, nu_s = 0.5, nu_p = 0.5, F = 2.0, axis = :y, conformation = true)
    h = 1 / Ny
    conf = conformation ? FVConformation3D(; lambda, nu_p) : nothing
    nu = conformation ? nu_s : nu_s + nu_p
    if axis === :y
        return fv_steady_problem(2, Ny, 2; h, nu, bc = _FVCT_YWALLS, conformation = conf,
                                 source = (x, y, z) -> (F, 0.0, 0.0, 0.0))
    else
        return fv_steady_problem(Ny, 2, 2; h, nu, bc = _FVCT_XWALLS, conformation = conf,
                                 source = (x, y, z) -> (0.0, 0.0, F, 0.0))
    end
end

@testset "FV 3D conformation tensor (CONF-1)" begin
    rng = MersenneTwister(20261111)
    law = PolynomialRelaxation()

    @testset "(a) pointwise law" begin
        for trial in 1:3
            c = _fvct_tup(Symmetric(I + 0.3 .* randn(rng, 3, 3)))
            d = KrakeNN._fvc_minus_identity(c)
            D = _fvct_mat(d)
            @test maximum(abs, collect(KrakeNN._fvc_square(d)) .- collect(_fvct_tup(D * D))) <= 1e-14
            # Oldroyd-B: Rlx = c - I exactly; Giesekus: (c - I) + alpha (c - I)^2
            @test fv_relaxation(law, (1.0, 0.0), c) == d
            @test maximum(abs, collect(fv_relaxation(law, (1.0, 0.4), c)) .-
                               collect(_fvct_tup(D + 0.4 * D * D))) <= 1e-14
            # upper-convected stretching against L c + c L^T
            Lm = randn(rng, 3, 3)
            C = _fvct_mat(c)
            @test maximum(abs, collect(fv_upper_convected(_fvct_Ltup(Lm), c)) .-
                               collect(_fvct_tup(Lm * C + C * Lm'))) <= 1e-13
            # dual derivative in c and L against central differences (polynomial: exact up to rounding)
            x0 = [collect(c); vec(permutedims(Lm))]
            f(x) = collect(fv_conformation_source(law, (1.0, 0.3), 0.7, Tuple(x[7:15]), Tuple(x[1:6])))
            Jd = ForwardDiff.jacobian(f, x0)
            Jfd = similar(Jd); e = 1e-5
            for q in eachindex(x0)
                xp = copy(x0); xm = copy(x0); xp[q] += e; xm[q] -= e
                Jfd[:, q] = (f(xp) - f(xm)) / (2 * e)
            end
            @test maximum(abs, Jd - Jfd) <= 1e-8 * max(1, maximum(abs, Jd))
            # frame indifference: R(Q c Q^T) = Q R(c) Q^T, and the stretching term with L -> Q L Q^T
            Q = Matrix(qr(randn(rng, 3, 3)).Q)
            cq = _fvct_tup(Q * C * Q')
            @test maximum(abs, collect(fv_relaxation(law, (1.0, 0.4), cq)) .-
                               collect(_fvct_tup(Q * _fvct_mat(fv_relaxation(law, (1.0, 0.4), c)) * Q'))) <= 1e-13
            @test maximum(abs, collect(fv_upper_convected(_fvct_Ltup(Q * Lm * Q'), cq)) .-
                               collect(_fvct_tup(Q * (Lm * C + C * Lm') * Q'))) <= 1e-12
        end
        # rest is a fixed point of every law; dual at c = I is defined
        cI = (1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
        @test all(iszero, fv_relaxation(law, (1.0, 0.4), cI))
        gI = ForwardDiff.jacobian(x -> collect(fv_relaxation(law, (1.0, 0.4), Tuple(x))), collect(cI))
        @test gI == Matrix(1.0I, 6, 6)
        # Oldroyd-B simple shear fixed point: c = [1 + 2Wi^2, Wi, 0, 1, 0, 1]
        for (lambda, gd) in ((1.0, 0.5), (0.5, 3.0), (2.0, 1.0))
            Wi = lambda * gd
            c = (1 + 2Wi^2, Wi, 0.0, 1.0, 0.0, 1.0)
            @test maximum(abs, fv_conformation_source(law, (1.0, 0.0), lambda, _fvct_shearL(gd), c)) <=
                  1e-14 * (1 + Wi^2) / lambda
        end
        # constructor checks
        @test FVConformation3D(; lambda = 1, nu_p = 0.5).theta == (1.0, 0.0)
        @test FVConformation3D(; lambda = 1, nu_p = 0.5, alpha = 0.4).theta == (1.0, 0.4)
        @test_throws ArgumentError FVConformation3D(; lambda = 0.0, nu_p = 0.5)
        @test_throws ArgumentError FVConformation3D(; lambda = 1.0, nu_p = -0.1)
        @test_throws DimensionMismatch FVConformation3D(; lambda = 1.0, nu_p = 0.1, theta = (1.0,))
        @test relaxation_nparams(law) == 2
    end

    @testset "(b) pointwise Giesekus simple shear vs closed form" begin
        alpha = 0.4
        for T in (BigFloat, Float64)
            setprecision(BigFloat, 256) do
                tol = T === BigFloat ? T(1e-60) : 1e-14
                rtol = T === BigFloat ? 1e-25 : 1e-12
                theta = T === BigFloat ? (big"1.0", big"0.4") : (1.0, alpha)
                c = (one(T), zero(T), zero(T), one(T), zero(T), one(T))
                for Wi in (big"0.1", big"0.5", big"1", big"2")
                    lambda = one(T); WiT = T(Wi)
                    cs, it, ok = _fvct_pointwise_newton(law, theta, lambda, _fvct_shearL(WiT / lambda), c; tol)
                    @test ok
                    num = _fvct_viscometric(cs, WiT)
                    ref = _fvct_giesekus_closed(theta[2], WiT)
                    @test all(abs(num[q] - ref[q]) <= rtol * abs(ref[q]) for q in 1:3)
                    @test abs(cs[3]) <= tol && abs(cs[5]) <= tol && abs(cs[6] - 1) <= 10tol
                    c = Tuple(cs)                       # continuation in Wi
                end
            end
        end
    end

    @testset "(c) default path and lowering" begin
        Nx, Ny, Nz = 3, 4, 2
        src = (x, y, z) -> (0.5, 0.1, 0.0, 0.0)
        bvel = (s, x, y, z) -> s === :N ? (0.7, 0.0, 0.3) : (0.0, 0.0, 0.0)
        solid = falses(Nx, Ny, Nz); solid[2, 2, 1] = true
        p0 = fv_steady_problem(Nx, Ny, Nz; nu = 0.3, bc = _FVCT_YWALLS, source = src,
                               boundary_velocity = bvel, solid)
        @test p0.conformation === nothing
        @test KrakeNN._fv_nfields(p0.grid) == 4
        @test fv_nunknowns(p0) == 4 * Nx * Ny * Nz
        # nu_p = 0: the four flow rows are the Newtonian ones, bit for bit
        pc = fv_steady_problem(Nx, Ny, Nz; nu = 0.3, bc = _FVCT_YWALLS, source = src,
                               boundary_velocity = bvel, solid,
                               conformation = FVConformation3D(; lambda = 0.8, nu_p = 0.0, alpha = 0.3))
        @test KrakeNN._fv_nfields(pc.grid) == 10
        @test pc.tag == p0.tag
        g0 = p0.grid; gc = pc.grid
        Uc = 0.4 .* randn(rng, fv_nunknowns(pc))
        U0 = zeros(fv_nunknowns(p0))
        for k in 1:Nz, j in 1:Ny, i in 1:Nx, a in 1:4
            U0[fv_index(g0, i, j, k, a)] = Uc[fv_index(gc, i, j, k, a)]
        end
        R0 = fv_residual(U0, p0); Rc = fv_residual(Uc, pc)
        @test all(Rc[fv_index(gc, i, j, k, a)] == R0[fv_index(g0, i, j, k, a)]
                  for k in 1:Nz, j in 1:Ny, i in 1:Nx, a in 1:4)
        # state at rest: c = I in fluid cells, 0 in the solid
        Ur = _fvct_rest(pc)
        cf = fv_conformation_fields(pc, Ur)
        @test cf.xx[1, 1, 1] == 1 && cf.zz[3, 4, 2] == 1 && cf.xy[1, 1, 1] == 0 && cf.xx[2, 2, 1] == 0
        @test_throws ArgumentError fv_conformation_fields(p0, U0)
        # lowering errors
        conf = FVConformation3D(; lambda = 1.0, nu_p = 0.5)
        @test_throws ArgumentError fv_steady_problem(3, 4, 2; conformation = conf,
            bc = (W = :inlet, E = :outlet, S = :wall, N = :wall, B = :periodic, T = :periodic))
        @test_throws ArgumentError fv_steady_problem(3, 4, 2; conformation = conf, bc = _FVCT_YWALLS,
            boundary_velocity = (s, x, y, z) -> s === :N ? (0.0, 0.2, 0.0) : (0.0, 0.0, 0.0))
        @test_throws ArgumentError fv_steady_problem(3, 4, 2; conformation = conf, bc = _FVCT_YWALLS,
                                                     energy = FVEnergy3D(; alpha = 0.1))
        @test_throws ArgumentError fv_steady_problem(3, 4, 2; conformation = conf, bc = _FVCT_YWALLS,
                                                     rheology = CarreauYasuda(0.1, 0.002, 3.0, 2.0, 0.4))
    end

    @testset "(d) coloured Jacobian == dense ForwardDiff" begin
        Nx, Ny, Nz = 3, 5, 4
        solid = falses(Nx, Ny, Nz); solid[2, 3, 2] = true
        bvel = (s, x, y, z) -> s === :N ? (0.7, 0.0, 0.3) : (0.0, 0.0, 0.0)
        conf = FVConformation3D(; lambda = 0.8, nu_p = 0.4, alpha = 0.3)
        cases = (
            ("Giesekus, moving wall, solid voxel, body force",
             fv_steady_problem(Nx, Ny, Nz; h = 0.2, nu = 0.3, bc = _FVCT_YWALLS, solid,
                               boundary_velocity = bvel, conformation = conf,
                               source = (x, y, z) -> (0.5, 0.0, 0.0, 0.0))),
            ("Giesekus, imposed mean velocity, Stokes",
             fv_steady_problem(Nx, Ny, Nz; h = 0.2, nu = 0.3, bc = _FVCT_YWALLS, solid, conv = false,
                               conformation = conf, source = (x, y, z) -> (0.5, 0.0, 0.0, 0.0),
                               u_mean_target = 0.2)),
        )
        for (name, p) in cases
            @testset "$name" begin
                S = fv_jacobian_pattern(p)
                n = fv_nunknowns(p)
                s = fv_newton_solver(p)
                g = p.grid
                for trial in 1:2
                    U = 0.4 .* randn(rng, n)
                    for cell in 1:fv_ncells(p), m in (1, 4, 6)       # c near I
                        U[KrakeNN._fv_cell_index(g, cell, KrakeNN._fv3_conf_slot(g.layout, m))] += 1
                    end
                    Jd = _fvct_dense(p, U)
                    @test all(S[k] for k in findall(!iszero, Jd))   # pattern is a superset
                    fv_jacobian!(s, U)
                    @test maximum(abs, Matrix(s.J) - Jd) <= 1e-12 * maximum(abs, Jd)
                end
                @info "3D conformation coloured Jacobian" name nnz(S) n
            end
        end
    end

    @testset "(e) Oldroyd-B plane Poiseuille" begin
        F = 2.0; nu_s = 0.5; nu_p = 0.5; nu0 = nu_s + nu_p
        for lambda in (1.0, 0.5)                   # Wi = lambda * gdot_wall = lambda F H / (2 nu0)
            G = nu_p / lambda
            errs = Float64[]
            for Ny in (16, 32, 64)
                h = 1 / Ny
                p = _fvct_channel(Ny; lambda, nu_s, nu_p, F)
                U = _fvct_rest(p)
                st = _fvct_solve!(p, U)
                @test st.converged
                @test st.iterations <= 8
                u, v, w, _ = fv_fields(p, U)
                c = fv_conformation_fields(p, U)
                # same discrete velocity as the Newtonian solvent nu0
                pn = _fvct_channel(Ny; lambda, nu_s, nu_p, F, conformation = false)
                Un = _fvct_rest(pn)
                @test _fvct_solve!(pn, Un).converged
                un = fv_fields(pn, Un)[1]
                y = [(j - 0.5) * h for j in 1:Ny]
                ue = F / (2nu0) .* y .* (1 .- y); Umax = maximum(ue)
                @test maximum(abs, u .- un) <= 1e-9 * Umax
                @test maximum(abs, v) <= 1e-10 * Umax && maximum(abs, w) <= 1e-10 * Umax
                e = maximum(abs(u[i, j, k] - ue[j]) for i in 1:2, j in 1:Ny, k in 1:2) / Umax
                push!(errs, e)
                @test e <= 2 * h^2
                # conformation at the cell centres
                gd = F / nu0 .* (0.5 .- y)
                cxy = lambda .* gd; cxx = 1 .+ 2 .* cxy .^ 2
                sxy = maximum(abs, cxy); sxx = maximum(cxx .- 1)
                @test maximum(abs(c.xy[i, j, k] - cxy[j]) for i in 1:2, j in 1:Ny, k in 1:2) <= 1e-8 * sxy
                @test maximum(abs(c.xx[i, j, k] - cxx[j]) for i in 1:2, j in 1:Ny, k in 1:2) <= 1e-8 * sxx
                N1 = G .* (c.xx .- c.yy); N1e = 2nu_p * lambda .* gd .^ 2
                @test maximum(abs(N1[i, j, k] - N1e[j]) for i in 1:2, j in 1:Ny, k in 1:2) <= 1e-8 * maximum(N1e)
                @test maximum(abs, G .* (c.yy .- c.zz)) <= 1e-10 * maximum(N1e)
                @test maximum(abs, c.yy .- 1) <= 1e-10 * sxx && maximum(abs, c.zz .- 1) <= 1e-10 * sxx
                @test maximum(abs, c.xz) <= 1e-10 * sxy && maximum(abs, c.yz) <= 1e-10 * sxy
                # tau_xy = eta_p gdot
                @test maximum(abs(G * c.xy[1, j, 1] - nu_p * gd[j]) for j in 1:Ny) <= 1e-8 * nu_p * maximum(abs, gd)
                # odd-even indicator: interior second differences of u constant (-F h^2 / nu0), of c_xy zero
                d2u = [u[1, j + 1, 1] - 2 * u[1, j, 1] + u[1, j - 1, 1] for j in 2:Ny-1]
                @test maximum(abs, d2u .+ F * h^2 / nu0) <= 1e-8 * Umax
                d2c = [c.xy[1, j + 1, 1] - 2 * c.xy[1, j, 1] + c.xy[1, j - 1, 1] for j in 2:Ny-1]
                @test maximum(abs, d2c) <= 1e-8 * sxy
                @info "OB Poiseuille" lambda Ny iterations = st.iterations uerr = e
            end
            @test log2(errs[2] / errs[3]) >= 1.9
        end
    end

    @testset "(f) rotated channel: walls on x, flow along z" begin
        Ny = 16; lambda = 1.0
        py = _fvct_channel(Ny; lambda); Uy = _fvct_rest(py)
        @test _fvct_solve!(py, Uy).converged
        px = _fvct_channel(Ny; lambda, axis = :x); Ux = _fvct_rest(px)
        @test _fvct_solve!(px, Ux).converged
        uy = fv_fields(py, Uy)[1]; cy = fv_conformation_fields(py, Uy)
        wx = fv_fields(px, Ux)[3]; cx = fv_conformation_fields(px, Ux)
        @test maximum(abs(wx[j, 1, 1] - uy[1, j, 1]) for j in 1:Ny) <= 1e-10 * maximum(abs, uy)
        @test maximum(abs(cx.xz[j, 1, 1] - cy.xy[1, j, 1]) for j in 1:Ny) <= 1e-10 * maximum(abs, cy.xy)
        @test maximum(abs(cx.zz[j, 1, 1] - cy.xx[1, j, 1]) for j in 1:Ny) <= 1e-10 * maximum(cy.xx)
        @test maximum(abs, cx.xx .- 1) <= 1e-10 && maximum(abs, cx.yy .- 1) <= 1e-10
    end

    @testset "(g) Giesekus plane Couette vs closed form" begin
        alpha = 0.4; lambda = 1.0; nu_s = 0.5; nu_p = 0.5; Ny = 8
        conf = FVConformation3D(; lambda, nu_p, alpha)
        U = nothing
        for Wi in (0.1, 0.5, 1.0, 2.0)
            Uw = Wi / lambda                        # H = 1: gdot = Uw
            p = fv_steady_problem(2, Ny, 2; h = 1 / Ny, nu = nu_s, bc = _FVCT_YWALLS, conformation = conf,
                                  boundary_velocity = (s, x, y, z) -> s === :N ? (Uw, 0.0, 0.0) : (0.0, 0.0, 0.0))
            U === nothing && (U = _fvct_rest(p))    # warm start from the previous Wi
            st = _fvct_solve!(p, U)
            @test st.converged
            u = fv_fields(p, U)[1]
            c = fv_conformation_fields(p, U)
            @test maximum(abs(u[i, j, k] - Uw * (j - 0.5) / Ny) for i in 1:2, j in 1:Ny, k in 1:2) <= 1e-10 * Uw
            ref = _fvct_giesekus_closed(alpha, Wi)
            worst = 0.0
            for k in 1:2, j in 1:Ny, i in 1:2
                cc = (c.xx[i, j, k], c.xy[i, j, k], c.xz[i, j, k], c.yy[i, j, k], c.yz[i, j, k], c.zz[i, j, k])
                num = _fvct_viscometric(cc, Wi)
                worst = max(worst, maximum(abs(num[q] - ref[q]) / abs(ref[q]) for q in 1:3))
            end
            @test worst <= 1e-8
            @info "Giesekus Couette" Wi iterations = st.iterations worst
        end
    end

    @testset "(h) square-duct slice 12 x 12, De 0.1 and 0.2 (CONF-2)" begin
        # 1 x N x N periodic slice of a square duct of side 1, mean axial velocity 1,
        # eta_s = eta_p = 1 (beta = 0.5, Re = 0.5), lambda = De (U = H = 1), as the
        # RheoTool case and benchmarks/ve_duct/slice_conformation.jl.
        N = 12; M = N + 1
        duct(conf) = fv_steady_problem(1, N, N; h = 1 / N, nu = 1.0, bc = _FVCT_DUCT, conformation = conf,
                                       source = (x, y, z) -> (50.0, 0.0, 0.0, 0.0), u_mean_target = 1.0)
        rest(p) = (z = zeros(1, N, N); fv_state(p, z, z, z, z))
        giesekus(de) = FVConformation3D(; lambda = de, nu_p = 1.0, alpha = 0.4)
        ok(st) = st.converged || (st.stagnated && last(st.inf) <= 1e-11)
        newton = (; atol = 1e-13, rtol = 1e-12, maxit = 30)
        secondary(p, U) = (f = fv_fields(p, U); (f[2][1, :, :], f[3][1, :, :], f[1][1, :, :]))

        # Giesekus De 0.1 from rest (u = v = w = 0, c = I, Fx = 50)
        p = duct(giesekus(0.1))
        s = fv_newton_solver(p)
        U = rest(p)
        st = fv_newton!(s, U; newton...)
        @test ok(st)
        @test st.iterations <= 15
        v, w, u = secondary(p, U)
        Um = sum(u) / length(u)
        @test abs(Um - 1) <= 1e-10
        sp = sqrt.(v .^ 2 .+ w .^ 2)
        m1 = maximum(sp)
        @test 1e-4 < m1 < 1e-2                     # RheoTool 200^2: 1.29e-3
        tol = 1e-6 * m1 + 1e-13
        # mirror y -> 1 - y, z -> 1 - z and the diagonal y <-> z
        @test maximum(abs(v[j, k] + v[M - j, k]) for j in 1:N, k in 1:N) <= tol
        @test maximum(abs(v[j, k] - v[j, M - k]) for j in 1:N, k in 1:N) <= tol
        @test maximum(abs(w[j, k] - w[M - j, k]) for j in 1:N, k in 1:N) <= tol
        @test maximum(abs(w[j, k] + w[j, M - k]) for j in 1:N, k in 1:N) <= tol
        @test maximum(abs(v[j, k] - w[k, j]) for j in 1:N, k in 1:N) <= tol
        # eight vortices, sense of the RheoTool solution (checked in its De 0.1 field):
        # along the diagonal from the corner toward the centre, along the wall
        # bisectors (between cells N/2 and N/2 + 1) toward the walls
        @test all(v[j, j] > 0 && w[j, j] > 0 for j in 2:5)
        @test all((v[j, 6] + v[j, 7]) / 2 < 0 for j in 2:4)
        @test all((w[6, k] + w[7, k]) / 2 < 0 for k in 2:4)
        # c stays positive definite
        c = fv_conformation_fields(p, U)
        mineig = minimum(eigmin(Symmetric(_fvct_mat((c.xx[1, j, k], c.xy[1, j, k], c.xz[1, j, k],
                                                    c.yy[1, j, k], c.yz[1, j, k], c.zz[1, j, k]))))
                         for j in 1:N, k in 1:N)
        @test mineig > 0
        @info "Giesekus duct slice" N De = 0.1 iterations = st.iterations max_uxy = m1 mineig

        # continuation: new coefficients on the same solver, bit-identical residual
        fv_set_conformation!(s, giesekus(0.2))
        p2 = duct(giesekus(0.2))
        X = U .+ 1e-3 .* randn(rng, length(U))
        @test fv_residual(X, s.prob) == fv_residual(X, p2)
        @test s.prob.conformation === p2.conformation
        nf = s.n_factorize
        st2 = fv_newton!(s, U; newton...)
        @test ok(st2)
        @test st2.iterations <= 10
        @test s.n_factorize == nf                  # no new symbolic analysis
        m2 = maximum(hypot.(secondary(p, U)[1], secondary(p, U)[2]))
        @test m2 > m1                              # RheoTool: 1.29e-3 -> 2.88e-3
        @info "Giesekus duct slice" N De = 0.2 iterations = st2.iterations max_uxy = m2
        @test_throws ArgumentError fv_set_conformation!(s, FVConformation3D(; lambda = 0.2, nu_p = 1.0,
                                                        law = _FVCTOtherLaw(), theta = (1.0, 0.4)))
        p0 = fv_steady_problem(1, N, N; h = 1 / N, bc = _FVCT_DUCT)
        @test_throws ArgumentError fv_set_conformation!(fv_newton_solver(p0), giesekus(0.2))

        # Oldroyd-B (b2 = 0): N2 = 0, no secondary flow
        pob = duct(FVConformation3D(; lambda = 0.1, nu_p = 1.0))
        Uob = rest(pob)
        stob = fv_newton!(fv_newton_solver(pob), Uob; newton...)
        @test ok(stob)
        vo, wo, uo = secondary(pob, Uob)
        @test maximum(abs, vo) <= 1e-12 && maximum(abs, wo) <= 1e-12
        co = fv_conformation_fields(pob, Uob)
        N1 = co.xx .- co.yy
        @test maximum(abs, co.yy .- co.zz) <= 1e-12 * maximum(abs, N1)
        @test maximum(abs, co.yz) <= 1e-12 * maximum(abs, N1)
        @info "Oldroyd-B duct slice" N iterations = stob.iterations max_vw = max(maximum(abs, vo), maximum(abs, wo))
    end
end
