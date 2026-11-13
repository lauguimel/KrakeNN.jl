# V3 of the 3D steady FV solver (plan "solveur FV stationnaire 3D", step 4,
# increment I2), CPU only, Float64, tiny grids (6 × 5 × 4):
#   (1) composite PIV objective (FVPIVObjective): the device path (contraction
#       kernel, matrix products, scatter kernel, dense linear terms) against
#       the host reference built on piv_misfit, value and ∂𝒥/∂U (ForwardDiff);
#   (2) adjoint gradient vs centred finite differences of the converged
#       objective, every map × objective: uniform ν, ν field (directions),
#       side velocity, x source on a lid-driven duct with an off-centre
#       obstacle (PIV planes with thickness/window/mask + flow rate + wall
#       force; flow rate; wall force), and Carreau (a = 2, ε = 0) and NN law
#       parameters plus uniform ν on a flow-rate-controlled duct (PIV + Fx;
#       Fx; wall force). Every primal solve starts from rest. Gate frozen
#       from the 2D C4/M3 tests: best of two fixed steps, error <= 2e-7
#       absolute + 1e-6 relative. Law contraction == parameter duals (1e-12);
#       adjoint == forward tangent (1e-10);
#   (3) `abs` convention of the upwind flux at a zero-flux face (symmetric
#       duct state). ForwardDiff differentiates abs by the sign bit of the
#       value: abs'(+0.0) = +1, abs'(-0.0) = -1 (NOT abs'(0) = 0). The
#       assembled Jacobian therefore holds ONE of the two one-sided
#       derivatives, chosen by the signed zero of each face flux; the test
#       measures both limits separately, checks that the AD entry is one of
#       them, and that their jump matches conv h²/2 (2u_P − u_S − u_N).
#       No FD equality is required there (non-differentiable point).
using Test
using LinearAlgebra
using Random
using ForwardDiff
using KrakeNN

const _I2_NEWTON = (; atol = 1e-14, rtol = 1e-15, maxit = 40)
const _I2_LOG = Dict{String,Float64}()
_i2_log!(key, e) = (_I2_LOG[key] = max(get(_I2_LOG, key, 0.0), e); e)

_i2_bc = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)
function _i2_solid(N)
    solid = falses(N...)
    solid[3, 2:3, 2] .= true          # off-centre obstacle (no symmetry plane)
    solid[4, 2, 2] = true
    return solid
end

_i2_restart!(s, base) = (copyto!(s.prob.nu, base.nu); copyto!(s.prob.bvel, base.bvel);
                         copyto!(s.prob.src, base.src); s)

function _i2_solve!(s, U)
    st = fv_newton!(s, U; _I2_NEWTON...)
    st.converged || st.stagnated || error("Newton did not converge")
    @test last(st.inf) < 1e-11
    return U
end

function _i2_objective(sfd, base, map, θ, obj, U0)
    _i2_restart!(sfd, base)
    fv_set_params!(sfd, map, θ)
    U = _i2_solve!(sfd, copy(U0))
    return fv_objective(obj, U, sfd.prob)
end

_i2_scale(θ, d) = count(!iszero, d) == 1 ? max(abs(θ[findfirst(!iszero, d)]), 1e-3) / norm(d, Inf) :
                                          max(norm(θ, Inf), 1e-3) / norm(d, Inf)

# best-of-two centred differences along d; returns (absolute error, derivative)
function _i2_fd(sfd, base, map, θ, d, obj, U0, gd; steps = (1e-4, 2e-5))
    sc = _i2_scale(θ, d)
    errs = Float64[]
    for e in steps
        ε = e * sc
        fd = (_i2_objective(sfd, base, map, θ .+ ε .* d, obj, U0) -
              _i2_objective(sfd, base, map, θ .- ε .* d, obj, U0)) / (2ε)
        push!(errs, abs(fd - gd))
    end
    return minimum(errs)
end

# PIV planes on the 6 × 5 × 4 grid of spacing h = 0.2: z-normal stereo plane
# with light-sheet thickness, window and a mask; x-normal plain plane; y-normal
# plane with thickness and a one-axis window.
function _i2_piv(p, U, rng)
    g = p.grid; h = g.h
    tmpl = zeros(g.Nx, g.Ny, g.Nz)
    planes = (PIVPlane(; normal = :z, position = 0.37, xs = 0.15:0.25:1.05, ys = 0.12:0.2:0.9,
                       window = (0.2, 0.15), thickness = 0.08, stereo = true),
              PIVPlane(; normal = :x, position = 0.53, xs = 0.1:0.2:0.9, ys = 0.1:0.2:0.7),
              PIVPlane(; normal = :y, position = 0.41, xs = 0.2:0.3:1.1, ys = 0.15:0.25:0.65,
                       window = (0.1, 0.0), thickness = 0.05))
    ops = map(pl -> piv_operator(pl, tmpl; h = (h, h, h)), planes)
    u, v, w, _ = fv_fields(p, U)
    # "measured" field: the state perturbed by 5 % and a smooth mode
    pert(f) = 1.05 .* f .+ 0.02 .* [sin(1.3i + 0.7j - 0.4k) for i in 1:g.Nx, j in 1:g.Ny, k in 1:g.Nz]
    data = Any[piv_apply(op, (pert(u), pert(v), pert(w))) for op in ops]
    m1 = ones(size(data[1][1])); m1[2, 3] = 0; m1[4, 1] = 0
    data[1] = map(d -> (d = copy(d); d[m1 .== 0] .= NaN; d), data[1])
    return ops, Tuple(data), (m1, nothing, nothing)
end

# ------------------------------------------------------------------ (1) objective
@testset "FVPIVObjective: device path == piv_misfit reference" begin
    rng = MersenneTwister(5)
    N = (6, 5, 4)
    p = fv_steady_problem(N...; h = 0.2, nu = 0.05, bc = _i2_bc, solid = _i2_solid(N),
                          source = (x, y, z) -> (1.0, 0.0, 0.0, 0.0),
                          boundary_velocity = (s, x, y, z) -> s === :T ? (0.2, 0.1, 0.0) : (0.0, 0.0, 0.0))
    U = 0.3 .* randn(rng, fv_nunknowns(p))
    ops, data, mask = _i2_piv(p, U, rng)
    terms = ((fv_flow_rate(p, 2), 0.3), (FVWallForce(1; region = :solid), 0.5, 0.01))
    obj = FVPIVObjective(p, ops, data; mask, sigma_u = 0.05, terms)
    @test fv_objective_explicit(obj)
    J, gU = fv_objective_gradient!(obj, U, p)
    Jref = fv_objective(obj, U, p)
    gref = ForwardDiff.gradient(X -> fv_objective(obj, X, p), U)
    @test abs(J - Jref) <= 1e-12 * abs(Jref)
    @test norm(gU - gref) <= 1e-12 * norm(gref)
    # PIV part alone: each plane is piv_misfit / σ²
    objp = FVPIVObjective(p, ops, data; mask, sigma_u = 0.05)
    u, v, w, _ = fv_fields(p, U)
    Lsum = sum(piv_misfit(ops[q], (u, v, w), data[q]; mask = mask[q])[1] for q in 1:3)
    @test abs(fv_objective_gradient!(objp, U, p)[1] - Lsum / 0.05^2) <= 1e-12 * Lsum / 0.05^2
    @test !fv_objective_explicit(objp)
    # (device bytes of this path are measured on the GPU by benchmarks/fv3d/i2_adjoint_measure.jl;
    #  on the CPU backend each kernel launch allocates its task, so no CPU allocation gate here)
    @info "I2 PIV objective vs reference" rel_value = abs(J - Jref) / abs(Jref) rel_grad = norm(gU - gref) / norm(gref)
end

# ------------------------------------------------------------------ (2) gradients
@testset "3D adjoint vs centred FD: Newtonian duct with lid and obstacle" begin
    rng = MersenneTwister(11)
    N = (6, 5, 4)
    lid = (s, x, y, z) -> s === :T ? (0.2, 0.1, 0.0) : (0.0, 0.0, 0.0)
    base = fv_steady_problem(N...; h = 0.2, nu = (x, y, z) -> 0.05 * (1 + 0.3x - 0.2y + 0.1z),
                             bc = _i2_bc, solid = _i2_solid(N),
                             source = (x, y, z) -> (1.0, 0.0, 0.0, 0.0), boundary_velocity = lid)
    U0 = zeros(fv_nunknowns(base))
    s0 = fv_newton_solver(base)
    Ub = _i2_solve!(s0, copy(U0))                 # from rest
    ops, data, mask = _i2_piv(base, Ub, rng)
    objectives = (("PIV+Q+F", FVPIVObjective(base, ops, data; mask, sigma_u = 0.05,
                                            terms = ((fv_flow_rate(base, 2), 0.3),
                                                     (FVWallForce(1; region = :solid), 0.5, 0.01)))),
                  ("flow rate", fv_flow_rate(base, 5)),
                  ("wall force", FVWallForce(1; region = :solid)),
                  ("lid force", FVWallForce(1; region = :T, pressure = :linear)))
    nc = fv_ncells(base)
    maps = (("uniform nu", FVUniformViscosity(), [0.04], [[1.0]]),
            ("nu field", FVViscosityField(), vec(copy(base.nu)),
             [randn(rng, nc), vec([sin(1.1i) * cos(0.7j + 0.3k) for i in 1:6, j in 1:5, k in 1:4])]),
            ("side velocity T", FVSideVelocity(base, :T; shape = (x, y, z) -> (1.0, 0.5, 0.0)), [0.2], [[1.0]]),
            ("x source", FVUniformSource(1), [1.0], [[1.0]]))
    sfd = fv_newton_solver(base)
    for (mname, map, θ, dirs) in maps
        sθ = _i2_restart!(fv_newton_solver(base), base)
        fv_set_params!(sθ, map, θ)
        Uθ = _i2_solve!(sθ, copy(U0))             # from rest, at θ
        ws = fv_adjoint_workspace(sθ, map)
        for (oname, obj) in objectives
            r = fv_steady_gradient(sθ, Uθ, θ, obj; params = map, ws)
            @test r.residual_inf <= 1e-12
            @test all(isfinite, r.grad)
            for d in dirs
                gd = dot(r.grad, d)
                err = _i2_fd(sfd, base, map, θ, d, obj, Uθ, gd)
                @test err <= 2e-7 + 1e-6 * abs(gd)
                _i2_log!("$mname / $oname (rel)", err / max(abs(gd), 1e-300))
            end
            if oname == "PIV+Q+F"
                d = dirs[end]
                dU = fv_steady_tangent(sθ, Uθ, θ, d; params = map, ws)
                tg = dot(Array(r.gU), dU) + dot(r.explicit, d)
                @test abs(tg - dot(r.grad, d)) <= 1e-10 * max(abs(tg), norm(r.grad) * norm(d))
            end
        end
    end
end

function _i2_law_problem(law, theta, fields, eps; nu = 0.05)
    N = (6, 5, 4)
    return fv_steady_problem(N...; h = 0.2, nu, bc = _i2_bc, solid = _i2_solid(N),
                             source = (x, y, z) -> (0.0, 0.0, 0.0, 0.0), u_mean_target = 0.4,
                             viscosity = law === nothing ? nothing : FVLawViscosity(law, theta, fields; eps))
end

@testset "3D adjoint vs centred FD: flow-rate-controlled duct, Carreau / NN / uniform nu" begin
    rng = MersenneTwister(23)
    field = [0.2 * (i - 0.5) / 6 + 0.1 * (j - 0.5) / 5 - 0.15 * (k - 0.5) / 4 for i in 1:6, j in 1:5, k in 1:4]
    carreau = ClassicalLaw(CarreauYasuda(0.05, 0.01, 2.0, 2.0, 0.5))
    nn = LearnedViscosity(TinyMLP{2,2,1}(); nu_ref = 0.05, monotone = false)
    θnn = law_init(nn; rng = MersenneTwister(33)); θnn[3:4] .= (0.3, -0.2)
    θcy = [0.05, 0.01, 2.0, 2.0, 0.5]
    cases = (("Carreau a=2 eps=0", _i2_law_problem(carreau, θcy, (), 0.0), FVLawParameters(5), θcy,
              # a = 2 exactly is what makes eps = 0 admissible: no FD step along a
              # (its adjoint component is still checked against parameter duals)
              [Float64.(1:5 .== q) for q in (1, 2, 3, 5)]),
             ("NN", _i2_law_problem(nn, θnn, (field,), 0.02), FVLawParameters(length(θnn)), θnn,
              [randn(rng, length(θnn)), randn(rng, length(θnn))]),
             ("uniform nu", _i2_law_problem(nothing, nothing, (), 0.0), FVUniformViscosity(), [0.05], [[1.0]]))
    for (cname, base, map, θ, dirs) in cases
        U0 = KrakeNN._fv_initial_force!(zeros(fv_nunknowns(base)), base)
        s = fv_newton_solver(base)
        U = _i2_solve!(s, copy(U0))               # from rest
        Fx = fv_flow_force_objective(base)
        ops, data, mask = _i2_piv(base, U, rng)
        objectives = (("PIV+Fx", FVPIVObjective(base, ops, data; mask, sigma_u = 0.05,
                                               terms = ((Fx, 4.0, 1.1 * fv_flow_force(base, U)),))),
                      ("Fx", Fx), ("wall force", FVWallForce(1; region = :solid)))
        sfd = fv_newton_solver(base)
        ws = fv_adjoint_workspace(s, map)
        for (oname, obj) in objectives
            r = fv_steady_gradient(s, U, θ, obj; params = map, ws)
            @test r.residual_inf <= 1e-12
            if map isa FVLawParameters
                rd = fv_steady_gradient(s, U, θ, obj; params = map, ws, law_gradient = :duals, finalize = false)
                e = norm(r.grad - rd.grad) / max(norm(rd.grad), eps())
                @test e <= 1e-12
                _i2_log!("$cname contraction vs duals", e)
            end
            for d in dirs
                gd = dot(r.grad, d)
                err = _i2_fd(sfd, base, map, θ, d, obj, U, gd)
                @test err <= 2e-7 + 1e-6 * abs(gd)
                _i2_log!("$cname / $oname (rel)", err / max(abs(gd), 1e-300))
            end
        end
    end
end

# ------------------------------------------------------------------ (3) abs convention
@testset "abs convention of the upwind flux at zero-flux faces" begin
    N = (4, 5, 4); h = 0.25
    p = fv_steady_problem(N...; h, nu = 0.05, bc = _i2_bc)
    g = p.grid
    # symmetric duct state: u = u(y, z) only, v = w = p = 0 -> every y/z face carries F = 0
    uf(i, j, k) = sin(pi * (j - 0.5) / 5)^1 * (1 + 0.3 * (j - 0.5) / 5) * sin(pi * (k - 0.5) / 4)
    U = fv_state(p, [uf(i, j, k) for i in 1:4, j in 1:5, k in 1:4], zeros(N...), zeros(N...), zeros(N...))
    s = fv_newton_solver(p)
    fv_jacobian!(s, U)
    i, j, k = 2, 2, 3
    row = fv_index(g, i, j, k, 1); col = fv_index(g, i, j, k, 2)
    ad = s.J[row, col]                                 # assembled (coloured duals) value
    R0 = fv_residual(U, p)[row]
    side(ε) = (V = copy(U); V[col] += ε; (fv_residual(V, p)[row] - R0) / ε)
    right = side(1e-7); left = side(-1e-7)
    uP, uS, uN = uf(i, j, k), uf(i, j - 1, k), uf(i, j + 1, k)
    jump = g.conv * h^2 / 2 * (2uP - uS - uN)
    @test ForwardDiff.derivative(abs, 0.0) == 1.0 && ForwardDiff.derivative(abs, -0.0) == -1.0
    picked = abs(ad - right) <= 1e-6 * abs(right - left) ? :right :
             abs(ad - left) <= 1e-6 * abs(right - left) ? :left : :neither
    @test picked !== :neither                           # a one-sided derivative, not the mean
    @test abs(ad - (right + left) / 2) > 0.1 * abs(right - left)
    @test abs((right - left) - jump) <= 1e-6 * abs(jump)
    @test abs(jump) > 1e-3 * h^2                         # a real kink, not a smooth point
    # the smooth (continuity) row has equal one-sided derivatives
    rc = fv_index(g, i, j, k, 4)
    Rc0 = fv_residual(U, p)[rc]
    sidec(ε) = (V = copy(U); V[col] += ε; (fv_residual(V, p)[rc] - Rc0) / ε)
    @test abs(sidec(1e-7) - sidec(-1e-7)) <= 1e-6 * abs(s.J[rc, col])
    @info "I2 abs convention (u row, v column, cell (2,2,3))" ad picked right left jump_measured = right - left jump_predicted = jump
end

@info "I2 worst agreement per map / objective\n" * join(("  $k: $(round(v; sigdigits = 3))" for (k, v) in sort(collect(_I2_LOG))), "\n")
