# Shared setup of the derivative gate of the 3D FV conformation adjoint (plan
# "discrete viscoelastic adjoints", step F2): the square-duct slice, the rows of
# the coverage matrix, the converged states, the test objective and the step
# scales. Included at top level by
#   test/analytical/incns_fv3d_conformation_gate_test.jl   (the gate)
#   benchmarks/fv3d/ve_adjoint_step_study.jl               (the step study)
# so that both use the same cases. No @test here.
using LinearAlgebra
using Random
using KrakeNN

# ---------------------------------------------------------------- G2 thresholds
# Centred FD gate G2: two steps eps = h s_k, min over the steps of
# |FD - g_k| <= VE_G2_REL |g_k| + VE_G2_ABS |J| / s_k (the plan's target).
# Step study benchmarks/fv3d/ve_adjoint_step_study.jl, GPU run of 2026-11-16 on
# the 1 x 8 x 8 slice: Richardson ratio 4.00 for h = 1e-2 ... 1e-5 for Giesekus
# alpha (De 1) and FENE-P L2 (De 0.5), adjoint vs FD down to 2.3e-11 (alpha) and
# 1.3e-10 (L2) relative, Newton atol 1e-15 instead of 1e-14 changes nothing; the
# steps 1e-4 and 1e-5 pass this gate. The learned-law direction was added to the
# study afterwards (step F4).
const VE_G2_STEPS = (1e-4, 1e-5)
const VE_G2_REL = 1e-6
const VE_G2_ABS = 1e-10

const VE_BC = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)
const VE_NEWTON = (; atol = 1e-14, rtol = 1e-15, maxit = 40)

# 1 x N x N slice, h = 1/N (duct side H = 1), imposed mean axial velocity 1, so
# De = lambda U / H = lambda; eta_s = nu_s, eta_p = nu_p (rho = 1)
ve_slice(conf; nu_s = 1.0, N = 8) =
    fv_steady_problem(1, N, N; h = 1 / N, nu = nu_s, bc = VE_BC, conformation = conf,
                      source = (x, y, z) -> (50.0, 0.0, 0.0, 0.0), u_mean_target = 1.0)
ve_rest(p) = (z = zeros(1, p.grid.Ny, p.grid.Nz); fv_state(p, z, z, z, z))

function ve_solve!(s, U; atol = VE_NEWTON.atol, rtol = VE_NEWTON.rtol, maxit = VE_NEWTON.maxit)
    st = fv_newton!(s, U; atol, rtol, maxit)
    # below atol, or stopped (stagnation, maxit at a tight atol) at round-off level
    st.converged || last(st.inf) <= 1e-11 || error("Newton did not converge (residual $(last(st.inf)))")
    return st
end

# step scale s_k of the plan: 1 for alpha and epsilon, the value for lambda,
# nu_p, nu_s, L2 - 3 for L2
ve_scale(name, value) = name in (:alpha, :epsilon) ? 1.0 : name === :L2 ? value - 3 : value

# One row of the coverage matrix: law label, De (= lambda), the law parameter
# (name, value), the conformation at (lambda, nu_p, law parameter), the map and θ.
struct VERow
    label::String
    De::Float64
    par::Union{Nothing,Tuple{Symbol,Float64}}
    conf::Function                       # (lambda, nu_p, par) -> FVConformation3D
end

ve_names(r::VERow) = r.par === nothing ? [:lambda, :nu_p, :nu_s] : [:lambda, :nu_p, :nu_s, r.par[1]]
ve_map(r::VERow) = r.par === nothing ? FVRelaxationParameters(:lambda, :nu_p, :nu_s) :
                   FVRelaxationParameters(:lambda, :nu_p, :nu_s, :theta)
ve_theta(r::VERow) = r.par === nothing ? [r.De, 1.0, 1.0] : [r.De, 1.0, 1.0, r.par[2]]
ve_scales(r::VERow) = [ve_scale(n, v) for (n, v) in zip(ve_names(r), ve_theta(r))]

const VE_ALPHA = 0.3
ve_giesekus(De) = VERow("Giesekus De=$De", De, (:alpha, VE_ALPHA),
                        (lam, nup, a) -> FVConformation3D(; lambda = lam, nu_p = nup, alpha = a))
ve_fenep(De = 0.5) = VERow("FENE-P De=$De", De, (:L2, 50.0),
                           (lam, nup, L2) -> FVConformation3D(; lambda = lam, nu_p = nup, law = FENEPRelaxation(),
                                                              theta = (L2,)))

# rows of the coverage matrix except the learned law (step F4); De: Giesekus 0.1
# and 1, the other laws 0.5
function ve_rows()
    return [
        VERow("Oldroyd-B De=0.5", 0.5, nothing, (lam, nup, _) -> FVConformation3D(; lambda = lam, nu_p = nup)),
        ve_giesekus(0.1),
        ve_giesekus(1.0),
        VERow("linear PTT De=0.5", 0.5, (:epsilon, 0.2),
              (lam, nup, e) -> FVConformation3D(; lambda = lam, nu_p = nup, law = LinearPTTRelaxation(),
                                                theta = (e,))),
        VERow("exponential PTT De=0.5", 0.5, (:epsilon, 0.2),
              (lam, nup, e) -> FVConformation3D(; lambda = lam, nu_p = nup, law = ExponentialPTTRelaxation(),
                                                theta = (e,))),
        ve_fenep(0.5),
    ]
end

# Converged state of a row at θ from rest: continuation in lambda only (fixed
# nu_p, nu_s and law parameter), each stage started from the previous state.
function ve_state(r::VERow; N = 8, to = Array, linsolve = CPUBackendTag())
    par = r.par === nothing ? 0.0 : r.par[2]
    ramp = filter(<(r.De), [0.1, 0.25, 0.5, 0.75])
    p = ve_slice(r.conf(isempty(ramp) ? r.De : first(ramp), 1.0, par); N)
    s = fv_newton_solver(p; to, linsolve)
    U = to(ve_rest(p))
    for lam in [ramp; r.De]
        fv_set_conformation!(s, r.conf(lam, 1.0, par))
        ve_solve!(s, U)
    end
    m = ve_map(r)
    fv_set_params!(s, m, ve_theta(r))             # same coefficients, through the map
    ve_solve!(s, U)
    fv_check_conformation_domain(s.prob, U)
    return s, U
end

# Data state at θ* = θ + 0.1 s (additive), warm start from the state U at θ, on
# the backend `to`; returned on the host.
function ve_data_state(r::VERow, s, U; to = Array, linsolve = CPUBackendTag())
    sd = fv_newton_solver(KrakeNN._fv_host_problem(s.prob); to, linsolve)
    fv_set_params!(sd, ve_map(r), ve_theta(r) .+ 0.1 .* ve_scales(r))
    Ud = copy(U)
    ve_solve!(sd, Ud)
    fv_check_conformation_domain(sd.prob, Ud)
    return Array(Ud)
end

# section mean of N1 / G = c_xx - c_yy (linear functional on the conformation unknowns)
function ve_n1_functional(p)
    g = p.grid
    sxx = KrakeNN._fv3_conf_slot(g.layout, 1); syy = KrakeNN._fv3_conf_slot(g.layout, 4)
    idx = Int[]; w = Float64[]
    a = 1 / (g.Nx * g.Ny * g.Nz)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        push!(idx, fv_index(g, i, j, k, sxx)); push!(w, a)
        push!(idx, fv_index(g, i, j, k, syy)); push!(w, -a)
    end
    return FVLinearFunctional(idx, w)
end

# Test objective (plan R2) on the backend of `to`: stereo x-normal PIV plane at
# the cell centres of the section (3 components), axial and secondary velocities
# weighted separately (1/Σu*², 1/Σ(v*² + w*²), as mask weights), plus
# (Fx - Fx*)² / (2 Fx*²) and, with `conformation`, (N1/G - N1*/G)² / (2 (N1*/G)²),
# data from the host state Ud. Laws without secondary flow (Oldroyd-B, PTT,
# FENE-P: Σ(v*² + w*²) at round-off) weight the secondary components by 1/Σu*².
function ve_objective(p, Ud::Vector{Float64}; conformation = true, to = Array)
    ph = KrakeNN._fv_host_problem(p)
    g = ph.grid
    N = g.Ny
    u, v, w, _ = fv_fields(ph, Ud)
    sa = sum(abs2, u); ss = sum(abs2, v) + sum(abs2, w)
    ws = ss > 1e-20 * sa ? 1 / ss : 1 / sa
    cs = ((1:N) .- 0.5) ./ N
    plane = PIVPlane(; normal = :x, position = 0.5 * g.h, xs = cs, ys = cs, stereo = true)
    op = piv_operator(plane, to(zeros(g.Nx, g.Ny, g.Nz)); h = (g.h, g.h, g.h))
    data = piv_apply(KrakeNN._fv_piv_host_op(op), (u, v, w))              # (v, w, u) samples
    mask = (fill(ws, N, N), fill(ws, N, N), fill(1 / sa, N, N))
    iF = KrakeNN._fv_force_index(ph)
    Fs = Ud[iF]
    terms = Any[(fv_flow_force_objective(p), 1 / Fs^2, Fs)]
    if conformation
        n1 = ve_n1_functional(ph)
        N1s = fv_objective(n1, Ud, ph)
        push!(terms, (n1, 1 / N1s^2, N1s))
    end
    return FVPIVObjective(p, (op,), (data,); mask = (mask,), terms = Tuple(terms)), (; sa, ss, Fs)
end

ve_J(obj, U, p) = fv_objective_gradient!(obj, U, p)[1]

# objective reconverged at θp on the FD solver sf, warm start U0 (state domain checked)
function ve_J_at(sf, m, θp, U0, obj; atol = VE_NEWTON.atol, rtol = VE_NEWTON.rtol)
    fv_set_params!(sf, m, θp)
    Up = copy(U0)
    st = ve_solve!(sf, Up; atol, rtol)
    fv_check_conformation_domain(sf.prob, Up)
    return ve_J(obj, Up, sf.prob), last(st.inf)
end

# centred FD of the reconverged objective along direction d (or parameter k)
# with step h s
function ve_fd(sf, m, θ, d::AbstractVector, h, s, U0, obj; kw...)
    ε = h * s
    Jp, rp = ve_J_at(sf, m, θ .+ ε .* d, U0, obj; kw...)
    Jm, rm = ve_J_at(sf, m, θ .- ε .* d, U0, obj; kw...)
    return (Jp - Jm) / (2ε), max(rp, rm)
end
ve_fd(sf, m, θ, k::Integer, h, s, U0, obj; kw...) = ve_fd(sf, m, θ, Float64.(1:length(θ) .== k), h, s, U0, obj; kw...)

# ---------------------------------------------------------------- learned law (F4)
# NeuralRelaxation{4} around Giesekus alpha0 = 0.3 at De = 0.5, invariants scaled
# by 10 (I1 - 3 reaches O(10) near the walls); parameters: hidden layer drawn
# N(0, 1) with seed VE_NN_SEED, output weights zero ("start", the law is then
# Giesekus exactly) or N(0, 0.05²) with the same seed ("second" point, hidden
# derivatives visible). Map: all 7H = 28 entries, s = 1.
const VE_NN_SEED = 21
const VE_NN_DE = 0.5
ve_neural_law() = NeuralRelaxation{4}(; alpha0 = VE_ALPHA, scale = 10.0)
ve_neural_theta(point::Symbol) =
    relaxation_init(ve_neural_law(); rng = MersenneTwister(VE_NN_SEED),
                    output_std = point === :start ? 0.0 : 0.05)
ve_neural_conf(lam, θnet) = FVConformation3D(; lambda = lam, nu_p = 1.0, law = ve_neural_law(), theta = θnet)
ve_neural_map() = FVRelaxationParameters(:theta)
# parameter indices: hidden weight W[1, 1], hidden bias b[2], output V[2, 1] (b1), V[3, 2] (b2)
const VE_NN_H = 4
const VE_NN_HIDDEN = (1, 3VE_NN_H + 2)
const VE_NN_OUTPUT = (5VE_NN_H + 1, 6VE_NN_H + 2)

# converged state of the learned law at θnet, from rest by continuation in lambda
function ve_neural_state(θnet; N = 8, to = Array, linsolve = CPUBackendTag())
    p = ve_slice(ve_neural_conf(0.1, θnet); N)
    s = fv_newton_solver(p; to, linsolve)
    U = to(ve_rest(p))
    for lam in (0.1, 0.25, VE_NN_DE)
        fv_set_conformation!(s, ve_neural_conf(lam, θnet))
        ve_solve!(s, U)
    end
    fv_check_conformation_domain(s.prob, U)
    return s, U
end

# data at θ* = θnet + 0.1 (every network weight, output weights included)
function ve_neural_data(s, U, θnet; to = Array, linsolve = CPUBackendTag())
    sd = fv_newton_solver(KrakeNN._fv_host_problem(s.prob); to, linsolve)
    fv_set_params!(sd, ve_neural_map(), θnet .+ 0.1)
    Ud = copy(U)
    ve_solve!(sd, Ud)
    fv_check_conformation_domain(sd.prob, Ud)
    return Array(Ud)
end
