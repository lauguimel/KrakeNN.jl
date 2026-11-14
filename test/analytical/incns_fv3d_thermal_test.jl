# Increment I3 of the 3D steady FV solver (plan "solveur FV stationnaire 3D",
# step 5): the opt-in energy equation (FVEnergy3D, temperature in slot 5).
# CPU only, Float64, tiny grids:
#   (a) dimensional reduction (V0a): with pins cleared on both sides, Nz = 3
#       periodic, z-uniform data and w = 0, every 3D row (u, v, p, T) equals h
#       times the 2D row and the w row vanishes; without and with viscous
#       heating, source, flow rate, solids, a learned law of the state
#       temperature, a thermal GNF law and first-order upwind;
#   (b) coloured Jacobian == dense ForwardDiff Jacobian on 4x5x6 with energy
#       rows: solids, prescribed-temperature and adiabatic walls, a pin,
#       periodicity, imposed flow rate, Newtonian / learned T-law / thermal GNF;
#   (c) adjoint of thermal objectives with respect to law parameters: law
#       contraction == parameter duals (1e-12), adjoint residual (1e-10),
#       centred full-resolve FD (best of two steps, 2e-7 + 1e-6 |g|);
#   (d) thermal Newton: stopping gates, the Nz = 3 solve equals the 2D solve,
#       branch K (ILU(0) and block SIMPLE) equals branch D, heating
#       continuation smoke test;
#   (e) argument checks of the builder and of FVEnergy3D.
using Test
using LinearAlgebra
using Random
using ForwardDiff
using KrakeNN
using KrakeNN.SparseArrays

const _FV3E_TW = 468.15

_fv3e_clear_pins!(tag) = replace!(tag, FV_CELL_PINNED => FV_CELL_FLUID)
_fv3e_dense_jacobian(p, U) =
    ForwardDiff.jacobian((R, X) -> fv_residual!(R, X, p), zeros(length(U)), U)
_fv3e_rel(a, b) = norm(a - b) / max(norm(a), norm(b), eps(Float64))

# learned law of (shear rate, absolute temperature), as in the 2D T4 test
function _fv3e_learned()
    law = LearnedViscosity(TinyMLP{2,2,1}(; shift = (0.0, _FV3E_TW), scale = (1.0, 2.0));
                           nu_ref = 0.3, monotone = false)
    return law, [0.3, -0.2, 0.4, -0.3, 0.1, -0.15, 0.2, -0.15, 0.04]
end
_fv3e_tlaw() = ((law, theta) = _fv3e_learned();
                FVLawViscosity(law, theta, (FVStateTemperature(),); eps = 0.02))
# thermal GNF: Carreau-Yasuda (a = 2, regular at rest) with an Arrhenius shift;
# temperatures of order 1 so that the shift is strong
_fv3e_tgnf() = FVGNFViscosity(CarreauYasuda(0.3, 0.05, 1.0, 2.0, 0.7;
                                            thermal = ArrheniusCoupling(1.0, 2.0)), 0.0;
                              temperature = FVStateTemperature())

# 2D energy problem and its z-extruded 3D counterpart (Nz = 3, periodic z);
# `walls` uses the 3D keys E, W, N, S.
function _fv3e_pair(Nx, Ny; bc, walls, alpha = 0.1, sigma = 0.0, q = nothing,
                    scheme = :linear_upwind, nu = 0.05, solid = nothing, source = nothing,
                    bvel2 = (s, x, y) -> (0.0, 0.0), viscosity = nothing, u_mean_target = nothing)
    Nz = 3; h = 1 / Ny
    e2 = FVEnergy(; alpha, sigma, scheme, source = q,
                  right = get(walls, :E, :adiabatic), left = get(walls, :W, :adiabatic),
                  top = get(walls, :N, :adiabatic), bottom = get(walls, :S, :adiabatic))
    e3 = FVEnergy3D(; alpha, sigma, scheme, boundary = walls,
                    source = q isa Function ? (x, y, z) -> q(x, y) : q)
    nu3 = nu isa Function ? (x, y, z) -> nu(x, y) : nu
    src3 = source === nothing ? nothing : (x, y, z) -> (s = source(x, y); (s[1], s[2], 0.0, s[3]))
    p2 = fv_steady_problem(Nx, Ny; h, bc, nu, solid, source, viscosity, u_mean_target,
                           boundary_velocity = bvel2, energy = e2)
    p3 = fv_steady_problem(Nx, Ny, Nz; h, nu = nu3, bc = (; bc..., B = :periodic, T = :periodic),
                           solid = solid === nothing ? nothing : repeat(solid, 1, 1, Nz),
                           source = src3, viscosity, u_mean_target,
                           boundary_velocity = (s, x, y, z) -> (bvel2(s, x, y)..., 0.0), energy = e3)
    _fv3e_clear_pins!(p2.tag); _fv3e_clear_pins!(p3.tag)
    return p2, p3
end

# Thermal Newton stops on its scaled 1e-10 gate (atol is not used); two explicit
# corrections resolve the much smaller finite-difference signals.
function _fv3e_solve!(s, U; maxit = 30)
    st = fv_newton!(s, U; maxit)
    @test st.converged
    for _ in 1:2
        fv_newton_finalize!(s, U)
        fv_residual!(s.R, U, s.prob)
        lin_solve!(s.dU, s.cache, s.R)
        U .-= s.dU
    end
    return st
end

# centred full-resolve differences of every objective along `dir`, one column
# per step (steps relative to the parameter scale, as in the I2 test)
function _fv3e_fd(sfd, pmap, theta, U, objectives, dir; steps = (1e-4, 2e-5))
    sc = max(norm(theta, Inf), 1e-3) / norm(dir, Inf)
    fd = zeros(length(objectives), length(steps))
    for (q, e) in enumerate(steps)
        step = e * sc
        J = zeros(length(objectives), 2)
        for (side, sg) in enumerate((-1.0, 1.0))
            fv_set_params!(sfd, pmap, theta .+ (sg * step) .* dir)
            V = copy(U)
            _fv3e_solve!(sfd, V)
            for (j, obj) in enumerate(objectives)
                J[j, side] = fv_objective(obj, V, sfd.prob)
            end
        end
        fd[:, q] .= (J[:, 2] .- J[:, 1]) ./ (2step)
    end
    fv_set_params!(sfd, pmap, theta)
    return fd
end

function _fv3e_adjoint_problem(target)
    N = (6, 5, 4)
    solid = falses(N...)
    solid[3, 2:3, 2] .= true          # off-centre obstacle (no symmetry plane)
    solid[4, 2, 2] = true
    return fv_steady_problem(N...; h = 0.2, solid, u_mean_target = target,
                             bc = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall),
                             source = (x, y, z) -> (1.4, 0.07 * sin(2x + 3y + 5z), 0.0, 0.0),
                             viscosity = _fv3e_tlaw(),
                             energy = FVEnergy3D(; alpha = 0.08, sigma = 0.6, boundary = (S = _FV3E_TW,)))
end

@testset "IncNS steady FV 3D energy (I3, CPU)" begin
    rng = MersenneTwister(2027)
    TW = _FV3E_TW

    @testset "(a) z-uniform 3D residual = h x 2D residual, energy rows" begin
        lid(s, x, y) = s === :N ? (1.0, 0.0) : (0.0, 0.0)
        solid = falses(7, 5); solid[3:4, 1:2] .= true
        nuxy(x, y) = 0.05 * (1 + x * y) + 0.02 * sin(3x)
        box = (W = :wall, E = :wall, S = :wall, N = :wall)
        xper = (W = :periodic, E = :periodic, S = :wall, N = :wall)
        # (name, keywords, mean temperature of the random states)
        cases = (
            ("Newtonian closed box, no heating",
             (; bc = box, nu = nuxy, bvel2 = lid, walls = (S = TW, N = TW + 0.5)), TW + 0.25),
            ("Newtonian x-periodic, heating + source + flow rate + solids",
             (; bc = xper, nu = nuxy, solid, sigma = 0.5, q = (x, y) -> 0.3 + 0.2 * x * y,
              source = (x, y) -> (0.5, 0.1x, 0.0), u_mean_target = 0.2, walls = (S = TW,)), TW),
            ("learned T-law closed box, heating + solids, Dirichlet W/E/S",
             (; bc = box, nu = nuxy, bvel2 = lid, solid, sigma = 0.6, viscosity = _fv3e_tlaw(),
              walls = (W = TW, E = TW + 0.3, S = TW + 0.1)), TW + 0.1),
            ("thermal GNF x-periodic, heating + source, upwind",
             (; bc = xper, nu = 0.3, sigma = 0.3, scheme = :upwind, viscosity = _fv3e_tgnf(),
              source = (x, y) -> (1.0, 0.0, 0.0), q = 0.2, walls = (S = 1.0, N = 1.5)), 1.25),
        )
        slot_pairs = ((1, 1), (2, 2), (4, 3), (5, 4))      # (3D slot, 2D slot): u, v, p, T
        for (name, kw, T0) in cases
            @testset "$name" begin
                p2, p3 = _fv3e_pair(7, 5; kw...)
                g2, g3 = p2.grid, p3.grid
                h = g3.h
                @test !any(==(FV_CELL_PINNED), p3.tag)
                @test fv_nunknowns(p3) == 5 * 7 * 5 * 3 + (p3.flowrate !== nothing)
                @test p3.energy.Tref == p2.energy.Tref
                for trial in 1:2
                    u = randn(rng, 7, 5); v = randn(rng, 7, 5); pr = randn(rng, 7, 5)
                    T = T0 .+ 0.2 .* randn(rng, 7, 5)
                    U2 = fv_state(p2, u, v, pr, T)
                    U3 = fv_state(p3, repeat(u, 1, 1, 3), repeat(v, 1, 1, 3), zeros(7, 5, 3),
                                  repeat(pr, 1, 1, 3), repeat(T, 1, 1, 3))
                    @test fv_fields(p3, U3)[5] ≈ repeat(T, 1, 1, 3) rtol = 1e-15
                    if p2.flowrate !== nothing
                        U2[end] = U3[end] = randn(rng)
                    end
                    R2 = fv_residual(U2, p2); R3 = fv_residual(U3, p3)
                    # one scale per equation, from the fluid rows (solid rows are identities)
                    scales = [h * maximum(abs(R2[fv_index(g2, i, j, b)])
                                          for j in 1:5, i in 1:7 if p2.tag[i, j] != FV_CELL_SOLID)
                              for (_, b) in slot_pairs]
                    worst = zeros(length(slot_pairs)); worst_w = 0.0
                    for k in 1:3, j in 1:5, i in 1:7
                        hs = p3.tag[i, j, k] == FV_CELL_SOLID ? 1.0 : h
                        for (q, (a, b)) in enumerate(slot_pairs)
                            worst[q] = max(worst[q],
                                           abs(R3[fv_index(g3, i, j, k, a)] - hs * R2[fv_index(g2, i, j, b)]))
                        end
                        worst_w = max(worst_w, abs(R3[fv_index(g3, i, j, k, 3)]))
                    end
                    @test all(worst .<= 1e-14 .* scales)
                    @test worst_w <= 1e-14 * max(scales[1], scales[2])
                    @test scales[4] > 0
                    p2.flowrate === nothing || @test abs(R3[end] - R2[end]) <= 1e-14 * max(1, abs(R2[end]))
                    trial == 1 && @info "I3 reduction" name relative = Tuple(worst ./ scales)
                end
            end
        end
    end

    @testset "(b) coloured Jacobian == dense on 4x5x6 with energy rows" begin
        Nx, Ny, Nz = 4, 5, 6
        nu3(x, y, z) = 0.05 * (1 + x * y + 0.5z) + 0.02 * sin(3x + 2z)
        solid = falses(Nx, Ny, Nz); solid[2:3, 1:2, 3] .= true; solid[4, 5, 6] = true
        lidN(s, x, y, z) = s === :N ? (0.3, 0.0, 0.2) : (0.0, 0.0, 0.0)
        lidT(s, x, y, z) = s === :T ? (1.0, 0.4, 0.0) : (0.0, 0.0, 0.0)
        zper = (W = :wall, E = :wall, S = :wall, N = :wall, B = :periodic, T = :periodic)
        xper = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)
        cases = (
            ("Newtonian, periodic z + solids, lid (u, w), heating + source",
             fv_steady_problem(Nx, Ny, Nz; h = 0.2, nu = nu3, bc = zper, solid, boundary_velocity = lidN,
                               energy = FVEnergy3D(; alpha = 0.1, sigma = 0.4, boundary = (S = TW, N = TW + 0.5),
                                                   source = (x, y, z) -> 0.3 + 0.2 * x * z))),
            ("learned T-law, x-periodic, flow rate + solids, Dirichlet B/N",
             fv_steady_problem(Nx, Ny, Nz; h = 0.2, bc = xper, solid, viscosity = _fv3e_tlaw(),
                               source = (x, y, z) -> (0.5, 0.0, 0.0, 0.0), u_mean_target = 0.2,
                               energy = FVEnergy3D(; alpha = 0.08, sigma = 0.6, source = 0.2,
                                                   boundary = (B = TW, N = TW + 0.3)))),
            ("thermal GNF closed box, lid T (pin), upwind, Dirichlet W/E",
             fv_steady_problem(Nx, Ny, Nz; h = 0.2, boundary_velocity = lidT, viscosity = _fv3e_tgnf(),
                               energy = FVEnergy3D(; alpha = 0.1, sigma = 0.3, scheme = :upwind,
                                                   boundary = (W = 1.0, E = 1.5)))),
        )
        for (name, p) in cases
            @testset "$name" begin
                g = p.grid
                @test any(==(FV_CELL_PINNED), p.tag)
                S = fv_jacobian_pattern(p)
                colors, ncol = fv_jacobian_coloring(S, p)
                rows = diff(sparse(transpose(S)).colptr)[1:fv_nunknowns(g)]   # not the dense Fx row
                n = fv_nunknowns(p)
                theta_rows = vec([fv_index(g, i, j, k, 5) for i in 1:Nx, j in 1:Ny, k in 1:Nz])
                s = fv_newton_solver(p)
                for trial in 1:2
                    U = 0.5 .* randn(rng, n)
                    U[theta_rows] .= 0.2 .* randn(rng, length(theta_rows))   # T = T_ref +- 0.2
                    Jd = _fv3e_dense_jacobian(p, U)
                    @test all(S[k] for k in findall(!iszero, Jd))   # pattern is a superset
                    @test any(!iszero, Jd[theta_rows, :])            # energy rows are active
                    fv_jacobian!(s, U)
                    @test Matrix(s.J) == Jd                          # bit for bit
                    asm = fv_jacobian_assembler(S, colors, ncol; chunk = 8, order = :csr)
                    nz = zeros(nnz(S))
                    fv_assemble_jacobian!(nz, asm, U, p)
                    @test nz == nonzeros(sparse(transpose(s.J)))    # cuDSS (CSR) order
                end
                @info "I3 coloured Jacobian" name ncol nnz(S) max_row = maximum(rows) max_energy_row = maximum(rows[theta_rows])
            end
        end
    end

    @testset "(c) thermal adjoint vs duals and centred FD (law parameters)" begin
        _, theta = _fv3e_learned()
        pmap = FVLawParameters(length(theta))
        for target in (nothing, 0.12)
            p = _fv3e_adjoint_problem(target)
            @test KrakeNN._fv_law_problem_slots(p) === Val(7)
            s = fv_newton_solver(p)
            U = fv_state(p, 0.0, 0.0, 0.0, 0.0, TW)
            st = _fv3e_solve!(s, U)
            @test st.gate in (:tolerance, :roundoff_floor)
            @test length(st.thermal_scales.equations) == (target === nothing ? 5 : 6)
            ws = fv_adjoint_workspace(s, pmap)
            objectives = FVObjective[fv_mean_temperature(p; T_wall = TW, target = 0.25),
                                     fv_temperature_l2(p, TW + 0.25; weights = 0.7)]
            target === nothing || push!(objectives, fv_flow_force_objective(p))
            gradients = Vector{Float64}[]
            for obj in objectives
                r = fv_steady_gradient(s, U, theta, obj; params = pmap, ws)
                d = fv_steady_gradient(s, U, theta, obj; params = pmap, ws, law_gradient = :duals,
                                       finalize = false)
                @test all(isfinite, r.grad) && norm(r.grad) > 0
                @test _fv3e_rel(r.grad, d.grad) <= 1e-12
                @test norm(transpose(s.J) * r.λ - r.gU) / max(norm(r.gU), eps()) <= 1e-10
                push!(gradients, r.grad)
            end
            sfd = fv_newton_solver(p)
            drng = MersenneTwister(4018)
            for k in 1:2
                dir = normalize(randn(drng, length(theta)))
                fd = _fv3e_fd(sfd, pmap, theta, U, objectives, dir)
                for (j, gr) in enumerate(gradients)
                    gd = dot(gr, dir)
                    err = minimum(abs.(fd[j, :] .- gd))
                    @test err <= 2e-7 + 1e-6 * abs(gd)
                    @info "I3 thermal adjoint FD" target objective = j direction = k derivative = gd abs_error = err rel_error = err / max(abs(gd), 1e-300)
                end
            end
        end
    end

    @testset "(d) thermal Newton: gates, 3D == 2D, branch K, continuation" begin
        visc = _fv3e_tlaw()
        p2 = fv_steady_problem(6, 5; h = 0.2, bc = (W = :periodic, E = :periodic, S = :wall, N = :wall),
                               source = (x, y) -> (1.0, 0.0, 0.0), viscosity = visc,
                               energy = FVEnergy(; alpha = 0.1, sigma = 0.2, source = 0.3,
                                                 bottom = TW, top = TW + 0.5))
        p3 = fv_steady_problem(6, 5, 3; h = 0.2, viscosity = visc,
                               bc = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :periodic, T = :periodic),
                               source = (x, y, z) -> (1.0, 0.0, 0.0, 0.0),
                               energy = FVEnergy3D(; alpha = 0.1, sigma = 0.2, source = 0.3,
                                                   boundary = (S = TW, N = TW + 0.5)))
        s2 = fv_newton_solver(p2)
        U2 = fv_state(p2, (x, y) -> 0.0, (x, y) -> 0.0, (x, y) -> 0.0, TW)
        _fv3e_solve!(s2, U2)
        s3 = fv_newton_solver(p3)
        U3 = fv_state(p3, 0.0, 0.0, 0.0, 0.0, TW)
        st3 = _fv3e_solve!(s3, U3)
        @test st3.gate in (:tolerance, :roundoff_floor)
        @test last(st3.temperature_increment) <= 1e-10 * st3.thermal_scales.deltaT
        @test length(st3.thermal_scales.equations) == 5
        u2, v2, pr2, T2 = fv_fields(p2, U2)
        u3, v3, w3, pr3, T3 = fv_fields(p3, U3)
        ext(a) = repeat(a, 1, 1, 3)
        @test maximum(abs, u3 - ext(u2)) <= 1e-8 * maximum(abs, u2)
        @test maximum(abs, v3 - ext(v2)) <= 1e-8 * maximum(abs, u2)
        @test maximum(abs, w3) <= 1e-8 * maximum(abs, u2)
        @test maximum(abs, pr3 - ext(pr2)) <= 1e-8 * max(maximum(abs, pr2), 1.0)
        @test maximum(abs, T3 - ext(T2)) <= 1e-8 * maximum(abs, T2 .- TW)
        @test maximum(T2) > TW + 0.5                        # heating and source raise T
        @info "I3 Newton 3D vs 2D" iterations = st3.iterations gate = st3.gate du = maximum(abs, u3 - ext(u2)) dT = maximum(abs, T3 - ext(T2))

        # branch K: temperature joins the velocity block of P2 automatically
        for (kname, tag) in (("ILU(0)", KrylovBackendTag(; precond = :ilu0, rtol = 1e-10, rtol_transpose = 1e-11,
                                                         strict = false)),
                             ("block SIMPLE, Schur LU",
                              KrylovBackendTag(; precond = :simple, schur = :lu, rtol = 1e-10,
                                               rtol_transpose = 1e-11, strict = false)),
                             # default tag: P3L (block upper, LSC, 4 cycles) on a 3D Jacobian with energy
                             ("default (P3L)", KrylovBackendTag(; rtol = 1e-10, rtol_transpose = 1e-11,
                                                                strict = false)))
            sK = fv_newton_solver(p3; linsolve = tag)
            UK = fv_state(p3, 0.0, 0.0, 0.0, 0.0, TW)
            stK = _fv3e_solve!(sK, UK; maxit = 40)
            @test norm(UK - U3, Inf) <= 1e-9 * norm(U3, Inf)
            if kname != "ILU(0)"
                # temperature sits in the velocity block; only pressure forms the Schur part
                blk = lin_krylov_blocks(sK.cache)
                @test sK.J.nfields == 5 && sK.J.pslot == 4
                @test blk.np == fv_ncells(p3) && blk.nu == 4 * fv_ncells(p3)
            end
            if kname == "default (P3L)"
                bA = sK.cache.backend
                @test bA.precond === :mg && bA.block === :upper && bA.schur_approx === :lsc &&
                      bA.mg_cycles == 4
            end
            @info "I3 branch K vs D" kname iterations = stK.iterations gate = stK.gate difference = norm(UK - U3, Inf)
        end

        # heating continuation from the isothermal state (equal walls, no source)
        pc = fv_steady_problem(6, 5, 3; h = 0.2, viscosity = _fv3e_tgnf(),
                               bc = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall),
                               source = (x, y, z) -> (1.0, 0.0, 0.0, 0.0),
                               energy = FVEnergy3D(; alpha = 0.1, sigma = 0.5, boundary = (S = 1.0, N = 1.0)))
        rc = fv_thermal_continuation(pc; initial_step = 0.5)
        @test rc.converged
        @test rc.last_sigma ≈ 0.5
        @test all(in((:tolerance, :roundoff_floor)), rc.gates)
        @test size(rc.w) == (6, 5, 3) && size(rc.T) == (6, 5, 3)
        @test maximum(rc.T) > 1.0
        @info "I3 continuation" sigmas = Tuple(h.sigma for h in rc.history) Tmax = maximum(rc.T) halvings = rc.halvings
    end

    @testset "(e) argument checks" begin
        e3 = FVEnergy3D(; alpha = 0.1, boundary = (S = 1.0,))
        box = (W = :wall, E = :wall, S = :wall, N = :wall, B = :wall, T = :wall)
        xper = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)
        @test_throws ArgumentError FVEnergy3D(; alpha = 0.1, boundary = (bottom = 1.0,))
        @test_throws ArgumentError FVEnergy3D(; alpha = -1.0)
        @test_throws ArgumentError FVEnergy3D(; alpha = 0.1, scheme = :central)
        @test_throws ArgumentError fv_steady_problem(4, 4, 3; energy = FVEnergy(; alpha = 0.1, bottom = 1.0))
        @test_throws ArgumentError fv_steady_problem(4, 4; energy = e3)
        @test_throws ArgumentError fv_steady_problem(4, 4, 3; energy = e3,
            bc = (W = :inlet, E = :outlet, S = :wall, N = :wall, B = :wall, T = :wall))
        @test_throws ArgumentError fv_steady_problem(4, 4, 3; viscosity = _fv3e_tlaw())
        @test_throws ArgumentError fv_steady_problem(4, 4, 3; viscosity = _fv3e_tgnf())
        @test_throws ArgumentError fv_steady_problem(4, 4, 3; bc = xper,
            energy = FVEnergy3D(; alpha = 0.1, boundary = (W = 1.0, S = 1.0)))
        @test_throws ArgumentError fv_steady_problem(4, 4, 3; bc = box, energy = FVEnergy3D(; alpha = 0.1))
        slab = falses(4, 4, 3); slab[2, :, :] .= true     # the E-side pocket has no Dirichlet wall
        @test_throws ArgumentError fv_steady_problem(4, 4, 3; solid = slab,
            energy = FVEnergy3D(; alpha = 0.1, boundary = (W = 1.0,)))
        p0 = fv_steady_problem(4, 4, 3)
        @test p0.energy === nothing && fv_nunknowns(p0) == 4 * 48
        @test_throws ArgumentError fv_state(p0, 0.0, 0.0, 0.0, 0.0, 1.0)
        p1 = fv_steady_problem(4, 4, 3; energy = e3)
        @test fv_nunknowns(p1) == 5 * 48 && p1.energy.Tref == 1.0
        @test_throws DimensionMismatch fv_state(p1, 0.0, 0.0, 0.0, 0.0, zeros(4, 4))
    end
end
