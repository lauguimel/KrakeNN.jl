# Increment I4 of the 3D steady FV solver (branch K): restarted GMRES with an
# ILU(0) preconditioner on the assembled Jacobian (`KrylovBackendTag`), against
# the direct CPU branch (`CPUBackendTag`, UMFPACK). CPU only, Float64, the
# 6 × 5 × 4 grids of the I2 test:
#   (1) one linear solve, forward and transposed, single and multiple right-hand
#       sides, after a refactorization: true residual <= the GMRES tolerance
#       and the same solution as the direct solve; the CSR matrix of branch K
#       equals the CSC matrix of branch D; pivot report; strict
#       non-convergence raises;
#   (2) Newton from rest with branch K converges to the state of branch D;
#   (3) adjoint gradient through branch K == through branch D, on a
#       Newtonian duct (uniform ν, ν field) and on a flow-rate-controlled
#       Carreau duct (law parameters, border = :krylov and :ilu), plus one
#       centred finite-difference check with branch-K Newton re-solves;
#   (4) preconditioner P2 (`precond = :simple`, Schur LU and Schur ILU(0)):
#       the assembled Schur approximation equals C − D diag(A)⁻¹ G built
#       densely, linear solves, Newton state and adjoint gradients equal
#       branch D;
#   (5) preconditioner P3 (`precond = :mg`, block SIMPLE with aggregation
#       multigrid blocks): hierarchy properties on synthetic 3D matrices
#       (P 1 = 1, solid and fluid never aggregated together, coarse operators
#       = Pᵀ A P, symmetric when A is, hierarchy of Aᵀ = transposed hierarchy,
#       Jacobi cycle on Aᵀ = transpose of the cycle on A, convergent cycle on
#       an SPD matrix), then linear solves, Newton state, adjoint gradients
#       and one FD check against branch D;
#   (6) outer block structure and Schur approximations (`block`,
#       `schur_approx`, `schur_q`, `schur_shift`, `ablock = :lu`): assembled S̃
#       and one preconditioner application (forward and transposed side, exact
#       inner solves) against a dense reference, linear solves with exact and
#       with ILU / multigrid inner solves, Newton state and adjoint gradients
#       against branch D;
#   (7) preconditioner P4 (`precond = :vanka`, monolithic multigrid with an
#       additive Vanka smoother): hierarchy properties on the FV Jacobians
#       (sum restriction, trilinear or constant prolongation with P 1 = 1,
#       coarse operators = R A P, one patch per fluid pressure unknown, every
#       fluid unknown covered, stored patch inverses exact, fluid pressure
#       diagonals positive on every level, same checks on the hierarchy of
#       Jᵀ), linear solves, Newton state, adjoint gradients and one FD check
#       against branch D, stationary residual ratios of one cycle (logged).
using Test
using LinearAlgebra
using Random
using KrakeNN

const _I4_NEWTON = (; atol = 1e-13, rtol = 1e-15, maxit = 60)
# Newton / adjoint tag: tight tolerances so that K and D agree to ~1e-9;
# strict = false so a round-off-limited last Newton step does not raise.
const _I4_TAG = KrylovBackendTag(; rtol = 1e-10, rtol_transpose = 1e-11, strict = false)

_i4_bc = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall)
function _i4_solid(N)
    solid = falses(N...)
    solid[3, 2:3, 2] .= true
    solid[4, 2, 2] = true
    return solid
end
_i4_lid = (s, x, y, z) -> s === :T ? (0.2, 0.1, 0.0) : (0.0, 0.0, 0.0)

_i4_duct() = fv_steady_problem(6, 5, 4; h = 0.2, nu = (x, y, z) -> 0.05 * (1 + 0.3x - 0.2y + 0.1z),
                               bc = _i4_bc, solid = _i4_solid((6, 5, 4)),
                               source = (x, y, z) -> (1.0, 0.0, 0.0, 0.0), boundary_velocity = _i4_lid)

_i4_carreau() = fv_steady_problem(6, 5, 4; h = 0.2, nu = 0.05, bc = _i4_bc, solid = _i4_solid((6, 5, 4)),
                                  source = (x, y, z) -> (0.0, 0.0, 0.0, 0.0), u_mean_target = 0.4,
                                  viscosity = FVLawViscosity(ClassicalLaw(CarreauYasuda(0.05, 0.01, 2.0, 2.0, 0.5)),
                                                             [0.05, 0.01, 2.0, 2.0, 0.5], (), 0.0))

_i4_U0(p) = KrakeNN._fv_initial_force!(zeros(fv_nunknowns(p)), p)

function _i4_solve!(s, U)
    st = fv_newton!(s, U; _I4_NEWTON...)
    @test st.converged || st.stagnated
    @test last(st.inf) < 1e-11
    return U
end

_i4_restart!(s, base) = (copyto!(s.prob.nu, base.nu); copyto!(s.prob.bvel, base.bvel);
                         copyto!(s.prob.src, base.src); s)

# ------------------------------------------------------------------ (1) linear solves
@testset "Krylov (GMRES + ILU(0)) solve == direct solve" begin
    rng = MersenneTwister(41)
    for (pname, p) in (("duct", _i4_duct()), ("Carreau + flow rate", _i4_carreau()))
        sD = fv_newton_solver(p)
        U = _i4_solve!(sD, _i4_U0(p))
        U2 = U .+ 1e-3 .* randn(rng, length(U))          # a second state for the refactorization
        tag = KrylovBackendTag(; rtol = 1e-10, rtol_transpose = 1e-10)
        sK = fv_newton_solver(p; linsolve = tag)
        @test sK.J isa KrylovCSRMatrix
        @test sK.J.border == (p.flowrate === nothing ? 0 : KrakeNN._fv_force_index(p))
        cD = cK = nothing                                 # caches kept across the two states
        for (k, Uk) in enumerate((U, U2))
            fv_jacobian!(sD, Uk); fv_jacobian!(sK, Uk)
            J = sD.J
            if k == 1
                cD = lin_factorize(J; backend = CPUBackendTag(), spd = false)
                cK = lin_factorize(sK.J; backend = tag)
            else
                lin_refactorize!(cD); lin_refactorize!(cK)       # values already in J / sK.J
            end
            # same matrix in both storages
            x = randn(rng, size(J, 1))
            w = similar(x); KrakeNN._krylov_matvec!(w, cK.factor.fwd, x)
            @test norm(w - J * x) <= 1e-13 * norm(J * x)
            @test lin_supports_transpose(cK)
            piv = lin_krylov_pivot(cK)
            @test piv.min_ratio > 0
            b = randn(rng, size(J, 1))
            for tr in (false, true)
                Jop = tr ? transpose(J) : J
                xD = lin_solve!(cD, b; transpose = tr)
                xK = lin_solve!(cK, b; transpose = tr)
                @test xK isa Vector{Float64}
                @test norm(Jop * xK - b) <= 1e-10 * norm(b) * (1 + 1e-6)
                @test norm(xK - xD) <= 1e-5 * norm(xD)
                e = last(lin_krylov_log(cK))
                @test e.transpose == tr && e.converged && e.iters > 0
                # in place and multiple right-hand sides
                y = zeros(length(b)); lin_solve!(y, cK, b; transpose = tr)
                @test norm(Jop * y - b) <= 1e-10 * norm(b) * (1 + 1e-6)
                B = hcat(b, randn(rng, length(b)))
                X = lin_solve!(cK, B; transpose = tr)
                @test all(norm(Jop * X[:, c] - B[:, c]) <= 1e-10 * norm(B[:, c]) * (1 + 1e-6) for c in 1:2)
            end
            @info "I4 Krylov solve ($pname, state $k)" iters_forward = lin_krylov_log(cK)[end-7].iters iters_transpose = lin_krylov_log(cK)[end].iters min_pivot_ratio = piv.min_ratio small_pivots = piv.n_small
        end
        # strict non-convergence raises; pin_k0 is refused
        cS = lin_factorize(sK.J; backend = KrylovBackendTag(; rtol = 1e-15, maxiter = 1))
        @test_throws ErrorException lin_solve!(cS, randn(rng, size(sD.J, 1)))
        @test_throws ArgumentError lin_factorize(sK.J; backend = tag, pin_k0 = 1)
    end
end

# ------------------------------------------------------------------ (2) Newton
@testset "Newton with branch K converges to the branch-D state" begin
    for (pname, p) in (("duct", _i4_duct()), ("Carreau + flow rate", _i4_carreau()))
        UD = _i4_solve!(fv_newton_solver(p), _i4_U0(p))
        sK = fv_newton_solver(p; linsolve = _I4_TAG)
        UK = _i4_solve!(sK, _i4_U0(p))
        @test norm(UK - UD, Inf) <= 1e-9 * norm(UD, Inf)
        # default tag (rtol = 1e-6: inexact Newton) reaches the same state
        s6 = fv_newton_solver(p; linsolve = KrylovBackendTag())
        U6 = _i4_solve!(s6, _i4_U0(p))
        @test norm(U6 - UD, Inf) <= 1e-9 * norm(UD, Inf)
        @info "I4 Newton K vs D ($pname)" rel_tight = norm(UK - UD, Inf) / norm(UD, Inf) rel_default = norm(U6 - UD, Inf) / norm(UD, Inf) gmres_iters = [e.iters for e in lin_krylov_log(s6.cache)]
    end
end

# ------------------------------------------------------------------ (3) adjoint
function _i4_grad_pair(base, map, θ, objs; tag = _I4_TAG)
    sD = _i4_restart!(fv_newton_solver(base), base); fv_set_params!(sD, map, θ)
    sK = _i4_restart!(fv_newton_solver(base; linsolve = tag), base); fv_set_params!(sK, map, θ)
    UD = _i4_solve!(sD, _i4_U0(base))
    UK = _i4_solve!(sK, _i4_U0(base))
    out = []
    for (oname, obj) in objs
        rD = fv_steady_gradient(sD, UD, θ, obj; params = map)
        rK = fv_steady_gradient(sK, UK, θ, obj; params = map)
        e = norm(rK.grad - rD.grad) / max(norm(rD.grad), eps())
        @test e <= 1e-8
        @test rK.residual_inf <= 1e-11
        tr = filter(x -> x.transpose, lin_krylov_log(sK.cache))
        @test !isempty(tr) && last(tr).converged
        push!(out, (oname, e, rK, sK, UK))
    end
    return out
end

@testset "3D adjoint gradient: branch K == branch D, FD check" begin
    rng = MersenneTwister(43)
    base = _i4_duct()
    nc = fv_ncells(base)
    objs = (("flow rate", fv_flow_rate(base, 2)), ("wall force", FVWallForce(1; region = :solid)))
    for (mname, map, θ) in (("uniform nu", FVUniformViscosity(), [0.04]),
                            ("nu field", FVViscosityField(), vec(copy(base.nu))))
        res = _i4_grad_pair(base, map, θ, objs)
        @info "I4 adjoint K vs D (duct, $mname)" rel = [r[2] for r in res]
    end
    # Carreau law parameters on the flow-rate-controlled duct, both border treatments
    cb = _i4_carreau()
    θcy = [0.05, 0.01, 2.0, 2.0, 0.5]
    cobjs = (("Fx", fv_flow_force_objective(cb)), ("wall force", FVWallForce(1; region = :solid)))
    for border in (:krylov, :ilu)
        tag = KrylovBackendTag(; rtol = 1e-10, rtol_transpose = 1e-11, strict = false, border)
        res = _i4_grad_pair(cb, FVLawParameters(5), θcy, cobjs; tag)
        @info "I4 adjoint K vs D (Carreau + flow rate, border = $border)" rel = [r[2] for r in res]
    end
    # one centred FD check through branch-K Newton re-solves (uniform nu, flow rate)
    map = FVUniformViscosity(); θ = [0.04]; obj = fv_flow_rate(base, 2)
    (_, _, rK, _, UK) = first(_i4_grad_pair(base, map, θ, (("flow rate", obj),)))
    sfd = fv_newton_solver(base; linsolve = _I4_TAG)
    J(t) = (_i4_restart!(sfd, base); fv_set_params!(sfd, map, [t]);
            fv_objective(obj, _i4_solve!(sfd, copy(UK)), sfd.prob))
    gd = rK.grad[1]
    err = minimum(abs((J(θ[1] + ε) - J(θ[1] - ε)) / (2ε) - gd) for ε in (4e-6, 8e-7))
    @test err <= 2e-7 + 1e-6 * abs(gd)
    @info "I4 adjoint via K vs centred FD (uniform nu, flow rate)" gd err
end

# ------------------------------------------------------------------ (4) P2 = block SIMPLE
# Dense reference of the SIMPLE Schur approximation of the matrix J.
function _i4_dense_schur(J::AbstractMatrix, nf::Int, ps::Int, nloc::Int)
    U = [i for i in 1:nloc if (i - 1) % nf + 1 != ps]
    P = [i for i in 1:nloc if (i - 1) % nf + 1 == ps]
    A = Matrix(J[U, U]); G = Matrix(J[U, P]); D = Matrix(J[P, U]); C = Matrix(J[P, P])
    return C - D * (G ./ diag(A))
end

@testset "Krylov P2 (block SIMPLE) == direct: Schur matrix, solves, Newton, gradient" begin
    rng = MersenneTwister(47)
    for (pname, p) in (("duct", _i4_duct()), ("Carreau + flow rate", _i4_carreau()))
        sD = fv_newton_solver(p)
        UD = _i4_solve!(sD, _i4_U0(p))
        for schur in (:lu, :ilu0)
            tag = KrylovBackendTag(; precond = :simple, schur, rtol = 1e-10, rtol_transpose = 1e-10)
            sK = fv_newton_solver(p; linsolve = tag)
            @test sK.J.nfields == 4 && sK.J.pslot == 4 && sK.J.nlocal == 4 * fv_ncells(p)
            fv_jacobian!(sD, UD); fv_jacobian!(sK, UD)
            J = sD.J
            cD = lin_factorize(J; backend = CPUBackendTag(), spd = false)
            cK = lin_factorize(sK.J; backend = tag)
            blk = lin_krylov_blocks(cK)
            @test blk.np == fv_ncells(p) && blk.nu == 3 * fv_ncells(p) && blk.schur === schur
            @test lin_krylov_pivot(cK).min_ratio > 0
            # Schur approximation assembled on the device == dense C − D diag(A)⁻¹ G
            sm = cK.factor.fwd.simple
            Sk = zeros(sm.np, sm.np)
            for i in 1:sm.np, q in sm.Srp[i]:(sm.Srp[i+1]-1)
                Sk[i, sm.Scv[q]] = sm.Sval[q]
            end
            Sref = _i4_dense_schur(J, 4, 4, sK.J.nlocal)
            @test norm(Sk - Sref) <= 1e-12 * norm(Sref)
            b = randn(rng, size(J, 1))
            for tr in (false, true)
                Jop = tr ? transpose(J) : J
                xD = lin_solve!(cD, b; transpose = tr)
                xK = lin_solve!(cK, b; transpose = tr)
                @test norm(Jop * xK - b) <= 1e-10 * norm(b) * (1 + 1e-6)
                @test norm(xK - xD) <= 1e-5 * norm(xD)
                @test last(lin_krylov_log(cK)).converged
            end
            # second state: refactorization of both blocks and of the Schur matrix
            U2 = UD .+ 1e-3 .* randn(rng, length(UD))
            fv_jacobian!(sD, U2); fv_jacobian!(sK, U2)
            lin_refactorize!(cD); lin_refactorize!(cK)
            Sref2 = _i4_dense_schur(sD.J, 4, 4, sK.J.nlocal)
            Sk2 = zeros(sm.np, sm.np)
            for i in 1:sm.np, q in sm.Srp[i]:(sm.Srp[i+1]-1)
                Sk2[i, sm.Scv[q]] = sm.Sval[q]
            end
            @test norm(Sk2 - Sref2) <= 1e-12 * norm(Sref2)
            for tr in (false, true)
                xK = lin_solve!(cK, b; transpose = tr)
                @test norm((tr ? transpose(sD.J) : sD.J) * xK - b) <= 1e-10 * norm(b) * (1 + 1e-6)
            end
            log = lin_krylov_log(cK)
            @info "I4 P2 solve ($pname, schur = $schur)" iters = [e.iters for e in log] transposed = [e.transpose for e in log]
        end
        # Newton with P2 reaches the branch-D state
        sK = fv_newton_solver(p; linsolve = KrylovBackendTag(; precond = :simple, rtol = 1e-10,
                                                                rtol_transpose = 1e-11, strict = false))
        UK = _i4_solve!(sK, _i4_U0(p))
        @test norm(UK - UD, Inf) <= 1e-9 * norm(UD, Inf)
        @info "I4 Newton P2 vs D ($pname)" rel = norm(UK - UD, Inf) / norm(UD, Inf) gmres_iters = [e.iters for e in lin_krylov_log(sK.cache)]
    end
    # P2 needs the block layout of the matrix
    p = _i4_duct()
    s0 = fv_newton_solver(p; linsolve = KrylovBackendTag())
    fv_jacobian!(s0, _i4_U0(p))
    bare = KrylovCSRMatrix(s0.J.n, s0.J.rowptr, s0.J.colval, s0.J.nzval, 0)
    @test_throws ArgumentError lin_factorize(bare; backend = KrylovBackendTag(; precond = :simple))
    @test_throws ArgumentError lin_factorize(s0.J; backend = KrylovBackendTag(; precond = :bogus))
    # adjoint gradients through P2 == through branch D
    tagP2 = KrylovBackendTag(; precond = :simple, rtol = 1e-10, rtol_transpose = 1e-11, strict = false)
    base = _i4_duct()
    res = _i4_grad_pair(base, FVUniformViscosity(), [0.04],
                        (("flow rate", fv_flow_rate(base, 2)), ("wall force", FVWallForce(1; region = :solid)));
                        tag = tagP2)
    @info "I4 adjoint P2 vs D (duct, uniform nu)" rel = [r[2] for r in res]
    cb = _i4_carreau()
    res = _i4_grad_pair(cb, FVLawParameters(5), [0.05, 0.01, 2.0, 2.0, 0.5],
                        (("Fx", fv_flow_force_objective(cb)), ("wall force", FVWallForce(1; region = :solid)));
                        tag = tagP2)
    @info "I4 adjoint P2 vs D (Carreau + flow rate)" rel = [r[2] for r in res]
end

# ------------------------------------------------------------------ (5) P3 = block SIMPLE + multigrid
# Synthetic 3D operator on a cell grid: 7-point Laplacian (Dirichlet outer and
# solid walls), identity rows in solid cells, optional first-order upwind
# convection in +x (non-symmetric).
function _i4_lap3(dims, solid; conv = 0.0)
    Nx, Ny, Nz = dims
    id(i, j, k) = i + (j - 1) * Nx + (k - 1) * Nx * Ny
    rows = Int[]; cols = Int[]; vals = Float64[]
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        c = id(i, j, k)
        if solid[c]
            push!(rows, c); push!(cols, c); push!(vals, 1.0)
            continue
        end
        for (di, dj, dk) in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
            ii, jj, kk = i + di, j + dj, k + dk
            (1 <= ii <= Nx && 1 <= jj <= Ny && 1 <= kk <= Nz) || continue
            nb = id(ii, jj, kk)
            solid[nb] && continue
            push!(rows, c); push!(cols, nb); push!(vals, di == -1 ? -1.0 - conv : -1.0)
        end
        push!(rows, c); push!(cols, c); push!(vals, 6.0 + conv)
    end
    return KrakeNN.sparse(rows, cols, vals, prod(dims), prod(dims))
end

# CSR triple (Int32, sorted columns) of a sparse matrix
function _i4_csr(B)
    T = KrakeNN.sparse(transpose(B))
    return Vector{Int32}(T.colptr), Vector{Int32}(KrakeNN.rowvals(T)), Vector{Float64}(KrakeNN.nonzeros(T))
end

function _i4_mg(B, geom, tag)
    rp, cv, v = _i4_csr(B)
    n = size(B, 1)
    nodes = KrakeNN._mg_nodes(Int32.(1:n), fill(Int32(1), n), geom)
    mg = KrakeNN._mg_build(rp, cv, v, nodes, tag)
    KrakeNN._mg_refresh!(mg, tag)
    return mg
end

# dense matrix of the (linear) multigrid operator b -> x
function _i4_mg_dense(mg, n)
    M = zeros(n, n); e = zeros(n); x = zeros(n)
    for c in 1:n
        fill!(e, 0.0); e[c] = 1.0
        KrakeNN._mg_apply!(x, mg, e)
        M[:, c] .= x
    end
    return M
end

const _I4_P3 = KrylovBackendTag(; precond = :mg, mg_coarse = 16, rtol = 1e-10, rtol_transpose = 1e-11,
                                strict = false)

@testset "Krylov P3 (block SIMPLE + multigrid): hierarchy, solves, Newton, gradient" begin
    rng = MersenneTwister(53)
    dims = (6, 5, 4)
    solid = vec(_i4_solid(dims))
    geom = (; dims, solid)
    n = prod(dims)
    tag = KrylovBackendTag(; precond = :mg, mg_coarse = 4)
    # (a) hierarchy of a symmetric matrix
    B = _i4_lap3(dims, solid)
    mg = _i4_mg(B, geom, tag)
    nl = length(mg.levels)
    @test nl >= 3
    @test mg.info.sizes[1] == n && issorted(mg.info.sizes; rev = true)
    for l in 1:nl-1
        P = KrakeNN._mg_prolongation(mg, l)
        @test P * ones(size(P, 2)) == ones(size(P, 1))                              # constant preserved
        Al = KrakeNN._mg_level_matrix(mg, l)
        Ac = Matrix(KrakeNN._mg_level_matrix(mg, l + 1))
        @test norm(Ac - Matrix(P' * Al * P)) <= 1e-13 * norm(Ac)                   # Galerkin RAP
        @test norm(Ac - Ac') <= 1e-14 * norm(Ac)                                   # symmetric
    end
    P1 = KrakeNN._mg_prolongation(mg, 1)
    for ag in axes(P1, 2)
        members = findall(!iszero, P1[:, ag])
        @test all(==(solid[members[1]]), solid[members])                           # solid / fluid apart
    end
    # convergent cycle on an SPD matrix (W-cycle, l1-Jacobi, exact coarsest)
    M = _i4_mg_dense(mg, n)
    rho = maximum(abs, eigvals(Matrix(1.0I, n, n) - M * Matrix(B)))
    @test rho < 1
    @info "I4 P3 hierarchy (SPD Laplacian)" sizes = mg.info.sizes nnz = mg.info.nnz rho
    # (b) non-symmetric matrix: hierarchy of Aᵀ = transposed hierarchy; with a
    # Jacobi smoother the cycle on Aᵀ is the transpose of the cycle on A
    # (l1-Jacobi uses row sums, which differ for Aᵀ: hierarchy check only)
    Bn = _i4_lap3(dims, solid; conv = 2.0)
    for (sm, cyc) in ((:jacobi, :W), (:jacobi, :V), (:l1jacobi, :W))
        tj = KrylovBackendTag(; precond = :mg, mg_coarse = 4, mg_smoother = sm, mg_cycle = cyc)
        mgf = _i4_mg(Bn, geom, tj)
        mgt = _i4_mg(KrakeNN.sparse(transpose(Bn)), geom, tj)
        @test length(mgt.levels) == length(mgf.levels)
        for l in 2:length(mgf.levels)
            Af = Matrix(KrakeNN._mg_level_matrix(mgf, l)); At = Matrix(KrakeNN._mg_level_matrix(mgt, l))
            @test norm(At - Af') <= 1e-13 * norm(Af)
        end
        if sm === :jacobi
            Mf = _i4_mg_dense(mgf, n); Mt = _i4_mg_dense(mgt, n)
            @test norm(Mt - Mf') <= 1e-11 * norm(Mf)
        end
        @info "I4 P3 hierarchy (non-symmetric, smoother $sm, cycle $cyc)" sizes = mgf.info.sizes opc = mgf.info.operator_complexity
    end
    # (c) solves on the FV Jacobians
    for (pname, p) in (("duct", _i4_duct()), ("Carreau + flow rate", _i4_carreau()))
        sD = fv_newton_solver(p)
        UD = _i4_solve!(sD, _i4_U0(p))
        for (label, tg) in (("P3", KrylovBackendTag(; precond = :mg, mg_coarse = 16, rtol = 1e-10,
                                                    rtol_transpose = 1e-10)),
                            ("P3 Schur LU", KrylovBackendTag(; precond = :mg, schur = :lu, mg_coarse = 16,
                                                             rtol = 1e-10, rtol_transpose = 1e-10)),
                            ("P2 Schur MG", KrylovBackendTag(; precond = :simple, schur = :mg, mg_coarse = 16,
                                                             rtol = 1e-10, rtol_transpose = 1e-10)),
                            ("P3 V ILU(0)", KrylovBackendTag(; precond = :mg, mg_coarse = 16, mg_cycle = :V,
                                                             mg_smoother = :ilu0, rtol = 1e-10,
                                                             rtol_transpose = 1e-10)))
            sK = fv_newton_solver(p; linsolve = tg)
            @test sK.J.geom !== nothing && sK.J.geom.dims == (6, 5, 4)
            fv_jacobian!(sD, UD); fv_jacobian!(sK, UD)
            cD = lin_factorize(sD.J; backend = CPUBackendTag(), spd = false)
            cK = lin_factorize(sK.J; backend = tg)
            blk = lin_krylov_blocks(cK)
            @test blk.schur === KrakeNN._krylov_schur_kind(tg)
            tg.precond === :mg && @test blk.mg_A.levels >= 2
            blk.schur === :mg && @test blk.mg_S.levels >= 2
            @test lin_krylov_pivot(cK).min_ratio > 0
            b = randn(rng, size(sD.J, 1))
            for state in 1:2
                if state == 2                                   # refactorization: new values everywhere
                    U2 = UD .+ 1e-3 .* randn(rng, length(UD))
                    fv_jacobian!(sD, U2); fv_jacobian!(sK, U2)
                    lin_refactorize!(cD); lin_refactorize!(cK)
                end
                for tr in (false, true)
                    Jop = tr ? transpose(sD.J) : sD.J
                    xD = lin_solve!(cD, b; transpose = tr)
                    xK = lin_solve!(cK, b; transpose = tr)
                    @test norm(Jop * xK - b) <= 1e-10 * norm(b) * (1 + 1e-6)
                    @test norm(xK - xD) <= 1e-5 * norm(xD)
                    @test last(lin_krylov_log(cK)).converged
                end
            end
            @info "I4 $label solve ($pname)" iters = [e.iters for e in lin_krylov_log(cK)] mg_A = (blk.mg_A === nothing ? nothing : blk.mg_A.sizes) mg_S = (blk.mg_S === nothing ? nothing : blk.mg_S.sizes)
        end
        # Newton with P3 reaches the branch-D state
        sK = fv_newton_solver(p; linsolve = _I4_P3)
        UK = _i4_solve!(sK, _i4_U0(p))
        @test norm(UK - UD, Inf) <= 1e-9 * norm(UD, Inf)
        @info "I4 Newton P3 vs D ($pname)" rel = norm(UK - UD, Inf) / norm(UD, Inf) gmres_iters = [e.iters for e in lin_krylov_log(sK.cache)]
    end
    # P3 needs the geometry; option checks
    p = _i4_duct()
    s0 = fv_newton_solver(p; linsolve = KrylovBackendTag())
    fv_jacobian!(s0, _i4_U0(p))
    nogeom = KrylovCSRMatrix(s0.J.n, s0.J.rowptr, s0.J.colval, s0.J.nzval, s0.J.border, s0.J.nfields,
                             s0.J.pslot, s0.J.nlocal)
    @test_throws ArgumentError lin_factorize(nogeom; backend = KrylovBackendTag(; precond = :mg))
    @test_throws ArgumentError lin_factorize(s0.J; backend = KrylovBackendTag(; precond = :mg, mg_cycle = :K))
    @test_throws ArgumentError lin_factorize(s0.J; backend = KrylovBackendTag(; precond = :mg, mg_smoother = :gs))
    # adjoint gradients through P3 == through branch D
    base = _i4_duct()
    res = _i4_grad_pair(base, FVUniformViscosity(), [0.04],
                        (("flow rate", fv_flow_rate(base, 2)), ("wall force", FVWallForce(1; region = :solid)));
                        tag = _I4_P3)
    @info "I4 adjoint P3 vs D (duct, uniform nu)" rel = [r[2] for r in res]
    cb = _i4_carreau()
    res = _i4_grad_pair(cb, FVLawParameters(5), [0.05, 0.01, 2.0, 2.0, 0.5],
                        (("Fx", fv_flow_force_objective(cb)), ("wall force", FVWallForce(1; region = :solid)));
                        tag = _I4_P3)
    @info "I4 adjoint P3 vs D (Carreau + flow rate)" rel = [r[2] for r in res]
    # one centred FD check through P3 Newton re-solves (uniform nu, flow rate)
    map = FVUniformViscosity(); θ = [0.04]; obj = fv_flow_rate(base, 2)
    (_, _, rK, _, UK) = first(_i4_grad_pair(base, map, θ, (("flow rate", obj),); tag = _I4_P3))
    sfd = fv_newton_solver(base; linsolve = _I4_P3)
    Jfd(t) = (_i4_restart!(sfd, base); fv_set_params!(sfd, map, [t]);
              fv_objective(obj, _i4_solve!(sfd, copy(UK)), sfd.prob))
    gd = rK.grad[1]
    err = minimum(abs((Jfd(θ[1] + ε) - Jfd(θ[1] - ε)) / (2ε) - gd) for ε in (4e-6, 8e-7))
    @test err <= 2e-7 + 1e-6 * abs(gd)
    @info "I4 adjoint via P3 vs centred FD (uniform nu, flow rate)" gd err
end

# ------------------------------------------------------------------ (6) block structure + Schur approximations
# Dense blocks of J on the interleaved layout (nf fields, pressure slot ps).
function _i4_dense_blocks(J, nf, ps, nloc)
    U = [i for i in 1:nloc if (i - 1) % nf + 1 != ps]
    P = [i for i in 1:nloc if (i - 1) % nf + 1 == ps]
    return (; U, P, A = Matrix(J[U, U]), G = Matrix(J[U, P]), D = Matrix(J[P, U]), C = Matrix(J[P, P]))
end

_i4_dense_q(A, qkind, qscale = 0.0) = qkind === :rowsum ? sign.(diag(A)) .* vec(sum(abs.(A); dims = 2)) :
                                      qkind === :scalar ? fill(qscale, size(A, 1)) : diag(A)

# S̃ = C − D Q⁻¹ G − σ diag(D Q⁻¹ G)
function _i4_dense_shat(B, q, shift)
    DQG = B.D * (B.G ./ q)
    return B.C - DQG - shift * Diagonal(diag(DQG))
end

# F_p = mean of the diagonal velocity-component blocks (cells = pressure unknowns)
function _i4_dense_fp(J, nf, nvel, ncell)
    F = zeros(ncell, ncell)
    for m in 1:nvel
        idx = [(c - 1) * nf + m for c in 1:ncell]
        F .+= Matrix(J[idx, idx])
    end
    return F ./ nvel
end

# One application z = M⁻¹ v with exact inner solves (dense reference)
function _i4_dense_apply(J, v, nf, ps, nloc, tag)
    B = _i4_dense_blocks(J, nf, ps, nloc)
    q = _i4_dense_q(B.A, KrakeNN._bs_qkind(tag), tag.schur_qscale)
    Sh = _i4_dense_shat(B, q, tag.schur_shift)
    F = _i4_dense_fp(J, nf, ps - 1, length(B.P))
    function sinv(r)
        if tag.schur_approx === :lsc
            y = Sh \ r
            return Sh \ (tag.lsc_alpha * (B.C * y) - B.D * ((B.A * ((B.G * y) ./ q)) ./ q))
        elseif tag.schur_approx === :pcd
            return F * (Sh \ (r ./ diag(F)))
        end
        return Sh \ r
    end
    vU = v[B.U]; vP = v[B.P]
    if tag.block === :upper
        p = sinv(vP); u = B.A \ (vU - B.G * p)
    else
        uh = B.A \ vU; p = sinv(vP - B.D * uh)
        u = tag.block === :simple ? uh - (B.G * p) ./ q : uh
    end
    z = copy(v); z[B.U] = u; z[B.P] = p
    return z, Sh
end

_i4_bs_tag(; kw...) = KrylovBackendTag(; rtol = 1e-10, rtol_transpose = 1e-10, kw...)

@testset "Krylov block structure + Schur approximations == dense reference / direct" begin
    rng = MersenneTwister(59)
    exact_cfgs = ((:upper, :simple, :auto, 0.0, 1.0), (:lower, :simplec, :auto, 0.0, 1.0),
                  (:simple, :simple, :auto, 2.0, 1.0), (:simple, :simple, :rowsum, 0.0, 1.0),
                  (:upper, :lsc, :auto, 0.0, 1.0), (:simple, :lsc, :auto, 0.0, 0.5),
                  (:upper, :pcd, :auto, 0.0, 1.0), (:lower, :pcd, :auto, 1.0, 1.0),
                  (:upper, :lsc, :scalar, 0.0, 1.0))
    inexact_tags = (("P3 upper LSC", _i4_bs_tag(; precond = :mg, mg_coarse = 16, block = :upper, schur_approx = :lsc)),
                    ("P3 upper PCD", _i4_bs_tag(; precond = :mg, mg_coarse = 16, block = :upper, schur_approx = :pcd)),
                    ("P3 SIMPLEC shift 4", _i4_bs_tag(; precond = :mg, mg_coarse = 16, schur_approx = :simplec,
                                                      schur_shift = 4.0)),
                    ("P2 lower PCD", _i4_bs_tag(; precond = :simple, block = :lower, schur_approx = :pcd)),
                    ("P2 exact A, Schur MG", _i4_bs_tag(; precond = :simple, ablock = :lu, schur = :mg,
                                                        mg_coarse = 16, block = :upper)))
    for (pname, p) in (("duct", _i4_duct()), ("Carreau + flow rate", _i4_carreau()))
        sD = fv_newton_solver(p)
        UD = _i4_solve!(sD, _i4_U0(p))
        nloc = 4 * fv_ncells(p)
        # (a) exact inner solves: S̃ and M⁻¹ v against the dense reference
        for (block, approx, qk, shift, alpha) in exact_cfgs
            tag = _i4_bs_tag(; precond = :simple, ablock = :lu, schur = :lu, block, schur_approx = approx,
                             schur_q = qk, schur_shift = shift, lsc_alpha = alpha,
                             schur_qscale = qk === :scalar ? 0.2^3 : 0.0)   # cell volume (h = 0.2)
            sK = fv_newton_solver(p; linsolve = tag)
            fv_jacobian!(sD, UD); fv_jacobian!(sK, UD)
            cD = lin_factorize(sD.J; backend = CPUBackendTag(), spd = false)
            cK = lin_factorize(sK.J; backend = tag)
            blk = lin_krylov_blocks(cK)
            @test blk.ablock === :lu && blk.block === block && blk.schur_approx === approx
            @test lin_krylov_pivot(cK).min_ratio > 0
            sm = cK.factor.fwd.simple
            @test sm.bs !== nothing && sm.Ablk isa KrakeNN._KrylovALU
            v = randn(rng, size(sD.J, 1))
            zref, Sh = _i4_dense_apply(sD.J, v, 4, 4, nloc, tag)
            Sk = zeros(sm.np, sm.np)
            for i in 1:sm.np, q in sm.Srp[i]:(sm.Srp[i+1]-1)
                Sk[i, sm.Scv[q]] = sm.Sval[q]
            end
            @test norm(Sk - Sh) <= 1e-12 * norm(Sh)
            z = similar(v); KrakeNN._simple_apply!(z, sm, v)
            @test norm(z - zref) <= 1e-9 * norm(zref)
            # transposed side: the same construction on the blocks of Jᵀ
            tr = KrakeNN._krylov_tside!(cK.factor)
            JT = KrakeNN.sparse(transpose(sD.J))
            ztref, _ = _i4_dense_apply(JT, v, 4, 4, nloc, tag)
            KrakeNN._simple_apply!(z, tr.simple, v)
            @test norm(z - ztref) <= 1e-9 * norm(ztref)
            b = randn(rng, size(sD.J, 1))
            for trp in (false, true)
                Jop = trp ? transpose(sD.J) : sD.J
                xD = lin_solve!(cD, b; transpose = trp)
                xK = lin_solve!(cK, b; transpose = trp)
                @test norm(Jop * xK - b) <= 1e-10 * norm(b) * (1 + 1e-6)
                @test norm(xK - xD) <= 1e-5 * norm(xD)
                @test last(lin_krylov_log(cK)).converged
            end
            @info "I4 block $block / $approx (q $qk, σ $shift, α $alpha), exact inner ($pname)" iters = [e.iters for e in lin_krylov_log(cK)]
        end
        # (b) ILU / multigrid inner solves, two states (refactorization)
        for (label, tag) in inexact_tags
            sK = fv_newton_solver(p; linsolve = tag)
            fv_jacobian!(sD, UD); fv_jacobian!(sK, UD)
            cD = lin_factorize(sD.J; backend = CPUBackendTag(), spd = false)
            cK = lin_factorize(sK.J; backend = tag)
            b = randn(rng, size(sD.J, 1))
            for state in 1:2
                if state == 2
                    U2 = UD .+ 1e-3 .* randn(rng, length(UD))
                    fv_jacobian!(sD, U2); fv_jacobian!(sK, U2)
                    lin_refactorize!(cD); lin_refactorize!(cK)
                end
                for trp in (false, true)
                    Jop = trp ? transpose(sD.J) : sD.J
                    xD = lin_solve!(cD, b; transpose = trp)
                    xK = lin_solve!(cK, b; transpose = trp)
                    @test norm(Jop * xK - b) <= 1e-10 * norm(b) * (1 + 1e-6)
                    @test norm(xK - xD) <= 1e-5 * norm(xD)
                    @test last(lin_krylov_log(cK)).converged
                end
            end
            @info "I4 $label solve ($pname)" iters = [e.iters for e in lin_krylov_log(cK)]
        end
        # (c) Newton with block upper + LSC (multigrid inner) reaches the branch-D state
        tagN = KrylovBackendTag(; precond = :mg, mg_coarse = 16, block = :upper, schur_approx = :lsc,
                                rtol = 1e-10, rtol_transpose = 1e-11, strict = false)
        sK = fv_newton_solver(p; linsolve = tagN)
        UK = _i4_solve!(sK, _i4_U0(p))
        @test norm(UK - UD, Inf) <= 1e-9 * norm(UD, Inf)
        @info "I4 Newton block upper + LSC vs D ($pname)" rel = norm(UK - UD, Inf) / norm(UD, Inf) gmres_iters = [e.iters for e in lin_krylov_log(sK.cache)]
    end
    # option checks
    p = _i4_duct()
    s0 = fv_newton_solver(p; linsolve = KrylovBackendTag())
    fv_jacobian!(s0, _i4_U0(p))
    @test_throws ArgumentError lin_factorize(s0.J; backend = KrylovBackendTag(; precond = :simple, block = :diag))
    @test_throws ArgumentError lin_factorize(s0.J; backend = KrylovBackendTag(; precond = :simple, schur_approx = :bfbt))
    @test_throws ArgumentError lin_factorize(s0.J; backend = KrylovBackendTag(; precond = :simple, schur_shift = -1.0))
    @test_throws ArgumentError lin_factorize(s0.J; backend = KrylovBackendTag(; precond = :simple, ablock = :gs))
    @test_throws ArgumentError lin_factorize(s0.J; backend = KrylovBackendTag(; precond = :simple, schur_q = :scalar))
    # adjoint gradients == branch D: PCD (ILU inner) on the duct, LSC (multigrid inner) on the Carreau duct
    tagG = (; rtol = 1e-10, rtol_transpose = 1e-11, strict = false)
    base = _i4_duct()
    res = _i4_grad_pair(base, FVUniformViscosity(), [0.04],
                        (("flow rate", fv_flow_rate(base, 2)), ("wall force", FVWallForce(1; region = :solid)));
                        tag = KrylovBackendTag(; precond = :simple, block = :upper, schur_approx = :pcd, tagG...))
    @info "I4 adjoint block upper + PCD vs D (duct, uniform nu)" rel = [r[2] for r in res]
    cb = _i4_carreau()
    res = _i4_grad_pair(cb, FVLawParameters(5), [0.05, 0.01, 2.0, 2.0, 0.5],
                        (("Fx", fv_flow_force_objective(cb)), ("wall force", FVWallForce(1; region = :solid)));
                        tag = KrylovBackendTag(; precond = :mg, mg_coarse = 16, block = :upper,
                                               schur_approx = :lsc, schur_shift = 1.0, tagG...))
    @info "I4 adjoint block upper + LSC + shift vs D (Carreau + flow rate)" rel = [r[2] for r in res]
end

# ------------------------------------------------------------------ (7) P4 = monolithic Vanka multigrid
const _I4_P4 = KrylovBackendTag(; precond = :vanka, mg_coarse = 16, rtol = 1e-10, rtol_transpose = 1e-11,
                                strict = false)

# Hierarchy checks of one monolithic Vanka preconditioner `sm` built on the
# local block `Jloc` (host sparse matrix, `nf` fields per cell, pressure slot
# `ps`) of the grid `geom`: level 1 = Jloc; on every level one patch per fluid
# pressure unknown (at most 10 dofs, fluid only, every fluid unknown in some
# patch) with the stored inverse equal to the inverse of the local matrix;
# restriction by sums, prolongation rows summing to 1 with the own block
# dominant, solid and fluid nodes never in the same coarse node, coarse
# operators R A P; fluid pressure diagonals non-zero (positive when
# `positive`) on every level. Returns the level sizes.
function _i4_vanka_check(sm, Jloc, geom, nf, ps; positive = true)
    nl = length(sm.levels)
    @test nl >= 2
    nloc = size(Jloc, 1)
    A1 = KrakeNN._vanka_level_matrix(sm, 1)
    @test norm(A1 - Jloc) <= 1e-13 * norm(Jloc)
    nd = KrakeNN._mg_nodes(Int32[(i - 1) ÷ nf + 1 for i in 1:nloc], Int32[(i - 1) % nf + 1 for i in 1:nloc], geom)
    sizes = Int[]
    for l in 1:nl
        Al = KrakeNN._vanka_level_matrix(sm, l)
        n = size(Al, 1)
        push!(sizes, n)
        pres = [i for i in 1:n if nd.comp[i] == ps && nd.cls[i] == 0]
        fluid = [i for i in 1:n if nd.cls[i] == 0]
        @test all(Al[i, i] != 0 for i in pres)
        positive && @test all(Al[i, i] > 0 for i in pres)
        L = sm.levels[l]
        if l == nl
            @test L.np == 0 && L.coarse !== nothing
            break
        end
        @test L.np == length(pres) && 1 <= L.M <= 10
        covered = falses(n)
        for t in 1:L.np
            dofs, Mi = KrakeNN._vanka_patch(sm, l, t)
            @test pres[t] in dofs && issorted(dofs) && allunique(dofs)
            @test all(nd.cls[d] == 0 for d in dofs)
            Mref = inv(Matrix(Al[dofs, dofs]))
            @test norm(Mi - Mref) <= 1e-8 * norm(Mref)
            covered[dofs] .= true
        end
        @test all(covered[fluid])
        R = KrakeNN._vanka_restriction_matrix(sm, l)
        P = KrakeNN._vanka_prolongation_matrix(sm, l)
        Ac = KrakeNN._vanka_level_matrix(sm, l + 1)
        agg, ndc, nc = KrakeNN._mg_aggregate(nd)
        @test size(R) == (nc, n) && size(P) == (n, nc) && size(Ac) == (nc, nc)
        @test all(==(1.0), KrakeNN.nonzeros(R))
        @test R * ones(n) == [count(==(c), agg) for c in 1:nc]
        @test norm(P * ones(nc) - ones(n)) <= 1e-13
        @test all(P[a, agg[a]] >= maximum(P[a, :]) - 1e-12 for a in 1:n)
        @test all(ndc.cls[agg[a]] == nd.cls[a] for a in 1:n)
        @test norm(Matrix(Ac) - Matrix(R * Al * P)) <= 1e-12 * norm(Matrix(Ac))
        nd = ndc
    end
    return sizes
end

@testset "Krylov P4 (monolithic Vanka multigrid): hierarchy, solves, Newton, gradient" begin
    rng = MersenneTwister(59)
    for (pname, p) in (("duct", _i4_duct()), ("Carreau + flow rate", _i4_carreau()))
        sD = fv_newton_solver(p)
        UD = _i4_solve!(sD, _i4_U0(p))
        for (label, tg) in (("P4 W linear", KrylovBackendTag(; precond = :vanka, mg_coarse = 16, rtol = 1e-10,
                                                           rtol_transpose = 1e-10)),
                            ("P4 V constant", KrylovBackendTag(; precond = :vanka, mg_coarse = 16, mg_cycle = :V,
                                                              vanka_prolong = :constant, mg_sweeps = 1,
                                                              rtol = 1e-10, rtol_transpose = 1e-10)))
            sK = fv_newton_solver(p; linsolve = tg)
            fv_jacobian!(sD, UD); fv_jacobian!(sK, UD)
            J = sD.J
            nloc = sK.J.nlocal
            cD = lin_factorize(J; backend = CPUBackendTag(), spd = false)
            cK = lin_factorize(sK.J; backend = tg)
            blk = lin_krylov_blocks(cK)
            @test blk.schur === :none && blk.mg_A === nothing && blk.np == fv_ncells(p)
            vk = blk.vanka
            @test vk.levels >= 2 && vk.sizes[1] == nloc && issorted(vk.sizes; rev = true)
            @test vk.prolong === tg.vanka_prolong && vk.periodic == (true, false, false)
            @test vk.patches[1] == count(!, sK.J.geom.solid) && vk.patch_max[1] <= 10
            @test lin_krylov_pivot(cK).min_ratio > 0
            sizes = _i4_vanka_check(cK.factor.fwd.simple, J[1:nloc, 1:nloc], sK.J.geom, 4, 4)
            @test sizes == vk.sizes
            b = randn(rng, size(J, 1))
            for state in 1:2
                if state == 2                                   # refactorization: new values on every level
                    U2 = UD .+ 1e-3 .* randn(rng, length(UD))
                    fv_jacobian!(sD, U2); fv_jacobian!(sK, U2)
                    lin_refactorize!(cD); lin_refactorize!(cK)
                end
                for tr in (false, true)
                    Jop = tr ? transpose(sD.J) : sD.J
                    xD = lin_solve!(cD, b; transpose = tr)
                    xK = lin_solve!(cK, b; transpose = tr)
                    @test norm(Jop * xK - b) <= 1e-10 * norm(b) * (1 + 1e-6)
                    @test norm(xK - xD) <= 1e-5 * norm(xD)
                    @test last(lin_krylov_log(cK)).converged
                end
            end
            # hierarchy of the transposed side: the same rule on the local block of Jᵀ
            Jt = KrakeNN.sparse(transpose(sD.J[1:nloc, 1:nloc]))
            _i4_vanka_check(cK.factor.tr.simple, Jt, sK.J.geom, 4, 4; positive = false)
            # stationary iteration of one application on the local block: finite ratios (logged, not gated)
            st = lin_krylov_stationary(cK, b; iters = 6, local_block = true)
            stt = lin_krylov_stationary(cK, b; iters = 6, local_block = true, transpose = true)
            @test length(st.ratios) == 6 && all(isfinite, st.ratios) && all(isfinite, stt.ratios)
            @test abs(st.norms[1] - norm(b[1:nloc])) <= 1e-12 * norm(b)
            @info "I4 $label solve ($pname)" iters = [e.iters for e in lin_krylov_log(cK)] sizes = vk.sizes patches = vk.patches operator_complexity = vk.operator_complexity stationary_forward = st.ratios stationary_transposed = stt.ratios
        end
        # Newton with P4 reaches the branch-D state
        sK = fv_newton_solver(p; linsolve = _I4_P4)
        UK = _i4_solve!(sK, _i4_U0(p))
        @test norm(UK - UD, Inf) <= 1e-9 * norm(UD, Inf)
        @info "I4 Newton P4 vs D ($pname)" rel = norm(UK - UD, Inf) / norm(UD, Inf) gmres_iters = [e.iters for e in lin_krylov_log(sK.cache)]
    end
    # P4 needs the geometry and the block layout; option checks; the stationary helper needs a Krylov cache
    p = _i4_duct()
    s0 = fv_newton_solver(p; linsolve = KrylovBackendTag())
    fv_jacobian!(s0, _i4_U0(p))
    nogeom = KrylovCSRMatrix(s0.J.n, s0.J.rowptr, s0.J.colval, s0.J.nzval, s0.J.border, s0.J.nfields,
                             s0.J.pslot, s0.J.nlocal)
    @test_throws ArgumentError lin_factorize(nogeom; backend = KrylovBackendTag(; precond = :vanka))
    bare = KrylovCSRMatrix(s0.J.n, s0.J.rowptr, s0.J.colval, s0.J.nzval, 0)
    @test_throws ArgumentError lin_factorize(bare; backend = KrylovBackendTag(; precond = :vanka))
    @test_throws ArgumentError lin_factorize(s0.J; backend = KrylovBackendTag(; precond = :vanka, vanka_omega = 0.0))
    @test_throws ArgumentError lin_factorize(s0.J; backend = KrylovBackendTag(; precond = :vanka, vanka_prolong = :bogus))
    @test_throws ArgumentError lin_factorize(s0.J; backend = KrylovBackendTag(; precond = :vanka, mg_cycle = :K))
    sDd = fv_newton_solver(p)
    fv_jacobian!(sDd, _i4_U0(p))
    cDd = lin_factorize(sDd.J; backend = CPUBackendTag(), spd = false)
    @test_throws ArgumentError lin_krylov_stationary(cDd, randn(MersenneTwister(1), size(sDd.J, 1)))
    # adjoint gradients through P4 == through branch D
    base = _i4_duct()
    res = _i4_grad_pair(base, FVUniformViscosity(), [0.04],
                        (("flow rate", fv_flow_rate(base, 2)), ("wall force", FVWallForce(1; region = :solid)));
                        tag = _I4_P4)
    @info "I4 adjoint P4 vs D (duct, uniform nu)" rel = [r[2] for r in res]
    cb = _i4_carreau()
    res = _i4_grad_pair(cb, FVLawParameters(5), [0.05, 0.01, 2.0, 2.0, 0.5],
                        (("Fx", fv_flow_force_objective(cb)), ("wall force", FVWallForce(1; region = :solid)));
                        tag = _I4_P4)
    @info "I4 adjoint P4 vs D (Carreau + flow rate)" rel = [r[2] for r in res]
    # one centred FD check through P4 Newton re-solves (uniform nu, flow rate)
    map = FVUniformViscosity(); θ = [0.04]; obj = fv_flow_rate(base, 2)
    (_, _, rK, _, UK) = first(_i4_grad_pair(base, map, θ, (("flow rate", obj),); tag = _I4_P4))
    sfd = fv_newton_solver(base; linsolve = _I4_P4)
    Jfd4(t) = (_i4_restart!(sfd, base); fv_set_params!(sfd, map, [t]);
               fv_objective(obj, _i4_solve!(sfd, copy(UK)), sfd.prob))
    gd = rK.grad[1]
    err = minimum(abs((Jfd4(θ[1] + ε) - Jfd4(θ[1] - ε)) / (2ε) - gd) for ε in (4e-6, 8e-7))
    @test err <= 2e-7 + 1e-6 * abs(gd)
    @info "I4 adjoint via P4 vs centred FD (uniform nu, flow rate)" gd err
end
