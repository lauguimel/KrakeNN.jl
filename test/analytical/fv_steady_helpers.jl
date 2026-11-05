# Test-only helpers for the C1 steady FV residual: a coloured sparse Jacobian
# (ForwardDiff duals through the residual kernel, CPU), a Newton loop built on
# it, and the manufactured Stokes / Navier-Stokes solution with variable
# viscosity. NOT the C2 assembler: the colouring here is a fixed 5x5 cell
# pattern, valid because a cell residual reads only cells within a 5x5 box
# (checked in incns_fv_steady_residual.jl), and needs N divisible by 5 on
# periodic sides.

using LinearAlgebra
using Kraken.SparseArrays
using ForwardDiff

struct _FVC1TestTag end

function _fvc1_sparse_jacobian(p, U)
    g = p.grid
    Nx, Ny = g.Nx, g.Ny
    n = fv_nunknowns(p)
    D = ForwardDiff.Dual{_FVC1TestTag,Float64,25}
    Ud = Vector{D}(undef, n)
    Rd = Vector{D}(undef, n)
    I = Int[]; Jc = Int[]; V = Float64[]
    perx = g.bcE == FV_BC_PERIODIC
    pery = g.bcN == FV_BC_PERIODIC
    ((perx && Nx % 5 != 0) || (pery && Ny % 5 != 0)) &&
        throw(ArgumentError("box colouring needs a periodic size divisible by 5"))
    for comp in 1:3
        for j in 1:Ny, i in 1:Nx, c in 1:3
            k = fv_index(g, i, j, c)
            col = (i - 1) % 5 + 5 * ((j - 1) % 5) + 1
            Ud[k] = D(U[k], ForwardDiff.Partials(ntuple(m -> (c == comp && m == col) ? 1.0 : 0.0, 25)))
        end
        fv_residual!(Rd, Ud, p)
        for j in 1:Ny, i in 1:Nx, c in 1:3
            r = fv_index(g, i, j, c)
            for dj in -2:2, di in -2:2
                ii = i + di; jj = j + dj
                perx ? (ii = mod1(ii, Nx)) : (1 <= ii <= Nx || continue)
                pery ? (jj = mod1(jj, Ny)) : (1 <= jj <= Ny || continue)
                col = (ii - 1) % 5 + 5 * ((jj - 1) % 5) + 1
                val = ForwardDiff.partials(Rd[r])[col]
                val == 0 && continue
                push!(I, r); push!(Jc, fv_index(g, ii, jj, comp)); push!(V, val)
            end
        end
    end
    return sparse(I, Jc, V, n, n, (a, b) -> a)
end

function _fvc1_newton(p, U0; tol = 1e-11, maxit = 30)
    U = copy(U0)
    for it in 1:maxit
        R = fv_residual(U, p)
        norm(R, Inf) < tol && return U, it - 1
        U .-= _fvc1_sparse_jacobian(p, U) \ R
    end
    return U, maxit
end

# Manufactured solution on [0,1]^2: divergence-free velocity, pressure with
# non-zero normal gradient at the walls, viscosity varying in x and y (so the
# transpose part of the stress tensor does not vanish).
const _FVC1_K = 2pi
_fvc1_uex(x) = [sin(_FVC1_K * x[1]) * cos(_FVC1_K * x[2]), -cos(_FVC1_K * x[1]) * sin(_FVC1_K * x[2])]
_fvc1_pex(x) = 0.5 * cos(_FVC1_K * x[1]) * sin(2 * _FVC1_K * x[2])
_fvc1_nuex(x) = 1 + 0.4 * sin(_FVC1_K * (x[1] + 2 * x[2]))

# s = conv (u.grad)u + grad p - div[nu (grad u + grad u^T)], by nested ForwardDiff
function _fvc1_source(x, conv)
    G(y) = ForwardDiff.jacobian(_fvc1_uex, y)
    σ(y) = (Gy = G(y); vec(_fvc1_nuex(y) .* (Gy .+ Gy')))
    Dσ = ForwardDiff.jacobian(σ, x)      # rows: σxx, σyx, σxy, σyy
    divσ = (Dσ[1, 1] + Dσ[3, 2], Dσ[2, 1] + Dσ[4, 2])
    gp = ForwardDiff.gradient(_fvc1_pex, x)
    c = conv ? G(x) * _fvc1_uex(x) : zeros(2)
    return (c[1] + gp[1] - divσ[1], c[2] + gp[2] - divσ[2], 0.0)
end

function _fvc1_mms_errors(N, bc, conv)
    p = fv_steady_problem(N, N; nu = (x, y) -> _fvc1_nuex([x, y]), bc = bc, conv = conv,
                          boundary_velocity = (s, x, y) -> Tuple(_fvc1_uex([x, y])),
                          source = (x, y) -> _fvc1_source([x, y], conv))
    Ue = fv_state(p, (x, y) -> _fvc1_uex([x, y])[1], (x, y) -> _fvc1_uex([x, y])[2],
                  (x, y) -> _fvc1_pex([x, y]))
    U, it = _fvc1_newton(p, Ue)
    u, v, pp = fv_fields(p, U)
    ue, ve, pe = fv_fields(p, Ue)
    eu = sqrt(sum(abs2, u - ue) + sum(abs2, v - ve)) / N
    pp .-= sum(pp) / N^2
    pe .-= sum(pe) / N^2
    ep = sqrt(sum(abs2, pp - pe)) / N
    return eu, ep, it
end
