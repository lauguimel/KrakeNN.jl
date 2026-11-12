# Opt-in conformation tensor of the steady 3D FV problem (plan
# .orchestrator/fv-conformation-plan.md, rung CONF-1): six extra unknowns per
# cell, the conformation c itself (not log c), solved together with
# (u, v, w, p) by the shared Newton solver and its coloured dual Jacobian.
# A 3D problem built with `conformation = nothing` keeps the isothermal or
# energy path: no method below applies to it (dispatch on the type parameter).
#
# Unknowns: c occupies the LAST six scalar slots of the cell layout
# (c_xx, c_xy, c_xz, c_yy, c_yz, c_zz), after any energy slot; the slot is
# computed from the layout, never hard-coded.
#
# Residual of the conformation rows of a fluid cell P (cell measure h^3):
#   R_c = sum_f negp(F_f) (c_N - c_P) + h^3 [ -(L c_P + c_P L^T) + Rlx(c_P) / lambda ]
# - advection: non-conservative first-order upwind, sum_f F_f c_upwind - c_P sum_f F_f
#   (the continuity defect is not a source), F_f the Rhie-Chow mass flux of the
#   continuity row (same expression, same face viscosity);
# - L = Green-Gauss cell velocity gradient of the momentum rows (wall faces at
#   the wall velocity);
# - no boundary condition on c: on walls the normal velocity is zero (checked at
#   lowering) so no face carries a flux of c; only wall and periodic sides.
# Polymer stress in momentum, tau_p = G S(c), G = nu_p / lambda (kinematic),
# S = relaxation_stress(law, theta, c) (c - I except FENE-P: f c - I):
#   R_mom -= sum_f h^2 G S(c_f) . (sx e_n),
#   c_f = mean(c_P, c_N) on interior faces, the linear extrapolation
#   (3 c_P - c_O) / 2 from the opposite neighbour O on wall faces (c_P when O
#   is not a fluid neighbour). A 1D model of this discretisation reproduces the
#   Oldroyd-B plane-Poiseuille conformation to round-off and the Newtonian
#   (eta_0 = eta_s + eta_p) velocity; holding c_P on the wall face instead
#   makes c first order at the wall (prototype, CONF-1, 2026-11-11).
# Solid cells: identity rows. Pinned cells: only the continuity row changes.
# Pseudo-transient continuation: `fv_newton!(...; dtau0, shift_conformation =
# true)` adds V/dtau to the diagonal of the six c rows as well as the momentum
# rows (Jacobian only; the steady residual is unchanged).
# Not available yet: energy together with conformation, state-dependent solvent
# viscosity (GNF / law), inlet and outlet sides, adjoint parameter maps.

"""
    FVConformation3D(; lambda, nu_p, alpha = 0, law = PolynomialRelaxation(), theta = nothing)

Opt-in steady conformation equation for the 3D [`fv_steady_problem`](@ref)`(Nx,
Ny, Nz; conformation = FVConformation3D(...))`:

```math
\\mathbf{u}\\cdot\\nabla\\mathbf{c} - \\mathbf{L}\\mathbf{c} - \\mathbf{c}\\mathbf{L}^T
+ \\frac{1}{\\lambda}\\left[b_0\\mathbf{I} + b_1(\\mathbf{c}-\\mathbf{I}) + b_2(\\mathbf{c}-\\mathbf{I})^2\\right] = 0,
\\qquad \\boldsymbol{\\tau}_p = \\frac{\\nu_p}{\\lambda}\\,\\mathbf{S}(\\mathbf{c}),
```

`L_ij = du_i/dx_j`, with the polymer stress `τ_p` added to the momentum
traction (kinematic units, as `nu`: the `nu` of the problem is the solvent
viscosity). `lambda > 0` is the relaxation time, `nu_p >= 0` the polymer
viscosity. `law`/`theta` give `(b0, b1, b2)` and the stress `S(c)` (`c - I`
except FENE-P), see [`AbstractRelaxationLaw`](@ref); the default
`PolynomialRelaxation()` with `theta = (1, alpha)` is Oldroyd-B (`alpha = 0`)
or Giesekus (mobility `alpha`). Other laws: `LinearPTTRelaxation()` and
`ExponentialPTTRelaxation()` with `theta = (epsilon,)`, `FENEPRelaxation()`
with `theta = (L2,)`; `theta` is then required.

The six components `(c_xx, c_xy, c_xz, c_yy, c_yz, c_zz)` are the last six cell
unknowns; `fv_state` starts them at `c = I`, [`fv_conformation_fields`](@ref)
unpacks them. Only wall (zero normal velocity) and periodic sides, a Newtonian
solvent viscosity and no energy equation are supported in this rung.
"""
struct FVConformation3D{T,L<:AbstractRelaxationLaw,P}
    lambda::T
    nu_p::T
    law::L
    theta::P
end

function FVConformation3D(; lambda::Real, nu_p::Real, alpha::Real = 0.0,
                          law::AbstractRelaxationLaw = PolynomialRelaxation(), theta = nothing)
    isfinite(lambda) && lambda > 0 || throw(ArgumentError("lambda must be finite and positive"))
    isfinite(nu_p) && nu_p >= 0 || throw(ArgumentError("nu_p must be finite and nonnegative"))
    isfinite(alpha) || throw(ArgumentError("alpha must be finite"))
    if theta === nothing
        law isa PolynomialRelaxation ||
            throw(ArgumentError("theta is required for $(nameof(typeof(law)))"))
        theta = (1.0, float(alpha))
    end
    l, p = promote(float(lambda), float(nu_p))
    th = Tuple(theta)
    length(th) == relaxation_nparams(law) ||
        throw(DimensionMismatch("theta has $(length(th)) entries, the law takes $(relaxation_nparams(law))"))
    all(x -> x isa Real && isfinite(x), th) || throw(ArgumentError("theta must be finite reals"))
    relaxation_check(law, th)
    return FVConformation3D(l, p, law, th)
end

# Isbits kernel argument (no arrays): G = nu_p / lambda.
struct _FVConformationData3D{T,L<:AbstractRelaxationLaw,P}
    lambda::T
    G::T
    nu_p::T
    law::L
    theta::P
end

# Scalar slots added to the 3D cell layout by the `conformation` keyword.
_fv3_conformation_nscalars(::Nothing) = 0
_fv3_conformation_nscalars(::FVConformation3D) = 6
_fv3_conformation_nscalars(c) =
    throw(ArgumentError("conformation must be nothing or FVConformation3D, got $(typeof(c))"))

# ------------------------------------------------------------------ lowering (host, once)
_fv3_lower_conformation(::Nothing, g, tag, bvel, visc, energy, ::Type) = nothing

function _fv3_lower_conformation(c::FVConformation3D, g::FVGrid3D, tag, bvel, visc, energy,
                                 ::Type{T}) where {T}
    energy === nothing ||
        throw(ArgumentError("conformation together with an energy equation is not available yet"))
    visc isa FVNewtonianViscosity ||
        throw(ArgumentError("conformation requires a Newtonian solvent viscosity (nu field) in this version"))
    isbitstype(typeof(c.law)) || throw(ArgumentError("relaxation law must be isbits"))
    for d in 1:6
        bc = _fv_side_bc(g, d)
        bc in (FV_BC_WALL, FV_BC_PERIODIC) ||
            throw(ArgumentError("conformation supports only wall and periodic sides (side $(_FV3_SIDES[d]))"))
        if bc == FV_BC_WALL
            n = _fv3_axis(d)
            all(iszero, view(bvel, n, :, :, d)) ||
                throw(ArgumentError("conformation requires zero normal wall velocity (side $(_FV3_SIDES[d]))"))
        end
    end
    lambda, nu_p = T(c.lambda), T(c.nu_p)
    theta = map(T, c.theta)
    isfinite(lambda) && lambda > 0 && isfinite(nu_p) && nu_p >= 0 && all(isfinite, theta) ||
        throw(ArgumentError("conformation coefficients must remain finite in $T"))
    relaxation_check(c.law, theta)
    return _FVConformationData3D(lambda, nu_p / lambda, nu_p, c.law, theta)
end

const _FVConfProblem3D = FVSteadyProblem3D{<:FVGrid3D,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,Nothing,<:Any,
                                           <:_FVConformationData3D}

# ------------------------------------------------------------------ cell access
@inline _fv3_conf_slot(layout::FVFieldLayout3D{NS}, m) where {NS} = _fv_scalar_slot(layout, NS - 6 + m)
@inline _fv3_conf_cell(U, g, i, j, k) =
    ntuple(m -> @inbounds(U[fv_index(g, i, j, k, _fv3_conf_slot(g.layout, m))]), Val(6))

@inline _fvc_mean(a, b) = ntuple(m -> (a[m] + b[m]) / 2, Val(6))
@inline _fvc_extrapolate(a, b) = ntuple(m -> (3 * a[m] - b[m]) / 2, Val(6))
@inline _fvc_upwind_acc(acc, f, cN, cP) = ntuple(m -> acc[m] + f * (cN[m] - cP[m]), Val(6))
# helpers keep closures off reassigned locals (a captured, reassigned local is boxed)
@inline _fvc_axpy(a, f, x) = ntuple(m -> a[m] + f * x[m], Val(6))

"""Conformation on face d of cell (i, j, k): mean on interior faces, linear
extrapolation from the opposite fluid neighbour on wall faces."""
@inline function _fv3_conf_face(U, tag, g, cP, i, j, k, d)
    kind, ii, jj, kk = _fv_face_kind(g, tag, i, j, k, d)
    if kind == _FV_F_INT
        return _fvc_mean(cP, _fv3_conf_cell(U, g, ii, jj, kk))
    end
    opposite = isodd(d) ? d + 1 : d - 1
    kindO, oi, oj, ok = _fv_face_kind(g, tag, i, j, k, opposite)
    if kindO == _FV_F_INT
        return _fvc_extrapolate(cP, _fv3_conf_cell(U, g, oi, oj, ok))
    end
    return cP
end

"""Polymer force sum_f h^2 G S(c_f) . (sx e_n) on cell (i, j, k), per component."""
@inline function _fv3_polymer_force(U, tag, g, conf, cP, i, j, k, z)
    fx = z; fy = z; fz = z
    h2 = g.h * g.h
    for d in 1:6
        n = _fv3_axis(d)
        dc = relaxation_stress(conf.law, conf.theta, _fv3_conf_face(U, tag, g, cP, i, j, k, d))
        w = h2 * _fv3_sign(d) * conf.G
        fx += w * _fvc_get(dc, 1, n)
        fy += w * _fvc_get(dc, 2, n)
        fz += w * _fvc_get(dc, 3, n)
    end
    return fx, fy, fz
end

"""
    _fv3_conformation_rows(U, tag, nu, bvel, g, visc, conf, cP, gP, i, j, k, z) -> NTuple{6}

The six conformation rows of fluid cell `(i, j, k)` (see the file header). The
mass flux is the continuity flux of `_fv3_cell_residual`; branches depend on
integer geometry only (upwinding through `abs`), so duals pass every branch.
"""
@inline function _fv3_conformation_rows(U, tag, nu, bvel, g, visc, conf, cP, gP, i, j, k, z)
    uP, vP, wP, pP = _fv3_cell(U, g, i, j, k)
    h = g.h
    @inbounds nuP = nu[i, j, k]
    adv = ntuple(_ -> z, Val(6))
    for d in 1:6
        kind, ii, jj, kk = _fv_face_kind(g, tag, i, j, k, d)
        # wall faces: zero normal velocity (lowering), hence no flux of c
        kind == _FV_F_INT || continue
        n = _fv3_axis(d)
        sx = _fv3_sign(d)
        uN, vN, wN, pN = _fv3_cell(U, g, ii, jj, kk)
        gN = _fv3_cell_grad(U, tag, bvel, g, ii, jj, kk)
        @inbounds nuN = nu[ii, jj, kk]
        du = (uN - uP) / h + z
        dv = (vN - vP) / h + z
        dw = (wN - wP) / h + z
        Gt = _fv3_mean9(gP, gN)
        nuf = _fv_face_nu3(visc, nuP, nuN, _fv3_face_grad(Gt, n, sx, du, dv, dw), i, j, k, ii, jj, kk)
        F = _fv3_interior_mass_flux(h, g.conv, _fv3_pick3(n, uP, vP, wP), _fv3_pick3(n, uN, vN, wN),
                                    pP, pN, gP[9 + n], gN[9 + n], nuf, sx)
        adv = _fvc_upwind_acc(adv, _fv_negp(F), _fv3_conf_cell(U, g, ii, jj, kk), cP)
    end
    h3 = h * h * h
    s = fv_conformation_source(conf.law, conf.theta, conf.lambda, _fv3_first9(gP), cP)
    return _fvc_axpy(adv, h3, s)
end

@inline function _fv3_conf_residual_cell!(R, U, tag, nu, bvel, src, g, visc, conf, i, j, k)
    ru, rv, rw, rc = _fv3_cell_residual(U, tag, nu, bvel, src, g, i, j, k, visc)
    b = fv_index(g, i, j, k, 0)
    layout = g.layout
    @inbounds tP = tag[i, j, k]
    if tP == FV_CELL_SOLID
        @inbounds begin
            R[b + 1] = ru
            R[b + 2] = rv
            R[b + 3] = rw
            R[b + 4] = rc
        end
        @inbounds for scalar in 1:_fv_nscalars(layout)
            slot = _fv_scalar_slot(layout, scalar)
            R[b + slot] = U[b + slot]
        end
        return nothing
    end
    @inbounds uP = U[b + 1]
    z = zero(uP) + zero(eltype(nu)) + zero(eltype(bvel)) + zero(eltype(src)) + _fv_visc_zero(visc) +
        zero(conf.lambda)
    cP = _fv3_conf_cell(U, g, i, j, k)
    fx, fy, fz = _fv3_polymer_force(U, tag, g, conf, cP, i, j, k, z)
    gP = _fv3_cell_grad(U, tag, bvel, g, i, j, k)
    rcs = _fv3_conformation_rows(U, tag, nu, bvel, g, visc, conf, cP, gP, i, j, k, z)
    @inbounds begin
        R[b + 1] = ru - fx
        R[b + 2] = rv - fy
        R[b + 3] = rw - fz
        R[b + 4] = rc
    end
    @inbounds for m in 1:6
        R[b + _fv3_conf_slot(layout, m)] = rcs[m]
    end
    return nothing
end

@kernel function _fv3_conf_residual_kernel!(R, @Const(U), @Const(tag), @Const(nu), @Const(bvel),
                                            @Const(src), g, visc, conf)
    i, j, k = @index(Global, NTuple)
    _fv3_conf_residual_cell!(R, U, tag, nu, bvel, src, g, visc, conf, i, j, k)
end

"""
    fv_residual!(R, U, prob::FVSteadyProblem3D; ...)   (conformation active)

Conformation-active 3D residual: one kernel writes the four flow rows (with the
polymer stress in the momentum traction) and the six conformation rows of every
cell, then the optional mean-flow rows; same contract as the isothermal method.
"""
function fv_residual!(R, U, prob::_FVConfProblem3D; workgroup::Int = 64, sync::Bool = true,
                      mass_rows::Bool = true)
    g = prob.grid
    n = fv_nunknowns(prob)
    length(U) == n || throw(DimensionMismatch("length(U) = $(length(U)), expected $n"))
    length(R) == n || throw(DimensionMismatch("length(R) = $(length(R)), expected $n"))
    backend = KernelAbstractions.get_backend(U)
    _fv3_conf_residual_kernel!(backend, workgroup)(R, U, prob.tag, prob.nu, prob.bvel, prob.src,
        g, prob.visc, prob.conformation; ndrange = (g.Nx, g.Ny, g.Nz))
    _fv_flow_residual!(R, U, prob, backend, workgroup; constraint = mass_rows)
    sync && KernelAbstractions.synchronize(backend)
    return R
end

function fv_residual(U, prob::_FVConfProblem3D)
    T = promote_type(eltype(U), eltype(prob.nu), eltype(prob.bvel), eltype(prob.src),
                     _fv_visc_eltype(prob.visc), typeof(prob.conformation.lambda))
    R = similar(U, T)
    return fv_residual!(R, U, prob)
end

# ------------------------------------------------------------------ Jacobian pattern
"""
    fv_jacobian_pattern(prob::FVSteadyProblem3D)   (conformation active)

Isothermal 3D pattern plus: momentum row `a` on `c_(a,b)`, `b = 1:3` (all six
components when the law's stress is nonlinear, `relaxation_stress_linear(law)
== false`, e.g. FENE-P), at the star (face mean, wall extrapolation from the
opposite neighbour); conformation
row `m` on its own component at the star (upwind), on all six components at P
(source), on `u, v, w` at the star (face flux and cell gradient) and on `p` as
the continuity row (Rhie-Chow flux).
"""
function fv_jacobian_pattern(prob::_FVConfProblem3D)
    g = prob.grid
    tag = Array(prob.tag)
    n = fv_nunknowns(prob)
    I = Int[]; J = Int[]
    sizehint!(I, 60 * n); sizehint!(J, 60 * n)
    add(row, comp, cells) = for c in cells
        c[1] == 0 && continue
        push!(I, row); push!(J, fv_index(g, c[1], c[2], c[3], comp))
    end
    slot(m) = _fv3_conf_slot(g.layout, m)
    linear_stress = relaxation_stress_linear(prob.conformation.law)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        _fv3_push_newtonian!(I, J, g, tag, i, j, k)       # flow rows; diagonal of every scalar slot
        tag[i, j, k] == FV_CELL_SOLID && continue
        s = _fv3_newtonian_dependencies(g, tag, i, j, k)
        star = (s.P, s.nbs...)
        for a in 1:3
            if linear_stress
                for b in 1:3
                    add(fv_index(g, i, j, k, a), slot(_fvc_sym(a, b)), star)
                end
            else
                for q in 1:6
                    add(fv_index(g, i, j, k, a), slot(q), star)
                end
            end
        end
        for m in 1:6
            row = fv_index(g, i, j, k, slot(m))
            add(row, slot(m), star)
            for q in 1:6
                add(row, slot(q), (s.P,))
            end
            for a in 1:3
                add(row, a, star)
            end
            add(row, 4, s.pres)
        end
    end
    _fv_flow_pattern!(I, J, prob)
    return sparse(I, J, trues(length(I)), n, n, |)
end

# ------------------------------------------------------------------ pseudo-transient shift on c rows
# Hooks of `fv_jacobian!(...; shift_conformation = true)` (fv_newton.jl): the
# diagonal of the six conformation rows of every cell gets `sigma = V/dtau`,
# as the momentum rows (solid cells included, as there: identity rows, so the
# converged state is unchanged).
_fv_check_conformation_shift(::_FVConfProblem3D) = nothing

@kernel function _fv3_conf_shift_kernel!(nzval, @Const(diag), sigma, layout)
    c = @index(Global)
    b = _fv_cell_base(layout, c)
    @inbounds for m in 1:6
        nzval[diag[b + _fv3_conf_slot(layout, m)]] += sigma
    end
end

function _fv_shift_conformation!(nzval, diag, sigma, prob::_FVConfProblem3D, backend)
    _fv3_conf_shift_kernel!(backend, 256)(nzval, diag, sigma, prob.grid.layout; ndrange = fv_ncells(prob))
    return nzval
end

# ------------------------------------------------------------------ coefficient update (continuation)
# Same lowering arithmetic as `_fv3_lower_conformation`; the geometric checks
# (sides, wall velocity, solvent model) do not depend on the coefficients and
# were done when the problem was built.
function _fv3_conformation_data(old::_FVConformationData3D, c::FVConformation3D)
    T = typeof(old.lambda)
    lambda, nu_p = T(c.lambda), T(c.nu_p)
    theta = map(T, c.theta)
    isfinite(lambda) && lambda > 0 && isfinite(nu_p) && nu_p >= 0 && all(isfinite, theta) ||
        throw(ArgumentError("conformation coefficients must remain finite in $T"))
    new = _FVConformationData3D(lambda, nu_p / lambda, nu_p, c.law, theta)
    typeof(new) === typeof(old) ||
        throw(ArgumentError("the relaxation law and theta types must match those of the problem " *
                            "($(typeof(old)) vs $(typeof(new)))"))
    return new
end

"""
    fv_set_conformation(prob::FVSteadyProblem3D, conf::FVConformation3D) -> FVSteadyProblem3D

Copy of the conformation-active problem `prob` (host or device) with the
coefficients `lambda`, `nu_p`, `theta` of `conf`; arrays are shared, the law
type must be the same. Bit-identical to building the problem with `conf`.
"""
function fv_set_conformation(p::FVSteadyProblem3D, conf::FVConformation3D)
    p.conformation isa _FVConformationData3D ||
        throw(ArgumentError("problem has no conformation unknowns"))
    return FVSteadyProblem3D(p.grid, p.tag, p.nu, p.bvel, p.src, p.visc, p.immersed, p.energy,
                             p.flowrate, _fv3_conformation_data(p.conformation, conf))
end

"""
    fv_set_conformation!(solver::FVNewtonSolver, conf::FVConformation3D) -> solver

Replace the conformation coefficients (`lambda`, `nu_p`, `theta`) of the
solver's problem in place, e.g. for a continuation in the Deborah number. The
Jacobian pattern, colouring and the symbolic analysis held by the solver are
kept (the pattern does not depend on the coefficients); the next
[`fv_newton!`](@ref) refactorizes numerically. Stored factors are marked stale
for the adjoint.
"""
function fv_set_conformation!(s::FVNewtonSolver, conf::FVConformation3D)
    s.prob isa FVSteadyProblem3D ||
        throw(ArgumentError("fv_set_conformation! needs a 3D conformation-active solver"))
    s.prob = fv_set_conformation(s.prob, conf)
    s.asm_sig = (NaN, NaN, NaN)
    s.fac_sig = (NaN, NaN, NaN)
    return s
end

# ------------------------------------------------------------------ state, fields
_fv3_initial_conformation!(U, p::FVSteadyProblem3D) = _fv3_initial_conformation!(U, p, p.conformation)
_fv3_initial_conformation!(U, p, ::Nothing) = U
function _fv3_initial_conformation!(U, p, ::_FVConformationData3D)
    g = p.grid
    tag = Array(p.tag)
    for cell in 1:fv_ncells(p), m in 1:6
        rest = tag[cell] != FV_CELL_SOLID && _FVC_I[m] == _FVC_J[m]
        U[_fv_cell_index(g, cell, _fv3_conf_slot(g.layout, m))] = rest ? one(eltype(U)) : zero(eltype(U))
    end
    return U
end

"""
    fv_conformation_fields(prob::FVSteadyProblem3D, U) -> (; xx, xy, xz, yy, yz, zz)

Host `Nx × Ny × Nz` copies of the six conformation components of a
conformation-active 3D problem.
"""
function fv_conformation_fields(prob::FVSteadyProblem3D, U::AbstractVector)
    prob.conformation === nothing && throw(ArgumentError("problem has no conformation unknowns"))
    g = prob.grid
    length(U) == fv_nunknowns(prob) || throw(DimensionMismatch("state length does not match problem"))
    A = reshape(Array(U)[1:fv_nunknowns(g)], _fv_nfields(g), g.Nx, g.Ny, g.Nz)
    c = ntuple(m -> A[_fv3_conf_slot(g.layout, m), :, :, :], 6)
    return (; xx = c[1], xy = c[2], xz = c[3], yy = c[4], yz = c[5], zz = c[6])
end
