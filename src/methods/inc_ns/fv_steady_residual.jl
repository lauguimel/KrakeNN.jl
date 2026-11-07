# Steady 2D colocated finite-volume residual R(u, v, p) of the incompressible
# Navier-Stokes (or Stokes) equations, written as ONE KernelAbstractions kernel
# that is generic in the element type (Float64, Float32, ForwardDiff.Dual, ...).
#
# Roadmap "differentiable FV + cuDSS", capability C1. The same kernel is meant
# to serve the Newton solve (C2: the Jacobian is assembled by evaluating this
# kernel on dual numbers), the GNF extension (C3) and the discrete adjoint (C4:
# dR/dtheta by duals on the parameter arrays). There is no CPU mirror: the CPU
# path is the same kernel on the KernelAbstractions CPU backend.
#
# Discretisation (uniform Cartesian grid, spacing h, cell-centred u, v, p);
# the choices are those of the J0 prototype (benchmarks/fv_newton_j0 on
# feat/fv-newton-j0), which this file reproduces to round-off:
#   momentum   sum_f [ conv * F_f * q_upwind - h * t_f ] + h * sum_f p_f n_f - h^2 s_mom = 0
#              t_f = nu_f (grad u + grad u^T) . n   (full stress tensor, variable nu)
#              first-order upwind convection, written branch-free with abs()
#   continuity sum_f F_f - h^2 s_mass = 0,  F_f = h * u_f . n
#   face velocity (Rhie-Chow, decision D-RC (ii)):
#              u_f . n = mean(u) . n - d_f * [ (p_N - p_P)/h - mean(grad p) . n ]
#              d_f = h^2 / (4 nu_f + conv * |mean(u) . n| * h)
#   nu_f = arithmetic mean of the two cell viscosities (Newtonian, prob.nu), or
#          the generalised-Newtonian law at the face strain rate (C3,
#          fv_gnf.jl): the face velocity gradient is the compact normal
#          difference plus the mean of the two cell gradients tangentially;
#          a wall face uses the one-sided normal difference, with tangential
#          gradients extrapolated to immersed L/Q nominal faces. The hook is
#          _fv_face_nu / _fv_wall_nu on prob.visc.
#   Pinned cell (one per closed fluid pocket): the continuity row is replaced
#   by p_P = 0. Solid cells carry identity rows (u, v, p) = 0; a fluid face next
#   to a solid cell is a halfway no-slip wall by default. A lowered immersed
#   geometry replaces only these faces with its prescribed ghost closure.
#
# Unknown ordering: k = 3*(c-1) + {1:u, 2:v, 3:p}, c = i + (j-1)*Nx.
# M3 appends one lambda per immersed (solid body, fluid region) pair.

# ------------------------------------------------------------------ codes
const FV_BC_WALL = 0       # prescribed velocity (no-slip or moving wall)
const FV_BC_INLET = 1      # prescribed velocity, same face treatment as a wall
const FV_BC_OUTLET = 2     # zero normal gradient of velocity, p_face = 0
const FV_BC_PERIODIC = 3

const FV_CELL_FLUID = 0x00
const FV_CELL_SOLID = 0x01
const FV_CELL_PINNED = 0x02   # fluid cell whose continuity row is p = 0

const _FV_F_INT = 0
const _FV_F_DIRICHLET = 1     # wall / inlet / solid neighbour
const _FV_F_OUTLET = 2

# side / direction index d: 1 = E, 2 = W, 3 = N, 4 = S
const _FV_SIDES = (:E, :W, :N, :S)

"""
    FVGrid2D{T}

Isbits description of a uniform `Nx × Ny` grid of spacing `h` and of its
domain-boundary conditions, passed by value to the residual kernel.

Fields: `Nx`, `Ny`, `h`, `conv` (1 = Navier-Stokes, 0 = Stokes; multiplies the
convective flux and the convective part of the Rhie-Chow coefficient), and the
side codes `bcE`, `bcW`, `bcN`, `bcS` (`FV_BC_WALL`, `FV_BC_INLET`,
`FV_BC_OUTLET`, `FV_BC_PERIODIC`; periodic sides come in pairs).
"""
struct FVGrid2D{T}
    Nx::Int
    Ny::Int
    h::T
    conv::T
    bcE::Int
    bcW::Int
    bcN::Int
    bcS::Int
end

"""
    FVSteadyProblem2D(grid, tag, nu, bvel, src[, visc, immersed])

Everything the steady FV residual needs besides the state, as device (or host)
arrays plus the isbits [`FVGrid2D`](@ref). Each array may have its own element
type; the residual is evaluated in the promoted type, so a parameter array of
`ForwardDiff.Dual` (C4: dR/dθ) is as legal as a dual state (C2: dR/dU).

- `tag::AbstractMatrix{UInt8}` (`Nx × Ny`): `FV_CELL_FLUID`, `FV_CELL_SOLID`
  or `FV_CELL_PINNED` (see [`fv_pressure_pins!`](@ref)).
- `nu::AbstractMatrix` (`Nx × Ny`): cell kinematic viscosity.
- `bvel::AbstractArray` (`2 × L × 4`, `L = max(Nx, Ny)`): velocity `(u, v)` of
  the boundary face number `l` (j on E/W sides, i on N/S sides) of side
  `d ∈ (E, W, N, S)`; read on wall and inlet sides only.
- `src::AbstractVector` (length `3 Nx Ny`, unknown ordering): volumetric
  sources `(s_x, s_y, s_mass)`; momentum rows subtract `h² s_x`, `h² s_y`,
  continuity rows `h² s_mass` (zero for a physical flow; MMS uses it).
- `visc`: viscosity model, [`FVNewtonianViscosity`](@ref) (default: the cell
  field `nu`) or [`FVGNFViscosity`](@ref) (generalised-Newtonian law of the
  face strain rate; `nu` is then not read). Its parameters may be duals too.
- `immersed`: `nothing` (the original halfway walls), or a lowered
  [`FVImmersedBoundary2D`](@ref) with fixed cell tags and cut-face data.

Build with [`fv_steady_problem`](@ref); move to a device with
[`fv_adapt`](@ref).
"""
struct FVSteadyProblem2D{G<:FVGrid2D,TA,NA,BA,SA,V,I}
    grid::G
    tag::TA
    nu::NA
    bvel::BA
    src::SA
    visc::V
    immersed::I
end
FVSteadyProblem2D(grid, tag, nu, bvel, src) =
    FVSteadyProblem2D(grid, tag, nu, bvel, src, FVNewtonianViscosity())
FVSteadyProblem2D(grid, tag, nu, bvel, src, visc) =
    FVSteadyProblem2D(grid, tag, nu, bvel, src, visc, nothing)

# Arrays are separate kernel arguments: only the scalar closure description
# crosses the device boundary as a struct. Number types come from signatures.
_fv_immersed_args(::Nothing) = (nothing, nothing, nothing)
_fv_immersed_args(b) = (b.qcut, b.wvel, b.closure)
@inline _fv_array_zero(::Nothing) = false
@inline _fv_array_zero(::AbstractArray{T}) where {T} = zero(T)
_fv_immersed_eltype(::Nothing) = Bool
_fv_immersed_eltype(b) = promote_type(eltype(b.qcut), eltype(b.wvel), _fv_mass_eltype(b.mass))
_fv_adapt_immersed(to, ::Nothing) = nothing
_fv_adapt_immersed(to, b) = fv_adapt(to, b)

fv_nunknowns(g::FVGrid2D) = 3 * g.Nx * g.Ny
fv_nunknowns(p::FVSteadyProblem2D) = fv_nunknowns(p.grid) + _fv_npairs(p.immersed)

"""
    fv_index(grid, i, j, comp) -> Int

Position of unknown `comp` (1 = u, 2 = v, 3 = p) of cell `(i, j)` in the state
vector: `3 (i - 1 + (j - 1) Nx) + comp`.
"""
@inline fv_index(g::FVGrid2D, i, j, comp) = 3 * (i - 1 + (j - 1) * g.Nx) + comp

# ------------------------------------------------------------------ viscosity hooks
"""
    FVNewtonianViscosity()

Viscosity model of an [`FVSteadyProblem2D`](@ref) whose viscosity is the cell
field `prob.nu`, independent of the state (C1/C2): face value = arithmetic mean
of the two cells; C1/domain wall = the cell value, immersed L/Q wall = linear
extrapolation from P and the opposite fluid cell to the nominal face. The generalised-Newtonian model
is [`FVGNFViscosity`](@ref) (C3).
"""
struct FVNewtonianViscosity end

# face viscosity from the two cell values and the face velocity gradient
# (ux, uy, vx, vy); wall face: from the cell value and the wall-face gradient
@inline _fv_face_nu(::FVNewtonianViscosity, nuP, nuN, ux, uy, vx, vy) = (nuP + nuN) / 2
@inline _fv_wall_nu(::FVNewtonianViscosity, nuP, ux, uy, vx, vy) = nuP
# Coordinates select prescribed scalar-law fields; existing viscosity models
# retain their original arithmetic and their short hook signatures.
@inline _fv_face_nu(v, nuP, nuN, ux, uy, vx, vy, i, j, ii, jj) =
    _fv_face_nu(v, nuP, nuN, ux, uy, vx, vy)
@inline _fv_wall_nu(v, nuP, ux, uy, vx, vy, i, j) =
    _fv_wall_nu(v, nuP, ux, uy, vx, vy)
# Context for one oriented residual face; ordinary viscosity models are unchanged.
@inline _fv_face_viscosity(v, d) = v
# zero carrying the number type of the model's parameters (Bool: neutral)
@inline _fv_visc_zero(::FVNewtonianViscosity) = false
_fv_visc_eltype(::FVNewtonianViscosity) = Bool
_fv_adapt_viscosity(to, v) = v
_fv_validate_viscosity(v, g, nu) = nothing
_fv_with_fields(v, fields) = throw(ArgumentError("prescribed fields require FVLawViscosity"))

# ------------------------------------------------------------------ geometry
@inline _fv_val(U, g, i, j, comp) = @inbounds U[fv_index(g, i, j, comp)]

@inline function _fv_side_bc(g, d)
    return d == 1 ? g.bcE : (d == 2 ? g.bcW : (d == 3 ? g.bcN : g.bcS))
end

# (kind, ii, jj) of the face of cell (i, j) in direction d
@inline function _fv_face_kind(g, tag, i, j, d)
    ii = i + (d == 1) - (d == 2)
    jj = j + (d == 3) - (d == 4)
    if ii < 1 || ii > g.Nx || jj < 1 || jj > g.Ny
        bc = _fv_side_bc(g, d)
        if bc == FV_BC_PERIODIC
            ii = mod1(ii, g.Nx)
            jj = mod1(jj, g.Ny)
        elseif bc == FV_BC_OUTLET
            return _FV_F_OUTLET, i, j
        else
            return _FV_F_DIRICHLET, i, j
        end
    end
    @inbounds if tag[ii, jj] == FV_CELL_SOLID
        return _FV_F_DIRICHLET, i, j
    end
    return _FV_F_INT, ii, jj
end

# prescribed (u, v) of a Dirichlet face: domain sides read bvel, faces against
# solid cells are no-slip
@inline function _fv_dirichlet_velocity(g, bvel, i, j, d)
    ii = i + (d == 1) - (d == 2)
    jj = j + (d == 3) - (d == 4)
    if ii < 1 || ii > g.Nx || jj < 1 || jj > g.Ny
        l = d <= 2 ? j : i
        @inbounds return bvel[1, l, d], bvel[2, l, d]
    end
    z = zero(eltype(bvel))
    return z, z
end

@inline function _fv_immersed_face(g, qcut, i, j, d)
    ii = i + (d == 1) - (d == 2)
    jj = j + (d == 3) - (d == 4)
    return qcut !== nothing && 1 <= ii <= g.Nx && 1 <= jj <= g.Ny
end

"""Ghost value at the solid centre: linear P/wall or quadratic O/P/wall."""
@inline function _fv_ghost_value(uO, uP, uw, q, ghost)
    ghost == 1 && return uw + (uw - uP) * (1 - q) / q
    return (1 - q) / (1 + q) * uO - 2 * (1 - q) / q * uP +
           2 / (q * (1 + q)) * uw
end

"""
Wall data at the nominal Cartesian face: prescribed velocity, upwind ghost,
pressure, mean velocity, and normal differences. C1 keeps its original
arithmetic and prescribed inflow value; its geometric q is deliberately ignored.
"""
@inline function _fv_wall_data(U, bvel, g, i, j, d, qcut, wvel, closure)
    z = _fv_array_zero(U) + _fv_array_zero(bvel) + _fv_array_zero(qcut) + _fv_array_zero(wvel)
    uP = _fv_val(U, g, i, j, 1) + z; vP = _fv_val(U, g, i, j, 2) + z
    pP = _fv_val(U, g, i, j, 3) + z
    cut = _fv_immersed_face(g, qcut, i, j, d)
    if cut
        @inbounds uw, vw = wvel[1, d, i, j], wvel[2, d, i, j]
    else
        uw, vw = _fv_dirichlet_velocity(g, bvel, i, j, d)
    end
    uw += z; vw += z
    if !cut || closure.ghost == 0
        return uw, vw, uw, vw, pP, uw, vw, (uw - uP) * 2 / g.h, (vw - vP) * 2 / g.h
    end
    oi = i - (d == 1) + (d == 2)
    oj = j - (d == 3) + (d == 4)
    @inbounds q = qcut[d, i, j]
    uO = _fv_val(U, g, oi, oj, 1); vO = _fv_val(U, g, oi, oj, 2)
    uG = _fv_ghost_value(uO, uP, uw, q, closure.ghost)
    vG = _fv_ghost_value(vO, vP, vw, q, closure.ghost)
    pf = closure.pressure == 0 ? pP : pP + (pP - _fv_val(U, g, oi, oj, 3)) / 2
    return uw, vw, uG, vG, pf, (uP + uG) / 2, (vP + vG) / 2,
           (uG - uP) / g.h, (vG - vP) / g.h
end

"""
Viscous traction through a nominal wall face, shared by residual and force.
Immersed L/Q uses extrapolated given viscosity or the GNF law at the ghost
normal slope and the nominal-face tangential gradient supplied by
`_fv_wall_gradient`; C1/domain walls keep the cell values and arithmetic.
"""
@inline function _fv_wall_stress(nu, visc, g, i, j, d, gF, dun, dvn, qcut, closure)
    @inbounds nuf = nu[i, j]
    if _fv_immersed_face(g, qcut, i, j, d) && closure.ghost != 0
        oi = i - (d == 1) + (d == 2)
        oj = j - (d == 3) + (d == 4)
        @inbounds nuf = nuf + (nuf - nu[oi, oj]) / 2
    end
    sx = (d == 1 || d == 3) ? 1 : -1
    if d <= 2
        nuw = _fv_wall_nu(visc, nuf, sx * dun, gF[2], sx * dvn, gF[4], i, j)
        return 2 * nuw * dun, nuw * (dvn + sx * gF[2])
    end
    nuw = _fv_wall_nu(visc, nuf, gF[1], sx * dun, gF[3], sx * dvn, i, j)
    return nuw * (dun + sx * gF[3]), 2 * nuw * dvn
end

# face values (u, v, p) of the face of cell (i, j) in direction d
@inline function _fv_face_values(U, tag, bvel, g, i, j, d,
                                qcut = nothing, wvel = nothing, closure = nothing)
    uP = _fv_val(U, g, i, j, 1); vP = _fv_val(U, g, i, j, 2); pP = _fv_val(U, g, i, j, 3)
    kind, ii, jj = _fv_face_kind(g, tag, i, j, d)
    zb = _fv_array_zero(bvel) + _fv_array_zero(qcut) + _fv_array_zero(wvel)
    if kind == _FV_F_INT
        uN = _fv_val(U, g, ii, jj, 1); vN = _fv_val(U, g, ii, jj, 2); pN = _fv_val(U, g, ii, jj, 3)
        return (uP + uN) / 2 + zb, (vP + vN) / 2 + zb, (pP + pN) / 2 + zb
    elseif kind == _FV_F_OUTLET
        return uP + zb, vP + zb, zero(pP) + zb
    else
        _, _, _, _, pf, uf, vf, _, _ = _fv_wall_data(U, bvel, g, i, j, d, qcut, wvel, closure)
        return zero(uP) + uf, zero(vP) + vf, pf
    end
end

# Green-Gauss cell gradient: (ux, uy, vx, vy, px, py)
@inline function _fv_cell_grad(U, tag, bvel, g, i, j,
                              qcut = nothing, wvel = nothing, closure = nothing)
    ue, ve, pe = _fv_face_values(U, tag, bvel, g, i, j, 1, qcut, wvel, closure)
    uw, vw, pw = _fv_face_values(U, tag, bvel, g, i, j, 2, qcut, wvel, closure)
    un, vn, pn = _fv_face_values(U, tag, bvel, g, i, j, 3, qcut, wvel, closure)
    us, vs, ps = _fv_face_values(U, tag, bvel, g, i, j, 4, qcut, wvel, closure)
    ih = inv(g.h)
    return ((ue - uw) * ih, (un - us) * ih, (ve - vw) * ih, (vn - vs) * ih,
            (pe - pw) * ih, (pn - ps) * ih)
end

"""
Tangential velocity gradients at an immersed L/Q nominal face: extrapolate
`gP + (gP - gO)/2` from P and the opposite fluid cell O. This is exact for
quadratic velocity with Q ghosts. Normal and pressure entries are unused by
the wall stress; keep them unchanged. C1/domain walls return gP bit-for-bit.
"""
@inline function _fv_wall_gradient(U, tag, bvel, g, i, j, d, gP, qcut, wvel, closure)
    if _fv_immersed_face(g, qcut, i, j, d) && closure.ghost != 0
        oi = i - (d == 1) + (d == 2)
        oj = j - (d == 3) + (d == 4)
        gO = _fv_cell_grad(U, tag, bvel, g, oi, oj, qcut, wvel, closure)
        if d <= 2
            return (gP[1], gP[2] + (gP[2] - gO[2]) / 2,
                    gP[3], gP[4] + (gP[4] - gO[4]) / 2, gP[5], gP[6])
        end
        return (gP[1] + (gP[1] - gO[1]) / 2, gP[2],
                gP[3] + (gP[3] - gO[3]) / 2, gP[4], gP[5], gP[6])
    end
    return gP
end

@inline _fv_posp(F) = (F + abs(F)) / 2
@inline _fv_negp(F) = (F - abs(F)) / 2

"""
    fv_cell_residual(U, tag, nu, bvel, src, grid, i, j[, visc, qcut, wvel, closure, face_pair])
        -> (Ru, Rv, Rc)

Residual of the three equations of cell `(i, j)`. Branches depend only on the
integer geometry (tags, boundary codes), never on the values of `U`, `nu`,
`bvel` or `src`, so the function is traceable by global sparsity detectors and
differentiable by dual numbers through every branch (the upwind switch is
written with `abs`). `visc` is the viscosity model (default
[`FVNewtonianViscosity`](@ref)). The immersed arguments are the lowered cut arrays, isbits closure and optional
M3 face/pair indices. `face_pair` selects appended lambdas from `U`; `nothing`
preserves the uncorrected face flux. No geometry preserves the C1 walls.
"""
@inline function fv_cell_residual(U, tag, nu, bvel, src, g, i, j,
                                  visc = FVNewtonianViscosity(),
                                  qcut = nothing, wvel = nothing, closure = nothing, face_pair = nothing)
    uP = _fv_val(U, g, i, j, 1); vP = _fv_val(U, g, i, j, 2); pP = _fv_val(U, g, i, j, 3)
    h = g.h
    @inbounds nuP = nu[i, j]
    z = zero(uP) + zero(nuP) + zero(eltype(bvel)) + zero(eltype(src)) + _fv_visc_zero(visc) +
        _fv_array_zero(qcut) + _fv_array_zero(wvel)
    @inbounds tP = tag[i, j]
    if tP == FV_CELL_SOLID
        return z + uP, z + vP, z + pP
    end
    gP = _fv_cell_grad(U, tag, bvel, g, i, j, qcut, wvel, closure)
    Ru = z; Rv = z; Rc = z
    for d in 1:4
        face_visc = _fv_face_viscosity(visc, d)
        isx = d <= 2
        sx = (d == 1 || d == 3) ? 1 : -1
        kind, ii, jj = _fv_face_kind(g, tag, i, j, d)
        if kind == _FV_F_INT
            uN = _fv_val(U, g, ii, jj, 1); vN = _fv_val(U, g, ii, jj, 2); pN = _fv_val(U, g, ii, jj, 3)
            gN = _fv_cell_grad(U, tag, bvel, g, ii, jj, qcut, wvel, closure)
            @inbounds nuN = nu[ii, jj]
            dun = (uN - uP) / h
            dvn = (vN - vP) / h
            # face velocity gradient: compact normal difference, mean of the
            # two cell gradients tangentially (read by a GNF model only)
            if isx
                nuf = _fv_face_nu(face_visc, nuP, nuN, sx * dun, (gP[2] + gN[2]) / 2,
                                  sx * dvn, (gP[4] + gN[4]) / 2, i, j, ii, jj)
            else
                nuf = _fv_face_nu(face_visc, nuP, nuN, (gP[1] + gN[1]) / 2, sx * dun,
                                  (gP[3] + gN[3]) / 2, sx * dvn, i, j, ii, jj)
            end
            ubn = sx * (isx ? (uP + uN) / 2 : (vP + vN) / 2)
            # Rhie-Chow coefficient, decision D-RC (ii)
            df = h * h / (4 * nuf + g.conv * abs(ubn) * h)
            dpc = (pN - pP) / h
            dpa = sx * (isx ? (gP[5] + gN[5]) / 2 : (gP[6] + gN[6]) / 2)
            F = h * (ubn - df * (dpc - dpa))
            Ru += g.conv * (_fv_posp(F) * uP + _fv_negp(F) * uN)
            Rv += g.conv * (_fv_posp(F) * vP + _fv_negp(F) * vN)
            if isx
                tx = 2 * nuf * dun
                ty = nuf * (dvn + sx * (gP[2] + gN[2]) / 2)
            else
                tx = nuf * (dun + sx * (gP[3] + gN[3]) / 2)
                ty = 2 * nuf * dvn
            end
            Ru -= h * tx; Rv -= h * ty
            pf = (pP + pN) / 2
            Rc += F
        elseif kind == _FV_F_OUTLET
            F = h * sx * (isx ? uP : vP)
            Ru += g.conv * F * uP
            Rv += g.conv * F * vP
            pf = zero(pP)
            Rc += F
        else
            au, av, tx, ty, pf, F = _fv_wall_flux_components(U, tag, nu, bvel, g, i, j, d,
                                                            face_visc, qcut, wvel, closure, gP, face_pair)
            Ru += au; Rv += av
            Ru -= h * tx; Rv -= h * ty
            Rc += F
        end
        if isx
            Ru += h * sx * pf
        else
            Rv += h * sx * pf
        end
    end
    b = fv_index(g, i, j, 0)
    @inbounds begin
        Ru -= h * h * src[b + 1]
        Rv -= h * h * src[b + 2]
        Rc -= h * h * src[b + 3]
    end
    if tP == FV_CELL_PINNED
        Rc = z + pP
    end
    return Ru, Rv, Rc
end

@kernel function _fv_residual_kernel!(R, @Const(U), @Const(tag), @Const(nu),
                                      @Const(bvel), @Const(src), g, visc,
                                      @Const(qcut), @Const(wvel), closure, @Const(face_pair))
    i, j = @index(Global, NTuple)
    ru, rv, rc = fv_cell_residual(U, tag, nu, bvel, src, g, i, j, visc, qcut, wvel, closure, face_pair)
    b = fv_index(g, i, j, 0)
    @inbounds R[b + 1] = ru
    @inbounds R[b + 2] = rv
    @inbounds R[b + 3] = rc
end

"""
    fv_residual!(R, U, prob::FVSteadyProblem2D; workgroup = 64, sync = true) -> R

Evaluate the steady FV residual of the state `U` into `R` (both of length
`fv_nunknowns(prob)`, cell ordering of [`fv_index`](@ref), followed by M3 lambdas) on the backend of `U`.
All arrays of `prob` must live on the same backend as `U`. `eltype(R)` must
hold the promoted element type of `U` and of the arrays of `prob` (e.g.
`ForwardDiff.Dual` when either carries duals).

Allocation-free on the device (the GPU test checks `CUDA.@allocated == 0`).
`sync = false` skips the final `KernelAbstractions.synchronize`, for callers
that chain kernels on the same stream (C2 assembly). `mass_rows=false` is an
internal assembly option: evaluate only cell rows, leaving pair rows untouched
for their direct Jacobian assembly. Full residual callers keep the default.
"""
function fv_residual!(R, U, prob::FVSteadyProblem2D; workgroup::Int = 64, sync::Bool = true,
                      mass_rows::Bool = true)
    g = prob.grid
    n = fv_nunknowns(prob)
    length(U) == n || throw(DimensionMismatch("length(U) = $(length(U)), expected $n"))
    length(R) == n || throw(DimensionMismatch("length(R) = $(length(R)), expected $n"))
    backend = KernelAbstractions.get_backend(U)
    qcut, wvel, closure = _fv_immersed_args(prob.immersed)
    _fv_residual_kernel!(backend, workgroup)(R, U, prob.tag, prob.nu, prob.bvel,
                                             prob.src, g, prob.visc, qcut, wvel, closure, _fv_face_pair(prob.immersed);
                                             ndrange = (g.Nx, g.Ny))
    mass_rows && _fv_mass_residual!(R, U, prob, backend, workgroup)
    sync && KernelAbstractions.synchronize(backend)
    return R
end

"""
    fv_residual(U, prob) -> R

Allocating convenience wrapper of [`fv_residual!`](@ref); `R` has the promoted
element type of `U` and of the arrays of `prob`.
"""
function fv_residual(U, prob::FVSteadyProblem2D)
    T = promote_type(eltype(U), eltype(prob.nu), eltype(prob.bvel), eltype(prob.src),
                     _fv_visc_eltype(prob.visc), _fv_immersed_eltype(prob.immersed))
    R = similar(U, T)
    return fv_residual!(R, U, prob)
end

# ------------------------------------------------------------------ setup (CPU, once)
"""
    fv_pressure_pins!(tag, grid) -> Int

Mark one `FV_CELL_PINNED` cell in every closed fluid pocket of `tag` and
return the number of pins. A pocket is a connected set of fluid cells
(4-connectivity through fluid faces, periodic sides wrapped); it is closed
when none of its cells has a face on an outlet side, so its pressure is only
defined up to a constant. The pinned cell is the pocket's cell of lowest
linear index `i + (j - 1) Nx`. Existing pins are cleared first. CPU, once per
geometry (host arrays).
"""
function fv_pressure_pins!(tag::AbstractMatrix{UInt8}, g::FVGrid2D)
    Nx, Ny = g.Nx, g.Ny
    size(tag) == (Nx, Ny) || throw(DimensionMismatch("tag must be $Nx × $Ny"))
    for k in eachindex(tag)
        tag[k] == FV_CELL_PINNED && (tag[k] = FV_CELL_FLUID)
    end
    comp = zeros(Int, Nx, Ny)
    npins = 0
    stack = Tuple{Int,Int}[]
    for j0 in 1:Ny, i0 in 1:Nx
        (tag[i0, j0] == FV_CELL_SOLID || comp[i0, j0] != 0) && continue
        comp[i0, j0] = 1
        empty!(stack); push!(stack, (i0, j0))
        open = false
        while !isempty(stack)
            i, j = pop!(stack)
            for d in 1:4
                kind, ii, jj = _fv_face_kind(g, tag, i, j, d)
                if kind == _FV_F_OUTLET
                    open = true
                elseif kind == _FV_F_INT && comp[ii, jj] == 0
                    comp[ii, jj] = 1
                    push!(stack, (ii, jj))
                end
            end
        end
        # (i0, j0) is the first cell of this pocket in linear order
        if !open
            tag[i0, j0] = FV_CELL_PINNED
            npins += 1
        end
    end
    return npins
end

_fv_bc_code(s::Symbol) =
    s === :wall ? FV_BC_WALL : s === :inlet ? FV_BC_INLET :
    s === :outlet ? FV_BC_OUTLET : s === :periodic ? FV_BC_PERIODIC :
    throw(ArgumentError("unknown FV boundary condition :$s (expected :wall, :inlet, :outlet, :periodic)"))

"""
    fv_steady_problem(Nx, Ny; h, nu, bc, boundary_velocity, solid, immersed, source, conv, T)
        -> FVSteadyProblem2D (host arrays)

Build the host-side problem of the steady FV residual.

- `h`: grid spacing (default `1/Ny`, unit-height domain).
- `nu`: a number (uniform) or a `(x, y) -> ν` function evaluated at cell
  centres, or an `Nx × Ny` matrix.
- `bc`: `NamedTuple` with keys `W, E, S, N`, values `:wall`, `:inlet`,
  `:outlet` or `:periodic` (default all `:wall`).
- `boundary_velocity`: `(side::Symbol, x, y) -> (u, v)` evaluated at the
  boundary face centres of wall and inlet sides (default no-slip).
- `solid`: optional `Nx × Ny` Bool/integer mask (nonzero = solid).
- `immersed`: a lowered [`FVImmersedBoundary2D`](@ref) with the same dimensions
  and spacing `h` (checked); supplies the solid mask and cut-face values.
  Cannot be combined with `solid`.
- `wall_flux`: physical outward wall flux per M3 body/region pair: scalar,
  pair-ordered vector, or `(body, region) -> Q`. Use exact zero for rigid
  closed walls. Analytic geometry without a target uses reported quadrature;
  raw geometry requires this keyword.
- `source`: `nothing`, a `(x, y) -> (s_x, s_y, s_mass)` function at cell
  centres, or a length-`3 Nx Ny` vector.
- `conv`: `true` (Navier-Stokes, default) or `false` (Stokes).
- `T`: element type of the arrays (default `Float64`).
- `rheology`: `nothing` (default; Newtonian, viscosity `nu`) or a
  generalised-Newtonian law of `src/rheology/models.jl` (`PowerLaw`,
  `CarreauYasuda`, `Cross`, `Bingham`, `HerschelBulkley`, `Newtonian`) in
  kinematic FV units; `nu` is then ignored. `gnf_eps` is the strain-rate
  regularisation `ε` of [`FVGNFViscosity`](@ref) (`γ̇_ε = √(2 S:S + ε²)`).
- `viscosity`: an explicit viscosity model, including [`FVLawViscosity`](@ref)
  for a scalar law with prescribed cell fields. Mutually exclusive with
  `rheology`; its own `eps` supplies the shear-rate regularisation.

Pressure pins are set by [`fv_pressure_pins!`](@ref). Cell centres are at
`((i - 1/2) h, (j - 1/2) h)`.
"""
function fv_steady_problem(Nx::Integer, Ny::Integer; h::Real = 1 / Ny, nu = 1.0,
                           bc = (W = :wall, E = :wall, S = :wall, N = :wall),
                           boundary_velocity = (side, x, y) -> (0.0, 0.0),
                           solid = nothing, source = nothing, conv::Bool = true,
                           T::Type{<:Real} = Float64, rheology = nothing,
                           gnf_eps::Real = 0.0, immersed = nothing, wall_flux = nothing,
                           viscosity = nothing)
    Nx = Int(Nx); Ny = Int(Ny)
    (_fv_bc_code(bc.W) == FV_BC_PERIODIC) == (_fv_bc_code(bc.E) == FV_BC_PERIODIC) ||
        throw(ArgumentError("periodic W/E sides must be paired"))
    (_fv_bc_code(bc.S) == FV_BC_PERIODIC) == (_fv_bc_code(bc.N) == FV_BC_PERIODIC) ||
        throw(ArgumentError("periodic S/N sides must be paired"))
    g = FVGrid2D{T}(Nx, Ny, T(h), conv ? one(T) : zero(T),
                    _fv_bc_code(bc.E), _fv_bc_code(bc.W), _fv_bc_code(bc.N), _fv_bc_code(bc.S))
    xc(i) = (i - 0.5) * h
    yc(j) = (j - 0.5) * h
    tag = fill(FV_CELL_FLUID, Nx, Ny)
    if immersed !== nothing
        solid === nothing || throw(ArgumentError("specify either solid or immersed, not both"))
        size(immersed.tag) == (Nx, Ny) || throw(DimensionMismatch("immersed geometry must be $Nx × $Ny"))
        immersed.h == h || throw(ArgumentError("immersed geometry spacing $(immersed.h) does not match h=$h"))
        solid = immersed.tag .== FV_CELL_SOLID
    end
    if solid !== nothing
        size(solid) == (Nx, Ny) || throw(DimensionMismatch("solid must be $Nx × $Ny"))
        for k in eachindex(tag)
            solid[k] != 0 && (tag[k] = FV_CELL_SOLID)
        end
    end
    fv_pressure_pins!(tag, g)
    immersed = _fv_lower_mass(immersed, g, tag, wall_flux)
    nuc = if nu isa Number
        fill(T(nu), Nx, Ny)
    elseif nu isa AbstractMatrix
        size(nu) == (Nx, Ny) || throw(DimensionMismatch("nu must be $Nx × $Ny"))
        T.(nu)
    else
        T[nu(xc(i), yc(j)) for i in 1:Nx, j in 1:Ny]
    end
    L = max(Nx, Ny)
    bvel = zeros(T, 2, L, 4)
    for (d, s) in enumerate(_FV_SIDES)
        code = _fv_bc_code(getproperty(bc, s))
        (code == FV_BC_WALL || code == FV_BC_INLET) || continue
        nl = d <= 2 ? Ny : Nx
        for l in 1:nl
            x = d == 1 ? Nx * h : d == 2 ? 0.0 : xc(l)
            y = d == 3 ? Ny * h : d == 4 ? 0.0 : yc(l)
            uv = boundary_velocity(s, x, y)
            bvel[1, l, d] = uv[1]; bvel[2, l, d] = uv[2]
        end
    end
    n = 3 * Nx * Ny
    srcv = if source === nothing
        zeros(T, n)
    elseif source isa AbstractVector
        length(source) == n || throw(DimensionMismatch("source must have length $n"))
        T.(source)
    else
        s = zeros(T, n)
        for j in 1:Ny, i in 1:Nx
            f = source(xc(i), yc(j))
            for c in 1:3
                s[fv_index(g, i, j, c)] = f[c]
            end
        end
        s
    end
    viscosity === nothing || rheology === nothing ||
        throw(ArgumentError("specify either viscosity or rheology, not both"))
    visc = viscosity === nothing ?
        (rheology === nothing ? FVNewtonianViscosity() : FVGNFViscosity(rheology, T(gnf_eps))) : viscosity
    _fv_validate_viscosity(visc, g, nuc)
    return FVSteadyProblem2D(g, tag, nuc, bvel, srcv, visc, immersed)
end

"""
    fv_adapt(to, prob::FVSteadyProblem2D) -> FVSteadyProblem2D

Copy the arrays of `prob` with the array constructor `to` (e.g. `CuArray`,
or `Array` to come back to the host). The grid is passed through.
"""
fv_adapt(to, p::FVSteadyProblem2D) =
    FVSteadyProblem2D(p.grid, to(p.tag), to(p.nu), to(p.bvel), to(p.src), _fv_adapt_viscosity(to, p.visc),
                      _fv_adapt_immersed(to, p.immersed))

"""
    fv_state(prob, u, v, p) -> Vector

Pack cell fields (`Nx × Ny` matrices, or `(x, y) -> value` functions evaluated
at cell centres) into a host state vector in the unknown ordering.
M3 lambda entries are initialized to zero.
"""
function fv_state(prob::FVSteadyProblem2D, u, v, p)
    g = prob.grid
    T = eltype(prob.nu)
    U = zeros(T, fv_nunknowns(prob))
    get(f::AbstractMatrix, i, j) = f[i, j]
    get(f, i, j) = f((i - 0.5) * g.h, (j - 0.5) * g.h)
    for j in 1:g.Ny, i in 1:g.Nx
        U[fv_index(g, i, j, 1)] = get(u, i, j)
        U[fv_index(g, i, j, 2)] = get(v, i, j)
        U[fv_index(g, i, j, 3)] = get(p, i, j)
    end
    return U
end

"""
    fv_fields(prob, U) -> (u, v, p)

Unpack the cell entries of a (host) state vector into three `Nx × Ny` matrices.
Any appended M3 lambda entries remain accessible in the original vector.
"""
function fv_fields(prob::FVSteadyProblem2D, U::AbstractVector)
    g = prob.grid
    length(U) == fv_nunknowns(prob) || throw(DimensionMismatch("state length does not match problem"))
    A = reshape(Array(U)[1:fv_nunknowns(g)], 3, g.Nx, g.Ny)
    return A[1, :, :], A[2, :, :], A[3, :, :]
end

# ------------------------------------------------------------------ platform
"""
    residual(prob::FVSteadyProblem2D, m::IncNS, U, p = nothing) -> R

Platform-contract entry of the steady FV residual (capability
`SteadyResidual`, declared by `IncNS(:fv_newton)` only): allocating
[`fv_residual`](@ref). `p` is unused — the parameters (viscosity field,
sources, boundary velocities) live in `prob`. Other IncNS drivers raise an
`ArgumentError`: this is the residual of the coupled colocated FV
discretisation that `:fv_newton` solves, not of the SIMPLE drivers (their
converged state depends on the relaxation factor).
"""
function residual(prob::FVSteadyProblem2D, m::IncNS, U, p = nothing)
    m.driver === :fv_newton ||
        throw(ArgumentError("IncNS(:$(m.driver)) does not declare SteadyResidual; " *
                            "the steady FV residual belongs to IncNS(:fv_newton)"))
    return fv_residual(U, prob)
end
