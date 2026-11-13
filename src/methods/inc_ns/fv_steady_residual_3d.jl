# Steady 3D colocated finite-volume residual R(u, v, w, p) of the
# incompressible Navier-Stokes (or Stokes) equations: the 3D counterpart of
# fv_steady_residual.jl, written as a separate KernelAbstractions kernel that is
# generic in the element type (Float64, Float32, ForwardDiff.Dual, ...).
#
# Plan "solveur FV stationnaire 3D", increment I1 (steps 1-2). The 2D kernel is
# left untouched (its arithmetic order is protected); the Newton solver, the
# coloured dual assembly and the linear-solve seam are shared through the
# abstract FVSteadyProblem and the accessors fv_ncells / _fv_nextra.
#
# Discretisation (uniform Cartesian grid, spacing h, cell-centred u, v, w, p),
# face by face exactly as in 2D with face area h^2 and cell volume h^3:
#   momentum   sum_f [ conv F_f q_upwind - h^2 t_f ] + h^2 sum_f p_f n_f - h^3 s = 0
#              t_f = nu_f (grad u + grad u^T) . n, first-order upwind via abs()
#   continuity sum_f F_f - h^3 s_mass = 0,  F_f = h^2 u_f . n
#   Rhie-Chow  u_f . n = mean(u) . n - d_f [ (p_N - p_P)/h - mean(grad p) . n ]
#              d_f = h^2 / (4 nu_f + conv |mean(u) . n| h)   (the 2D coefficient,
#              kept so that a z-uniform 3D state reproduces the 2D residual)
#   face velocity gradient: compact normal difference, mean of the two
#   Green-Gauss cell gradients tangentially (9 components); a wall face uses
#   the one-sided normal difference and the cell gradient tangentially.
#   nu_f: arithmetic mean (Newtonian), or the law at the face invariant
#   I1 = S:S (+ eps^2/2) through the 9-component hooks _fv_face_nu3/_fv_wall_nu3.
#   Pinned cell (one per closed fluid pocket, 6-connectivity): continuity row
#   replaced by p_P = 0. Solid cells: identity rows; a fluid face next to a
#   solid cell is a halfway no-slip wall.
#
# Unknown ordering: cell c = i + (j-1) Nx + (k-1) Nx Ny,
# index = nfields (c-1) + {1:u, 2:v, 3:w, 4:p, 5:scalar1, ...}.
# Direction d: 1 = E (+x), 2 = W (-x), 3 = N (+y), 4 = S (-y), 5 = T (+z), 6 = B (-z).
# Not available: immersed boundaries (voxel solids only). The adjoint (I2) is
# in fv_adjoint_3d.jl, the opt-in energy equation (I3, slot 5) in fv_energy_3d.jl.

"""
    FVFieldLayout3D{NS}()

Internal, isbits 3D cell layout: three velocity components (slots 1:3),
pressure (slot 4) and `NS` scalar slots (5, 6, ...), identity rows unless a
scalar model is active (slot 5 = temperature with [`FVEnergy3D`](@ref)).
"""
struct FVFieldLayout3D{NS} <: AbstractFVFieldLayout
    function FVFieldLayout3D{NS}() where {NS}
        NS isa Int && NS >= 0 || throw(ArgumentError("scalar count must be a nonnegative Int"))
        return new{NS}()
    end
end
@inline _fv_nscalars(::FVFieldLayout3D{NS}) where {NS} = NS
@inline _fv_nfields(layout::FVFieldLayout3D) = 4 + _fv_nscalars(layout)
@inline _fv_velocity_slot(::FVFieldLayout3D, component) = component
@inline _fv_pressure_slot(::FVFieldLayout3D) = 4
@inline _fv_scalar_slot(::FVFieldLayout3D, scalar) = 4 + scalar

# Momentum-row hooks of the shared Newton norms and pseudo-transient shift.
@inline _fv_cell_measure(h, ::FVFieldLayout3D) = h * h * h
@inline function _fv_shift_momentum!(nzval, diag, b, sigma, layout::FVFieldLayout3D)
    @inbounds for a in 1:3
        nzval[diag[b + _fv_velocity_slot(layout, a)]] += sigma
    end
    return nothing
end
@inline function _fv_momentum_sq(R, b, layout::FVFieldLayout3D)
    @inbounds ru = R[b + 1]; @inbounds rv = R[b + 2]; @inbounds rw = R[b + 3]
    return ru * ru + rv * rv + rw * rw
end
@inline function _fv_momentum_max(mi, R, b, layout::FVFieldLayout3D)
    @inbounds ru = R[b + 1]; @inbounds rv = R[b + 2]; @inbounds rw = R[b + 3]
    return max(mi, abs(ru), abs(rv), abs(rw))
end
@inline _fv_momentum_l2(l2, layout::FVFieldLayout3D) = hypot(l2[1], l2[2], l2[3])

"""
    FVGrid3D{T}

Isbits description of a uniform `Nx × Ny × Nz` grid of spacing `h` and of its
six side codes (`bcE`, `bcW`, `bcN`, `bcS`, `bcT`, `bcB`: `FV_BC_WALL`,
`FV_BC_INLET`, `FV_BC_OUTLET`, `FV_BC_PERIODIC`), passed by value to the 3D
residual kernel. `conv` is 1 (Navier-Stokes) or 0 (Stokes).
"""
struct FVGrid3D{T,L<:FVFieldLayout3D}
    Nx::Int
    Ny::Int
    Nz::Int
    h::T
    conv::T
    bcE::Int
    bcW::Int
    bcN::Int
    bcS::Int
    bcT::Int
    bcB::Int
    layout::L
end
@inline _fv_nfields(g::FVGrid3D) = _fv_nfields(g.layout)
@inline _fv_cell_base(g::FVGrid3D, cell) = _fv_cell_base(g.layout, cell)
@inline _fv_cell_index(g::FVGrid3D, cell, component) = _fv_cell_index(g.layout, cell, component)
@inline _fv_cell_from_index(g::FVGrid3D, index) = _fv_cell_from_index(g.layout, index)
fv_nunknowns(g::FVGrid3D) = _fv_nfields(g) * g.Nx * g.Ny * g.Nz

"""
    fv_index(grid::FVGrid3D, i, j, k, comp) -> Int

Position of unknown `comp` (1 = u, 2 = v, 3 = w, 4 = p, then scalar slots) of
cell `(i, j, k)`: `nfields (i - 1 + (j - 1) Nx + (k - 1) Nx Ny) + comp`.
"""
@inline fv_index(g::FVGrid3D, i, j, k, comp) =
    _fv_cell_index(g, i + (j - 1) * g.Nx + (k - 1) * g.Nx * g.Ny, comp)

"""
    FVSteadyProblem3D(grid, tag, nu, bvel, src, visc, immersed, energy, flowrate)

3D counterpart of [`FVSteadyProblem2D`](@ref); build with
[`fv_steady_problem`](@ref)`(Nx, Ny, Nz; ...)`.

- `tag` (`Nx × Ny × Nz`, `UInt8`): `FV_CELL_FLUID`, `FV_CELL_SOLID`, `FV_CELL_PINNED`.
- `nu` (`Nx × Ny × Nz`): cell kinematic viscosity (Newtonian model).
- `bvel` (`3 × L × L × 6`, `L = max(Nx, Ny, Nz)`): `(u, v, w)` of the boundary
  face `(l1, l2)` of side `d`; `(l1, l2) = (j, k)` on E/W, `(i, k)` on N/S,
  `(i, j)` on T/B. Read on wall and inlet sides only.
- `src` (cell unknown ordering): `(s_x, s_y, s_z, s_mass)` per cell; rows
  subtract `h³ s`.
- `visc`: [`FVNewtonianViscosity`](@ref), [`FVGNFViscosity`](@ref) or
  [`FVLawViscosity`](@ref) (prescribed fields as `Nx × Ny × Nz` arrays).
- `immersed`: `nothing` (not available in 3D).
- `energy`: `nothing`, or the lowered [`FVEnergy3D`](@ref) model (temperature
  in slot 5; see fv_energy_3d.jl).
- `flowrate`: `nothing`, or the lowered mean-x-velocity target with the
  global force unknown Fx (periodic x required), as in 2D.
"""
struct FVSteadyProblem3D{G<:FVGrid3D,TA,NA,BA,SA,V,I,E,F} <: FVSteadyProblem
    grid::G
    tag::TA
    nu::NA
    bvel::BA
    src::SA
    visc::V
    immersed::I
    energy::E
    flowrate::F
end

fv_ncells(p::FVSteadyProblem3D) = p.grid.Nx * p.grid.Ny * p.grid.Nz
@inline _fv_extra_index(p::FVSteadyProblem3D, k) = fv_nunknowns(p.grid) + k
fv_nunknowns(p::FVSteadyProblem3D) = _fv_extra_index(p, _fv_nextra(p))

const _FV3_SIDES = (:E, :W, :N, :S, :T, :B)

# ------------------------------------------------------------------ viscosity hooks (9 components)
# G = (ux, uy, uz, vx, vy, vz, wx, wy, wz): entry 3(a-1)+b is d u_a / d x_b.
@inline function _fv_strain_invariant3(G)
    sxy = (G[2] + G[4]) / 2
    sxz = (G[3] + G[7]) / 2
    syz = (G[6] + G[8]) / 2
    return G[1] * G[1] + G[5] * G[5] + G[9] * G[9] + 2 * (sxy * sxy + sxz * sxz + syz * syz)
end
@inline _fv_face_nu3(::FVNewtonianViscosity, nuP, nuN, G, i, j, k, ii, jj, kk) = (nuP + nuN) / 2
@inline _fv_wall_nu3(::FVNewtonianViscosity, nuP, G, i, j, k) = nuP
@inline _fv_gnf_nu3(v::FVGNFViscosity, G) =
    gnf_viscosity(v.law, _fv_strain_invariant3(G) + v.eps * v.eps / 2)
@inline _fv_face_nu3(v::FVGNFViscosity, nuP, nuN, G, i, j, k, ii, jj, kk) = _fv_gnf_nu3(v, G)
@inline _fv_wall_nu3(v::FVGNFViscosity, nuP, G, i, j, k) = _fv_gnf_nu3(v, G)
@inline _fv_law_nu3(v::FVLawViscosity, G, fields::Tuple) =
    _fv_law_invariant(v.law, v.theta, _fv_strain_invariant3(G) + v.eps * v.eps / 2, fields)
@inline function _fv_face_nu3(v::FVLawViscosity, nuP, nuN, G, i, j, k, ii, jj, kk)
    fields = map(v.fields) do f
        @inbounds (f[i, j, k] + f[ii, jj, kk]) / 2
    end
    return _fv_law_nu3(v, G, fields)
end
@inline function _fv_wall_nu3(v::FVLawViscosity, nuP, G, i, j, k)
    fields = map(f -> @inbounds(f[i, j, k]), v.fields)
    return _fv_law_nu3(v, G, fields)
end

_fv3_validate_viscosity(::FVNewtonianViscosity, g, nu) = nothing
function _fv3_validate_viscosity(v::FVGNFViscosity, g, nu)
    isbitstype(typeof(v.law)) || throw(ArgumentError("FV GNF law must be isbits"))
    return nothing
end
function _fv3_validate_viscosity(v::FVLawViscosity, g, nu)
    isbitstype(typeof(v.law)) || throw(ArgumentError("FV scalar law must be isbits"))
    length(v.theta) == law_nparams(v.law) ||
        throw(DimensionMismatch("theta does not match the law parameter count"))
    length(v.fields) == law_ninputs(v.law) - 1 ||
        throw(ArgumentError("prescribed field count does not match the law"))
    isfinite(v.eps) && v.eps >= 0 && (v.eps > 0 || _fv_law_regular_at_rest(v.law, v.theta)) ||
        throw(ArgumentError("scalar law must be regular at rest: use eps > 0"))
    any(f -> f isa _FVAnyStateTemperature, v.fields) && _fv_nfields(g) < 5 &&
        throw(ArgumentError("FVStateTemperature requires an active 3D energy model (FVEnergy3D)"))
    all(f -> f isa _FVAnyStateTemperature ||
             (f isa AbstractArray{<:Any,3} && size(f) == (g.Nx, g.Ny, g.Nz)), v.fields) ||
        throw(DimensionMismatch("3D law fields must be $(g.Nx) × $(g.Ny) × $(g.Nz) arrays " *
                                "or FVStateTemperature()"))
    return nothing
end
_fv3_validate_viscosity(v, g, nu) =
    throw(ArgumentError("viscosity model $(nameof(typeof(v))) is not available in the 3D FV residual"))

# ------------------------------------------------------------------ geometry
@inline function _fv_side_bc(g::FVGrid3D, d)
    return d == 1 ? g.bcE : d == 2 ? g.bcW : d == 3 ? g.bcN :
           d == 4 ? g.bcS : d == 5 ? g.bcT : g.bcB
end
@inline _fv3_axis(d) = (d + 1) >> 1
@inline _fv3_sign(d) = isodd(d) ? 1 : -1
@inline _fv3_step(i, j, k, d) =
    (i + (d == 1) - (d == 2), j + (d == 3) - (d == 4), k + (d == 5) - (d == 6))

# (kind, ii, jj, kk) of the face of cell (i, j, k) in direction d
@inline function _fv_face_kind(g::FVGrid3D, tag, i, j, k, d)
    ii, jj, kk = _fv3_step(i, j, k, d)
    if ii < 1 || ii > g.Nx || jj < 1 || jj > g.Ny || kk < 1 || kk > g.Nz
        bc = _fv_side_bc(g, d)
        if bc == FV_BC_PERIODIC
            ii = mod1(ii, g.Nx)
            jj = mod1(jj, g.Ny)
            kk = mod1(kk, g.Nz)
        elseif bc == FV_BC_OUTLET
            return _FV_F_OUTLET, i, j, k
        else
            return _FV_F_DIRICHLET, i, j, k
        end
    end
    @inbounds if tag[ii, jj, kk] == FV_CELL_SOLID
        return _FV_F_DIRICHLET, i, j, k
    end
    return _FV_F_INT, ii, jj, kk
end

# prescribed (u, v, w) of a Dirichlet face: domain sides read bvel, faces
# against solid cells are no-slip
@inline function _fv3_dirichlet_velocity(g, bvel, i, j, k, d)
    ii, jj, kk = _fv3_step(i, j, k, d)
    if ii < 1 || ii > g.Nx || jj < 1 || jj > g.Ny || kk < 1 || kk > g.Nz
        l1 = d <= 2 ? j : i
        l2 = d <= 4 ? k : j
        @inbounds return bvel[1, l1, l2, d], bvel[2, l1, l2, d], bvel[3, l1, l2, d]
    end
    z = zero(eltype(bvel))
    return z, z, z
end

@inline _fv3_cell(U, g, i, j, k) = @inbounds (U[fv_index(g, i, j, k, 1)], U[fv_index(g, i, j, k, 2)],
                                             U[fv_index(g, i, j, k, 3)], U[fv_index(g, i, j, k, 4)])

# face values (u, v, w, p) of the face of cell (i, j, k) in direction d
@inline function _fv3_face_values(U, tag, bvel, g, i, j, k, d)
    uP, vP, wP, pP = _fv3_cell(U, g, i, j, k)
    kind, ii, jj, kk = _fv_face_kind(g, tag, i, j, k, d)
    zb = _fv_array_zero(bvel)
    if kind == _FV_F_INT
        uN, vN, wN, pN = _fv3_cell(U, g, ii, jj, kk)
        return (uP + uN) / 2 + zb, (vP + vN) / 2 + zb, (wP + wN) / 2 + zb, (pP + pN) / 2 + zb
    elseif kind == _FV_F_OUTLET
        return uP + zb, vP + zb, wP + zb, zero(pP) + zb
    else
        uw, vw, ww = _fv3_dirichlet_velocity(g, bvel, i, j, k, d)
        return zero(uP) + uw, zero(vP) + vw, zero(wP) + ww, pP + zb
    end
end

# Green-Gauss cell gradient: (ux, uy, uz, vx, vy, vz, wx, wy, wz, px, py, pz)
@inline function _fv3_cell_grad(U, tag, bvel, g, i, j, k)
    fe = _fv3_face_values(U, tag, bvel, g, i, j, k, 1)
    fw = _fv3_face_values(U, tag, bvel, g, i, j, k, 2)
    fn = _fv3_face_values(U, tag, bvel, g, i, j, k, 3)
    fs = _fv3_face_values(U, tag, bvel, g, i, j, k, 4)
    ft = _fv3_face_values(U, tag, bvel, g, i, j, k, 5)
    fb = _fv3_face_values(U, tag, bvel, g, i, j, k, 6)
    ih = inv(g.h)
    return ((fe[1] - fw[1]) * ih, (fn[1] - fs[1]) * ih, (ft[1] - fb[1]) * ih,
            (fe[2] - fw[2]) * ih, (fn[2] - fs[2]) * ih, (ft[2] - fb[2]) * ih,
            (fe[3] - fw[3]) * ih, (fn[3] - fs[3]) * ih, (ft[3] - fb[3]) * ih,
            (fe[4] - fw[4]) * ih, (fn[4] - fs[4]) * ih, (ft[4] - fb[4]) * ih)
end

# Both branches carry the promoted number type (adding exact zeros).
@inline _fv3_pick(b, n, normal, tangential) =
    b == n ? normal + zero(tangential) : tangential + zero(normal)
@inline _fv3_pick3(n, a, b, c) = n == 1 ? a : (n == 2 ? b : c)

# Face velocity gradient: column n (normal axis) = sx * one-sided/compact normal
# difference, other columns = the supplied tangential gradient Gt.
@inline function _fv3_face_grad(Gt, n, sx, du, dv, dw)
    return (_fv3_pick(1, n, sx * du, Gt[1]), _fv3_pick(2, n, sx * du, Gt[2]), _fv3_pick(3, n, sx * du, Gt[3]),
            _fv3_pick(1, n, sx * dv, Gt[4]), _fv3_pick(2, n, sx * dv, Gt[5]), _fv3_pick(3, n, sx * dv, Gt[6]),
            _fv3_pick(1, n, sx * dw, Gt[7]), _fv3_pick(2, n, sx * dw, Gt[8]), _fv3_pick(3, n, sx * dw, Gt[9]))
end

# Viscous traction nu (grad u + grad u^T) . (sx e_n) through a face of normal
# axis n: t_n = 2 nu d_n, t_c = nu (d_c + sx d u_n / d x_c) for c != n, with
# d_c the normal difference of component c and Gt the tangential gradient.
@inline function _fv3_traction(nuf, n, sx, du, dv, dw, Gt)
    gn1 = _fv3_pick3(n, Gt[1], Gt[4], Gt[7])
    gn2 = _fv3_pick3(n, Gt[2], Gt[5], Gt[8])
    gn3 = _fv3_pick3(n, Gt[3], Gt[6], Gt[9])
    t1 = n == 1 ? 2 * nuf * du : nuf * (du + sx * gn1)
    t2 = n == 2 ? 2 * nuf * dv : nuf * (dv + sx * gn2)
    t3 = n == 3 ? 2 * nuf * dw : nuf * (dw + sx * gn3)
    return t1, t2, t3
end

@inline _fv3_mean9(gP, gN) = ntuple(q -> (gP[q] + gN[q]) / 2, Val(9))
@inline _fv3_first9(gP) = ntuple(q -> gP[q], Val(9))

"""Oriented 3D Rhie--Chow mass flux through a face of area h^2 (2D coefficient d_f)."""
@inline function _fv3_interior_mass_flux(h, conv, uPn, uNn, pP, pN, dpP, dpN, nuf, sx)
    ubn = sx * (uPn + uNn) / 2
    df = h * h / (4 * nuf + conv * abs(ubn) * h)
    dpc = (pN - pP) / h
    dpa = sx * (dpP + dpN) / 2
    return h * h * (ubn - df * (dpc - dpa))
end

"""
    _fv3_cell_residual(U, tag, nu, bvel, src, grid::FVGrid3D, i, j, k, visc) -> (Ru, Rv, Rw, Rc)

Residual of the four flow equations of cell `(i, j, k)`. Branches depend only
on integer geometry (tags, side codes), never on values, so the function is
differentiable by dual numbers through every branch (upwind written with `abs`).
"""
@inline function _fv3_cell_residual(U, tag, nu, bvel, src, g::FVGrid3D, i, j, k, visc)
    uP, vP, wP, pP = _fv3_cell(U, g, i, j, k)
    h = g.h
    @inbounds nuP = nu[i, j, k]
    z = zero(uP) + zero(nuP) + zero(eltype(bvel)) + zero(eltype(src)) + _fv_visc_zero(visc)
    @inbounds tP = tag[i, j, k]
    if tP == FV_CELL_SOLID
        return z + uP, z + vP, z + wP, z + pP
    end
    gP = _fv3_cell_grad(U, tag, bvel, g, i, j, k)
    Ru = z; Rv = z; Rw = z; Rc = z
    h2 = h * h
    for d in 1:6
        n = _fv3_axis(d)
        sx = _fv3_sign(d)
        # identity except for the adjoint's law-capture wrapper (one slot per face)
        vf = _fv_face_viscosity(visc, d)
        uPn = _fv3_pick3(n, uP, vP, wP)
        kind, ii, jj, kk = _fv_face_kind(g, tag, i, j, k, d)
        if kind == _FV_F_INT
            uN, vN, wN, pN = _fv3_cell(U, g, ii, jj, kk)
            gN = _fv3_cell_grad(U, tag, bvel, g, ii, jj, kk)
            @inbounds nuN = nu[ii, jj, kk]
            du = (uN - uP) / h + z
            dv = (vN - vP) / h + z
            dw = (wN - wP) / h + z
            Gt = _fv3_mean9(gP, gN)
            nuf = _fv_face_nu3(vf, nuP, nuN, _fv3_face_grad(Gt, n, sx, du, dv, dw),
                               i, j, k, ii, jj, kk)
            uNn = _fv3_pick3(n, uN, vN, wN)
            F = _fv3_interior_mass_flux(h, g.conv, uPn, uNn, pP, pN,
                                        gP[9 + n], gN[9 + n], nuf, sx)
            Ru += g.conv * (_fv_posp(F) * uP + _fv_negp(F) * uN)
            Rv += g.conv * (_fv_posp(F) * vP + _fv_negp(F) * vN)
            Rw += g.conv * (_fv_posp(F) * wP + _fv_negp(F) * wN)
            t1, t2, t3 = _fv3_traction(nuf, n, sx, du, dv, dw, Gt)
            Ru -= h2 * t1; Rv -= h2 * t2; Rw -= h2 * t3
            pf = (pP + pN) / 2 + z
            Rc += F
        elseif kind == _FV_F_OUTLET
            F = h2 * sx * uPn
            Ru += g.conv * F * uP
            Rv += g.conv * F * vP
            Rw += g.conv * F * wP
            pf = zero(pP) + z
            Rc += F
        else
            uw, vw, ww = _fv3_dirichlet_velocity(g, bvel, i, j, k, d)
            uw += z; vw += z; ww += z
            du = (uw - uP) * 2 / h
            dv = (vw - vP) * 2 / h
            dw = (ww - wP) * 2 / h
            Gt = _fv3_first9(gP)
            nuw = _fv_wall_nu3(vf, nuP, _fv3_face_grad(Gt, n, sx, du, dv, dw), i, j, k)
            F = h2 * sx * _fv3_pick3(n, uw, vw, ww)
            Ru += g.conv * (_fv_posp(F) * uP + _fv_negp(F) * uw)
            Rv += g.conv * (_fv_posp(F) * vP + _fv_negp(F) * vw)
            Rw += g.conv * (_fv_posp(F) * wP + _fv_negp(F) * ww)
            t1, t2, t3 = _fv3_traction(nuw, n, sx, du, dv, dw, Gt)
            Ru -= h2 * t1; Rv -= h2 * t2; Rw -= h2 * t3
            pf = pP + z
            Rc += F
        end
        if n == 1
            Ru += h2 * sx * pf
        elseif n == 2
            Rv += h2 * sx * pf
        else
            Rw += h2 * sx * pf
        end
    end
    b = fv_index(g, i, j, k, 0)
    h3 = h2 * h
    @inbounds begin
        Ru -= h3 * src[b + 1]
        Rv -= h3 * src[b + 2]
        Rw -= h3 * src[b + 3]
        Rc -= h3 * src[b + 4]
    end
    if tP == FV_CELL_PINNED
        Rc = z + pP
    end
    return Ru, Rv, Rw, Rc
end

@inline function _fv3_residual_cell!(R, U, tag, nu, bvel, src, g, visc, i, j, k)
    ru, rv, rw, rc = _fv3_cell_residual(U, tag, nu, bvel, src, g, i, j, k, visc)
    b = fv_index(g, i, j, k, 0)
    @inbounds begin
        R[b + 1] = ru
        R[b + 2] = rv
        R[b + 3] = rw
        R[b + 4] = rc
    end
    @inbounds for scalar in 1:_fv_nscalars(g.layout)
        slot = _fv_scalar_slot(g.layout, scalar)
        R[b + slot] = U[b + slot]
    end
    return nothing
end

@kernel function _fv3_residual_kernel!(R, @Const(U), @Const(tag), @Const(nu),
                                       @Const(bvel), @Const(src), g, visc)
    i, j, k = @index(Global, NTuple)
    _fv3_residual_cell!(R, U, tag, nu, bvel, src, g, visc, i, j, k)
end

"""
    fv_residual!(R, U, prob::FVSteadyProblem3D; workgroup = 64, sync = true) -> R

3D steady FV residual into `R` on the backend of `U` (one kernel over the
cells, then the optional mean-flow rows). Same contract as the 2D method:
`eltype(R)` holds the promoted element type, no device allocation,
`sync = false` chains kernels, `mass_rows = false` skips the analytic
constraint row (coloured assembly).
"""
function fv_residual!(R, U, prob::FVSteadyProblem3D; workgroup::Int = 64, sync::Bool = true,
                      mass_rows::Bool = true)
    g = prob.grid
    n = fv_nunknowns(prob)
    length(U) == n || throw(DimensionMismatch("length(U) = $(length(U)), expected $n"))
    length(R) == n || throw(DimensionMismatch("length(R) = $(length(R)), expected $n"))
    backend = KernelAbstractions.get_backend(U)
    _fv3_residual_kernel!(backend, workgroup)(R, U, prob.tag, prob.nu, prob.bvel, prob.src,
                                              g, prob.visc; ndrange = (g.Nx, g.Ny, g.Nz))
    _fv_flow_residual!(R, U, prob, backend, workgroup; constraint = mass_rows)
    sync && KernelAbstractions.synchronize(backend)
    return R
end

function fv_residual(U, prob::FVSteadyProblem3D)
    T = promote_type(eltype(U), eltype(prob.nu), eltype(prob.bvel), eltype(prob.src),
                     _fv_visc_eltype(prob.visc))
    R = similar(U, T)
    return fv_residual!(R, U, prob)
end

# ------------------------------------------------------------------ mean-flow control (3D)
function _fv3_lower_flowrate(target::Real, g, tag, src, ::Type{T}) where {T}
    isfinite(target) || throw(ArgumentError("u_mean_target must be finite"))
    g.bcE == FV_BC_PERIODIC && g.bcW == FV_BC_PERIODIC ||
        throw(ArgumentError("mean-flow control requires periodic x boundaries"))
    cells = findall(!=(FV_CELL_SOLID), vec(tag))
    isempty(cells) && throw(ArgumentError("mean-flow control requires fluid cells"))
    force = src[_fv_cell_index(g, first(cells), 1)]
    all(c -> src[_fv_cell_index(g, c, 1)] == force, cells) ||
        throw(ArgumentError("mean-flow control replaces a uniform x source with Fx"))
    isfinite(force) && isfinite(T(target)) || throw(ArgumentError("flow data must remain finite in $T"))
    return _FVFlowRate(T(target), T(force), length(cells))
end
_fv3_lower_flowrate(::Nothing, g, tag, src, T) = nothing

@kernel function _fv3_flow_force_kernel!(R, @Const(U), @Const(tag), @Const(src), g, force_index)
    cell = @index(Global)
    @inbounds if tag[cell] != FV_CELL_SOLID
        row = _fv_cell_index(g, cell, 1)
        R[row] += g.h * g.h * g.h * (src[row] - U[force_index])
    end
end

function _fv_flow_residual!(R, U, p::FVSteadyProblem3D, backend, workgroup; constraint = true)
    p.flowrate === nothing && return
    index = _fv_force_index(p); g = p.grid
    _fv3_flow_force_kernel!(backend, workgroup)(R, U, p.tag, p.src, g, index; ndrange = fv_ncells(p))
    constraint && _fv_flow_constraint_kernel!(backend, 1)(R, U, p.tag, g, p.flowrate, index; ndrange = 1)
    return nothing
end

function _fv_flow_pattern!(I, J, p::FVSteadyProblem3D)
    p.flowrate === nothing && return
    index = _fv_force_index(p); g = p.grid
    tag = p.tag isa Array ? p.tag : Array(p.tag)     # host copy of a device problem's tags
    for cell in 1:fv_ncells(p)
        tag[cell] == FV_CELL_SOLID && continue
        velocity = _fv_cell_index(g, cell, 1)
        push!(I, velocity, index); push!(J, index, velocity)
    end
    push!(I, index); push!(J, index)
    return nothing
end

function _fv_flow_jacobian!(nzval, asm, p::FVSteadyProblem3D, backend)
    p.flowrate === nothing && return
    _fv_flow_jacobian_kernel!(backend, 256)(nzval, asm.nzrow, asm.nzcol, _fv_force_index(p),
        p.grid.h^3, inv(p.flowrate.nfluid); ndrange = asm.nnz)
    return nothing
end

# ------------------------------------------------------------------ setup (CPU, once)
"""
    fv_pressure_pins!(tag::Array{UInt8,3}, grid::FVGrid3D) -> Int

3D pressure pins: one `FV_CELL_PINNED` cell per closed fluid pocket
(6-connectivity through fluid faces, periodic sides wrapped; a pocket is open
when one of its cells has an outlet face), at the pocket's lowest linear
index `i + (j - 1) Nx + (k - 1) Nx Ny`. Existing pins are cleared first.
"""
function fv_pressure_pins!(tag::AbstractArray{UInt8,3}, g::FVGrid3D)
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    size(tag) == (Nx, Ny, Nz) || throw(DimensionMismatch("tag must be $Nx × $Ny × $Nz"))
    for c in eachindex(tag)
        tag[c] == FV_CELL_PINNED && (tag[c] = FV_CELL_FLUID)
    end
    comp = falses(Nx, Ny, Nz)
    npins = 0
    stack = NTuple{3,Int}[]
    for k0 in 1:Nz, j0 in 1:Ny, i0 in 1:Nx
        (tag[i0, j0, k0] == FV_CELL_SOLID || comp[i0, j0, k0]) && continue
        comp[i0, j0, k0] = true
        empty!(stack); push!(stack, (i0, j0, k0))
        open = false
        while !isempty(stack)
            i, j, k = pop!(stack)
            for d in 1:6
                kind, ii, jj, kk = _fv_face_kind(g, tag, i, j, k, d)
                if kind == _FV_F_OUTLET
                    open = true
                elseif kind == _FV_F_INT && !comp[ii, jj, kk]
                    comp[ii, jj, kk] = true
                    push!(stack, (ii, jj, kk))
                end
            end
        end
        if !open
            tag[i0, j0, k0] = FV_CELL_PINNED
            npins += 1
        end
    end
    return npins
end

"""
    fv_steady_problem(Nx, Ny, Nz; h, nu, bc, boundary_velocity, solid, source, conv,
                      T, rheology, gnf_eps, viscosity, u_mean_target, energy) -> FVSteadyProblem3D

Host-side 3D steady FV problem (same keywords as the 2D builder, one more
dimension). `bc` has keys `W, E, S, N, B, T` (x-, x+, y-, y+, z-, z+), values
`:wall`, `:inlet`, `:outlet`, `:periodic`. `boundary_velocity(side, x, y, z)
-> (u, v, w)` at the boundary face centres. `nu`: number, `(x, y, z) -> ν` or
an `Nx × Ny × Nz` array. `solid`: optional `Nx × Ny × Nz` mask. `source`:
`nothing`, `(x, y, z) -> (s_x, s_y, s_z, s_mass)` or a vector in the unknown
ordering. `rheology`/`gnf_eps` build an [`FVGNFViscosity`](@ref); `viscosity`
passes an explicit model ([`FVLawViscosity`](@ref) with `Nx × Ny × Nz` fields).
`u_mean_target` replaces a uniform x source by the global force unknown Fx
(periodic x). Cell centres are at `((i - 1/2) h, (j - 1/2) h, (k - 1/2) h)`.
`energy = FVEnergy3D(...)` adds the steady temperature equation (slot 5; wall
and periodic sides only); `energy = nothing` keeps the isothermal problem.
Immersed boundaries are not available in 3D.
"""
function fv_steady_problem(Nx::Integer, Ny::Integer, Nz::Integer; h::Real = 1 / Ny, nu = 1.0,
                           bc = (W = :wall, E = :wall, S = :wall, N = :wall, B = :wall, T = :wall),
                           boundary_velocity = (side, x, y, z) -> (0.0, 0.0, 0.0),
                           solid = nothing, source = nothing, conv::Bool = true,
                           T::Type{<:Real} = Float64, rheology = nothing, gnf_eps::Real = 0.0,
                           viscosity = nothing, u_mean_target = nothing,
                           immersed = nothing, energy = nothing)
    immersed === nothing || throw(ArgumentError("immersed boundaries are not available in the 3D FV residual"))
    Nx = Int(Nx); Ny = Int(Ny); Nz = Int(Nz)
    for (a, b) in ((:W, :E), (:S, :N), (:B, :T))
        (_fv_bc_code(getproperty(bc, a)) == FV_BC_PERIODIC) ==
            (_fv_bc_code(getproperty(bc, b)) == FV_BC_PERIODIC) ||
            throw(ArgumentError("periodic $a/$b sides must be paired"))
    end
    g = FVGrid3D(Nx, Ny, Nz, T(h), conv ? one(T) : zero(T),
                 _fv_bc_code(bc.E), _fv_bc_code(bc.W), _fv_bc_code(bc.N), _fv_bc_code(bc.S),
                 _fv_bc_code(bc.T), _fv_bc_code(bc.B),
                 energy === nothing ? FVFieldLayout3D{0}() : FVFieldLayout3D{1}())
    xc(i) = (i - 0.5) * h
    tag = fill(FV_CELL_FLUID, Nx, Ny, Nz)
    if solid !== nothing
        size(solid) == (Nx, Ny, Nz) || throw(DimensionMismatch("solid must be $Nx × $Ny × $Nz"))
        for c in eachindex(tag)
            solid[c] != 0 && (tag[c] = FV_CELL_SOLID)
        end
    end
    fv_pressure_pins!(tag, g)
    nuc = if nu isa Number
        fill(T(nu), Nx, Ny, Nz)
    elseif nu isa AbstractArray
        size(nu) == (Nx, Ny, Nz) || throw(DimensionMismatch("nu must be $Nx × $Ny × $Nz"))
        T.(nu)
    else
        T[nu(xc(i), xc(j), xc(k)) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
    end
    L = max(Nx, Ny, Nz)
    bvel = zeros(T, 3, L, L, 6)
    dims = (Nx, Ny, Nz)
    for (d, s) in enumerate(_FV3_SIDES)
        code = _fv_bc_code(getproperty(bc, s))
        (code == FV_BC_WALL || code == FV_BC_INLET) || continue
        n = _fv3_axis(d)
        t1, t2 = n == 1 ? (2, 3) : n == 2 ? (1, 3) : (1, 2)
        for l2 in 1:dims[t2], l1 in 1:dims[t1]
            x = zeros(3)
            x[n] = isodd(d) ? dims[n] * h : 0.0
            x[t1] = xc(l1); x[t2] = xc(l2)
            uvw = boundary_velocity(s, x[1], x[2], x[3])
            for c in 1:3
                bvel[c, l1, l2, d] = uvw[c]
            end
        end
    end
    n = fv_nunknowns(g)
    srcv = if source === nothing
        zeros(T, n)
    elseif source isa AbstractVector
        if length(source) == n
            T.(source)
        elseif energy !== nothing && length(source) == 4 * Nx * Ny * Nz
            s = zeros(T, n)
            for cell in 1:(Nx * Ny * Nz), c in 1:4
                s[_fv_cell_index(g, cell, c)] = source[4 * (cell - 1) + c]
            end
            s
        else
            throw(DimensionMismatch(energy === nothing ? "source must have length $n" :
                "source must have length $n (or $(4 * Nx * Ny * Nz) flow entries with energy)"))
        end
    else
        s = zeros(T, n)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            f = source(xc(i), xc(j), xc(k))
            for c in 1:4
                s[fv_index(g, i, j, k, c)] = f[c]
            end
        end
        s
    end
    viscosity === nothing || rheology === nothing ||
        throw(ArgumentError("specify either viscosity or rheology, not both"))
    visc = viscosity === nothing ?
        (rheology === nothing ? FVNewtonianViscosity() : FVGNFViscosity(rheology, T(gnf_eps))) : viscosity
    _fv3_validate_viscosity(visc, g, nuc)
    thermal = _fv3_lower_energy(energy, g, tag, T)
    _fv_has_state_temperature(visc) && thermal === nothing &&
        throw(ArgumentError("state-temperature viscosity requires an active energy model"))
    flowrate = _fv3_lower_flowrate(u_mean_target, g, tag, srcv, T)
    return FVSteadyProblem3D(g, tag, nuc, bvel, srcv, visc, nothing, thermal, flowrate)
end

"""
    fv_adapt(to, prob::FVSteadyProblem3D) -> FVSteadyProblem3D

Copy the arrays of `prob` with `to` (e.g. `CuArray`, or `Array` back to host).
"""
fv_adapt(to, p::FVSteadyProblem3D) =
    FVSteadyProblem3D(p.grid, to(p.tag), to(p.nu), to(p.bvel), to(p.src), _fv_adapt_viscosity(to, p.visc),
                      nothing, _fv_adapt_energy(to, p.energy), p.flowrate)

"""
    fv_state(prob::FVSteadyProblem3D, u, v, w, p) -> Vector

Pack cell fields (`Nx × Ny × Nz` arrays or `(x, y, z) -> value` functions at
cell centres) into a host state vector; a controlled Fx starts at the uniform
x source.
"""
function fv_state(prob::FVSteadyProblem3D, u, v, w, p)
    g = prob.grid
    U = zeros(eltype(prob.nu), fv_nunknowns(prob))
    get(f::AbstractArray, i, j, k) = f[i, j, k]
    get(f, i, j, k) = f((i - 0.5) * g.h, (j - 0.5) * g.h, (k - 0.5) * g.h)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        for (c, f) in enumerate((u, v, w, p))
            U[fv_index(g, i, j, k, c)] = get(f, i, j, k)
        end
    end
    _fv_initial_force!(U, prob)
    return U
end

"""
    fv_fields(prob::FVSteadyProblem3D, U) -> (u, v, w, p)

Unpack the cell entries into `Nx × Ny × Nz` arrays (host copies).
"""
function fv_fields(prob::FVSteadyProblem3D, U::AbstractVector)
    g = prob.grid
    length(U) == fv_nunknowns(prob) || throw(DimensionMismatch("state length does not match problem"))
    A = reshape(Array(U)[1:fv_nunknowns(g)], _fv_nfields(g), g.Nx, g.Ny, g.Nz)
    return ntuple(c -> A[c, :, :, :], 4)
end
