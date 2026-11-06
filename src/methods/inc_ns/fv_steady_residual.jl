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
#          a wall face uses the one-sided normal difference and the cell
#          gradient. The hook is _fv_face_nu / _fv_wall_nu on prob.visc.
#   Pinned cell (one per closed fluid pocket): the continuity row is replaced
#   by p_P = 0. Solid cells carry identity rows (u, v, p) = 0; a fluid face next
#   to a solid cell is a no-slip wall.
#
# Unknown ordering: k = 3*(c-1) + {1:u, 2:v, 3:p}, c = i + (j-1)*Nx.

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
    FVSteadyProblem2D(grid, tag, nu, bvel, src[, visc])

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

Build with [`fv_steady_problem`](@ref); move to a device with
[`fv_adapt`](@ref).
"""
struct FVSteadyProblem2D{G<:FVGrid2D,TA,NA,BA,SA,V}
    grid::G
    tag::TA
    nu::NA
    bvel::BA
    src::SA
    visc::V
end
FVSteadyProblem2D(grid, tag, nu, bvel, src) =
    FVSteadyProblem2D(grid, tag, nu, bvel, src, FVNewtonianViscosity())

fv_nunknowns(g::FVGrid2D) = 3 * g.Nx * g.Ny
fv_nunknowns(p::FVSteadyProblem2D) = fv_nunknowns(p.grid)

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
of the two cells, wall face = the cell value. The generalised-Newtonian model
is [`FVGNFViscosity`](@ref) (C3).
"""
struct FVNewtonianViscosity end

# face viscosity from the two cell values and the face velocity gradient
# (ux, uy, vx, vy); wall face: from the cell value and the wall-face gradient
@inline _fv_face_nu(::FVNewtonianViscosity, nuP, nuN, ux, uy, vx, vy) = (nuP + nuN) / 2
@inline _fv_wall_nu(::FVNewtonianViscosity, nuP, ux, uy, vx, vy) = nuP
# zero carrying the number type of the model's parameters (Bool: neutral)
@inline _fv_visc_zero(::FVNewtonianViscosity) = false
_fv_visc_eltype(::FVNewtonianViscosity) = Bool

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

# face values (u, v, p) of the face of cell (i, j) in direction d
@inline function _fv_face_values(U, tag, bvel, g, i, j, d)
    uP = _fv_val(U, g, i, j, 1); vP = _fv_val(U, g, i, j, 2); pP = _fv_val(U, g, i, j, 3)
    kind, ii, jj = _fv_face_kind(g, tag, i, j, d)
    zb = zero(eltype(bvel))
    if kind == _FV_F_INT
        uN = _fv_val(U, g, ii, jj, 1); vN = _fv_val(U, g, ii, jj, 2); pN = _fv_val(U, g, ii, jj, 3)
        return (uP + uN) / 2 + zb, (vP + vN) / 2 + zb, (pP + pN) / 2
    elseif kind == _FV_F_OUTLET
        return uP + zb, vP + zb, zero(pP)
    else
        uw, vw = _fv_dirichlet_velocity(g, bvel, i, j, d)
        return zero(uP) + uw, zero(vP) + vw, pP
    end
end

# Green-Gauss cell gradient: (ux, uy, vx, vy, px, py)
@inline function _fv_cell_grad(U, tag, bvel, g, i, j)
    ue, ve, pe = _fv_face_values(U, tag, bvel, g, i, j, 1)
    uw, vw, pw = _fv_face_values(U, tag, bvel, g, i, j, 2)
    un, vn, pn = _fv_face_values(U, tag, bvel, g, i, j, 3)
    us, vs, ps = _fv_face_values(U, tag, bvel, g, i, j, 4)
    ih = inv(g.h)
    return ((ue - uw) * ih, (un - us) * ih, (ve - vw) * ih, (vn - vs) * ih,
            (pe - pw) * ih, (pn - ps) * ih)
end

@inline _fv_posp(F) = (F + abs(F)) / 2
@inline _fv_negp(F) = (F - abs(F)) / 2

"""
    fv_cell_residual(U, tag, nu, bvel, src, grid, i, j[, visc]) -> (Ru, Rv, Rc)

Residual of the three equations of cell `(i, j)`. Branches depend only on the
integer geometry (tags, boundary codes), never on the values of `U`, `nu`,
`bvel` or `src`, so the function is traceable by global sparsity detectors and
differentiable by dual numbers through every branch (the upwind switch is
written with `abs`). `visc` is the viscosity model (default
[`FVNewtonianViscosity`](@ref)).
"""
@inline function fv_cell_residual(U, tag, nu, bvel, src, g, i, j,
                                  visc = FVNewtonianViscosity())
    uP = _fv_val(U, g, i, j, 1); vP = _fv_val(U, g, i, j, 2); pP = _fv_val(U, g, i, j, 3)
    h = g.h
    @inbounds nuP = nu[i, j]
    z = zero(uP) + zero(nuP) + zero(eltype(bvel)) + zero(eltype(src)) + _fv_visc_zero(visc)
    @inbounds tP = tag[i, j]
    if tP == FV_CELL_SOLID
        return z + uP, z + vP, z + pP
    end
    gP = _fv_cell_grad(U, tag, bvel, g, i, j)
    Ru = z; Rv = z; Rc = z
    for d in 1:4
        isx = d <= 2
        sx = (d == 1 || d == 3) ? 1 : -1
        kind, ii, jj = _fv_face_kind(g, tag, i, j, d)
        if kind == _FV_F_INT
            uN = _fv_val(U, g, ii, jj, 1); vN = _fv_val(U, g, ii, jj, 2); pN = _fv_val(U, g, ii, jj, 3)
            gN = _fv_cell_grad(U, tag, bvel, g, ii, jj)
            @inbounds nuN = nu[ii, jj]
            dun = (uN - uP) / h
            dvn = (vN - vP) / h
            # face velocity gradient: compact normal difference, mean of the
            # two cell gradients tangentially (read by a GNF model only)
            if isx
                nuf = _fv_face_nu(visc, nuP, nuN, sx * dun, (gP[2] + gN[2]) / 2,
                                  sx * dvn, (gP[4] + gN[4]) / 2)
            else
                nuf = _fv_face_nu(visc, nuP, nuN, (gP[1] + gN[1]) / 2, sx * dun,
                                  (gP[3] + gN[3]) / 2, sx * dvn)
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
            uw, vw = _fv_dirichlet_velocity(g, bvel, i, j, d)
            F = h * sx * (isx ? uw : vw)
            Ru += g.conv * (_fv_posp(F) * uP + _fv_negp(F) * uw)
            Rv += g.conv * (_fv_posp(F) * vP + _fv_negp(F) * vw)
            dun = (uw - uP) * 2 / h
            dvn = (vw - vP) * 2 / h
            if isx
                nuw = _fv_wall_nu(visc, nuP, sx * dun, gP[2], sx * dvn, gP[4])
                tx = 2 * nuw * dun
                ty = nuw * (dvn + sx * gP[2])
            else
                nuw = _fv_wall_nu(visc, nuP, gP[1], sx * dun, gP[3], sx * dvn)
                tx = nuw * (dun + sx * gP[3])
                ty = 2 * nuw * dvn
            end
            Ru -= h * tx; Rv -= h * ty
            pf = pP
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
                                      @Const(bvel), @Const(src), g, visc)
    i, j = @index(Global, NTuple)
    ru, rv, rc = fv_cell_residual(U, tag, nu, bvel, src, g, i, j, visc)
    b = fv_index(g, i, j, 0)
    @inbounds R[b + 1] = ru
    @inbounds R[b + 2] = rv
    @inbounds R[b + 3] = rc
end

"""
    fv_residual!(R, U, prob::FVSteadyProblem2D; workgroup = 64, sync = true) -> R

Evaluate the steady FV residual of the state `U` into `R` (both of length
`3 Nx Ny`, unknown ordering of [`fv_index`](@ref)) on the backend of `U`.
All arrays of `prob` must live on the same backend as `U`. `eltype(R)` must
hold the promoted element type of `U` and of the arrays of `prob` (e.g.
`ForwardDiff.Dual` when either carries duals).

Allocation-free on the device (the GPU test checks `CUDA.@allocated == 0`).
`sync = false` skips the final `KernelAbstractions.synchronize`, for callers
that chain kernels on the same stream (C2 assembly).
"""
function fv_residual!(R, U, prob::FVSteadyProblem2D; workgroup::Int = 64, sync::Bool = true)
    g = prob.grid
    n = fv_nunknowns(g)
    length(U) == n || throw(DimensionMismatch("length(U) = $(length(U)), expected $n"))
    length(R) == n || throw(DimensionMismatch("length(R) = $(length(R)), expected $n"))
    backend = KernelAbstractions.get_backend(U)
    _fv_residual_kernel!(backend, workgroup)(R, U, prob.tag, prob.nu, prob.bvel,
                                             prob.src, g, prob.visc; ndrange = (g.Nx, g.Ny))
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
                     _fv_visc_eltype(prob.visc))
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
    fv_steady_problem(Nx, Ny; h, nu, bc, boundary_velocity, solid, source, conv, T)
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
- `source`: `nothing`, a `(x, y) -> (s_x, s_y, s_mass)` function at cell
  centres, or a length-`3 Nx Ny` vector.
- `conv`: `true` (Navier-Stokes, default) or `false` (Stokes).
- `T`: element type of the arrays (default `Float64`).
- `rheology`: `nothing` (default; Newtonian, viscosity `nu`) or a
  generalised-Newtonian law of `src/rheology/models.jl` (`PowerLaw`,
  `CarreauYasuda`, `Cross`, `Bingham`, `HerschelBulkley`, `Newtonian`) in
  kinematic FV units; `nu` is then ignored. `gnf_eps` is the strain-rate
  regularisation `ε` of [`FVGNFViscosity`](@ref) (`γ̇_ε = √(2 S:S + ε²)`).

Pressure pins are set by [`fv_pressure_pins!`](@ref). Cell centres are at
`((i - 1/2) h, (j - 1/2) h)`.
"""
function fv_steady_problem(Nx::Integer, Ny::Integer; h::Real = 1 / Ny, nu = 1.0,
                           bc = (W = :wall, E = :wall, S = :wall, N = :wall),
                           boundary_velocity = (side, x, y) -> (0.0, 0.0),
                           solid = nothing, source = nothing, conv::Bool = true,
                           T::Type{<:Real} = Float64, rheology = nothing,
                           gnf_eps::Real = 0.0)
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
    if solid !== nothing
        size(solid) == (Nx, Ny) || throw(DimensionMismatch("solid must be $Nx × $Ny"))
        for k in eachindex(tag)
            solid[k] != 0 && (tag[k] = FV_CELL_SOLID)
        end
    end
    fv_pressure_pins!(tag, g)
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
    visc = rheology === nothing ? FVNewtonianViscosity() : FVGNFViscosity(rheology, T(gnf_eps))
    return FVSteadyProblem2D(g, tag, nuc, bvel, srcv, visc)
end

"""
    fv_adapt(to, prob::FVSteadyProblem2D) -> FVSteadyProblem2D

Copy the arrays of `prob` with the array constructor `to` (e.g. `CuArray`,
or `Array` to come back to the host). The grid is passed through.
"""
fv_adapt(to, p::FVSteadyProblem2D) =
    FVSteadyProblem2D(p.grid, to(p.tag), to(p.nu), to(p.bvel), to(p.src), p.visc)

"""
    fv_state(prob, u, v, p) -> Vector

Pack cell fields (`Nx × Ny` matrices, or `(x, y) -> value` functions evaluated
at cell centres) into a host state vector in the unknown ordering.
"""
function fv_state(prob::FVSteadyProblem2D, u, v, p)
    g = prob.grid
    T = eltype(prob.nu)
    U = zeros(T, fv_nunknowns(g))
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

Unpack a (host) state vector into three `Nx × Ny` matrices.
"""
function fv_fields(prob::FVSteadyProblem2D, U::AbstractVector)
    g = prob.grid
    A = reshape(Array(U), 3, g.Nx, g.Ny)
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
