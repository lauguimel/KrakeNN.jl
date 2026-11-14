# Wall slip laws for the steady FV residual.
#
# Convention (shared with the LBM slip and RheoTool's navierSlip, m = 1): at a
# slip wall the tangential velocity of the fluid relative to the wall is
#   s = f(τ_w; θ),
# where τ_w is the tangential shear stress the fluid exerts on the wall, signed
# along the same tangent as s. Navier slip is f(τ) = b τ (b = mobility, the
# slip length is λ = b μ). A law must be odd and non-decreasing in τ: the local
# equation g(s) = s - f(τ(s)) = 0 is then strictly increasing in s and its root
# lies between 0 and the no-slip-free velocity difference a (see below).
#
# Discretisation. At one wall face of cell P the wall shear rate is written
# γ = κ (a - s), with
# - a grid-aligned face: κ = 2/h, a = u_t(P) - u_t,wall (the linear ghost of
#   the no-slip closure, kept unchanged);
# - an immersed (cut) face: the wall derivative of the ghost interpolant along
#   the axial link e (linear: P, wall; quadratic: O, P, wall), divided by
#   e·n to turn it into a normal derivative. The tangential derivative of u_t
#   along the wall is neglected, which is exact for flow parallel to a straight
#   wall and first order where the slip varies along the wall.
# τ_w = ν_w(γ) γ with the residual's own wall viscosity hook (simple shear),
# then s solves s = f(ν_w(γ) γ) by a safeguarded Newton iteration. Every
# iteration runs on the full number type, so the converged s carries the exact
# implicit-function derivatives for every outer dual (Jacobian columns,
# parameter maps, nested adjoint duals). The wall velocity of the face becomes
# u_wall + s t; the existing ghost, convection and mass closures then read it.
# On grid-aligned faces the viscous traction is ν_w ∂u_t/∂n tangentially (the
# cross term is dropped so the traction equals the τ_w of the slip law) and
# 2 ν_w ∂u_n/∂n normally.

"""
    FVSlipLaw

Supertype of the wall slip laws of [`FVSlipWalls`](@ref). A law `L` defines

- `fv_slip_velocity(law, τ)`: the slip velocity for the wall shear stress `τ`
  (odd, non-decreasing, generic in the number type: it is evaluated on nested
  `ForwardDiff.Dual`s and on the GPU, so the law must be isbits);
- `fv_slip_parameters(law) -> Vector`: its parameters θ;
- `fv_slip_law(law, θ)`: the same law with parameters θ (any number type).

The number type of a law is the promotion of its numeric fields; a law with
non-numeric fields (e.g. a network) overloads `KrakeNN._fv_slip_law_eltype`.
"""
abstract type FVSlipLaw end

"""
    FVNavierSlip(b)

Navier slip `s = b τ_w` with mobility `b` (slip length `b μ`).
"""
struct FVNavierSlip{T} <: FVSlipLaw
    b::T
end
@inline fv_slip_velocity(l::FVNavierSlip, τ) = l.b * τ
fv_slip_parameters(l::FVNavierSlip) = [l.b]
fv_slip_law(::FVNavierSlip, θ) = FVNavierSlip(θ[1])

"""
    fv_slip_velocity(law, τ)

Slip velocity of `law` for the wall shear stress `τ` (see [`FVSlipLaw`](@ref)).
"""
function fv_slip_velocity end
"""
    fv_slip_parameters(law) -> Vector

Parameters θ of a slip law (see [`FVSlipLaw`](@ref)).
"""
function fv_slip_parameters end
"""
    fv_slip_law(law, θ)

The slip law of the same family as `law` with parameters `θ`.
"""
function fv_slip_law end

_fv_number_promote() = Bool
_fv_number_promote(T, rest...) = promote_type(T <: Number ? T : Bool, _fv_number_promote(rest...))
_fv_slip_law_eltype(::Type{L}) where {L} = _fv_number_promote(fieldtypes(L)...)
@inline _fv_slip_law_zero(l) = zero(_fv_slip_law_eltype(typeof(l)))

"""
    FVSlipWalls(law; sides = (), solid = false, immersed = false, maxit = 50)

Slip walls of an [`FVSteadyProblem2D`](@ref) (keyword `slip` of
[`fv_steady_problem`](@ref)). `law` is an [`FVSlipLaw`](@ref). `sides` lists the
domain sides (`:E`, `:W`, `:N`, `:S`, each a `:wall` side) that slip; `solid`
makes the faces against masked solid cells slip; `immersed` makes every cut
face of the immersed geometry slip (linear or quadratic ghost closure; the
cut normals come from the immersed surface metadata). All other walls stay
no-slip, and `slip = nothing` leaves the residual bit-identical. `maxit` bounds
the local Newton iteration. Not available with the energy equation.
"""
struct FVSlipWalls{L,A}
    law::L
    sides::NTuple{4,Bool}       # E, W, N, S (face directions d = 1..4)
    solid::Bool
    immersed::Bool
    maxit::Int
    normal::A                   # nothing, or 2 × 4 × Nx × Ny cut normals (fluid → solid)
end
function FVSlipWalls(law::FVSlipLaw; sides = (), solid::Bool = false, immersed::Bool = false,
                     maxit::Integer = 50)
    s = Symbol.(collect(sides))
    all(in((:E, :W, :N, :S)), s) || throw(ArgumentError("slip sides must be among :E, :W, :N, :S"))
    maxit >= 1 || throw(ArgumentError("maxit must be positive"))
    return FVSlipWalls(law, (:E in s, :W in s, :N in s, :S in s), solid, immersed, Int(maxit), nothing)
end

KernelAbstractions.Adapt.adapt_structure(to, w::FVSlipWalls) =
    FVSlipWalls(w.law, w.sides, w.solid, w.immersed, w.maxit,
                KernelAbstractions.Adapt.adapt(to, w.normal))
_fv_adapt_slip(to, ::Nothing) = nothing
_fv_adapt_slip(to, w::FVSlipWalls) =
    FVSlipWalls(w.law, w.sides, w.solid, w.immersed, w.maxit, w.normal === nothing ? nothing : to(w.normal))

_fv_slip_eltype(::Nothing) = Bool
_fv_slip_eltype(w::FVSlipWalls) =
    promote_type(_fv_slip_law_eltype(typeof(w.law)), w.normal === nothing ? Bool : eltype(w.normal))

_fv_lower_slip(::Nothing, g, immersed, thermal, visc) = nothing
function _fv_lower_slip(w::FVSlipWalls, g, immersed, thermal, visc)
    thermal === nothing || throw(ArgumentError("slip walls do not support the energy equation yet"))
    for (d, name) in enumerate((:E, :W, :N, :S))
        w.sides[d] && _fv_side_bc(g, d) != FV_BC_WALL &&
            throw(ArgumentError("slip side $name must be a :wall side"))
    end
    normal = nothing
    if w.immersed
        immersed === nothing && throw(ArgumentError("immersed slip requires an immersed geometry"))
        immersed.closure.ghost == 0 &&
            throw(ArgumentError("immersed slip requires the :linear or :quadratic ghost closure"))
        immersed.surface === nothing &&
            throw(ArgumentError("immersed slip requires the immersed surface metadata (cut normals)"))
        v = Array(immersed.surface.values)
        normal = v[3:4, :, :, :]
    end
    return FVSlipWalls(w.law, w.sides, w.solid, w.immersed, w.maxit, normal)
end

# ------------------------------------------------------------------ kernel side
# Slip walls bound to the cell viscosity array and the (state-bound) viscosity
# model of the current residual evaluation.
struct _FVSlipBound{W,N,V}
    walls::W
    nu::N
    visc::V
end
@inline _fv_bind_slip(::Nothing, nu, visc) = nothing
@inline _fv_bind_slip(w::FVSlipWalls, nu, visc) = _FVSlipBound(w, nu, visc)
@inline _fv_bind_slip(b::_FVSlipBound, nu, visc) = b

@inline _fv_slip_promote(z, ::Nothing) = z
@inline _fv_slip_promote(z, b::_FVSlipBound) =
    z + zero(eltype(b.nu)) + _fv_visc_zero(b.visc) + _fv_slip_law_zero(b.walls.law) +
    _fv_array_zero(b.walls.normal)

@inline _fv_slip_face(::Nothing, g, i, j, d, cut) = false
@inline function _fv_slip_face(b::_FVSlipBound, g, i, j, d, cut)
    w = b.walls
    cut && return w.immersed
    ii = i + (d == 1) - (d == 2)
    jj = j + (d == 3) - (d == 4)
    (ii < 1 || ii > g.Nx || jj < 1 || jj > g.Ny) && return @inbounds w.sides[d]
    return w.solid
end

# Innermost derivative of the local slip equation; outside every other dual.
struct _FVSlipTag end
ForwardDiff.:(≺)(::Type{T}, ::Type{_FVSlipTag}) where {T} = true
ForwardDiff.:(≺)(::Type{_FVSlipTag}, ::Type{T}) where {T} = false
ForwardDiff.:(≺)(::Type{_FVSlipTag}, ::Type{_FVSlipTag}) = false
ForwardDiff.:(≺)(::Type{_FVSurfaceTangentTag}, ::Type{_FVSlipTag}) = true
ForwardDiff.:(≺)(::Type{_FVSlipTag}, ::Type{_FVSurfaceTangentTag}) = false

@inline _fv_slip_wall_tau(visc, nuf, γ, i, j) =
    _fv_wall_nu(visc, nuf, zero(γ), γ, zero(γ), zero(γ), i, j) * γ

"""
Root s of s = f(ν_w(γ) γ), γ = κ (a - s), bracketed by 0 and a. Newton with
the exact derivative (nested dual), bisection when a step leaves the bracket.
"""
@inline function _fv_slip_solve(b::_FVSlipBound, a0, κ0, nuf0, i, j)
    law = b.walls.law; visc = b.visc
    z = zero(a0) + zero(κ0) + zero(nuf0) + _fv_slip_law_zero(law) + _fv_visc_zero(visc)
    a = a0 + z; κ = κ0 + z; nuf = nuf0 + z
    ap = _fv_ib_primal(a)
    tol = 8 * eps(typeof(float(ap))) * max(abs(ap), floatmin(typeof(float(ap))))
    lo = ap < 0 ? ap : zero(ap)
    hi = ap < 0 ? zero(ap) : ap
    s = z
    for _ in 1:b.walls.maxit
        sd = ForwardDiff.Dual{_FVSlipTag}(s, one(s))
        γ = κ * (a - sd)
        gd = sd - fv_slip_velocity(law, _fv_slip_wall_tau(visc, nuf, γ, i, j))
        gv = ForwardDiff.value(gd)
        dg = ForwardDiff.partials(gd, 1)
        sp = _fv_ib_primal(s)
        gp = _fv_ib_primal(gv)
        gp > 0 ? (hi = sp) : (lo = sp)
        dp = _fv_ib_primal(dg)
        step = gv / dg
        snew = s - step
        np = _fv_ib_primal(snew)
        if dp > 0 && lo <= np <= hi && isfinite(np)
            s = snew
            abs(_fv_ib_primal(step)) <= tol && break
        else
            s = zero(s) + (lo + hi) / 2
        end
    end
    return s
end

# Grid-aligned slip face (domain side or masked solid): tangential component
# u (N/S faces) or v (E/W faces), linear ghost, κ = 2/h.
@inline function _fv_slip_aligned(b::_FVSlipBound, g, i, j, d, uP, vP, uw, vw)
    @inbounds nuf = b.nu[i, j]
    κ = 2 / g.h
    if d <= 2
        s = _fv_slip_solve(b, vP - vw, κ, nuf, i, j)
        return uw + zero(s), vw + s
    end
    s = _fv_slip_solve(b, uP - uw, κ, nuf, i, j)
    return uw + s, vw + zero(s)
end

# Immersed cut face: wall slope of the ghost interpolant along the axial link.
@inline function _fv_slip_cut(b::_FVSlipBound, g, i, j, d, q, ghost, uO, vO, uP, vP, uw, vw)
    @inbounds nx = b.walls.normal[1, d, i, j]
    @inbounds ny = b.walls.normal[2, d, i, j]
    tx, ty = -ny, nx
    if ghost == 1
        cO = zero(q); cP = -1 / q; cw = 1 / q
    else
        cO = q / (1 + q); cP = -(1 + q) / q; cw = (2q + 1) / (q * (1 + q))
    end
    en = d == 1 ? nx : d == 2 ? -nx : d == 3 ? ny : -ny
    A = tx * (cO * uO + cP * uP + cw * uw) + ty * (cO * vO + cP * vP + cw * vw)
    κ = cw / (g.h * en)
    oi = i - (d == 1) + (d == 2)
    oj = j - (d == 3) + (d == 4)
    @inbounds nuP = b.nu[i, j]
    @inbounds nuf = nuP + (nuP - b.nu[oi, oj]) / 2
    s = _fv_slip_solve(b, -A / cw, κ, nuf, i, j)
    return uw + s * tx, vw + s * ty
end

# Traction of a grid-aligned slip face: tangential ν_w ∂u_t/∂n (= τ_w of the
# law), normal 2 ν_w ∂u_n/∂n, ν_w at the wall shear rate.
@inline function _fv_slip_aligned_stress(visc, nuf, i, j, d, dun, dvn)
    if d <= 2
        nuw = _fv_wall_nu(visc, nuf, zero(dvn), dvn, zero(dvn), zero(dvn), i, j)
        return 2 * nuw * dun, nuw * dvn
    end
    nuw = _fv_wall_nu(visc, nuf, zero(dun), dun, zero(dun), zero(dun), i, j)
    return nuw * dun, 2 * nuw * dvn
end

# ------------------------------------------------------------------ parameters
"""
    fv_with_slip_parameters(prob, θ) -> FVSteadyProblem2D

`prob` with the slip law rebuilt from `θ` (`fv_slip_law(prob.slip.law, θ)`),
for `FVProblemMap(fv_with_slip_parameters, length(fv_slip_parameters(law)))`
with the steady adjoint and [`fv_set_params!`](@ref).
"""
function fv_with_slip_parameters(p::FVSteadyProblem2D, θ::AbstractVector)
    w = p.slip
    w === nothing && throw(ArgumentError("the problem has no slip walls"))
    law = fv_slip_law(w.law, θ)
    return _fv_with(p; slip = FVSlipWalls(law, w.sides, w.solid, w.immersed, w.maxit, w.normal))
end

# ------------------------------------------------------------------ Jacobian pattern
# A cut slip face of cell C couples u and v of C and of its opposite cell O
# through the tangent. Its value reaches the rows within two cells of C (wall
# flux, cell gradients, Rhie–Chow, extrapolated wall gradients): add (u, v) of
# C and O to every row of that 5 × 5 box. Grid-aligned slip reads only the
# tangential component of P, already in the stencil.
function _fv_slip_pattern!(I, J, prob)
    w = prob.slip
    (w === nothing || !w.immersed || prob.immersed === nothing) && return
    g = prob.grid
    qcut = Array(prob.immersed.qcut)
    tag = Array(prob.tag)
    xper = g.bcE == FV_BC_PERIODIC; yper = g.bcN == FV_BC_PERIODIC
    cu = _fv_velocity_slot(g.layout, 1); cv = _fv_velocity_slot(g.layout, 2)
    for j in 1:g.Ny, i in 1:g.Nx, d in 1:4
        tag[i, j] == FV_CELL_SOLID && continue
        _fv_immersed_face(g, qcut, i, j, d) || continue
        ii, jj = i + (d == 1) - (d == 2), j + (d == 3) - (d == 4)
        tag[ii, jj] == FV_CELL_SOLID || continue
        oi, oj = i - (d == 1) + (d == 2), j - (d == 3) + (d == 4)
        cols = (fv_index(g, i, j, cu), fv_index(g, i, j, cv),
                fv_index(g, oi, oj, cu), fv_index(g, oi, oj, cv))
        for dj in -2:2, di in -2:2
            x, y = i + di, j + dj
            if x < 1 || x > g.Nx
                xper || continue
                x = mod1(x, g.Nx)
            end
            if y < 1 || y > g.Ny
                yper || continue
                y = mod1(y, g.Ny)
            end
            for e in 1:3, c in cols
                push!(I, fv_index(g, x, y, e)); push!(J, c)
            end
        end
    end
    return
end

# ------------------------------------------------------------------ M3 pair rows
# With immersed slip the M3 pair rows are no longer affine: the face mass flux
# reads the slipped wall velocity. Lowering widens each pair row to u and v at
# P and O of every face; each Jacobian assembly refreshes the coefficients at
# the current state (one work item per pair, its faces in sequence).
function _fv_slip_widen_mass(immersed, w, g)
    (w === nothing || !w.immersed || immersed === nothing) && return immersed
    m = _fv_mass(immersed)
    m === nothing && return immersed
    faceptr = Array(m.faceptr); faces = Array(m.faces)
    rowptr = Array(m.rowptr); rowcol = Array(m.rowcol); rowval = Array(m.rowval)
    V = eltype(rowval)
    cu = _fv_velocity_slot(g.layout, 1); cv = _fv_velocity_slot(g.layout, 2)
    nptr = Int32[1]; ncol = Int32[]; nval = V[]
    for k in 1:length(faceptr)-1
        coefficients = Dict{Int,V}()
        for q in rowptr[k]:rowptr[k+1]-1
            coefficients[rowcol[q]] = rowval[q]
        end
        for l in faceptr[k]:faceptr[k+1]-1
            d, i, j = faces[1, l], faces[2, l], faces[3, l]
            oi, oj = i - (d == 1) + (d == 2), j - (d == 3) + (d == 4)
            for c in (fv_index(g, i, j, cu), fv_index(g, i, j, cv),
                      fv_index(g, oi, oj, cu), fv_index(g, oi, oj, cv))
                coefficients[c] = get(coefficients, c, zero(V))
            end
        end
        for c in sort!(collect(keys(coefficients)))
            push!(ncol, c); push!(nval, coefficients[c])
        end
        push!(nptr, length(ncol) + 1)
    end
    widened = FVImmersedMass(m.face_pair, m.body, m.region, m.Q, m.A, m.quadrature_error,
        m.region_open, m.faceptr, m.faces, typeof(m.rowptr)(nptr), typeof(m.rowcol)(ncol),
        typeof(m.rowval)(nval))
    return FVImmersedBoundary2D(immersed.qcut, immersed.wvel, immersed.tag, immersed.h,
                                immersed.closure, immersed.surface, widened)
end

struct _FVSlipRowTag end

@inline function _fv_slip_face_flux(sb, g, i, j, d, q, closure, uw, vw, uO, vO, uP, vP)
    uw2, vw2 = _fv_slip_cut(sb, g, i, j, d, q, closure.ghost, uO, vO, uP, vP, uw, vw)
    uG = _fv_ghost_value(uO, uP, uw2, q, closure.ghost)
    vG = _fv_ghost_value(vO, vP, vw2, q, closure.ghost)
    return g.h * ((d == 1 || d == 3) ? 1 : -1) * (d <= 2 ? (uP + uG) / 2 : (vP + vG) / 2)
end

@kernel function _fv_slip_mass_rowval_kernel!(rowval, @Const(rowptr), @Const(rowcol),
        @Const(U), g, @Const(qcut), @Const(wvel), closure, @Const(faceptr), @Const(faces),
        @Const(A), @Const(nu), visc, slip)
    k = @index(Global)
    n = fv_nunknowns(g)
    sb = _fv_bind_slip(slip, nu, _fv_bind_viscosity(visc, U, g))
    @inbounds begin
        lo0 = rowptr[k]; hi0 = rowptr[k+1] - 1
        for q in lo0:hi0
            rowval[q] = rowcol[q] == n + k ? -A[k] : zero(eltype(rowval))
        end
        for l in faceptr[k]:faceptr[k+1]-1
            d, i, j = faces[1, l], faces[2, l], faces[3, l]
            oi = i - (d == 1) + (d == 2)
            oj = j - (d == 3) + (d == 4)
            cu = _fv_velocity_slot(g.layout, 1); cv = _fv_velocity_slot(g.layout, 2)
            E = eltype(U); o = one(E); z = zero(E)
            T = ForwardDiff.Dual{_FVSlipRowTag,E,4}
            uP = T(U[fv_index(g, i, j, cu)], ForwardDiff.Partials((o, z, z, z)))
            vP = T(U[fv_index(g, i, j, cv)], ForwardDiff.Partials((z, o, z, z)))
            uO = T(U[fv_index(g, oi, oj, cu)], ForwardDiff.Partials((z, z, o, z)))
            vO = T(U[fv_index(g, oi, oj, cv)], ForwardDiff.Partials((z, z, z, o)))
            F = _fv_slip_face_flux(sb, g, i, j, d, qcut[d, i, j], closure,
                                   wvel[1, d, i, j], wvel[2, d, i, j], uO, vO, uP, vP)
            cols = (fv_index(g, i, j, cu), fv_index(g, i, j, cv),
                    fv_index(g, oi, oj, cu), fv_index(g, oi, oj, cv))
            for c in 1:4
                col = cols[c]
                a = lo0; b = hi0
                while a <= b
                    mid = (a + b) >>> 1
                    if rowcol[mid] < col
                        a = mid + 1
                    elseif rowcol[mid] > col
                        b = mid - 1
                    else
                        rowval[mid] += ForwardDiff.partials(F)[c]
                        break
                    end
                end
            end
        end
    end
end

function _fv_slip_mass_rowval!(m, U, prob, backend)
    w = prob.slip
    (w === nothing || !w.immersed) && return
    b = prob.immersed
    _fv_slip_mass_rowval_kernel!(backend, 1)(m.rowval, m.rowptr, m.rowcol, U, prob.grid,
        b.qcut, b.wvel, b.closure, m.faceptr, m.faces, m.A, prob.nu, prob.visc, w;
        ndrange = length(m.Q))
    return
end
