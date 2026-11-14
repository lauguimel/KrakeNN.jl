# Steady discrete adjoint of the 3D FV problem (plan "solveur FV stationnaire
# 3D", step 4, increment I2).
#
# The adjoint layers of fv_adjoint.jl are dimension-generic (transposed solve
# with the Newton factors, parameter duals through the residual kernel,
# lambda-dot reductions, objective duals on the host); this file adds the
# dimension-specific methods only:
#   - problem plumbing: _fv_with, _fv_host_problem, law-model validation;
#   - parameter maps: FVSideVelocity on six sides, FVUniformSource with the
#     4th (mass) equation, the per-cell viscosity colouring and its gather,
#     FVLawParameters (buffers, problem rebuild, per-face law capture with six
#     slots, then the shared pullback/reduction kernels);
#   - objectives: 3D flow rate, the solved force Fx, and FVWallForce on 3D
#     walls (legacy kind, explicit dependence on the viscosity law and wall
#     velocities through host duals).
# The composite PIV objective is in fv_piv_objective.jl.

_fv_nflow(p::FVSteadyProblem3D) = 4

_fv_with(p::FVSteadyProblem3D; nu = p.nu, bvel = p.bvel, src = p.src, visc = p.visc,
         immersed = p.immersed, energy = p.energy, flowrate = p.flowrate) =
    FVSteadyProblem3D(p.grid, p.tag, nu, bvel, src, visc, immersed, energy, flowrate)

_fv_host_problem(p::FVSteadyProblem3D) = p.nu isa Array ? p : fv_adapt(Array, p)

# fv_set_params! validates the mapped viscosity model against the grid
_fv_validate_viscosity(v::FVLawViscosity, g::FVGrid3D, nu) = _fv3_validate_viscosity(v, g, nu)
_fv_validate_viscosity(v::FVGNFViscosity, g::FVGrid3D, nu) = _fv3_validate_viscosity(v, g, nu)

# ------------------------------------------------------------------ side velocity
"""
    FVSideVelocity(prob::FVSteadyProblem3D, side::Symbol; shape = nothing)

3D form: `side` in `:E, :W, :N, :S, :T, :B` (a wall or inlet side); `shape` is
`(x, y, z) -> (u, v, w)` at the boundary face centres. Default: unit velocity
along the first tangential axis (`(0, 1, 0)` on `:E`/`:W`, `(1, 0, 0)` on the
other sides), i.e. a lid speed.
"""
function FVSideVelocity(prob::FVSteadyProblem3D, side::Symbol; shape = nothing)
    d = findfirst(==(side), _FV3_SIDES)
    d === nothing && throw(ArgumentError("side must be one of $(_FV3_SIDES), got :$side"))
    g = prob.grid
    L = max(g.Nx, g.Ny, g.Nz)
    keep = ones(Float64, 3, L, L, 6); keep[:, :, :, d] .= 0
    shp = zeros(Float64, 3, L, L, 6)
    h = Float64(g.h)
    f = shape === nothing ? (d <= 2 ? ((x, y, z) -> (0.0, 1.0, 0.0)) : ((x, y, z) -> (1.0, 0.0, 0.0))) : shape
    dims = (g.Nx, g.Ny, g.Nz)
    n = _fv3_axis(d)
    t1, t2 = n == 1 ? (2, 3) : n == 2 ? (1, 3) : (1, 2)
    x = zeros(3)
    for l2 in 1:dims[t2], l1 in 1:dims[t1]
        x[n] = isodd(d) ? dims[n] * h : 0.0
        x[t1] = (l1 - 0.5) * h; x[t2] = (l2 - 0.5) * h
        uvw = f(x[1], x[2], x[3])
        for c in 1:3
            shp[c, l1, l2, d] = uvw[c]
        end
    end
    return FVSideVelocity(d, keep, shp)
end

# ------------------------------------------------------------------ viscosity field
"""
    fv_nu_coloring(prob::FVSteadyProblem3D) -> (colors, ncolors)

3D column colouring of `∂R/∂ν`: the viscosity of fluid cell `c` enters the
rows of `c` and of its (up to six) interior face neighbours.
"""
function fv_nu_coloring(prob::FVSteadyProblem3D)
    g = prob.grid; tag = Array(prob.tag)
    nc = fv_ncells(prob)
    I = Int[]; J = Int[]
    cell(i, j, k) = i + (j - 1) * g.Nx + (k - 1) * g.Nx * g.Ny
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        c = cell(i, j, k)
        push!(I, c); push!(J, c)
        tag[i, j, k] == FV_CELL_SOLID && continue
        for d in 1:6
            kind, ii, jj, kk = _fv_face_kind(g, tag, i, j, k, d)
            kind == _FV_F_INT && (push!(I, c); push!(J, cell(ii, jj, kk)))
        end
    end
    return fv_jacobian_coloring(sparse(I, J, trues(length(I)), nc, nc, |))
end

@inline function _fv3_rows_dot(λ, Rd, g, i, j, k, col)
    b = fv_index(g, i, j, k, 0)
    acc = 0.0
    @inbounds for slot in 1:_fv_nfields(g)
        acc += λ[b + slot] * ForwardDiff.partials(Rd[b + slot])[col]
    end
    return acc
end

# s[c] = −λᵀ ∂R/∂ν_c: rows of c and of its distinct interior face neighbours
@kernel function _fv3_nu_gather_kernel!(s, @Const(λ), @Const(Rd), @Const(tag), @Const(colors),
                                        g, offset, ::Val{N}) where {N}
    i, j, k = @index(Global, NTuple)
    c = i + (j - 1) * g.Nx + (k - 1) * g.Nx * g.Ny
    @inbounds col = colors[c] - offset
    if 1 <= col <= N
        acc = 0.0
        @inbounds if tag[i, j, k] != FV_CELL_SOLID
            acc = _fv3_rows_dot(λ, Rd, g, i, j, k, col)
            for d in 1:6
                kind, ii, jj, kk = _fv_face_kind(g, tag, i, j, k, d)
                fresh = kind == _FV_F_INT && !(ii == i && jj == j && kk == k)
                for e in 1:(d - 1)
                    k2, i2, j2, l2 = _fv_face_kind(g, tag, i, j, k, e)
                    fresh &= !(k2 == _FV_F_INT && i2 == ii && j2 == jj && l2 == kk)
                end
                if fresh
                    acc += _fv3_rows_dot(λ, Rd, g, ii, jj, kk, col)
                end
            end
        end
        @inbounds s[i, j, k] = -acc
    end
end

_fv_nu_gather!(backend, ws, prob::FVSteadyProblem3D, off, ::Val{N}) where {N} =
    _fv3_nu_gather_kernel!(backend, 64)(ws.s, ws.λ, ws.Rd, prob.tag, ws.nucolors, prob.grid, off,
                                        Val(N); ndrange = (prob.grid.Nx, prob.grid.Ny, prob.grid.Nz))

# ------------------------------------------------------------------ law parameters
const _FVLawProblem3D = FVSteadyProblem3D{<:FVGrid3D,<:Any,<:Any,<:Any,<:Any,<:FVLawViscosity}

_fv_law_problem_slots(p::FVSteadyProblem3D) = Val(6)

function _fv_param_buffers(m::FVLawParameters, p::_FVLawProblem3D, ::Type{T}) where {T}
    m.nparams == law_nparams(p.visc.law) ||
        throw(DimensionMismatch("FVLawParameters count does not match the law"))
    return similar(p.visc.theta, T)
end
function _fv_param_problem!(theta, ::FVLawParameters, p::_FVLawProblem3D, values)
    copyto!(theta, values)
    v = p.visc
    return _fv_with(p; visc = FVLawViscosity(v.law, theta, v.fields, v.eps))
end

# One residual traversal per cell; the capture wrapper records the law inputs
# of each of the six faces and seeds that face's viscosity (slot d), so the
# partials of the cell rows are ∂r_P/∂ν_{P,d} at fixed U.
@kernel function _fv3_law_capture_kernel!(inputs, sensitivity, @Const(U), @Const(lambda),
                                          @Const(tag), @Const(nu), @Const(bvel), @Const(src),
                                          g, visc, ::Val{N}) where {N}
    i, j, k = @index(Global, NTuple)
    c = i + (j - 1) * g.Nx + (k - 1) * g.Nx * g.Ny
    @inbounds for d in 1:N, a in 1:law_ninputs(visc.law)
        inputs[c, d, a] = 0.0
    end
    capture = _FVLawCapture(visc.law, inputs, c, 0, Val(N))
    v = FVLawViscosity(capture, visc.theta, visc.fields, visc.eps)
    ru, rv, rw, rc = _fv3_cell_residual(U, tag, nu, bvel, src, g, i, j, k, v)
    pu, pv = ForwardDiff.partials(ru), ForwardDiff.partials(rv)
    pw, pc = ForwardDiff.partials(rw), ForwardDiff.partials(rc)
    b = fv_index(g, i, j, k, 0)
    @inbounds for d in 1:N
        sensitivity[c, d] = -(lambda[b + 1] * pu[d] + lambda[b + 2] * pv[d] +
                              lambda[b + 3] * pw[d] + lambda[b + 4] * pc[d])
    end
end

function _fv_law_capture!(backend, w, U, λ, prob::FVSteadyProblem3D, v)
    g = prob.grid
    _fv3_law_capture_kernel!(backend, 64)(w.inputs, w.sensitivity, U, λ, prob.tag, prob.nu,
        prob.bvel, prob.src, g, v, Val(6); ndrange = (g.Nx, g.Ny, g.Nz))
    return nothing
end

# ------------------------------------------------------------------ objectives
"""
    fv_flow_rate(prob::FVSteadyProblem3D, i) -> FVLinearFunctional

Volume flow rate through the cell slab `i` (x = const): `h² Σ_{j,k} u(i, j, k)`
over its fluid cells.
"""
function fv_flow_rate(prob::FVSteadyProblem3D, i::Integer)
    g = prob.grid; tag = Array(prob.tag)
    idx = [fv_index(g, i, j, k, 1) for k in 1:g.Nz for j in 1:g.Ny if tag[i, j, k] != FV_CELL_SOLID]
    return FVLinearFunctional(idx, fill(Float64(g.h)^2, length(idx)))
end

# is face d of fluid cell (i, j, k) a wall face of `region`?
function _fv3_wall_face(g, tag, i, j, k, d, region)
    kind, _, _, _ = _fv_face_kind(g, tag, i, j, k, d)
    kind == _FV_F_DIRICHLET || return false
    ii, jj, kk = _fv3_step(i, j, k, d)
    inside = 1 <= ii <= g.Nx && 1 <= jj <= g.Ny && 1 <= kk <= g.Nz
    solid = inside || _fv_side_bc(g, d) == FV_BC_PERIODIC   # periodic wrap onto a solid cell
    return region === :solid ? solid : (!solid && _FV3_SIDES[d] === region)
end

function _fv3_check_wall_objective(f::FVWallForce)
    f.kind === :legacy && f.wall == 0 ||
        throw(ArgumentError("3D wall objectives support kind = :legacy and wall = 0 only"))
    return nothing
end

function _fv3_wall_faces(f::FVWallForce, prob)
    _fv3_check_wall_objective(f)
    g = prob.grid; tag = Array(prob.tag)
    faces = NTuple{4,Int}[]
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        tag[i, j, k] == FV_CELL_SOLID && continue
        for d in 1:6
            _fv3_wall_face(g, tag, i, j, k, d, f.region) && push!(faces, (i, j, k, d))
        end
    end
    return faces
end

# Force of the fluid on one wall face, component f.dir: the wall momentum flux
# of the residual without advection, h² (p_f n − ν_w (∇u + ∇uᵀ)·n), with the
# residual's own wall strain and viscosity hooks.
@inline function _fv3_wall_force_face(f::FVWallForce, U, prob, i, j, k, d)
    g = prob.grid; tag = prob.tag; bvel = prob.bvel
    h = g.h; h2 = h * h
    n = _fv3_axis(d); sx = _fv3_sign(d)
    uP, vP, wP, pP = _fv3_cell(U, g, i, j, k)
    gP = _fv3_cell_grad(U, tag, bvel, g, i, j, k)
    uw, vw, ww = _fv3_dirichlet_velocity(g, bvel, i, j, k, d)
    du = (uw - uP) * 2 / h
    dv = (vw - vP) * 2 / h
    dw = (ww - wP) * 2 / h
    Gt = _fv3_first9(gP)
    @inbounds nuP = prob.nu[i, j, k]
    # bound to the state temperature when the law reads it (identity otherwise)
    visc = _fv_bind_viscosity(prob.visc, U, g)
    nuw = _fv_wall_nu3(visc, nuP, _fv3_face_grad(Gt, n, sx, du, dv, dw), i, j, k)
    t = _fv3_traction(nuw, n, sx, du, dv, dw, Gt)
    pf = pP
    if f.pressure === :linear
        dop = isodd(d) ? d + 1 : d - 1
        kind, ii, jj, kk = _fv_face_kind(g, tag, i, j, k, dop)
        kind == _FV_F_INT && (pf = pP + (pP - _fv3_cell(U, g, ii, jj, kk)[4]) / 2)
    end
    return -h2 * t[f.dir] + (n == f.dir ? h2 * sx * pf : zero(pf))
end

function fv_objective(f::FVWallForce, U, prob::FVSteadyProblem3D)
    T = promote_type(eltype(U), eltype(prob.nu), eltype(prob.bvel), _fv_visc_eltype(prob.visc))
    F = zero(T)
    for (i, j, k, d) in _fv3_wall_faces(f, prob)
        F += _fv3_wall_force_face(f, U, prob, i, j, k, d)
    end
    return F
end

function fv_objective_support(f::FVWallForce, prob::FVSteadyProblem3D)
    g = prob.grid; tag = Array(prob.tag)
    cells = Set{Int}()
    cell(i, j, k) = i + (j - 1) * g.Nx + (k - 1) * g.Nx * g.Ny
    for (i, j, k, d) in _fv3_wall_faces(f, prob)
        push!(cells, cell(i, j, k))
        for e in 1:6
            kind, ii, jj, kk = _fv_face_kind(g, tag, i, j, k, e)
            kind == _FV_F_INT && push!(cells, cell(ii, jj, kk))
        end
    end
    return sort!([_fv_cell_index(g, c, comp) for c in cells for comp in 1:_fv_nfields(g)])
end

# State and parameter derivatives of the 3D wall force: chunked host duals on
# the support (tiny stencils; not on the hot path of the PIV objective).
_fv_objective_grad_u(obj::FVWallForce, Uh::Vector{Float64}, hprob::FVSteadyProblem3D; chunk::Int = 8) =
    invoke(_fv_objective_grad_u, Tuple{FVObjective,Vector{Float64},Any}, obj, Uh, hprob; chunk)
_fv_objective_grad_theta(obj::FVWallObjective, m::FVViscosityField, Uh, hprob::FVSteadyProblem3D, θ;
                         chunk::Int = 8) =
    invoke(_fv_objective_grad_theta, Tuple{FVObjective,FVViscosityField,Any,Any,Any},
           obj, m, Uh, hprob, θ; chunk)
_fv_objective_grad_theta(obj::FVWallObjective, m::FVUniformViscosity, Uh, hprob::FVSteadyProblem3D, θ;
                         chunk::Int = 8) =
    invoke(_fv_objective_grad_theta, Tuple{FVObjective,FVParameterMap,Any,Any,Any},
           obj, m, Uh, hprob, θ; chunk)
