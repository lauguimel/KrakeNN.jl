# M3: one affine compatibility equation per (whole solid body, fluid region).
# Topology is host-only; all residual and Jacobian evaluation uses lowered arrays.

struct _FVWallFluxQuadrature{V}
    Q::V
    error::V
end

struct FVImmersedMass{I,F,V,B,O}
    face_pair::I
    body::B
    region::B
    Q::V
    A::V
    quadrature_error::V
    region_open::O
    faceptr::B
    faces::F
    rowptr::B
    rowcol::B
    rowval::V
end

_fv_mass(::Nothing) = nothing
_fv_mass(b) = b.mass isa FVImmersedMass ? b.mass : nothing
function _fv_npairs(b)
    b === nothing && return 0
    if b.closure.mass == 3 && !(b.mass isa FVImmersedMass)
        throw(ArgumentError("M3 geometry must be lowered by fv_steady_problem with its physical wall_flux before constructing a positional problem"))
    end
    return _fv_mass(b) === nothing ? 0 : length(b.mass.Q)
end
_fv_face_pair(b) = _fv_mass(b) === nothing ? nothing : b.mass.face_pair
_fv_mass_eltype(::Nothing) = Bool
_fv_mass_eltype(m::Union{FVImmersedMass,_FVWallFluxQuadrature}) = eltype(m.Q)
_fv_adapt_mass(to, ::Nothing) = nothing
_fv_adapt_mass(to, m::_FVWallFluxQuadrature) = _FVWallFluxQuadrature(to(m.Q), to(m.error))
_fv_adapt_mass(to, m::FVImmersedMass) = FVImmersedMass(
    to(m.face_pair), to(m.body), to(m.region), to(m.Q), to(m.A),
    to(m.quadrature_error), to(m.region_open), to(m.faceptr), to(m.faces),
    to(m.rowptr), to(m.rowcol), to(m.rowval))

function _fv_wall_flux_quadrature(circles, annular, velocity)
    function integrate(k, n)
        cx, cy, radius = circles[k]
        sign = annular && k == 2 ? 1 : -1
        total = zero(radius)
        for l in 0:n-1
            angle = 2pi * l / n
            nx, ny = cos(angle), sin(angle)
            u, v = velocity(k, cx + radius * nx, cy + radius * ny)
            total += sign * radius * (u * nx + v * ny) * (2pi / n)
        end
        return total
    end
    fine = [integrate(k, 256) for k in eachindex(circles)]
    coarse = [integrate(k, 128) for k in eachindex(circles)]
    return _FVWallFluxQuadrature(fine, abs.(fine .- coarse))
end

function _fv_mass_components(tag, g, solid)
    labels = zeros(Int32, g.Nx, g.Ny)
    opens = Int32[]
    stack = Tuple{Int,Int}[]
    for j0 in 1:g.Ny, i0 in 1:g.Nx
        ((tag[i0, j0] == FV_CELL_SOLID) == solid && labels[i0, j0] == 0) || continue
        label = length(opens) + 1
        push!(opens, 0); push!(stack, (i0, j0)); labels[i0, j0] = label
        while !isempty(stack)
            i, j = pop!(stack)
            for d in 1:4
                if solid
                    # Whole solid bodies use ordinary 4-connectivity, including
                    # the exterior annular solid that reaches the domain edges.
                    ii, jj = i + (d == 1) - (d == 2), j + (d == 3) - (d == 4)
                    1 <= ii <= g.Nx && 1 <= jj <= g.Ny || continue
                    tag[ii, jj] == FV_CELL_SOLID || continue
                else
                    kind, ii, jj = _fv_face_kind(g, tag, i, j, d)
                    kind == _FV_F_OUTLET && (opens[label] = 1)
                    kind == _FV_F_INT || continue
                end
                labels[ii, jj] == 0 || continue
                labels[ii, jj] = label
                push!(stack, (ii, jj))
            end
        end
    end
    return labels, opens
end

"""Lower M3 pairs; `wall_flux(body,region)` is outward physical flux from fluid.

An exact scalar or pair-ordered vector is also accepted. Analytic circles may
fall back to periodic trapezoidal quadrature (256 points, error estimate from
128 points), reported in `quadrature_error` and by a warning. Raw geometries
must supply the target explicitly. In particular use `wall_flux=0` for rigid
closed walls; no arbitrary wall velocity is silently assigned zero flux.
"""
function _fv_lower_mass(b, g, tag, wall_flux)
    b === nothing && return nothing
    b.closure.mass == 3 || return b
    bodies, _ = _fv_mass_components(tag, g, true)
    regions, opens = _fv_mass_components(tag, g, false)
    pairmap = zeros(Int32, 4, g.Nx, g.Ny)
    pairs = Tuple{Int32,Int32}[]
    faces = Vector{NTuple{3,Int}}[]
    for j in 1:g.Ny, i in 1:g.Nx
        tag[i, j] == FV_CELL_SOLID && continue
        for d in 1:4
            q = b.qcut[d, i, j]
            ii, jj = i + (d == 1) - (d == 2), j + (d == 3) - (d == 4)
            interior = 1 <= ii <= g.Nx && 1 <= jj <= g.Ny
            if interior && tag[ii,jj] == FV_CELL_SOLID
                0 < _fv_ib_primal(q) <= 1 ||
                    _fv_ib_error("M3 immersed wall requires a cut fraction in (0,1]", (i,j), (ii,jj))
            else
                iszero(_fv_ib_primal(q)) && continue
            end
            1 <= ii <= g.Nx && 1 <= jj <= g.Ny ||
                _fv_ib_error("immersed face touches the domain boundary", (i, j), (ii, jj))
            tag[ii, jj] == FV_CELL_SOLID || _fv_ib_error("cut neighbour must be solid", (i, j), (ii, jj))
            touch = d <= 2 ? (j == 1 || j == g.Ny) : (i == 1 || i == g.Nx)
            x, y = (i - 1//2 + q * (ii-i)) * g.h, (j - 1//2 + q * (jj-j)) * g.h
            touch |= !(_fv_ib_primal(x) > 0 && _fv_ib_primal(x) < g.Nx*g.h &&
                       _fv_ib_primal(y) > 0 && _fv_ib_primal(y) < g.Ny*g.h)
            touch && _fv_ib_error("immersed wall face or physical cut touches the domain boundary", (i, j), (ii, jj))
            key = (bodies[ii, jj], regions[i, j])
            k = findfirst(==(key), pairs)
            if k === nothing
                push!(pairs, key); push!(faces, NTuple{3,Int}[]); k = length(pairs)
            end
            pairmap[d, i, j] = k
            push!(faces[k], (d, i, j))
        end
    end
    np = length(pairs)
    inherited = wall_flux === nothing && b.mass isa FVImmersedMass
    if inherited
        # Rebuilding a problem with an already lowered boundary keeps the
        # physical target and its quadrature diagnostic. Geometry-changing
        # maps must rebuild the boundary.
        wall_flux = b.mass.Q
    end
    fallback = wall_flux === nothing
    fallback && !(b.mass isa _FVWallFluxQuadrature) && np > 0 &&
        throw(ArgumentError("M3 raw geometry requires physical wall_flux(body,region), a scalar, or a pair-ordered vector"))
    wall_flux isa AbstractVector && length(wall_flux) != np &&
        throw(DimensionMismatch("wall_flux must have one value per body/region pair ($np)"))
    targets = Any[]; errors = Any[]
    for k in 1:np
        if fallback
            walls = unique([Int(b.surface.wall[d, i, j]) for (d, i, j) in faces[k]])
            push!(targets, sum(b.mass.Q[w] for w in walls))
            push!(errors, sum(b.mass.error[w] for w in walls))
        else
            q = wall_flux isa Number ? wall_flux : wall_flux isa AbstractVector ? wall_flux[k] : wall_flux(pairs[k]...)
            isfinite(_fv_ib_primal(q)) || throw(ArgumentError("wall_flux must be finite for body/region pair $(pairs[k])"))
            push!(targets, q)
            push!(errors, inherited ? b.mass.quadrature_error[k] : zero(q))
        end
    end
    fallback && np > 0 && @warn "M3 physical wall_flux uses circle quadrature; supply exact wall_flux for an analytical target" error_estimate=_fv_ib_primal.(errors)
    T = promote_type(eltype(b.qcut), eltype(b.wvel), typeof(g.h), map(typeof, targets)...)
    Q = T[targets...]; err = T[errors...]
    A = T[length(f) * g.h for f in faces]
    faceptr = Int32[1]; flatfaces = NTuple{3,Int}[]
    rowptr = Int32[1]; rowcol = Int32[]; rowval = T[]
    for k in 1:np
        append!(flatfaces, faces[k]); push!(faceptr, length(flatfaces)+1)
        coefficients = Dict{Int,T}()
        for (d, i, j) in faces[k]
            q = b.qcut[d, i, j]
            scale = g.h * ((d == 1 || d == 3) ? 1 : -1) / 2
            cP = b.closure.ghost == 1 ? 2-inv(q) : 3-2inv(q)
            cO = b.closure.ghost == 1 ? zero(q) : (1-q)/(1+q)
            comp = d <= 2 ? 1 : 2
            oi, oj = i - (d == 1) + (d == 2), j - (d == 3) + (d == 4)
            p, o = fv_index(g, i, j, comp), fv_index(g, oi, oj, comp)
            coefficients[p] = get(coefficients, p, zero(T)) + scale*cP
            coefficients[o] = get(coefficients, o, zero(T)) + scale*cO
        end
        coefficients[fv_nunknowns(g)+k] = -A[k]
        for c in sort!(collect(keys(coefficients)))
            push!(rowcol, c); push!(rowval, coefficients[c])
        end
        push!(rowptr, length(rowcol)+1)
    end
    packed = zeros(Int32, 3, length(flatfaces))
    for (l, face) in enumerate(flatfaces)
        packed[:, l] .= face
    end
    m = FVImmersedMass(pairmap, Int32[p[1] for p in pairs], Int32[p[2] for p in pairs],
        Q, A, err, Bool[x != 0 for x in opens], faceptr, packed, rowptr, rowcol, rowval)
    return FVImmersedBoundary2D(b.qcut, b.wvel, b.tag, b.h, b.closure, b.surface, m)
end

@inline function _fv_corrected_wall_mass(F, U, g, i, j, d, face_pair)
    face_pair === nothing && return F
    @inbounds k = face_pair[d, i, j]
    k == 0 && return F
    @inbounds return F - g.h * U[fv_nunknowns(g)+k]
end

@inline function _fv_mass_pair_residual(U, bvel, g, qcut, wvel, closure,
                                      faceptr, faces, Q, A, k)
    @inbounds value = -Q[k] - A[k]*U[fv_nunknowns(g)+k]
    @inbounds for l in faceptr[k]:faceptr[k+1]-1
        d, i, j = faces[1,l], faces[2,l], faces[3,l]
        _, _, _, _, _, uf, vf, _, _ = _fv_wall_data(U, bvel, g, i, j, d, qcut, wvel, closure)
        value += g.h * ((d == 1 || d == 3) ? 1 : -1) * (d <= 2 ? uf : vf)
    end
    return value
end

@kernel function _fv_mass_residual_kernel!(R, @Const(U), @Const(bvel), g,
        @Const(qcut), @Const(wvel), closure, @Const(faceptr), @Const(faces), @Const(Q), @Const(A))
    k = @index(Global)
    @inbounds R[fv_nunknowns(g)+k] = _fv_mass_pair_residual(U, bvel, g, qcut, wvel,
        closure, faceptr, faces, Q, A, k)
end

function _fv_mass_residual!(R, U, prob, backend, workgroup)
    m = _fv_mass(prob.immersed)
    m === nothing && return
    isempty(m.Q) && return
    b = prob.immersed
    _fv_mass_residual_kernel!(backend, workgroup)(R, U, prob.bvel, prob.grid,
        b.qcut, b.wvel, b.closure, m.faceptr, m.faces, m.Q, m.A; ndrange=length(m.Q))
end

function _fv_mass_pattern!(I, J, prob)
    m = _fv_mass(prob.immersed)
    m === nothing && return
    g = prob.grid; n = fv_nunknowns(g)
    for k in eachindex(m.Q)
        for l in m.faceptr[k]:m.faceptr[k+1]-1
            d, i, j = m.faces[:,l]
            row = fv_index(g, i, j, 1)
            push!(I, row, row+1); push!(J, n+k, n+k)
            if prob.tag[i,j] != FV_CELL_PINNED
                push!(I, row+2); push!(J, n+k)
            end
        end
        for l in m.rowptr[k]:m.rowptr[k+1]-1
            push!(I, n+k); push!(J, m.rowcol[l])
        end
    end
end

@kernel function _fv_mass_jacobian_kernel!(nzval, @Const(nzrow), @Const(nzcol),
        nlocal, @Const(rowptr), @Const(rowcol), @Const(rowval))
    q = @index(Global)
    @inbounds r = nzrow[q] - nlocal
    if r > 0
        @inbounds begin
            c = nzcol[q]; lo = rowptr[r]; hi = rowptr[r+1]-1
            while lo <= hi
                mid = (lo+hi) >>> 1
                if rowcol[mid] < c
                    lo = mid+1
                elseif rowcol[mid] > c
                    hi = mid-1
                else
                    nzval[q] = rowval[mid]
                    break
                end
            end
        end
    end
end

function _fv_mass_jacobian!(nzval, asm, prob, backend)
    m = _fv_mass(prob.immersed)
    m === nothing && return
    _fv_mass_jacobian_kernel!(backend, 256)(nzval, asm.nzrow, asm.nzcol,
        fv_nunknowns(prob.grid), m.rowptr, m.rowcol, m.rowval; ndrange=asm.nnz)
end

function _fv_check_mass(a, b)
    (a === nothing && b === nothing) && return
    (a isa FVImmersedMass && b isa FVImmersedMass) ||
        throw(ArgumentError("parameter map changed M3 mass topology; build a new solver"))
    for field in (:face_pair, :body, :region, :region_open, :faceptr, :faces, :rowptr, :rowcol)
        Array(getproperty(a,field)) == Array(getproperty(b,field)) ||
            throw(ArgumentError("parameter map changed M3 $field topology; build a new solver"))
    end
end
function _fv_copy_mass!(a, b)
    a === nothing && return
    for field in (:Q, :A, :quadrature_error, :rowval)
        x, y = getproperty(a,field), getproperty(b,field)
        x === y || copyto!(x,y)
    end
end
