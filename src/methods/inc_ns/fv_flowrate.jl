"""Lowered mean-fluid-x-velocity constraint; Fx replaces uniform x forcing.

The force is the final global unknown, after all immersed mass multipliers.
Its row and column are assembled analytically and excluded from colouring.
"""
struct _FVFlowRate{T}
    target::T
    initial_force::T
    nfluid::Int
end

_fv_lower_flowrate(::Nothing, g, tag, src, T) = nothing
function _fv_lower_flowrate(target::Real, g, tag, src, ::Type{T}) where {T}
    isfinite(target) || throw(ArgumentError("u_mean_target must be finite"))
    g.bcE == FV_BC_PERIODIC && g.bcW == FV_BC_PERIODIC ||
        throw(ArgumentError("mean-flow control requires periodic x boundaries"))
    cells = findall(!=(FV_CELL_SOLID), tag)
    isempty(cells) && throw(ArgumentError("mean-flow control requires fluid cells"))
    i, j = Tuple(first(cells)); force = src[fv_index(g,i,j,1)]
    all(c -> src[fv_index(g,Tuple(c)...,1)] == force, cells) ||
        throw(ArgumentError("mean-flow control replaces a uniform x source with Fx"))
    isfinite(force) && isfinite(T(target)) || throw(ArgumentError("flow data must remain finite in $T"))
    return _FVFlowRate(T(target), T(force), length(cells))
end

@inline _fv_force_index(p) = _fv_extra_index(p, _fv_npairs(p.immersed) + 1)
function _fv_initial_force!(U, p)
    p.flowrate === nothing || (U[_fv_force_index(p)] = p.flowrate.initial_force)
    return U
end

"""
    fv_flow_force(prob, U)

Read the solved uniform x body force for a problem with `u_mean_target`.
The scalar copy is host-side; this getter is not part of Newton iterations.
"""
function fv_flow_force(p::FVSteadyProblem2D, U)
    p.flowrate === nothing && throw(ArgumentError("Fx is an unknown only with u_mean_target"))
    return only(Array(view(U, _fv_force_index(p):_fv_force_index(p))))
end

@kernel function _fv_flow_force_kernel!(R, @Const(U), @Const(tag), @Const(src), g, force_index)
    cell = @index(Global)
    @inbounds if tag[cell] != FV_CELL_SOLID
        row = _fv_cell_index(g,cell,1)
        R[row] += g.h*g.h * (src[row] - U[force_index])
    end
end

# One deterministic reduction avoids atomics (which cannot add dual numbers)
# and device scratch allocation. It runs only on full residual evaluations.
@kernel function _fv_flow_constraint_kernel!(R, @Const(U), @Const(tag), g, flow, force_index)
    value = zero(eltype(R))
    @inbounds for cell in eachindex(tag)
        tag[cell] == FV_CELL_SOLID && continue
        value += U[_fv_cell_index(g,cell,1)]
    end
    @inbounds R[force_index] = value / flow.nfluid - flow.target
end

function _fv_flow_residual!(R, U, p, backend, workgroup; constraint=true)
    p.flowrate === nothing && return
    index = _fv_force_index(p); g = p.grid
    _fv_flow_force_kernel!(backend,workgroup)(R,U,p.tag,p.src,g,index;ndrange=g.Nx*g.Ny)
    constraint && _fv_flow_constraint_kernel!(backend,1)(R,U,p.tag,g,p.flowrate,index;ndrange=1)
    return nothing
end

function _fv_flow_pattern!(I,J,p)
    p.flowrate === nothing && return
    index = _fv_force_index(p); g = p.grid
    for j in 1:g.Ny, i in 1:g.Nx
        p.tag[i,j] == FV_CELL_SOLID && continue
        velocity = fv_index(g,i,j,1)
        push!(I,velocity,index); push!(J,index,velocity)
    end
    # A stored structural zero keeps diagonal indexing valid; no force shift.
    push!(I,index); push!(J,index)
    return nothing
end

@kernel function _fv_flow_jacobian_kernel!(nzval, @Const(rows), @Const(cols),
                                           force_index, h2, inv_nfluid)
    q = @index(Global)
    @inbounds begin
        row = rows[q]; col = cols[q]
        if row == force_index
            nzval[q] = col == force_index ? zero(h2) : inv_nfluid
        elseif col == force_index
            nzval[q] = -h2
        end
    end
end
function _fv_flow_jacobian!(nzval, asm, p, backend)
    p.flowrate === nothing && return
    _fv_flow_jacobian_kernel!(backend,256)(nzval,asm.nzrow,asm.nzcol,_fv_force_index(p),
        p.grid.h^2,inv(p.flowrate.nfluid);ndrange=asm.nnz)
    return nothing
end
