# Standalone steady SIMPLE incompressible solver for 2D manifold channels.
#
# Collocated cell-centred grid on [0,Lx] x [0,Ly]. Localized inlet velocity
# spans and pressure-reference outlet spans live on the west/east boundary.
# Full-cell immersed solids are represented by is_solid; fluid-solid momentum
# faces are no-slip Dirichlet faces, pressure-correction faces are homogeneous
# Neumann faces. Registered in src/KrakeNN.jl (exported driver; platform wrapper
# IncNS(:manifold) forwards here).

using KernelAbstractions
using LinearAlgebra
using SparseArrays

const _INCNS_MANIFOLD_OPERATOR_PATH =
    joinpath(@__DIR__, "..", "..", "fvfd", "operators_2d_grad_div_laplacian.jl")

if !isdefined(@__MODULE__, :pin_reference_dof)
    include(joinpath(@__DIR__, "..", "..", "solve", "poisson.jl"))
end
if !isdefined(@__MODULE__, :lin_factorize)
    include(joinpath(@__DIR__, "..", "..", "solve", "linear_solve.jl"))
end
if !isdefined(@__MODULE__, :gdl_divergence_2d!)
    include(_INCNS_MANIFOLD_OPERATOR_PATH)
end

_mf_lin(i, j, nx) = i + (j - 1) * nx
_mf_outlet_w(bc, j) = bc.outlet_side === :west && bc.outlet_mask[j]
_mf_outlet_e(bc, j) = bc.outlet_side === :east && bc.outlet_mask[j]
_mf_inlet_w(bc, j) = bc.inlet_side === :west && bc.inlet_mask[j]
_mf_inlet_e(bc, j) = bc.inlet_side === :east && bc.inlet_mask[j]

function _mf_boundary_spec(ny::Integer, inlet, outlet)
    inlet_side = Symbol(inlet.side)
    outlet_side = Symbol(outlet.side)
    inlet_side in (:west, :east) ||
        throw(ArgumentError("inlet.side must be :west or :east"))
    outlet_side in (:west, :east) ||
        throw(ArgumentError("outlet.side must be :west or :east"))

    inlet_j0 = Int(inlet.j0); inlet_j1 = Int(inlet.j1)
    outlet_j0 = Int(outlet.j0); outlet_j1 = Int(outlet.j1)
    1 <= inlet_j0 <= inlet_j1 <= ny ||
        throw(ArgumentError("inlet j0:j1 must lie inside 1:ny"))
    1 <= outlet_j0 <= outlet_j1 <= ny ||
        throw(ArgumentError("outlet j0:j1 must lie inside 1:ny"))
    inlet_side === outlet_side &&
        max(inlet_j0, outlet_j0) <= min(inlet_j1, outlet_j1) &&
        throw(ArgumentError("inlet and outlet spans overlap"))

    inlet_mask = falses(Int(ny))
    outlet_mask = falses(Int(ny))
    inlet_mask[inlet_j0:inlet_j1] .= true
    outlet_mask[outlet_j0:outlet_j1] .= true
    uin = Float64(inlet.u)
    return (; inlet_side, outlet_side, inlet_mask, outlet_mask, uin)
end

function _mf_check_mask(is_solid, nx::Int, ny::Int)
    is_solid === nothing && return falses(nx, ny)
    size(is_solid) == (nx, ny) ||
        throw(ArgumentError("is_solid must have size (nx, ny)"))
    return Matrix{Bool}(is_solid)
end

function _mf_simple_scheme(scheme::Symbol)
    s = Symbol(replace(lowercase(String(scheme)), '-' => '_'))
    s in (:simple, :simplec) ||
        throw(ArgumentError("scheme must be :simple or :simplec"))
    return s
end

function _mf_advection_scheme(advection::Symbol)
    scheme = Symbol(replace(lowercase(String(advection)), '-' => '_'))
    scheme in (:upwind, :linear_upwind) ||
        throw(ArgumentError("momentum_advection must be :upwind or :linear_upwind; got $advection"))
    return scheme
end

function _mf_assert_gridline(value::Real, h::Real, name::AbstractString)
    q = Float64(value) / Float64(h)
    isapprox(q, round(q); atol=1e-10, rtol=1e-10) ||
        throw(ArgumentError("$name=$value is not on a grid line with spacing $h"))
    return nothing
end

function manifold_full_cell_mask(nx::Integer, ny::Integer, Lx::Real, Ly::Real, plates)
    nx = Int(nx); ny = Int(ny)
    dx = Float64(Lx) / nx
    dy = Float64(Ly) / ny
    mask = falses(nx, ny)
    for (nplate, plate) in enumerate(plates)
        x0 = Float64(plate.x0); x1 = Float64(plate.x1)
        y0 = Float64(plate.y0); y1 = Float64(plate.y1)
        _mf_assert_gridline(x0, dx, "plates[$nplate].x0")
        _mf_assert_gridline(x1, dx, "plates[$nplate].x1")
        _mf_assert_gridline(y0, dy, "plates[$nplate].y0")
        _mf_assert_gridline(y1, dy, "plates[$nplate].y1")
        @inbounds for j in 1:ny, i in 1:nx
            x = (i - 0.5) * dx
            y = (j - 0.5) * dy
            mask[i, j] |= (x0 <= x <= x1) && (y0 <= y <= y1)
        end
    end
    return mask
end

function _mf_factorise(A::SparseMatrixCSC{Float64,Int};
                       pin_k0::Integer=0, spd::Bool=true)
    return lin_factorize(A; backend=CPUBackendTag(), spd=spd, pin_k0=Int(pin_k0))
end

_mf_solve!(cache::LinearSolveCache, b::Vector{Float64}) = lin_solve!(cache, b)

function _mf_positive_neighbour_sum(A::SparseMatrixCSC{Float64,Int})
    nb = zeros(Float64, size(A, 1))
    @inbounds for col in 1:size(A, 2)
        for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            row = A.rowval[ptr]
            row == col && continue
            a = A.nzval[ptr]
            a < 0.0 && (nb[row] += -a)
        end
    end
    return nb
end

function _mf_assemble_momentum_laplacian(nx, ny, dx, dy, is_solid, bc)
    n = nx * ny
    invdx2 = 1.0 / (dx * dx)
    invdy2 = 1.0 / (dy * dy)
    I = Int[]; J = Int[]; V = Float64[]
    sizehint!(I, 5n); sizehint!(J, 5n); sizehint!(V, 5n)

    @inbounds for j in 1:ny, i in 1:nx
        k = _mf_lin(i, j, nx)
        diag = 0.0
        if is_solid[i, j]
            diag = 1.0
        else
            if i > 1
                if is_solid[i - 1, j]
                    diag += 2.0 * invdx2
                else
                    push!(I, k); push!(J, _mf_lin(i - 1, j, nx)); push!(V, -invdx2)
                    diag += invdx2
                end
            elseif !_mf_outlet_w(bc, j)
                diag += 2.0 * invdx2
            end
            if i < nx
                if is_solid[i + 1, j]
                    diag += 2.0 * invdx2
                else
                    push!(I, k); push!(J, _mf_lin(i + 1, j, nx)); push!(V, -invdx2)
                    diag += invdx2
                end
            elseif !_mf_outlet_e(bc, j)
                diag += 2.0 * invdx2
            end
            if j > 1
                if is_solid[i, j - 1]
                    diag += 2.0 * invdy2
                else
                    push!(I, k); push!(J, _mf_lin(i, j - 1, nx)); push!(V, -invdy2)
                    diag += invdy2
                end
            else
                diag += 2.0 * invdy2
            end
            if j < ny
                if is_solid[i, j + 1]
                    diag += 2.0 * invdy2
                else
                    push!(I, k); push!(J, _mf_lin(i, j + 1, nx)); push!(V, -invdy2)
                    diag += invdy2
                end
            else
                diag += 2.0 * invdy2
            end
        end
        push!(I, k); push!(J, k); push!(V, diag)
    end
    return sparse(I, J, V, n, n)
end

function _mf_assemble_upwind_convection_operator(nx, ny, dx, dy, is_solid, bc,
                                                 uf, vf, uwest)
    n = nx * ny; invdx = 1.0 / dx; invdy = 1.0 / dy
    Iu = Int[]; Ju = Int[]; Vu = Float64[]
    Iv = Int[]; Jv = Int[]; Vv = Float64[]
    sizehint!(Iu, 5n); sizehint!(Ju, 5n); sizehint!(Vu, 5n)
    sizehint!(Iv, 5n); sizehint!(Jv, 5n); sizehint!(Vv, 5n)
    src_u = zeros(Float64, nx, ny)

    @inbounds for j in 1:ny, i in 1:nx
        is_solid[i, j] && continue
        k = _mf_lin(i, j, nx)
        diag_u = 0.0; diag_v = 0.0
        Fe = i < nx ? (!is_solid[i + 1, j] ? uf[i, j] : 0.0) : uf[i, j]
        Fw = i > 1 ? (!is_solid[i - 1, j] ? uf[i - 1, j] : 0.0) : uwest[j]
        Fn = j < ny ? (!is_solid[i, j + 1] ? vf[i, j] : 0.0) : 0.0
        Fs = j > 1 ? (!is_solid[i, j - 1] ? vf[i, j - 1] : 0.0) : 0.0

        if i < nx && !is_solid[i + 1, j]
            if Fe >= 0.0
                diag_u += Fe * invdx; diag_v += Fe * invdx
            else
                col = _mf_lin(i + 1, j, nx); c = Fe * invdx
                push!(Iu, k); push!(Ju, col); push!(Vu, c)
                push!(Iv, k); push!(Jv, col); push!(Vv, c)
            end
        elseif _mf_inlet_e(bc, j) && Fe < 0.0
            src_u[i, j] += Fe * bc.uin * invdx
        else
            diag_u += Fe * invdx
        end

        if i > 1 && !is_solid[i - 1, j]
            if Fw >= 0.0
                col = _mf_lin(i - 1, j, nx); c = -Fw * invdx
                push!(Iu, k); push!(Ju, col); push!(Vu, c)
                push!(Iv, k); push!(Jv, col); push!(Vv, c)
            else
                diag_u += -Fw * invdx; diag_v += -Fw * invdx
            end
        elseif _mf_inlet_w(bc, j) && Fw >= 0.0
            src_u[i, j] += -Fw * bc.uin * invdx
        else
            diag_u += -Fw * invdx
        end

        if j < ny && !is_solid[i, j + 1]
            if Fn >= 0.0
                diag_u += Fn * invdy; diag_v += Fn * invdy
            else
                col = _mf_lin(i, j + 1, nx); c = Fn * invdy
                push!(Iu, k); push!(Ju, col); push!(Vu, c)
                push!(Iv, k); push!(Jv, col); push!(Vv, c)
            end
        end

        if j > 1 && !is_solid[i, j - 1]
            if Fs >= 0.0
                col = _mf_lin(i, j - 1, nx); c = -Fs * invdy
                push!(Iu, k); push!(Ju, col); push!(Vu, c)
                push!(Iv, k); push!(Jv, col); push!(Vv, c)
            else
                diag_u += -Fs * invdy; diag_v += -Fs * invdy
            end
        end

        if diag_u != 0.0
            push!(Iu, k); push!(Ju, k); push!(Vu, diag_u)
        end
        if diag_v != 0.0
            push!(Iv, k); push!(Jv, k); push!(Vv, diag_v)
        end
    end

    return (; Cu=sparse(Iu, Ju, Vu, n, n),
            Cv=sparse(Iv, Jv, Vv, n, n),
            src_u)
end

function _mf_relaxed_momentum_operator(Amom_base::SparseMatrixCSC{Float64,Int},
                                       αu, nx, ny, is_solid, scheme_sym)
    ap_base = Vector(diag(Amom_base))
    extra = 1.0 / αu - 1.0
    Amom = Amom_base + spdiagm(0 => extra .* ap_base)
    ap = Vector(diag(Amom))
    ap_mat = reshape(ap, nx, ny)
    d_rc = αu ./ ap_mat
    if scheme_sym === :simple
        d = copy(d_rc)
    else
        anb = reshape(_mf_positive_neighbour_sum(Amom_base), nx, ny)
        d_rc_inv = 1.0 ./ d_rc
        denom = d_rc_inv .- anb
        @inbounds for idx in eachindex(denom, is_solid)
            !is_solid[idx] && denom[idx] <= 0.0 && (denom[idx] = d_rc_inv[idx])
        end
        d = 1.0 ./ denom
    end
    _mf_zero_solid!((d_rc, d), is_solid)
    return (; Amom, relax_old=reshape(extra .* ap_base, nx, ny), d_rc, d)
end

function _mf_momentum_dirichlet_rhs!(src_u, src_v, nx, ny, dx, dy, is_solid, bc)
    invdx2 = 1.0 / (dx * dx)
    fill!(src_u, 0.0)
    fill!(src_v, 0.0)
    @inbounds for j in 1:ny
        if !is_solid[1, j] && _mf_inlet_w(bc, j)
            src_u[1, j] += 2.0 * invdx2 * bc.uin
        end
        if !is_solid[nx, j] && _mf_inlet_e(bc, j)
            src_u[nx, j] += 2.0 * invdx2 * bc.uin
        end
    end
    return nothing
end

function _mf_assemble_pressure_operator(nx, ny, dx, dy, is_solid, bc, d_u, d_v)
    n = nx * ny
    invdx2 = 1.0 / (dx * dx)
    invdy2 = 1.0 / (dy * dy)
    I = Int[]; J = Int[]; V = Float64[]
    sizehint!(I, 5n); sizehint!(J, 5n); sizehint!(V, 5n)

    @inbounds for j in 1:ny, i in 1:nx
        k = _mf_lin(i, j, nx)
        diag = 0.0
        if is_solid[i, j]
            diag = 1.0
        else
            if i > 1 && !is_solid[i - 1, j]
                c = 0.5 * (d_u[i, j] + d_u[i - 1, j]) * invdx2
                push!(I, k); push!(J, _mf_lin(i - 1, j, nx)); push!(V, -c)
                diag += c
            elseif i == 1 && _mf_outlet_w(bc, j)
                diag += 2.0 * d_u[i, j] * invdx2
            end
            if i < nx && !is_solid[i + 1, j]
                c = 0.5 * (d_u[i, j] + d_u[i + 1, j]) * invdx2
                push!(I, k); push!(J, _mf_lin(i + 1, j, nx)); push!(V, -c)
                diag += c
            elseif i == nx && _mf_outlet_e(bc, j)
                diag += 2.0 * d_u[i, j] * invdx2
            end
            if j > 1 && !is_solid[i, j - 1]
                c = 0.5 * (d_v[i, j] + d_v[i, j - 1]) * invdy2
                push!(I, k); push!(J, _mf_lin(i, j - 1, nx)); push!(V, -c)
                diag += c
            end
            if j < ny && !is_solid[i, j + 1]
                c = 0.5 * (d_v[i, j] + d_v[i, j + 1]) * invdy2
                push!(I, k); push!(J, _mf_lin(i, j + 1, nx)); push!(V, -c)
                diag += c
            end
        end
        push!(I, k); push!(J, k); push!(V, diag)
    end
    return sparse(I, J, V, n, n)
end

function _mf_compact_gradient!(gx, gy, p, dx, dy, nx, ny, is_solid, bc)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    @inbounds for j in 1:ny, i in 1:nx
        if is_solid[i, j]
            gx[i, j] = 0.0; gy[i, j] = 0.0
        else
            pe = i < nx && !is_solid[i + 1, j] ? 0.5 * (p[i, j] + p[i + 1, j]) :
                 (i == nx && _mf_outlet_e(bc, j) ? 0.0 : p[i, j])
            pw = i > 1 && !is_solid[i - 1, j] ? 0.5 * (p[i - 1, j] + p[i, j]) :
                 (i == 1 && _mf_outlet_w(bc, j) ? 0.0 : p[i, j])
            pn = j < ny && !is_solid[i, j + 1] ? 0.5 * (p[i, j] + p[i, j + 1]) : p[i, j]
            ps = j > 1 && !is_solid[i, j - 1] ? 0.5 * (p[i, j - 1] + p[i, j]) : p[i, j]
            gx[i, j] = (pe - pw) * invdx
            gy[i, j] = (pn - ps) * invdy
        end
    end
    return nothing
end

function _mf_rhie_chow_faces!(uf, vf, uwest, u, v, p, gpx, gpy, d_u, d_v,
                              dx, dy, nx, ny, is_solid, bc)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    fill!(uf, 0.0); fill!(vf, 0.0); fill!(uwest, 0.0)
    @inbounds for j in 1:ny, i in 1:nx
        is_solid[i, j] && continue
        if i == 1
            if _mf_inlet_w(bc, j)
                uwest[j] = bc.uin
            elseif _mf_outlet_w(bc, j)
                gp_face = 2.0 * p[i, j] * invdx
                uwest[j] = u[i, j] - d_u[i, j] * (gp_face - gpx[i, j])
            end
        end
        if i < nx
            if !is_solid[i + 1, j]
                dbar = 0.5 * (d_u[i, j] + d_u[i + 1, j])
                gp_face = (p[i + 1, j] - p[i, j]) * invdx
                gp_cell = 0.5 * (gpx[i, j] + gpx[i + 1, j])
                uf[i, j] = 0.5 * (u[i, j] + u[i + 1, j]) - dbar * (gp_face - gp_cell)
            end
        elseif _mf_inlet_e(bc, j)
            uf[i, j] = bc.uin
        elseif _mf_outlet_e(bc, j)
            gp_face = -2.0 * p[i, j] * invdx
            uf[i, j] = u[i, j] - d_u[i, j] * (gp_face - gpx[i, j])
        end
        if j < ny && !is_solid[i, j + 1]
            dbar = 0.5 * (d_v[i, j] + d_v[i, j + 1])
            gp_face = (p[i, j + 1] - p[i, j]) * invdy
            gp_cell = 0.5 * (gpy[i, j] + gpy[i, j + 1])
            vf[i, j] = 0.5 * (v[i, j] + v[i, j + 1]) - dbar * (gp_face - gp_cell)
        end
    end
    return nothing
end

function _mf_convection_upwind!(conv_u, conv_v, u, v, uf, vf, uwest, dx, dy,
                                nx, ny, is_solid, bc)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    @inbounds for j in 1:ny, i in 1:nx
        if is_solid[i, j]
            conv_u[i, j] = 0.0; conv_v[i, j] = 0.0
            continue
        end
        Fe = i < nx ? (!is_solid[i + 1, j] ? uf[i, j] : 0.0) : uf[i, j]
        Fw = i > 1 ? (!is_solid[i - 1, j] ? uf[i - 1, j] : 0.0) : uwest[j]
        Fn = j < ny ? (!is_solid[i, j + 1] ? vf[i, j] : 0.0) : 0.0
        Fs = j > 1 ? (!is_solid[i, j - 1] ? vf[i, j - 1] : 0.0) : 0.0

        uE = i < nx && !is_solid[i + 1, j] ? (Fe >= 0 ? u[i, j] : u[i + 1, j]) :
             (_mf_inlet_e(bc, j) && Fe < 0 ? bc.uin : u[i, j])
        uW = i > 1 && !is_solid[i - 1, j] ? (Fw >= 0 ? u[i - 1, j] : u[i, j]) :
             (_mf_inlet_w(bc, j) && Fw >= 0 ? bc.uin : u[i, j])
        uN = j < ny && !is_solid[i, j + 1] ? (Fn >= 0 ? u[i, j] : u[i, j + 1]) : 0.0
        uS = j > 1 && !is_solid[i, j - 1] ? (Fs >= 0 ? u[i, j - 1] : u[i, j]) : 0.0

        vE = i < nx && !is_solid[i + 1, j] ? (Fe >= 0 ? v[i, j] : v[i + 1, j]) : 0.0
        vW = i > 1 && !is_solid[i - 1, j] ? (Fw >= 0 ? v[i - 1, j] : v[i, j]) : 0.0
        vN = j < ny && !is_solid[i, j + 1] ? (Fn >= 0 ? v[i, j] : v[i, j + 1]) : 0.0
        vS = j > 1 && !is_solid[i, j - 1] ? (Fs >= 0 ? v[i, j - 1] : v[i, j]) : 0.0

        conv_u[i, j] = (Fe * uE - Fw * uW) * invdx + (Fn * uN - Fs * uS) * invdy
        conv_v[i, j] = (Fe * vE - Fw * vW) * invdx + (Fn * vN - Fs * vS) * invdy
    end
    return nothing
end

function _mf_linear_upwind_x(q::AbstractMatrix, is_solid::AbstractMatrix{Bool},
                             nx::Int, i::Int, j::Int, F::Float64)
    low = F >= 0.0 ? Float64(q[i, j]) : Float64(q[i + 1, j])
    F >= 0.0 && i > 1 && !is_solid[i - 1, j] &&
        return low + 0.5 * (low - Float64(q[i - 1, j]))
    F < 0.0 && i + 1 < nx && !is_solid[i + 2, j] &&
        return low + 0.5 * (low - Float64(q[i + 2, j]))
    return low
end

function _mf_linear_upwind_y(q::AbstractMatrix, is_solid::AbstractMatrix{Bool},
                             ny::Int, i::Int, j::Int, F::Float64)
    low = F >= 0.0 ? Float64(q[i, j]) : Float64(q[i, j + 1])
    F >= 0.0 && j > 1 && !is_solid[i, j - 1] &&
        return low + 0.5 * (low - Float64(q[i, j - 1]))
    F < 0.0 && j + 1 < ny && !is_solid[i, j + 2] &&
        return low + 0.5 * (low - Float64(q[i, j + 2]))
    return low
end

function _mf_convection_linear_upwind!(conv_u, conv_v, u, v, uf, vf, uwest,
                                       dx, dy, nx, ny, is_solid, bc)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    @inbounds for j in 1:ny, i in 1:nx
        if is_solid[i, j]
            conv_u[i, j] = 0.0; conv_v[i, j] = 0.0
            continue
        end
        Fe = i < nx ? (!is_solid[i + 1, j] ? uf[i, j] : 0.0) : uf[i, j]
        Fw = i > 1 ? (!is_solid[i - 1, j] ? uf[i - 1, j] : 0.0) : uwest[j]
        Fn = j < ny ? (!is_solid[i, j + 1] ? vf[i, j] : 0.0) : 0.0
        Fs = j > 1 ? (!is_solid[i, j - 1] ? vf[i, j - 1] : 0.0) : 0.0

        uE = i < nx && !is_solid[i + 1, j] ? _mf_linear_upwind_x(u, is_solid, nx, i, j, Fe) :
             (_mf_inlet_e(bc, j) && Fe < 0 ? bc.uin : u[i, j])
        uW = i > 1 && !is_solid[i - 1, j] ? _mf_linear_upwind_x(u, is_solid, nx, i - 1, j, Fw) :
             (_mf_inlet_w(bc, j) && Fw >= 0 ? bc.uin : u[i, j])
        uN = j < ny && !is_solid[i, j + 1] ? _mf_linear_upwind_y(u, is_solid, ny, i, j, Fn) : 0.0
        uS = j > 1 && !is_solid[i, j - 1] ? _mf_linear_upwind_y(u, is_solid, ny, i, j - 1, Fs) : 0.0

        vE = i < nx && !is_solid[i + 1, j] ? _mf_linear_upwind_x(v, is_solid, nx, i, j, Fe) : 0.0
        vW = i > 1 && !is_solid[i - 1, j] ? _mf_linear_upwind_x(v, is_solid, nx, i - 1, j, Fw) : 0.0
        vN = j < ny && !is_solid[i, j + 1] ? _mf_linear_upwind_y(v, is_solid, ny, i, j, Fn) : 0.0
        vS = j > 1 && !is_solid[i, j - 1] ? _mf_linear_upwind_y(v, is_solid, ny, i, j - 1, Fs) : 0.0

        conv_u[i, j] = (Fe * uE - Fw * uW) * invdx + (Fn * uN - Fs * uS) * invdy
        conv_v[i, j] = (Fe * vE - Fw * vW) * invdx + (Fn * vN - Fs * vS) * invdy
    end
    return nothing
end

function _mf_convection!(conv_u, conv_v, u, v, uf, vf, uwest, dx, dy, nx, ny,
                         is_solid, bc, momentum_advection::Symbol)
    if momentum_advection === :upwind
        return _mf_convection_upwind!(conv_u, conv_v, u, v, uf, vf, uwest,
                                      dx, dy, nx, ny, is_solid, bc)
    elseif momentum_advection === :linear_upwind
        return _mf_convection_linear_upwind!(conv_u, conv_v, u, v, uf, vf,
                                             uwest, dx, dy, nx, ny, is_solid, bc)
    end
    throw(ArgumentError("momentum_advection must be :upwind or :linear_upwind; got $momentum_advection"))
end

function _mf_face_divergence!(div, uf, vf, uwest, dx, dy, nx, ny, is_solid)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    @inbounds for j in 1:ny, i in 1:nx
        if is_solid[i, j]
            div[i, j] = 0.0
        else
            ue = i < nx ? (!is_solid[i + 1, j] ? uf[i, j] : 0.0) : uf[i, j]
            uw = i > 1 ? (!is_solid[i - 1, j] ? uf[i - 1, j] : 0.0) : uwest[j]
            vn = j < ny ? (!is_solid[i, j + 1] ? vf[i, j] : 0.0) : 0.0
            vs = j > 1 ? (!is_solid[i, j - 1] ? vf[i, j - 1] : 0.0) : 0.0
            div[i, j] = (ue - uw) * invdx + (vn - vs) * invdy
        end
    end
    return nothing
end

function _mf_correct_faces!(uf, vf, uwest, pcorr, d_u, d_v, dx, dy, nx, ny, is_solid, bc)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    @inbounds for j in 1:ny, i in 1:nx
        is_solid[i, j] && continue
        if i < nx && !is_solid[i + 1, j]
            dbar = 0.5 * (d_u[i, j] + d_u[i + 1, j])
            uf[i, j] += dbar * (pcorr[i + 1, j] - pcorr[i, j]) * invdx
        elseif i == nx && _mf_outlet_e(bc, j)
            uf[i, j] += 2.0 * d_u[i, j] * (0.0 - pcorr[i, j]) * invdx
        end
        if i == 1 && _mf_outlet_w(bc, j)
            uwest[j] += 2.0 * d_u[i, j] * (pcorr[i, j] - 0.0) * invdx
        end
        if j < ny && !is_solid[i, j + 1]
            dbar = 0.5 * (d_v[i, j] + d_v[i, j + 1])
            vf[i, j] += dbar * (pcorr[i, j + 1] - pcorr[i, j]) * invdy
        end
    end
    return nothing
end

function _mf_zero_solid!(fields, is_solid)
    @inbounds for A in fields
        for idx in eachindex(A, is_solid)
            is_solid[idx] && (A[idx] = 0.0)
        end
    end
    return nothing
end

function _mf_checkerboard_metric(p, is_solid, nx, ny)
    vals = [p[i, j] for j in 1:ny, i in 1:nx if !is_solid[i, j]]
    isempty(vals) && return 0.0
    pbar = sum(vals) / length(vals)
    osc = 0.0
    tot = 0.0
    @inbounds for j in 2:ny-1, i in 2:nx-1
        if !is_solid[i, j] && !is_solid[i - 1, j] && !is_solid[i + 1, j] &&
           !is_solid[i, j - 1] && !is_solid[i, j + 1]
            lap = p[i + 1, j] + p[i - 1, j] + p[i, j + 1] + p[i, j - 1] - 4.0 * p[i, j]
            osc += lap^2
            tot += (p[i, j] - pbar)^2
        end
    end
    return sqrt(osc / max(tot, eps()))
end

function _mf_fluxes(p, uf, uwest, is_solid, dx, dy, nx, ny, bc)
    inflow = 0.0
    outflow = 0.0
    pin_sum = 0.0
    pin_n = 0
    @inbounds for j in 1:ny
        if bc.inlet_side === :west && bc.inlet_mask[j] && !is_solid[1, j]
            inflow += bc.uin * dy
            pin_sum += p[1, j]; pin_n += 1
        elseif bc.inlet_side === :east && bc.inlet_mask[j] && !is_solid[nx, j]
            inflow += -bc.uin * dy
            pin_sum += p[nx, j]; pin_n += 1
        end
        if bc.outlet_side === :west && bc.outlet_mask[j] && !is_solid[1, j]
            outflow += -uwest[j] * dy
        elseif bc.outlet_side === :east && bc.outlet_mask[j] && !is_solid[nx, j]
            outflow += uf[nx, j] * dy
        end
    end
    dp = pin_n == 0 ? 0.0 : pin_sum / pin_n
    imbalance = abs(outflow - inflow) / max(abs(inflow), eps())
    return (; inflow, outflow, mass_imbalance=imbalance, dp)
end

"""
    solve_incns_manifold(; nx, ny, Lx, Ly, Re, U_in,
                         is_solid=nothing, inlet, outlet, mu=nothing,
                         relax=(u=0.7,p=0.3), scheme=:simplec, tol=1e-7,
                         maxiter=4000, backend=CPU(), verbose=false,
                         momentum_advection=:linear_upwind)

Steady incompressible Navier-Stokes SIMPLE solver for a 2D manifold on a
collocated cell-centred grid. Registered in `src/KrakeNN.jl` and exported; also
reachable as `solve(params, IncNS(:manifold))` through the platform contract.

`inlet = (; side, j0, j1, u)` imposes a localized velocity Dirichlet span on the
west/east boundary. The inlet face flux is frozen in the pressure-correction
projection, so the pressure side is homogeneous Neumann there. `outlet =
(; side, j0, j1)` imposes the pressure reference: a localized Dirichlet
`p' = 0` span in the pressure-correction Poisson, with zero-gradient momentum
at the same velocity faces. Do not add a separate pressure pin.

`is_solid` marks full solid cells. Fluid-solid pressure-correction faces are
homogeneous Neumann; fluid-solid momentum faces are no-slip Dirichlet faces.
The returned `uf, vf` use the cavity convention: `uf[i,j]` is the east face of
cell `(i,j)`, `vf[i,j]` is the north face. Those face fields, together with
`is_solid`, `dx`, `dy`, `nx`, and `ny`, are the handoff contract consumed by
`solve_scalar_transport`.
"""
function solve_incns_manifold(; nx::Integer, ny::Integer, Lx::Real, Ly::Real,
                              Re::Real, U_in::Real,
                              is_solid::Union{Nothing,AbstractMatrix{Bool}}=nothing,
                              inlet, outlet, mu=nothing,
                              relax=(u=0.7, p=0.3), tol::Real=1e-7,
                              maxiter::Integer=4000, backend=CPU(),
                              verbose::Bool=false, scheme::Symbol=:simplec,
                              momentum_advection::Symbol=:linear_upwind)
    nx = Int(nx); ny = Int(ny)
    Lx = Float64(Lx); Ly = Float64(Ly); Re = Float64(Re); U_in = Float64(U_in)
    dx = Lx / nx
    dy = Ly / ny
    μ = mu === nothing ? abs(U_in) * Ly / Re : Float64(mu)
    αu = Float64(relax.u); αp = Float64(relax.p)
    0.0 < αu <= 1.0 || throw(ArgumentError("relax.u must lie in (0, 1]"))
    0.0 < αp <= 1.0 || throw(ArgumentError("relax.p must lie in (0, 1]"))
    scheme_sym = _mf_simple_scheme(scheme)
    momentum_scheme = _mf_advection_scheme(momentum_advection)

    solid = _mf_check_mask(is_solid, nx, ny)
    bc = _mf_boundary_spec(ny, inlet, outlet)
    xcenters = [(i - 0.5) * dx for i in 1:nx]
    ycenters = [(j - 0.5) * dy for j in 1:ny]

    Lmom = _mf_assemble_momentum_laplacian(nx, ny, dx, dy, solid, bc)
    Amom_visc = μ .* Lmom
    ap_visc = Vector(diag(Amom_visc))
    extra = 1.0 / αu - 1.0
    Amom = Amom_visc + spdiagm(0 => extra .* ap_visc)
    ap = Vector(diag(Amom))
    ap_mat = reshape(ap, nx, ny)
    d_rc_u = αu ./ ap_mat
    d_rc_v = αu ./ ap_mat
    if scheme_sym === :simple
        d_u = d_rc_u
        d_v = d_rc_v
    else
        anb = reshape(_mf_positive_neighbour_sum(Amom_visc), nx, ny)
        denom = 1.0 ./ d_rc_u .- anb
        fluid_min = minimum(denom[.!solid])
        fluid_min > 0.0 ||
            throw(ArgumentError("SIMPLEC correction denominator must be positive; use relax.u < 1"))
        d_u = 1.0 ./ denom
        d_v = 1.0 ./ denom
    end
    _mf_zero_solid!((d_rc_u, d_rc_v, d_u, d_v), solid)
    mom_op = nothing
    p_op = nothing
    if momentum_scheme === :upwind
        mom_op = _mf_factorise(Amom)
        Ap = _mf_assemble_pressure_operator(nx, ny, dx, dy, solid, bc, d_u, d_v)
        p_op = _mf_factorise(Ap; pin_k0=0)
    end

    src_u = zeros(Float64, nx, ny)
    src_v = zeros(Float64, nx, ny)
    _mf_momentum_dirichlet_rhs!(src_u, src_v, nx, ny, dx, dy, solid, bc)
    src_u .*= μ

    u = zeros(Float64, nx, ny)
    v = zeros(Float64, nx, ny)
    p = zeros(Float64, nx, ny)
    if bc.inlet_side === :west && bc.outlet_side === :east
        dp0 = 12.0 * μ * abs(bc.uin) * Lx / (Ly * Ly)
        @inbounds for j in 1:ny, i in 1:nx
            p[i, j] = dp0 * (1.0 - xcenters[i] / Lx)
        end
    elseif bc.inlet_side === :east && bc.outlet_side === :west
        dp0 = 12.0 * μ * abs(bc.uin) * Lx / (Ly * Ly)
        @inbounds for j in 1:ny, i in 1:nx
            p[i, j] = dp0 * (xcenters[i] / Lx)
        end
    end
    _mf_zero_solid!((u, v, p), solid)

    gpx = zeros(Float64, nx, ny); gpy = zeros(Float64, nx, ny)
    uf = zeros(Float64, nx, ny); vf = zeros(Float64, nx, ny); uwest = zeros(Float64, ny)
    conv_u = zeros(Float64, nx, ny); conv_v = zeros(Float64, nx, ny)
    conv_lu_u = zeros(Float64, nx, ny); conv_lu_v = zeros(Float64, nx, ny)
    divstar = zeros(Float64, nx, ny); divcorr = zeros(Float64, nx, ny)
    pcorr = zeros(Float64, nx, ny)
    residual_history = Float64[]
    converged = false
    iters = 0
    vel_change = Inf
    ref_flux = max(abs(U_in), abs(bc.uin), eps())

    if momentum_scheme === :linear_upwind
        _mf_compact_gradient!(gpx, gpy, p, dx, dy, nx, ny, solid, bc)
        _mf_rhie_chow_faces!(uf, vf, uwest, u, v, p, gpx, gpy, d_rc_u, d_rc_v,
                             dx, dy, nx, ny, solid, bc)
    end

    for it in 1:Int(maxiter)
        iters = it
        if momentum_scheme === :upwind
            _mf_compact_gradient!(gpx, gpy, p, dx, dy, nx, ny, solid, bc)
            _mf_rhie_chow_faces!(uf, vf, uwest, u, v, p, gpx, gpy, d_rc_u, d_rc_v,
                                 dx, dy, nx, ny, solid, bc)
            _mf_convection!(conv_u, conv_v, u, v, uf, vf, uwest, dx, dy, nx, ny,
                            solid, bc, momentum_scheme)

            bu = vec(src_u .- conv_u .- gpx .+ (extra .* reshape(ap_visc, nx, ny)) .* u)
            bv = vec(src_v .- conv_v .- gpy .+ (extra .* reshape(ap_visc, nx, ny)) .* v)
            ustar = reshape(_mf_solve!(mom_op, bu), nx, ny)
            vstar = reshape(_mf_solve!(mom_op, bv), nx, ny)
        else
            adv = _mf_assemble_upwind_convection_operator(nx, ny, dx, dy, solid, bc,
                                                          uf, vf, uwest)
            uop = _mf_relaxed_momentum_operator(Amom_visc + adv.Cu, αu, nx, ny,
                                                solid, scheme_sym)
            vop = _mf_relaxed_momentum_operator(Amom_visc + adv.Cv, αu, nx, ny,
                                                solid, scheme_sym)
            d_rc_u = uop.d_rc
            d_rc_v = vop.d_rc
            d_u = uop.d
            d_v = vop.d
            mom_u_op = _mf_factorise(uop.Amom; spd=false)
            mom_v_op = _mf_factorise(vop.Amom; spd=false)
            Ap = _mf_assemble_pressure_operator(nx, ny, dx, dy, solid, bc, d_u, d_v)
            p_op = _mf_factorise(Ap; pin_k0=0)

            _mf_convection_upwind!(conv_u, conv_v, u, v, uf, vf, uwest,
                                   dx, dy, nx, ny, solid, bc)
            _mf_convection_linear_upwind!(conv_lu_u, conv_lu_v, u, v, uf, vf, uwest,
                                          dx, dy, nx, ny, solid, bc)
            _mf_compact_gradient!(gpx, gpy, p, dx, dy, nx, ny, solid, bc)

            bu = vec(src_u .- adv.src_u .- (conv_lu_u .- conv_u) .- gpx .+
                     uop.relax_old .* u)
            bv = vec(src_v .- (conv_lu_v .- conv_v) .- gpy .+ vop.relax_old .* v)
            ustar = reshape(_mf_solve!(mom_u_op, bu), nx, ny)
            vstar = reshape(_mf_solve!(mom_v_op, bv), nx, ny)
        end

        umax_prev = max(maximum(abs, u), maximum(abs, v), ref_flux * eps())
        du = 0.0
        @inbounds for idx in eachindex(u)
            du = max(du, abs(ustar[idx] - u[idx]), abs(vstar[idx] - v[idx]))
            u[idx] = ustar[idx]
            v[idx] = vstar[idx]
        end
        _mf_zero_solid!((u, v), solid)
        vel_change = du / umax_prev

        _mf_compact_gradient!(gpx, gpy, p, dx, dy, nx, ny, solid, bc)
        _mf_rhie_chow_faces!(uf, vf, uwest, u, v, p, gpx, gpy, d_rc_u, d_rc_v,
                             dx, dy, nx, ny, solid, bc)
        _mf_face_divergence!(divstar, uf, vf, uwest, dx, dy, nx, ny, solid)

        pcorr .= reshape(_mf_solve!(p_op, vec(divstar)), nx, ny)
        _mf_correct_faces!(uf, vf, uwest, pcorr, d_u, d_v, dx, dy, nx, ny, solid, bc)
        _mf_compact_gradient!(gpx, gpy, pcorr, dx, dy, nx, ny, solid, bc)
        @. u = u + d_u * gpx
        @. v = v + d_v * gpy
        @. p = p - αp * pcorr
        _mf_zero_solid!((u, v, p), solid)

        _mf_face_divergence!(divcorr, uf, vf, uwest, dx, dy, nx, ny, solid)
        res = sqrt(sum(abs2, divcorr) / max(count(!, solid), 1)) * min(dx, dy) / ref_flux
        push!(residual_history, res)
        verbose && (it <= 5 || it % 100 == 0) &&
            @info "manifold SIMPLE" scheme=scheme_sym it res vel_change
        if res < tol && vel_change < tol
            converged = true
            break
        end
    end

    flux = _mf_fluxes(p, uf, uwest, solid, dx, dy, nx, ny, bc)
    checkerboard = _mf_checkerboard_metric(p, solid, nx, ny)

    return (; u, v, p, uf, vf, is_solid=solid, dx, dy, nx, ny, xcenters,
            ycenters, residual_history, iters, converged, vel_change,
            mass_imbalance=flux.mass_imbalance, dp=flux.dp, Re, mu=μ,
            U_in, Lx, Ly, checkerboard, scheme=scheme_sym,
            momentum_advection=momentum_scheme)
end
