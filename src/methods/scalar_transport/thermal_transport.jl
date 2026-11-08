# Standalone DECOUPLED STEADY scalar advection–diffusion solver ("thermal transport").
#
# Given a FROZEN face-normal velocity field (uf,vf) on a collocated cell-centred
# grid, solve the steady scalar transport equation
#
#   ∇·(u T) − DT ∇²T = 0
#
# to steady state with mixed Dirichlet / Neumann-flux / zero-gradient outflow
# boundary conditions. The default advection scheme keeps the stable first-order
# upwind operator in the matrix and applies a linear-upwind correction on the RHS
# by Picard deferred correction, reusing the same factorization. `advection =
# :upwind` preserves the legacy single-solve first-order path.
#
# Reuses the matrix-assembly pattern of inc_ns/simple.jl (`_incns_assemble_neg_laplacian`,
# 5-point Laplacian into a sparse CSC) and the cavity's first-order UPWIND
# advection logic (`_cavity_convection!`), transcribed here into MATRIX-ASSEMBLY
# coefficients (donor diagonal +F, acceptor off-diagonal −F). The solve goes
# through the shared factorize-once seam (`lin_factorize`/`lin_solve!`); since the
# upwind advection makes A NON-symmetric we use spd=false (LU/LDLᵀ fallback).
# Reimplements NOTHING from the operators or the linear-solve seam — calls/mirrors.
#
# KA + stdlib only, CPU by default. Registered in `src/KrakeNN.jl` (exported
# driver `solve_scalar_transport`); include guards keep the brick
# standalone-include-able.
#
# Public entry point:
#   solve_scalar_transport(; nx, ny, dx, dy, uf, vf, DT, is_solid, bc, backend,
#                          advection, deferred_passes)
#     -> NamedTuple(T, residual_history, iters, converged, dx, dy, ycenters,
#                   nx, ny, DT, Pe_cell)

using LinearAlgebra
using SparseArrays

# Factorize-once seam (linear_solve.jl pulls in poisson.jl for pin_reference_dof).
# Guarded so the brick is include-able standalone on a CPU-only box.
if !isdefined(@__MODULE__, :lin_factorize)
    include(joinpath(@__DIR__, "..", "..", "solve", "linear_solve.jl"))
end

# ---------------------------------------------------------------------------
# Sparse system assembly for steady scalar advection–diffusion.
#
# Layout: linear index k = i + (j-1)*nx, i in 1:nx (x), j in 1:ny (y). Same
# layout as simple.jl / cavity.jl. uf[i,j] = east face of cell (i,j) (x-normal),
# vf[i,j] = north face of cell (i,j) (y-normal); the west face of (i,j) is
# uf[i-1,j], the south face is vf[i,j-1] (FROZEN velocities).
#
# We assemble  A = ADV + DIFF  and the RHS b, so that A·T = b discretises
#   div(u T) − DT·lap(T) = 0  ⇒  ADV·T − DT·lap(T) = 0,  i.e.
#   A = ADV + (−DT·lap) = ADV + DT·(neg-Laplacian).
#
# DIFFUSION (mirrors _incns_assemble_neg_laplacian, scaled by DT):
#   The discrete (−Laplacian) contributes +DT·invh2 to the diagonal and
#   −DT·invh2 to each interior neighbour. Per wall:
#     :dirichlet -> ghost T_g = 2·T_wall − T_c  ⇒  +2·DT·invh2 on diagonal,
#                   wall-value source +2·DT·invh2·T_wall into b (cavity convention).
#     :flux      -> Neumann wall heat flux q. Zero-gradient ghost for diffusion
#                   (no diagonal face term); the flux enters b as a SOURCE on the
#                   wall-adjacent row:  +q/h  (q = DT·∂T/∂n, the conductive flux
#                   leaving the wall into the domain). NOT in A.
#     :outflow   -> zero-gradient (ghost = interior): no diffusive face term.
#
# ADVECTION (mirrors _cavity_convection! first-order upwind, as MATRIX coeffs):
#   conv = (Fe·T_e − Fw·T_w)/dx + (Fn·T_n − Fs·T_s)/dy with the upwind donor cell.
#   On an interior east face with flux Fe = uf[i,j]:
#     Fe >= 0 (outflow east, donor = own cell):  +Fe/dx on diag(k),
#     Fe <  0 (inflow from east, donor = i+1):   +Fe/dx on off-diag (k, i+1).
#   The matching face for cell (i+1) is its WEST face Fw = uf[i,j], handled when
#   the loop reaches (i+1) via the symmetric (i>1) branch. Boundary faces:
#     :dirichlet -> wall value T_wall is the upwind donor when flowing IN; its
#                   contribution Fwall·T_wall/h goes to b (sign per inflow/outflow).
#     :flux      -> adiabatic/heat-flux wall for diffusion; no advective
#                   boundary transport.
#     :outflow   -> zero-gradient (ghost = interior cell): the boundary face
#                   advects the INTERIOR upwind value, i.e. a boundary diagonal
#                   term with the appropriate face-flux sign.
#
# SOLIDS:
#   Solid cells are identity rows pinned to T=0. Fluid-solid faces contribute no
#   diffusive or advective flux, i.e. they are adiabatic impermeable walls.
# ---------------------------------------------------------------------------
function _st_check_shape(name::AbstractString, A, nx::Int, ny::Int)
    size(A) == (nx, ny) ||
        throw(ArgumentError("$name must have size ($nx, $ny); got $(size(A))"))
    return nothing
end

function _st_bc_kind(kind, side::Symbol)
    k = Symbol(kind)
    k in (:dirichlet, :flux, :outflow) ||
        throw(ArgumentError("bc.$side kind must be :dirichlet, :flux, or :outflow; got $k"))
    return k
end

function _st_advection_scheme(advection::Symbol)
    scheme = Symbol(replace(lowercase(String(advection)), '-' => '_'))
    scheme in (:upwind, :linear_upwind) ||
        throw(ArgumentError("advection must be :upwind or :linear_upwind; got $advection"))
    return scheme
end

function _st_side_bc(sidebc, n::Int, side::Symbol)
    kinds = fill(:flux, n)
    values = zeros(Float64, n)

    if hasproperty(sidebc, :kind)
        hasproperty(sidebc, :value) ||
            throw(ArgumentError("bc.$side must include value"))
        kind = _st_bc_kind(getproperty(sidebc, :kind), side)
        value = Float64(getproperty(sidebc, :value))
        fill!(kinds, kind)
        fill!(values, value)
    elseif sidebc isa AbstractVector
        covered = falses(n)
        for (iseg, seg) in enumerate(sidebc)
            for prop in (:lo, :hi, :kind, :value)
                hasproperty(seg, prop) ||
                    throw(ArgumentError("bc.$side segment $iseg must include $prop"))
            end
            lo = Int(getproperty(seg, :lo))
            hi = Int(getproperty(seg, :hi))
            1 <= lo <= hi <= n ||
                throw(ArgumentError("bc.$side segment $iseg range $lo:$hi must lie inside 1:$n"))
            any(@view covered[lo:hi]) &&
                throw(ArgumentError("bc.$side segments must not overlap"))
            kind = _st_bc_kind(getproperty(seg, :kind), side)
            value = Float64(getproperty(seg, :value))
            @inbounds for q in lo:hi
                kinds[q] = kind
                values[q] = value
                covered[q] = true
            end
        end
    else
        throw(ArgumentError("bc.$side must be a NamedTuple BC or a vector of segment NamedTuples"))
    end

    return (; kind=kinds, value=values)
end

function _st_boundary_bcs(bc, nx::Int, ny::Int)
    for side in (:west, :east, :south, :north)
        hasproperty(bc, side) || throw(ArgumentError("bc must include $side"))
    end
    return (west = _st_side_bc(getproperty(bc, :west), ny, :west),
            east = _st_side_bc(getproperty(bc, :east), ny, :east),
            south = _st_side_bc(getproperty(bc, :south), nx, :south),
            north = _st_side_bc(getproperty(bc, :north), nx, :north))
end

function _st_west_boundary_flux(uf::AbstractMatrix, vf::AbstractMatrix,
                                solid::AbstractMatrix{Bool}, nx::Int, ny::Int,
                                dx::Float64, dy::Float64, j::Int)
    Fe = (nx > 1 && !solid[2, j]) ? Float64(uf[1, j]) : 0.0
    Fn = (j < ny && !solid[1, j + 1]) ? Float64(vf[1, j]) : 0.0
    Fs = (j > 1 && !solid[1, j - 1]) ? Float64(vf[1, j - 1]) : 0.0
    return Fe + (dx / dy) * (Fn - Fs)
end

function _st_assemble_system(nx::Integer, ny::Integer, dx::Real, dy::Real,
                             uf::AbstractMatrix, vf::AbstractMatrix, DT::Real,
                             is_solid::AbstractMatrix;
                             bc, source = nothing)
    nx = Int(nx); ny = Int(ny)
    n = nx * ny
    dx = Float64(dx); dy = Float64(dy); DT = Float64(DT)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    invdx2 = 1.0 / (dx * dx)
    invdy2 = 1.0 / (dy * dy)

    lin(i, j) = i + (j - 1) * nx

    _st_check_shape("uf", uf, nx, ny)
    _st_check_shape("vf", vf, nx, ny)
    _st_check_shape("is_solid", is_solid, nx, ny)
    source !== nothing && _st_check_shape("source", source, nx, ny)
    solid = Matrix{Bool}(is_solid)

    bcs = _st_boundary_bcs(bc, nx, ny)
    bc_w = bcs.west;  bc_e = bcs.east
    bc_s = bcs.south; bc_n = bcs.north

    I = Int[]; J = Int[]; V = Float64[]
    sizehint!(I, 5n); sizehint!(J, 5n); sizehint!(V, 5n)
    b = zeros(Float64, n)

    @inbounds for j in 1:ny, i in 1:nx
        k = lin(i, j)
        diag = 0.0

        if solid[i, j]
            push!(I, k); push!(J, k); push!(V, 1.0)
            continue
        end

        # Face mass fluxes (frozen). Interior faces come from uf/vf. The west
        # boundary face is reconstructed from the adjacent divergence-free face
        # balance because the inc_ns handoff stores its true west face separately
        # as `uwest`, not in the public uf matrix. Fluid-solid faces are
        # impermeable.
        Fe = (i < nx) ? (solid[i + 1, j] ? 0.0 : Float64(uf[i, j])) : Float64(uf[i, j])
        Fn = (j < ny) ? (solid[i, j + 1] ? 0.0 : Float64(vf[i, j])) : Float64(vf[i, j])
        Fs = (j > 1)  ? (solid[i, j - 1] ? 0.0 : Float64(vf[i, j - 1])) : Float64(vf[i, j])
        Fw = (i > 1)  ? (solid[i - 1, j] ? 0.0 : Float64(uf[i - 1, j])) :
                         _st_west_boundary_flux(uf, vf, solid, nx, ny, dx, dy, j)

        # ===== EAST face (x) =====
        if i < nx
            if !solid[i + 1, j]
                # diffusion
                push!(I, k); push!(J, lin(i + 1, j)); push!(V, -DT * invdx2)
                diag += DT * invdx2
                # advection (upwind on Fe)
                if Fe >= 0          # donor = own cell
                    diag += Fe * invdx
                else                # donor = east neighbour
                    push!(I, k); push!(J, lin(i + 1, j)); push!(V, Fe * invdx)
                end
            end
        else
            # east boundary
            kind = bc_e.kind[j]
            value = bc_e.value[j]
            if kind === :dirichlet
                diag += 2.0 * DT * invdx2
                b[k] += 2.0 * DT * invdx2 * value
                # advection: wall value is the donor when flow enters (Fe<0),
                # interior is donor when flow leaves (Fe>=0).
                if Fe >= 0
                    diag += Fe * invdx
                else
                    b[k] -= Fe * invdx * value
                end
            elseif kind === :flux
                # Neumann: zero-grad diffusion ghost (no diag term); flux source.
                b[k] += value * invdx
            elseif kind === :outflow
                # zero-gradient: ghost = interior; advect interior upwind value.
                diag += Fe * invdx
            end
        end

        # ===== WEST face (x) =====
        if i > 1
            if !solid[i - 1, j]
                push!(I, k); push!(J, lin(i - 1, j)); push!(V, -DT * invdx2)
                diag += DT * invdx2
                # advection: west face flux Fw, OUT of cell k is -Fw (Fw>0 enters k).
                # conv contribution is -(Fw·T_w)/dx. Upwind donor on Fw:
                if Fw >= 0          # flow enters k from west: donor = west neighbour
                    push!(I, k); push!(J, lin(i - 1, j)); push!(V, -Fw * invdx)
                else                # flow leaves k to the west: donor = own cell
                    diag += -Fw * invdx
                end
            end
        else
            # west boundary
            kind = bc_w.kind[j]
            value = bc_w.value[j]
            if kind === :dirichlet
                diag += 2.0 * DT * invdx2
                b[k] += 2.0 * DT * invdx2 * value
                # Contribution: -(Fw*T_face)/dx. Flow into domain (Fw>0) uses the
                # wall value; outflow uses the interior cell value.
                if Fw >= 0
                    b[k] += Fw * invdx * value
                else
                    diag += -Fw * invdx
                end
            elseif kind === :flux
                b[k] += value * invdx
            elseif kind === :outflow
                # zero-gradient: nothing for diffusion; advection carries interior.
                diag += -Fw * invdx
            end
        end

        # ===== NORTH face (y) =====
        if j < ny
            if !solid[i, j + 1]
                push!(I, k); push!(J, lin(i, j + 1)); push!(V, -DT * invdy2)
                diag += DT * invdy2
                if Fn >= 0
                    diag += Fn * invdy
                else
                    push!(I, k); push!(J, lin(i, j + 1)); push!(V, Fn * invdy)
                end
            end
        else
            kind = bc_n.kind[i]
            value = bc_n.value[i]
            if kind === :dirichlet
                diag += 2.0 * DT * invdy2
                b[k] += 2.0 * DT * invdy2 * value
                if Fn >= 0
                    diag += Fn * invdy
                else
                    b[k] -= Fn * invdy * value
                end
            elseif kind === :flux
                b[k] += value * invdy
            elseif kind === :outflow
                diag += Fn * invdy
            end
        end

        # ===== SOUTH face (y) =====
        if j > 1
            if !solid[i, j - 1]
                push!(I, k); push!(J, lin(i, j - 1)); push!(V, -DT * invdy2)
                diag += DT * invdy2
                if Fs >= 0          # flow enters k from south: donor = south neighbour
                    push!(I, k); push!(J, lin(i, j - 1)); push!(V, -Fs * invdy)
                else                # flow leaves k to the south: donor = own cell
                    diag += -Fs * invdy
                end
            end
        else
            kind = bc_s.kind[i]
            value = bc_s.value[i]
            if kind === :dirichlet
                diag += 2.0 * DT * invdy2
                b[k] += 2.0 * DT * invdy2 * value
                if Fs >= 0
                    b[k] += Fs * invdy * value
                else
                    diag += -Fs * invdy
                end
            elseif kind === :flux
                b[k] += value * invdy
            elseif kind === :outflow
                # zero-gradient.
                diag += -Fs * invdy
            end
        end

        # Optional volumetric source S (manufactured-solution / heat-generation
        # term): the equation becomes div(uT) − DT·lap(T) = S, contributing +S to
        # the RHS row. Defaults to no source.
        if source !== nothing
            b[k] += Float64(source[i, j])
        end

        push!(I, k); push!(J, k); push!(V, diag)
    end

    A = sparse(I, J, V, n, n)
    return A, b
end

function _st_linear_upwind_x(T::AbstractMatrix, solid::AbstractMatrix{Bool},
                             nx::Int, dx::Float64, i::Int, j::Int, F::Float64)
    if F >= 0.0
        low = Float64(T[i, j])
        if i > 1 && !solid[i - 1, j]
            return low + 0.5 * (low - Float64(T[i - 1, j]))
        end
    else
        low = Float64(T[i + 1, j])
        if i + 1 < nx && !solid[i + 2, j]
            return low + 0.5 * (low - Float64(T[i + 2, j]))
        end
    end
    return low
end

function _st_linear_upwind_y(T::AbstractMatrix, solid::AbstractMatrix{Bool},
                             ny::Int, dy::Float64, i::Int, j::Int, F::Float64)
    if F >= 0.0
        low = Float64(T[i, j])
        if j > 1 && !solid[i, j - 1]
            return low + 0.5 * (low - Float64(T[i, j - 1]))
        end
    else
        low = Float64(T[i, j + 1])
        if j + 1 < ny && !solid[i, j + 2]
            return low + 0.5 * (low - Float64(T[i, j + 2]))
        end
    end
    return low
end

function _st_advection_deferred_correction!(corr::AbstractVector{Float64},
                                            T::AbstractMatrix,
                                            uf::AbstractMatrix, vf::AbstractMatrix,
                                            solid::AbstractMatrix{Bool},
                                            nx::Int, ny::Int,
                                            dx::Float64, dy::Float64)
    fill!(corr, 0.0)
    lin(i, j) = i + (j - 1) * nx
    invdx = 1.0 / dx
    invdy = 1.0 / dy

    @inbounds for j in 1:ny, i in 1:nx - 1
        (solid[i, j] || solid[i + 1, j]) && continue
        F = Float64(uf[i, j])
        low = F >= 0.0 ? Float64(T[i, j]) : Float64(T[i + 1, j])
        high = _st_linear_upwind_x(T, solid, nx, dx, i, j, F)
        delta = F * (low - high) * invdx
        corr[lin(i, j)] += delta
        corr[lin(i + 1, j)] -= delta
    end

    @inbounds for j in 1:ny - 1, i in 1:nx
        (solid[i, j] || solid[i, j + 1]) && continue
        F = Float64(vf[i, j])
        low = F >= 0.0 ? Float64(T[i, j]) : Float64(T[i, j + 1])
        high = _st_linear_upwind_y(T, solid, ny, dy, i, j, F)
        delta = F * (low - high) * invdy
        corr[lin(i, j)] += delta
        corr[lin(i, j + 1)] -= delta
    end

    return corr
end

"""
    solve_scalar_transport(; nx, ny, dx, dy, uf, vf, DT, is_solid=falses(nx,ny),
                           bc, backend=nothing, advection=:linear_upwind,
                           deferred_passes=4)

Standalone DECOUPLED STEADY scalar advection–diffusion ("thermal transport")
solver. Given a FROZEN face-normal velocity field, solve

    ∇·(u T) − DT ∇²T = 0

to steady state on a collocated cell-centred grid `nx × ny` with spacing
`dx, dy`. The assembled matrix is always the legacy first-order upwind
advection plus diffusion operator. With `advection=:linear_upwind` (default), a
second-order linear-upwind face flux is applied by deferred correction:
`A*T(k+1) = b + F_low(T(k)) - F_high(T(k))`. The factorization is reused for
every Picard RHS. With `advection=:upwind`, the legacy single-solve path is
preserved.

Arguments:
  `uf, vf`   FROZEN face-normal velocities. `uf[i,j]` = east face of cell
             `(i,j)` (x-normal), `vf[i,j]` = north face (y-normal). SAME layout
             as `inc_ns/simple.jl`.
  `DT`       scalar diffusivity (thermal diffusivity / conductivity proxy).
  `is_solid` full-cell solid mask. Solid rows are pinned to `T=0`, and
             fluid-solid faces are adiabatic/impermeable.
  `bc`       NamedTuple keyed `west/east/south/north`, each a
             whole-side `(kind::Symbol, value)` BC or a vector of segments
             `(lo, hi, kind, value)`. Segment ranges are cell indices along
             that side: `j` for west/east, `i` for south/north. Uncovered
             segment cells default to adiabatic `(kind=:flux, value=0)`.
               `:dirichlet` — imposed inlet/wall value `T = value`.
               `:flux`      — Neumann wall heat flux `q = value`; enters as a
                              SOURCE in `b` on the wall-adjacent row (q/h), NOT
                              in `A`; no advective boundary transport.
               `:outflow`   — zero-gradient (ghost = interior); advection carries
                              the interior upwind value.
  `source`   optional volumetric source field `S[i,j]` (heat generation /
             manufactured-solution term): the balance becomes
             `div(uT) − DT·lap(T) = S`. `nothing` (default) means no source.
  `advection` `:linear_upwind` for deferred second-order advection, or
             `:upwind` for the legacy first-order single solve.
  `deferred_passes` maximum Picard correction solves after the initial upwind
             solve; the factorization is reused.
  `deferred_tol` relative infinity-norm update tolerance for stopping the
             deferred correction.
  `backend`  optional backend tag; `nothing` (default) routes to the CPU seam.

Assembly mirrors `inc_ns/simple.jl` (5-point Laplacian → sparse CSC) and the
cavity's first-order upwind advection (`_cavity_convection!`), transcribed into
matrix coefficients (donor diagonal `+F`, acceptor off-diagonal `−F`). The solve
uses the shared factorize-once seam with `spd=false` (A is non-symmetric due to
upwind).

Returns a NamedTuple with fields:
  `T`                  steady scalar field (`nx × ny`).
  `residual_history`   single-element `[‖A·T − b‖]` (should be ~machine eps).
  `iters`              always `1` (single linear solve, no loop).
  `converged`          whether the residual is at machine-noise level.
  `dx, dy, ycenters`   grid metrics for analytic comparison.
  `nx, ny, DT`         echoed inputs.
  `Pe_cell`            cell Péclet number `max|u|·min(dx,dy)/DT`.
"""
function solve_scalar_transport(; nx::Integer, ny::Integer, dx::Real, dy::Real,
                                uf::AbstractMatrix, vf::AbstractMatrix, DT::Real,
                                is_solid::AbstractMatrix = falses(Int(nx), Int(ny)),
                                bc, source = nothing, backend = nothing,
                                advection::Symbol = :linear_upwind,
                                deferred_passes::Integer = 4,
                                deferred_tol::Real = 1e-8)
    nx = Int(nx); ny = Int(ny)
    dx = Float64(dx); dy = Float64(dy); DT = Float64(DT)
    scheme = _st_advection_scheme(advection)
    max_deferred = Int(deferred_passes)
    max_deferred >= 0 ||
        throw(ArgumentError("deferred_passes must be non-negative; got $deferred_passes"))
    tol_deferred = Float64(deferred_tol)
    tol_deferred > 0.0 ||
        throw(ArgumentError("deferred_tol must be positive; got $deferred_tol"))
    ycenters = [(j - 0.5) * dy for j in 1:ny]

    A, b = _st_assemble_system(nx, ny, dx, dy, uf, vf, DT, is_solid;
                               bc = bc, source = source)

    # A is non-symmetric (first-order upwind advection) -> spd=false routes the
    # seam to a sparse LU (the CPU spd=false branch gates on `issymmetric(A)`
    # and factorizes genuinely non-symmetric operators with `lu`). The solve is
    # a single factorize + solve through the shared factorize-once seam; the
    # cache would amortize re-solves if T were ever re-solved on a fresh RHS.
    btag = backend === nothing ? CPUBackendTag() : backend
    cache = lin_factorize(A; backend = btag, spd = false)
    Tvec = lin_solve!(cache, b)
    Tvec = Vector{Float64}(Tvec)
    T = reshape(Tvec, nx, ny)

    # Residual ‖A·T − b‖ (should be ~machine eps for a direct solve).
    resid = norm(A * Tvec .- b)
    residual_history = [resid]
    linear_converged = resid <= 1e-8 * max(norm(b), 1.0)

    deferred_passes_used = 0
    deferred_converged = scheme === :upwind
    deferred_rel_change = 0.0

    if scheme === :linear_upwind && max_deferred > 0
        solid = Matrix{Bool}(is_solid)
        corr = zeros(Float64, length(b))
        rhs = similar(b)
        _st_advection_deferred_correction!(corr, T, uf, vf, solid, nx, ny, dx, dy)
        if norm(corr, Inf) == 0.0
            deferred_converged = true
        else
            for _ in 1:max_deferred
                @. rhs = b + corr
                next = Vector{Float64}(lin_solve!(cache, rhs))
                lin_resid = norm(A * next .- rhs)
                push!(residual_history, lin_resid)
                denom = max(norm(next, Inf), eps(Float64))
                deferred_rel_change = norm(next .- Tvec, Inf) / denom
                deferred_passes_used += 1
                Tvec = next
                T = reshape(Tvec, nx, ny)
                linear_converged = lin_resid <= 1e-8 * max(norm(rhs), 1.0)
                if deferred_rel_change <= tol_deferred
                    deferred_converged = true
                    break
                end
                _st_advection_deferred_correction!(corr, T, uf, vf, solid,
                                                   nx, ny, dx, dy)
            end
        end
    end

    # Cell Péclet number Pe = max|u|·min(dx,dy)/DT.
    umax = max(maximum(abs, uf), maximum(abs, vf))
    Pe_cell = umax * min(dx, dy) / DT

    iters = 1 + deferred_passes_used
    converged = linear_converged
    return (; T, residual_history, iters, converged, dx, dy, ycenters,
            nx, ny, DT, Pe_cell, advection = scheme,
            deferred_passes_used, deferred_converged, deferred_rel_change)
end
