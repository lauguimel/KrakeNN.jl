# Standalone steady SIMPLE incompressible solver for the 2D lid-driven cavity.
#
# Collocated cell-centred grid on the unit square [0,1]^2. Top wall (north)
# moves at u = U_lid, v = 0; the other three walls are no-slip (u = v = 0).
# Re = U_lid * L / nu with L = 1, rho = 1 => mu = nu.
#
# This exercises the full SIMPLE pressure-velocity coupling (Poiseuille did not,
# because there p' ~ 0). The momentum equation is solved with an IMPLICIT
# symmetric viscous Laplacian (assembled once, CHOLMOD factorise-once) plus
# advection by DEFERRED CORRECTION (u.grad(u) evaluated from the current field
# and moved to the RHS) and momentum under-relaxation. Pressure-velocity coupling
# uses a proper Rhie-Chow face-velocity interpolation; the pressure-correction
# Poisson is all-Neumann (singular) and is pinned at one reference dof. There is
# NO divergence-floor gate here (that workaround was only valid for Poiseuille).
#
# Reuses the matrix-free KA grad/div operators conceptually but, to stay
# self-contained (registered in `src/KrakeNN.jl`; the platform wrapper
# IncNS(:cavity) forwards here), assembles its own SPD
# operators and uses compact transpose-consistent stencils, exactly like
# simple.jl. KA + stdlib only, CPU.
#
# Public entry point:
#   solve_incns_cavity(; nx, ny, U_lid, Re, relax, tol, maxiter, backend=CPU())
#     -> NamedTuple(u, v, p, residual_history, iters, converged, ...)

using KernelAbstractions
using LinearAlgebra
using SparseArrays

if !isdefined(@__MODULE__, :pin_reference_dof)
    include(joinpath(@__DIR__, "..", "..", "solve", "poisson.jl"))
end
# Factorize-once seam (poisson.jl includes it, but guard for standalone use).
if !isdefined(@__MODULE__, :lin_factorize)
    include(joinpath(@__DIR__, "..", "..", "solve", "linear_solve.jl"))
end

# ---------------------------------------------------------------------------
# SPD (-Laplacian) assembly for a cell-centred grid with per-wall Dirichlet /
# Neumann boundary conditions.
#
# Layout: k = i + (j-1)*nx, i in 1:nx (x), j in 1:ny (y).
# For a wall face with Dirichlet condition the ghost is u_g = 2*u_wall - u_c,
# so the missing face term (u_g - u_c)/h^2 = (2*u_wall - 2*u_c)/h^2 contributes
# +2/h^2 to the diagonal (the wall-value part 2*u_wall/h^2 is a RHS source,
# handled separately by _cavity_dirichlet_rhs!). For a Neumann wall the ghost
# is u_g = u_c, so no face term is added.
#
# Returned A is the POSITIVE-definite discrete (-Laplacian).
# ---------------------------------------------------------------------------
function _cavity_assemble_neg_laplacian(nx::Integer, ny::Integer, dx::Real, dy::Real;
                                        bc_w::Symbol, bc_e::Symbol,
                                        bc_s::Symbol, bc_n::Symbol)
    nx = Int(nx); ny = Int(ny)
    n = nx * ny
    invdx2 = 1.0 / (Float64(dx)^2)
    invdy2 = 1.0 / (Float64(dy)^2)
    lin(i, j) = i + (j - 1) * nx

    I = Int[]; J = Int[]; V = Float64[]
    sizehint!(I, 5n); sizehint!(J, 5n); sizehint!(V, 5n)

    @inbounds for j in 1:ny, i in 1:nx
        k = lin(i, j)
        diag = 0.0
        # west (i-1)
        if i > 1
            push!(I, k); push!(J, lin(i - 1, j)); push!(V, -invdx2); diag += invdx2
        elseif bc_w === :dirichlet
            diag += 2.0 * invdx2
        end
        # east (i+1)
        if i < nx
            push!(I, k); push!(J, lin(i + 1, j)); push!(V, -invdx2); diag += invdx2
        elseif bc_e === :dirichlet
            diag += 2.0 * invdx2
        end
        # south (j-1)
        if j > 1
            push!(I, k); push!(J, lin(i, j - 1)); push!(V, -invdy2); diag += invdy2
        elseif bc_s === :dirichlet
            diag += 2.0 * invdy2
        end
        # north (j+1)
        if j < ny
            push!(I, k); push!(J, lin(i, j + 1)); push!(V, -invdy2); diag += invdy2
        elseif bc_n === :dirichlet
            diag += 2.0 * invdy2
        end
        push!(I, k); push!(J, k); push!(V, diag)
    end
    return sparse(I, J, V, n, n)
end

# Dirichlet wall-value source for the (-Laplacian): for each Dirichlet wall face
# the ghost u_g = 2*u_wall - u_c contributes +2*u_wall/h^2 to (-Lap u)_c (moved
# to the RHS). uw_*,_ are the wall velocity values per side. Adds into `src`.
function _cavity_dirichlet_rhs!(src, nx, ny, dx, dy;
                                bc_w, bc_e, bc_s, bc_n,
                                uw_w, uw_e, uw_s, uw_n)
    invdx2 = 1.0 / (Float64(dx)^2)
    invdy2 = 1.0 / (Float64(dy)^2)
    fill!(src, 0.0)
    @inbounds for j in 1:ny, i in 1:nx
        if i == 1 && bc_w === :dirichlet
            src[i, j] += 2.0 * invdx2 * uw_w
        end
        if i == nx && bc_e === :dirichlet
            src[i, j] += 2.0 * invdx2 * uw_e
        end
        if j == 1 && bc_s === :dirichlet
            src[i, j] += 2.0 * invdy2 * uw_s
        end
        if j == ny && bc_n === :dirichlet
            src[i, j] += 2.0 * invdy2 * uw_n
        end
    end
    return nothing
end

# CHOLMOD factor wrapper with optional reference-dof pinning (singular pressure).
# Thin wrappers over the shared factorize-once seam (linear_solve.jl): the
# constant viscous momentum operator and the constant pressure Laplacian are each
# factorized ONCE and reused across all ~3000 outer iterations. cuDSS drops in on
# GPU by swapping the backend tag. Returns a LinearSolveCache.
function _cavity_factorise(A::SparseMatrixCSC{Float64,Int}; pin_k0::Integer=0)
    return lin_factorize(A; backend=CPUBackendTag(), spd=true, pin_k0=Int(pin_k0))
end

# Solve A * x = b reusing the cached factorization (pinning handled in the cache).
function _cavity_solve!(cache::LinearSolveCache, ::SparseMatrixCSC{Float64,Int}, b::Vector{Float64})
    return lin_solve!(cache, b)
end

# ---------------------------------------------------------------------------
# Convective term by deferred correction, evaluated on the current cell field.
#
# Builds the discrete advection -div(u_face * phi) for phi = u and phi = v,
# using Rhie-Chow face velocities (uf,vf) for the advecting flux and a first-
# order upwind reconstruction of the advected quantity (robust at Re=100). The
# result conv_u, conv_v has units of d(phi)/dt and is moved to the momentum RHS:
#   (A_mom) u* = -conv_u - dp/dx + dirichlet_src.
# (rho = 1; conv carries the sign convention conv = +div(u phi).)
#
# Wall fluxes: the face velocity uf/vf vanishes on solid walls (no-slip) except
# the north face on the lid, where the lid carries u=U_lid; that wall flux is
# included so the lid injects x-momentum. The advected face value at a wall uses
# the prescribed wall velocity.
# ---------------------------------------------------------------------------
function _cavity_convection!(conv_u, conv_v, u, v, uf, vf, dx, dy, nx, ny,
                             U_lid)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    @inbounds for j in 1:ny, i in 1:nx
        # face mass fluxes (x-normal east/west, y-normal north/south)
        Fe = (i < nx) ? uf[i, j] : 0.0              # east interior face vel
        Fw = (i > 1) ? uf[i - 1, j] : 0.0           # west = east face of west nb
        Fn = (j < ny) ? vf[i, j] : 0.0              # north interior face vel
        Fs = (j > 1) ? vf[i, j - 1] : 0.0           # south face

        # advected u (upwind). Wall ghost values: walls are no-slip (u=0) except
        # north lid (u=U_lid). East/west/south walls -> u_wall = 0.
        uE = (i < nx) ? (Fe >= 0 ? u[i, j] : u[i + 1, j]) : 0.0
        uW = (i > 1) ? (Fw >= 0 ? u[i - 1, j] : u[i, j]) : 0.0
        uN = (j < ny) ? (Fn >= 0 ? u[i, j] : u[i, j + 1]) :
             # north wall: lid carries U_lid; flux there is 0 (vf=0) so value
             # is irrelevant for u-advection but the wall convective flux of u
             # across the moving lid is handled via the viscous BC, not here.
             U_lid
        uS = (j > 1) ? (Fs >= 0 ? u[i, j - 1] : u[i, j]) : 0.0

        # advected v (upwind). All walls v_wall = 0 (incl. lid: v=0).
        vE = (i < nx) ? (Fe >= 0 ? v[i, j] : v[i + 1, j]) : 0.0
        vW = (i > 1) ? (Fw >= 0 ? v[i - 1, j] : v[i, j]) : 0.0
        vN = (j < ny) ? (Fn >= 0 ? v[i, j] : v[i, j + 1]) : 0.0
        vS = (j > 1) ? (Fs >= 0 ? v[i, j - 1] : v[i, j]) : 0.0

        conv_u[i, j] = (Fe * uE - Fw * uW) * invdx + (Fn * uN - Fs * uS) * invdy
        conv_v[i, j] = (Fe * vE - Fw * vW) * invdx + (Fn * vN - Fs * vS) * invdy
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Rhie-Chow face velocities for the cavity.
#
# Face velocity (interior faces only; wall faces are 0 by no-slip, and the north
# lid face carries v=0 too) =
#   ubar - dbar * (compact face pressure-grad - averaged cell pressure-grad).
# d_u,d_v are the SIMPLE response coefficients (αu / a_p with the under-relaxed
# diagonal). gpx,gpy are the current cell pressure gradients (+dp/dx,+dp/dy).
#
# uf[i,j] is the EAST face of cell (i,j) (defined for i in 1:nx-1).
# vf[i,j] is the NORTH face of cell (i,j) (defined for j in 1:ny-1).
# ---------------------------------------------------------------------------
function _cavity_rhie_chow_faces!(uf, vf, u, v, p, gpx, gpy, d_u, d_v,
                                  dx, dy, nx, ny)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    fill!(uf, 0.0)
    fill!(vf, 0.0)
    @inbounds for j in 1:ny, i in 1:nx
        if i < nx
            ubar = 0.5 * (u[i, j] + u[i + 1, j])
            dbar = 0.5 * (d_u[i, j] + d_u[i + 1, j])
            gp_face = (p[i + 1, j] - p[i, j]) * invdx
            gp_cell = 0.5 * (gpx[i, j] + gpx[i + 1, j])
            uf[i, j] = ubar - dbar * (gp_face - gp_cell)
        end
        if j < ny
            vbar = 0.5 * (v[i, j] + v[i, j + 1])
            dbar = 0.5 * (d_v[i, j] + d_v[i, j + 1])
            gp_face = (p[i, j + 1] - p[i, j]) * invdy
            gp_cell = 0.5 * (gpy[i, j] + gpy[i, j + 1])
            vf[i, j] = vbar - dbar * (gp_face - gp_cell)
        end
    end
    return nothing
end

# Divergence of the Rhie-Chow face field (cell-centred). Wall faces are 0.
# uf[i,j]=east face of (i,j); the west face of (i,j) is uf[i-1,j].
function _cavity_face_divergence!(div, uf, vf, dx, dy, nx, ny)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    @inbounds for j in 1:ny, i in 1:nx
        ue = (i < nx) ? uf[i, j] : 0.0
        uw = (i > 1) ? uf[i - 1, j] : 0.0
        vn = (j < ny) ? vf[i, j] : 0.0
        vs = (j > 1) ? vf[i, j - 1] : 0.0
        div[i, j] = (ue - uw) * invdx + (vn - vs) * invdy
    end
    return nothing
end

# Compact cell-centred pressure gradient consistent (discrete transpose) with the
# face divergence + the all-Neumann pressure Laplacian: wall faces use a zero
# normal gradient (homogeneous Neumann), interior faces use the average value.
# gx = +dp/dx, gy = +dp/dy.
function _cavity_compact_gradient!(gx, gy, p, dx, dy, nx, ny)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    @inbounds for j in 1:ny, i in 1:nx
        pe = (i < nx) ? 0.5 * (p[i, j] + p[i + 1, j]) : p[i, j]   # east face
        pw = (i > 1) ? 0.5 * (p[i - 1, j] + p[i, j]) : p[i, j]    # west face
        gx[i, j] = (pe - pw) * invdx
        pn = (j < ny) ? 0.5 * (p[i, j] + p[i, j + 1]) : p[i, j]   # north face
        ps = (j > 1) ? 0.5 * (p[i, j - 1] + p[i, j]) : p[i, j]    # south face
        gy[i, j] = (pn - ps) * invdy
    end
    return nothing
end

# Correct the FACE velocities in place from the Poisson solution `pcorr`, where
# Ap*pcorr = div(u*) with Ap = dscalar*(-Lap). The face divergence of
# d^f*grad_face(pcorr) equals -Ap*pcorr = -div(u*) (the assembled operator is the
# NEGATIVE Laplacian), so adding d^f*grad_face(pcorr) to the faces annihilates
# div(u*) in one shot:
#   uf[i,j] += d^f_east * (pcorr[i+1,j] - pcorr[i,j]) / dx   (interior x-faces),
#   vf[i,j] += d^f_north * (pcorr[i,j+1] - pcorr[i,j]) / dy  (interior y-faces).
# Wall faces carry no correction (no-slip + homogeneous Neumann pcorr).
function _cavity_correct_faces!(uf, vf, pcorr, d_u, d_v, dx, dy, nx, ny)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    @inbounds for j in 1:ny, i in 1:nx
        if i < nx
            dbar = 0.5 * (d_u[i, j] + d_u[i + 1, j])
            uf[i, j] += dbar * (pcorr[i + 1, j] - pcorr[i, j]) * invdx
        end
        if j < ny
            dbar = 0.5 * (d_v[i, j] + d_v[i, j + 1])
            vf[i, j] += dbar * (pcorr[i, j + 1] - pcorr[i, j]) * invdy
        end
    end
    return nothing
end

# Checkerboard metric: high-frequency content of the pressure field. We measure
# the ratio of the discrete (i+j) odd/even oscillation energy to the total
# pressure variance. A smooth field gives ~0; a checkerboard mode gives ~1.
function _cavity_checkerboard_metric(p, nx, ny)
    pbar = sum(p) / length(p)
    osc = 0.0
    tot = 0.0
    @inbounds for j in 2:ny-1, i in 2:nx-1
        # discrete Laplacian-like high-pass (5-point) normalised
        lap = p[i + 1, j] + p[i - 1, j] + p[i, j + 1] + p[i, j - 1] - 4 * p[i, j]
        osc += lap^2
        tot += (p[i, j] - pbar)^2
    end
    # normalise oscillation energy by variance; a smooth resolved field has
    # bounded ratio, a checkerboard saturates this near the stencil maximum.
    return sqrt(osc / max(tot, eps()))
end

"""
    solve_incns_cavity(; nx=128, ny=128, U_lid=1.0, Re=100.0,
                       relax=(u=0.7, p=0.3), tol=1e-7, maxiter=2000,
                       L=1.0, backend=CPU())

Steady SIMPLE incompressible solver for the 2D lid-driven cavity on `[0,L]^2`.

Top wall moves at `u = U_lid, v = 0`; other walls no-slip. `Re = U_lid*L/nu`,
`rho = 1` so `mu = nu = U_lid*L/Re`. Collocated cell-centred grid `nx x ny`.

Momentum: implicit symmetric viscous (-Laplacian) (CHOLMOD, factor-once) with
deferred-correction upwind advection moved to the RHS, momentum under-relaxation
`relax.u`. Pressure-velocity coupling: Rhie-Chow face velocities; the all-Neumann
pressure-correction Poisson is pinned at dof 1; pressure under-relaxation
`relax.p`. Converges when BOTH the continuity residual and the velocity-settle
fall below `tol`.

Returns a NamedTuple with `u, v, p` (nx x ny), `residual_history`, `iters`,
`converged`, plus grid metrics (`dx, dy, xcenters, ycenters`) and `nx, ny,
U_lid, Re, mu, L, checkerboard`.
"""
function solve_incns_cavity(; nx::Integer=128, ny::Integer=128,
                            U_lid::Real=1.0, Re::Real=100.0,
                            relax=(u=0.7, p=0.3),
                            tol::Real=1e-7, maxiter::Integer=2000,
                            L::Real=1.0, backend=CPU(),
                            verbose::Bool=false)
    nx = Int(nx); ny = Int(ny)
    U_lid = Float64(U_lid); Re = Float64(Re); L = Float64(L)
    αu = Float64(relax.u); αp = Float64(relax.p)
    mu = U_lid * L / Re            # rho = 1
    dx = L / nx
    dy = L / ny
    xcenters = [(i - 0.5) * dx for i in 1:nx]
    ycenters = [(j - 0.5) * dy for j in 1:ny]

    # ----- viscous (-Laplacian), Dirichlet on all four walls -----
    Lmom = _cavity_assemble_neg_laplacian(nx, ny, dx, dy;
                                          bc_w=:dirichlet, bc_e=:dirichlet,
                                          bc_s=:dirichlet, bc_n=:dirichlet)
    Amom_visc = mu .* Lmom                       # SPD viscous operator
    ap_visc = Vector(diag(Amom_visc))            # viscous diagonal, length n

    # Momentum under-relaxation built into the diagonal: solve
    #   (Amom_visc + (1/αu - 1) * Diag(ap_visc)) u* =
    #        b_visc - conv - grad p + (1/αu - 1) * Diag(ap_visc) * u_old
    # i.e. A_relaxed = Amom_visc + (1/αu - 1)*Diag(ap_visc). The added diagonal
    # keeps the operator SPD and dominant, so CHOLMOD stays valid.
    extra = (1.0 / αu - 1.0)
    Drelax = spdiagm(0 => extra .* ap_visc)
    Amom = Amom_visc + Drelax
    ap_relax = Vector(diag(Amom))                # relaxed diagonal a_p
    ap_visc_mat = reshape(ap_visc, nx, ny)       # matrix form for RHS broadcast
    mom_op = _cavity_factorise(Amom)

    # SIMPLE d-coefficient: d = αu / a_p (response of velocity to pressure grad).
    # Using the relaxed a_p (standard SIMPLE under-relaxed d).
    ap_mat = reshape(ap_relax, nx, ny)
    d_u = αu ./ ap_mat
    d_v = αu ./ ap_mat

    # ----- pressure-correction operator: d * (-Laplacian), all-Neumann -----
    Lp = _cavity_assemble_neg_laplacian(nx, ny, dx, dy;
                                        bc_w=:neumann, bc_e=:neumann,
                                        bc_s=:neumann, bc_n=:neumann)
    dscalar = sum(d_u) / length(d_u)
    Ap = dscalar .* Lp
    p_op = _cavity_factorise(Ap; pin_k0=1)       # singular -> pin reference dof

    # ----- Dirichlet velocity wall-value sources for momentum -----
    # u: walls W/E/S = 0, N (lid) = U_lid.
    src_u = zeros(Float64, nx, ny)
    _cavity_dirichlet_rhs!(src_u, nx, ny, dx, dy;
                           bc_w=:dirichlet, bc_e=:dirichlet,
                           bc_s=:dirichlet, bc_n=:dirichlet,
                           uw_w=0.0, uw_e=0.0, uw_s=0.0, uw_n=U_lid)
    src_u .*= mu
    # v: all walls 0.
    src_v = zeros(Float64, nx, ny)               # all-zero (no-slip v walls)

    # ----- fields -----
    u = zeros(Float64, nx, ny)
    v = zeros(Float64, nx, ny)
    p = zeros(Float64, nx, ny)
    gpx = zeros(Float64, nx, ny)
    gpy = zeros(Float64, nx, ny)
    uf = zeros(Float64, nx, ny)
    vf = zeros(Float64, nx, ny)
    conv_u = zeros(Float64, nx, ny)
    conv_v = zeros(Float64, nx, ny)
    divstar = zeros(Float64, nx, ny)
    pcorr = zeros(Float64, nx, ny)

    residual_history = Float64[]
    converged = false
    iters = 0
    vel_change = Inf

    # reference flux scale for residual normalisation (lid-driven)
    ref_flux = U_lid

    for it in 1:maxiter
        iters = it

        # ---- 1. current pressure gradient (cell, for the momentum source and
        #         for the Rhie-Chow cell-gradient term) ----
        _cavity_compact_gradient!(gpx, gpy, p, dx, dy, nx, ny)

        # ---- 2. Rhie-Chow face velocities from current u,v,p ----
        _cavity_rhie_chow_faces!(uf, vf, u, v, p, gpx, gpy, d_u, d_v,
                                 dx, dy, nx, ny)

        # ---- 3. deferred-correction convection on current field ----
        _cavity_convection!(conv_u, conv_v, u, v, uf, vf, dx, dy, nx, ny, U_lid)

        # ---- 4. momentum predictor (implicit viscous, deferred advection) ----
        # A_relaxed u* = b_visc - conv - grad p + relax_old_term.
        # relax_old_term = (1/αu - 1)*ap_visc .* u_old.
        bu = vec(src_u .- conv_u .- gpx .+ (extra .* ap_visc_mat) .* u)
        bv = vec(src_v .- conv_v .- gpy .+ (extra .* ap_visc_mat) .* v)
        ustar = reshape(_cavity_solve!(mom_op, Amom, bu), nx, ny)
        vstar = reshape(_cavity_solve!(mom_op, Amom, bv), nx, ny)

        umax_prev = max(maximum(abs, u), maximum(abs, v), U_lid * eps())
        du = 0.0
        @inbounds for idx in eachindex(u)
            du = max(du, abs(ustar[idx] - u[idx]), abs(vstar[idx] - v[idx]))
            u[idx] = ustar[idx]
            v[idx] = vstar[idx]
        end
        vel_change = du / umax_prev

        # ---- 5. Rhie-Chow faces from the predictor (continuity RHS) ----
        # uf*, vf* = ubar - d^f (face grad p - avg cell grad p). The cell-grad
        # term decouples the checkerboard mode; it stays attached to the CURRENT
        # pressure p. The pressure-correction below uses the PURE face gradient,
        # so its face divergence is exactly Ap = dscalar*Lp (5-point), making the
        # face projection consistent and idempotent.
        _cavity_compact_gradient!(gpx, gpy, p, dx, dy, nx, ny)
        _cavity_rhie_chow_faces!(uf, vf, u, v, p, gpx, gpy, d_u, d_v,
                                 dx, dy, nx, ny)

        # ---- 6. continuity residual = div(u*_face) ----
        _cavity_face_divergence!(divstar, uf, vf, dx, dy, nx, ny)
        res = sqrt(sum(abs2, divstar) / length(divstar)) * dx / max(ref_flux, eps())
        push!(residual_history, res)

        if verbose && (it <= 5 || it % 100 == 0)
            @info "cavity SIMPLE" it res vel_change
        end

        if res < tol && vel_change < tol
            converged = true
            break
        end

        # ---- 7. pressure-correction Poisson: Ap p' = div(u*_face) ----
        bp = vec(divstar)
        pcorr .= reshape(_cavity_solve!(p_op, Ap, bp), nx, ny)
        pcorr .-= sum(pcorr) / length(pcorr)   # Neumann gauge

        # ---- 8. correct the FACE velocities directly (exact projection) ----
        # The assembled Ap = dscalar*(-Lap) solves Ap*pcorr = div(u*), so pcorr is
        # MINUS the physical pressure correction p'_phys = -pcorr. The physical
        # face/cell velocity correction u' = -d*grad(p'_phys) = +d*grad(pcorr),
        # and the pressure update is p += αp*p'_phys = p - αp*pcorr. With the +
        # sign the face divergence is annihilated in one shot (verified).
        _cavity_correct_faces!(uf, vf, pcorr, d_u, d_v, dx, dy, nx, ny)

        # ---- 9. correct CELL velocities with the cell pressure-corr gradient ----
        _cavity_compact_gradient!(gpx, gpy, pcorr, dx, dy, nx, ny)
        @. u = u + d_u * gpx
        @. v = v + d_v * gpy

        # ---- 10. correct pressure (under-relaxed) ----
        @. p = p - αp * pcorr
    end

    checkerboard = _cavity_checkerboard_metric(p, nx, ny)

    return (; u, v, p, residual_history, iters, converged, vel_change,
            dx, dy, xcenters, ycenters, nx, ny, U_lid, Re, mu, L, checkerboard)
end
