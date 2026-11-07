# Standalone UNSTEADY incompressible Navier-Stokes driver (2D, laminar,
# Newtonian): incremental pressure-correction projection (fractional step).
#
# Collocated cell-centred grid, same layout and conventions as simple.jl /
# cavity.jl (k = i + (j-1)*nx). Per-axis boundary conditions: :periodic or
# :wall (no-slip u = v = 0; homogeneous Dirichlet for velocity, homogeneous
# Neumann for pressure). Validated cases: Taylor-Green vortex (periodic x/y)
# and impulsively-started body-force plane Poiseuille (periodic x, walls y).
#
# TIME DISCRETISATION per step (dt, nu constant):
#   * advection  EXPLICIT Adams-Bashforth 2 (Euler on the first step),
#     conservative central (2nd-order) fluxes advected by the divergence-free
#     FACE velocities left by the previous projection.
#   * diffusion  IMPLICIT theta scheme: Crank-Nicolson (theta = 1/2, default,
#     2nd order in time) or backward Euler (theta = 1, robust fallback).
#   * pressure   incremental correction: the predictor carries grad(p^n);
#     ONE pressure Poisson per step yields the increment phi = p^{n+1} - p^n.
#
# Momentum predictor (per velocity component, f = constant body force):
#   (I/dt + nu*theta*(-Lap)) u* = u^n/dt - conv_AB2 - grad(p^n) + f
#                                 - nu*(1-theta)*(-Lap) u^n
# Projection (face velocities feed the Poisson RHS):
#   (-Lap) phi = -div(uf*)/dt
#   uf^{n+1} = uf* - dt*grad_face(phi)      (face divergence -> 0 exactly)
#   u^{n+1}  = u*  - dt*grad_cell(phi)      (compact cell gradient)
#   p^{n+1}  = p^n + phi
#
# FACE VELOCITIES / CHECKERBOARD (collocated grid): a dedicated face-velocity
# field is reconstructed from the predictor, fed to the Poisson, and projected
# EXACTLY each step (compact face divergence annihilated to machine
# precision); the advecting fluxes are these divergence-free faces. Two
# momentum interpolations are available (`rhie_chow` kwarg):
#   :increment (default) — Zang/Kim-&-Choi incremental momentum
#     interpolation: uf* = avg(u*). The compact pressure-difference coupling
#     (the Rhie-Chow mechanism) rides on the face projection of the increment
#     phi, so the face-vs-cell deviation is O(dt^2*h^2) and the scheme is
#     cleanly 2nd order in time (measured 2.00 on Taylor-Green
#     self-convergence at fixed grid).
#   :full — classical d = dt Rhie-Chow deviation against the FULL pressure:
#     uf* = avg(u*) - dt*(grad_f p^n - avg grad_c p^n). Strongest
#     pressure-velocity coupling (actively annihilates the p-checkerboard
#     mode every step) but the dt-scaled deviation against the FULL pressure
#     enters the advecting flux at O(dt*h^2): measured temporal order
#     degrades to ~1.1-1.6 (the classic time-step-dependent momentum
#     interpolation defect, Choi IJNMF 1999). Prefer for steady-dominated or
#     strongly pressure-coupled runs; use :increment for time-accurate runs.
#
# SIGN NOTE (the cavity.jl trap): the assembled pressure operator is the
# POSITIVE-definite (-Laplacian). cavity.jl solves Ap*pcorr = +div(u*) so its
# pcorr is MINUS the physical correction. Here the minus sign is placed in the
# RHS (b = -div(uf*)/dt), so phi IS the physical pressure increment and the
# updates above use plain minus/plus signs. Mixing the two conventions
# diverges within a few steps — do not "fix" one side without the other.
#
# FACTORIZE-ONCE: both implicit operators are constant (geometry + dt + nu
# only), so they are assembled and factorized ONCE before the time loop via
# the lin_factorize/lin_solve! seam (linear_solve.jl); the loop only performs
# back-substitutions. `nfactorizations == 2` for ANY number of steps while
# `nlinsolves == 3*nsteps` (2 momentum + 1 Poisson per step). cuDSS drops in
# on GPU by swapping the backend tag in the seam.
#
# Backend-parametric like simple.jl: takes `backend = CPU()` (KA) so the GPU
# path can slot in later; VALIDATED ON CPU (plain loops + CHOLMOD here).
# KernelAbstractions + stdlib only. Registered in `src/KrakeNN.jl` (exported
# driver; the platform wrapper IncNS(:projection) in method.jl forwards here).
# Include guards keep it standalone-include-able.
#
# Public entry point:
#   solve_incns_projection(; nx, ny, Lx, Ly, nu, dt, nsteps, bc_x, bc_y,
#                          fx, fy, u0, v0, p0, scheme, backend, callback)
#     -> NamedTuple(u, v, p, uf, vf, ke_history, max_div_inf, ...)

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
# Sparse SPD (-Laplacian) assembly, per-axis BCs. Copied locally from
# simple.jl `_incns_assemble_neg_laplacian` (same convention, kept standalone
# like cavity.jl does for its helpers).
#   :periodic   -> wrap neighbour.
#   :dirichlet0 -> wall ghost u_g = -u_c (value 0 at the wall FACE): +2/h^2 on
#                  the diagonal per wall face (cavity.jl velocity convention).
#   :neumann    -> wall ghost u_g = u_c: no face term (pressure walls).
# Returned A is the POSITIVE-definite discrete (-Laplacian).
# ---------------------------------------------------------------------------
function _proj_assemble_neg_laplacian(nx::Integer, ny::Integer, dx::Real, dy::Real;
                                      bc_x::Symbol, bc_y::Symbol)
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

        # ---- x-direction ----
        if i < nx
            push!(I, k); push!(J, lin(i + 1, j)); push!(V, -invdx2)
            diag += invdx2
        elseif bc_x === :periodic
            push!(I, k); push!(J, lin(1, j)); push!(V, -invdx2)
            diag += invdx2
        elseif bc_x === :dirichlet0
            diag += 2.0 * invdx2
        end # :neumann -> nothing
        if i > 1
            push!(I, k); push!(J, lin(i - 1, j)); push!(V, -invdx2)
            diag += invdx2
        elseif bc_x === :periodic
            push!(I, k); push!(J, lin(nx, j)); push!(V, -invdx2)
            diag += invdx2
        elseif bc_x === :dirichlet0
            diag += 2.0 * invdx2
        end

        # ---- y-direction ----
        if j < ny
            push!(I, k); push!(J, lin(i, j + 1)); push!(V, -invdy2)
            diag += invdy2
        elseif bc_y === :periodic
            push!(I, k); push!(J, lin(i, 1)); push!(V, -invdy2)
            diag += invdy2
        elseif bc_y === :dirichlet0
            diag += 2.0 * invdy2
        end
        if j > 1
            push!(I, k); push!(J, lin(i, j - 1)); push!(V, -invdy2)
            diag += invdy2
        elseif bc_y === :periodic
            push!(I, k); push!(J, lin(i, ny)); push!(V, -invdy2)
            diag += invdy2
        elseif bc_y === :dirichlet0
            diag += 2.0 * invdy2
        end

        push!(I, k); push!(J, k); push!(V, diag)
    end

    return sparse(I, J, V, n, n)
end

# ---------------------------------------------------------------------------
# Conservative central advection conv = +div(u_face * phi) for phi = u, v.
# Advecting flux: the stored divergence-free FACE velocities (uf, vf) from the
# previous projection. Advected face value: 2nd-order central average (upwind
# would clamp the spatial order to 1 — measured on Taylor-Green).
# Wall faces: no-slip => face flux 0, so the wall term vanishes identically.
# Face layout (cavity.jl convention): uf[i,j] = EAST face of cell (i,j),
# vf[i,j] = NORTH face; in periodic x, uf[nx,j] is the wrap face nx|1.
# ---------------------------------------------------------------------------
function _proj_advect!(conv_u, conv_v, u, v, uf, vf, dx, dy, nx, ny,
                       perx::Bool, pery::Bool)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    @inbounds for j in 1:ny, i in 1:nx
        ie = i == nx ? 1 : i + 1
        iw = i == 1 ? nx : i - 1
        jn = j == ny ? 1 : j + 1
        js = j == 1 ? ny : j - 1

        if i < nx || perx                      # east face
            Fe = uf[i, j]
            uE = 0.5 * (u[i, j] + u[ie, j]); vE = 0.5 * (v[i, j] + v[ie, j])
        else
            Fe = 0.0; uE = 0.0; vE = 0.0       # no-slip wall: zero flux
        end
        if i > 1 || perx                       # west face
            Fw = uf[iw, j]
            uW = 0.5 * (u[iw, j] + u[i, j]); vW = 0.5 * (v[iw, j] + v[i, j])
        else
            Fw = 0.0; uW = 0.0; vW = 0.0
        end
        if j < ny || pery                      # north face
            Fn = vf[i, j]
            uN = 0.5 * (u[i, j] + u[i, jn]); vN = 0.5 * (v[i, j] + v[i, jn])
        else
            Fn = 0.0; uN = 0.0; vN = 0.0
        end
        if j > 1 || pery                       # south face
            Fs = vf[i, js]
            uS = 0.5 * (u[i, js] + u[i, j]); vS = 0.5 * (v[i, js] + v[i, j])
        else
            Fs = 0.0; uS = 0.0; vS = 0.0
        end

        conv_u[i, j] = (Fe * uE - Fw * uW) * invdx + (Fn * uN - Fs * uS) * invdy
        conv_v[i, j] = (Fe * vE - Fw * vW) * invdx + (Fn * vN - Fs * vS) * invdy
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Rhie-Chow face velocities from cell fields:
#   uf = ubar - dcoef * (compact face grad p - averaged cell grad p),
# with dcoef = dt (the time-discrete velocity response to a pressure
# gradient). Wall faces stay 0 (no-slip). dcoef = 0 gives plain averaging
# (used once, for the initial face field).
# ---------------------------------------------------------------------------
function _proj_faces_from_cells!(uf, vf, u, v, p, gpx, gpy, dcoef,
                                 dx, dy, nx, ny, perx::Bool, pery::Bool)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    fill!(uf, 0.0)
    fill!(vf, 0.0)
    @inbounds for j in 1:ny, i in 1:nx
        if i < nx || perx
            ie = i == nx ? 1 : i + 1
            ubar = 0.5 * (u[i, j] + u[ie, j])
            gp_face = (p[ie, j] - p[i, j]) * invdx
            gp_cell = 0.5 * (gpx[i, j] + gpx[ie, j])
            uf[i, j] = ubar - dcoef * (gp_face - gp_cell)
        end
        if j < ny || pery
            jn = j == ny ? 1 : j + 1
            vbar = 0.5 * (v[i, j] + v[i, jn])
            gp_face = (p[i, jn] - p[i, j]) * invdy
            gp_cell = 0.5 * (gpy[i, j] + gpy[i, jn])
            vf[i, j] = vbar - dcoef * (gp_face - gp_cell)
        end
    end
    return nothing
end

# Cell-centred divergence of the face field. Wall faces contribute 0.
function _proj_face_divergence!(divf, uf, vf, dx, dy, nx, ny,
                                perx::Bool, pery::Bool)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    @inbounds for j in 1:ny, i in 1:nx
        ue = (i < nx || perx) ? uf[i, j] : 0.0
        uw = i > 1 ? uf[i - 1, j] : (perx ? uf[nx, j] : 0.0)
        vn = (j < ny || pery) ? vf[i, j] : 0.0
        vs = j > 1 ? vf[i, j - 1] : (pery ? vf[i, ny] : 0.0)
        divf[i, j] = (ue - uw) * invdx + (vn - vs) * invdy
    end
    return nothing
end

# Compact cell-centred gradient (face-average then difference), the discrete
# transpose of the face divergence. Periodic axes wrap; wall axes use the
# homogeneous-Neumann face value p_c (pressure walls). gx = +dp/dx, gy = +dp/dy.
function _proj_compact_gradient!(gx, gy, p, dx, dy, nx, ny,
                                 perx::Bool, pery::Bool)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    @inbounds for j in 1:ny, i in 1:nx
        pe = i < nx ? 0.5 * (p[i, j] + p[i + 1, j]) :
             (perx ? 0.5 * (p[i, j] + p[1, j]) : p[i, j])
        pw = i > 1 ? 0.5 * (p[i - 1, j] + p[i, j]) :
             (perx ? 0.5 * (p[nx, j] + p[i, j]) : p[i, j])
        gx[i, j] = (pe - pw) * invdx
        pn = j < ny ? 0.5 * (p[i, j] + p[i, j + 1]) :
             (pery ? 0.5 * (p[i, j] + p[i, 1]) : p[i, j])
        ps = j > 1 ? 0.5 * (p[i, j - 1] + p[i, j]) :
             (pery ? 0.5 * (p[i, ny] + p[i, j]) : p[i, j])
        gy[i, j] = (pn - ps) * invdy
    end
    return nothing
end

# Correct the FACE velocities with the compact face gradient of phi:
#   uf -= dtc * (phi_E - phi_C)/dx  (interior/periodic x-faces),
#   vf -= dtc * (phi_N - phi_C)/dy  (interior/periodic y-faces).
# The face divergence of grad_face(phi) is EXACTLY -Ap*phi (Ap = assembled
# -Laplacian with matching BCs), so with Ap*phi = -div(uf*)/dtc this update
# annihilates the face divergence in one shot (machine precision).
function _proj_correct_faces!(uf, vf, phi, dtc, dx, dy, nx, ny,
                              perx::Bool, pery::Bool)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    @inbounds for j in 1:ny, i in 1:nx
        if i < nx
            uf[i, j] -= dtc * (phi[i + 1, j] - phi[i, j]) * invdx
        elseif perx
            uf[i, j] -= dtc * (phi[1, j] - phi[i, j]) * invdx
        end
        if j < ny
            vf[i, j] -= dtc * (phi[i, j + 1] - phi[i, j]) * invdy
        elseif pery
            vf[i, j] -= dtc * (phi[i, 1] - phi[i, j]) * invdy
        end
    end
    return nothing
end

# Fill a field from `nothing` (zeros), a function (x,y) -> value sampled at
# cell centres, or an array (copied).
function _proj_init_field(init, xc, yc, nx, ny)
    f = zeros(Float64, nx, ny)
    if init isa Function
        @inbounds for j in 1:ny, i in 1:nx
            f[i, j] = Float64(init(xc[i], yc[j]))
        end
    elseif init isa AbstractMatrix
        size(init) == (nx, ny) ||
            throw(ArgumentError("initial field must be (nx, ny) = ($nx, $ny)"))
        f .= Float64.(init)
    elseif init !== nothing
        throw(ArgumentError("initial field must be nothing, a Function or a Matrix"))
    end
    return f
end

"""
    solve_incns_projection(; nx, ny, Lx, Ly, nu, dt, nsteps,
                           bc_x=:periodic, bc_y=:periodic,
                           fx=0.0, fy=0.0,
                           u0=nothing, v0=nothing, p0=nothing,
                           scheme=:cn, rhie_chow=:increment,
                           backend=CPU(), callback=nothing)

Unsteady 2D incompressible Navier-Stokes (laminar, Newtonian, rho = 1, constant
`nu`) on `[0,Lx] x [0,Ly]`, collocated cell-centred `nx x ny` grid, advanced
`nsteps` steps of size `dt` by an INCREMENTAL pressure-correction projection
(fractional step): AB2 explicit advection (Euler first step), implicit
Crank-Nicolson diffusion (`scheme=:cn`, default; `:be` = backward Euler
fallback), ONE pressure Poisson per step.

Per-axis BCs `bc_x`, `bc_y`: `:periodic` or `:wall` (no-slip `u = v = 0`;
homogeneous Neumann for pressure). `fx, fy` are constant body-force components
(e.g. `fx = G` drives a periodic channel). Initial conditions `u0, v0, p0`:
`nothing` (zero), a function `(x, y) -> value` sampled at cell centres, or an
`(nx, ny)` matrix.

`rhie_chow` selects the face-velocity (momentum) interpolation that feeds the
Poisson RHS: `:increment` (default — incremental interpolation, faces
`avg(u*)` plus the EXACT face projection of the pressure increment; 2nd order
in time) or `:full` (classical `d = dt` Rhie-Chow deviation against the full
pressure; strongest checkerboard damping but O(dt*h^2) time-accuracy defect —
see the header note). Both keep the face field divergence-free to machine
precision.

FACTORIZE-ONCE: the momentum Helmholtz operator `I/dt + nu*theta*(-Lap)` and
the pressure `(-Lap)` are constant, assembled and factorized ONCE before the
time loop ([`lin_factorize`](@ref), CHOLMOD, the singular pressure operator
pinned at dof 1) and reused for every step ([`lin_solve!`](@ref)):
`nfactorizations == 2` regardless of `nsteps`, `nlinsolves == 3*nsteps`.

`callback(step, t, u, v, p)` (if given) runs after each completed step with the
LIVE arrays — copy what you keep.

Returns a NamedTuple:
  `u, v, p`        cell fields after `nsteps` steps (p in the zero-mean gauge
                   of its increments, defined up to a constant)
  `uf, vf`         divergence-free face velocities (east/north faces)
  `t_final`        `nsteps*dt`
  `ke_history`     kinetic energy `0.5*sum(u^2+v^2)*dx*dy`, length `nsteps+1`
                   (entry 1 = initial condition)
  `max_div_inf`    max over steps of the post-projection face-divergence Linf
  `div_inf_final`  post-projection face-divergence Linf at the last step
  `nfactorizations, nlinsolves`  factorize-once receipts
  plus grid metadata (`dx, dy, xcenters, ycenters, nx, ny, nu, dt, nsteps,
  scheme`).

Validated against analytic transients: Taylor-Green vortex (spatial/temporal
order, energy decay) in `test/analytical/incns_unsteady_taylor_green.jl` and
impulsively-started plane Poiseuille in
`test/analytical/incns_unsteady_startup_channel.jl`. Registered in
`src/KrakeNN.jl` and exported; also reachable as
`solve(params, IncNS(:projection))` through the platform contract.
"""
function solve_incns_projection(; nx::Integer, ny::Integer,
                                Lx::Real, Ly::Real, nu::Real,
                                dt::Real, nsteps::Integer,
                                bc_x::Symbol = :periodic,
                                bc_y::Symbol = :periodic,
                                fx::Real = 0.0, fy::Real = 0.0,
                                u0 = nothing, v0 = nothing, p0 = nothing,
                                scheme::Symbol = :cn,
                                rhie_chow::Symbol = :increment,
                                backend = CPU(),
                                callback = nothing)
    nx = Int(nx); ny = Int(ny); nsteps = Int(nsteps)
    Lx = Float64(Lx); Ly = Float64(Ly)
    nu = Float64(nu); dt = Float64(dt)
    fx = Float64(fx); fy = Float64(fy)
    nsteps >= 1 || throw(ArgumentError("nsteps must be >= 1"))
    dt > 0 || throw(ArgumentError("dt must be positive"))
    bc_x in (:periodic, :wall) || throw(ArgumentError("bc_x must be :periodic or :wall"))
    bc_y in (:periodic, :wall) || throw(ArgumentError("bc_y must be :periodic or :wall"))
    scheme in (:cn, :be) || throw(ArgumentError("scheme must be :cn or :be"))
    rhie_chow in (:increment, :full) ||
        throw(ArgumentError("rhie_chow must be :increment or :full"))

    theta = scheme === :cn ? 0.5 : 1.0
    rc_dcoef = rhie_chow === :full ? Float64(dt) : 0.0
    perx = bc_x === :periodic
    pery = bc_y === :periodic

    dx = Lx / nx
    dy = Ly / ny
    xcenters = [(i - 0.5) * dx for i in 1:nx]
    ycenters = [(j - 0.5) * dy for j in 1:ny]
    n = nx * ny

    # ----- constant operators: assemble + factorize ONCE (before the loop) ----
    # Momentum Helmholtz: I/dt + nu*theta*(-Lap), velocity walls = :dirichlet0.
    Lmom = _proj_assemble_neg_laplacian(nx, ny, dx, dy;
                                        bc_x = perx ? :periodic : :dirichlet0,
                                        bc_y = pery ? :periodic : :dirichlet0)
    Amom = spdiagm(0 => fill(1.0 / dt, n)) + (nu * theta) .* Lmom
    mom_cache = lin_factorize(Amom; backend = CPUBackendTag(), spd = true)

    # Pressure: (-Lap), pressure walls = :neumann. Singular for every BC combo
    # here (constant nullspace) -> pin reference dof 1; the per-step RHS has
    # zero sum (telescoping face divergence), so pinning is consistent.
    Ap = _proj_assemble_neg_laplacian(nx, ny, dx, dy;
                                      bc_x = perx ? :periodic : :neumann,
                                      bc_y = pery ? :periodic : :neumann)
    p_cache = lin_factorize(Ap; backend = CPUBackendTag(), spd = true, pin_k0 = 1)

    nfactorizations = 2          # constant: the whole point of the seam
    nlinsolves = 0

    # ----- fields -----
    u = _proj_init_field(u0, xcenters, ycenters, nx, ny)
    v = _proj_init_field(v0, xcenters, ycenters, nx, ny)
    p = _proj_init_field(p0, xcenters, ycenters, nx, ny)
    uf = zeros(Float64, nx, ny)
    vf = zeros(Float64, nx, ny)
    gpx = zeros(Float64, nx, ny)
    gpy = zeros(Float64, nx, ny)
    conv_u = zeros(Float64, nx, ny)
    conv_v = zeros(Float64, nx, ny)
    convo_u = zeros(Float64, nx, ny)   # AB2 history (step n-1)
    convo_v = zeros(Float64, nx, ny)
    bu = zeros(Float64, nx, ny)
    bv = zeros(Float64, nx, ny)
    divf = zeros(Float64, nx, ny)
    lapu = zeros(Float64, n)           # (-Lap) u^n matvec work (CN explicit half)
    lapv = zeros(Float64, n)
    phi = zeros(Float64, nx, ny)
    gphix = zeros(Float64, nx, ny)
    gphiy = zeros(Float64, nx, ny)

    # Initial faces: plain averaging (dcoef = 0). Any O(h^2) divergence of the
    # averaged initial field is annihilated by the first projection.
    _proj_faces_from_cells!(uf, vf, u, v, p, gpx, gpy, 0.0,
                            dx, dy, nx, ny, perx, pery)

    cellvol = dx * dy
    ke_history = Float64[0.5 * (sum(abs2, u) + sum(abs2, v)) * cellvol]
    sizehint!(ke_history, nsteps + 1)
    max_div_inf = 0.0
    div_inf_final = 0.0
    nu_expl = nu * (1.0 - theta)       # 0 for :be
    invdt = 1.0 / dt

    for step in 1:nsteps
        # ---- 1. advection on the current field (AB2, Euler first step) ----
        _proj_advect!(conv_u, conv_v, u, v, uf, vf, dx, dy, nx, ny, perx, pery)

        # ---- 2. current pressure gradient (compact cell gradient) ----
        _proj_compact_gradient!(gpx, gpy, p, dx, dy, nx, ny, perx, pery)

        # ---- 3. momentum predictor RHS ----
        # bu = u/dt - conv_AB2 - dp/dx + fx - nu*(1-theta)*(-Lap u)  (CN half).
        if nu_expl != 0.0
            mul!(lapu, Lmom, vec(u))   # lapu = (-Lap) u^n (walls: ghost -u_c)
            mul!(lapv, Lmom, vec(v))
        else
            fill!(lapu, 0.0); fill!(lapv, 0.0)
        end
        lapu_m = reshape(lapu, nx, ny)
        lapv_m = reshape(lapv, nx, ny)
        if step == 1
            @. bu = u * invdt - conv_u - gpx + fx - nu_expl * lapu_m
            @. bv = v * invdt - conv_v - gpy + fy - nu_expl * lapv_m
        else
            @. bu = u * invdt - (1.5 * conv_u - 0.5 * convo_u) - gpx + fx -
                    nu_expl * lapu_m
            @. bv = v * invdt - (1.5 * conv_v - 0.5 * convo_v) - gpy + fy -
                    nu_expl * lapv_m
        end

        # ---- 4. momentum solves (factorize-once cache, 2 back-subs) ----
        ustar = reshape(lin_solve!(mom_cache, vec(bu)), nx, ny)
        vstar = reshape(lin_solve!(mom_cache, vec(bv)), nx, ny)
        nlinsolves += 2

        # ---- 5. face velocities of the predictor ----
        # :increment -> plain avg(u*) (rc_dcoef = 0): the compact pressure
        #   coupling is carried by the face projection of phi below.
        # :full -> classical Rhie-Chow d = dt deviation against p^n
        #   (rc_dcoef = dt). See the header trade-off note.
        _proj_faces_from_cells!(uf, vf, ustar, vstar, p, gpx, gpy, rc_dcoef,
                                dx, dy, nx, ny, perx, pery)

        # ---- 6. pressure Poisson: (-Lap) phi = -div(uf*)/dt ----
        # (minus in the RHS => phi IS the physical increment, see header note)
        _proj_face_divergence!(divf, uf, vf, dx, dy, nx, ny, perx, pery)
        bphi = vec(divf) .* (-invdt)
        phi .= reshape(lin_solve!(p_cache, bphi), nx, ny)
        nlinsolves += 1
        phi .-= sum(phi) / length(phi)         # zero-mean gauge

        # ---- 7. project faces (exact) and cells (compact gradient) ----
        _proj_correct_faces!(uf, vf, phi, dt, dx, dy, nx, ny, perx, pery)
        _proj_compact_gradient!(gphix, gphiy, phi, dx, dy, nx, ny, perx, pery)
        @. u = ustar - dt * gphix
        @. v = vstar - dt * gphiy

        # ---- 8. incremental pressure update ----
        @. p = p + phi

        # ---- 9. diagnostics + AB2 history ----
        _proj_face_divergence!(divf, uf, vf, dx, dy, nx, ny, perx, pery)
        div_inf_final = maximum(abs, divf)
        max_div_inf = max(max_div_inf, div_inf_final)
        push!(ke_history, 0.5 * (sum(abs2, u) + sum(abs2, v)) * cellvol)
        copyto!(convo_u, conv_u)
        copyto!(convo_v, conv_v)

        callback !== nothing && callback(step, step * dt, u, v, p)
    end

    return (; u, v, p, uf, vf, t_final = nsteps * dt,
            ke_history, max_div_inf, div_inf_final,
            nfactorizations, nlinsolves,
            dx, dy, xcenters, ycenters, nx, ny, nu, dt, nsteps, scheme,
            rhie_chow)
end
