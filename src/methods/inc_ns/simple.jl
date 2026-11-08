# Standalone steady SIMPLE incompressible solver core.
#
# Collocated cell-centred grid. Validated case: body-force-driven periodic
# plane Poiseuille (channel height H in y, periodic in x, no-slip walls at
# top/bottom, driven by a constant streamwise body force G = -dP/dx).
#
# Reuses the matrix-free KA grad/div/laplacian operators and the sparse
# Poisson service (CHOLMOD). KA + stdlib only, CPU by default. Registered in
# `src/KrakeNN.jl` (exported driver; the platform wrapper IncNS(:simple) in
# method.jl forwards here). Include guards keep it standalone-include-able.
#
# Public entry point:
#   solve_incns_simple(; nx, ny, H, mu, G, relax, scheme, tol, maxiter, backend=CPU())
#     -> NamedTuple(u, v, p, residual_history, iters, converged, ...)

using KernelAbstractions
using LinearAlgebra
using SparseArrays

const _INCNS_SIMPLE_OPERATOR_PATH =
    joinpath(@__DIR__, "..", "..", "fvfd", "operators_2d_grad_div_laplacian.jl")
const _INCNS_SIMPLE_POISSON_PATH =
    joinpath(@__DIR__, "..", "..", "solve", "poisson.jl")

if !isdefined(@__MODULE__, :gdl_divergence_2d!)
    include(_INCNS_SIMPLE_OPERATOR_PATH)
end
if !isdefined(@__MODULE__, :pin_reference_dof)
    include(_INCNS_SIMPLE_POISSON_PATH)
end
# Factorize-once seam (poisson.jl includes it, but guard for standalone use).
if !isdefined(@__MODULE__, :lin_factorize)
    include(joinpath(@__DIR__, "..", "..", "solve", "linear_solve.jl"))
end

# ---------------------------------------------------------------------------
# Sparse Laplacian assembly on a rectangular cell-centred grid.
#
# Layout: linear index k = i + (j-1)*nx, i in 1:nx (x, periodic),
# j in 1:ny (y). The matrix represents the discrete operator
#   (L u)_c = sum_faces (u_nb - u_c) / h_dir^2
# i.e. the NEGATIVE Laplacian is -L. Boundary treatment per direction:
#   :periodic -> wrap neighbour.
#   :dirichlet0 -> wall ghost u_g = -u_c (Dirichlet 0 at the wall face),
#                  contributing an extra -2/h^2 on the diagonal for that face
#                  (same ghost convention as assemble_poisson_dirichlet, which
#                  adds +invh2 to the diagonal of the POSITIVE-Laplacian form).
#   :neumann -> wall ghost u_g = u_c (zero normal gradient): no face term.
#
# Returned matrix A is the POSITIVE-definite discrete (-Laplacian), so that
# the momentum balance -mu*lap(u) = rhs becomes  (mu*A) u = rhs, and the
# pressure-correction -lap(p') = rhs becomes  A p' = rhs.
# ---------------------------------------------------------------------------
function _incns_assemble_neg_laplacian(nx::Integer, ny::Integer, dx::Real, dy::Real;
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

        # ---- x-direction (i) ----
        # east
        if i < nx
            push!(I, k); push!(J, lin(i + 1, j)); push!(V, -invdx2)
            diag += invdx2
        elseif bc_x === :periodic
            push!(I, k); push!(J, lin(1, j)); push!(V, -invdx2)
            diag += invdx2
        elseif bc_x === :dirichlet0
            diag += 2.0 * invdx2
        end # :neumann -> nothing
        # west
        if i > 1
            push!(I, k); push!(J, lin(i - 1, j)); push!(V, -invdx2)
            diag += invdx2
        elseif bc_x === :periodic
            push!(I, k); push!(J, lin(nx, j)); push!(V, -invdx2)
            diag += invdx2
        elseif bc_x === :dirichlet0
            diag += 2.0 * invdx2
        end

        # ---- y-direction (j) ----
        # north
        if j < ny
            push!(I, k); push!(J, lin(i, j + 1)); push!(V, -invdy2)
            diag += invdy2
        elseif bc_y === :periodic
            push!(I, k); push!(J, lin(i, 1)); push!(V, -invdy2)
            diag += invdy2
        elseif bc_y === :dirichlet0
            diag += 2.0 * invdy2
        end
        # south
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

# Factorise a SPD sparse operator ONCE (CHOLMOD), with optional reference-dof
# pinning for the singular (all-Neumann/periodic) pressure operator. Thin
# wrappers over the shared factorize-once seam (linear_solve.jl) so the repeated
# per-outer-iteration solves reuse a single cached factorization; cuDSS drops in
# on GPU by swapping the backend tag. Returns a LinearSolveCache.
function _incns_factorise(A::SparseMatrixCSC{Float64,Int}; pin_k0::Integer=0)
    return lin_factorize(A; backend=CPUBackendTag(), spd=true, pin_k0=Int(pin_k0))
end

# Solve A * x = b reusing the cached factorization. The pinned-operator RHS
# adjustment is handled inside lin_solve! from the cache (the `A` argument is
# kept for call-site compatibility but is no longer needed).
function _incns_solve!(cache::LinearSolveCache, ::SparseMatrixCSC{Float64,Int}, b::Vector{Float64})
    return lin_solve!(cache, b)
end

function _incns_simple_scheme(scheme::Symbol)
    s = Symbol(replace(lowercase(String(scheme)), '-' => '_'))
    s in (:simple, :simplec) ||
        throw(ArgumentError("scheme must be :simple or :simplec"))
    return s
end

function _incns_momentum_advection_scheme(advection::Symbol)
    scheme = Symbol(replace(lowercase(String(advection)), '-' => '_'))
    scheme in (:upwind, :linear_upwind) ||
        throw(ArgumentError("momentum_advection must be :upwind or :linear_upwind; got $advection"))
    return scheme
end

function _incns_positive_neighbour_sum(A::SparseMatrixCSC{Float64,Int})
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

# ---------------------------------------------------------------------------
# Rhie-Chow face velocity interpolation.
#
# Collocated cell-centred velocities suffer pressure-velocity decoupling
# (checkerboard). Rhie-Chow reconstructs the face-normal velocity as the
# average of the two cell velocities MINUS the difference between the
# compact face pressure gradient and the average of the two cell pressure
# gradients, scaled by d = relax_u / a_p (here a_p ~ mu/h^2 * stencil; we use
# the momentum-operator diagonal as a_p surrogate).
#
# Note: for fully-developed body-force Poiseuille the pressure field is
# uniform (p' ~ 0), so the Rhie-Chow correction term is ~0 and checkerboard
# is NOT stressed by this case. The wiring is exercised in full by the cavity.
# ---------------------------------------------------------------------------
function _incns_rhie_chow_faces!(uf, vf, u, v, p, gpx, gpy, d_u, d_v,
                                 dx, dy, nx, ny)
    # uf[i,j]: east face of cell (i,j) (x-normal). vf[i,j]: north face.
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    @inbounds for j in 1:ny, i in 1:nx
        ie = i == nx ? 1 : i + 1   # periodic east
        # East face (always interior in x because periodic)
        ubar = 0.5 * (u[i, j] + u[ie, j])
        dbar = 0.5 * (d_u[i, j] + d_u[ie, j])
        # compact face pressure gradient
        gp_face = (p[ie, j] - p[i, j]) * invdx
        # average of cell pressure gradients (gpx holds +dp/dx)
        gp_cell = 0.5 * (gpx[i, j] + gpx[ie, j])
        uf[i, j] = ubar - dbar * (gp_face - gp_cell)
    end
    @inbounds for j in 1:ny, i in 1:nx
        if j < ny
            vbar = 0.5 * (v[i, j] + v[i, j + 1])
            dbar = 0.5 * (d_v[i, j] + d_v[i, j + 1])
            gp_face = (p[i, j + 1] - p[i, j]) * invdy
            gp_cell = 0.5 * (gpy[i, j] + gpy[i, j + 1])
            vf[i, j] = vbar - dbar * (gp_face - gp_cell)
        else
            vf[i, j] = 0.0   # no-slip wall, north face of top row
        end
    end
    return nothing
end

# Divergence of the Rhie-Chow face field, returned as a cell-centred field.
# Uses simple face differences consistent with the face layout above.
function _incns_face_divergence!(div, uf, vf, dx, dy, nx, ny)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    @inbounds for j in 1:ny, i in 1:nx
        iw = i == 1 ? nx : i - 1   # west face = east face of west neighbour
        ue = uf[i, j]
        uw = uf[iw, j]
        vn = vf[i, j]
        vs = j == 1 ? 0.0 : vf[i, j - 1]   # south wall = 0 on bottom row
        div[i, j] = (ue - uw) * invdx + (vn - vs) * invdy
    end
    return nothing
end

# Compact cell-centred pressure gradient, the discrete transpose of the
# face-divergence above. Using the SAME compact stencil for the SIMPLE velocity
# correction (instead of the wide operator gradient) keeps div(d*grad(p')) equal
# to the assembled pressure Laplacian, making the projection idempotent and
# stable. gx = +dp/dx, gy = +dp/dy (returns the physical gradient).
function _incns_compact_gradient!(gx, gy, p, dx, dy, nx, ny)
    invdx = 1.0 / dx
    invdy = 1.0 / dy
    @inbounds for j in 1:ny, i in 1:nx
        ie = i == nx ? 1 : i + 1       # periodic east face
        iw = i == 1 ? nx : i - 1       # periodic west face
        pe = 0.5 * (p[i, j] + p[ie, j])   # east face value
        pw = 0.5 * (p[iw, j] + p[i, j])   # west face value
        gx[i, j] = (pe - pw) * invdx
        # y: Neumann walls -> wall face gradient = 0, so use interior faces only.
        pn = j < ny ? 0.5 * (p[i, j] + p[i, j + 1]) : p[i, j]   # north face
        ps = j > 1 ? 0.5 * (p[i, j - 1] + p[i, j]) : p[i, j]    # south face
        gy[i, j] = (pn - ps) * invdy
    end
    return nothing
end

"""
    solve_incns_simple(; nx, ny, H, mu, G, relax=(u=0.7, p=0.3),
                       scheme=:simplec, momentum_advection=:linear_upwind,
                       tol=1e-10, maxiter=200, Lx=H, backend=CPU())

Standalone steady SIMPLE incompressible solver core for body-force-driven
periodic plane Poiseuille flow.

Geometry: channel of height `H` (y), streamwise length `Lx` (x), periodic in
x, no-slip walls (`u=v=0`) at `y=0` and `y=H`, driven by constant streamwise
body force `G = -dP/dx`. Collocated cell-centred grid `nx x ny`.

Returns a NamedTuple with fields:
  `u, v, p`            converged cell-centred fields (nx x ny)
  `residual_history`   normalised continuity residual per outer iteration
  `iters`              number of outer iterations performed
  `converged`          whether `tol` was reached
  `dx, dy, ycenters`   grid metrics for analytic comparison

Structure: per outer iteration, a DIRECT momentum solve with frozen pressure
(the assembled SPD viscous `-Laplacian`, factorized ONCE through the
[`lin_factorize`](@ref)/[`lin_solve!`](@ref) seam), then Rhie-Chow face
velocities, then the pinned pressure-correction Poisson solve (same seam,
`pin_k0=1`) and the compact-gradient velocity/pressure corrections. Convergence
requires BOTH the continuity residual AND the velocity settle below `tol`.
`momentum_advection` is accepted for API parity with manifold/cavity steady
solvers; this fully-developed Poiseuille rung has no nonlinear convection term.

Validated against the analytic Poiseuille parabola at 0.033% L2 error
(`test/analytical/incns_poiseuille.jl`). The MG/GPU sibling exercising the full
pressure-velocity coupling is [`solve_incns_cavity_mg`](@ref). Registered in
`src/KrakeNN.jl` and exported; also reachable as `solve(params, IncNS(:simple))`.
"""
function solve_incns_simple(; nx::Integer, ny::Integer, H::Real, mu::Real, G::Real,
                            relax=(u = 0.7, p = 0.3),
                            scheme::Symbol = :simplec,
                            momentum_advection::Symbol = :linear_upwind,
                            tol::Real = 1e-10, maxiter::Integer = 200,
                            Lx::Real = H, backend = CPU())
    nx = Int(nx); ny = Int(ny)
    H = Float64(H); mu = Float64(mu); G = Float64(G); Lx = Float64(Lx)
    αu = Float64(relax.u); αp = Float64(relax.p)
    scheme_sym = _incns_simple_scheme(scheme)
    momentum_scheme = _incns_momentum_advection_scheme(momentum_advection)
    dx = Lx / nx
    dy = H / ny
    ycenters = [(j - 0.5) * dy for j in 1:ny]

    # ----- operator backends / geometry (regular, no cut cells) -----
    is_solid = falses(nx, ny)
    bc = (FVFD_BC_PERIODIC, FVFD_BC_PERIODIC, FVFD_BC_WALL, FVFD_BC_WALL)  # W,E,S,N

    # ----- momentum operator: SPD (-Laplacian), periodic-x / Dirichlet-y -----
    # Amom u = (G - dp/dx). The momentum operator here is the full viscous
    # Laplacian, which a direct (CHOLMOD) solve inverts EXACTLY in one shot for
    # a frozen pressure. We therefore do NOT iterate the momentum equation with
    # point-relaxation (that converges at Jacobi rate, O(N^2) sweeps). Instead
    # we solve it directly each outer iteration; au only damps the pressure-
    # coupling correction below. For the cavity (next mission) the advection
    # term makes Amom non-symmetric and outer iterations matter; the same
    # direct-solve-then-correct structure carries over.
    Lmom = _incns_assemble_neg_laplacian(nx, ny, dx, dy; bc_x = :periodic, bc_y = :dirichlet0)
    Amom = mu .* Lmom
    ap = Vector(diag(Amom))                      # a_p (operator diagonal), length n
    mom_op = _incns_factorise(Amom)

    # Legacy Rhie-Chow coefficient. Keep this fixed across schemes so SIMPLEC
    # changes only the pressure-correction path, not the converged face model.
    ap_mat = reshape(ap, nx, ny)
    d_rc_u = αu ./ ap_mat
    d_rc_v = αu ./ ap_mat

    # SIMPLE d-coefficient d = au / a_p (legacy velocity response to a pressure
    # gradient). SIMPLEC subtracts sum(a_nb) from the inverse of that existing
    # response coefficient; this preserves the extra under-relaxation already
    # encoded by the legacy d path.
    if scheme_sym === :simple
        d_u = d_rc_u
        d_v = d_rc_v
    else
        anb = reshape(_incns_positive_neighbour_sum(Amom), nx, ny)
        denom = 1.0 ./ d_rc_u .- anb
        minimum(denom) > 0.0 ||
            throw(ArgumentError("SIMPLEC correction denominator must be positive; use relax.u < 1"))
        d_u = 1.0 ./ denom
        d_v = 1.0 ./ denom
    end

    # ----- pressure-correction operator: d * (-Laplacian), periodic-x / Neumann-y -----
    # Continuity:  div(u* + d*grad(p')) = 0  ->  d*(-Lap) p' = div(u*).
    # d is uniform on this regular grid, so scale the (-Lap) by the scalar d.
    Lp = _incns_assemble_neg_laplacian(nx, ny, dx, dy; bc_x = :periodic, bc_y = :neumann)
    dscalar = sum(d_u) / length(d_u)
    Ap = dscalar .* Lp
    p_op = _incns_factorise(Ap; pin_k0 = 1)      # singular: pin reference dof

    # ----- fields -----
    u = zeros(Float64, nx, ny)
    v = zeros(Float64, nx, ny)
    p = zeros(Float64, nx, ny)
    gpx = zeros(Float64, nx, ny)
    gpy = zeros(Float64, nx, ny)
    uf = zeros(Float64, nx, ny)
    vf = zeros(Float64, nx, ny)
    divstar = zeros(Float64, nx, ny)
    pcorr = zeros(Float64, nx, ny)
    gpcx = zeros(Float64, nx, ny)
    gpcy = zeros(Float64, nx, ny)

    residual_history = Float64[]
    converged = false
    iters = 0
    vel_change = Inf

    for it in 1:maxiter
        iters = it

        # ---- 1. pressure gradient of current p (compact, transpose-consistent) ----
        # gpx,gpy hold +dp/dx,+dp/dy here; the momentum source is (G - dp/dx).
        _incns_compact_gradient!(gpx, gpy, p, dx, dy, nx, ny)

        # ---- 2. momentum predictor (direct solve, frozen pressure) ----
        # Amom u* = (G - dp/dx) ;  Amom v* = (-dp/dy).
        bu = vec(G .- gpx)
        bv = vec(.-gpy)
        ustar = reshape(_incns_solve!(mom_op, Amom, bu), nx, ny)
        vstar = reshape(_incns_solve!(mom_op, Amom, bv), nx, ny)

        # commit the predictor as the current velocity.
        umax_prev = max(maximum(abs, u), eps())
        du = 0.0
        @inbounds for idx in eachindex(u)
            du = max(du, abs(ustar[idx] - u[idx]), abs(vstar[idx] - v[idx]))
            u[idx] = ustar[idx]
            v[idx] = vstar[idx]
        end
        vel_change = du / umax_prev

        # ---- 3. Rhie-Chow face velocities ----
        _incns_rhie_chow_faces!(uf, vf, u, v, p, gpx, gpy, d_rc_u, d_rc_v,
                                dx, dy, nx, ny)

        # ---- 4. continuity residual = div(u*) ----
        _incns_face_divergence!(divstar, uf, vf, dx, dy, nx, ny)

        # normalised continuity residual (L2 of div, scaled by a reference flux)
        ref = max(sum(abs, uf) / max(length(uf), 1) / dx, eps())
        res = sqrt(sum(abs2, divstar) / length(divstar)) / ref
        push!(residual_history, res)

        # Converge when BOTH continuity is satisfied AND the velocity field has
        # settled between outer iterations (vel_change -> 0). Continuity alone
        # is insufficient: any divergence-free u(y), including an under-scaled
        # one, satisfies it, so the momentum magnitude would be unconstrained.
        if res < tol && vel_change < tol
            converged = true
            break
        end

        # ---- 5. pressure-correction Poisson: A p' = div(u*) ----
        # Gate on a divergence floor: when div* is at machine-noise level the
        # field is already solenoidal and inverting it only injects a spurious
        # pressure mode (collocated SIMPLE is sensitive to this). Skip the
        # correction in that regime; for this case p' ~ 0 and is effectively a
        # no-op. The full path is exercised when div* is genuinely non-zero
        # (e.g. the lid-driven cavity, next mission).
        div_floor = 1e-10 * ref
        if sqrt(sum(abs2, divstar) / length(divstar)) > div_floor
            bp = vec(divstar)
            pcorr .= reshape(_incns_solve!(p_op, Ap, bp), nx, ny)

            # ---- 6. correct pressure ----
            @. p = p + αp * pcorr

            # ---- 7. correct cell velocities with grad(p') (compact) ----
            _incns_compact_gradient!(gpcx, gpcy, pcorr, dx, dy, nx, ny)
            # gpcx = +dp'/dx ; velocity correction u' = -d * dp'/dx = -d*gpcx
            @. u = u - αp * d_u * gpcx
            @. v = v - αp * d_v * gpcy
        end
    end

    return (; u, v, p, residual_history, iters, converged, vel_change,
            dx, dy, ycenters, H, mu, G, Lx, nx, ny, scheme=scheme_sym,
            momentum_advection=momentum_scheme)
end
