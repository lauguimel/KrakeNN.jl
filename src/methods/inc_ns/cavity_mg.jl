# Backend-parametric steady SIMPLE incompressible solver for the 2D lid-driven
# cavity, built so the SAME source runs on CPU now and CUDA later.
#
# WHAT IS DIFFERENT FROM cavity.jl
# --------------------------------
# cavity.jl allocates plain CPU `Array`s, assembles sparse `-Laplacian`s and
# factorizes them with CHOLMOD (via the linear-solve seam). That is fast on CPU
# but (a) under-uses the GPU and (b) is not the matrix-free MG path. This file:
#
#   * allocates EVERY field through a backend array type `atype` (Array on CPU,
#     CuArray later) — no plain `zeros`,
#   * does EVERY elementwise op as a KernelAbstractions `@kernel` launched on the
#     KA backend `backend_ka` — no host scalar-indexing loops over device arrays,
#   * solves BOTH the pressure-correction Poisson AND the viscous/momentum
#     predictor with the matrix-free multigrid `solve_poisson_mg`,
#   * reductions (residual norms, gauge means) go through `sum`/`norm`, which
#     dispatch to the device implementation on GPU.
#
# So a CUDA run is `backend_ka = CUDABackend()`, `atype = CuArray{Float64}` with
# NO other change. We validate on the CPU backend here (no GPU locally).
#
# LATENCY/ORCHESTRATION DESIGN (GPU)
# ----------------------------------
# The end-to-end GPU solve is LATENCY-bound, not FLOP-bound: tens of thousands
# of SIMPLE outer iterations, each made of many small kernels. Three measures
# keep the GPU fed:
#
#   1. The per-iteration physics is FUSED into four large kernels (gradient +
#      Rhie-Chow; convection + momentum RHS (+L² scaling); face divergence +
#      pressure RHS; all velocity/pressure corrections). Fusion only joins ops
#      whose per-cell results depend on data that is already globally consistent
#      — no fusion across a needed global barrier. Recomputing a neighbour's
#      compact pressure gradient inline (same FP ops, bit-identical) replaces a
#      stored-array dependency, so gradient+Rhie-Chow need no barrier between
#      them.
#   2. NO per-launch host synchronize: KA CPU kernels are synchronous and CUDA
#      launches are stream-ordered, so host syncs only happen where a host
#      scalar is read. Outer convergence norms are computed every `norm_stride`
#      iterations only (they only gate the STOP decision; the iterates are
#      unchanged — the solver merely stops up to norm_stride-1 iterations
#      later, i.e. slightly MORE converged).
#   3. `mg_cycles > 0` runs every inner MG solve (momentum + pressure) with a
#      FIXED number of V-cycles and ZERO residual checks / zero-mean
#      projections — no inner reductions or syncs at all, and a STATIC kernel
#      launch sequence (CUDA-graph-capturable later). `mg_cycles = 0` keeps
#      the legacy tolerance-driven inner solves.
#
# With `mg_cycles > 0` and off-stride iterations, one SIMPLE outer iteration
# performs ZERO host<->device synchronizations.
#
# MOMENTUM OPERATOR (Laplace vs Helmholtz)
# ----------------------------------------
# The momentum predictor is solved by MG, which inverts `(-∇² + σ)` (the optional
# Helmholtz shift added to poisson_mg.jl). We keep advection in the RHS (deferred
# correction, first-order upwind on Rhie-Chow face fluxes) and fold the SIMPLE
# under-relaxation into a pseudo-transient diagonal shift `τ = ρ V_cell / (αu Δt*)`
# implemented as the Helmholtz σ. Concretely, dividing the viscous momentum eqn by
# the kinematic viscosity ν=μ (ρ=1), the per-cell predictor is
#
#     (-∇² + σ) u* = ( -conv_u - ∂p/∂x + src_lid )/ν + σ u_old
#
# with σ = (1/αu - 1) * a_diag / ν, a_diag a representative central diagonal of
# the viscous operator. The +σ u_old term makes the fixed point converge to the
# under-relaxed steady solution while keeping the operator strongly diagonally
# dominant (textbook MG convergence) — exactly the SIMPLE momentum under-relax,
# but expressed so a PLAIN-shift MG can solve it.
#
# DIRICHLET CONVENTION (must match poisson_mg.jl)
# -----------------------------------------------
# The MG Dirichlet operator uses the GHOST-0 ("+1/h² per missing face") boundary,
# i.e. the wall value lives at the ghost CELL CENTRE one half-cell outside the
# boundary. A non-homogeneous wall value u_w is folded into the RHS as +u_w/h²
# at the boundary row (the missing face contributes (u_w - u_c)/h²). The lid
# (u = U_lid on north) is injected this way. All velocity gradient/divergence
# wall treatments below use the SAME ghost-0 convention so the discrete operators
# are mutually consistent.
#
# Public entry point:
#   solve_incns_cavity_mg(; nx, ny, U_lid, Re, relax, tol, maxiter,
#                         norm_stride=25, mg_cycles=3,
#                         backend_ka=CPU(), atype=Array{Float64}, ...)
#     -> NamedTuple(u, v, p, residual_history, iters, converged, ...)
#
# KA + stdlib only. Registered in `src/KrakeNN.jl` (exported driver; the
# platform wrapper IncNS(:cavity_mg) in method.jl forwards here).

using KernelAbstractions
using LinearAlgebra: norm

# Matrix-free multigrid (pressure + momentum) and its backend tags.
if !isdefined(@__MODULE__, :solve_poisson_mg)
    include(joinpath(@__DIR__, "..", "..", "solve", "poisson_mg.jl"))
end

# ---------------------------------------------------------------------------
# KA kernels. Square grid (nx == ny == N), cell-centred on [0,L]^2, h = L/N.
# Field layout u[i,j], i in 1:N (x), j in 1:N (y). North = j=N (lid), south = j=1,
# west = i=1, east = i=N. Faces: uf[i,j] = east face of (i,j) (i in 1:N-1 active),
# vf[i,j] = north face of (i,j) (j in 1:N-1 active); wall faces carry 0.
#
# The per-iteration physics is fused into FOUR kernels (see header). The compact
# cell-centred pressure gradient (homogeneous Neumann at the walls, consistent
# with the all-Neumann pressure Laplacian) is an @inline helper recomputed where
# needed — same FP expressions as a stored-gradient pass, so values (and the
# converged answer) are bit-identical to the unfused version.
# ---------------------------------------------------------------------------

@inline function _cav_gradx(p, i, j, invdx, N)
    @inbounds begin
        T = eltype(p)
        pe = (i < N) ? (p[i, j] + p[i + 1, j]) * T(0.5) : p[i, j]
        pw = (i > 1) ? (p[i - 1, j] + p[i, j]) * T(0.5) : p[i, j]
        return (pe - pw) * invdx
    end
end

@inline function _cav_grady(p, i, j, invdy, N)
    @inbounds begin
        T = eltype(p)
        pn = (j < N) ? (p[i, j] + p[i, j + 1]) * T(0.5) : p[i, j]
        ps = (j > 1) ? (p[i, j - 1] + p[i, j]) * T(0.5) : p[i, j]
        return (pn - ps) * invdy
    end
end

# FUSED kernel 1: cell-centred pressure gradient (gx=+dp/dx, gy=+dp/dy) AND
# Rhie-Chow face velocities. uf east faces (i<N), vf north faces (j<N); wall
# faces 0 (no-slip; lid face vf at north is 0 since v_lid=0). The neighbour's
# cell gradient is recomputed inline from p (bit-identical to a stored pass),
# which removes the global barrier a stored-gradient dependency would need.
@kernel function cav_grad_rhie_chow_kernel!(
    gx, gy, uf, vf, @Const(u), @Const(v), @Const(p),
    @Const(d_u), @Const(d_v), invdx, invdy, N,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            T = eltype(uf)
            half = T(0.5)
            gxc = _cav_gradx(p, i, j, invdx, N)
            gyc = _cav_grady(p, i, j, invdy, N)
            gx[i, j] = gxc
            gy[i, j] = gyc
            # east face of (i,j)
            if i < N
                ubar = half * (u[i, j] + u[i + 1, j])
                dbar = half * (d_u[i, j] + d_u[i + 1, j])
                gp_face = (p[i + 1, j] - p[i, j]) * invdx
                gp_cell = half * (gxc + _cav_gradx(p, i + 1, j, invdx, N))
                uf[i, j] = ubar - dbar * (gp_face - gp_cell)
            else
                uf[i, j] = zero(T)
            end
            # north face of (i,j)
            if j < N
                vbar = half * (v[i, j] + v[i, j + 1])
                dbar = half * (d_v[i, j] + d_v[i, j + 1])
                gp_face = (p[i, j + 1] - p[i, j]) * invdy
                gp_cell = half * (gyc + _cav_grady(p, i, j + 1, invdy, N))
                vf[i, j] = vbar - dbar * (gp_face - gp_cell)
            else
                vf[i, j] = zero(T)
            end
        end
    end
end

# FUSED kernel 2: deferred-correction convection (conv = +div(u_face * phi),
# first-order upwind on the Rhie-Chow face fluxes; wall fluxes vanish, the lid
# carries U_lid for the advected u value at the north wall but vf=0 there so it
# contributes no flux — matching cavity.jl exactly) AND the momentum RHS for the
# MG solve of (-∇² + σ) phi* = rhs, PRE-SCALED by L² (the unit-square MG
# rescaling; see the solver body).
#
# The PHYSICAL viscous momentum equation is  μ(-∇²)φ + conv = -∂p/∂x  with
# Dirichlet walls. Dividing by ν=μ (ρ=1) gives the MG-form operator (-∇²)φ, so
# the convective and pressure-gradient terms divide by ν. The Dirichlet wall
# SOURCE pairs with the MG operator's own boundary diagonal (already the post-÷ν
# viscous coefficient) and is therefore NOT divided by ν.
#
# WALL TREATMENT (verified against the assembled +2/h² operator).
# On a cell-centred grid the wall sits at the FACE (half a cell from the centre),
# so the Dirichlet ghost is u_g = 2 u_w - u_c. The resulting OPERATOR is bit-for-
# bit identical to the MG ghost-0 Dirichlet operator: BOTH have boundary diagonal
# (4 + n_wall)/h². The "+1/h² vs +2/h²" distinction lives in the SOURCE only, not
# the stencil. Only the moving lid (north, u-momentum) carries a source,
# +2 U_lid/h². The σ φ_old pseudo-transient term cancels at the fixed point
# (φ=φ_old), so the converged answer is σ-independent. `phi_old` is the current
# (pre-predictor) velocity, passed as u, v.
@kernel function cav_conv_mom_rhs_kernel!(
    rhs_u, rhs_v, @Const(u), @Const(v), @Const(uf), @Const(vf),
    @Const(gpx), @Const(gpy), invdx, invdy, invnu, sigma, invh2, U_lid, L2, N,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            T = eltype(rhs_u)
            Fe = (i < N) ? uf[i, j] : zero(T)
            Fw = (i > 1) ? uf[i - 1, j] : zero(T)
            Fn = (j < N) ? vf[i, j] : zero(T)
            Fs = (j > 1) ? vf[i, j - 1] : zero(T)

            uE = (i < N) ? (Fe >= 0 ? u[i, j] : u[i + 1, j]) : zero(T)
            uW = (i > 1) ? (Fw >= 0 ? u[i - 1, j] : u[i, j]) : zero(T)
            uN = (j < N) ? (Fn >= 0 ? u[i, j] : u[i, j + 1]) : T(U_lid)
            uS = (j > 1) ? (Fs >= 0 ? u[i, j - 1] : u[i, j]) : zero(T)

            vE = (i < N) ? (Fe >= 0 ? v[i, j] : v[i + 1, j]) : zero(T)
            vW = (i > 1) ? (Fw >= 0 ? v[i - 1, j] : v[i, j]) : zero(T)
            vN = (j < N) ? (Fn >= 0 ? v[i, j] : v[i, j + 1]) : zero(T)
            vS = (j > 1) ? (Fs >= 0 ? v[i, j - 1] : v[i, j]) : zero(T)

            conv_u = (Fe * uE - Fw * uW) * invdx + (Fn * uN - Fs * uS) * invdy
            conv_v = (Fe * vE - Fw * vW) * invdx + (Fn * vN - Fs * vS) * invdy

            # Lid source on the north row for the u-component only: +2 U_lid/h²
            # (the half-spacing ghost 2 u_w - u_c). All other walls homogeneous.
            src_u = zero(T)
            if j == N
                src_u += T(2) * T(U_lid) * invh2
            end
            src_v = zero(T)
            rhs_u[i, j] = ((-conv_u - gpx[i, j]) * invnu + src_u + sigma * u[i, j]) * L2
            rhs_v[i, j] = ((-conv_v - gpy[i, j]) * invnu + src_v + sigma * v[i, j]) * L2
        end
    end
end

# FUSED kernel 3: face divergence of the Rhie-Chow field (cell-centred; wall
# faces are 0) AND the scaled pressure-correction RHS rhs_p = div(u*) · L²/dval
# (MG solves the PLAIN unit-square Neumann Laplacian; the SIMPLE d-coefficient
# and domain scaling are folded into the RHS). `divu` is kept unscaled for the
# outer continuity-residual norm.
@kernel function cav_div_prhs_kernel!(
    divu, rhs_p, @Const(uf), @Const(vf), invdx, invdy, pscale, N,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            T = eltype(divu)
            ue = (i < N) ? uf[i, j] : zero(T)
            uw = (i > 1) ? uf[i - 1, j] : zero(T)
            vn = (j < N) ? vf[i, j] : zero(T)
            vs = (j > 1) ? vf[i, j - 1] : zero(T)
            d = (ue - uw) * invdx + (vn - vs) * invdy
            divu[i, j] = d
            rhs_p[i, j] = d * pscale
        end
    end
end

# FUSED kernel 4: ALL post-pressure-solve corrections in one pass —
#   * FACE velocities += d^f · grad_face(pcorr)  (exact projection: the MG
#     pressure operator solves (-∇²)pcorr = div(u*)/dscalar, so this annihilates
#     the face divergence in one shot),
#   * CELL velocities += d · grad_cell(pcorr)  (compact gradient inline),
#   * under-relaxed pressure update p -= αp · pcorr.
# Every output at (i,j) depends only on the (already globally consistent) pcorr
# stencil at (i,j), so no barrier is needed between the fused pieces.
@kernel function cav_apply_corrections_kernel!(
    uf, vf, u, v, p, @Const(pcorr), @Const(d_u), @Const(d_v),
    invdx, invdy, alpha_p, N,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            T = eltype(uf)
            half = T(0.5)
            if i < N
                dbar = half * (d_u[i, j] + d_u[i + 1, j])
                uf[i, j] += dbar * (pcorr[i + 1, j] - pcorr[i, j]) * invdx
            end
            if j < N
                dbar = half * (d_v[i, j] + d_v[i, j + 1])
                vf[i, j] += dbar * (pcorr[i, j + 1] - pcorr[i, j]) * invdy
            end
            u[i, j] += d_u[i, j] * _cav_gradx(pcorr, i, j, invdx, N)
            v[i, j] += d_v[i, j] * _cav_grady(pcorr, i, j, invdy, N)
            p[i, j] -= alpha_p * pcorr[i, j]
        end
    end
end

# Velocity-settle metric (max abs change), written without host scalar indexing:
# delta = phi_new - phi_old computed elementwise; the host calls maximum(abs, .).
@kernel function cav_delta_kernel!(du, @Const(unew), @Const(uold), @Const(vnew), @Const(vold), N)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            a = abs(unew[i, j] - uold[i, j])
            b = abs(vnew[i, j] - vold[i, j])
            du[i, j] = a > b ? a : b
        end
    end
end

# Subtract a HOST scalar (gauge / zero-mean) from a field, elementwise. Legacy
# (tolerance-driven) gauge path: the mean is reduced to the host first.
@kernel function cav_shift_kernel!(a, s, N)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            a[i, j] -= s
        end
    end
end

# Subtract a DEVICE-resident mean: mbuf is a 1x1 device array holding sum(a)
# (filled by `sum!`, which stays on-device), invNN = 1/(N*N). No host transfer,
# so the zero-mean gauge costs NO host sync in the fixed-cycles fast path.
@kernel function cav_shift_mean_kernel!(a, @Const(mbuf), invNN, N)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            a[i, j] -= mbuf[1, 1] * invNN
        end
    end
end

# STATIC two-stage gauge reduction (opt-in via `static_gauge=true`): column sums
# into a 1xN buffer, then a single-work-item final sum into the 1x1 gauge buffer.
# Functionally equivalent to `sum!(gbuf, a)` (last-ulp summation-order
# differences only) but ALLOCATION-FREE and a fixed two-launch sequence — the
# library `sum!` may allocate a temporary partial-reduction buffer on GPU, which
# CUDA stream capture rejects ("graph capture does not support asynchronous
# memory operations"). Required by the CUDA-graph executor; default OFF keeps
# the bit-identical `sum!` path.
@kernel function cav_colsum_kernel!(csum, @Const(a), N)
    j = @index(Global, Linear)
    @inbounds begin
        if j <= N
            T = eltype(csum)
            s = zero(T)
            for i in 1:N
                s += a[i, j]
            end
            csum[1, j] = s
        end
    end
end

@kernel function cav_gauge_finalize_kernel!(gbuf, @Const(csum), N)
    k = @index(Global, Linear)
    @inbounds begin
        if k == 1
            T = eltype(gbuf)
            s = zero(T)
            for j in 1:N
                s += csum[1, j]
            end
            gbuf[1, 1] = s
        end
    end
end

# Checkerboard metric (high-frequency pressure energy / variance). Uses a kernel
# to form the 5-point high-pass and the squared deviation, then device reductions.
@kernel function cav_checker_kernel!(osc, dev2, @Const(p), pbar, N)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            T = eltype(p)
            if 2 <= i <= N - 1 && 2 <= j <= N - 1
                lap = p[i + 1, j] + p[i - 1, j] + p[i, j + 1] + p[i, j - 1] - T(4) * p[i, j]
                osc[i, j] = lap * lap
                d = p[i, j] - pbar
                dev2[i, j] = d * d
            else
                osc[i, j] = zero(T)
                dev2[i, j] = zero(T)
            end
        end
    end
end

# Per-cell SIMPLE response coefficient d = αu / a_p, with the boundary-aware
# relaxed diagonal a_p = ν·(4 + n_wall)/h² / αu (n_wall = number of Dirichlet wall
# faces the cell touches). So d = αu² h² / (ν (4 + n_wall)). Using the per-cell
# (not uniform) diagonal — matching cavity.jl — keeps the Rhie-Chow correction
# from over-shooting at boundary/corner cells (which otherwise spikes v in the
# corners and weakens the interior centreline). `dnum = αu² h² / ν`.
@kernel function cav_fill_d_kernel!(d, dnum, N)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= N && j <= N
            nwall = 0
            if i == 1; nwall += 1; end
            if i == N; nwall += 1; end
            if j == 1; nwall += 1; end
            if j == N; nwall += 1; end
            d[i, j] = eltype(d)(dnum) / (4 + nwall)
        end
    end
end

# =============================================================================
# Per-iteration phase functions (function barrier / injectable executor seam).
#
# The SIMPLE outer iteration is split into four phases over a shared state
# NamedTuple `S` (all device buffers + host scalars, built ONCE in the solver —
# every buffer is allocation-stable across iterations):
#
#   phase 1  fused grad+Rhie-Chow  +  fused convection+momentum-RHS
#   phase 2  momentum predictor MG solves (u, v) + predictor copy
#   phase 3  fused grad+Rhie-Chow (predictor)  +  fused divergence+pressure-RHS
#   phase 4  pressure-correction MG solve + zero-mean gauge + fused corrections
#
# An OFF-STRIDE iteration is exactly phase1;2;3;4 — with `mg_cycles>0` this is a
# STATIC kernel-launch sequence with zero host syncs and zero device
# allocations (with `static_gauge=true`; the default `sum!` gauge may allocate
# a GPU reduction temporary), i.e. CUDA-graph-capturable as a single unit:
# `_cavity_mg_offstride_step!`. An ON-STRIDE (checked) iteration runs the same
# phases with the norm computations and the convergence break interleaved
# between them (in the solver loop), exactly as before this refactor — the
# launch/FP sequence is unchanged, so results are bit-identical.
#
# The solver accepts an injectable `offstride_executor(S)` replacing
# `_cavity_mg_offstride_step!` for off-stride iterations only; the CUDA-graph
# wrapper (cavity_mg_cuda.jl, loaded only in GPU jobs) captures the step once
# and replays the instantiated graph. This file stays CUDA-free.
#
# Host-side bookkeeping note: `S.counters` ([momentum cycles; pressure cycles])
# is incremented by HOST code inside phases 2/4. A replayed CUDA graph re-runs
# only the recorded DEVICE work, so captured iterations do not increment the
# counters (fixed-cycles mode is required there, where the per-iteration counts
# are constant anyway).
# =============================================================================

function _cav_phase1!(S)
    cav_grad_rhie_chow_kernel!(S.kab)(S.gpx, S.gpy, S.uf, S.vf, S.u, S.v, S.p,
                                      S.d_u, S.d_v, S.invdx, S.invdy, S.N;
                                      ndrange=(S.N, S.N))
    cav_conv_mom_rhs_kernel!(S.kab)(S.rhs_u, S.rhs_v, S.u, S.v, S.uf, S.vf,
                                    S.gpx, S.gpy, S.invdx, S.invdy, S.invnu,
                                    S.sigma_mom, S.invh2, S.U_lid, S.L2, S.N;
                                    ndrange=(S.N, S.N))
    return nothing
end

function _cav_phase2!(S)
    upred, ncu, _ = solve_poisson_mg(S.rhs_u, S.N;
                                     bc=:dirichlet, backend_ka=S.kab, atype=S.atype,
                                     tol=S.mom_mg_tol, maxcycles=S.mom_mg_maxcycles,
                                     sigma=S.sigma_mom_mg, u0=S.u,
                                     fixed_cycles=S.nfixed_mom, hier=S.mg_hier,
                                     mixed_precision=S.mom_mg_mixed_precision,
                                     hier_f32=S.mg_hier_f32)
    copyto!(S.u, upred)
    vpred, ncv, _ = solve_poisson_mg(S.rhs_v, S.N;
                                     bc=:dirichlet, backend_ka=S.kab, atype=S.atype,
                                     tol=S.mom_mg_tol, maxcycles=S.mom_mg_maxcycles,
                                     sigma=S.sigma_mom_mg, u0=S.v,
                                     fixed_cycles=S.nfixed_mom, hier=S.mg_hier,
                                     mixed_precision=S.mom_mg_mixed_precision,
                                     hier_f32=S.mg_hier_f32)
    copyto!(S.v, vpred)
    S.counters[1] += ncu + ncv
    return nothing
end

function _cav_phase3!(S)
    cav_grad_rhie_chow_kernel!(S.kab)(S.gpx, S.gpy, S.uf, S.vf, S.u, S.v, S.p,
                                      S.d_u, S.d_v, S.invdx, S.invdy, S.N;
                                      ndrange=(S.N, S.N))
    cav_div_prhs_kernel!(S.kab)(S.divstar, S.rhs_p, S.uf, S.vf, S.invdx, S.invdy,
                                S.prhs_scale, S.N; ndrange=(S.N, S.N))
    return nothing
end

function _cav_phase4!(S)
    pc, ncp, _ = solve_poisson_mg(S.rhs_p, S.N;
                                  bc=:neumann, backend_ka=S.kab, atype=S.atype,
                                  tol=S.mg_tol, maxcycles=S.mg_maxcycles,
                                  sigma=0.0, fixed_cycles=S.nfixed, hier=S.mg_hier,
                                  mixed_precision=S.mg_mixed_precision,
                                  hier_f32=S.mg_hier_f32)
    copyto!(S.pcorr, pc)
    S.counters[2] += ncp
    # Neumann gauge: zero-mean. Fixed-cycles path keeps the mean ON DEVICE; the
    # legacy path reduces to a host scalar exactly as before. `static_gauge`
    # swaps the library `sum!` for the fixed two-kernel reduction (allocation-
    # free, capture-safe; last-ulp summation-order difference only).
    if S.use_fixed
        if S.static_gauge
            cav_colsum_kernel!(S.kab)(S.csum, S.pcorr, S.N; ndrange=(S.N,))
            cav_gauge_finalize_kernel!(S.kab)(S.gbuf, S.csum, S.N; ndrange=(1,))
        else
            sum!(S.gbuf, S.pcorr)
        end
        cav_shift_mean_kernel!(S.kab)(S.pcorr, S.gbuf, S.invNN, S.N; ndrange=(S.N, S.N))
    else
        m = sum(S.pcorr) / (S.N * S.N)
        cav_shift_kernel!(S.kab)(S.pcorr, m, S.N; ndrange=(S.N, S.N))
    end
    cav_apply_corrections_kernel!(S.kab)(S.uf, S.vf, S.u, S.v, S.p, S.pcorr,
                                         S.d_u, S.d_v, S.invdx, S.invdy,
                                         S.alpha_p, S.N; ndrange=(S.N, S.N))
    return nothing
end

"""
    _cavity_mg_offstride_step!(S)

One full OFF-STRIDE SIMPLE outer iteration (no norms, no host syncs): phases
1-4 back-to-back. With `mg_cycles>0` (+ `static_gauge=true` for the gauge
reduction) this is a static, allocation-free device launch sequence — the unit
the CUDA-graph executor captures and replays.
"""
function _cavity_mg_offstride_step!(S)
    _cav_phase1!(S)
    _cav_phase2!(S)
    _cav_phase3!(S)
    _cav_phase4!(S)
    return nothing
end

"""
    solve_incns_cavity_mg(; nx=128, ny=128, U_lid=1.0, Re=100.0,
                          relax=(u=0.7, p=0.3), tol=1e-7, vel_tol=1e-6,
                          maxiter=8000, L=1.0, backend_ka=CPU(),
                          atype=Array{Float64}, mg_tol=1e-3, mg_maxcycles=20,
                          mom_mg_tol=1e-3, mom_mg_maxcycles=20,
                          norm_stride=25, mg_cycles=3, mom_mg_cycles=1,
                          verbose=false)

Backend-parametric steady SIMPLE incompressible solver for the 2D lid-driven
cavity on `[0,L]^2` (square grid, `nx == ny`). Top wall moves at `u=U_lid, v=0`;
the other three walls are no-slip. `Re = U_lid*L/nu`, `rho=1` so `mu=nu`.

Both the viscous/momentum predictor AND the pressure-correction Poisson are
solved with the matrix-free multigrid `solve_poisson_mg`: the momentum solve uses
a Helmholtz shift `σ` (under-relaxation/pseudo-transient) and the lid Dirichlet
value folded into the RHS (ghost-0 convention); the pressure solve is all-Neumann
(`bc=:neumann`, σ=0). Every field is allocated via `atype` and every elementwise
op is a KA `@kernel` launched on `backend_ka`, so passing `backend_ka=CUDABackend()`
and `atype=CuArray{Float64}` runs the identical source on the GPU.

Latency/orchestration keywords (see file header):
  `norm_stride`  compute the outer convergence norms (continuity residual +
                 velocity settle) only every `norm_stride` iterations (plus
                 iteration 1 and `maxiter`). The norms only gate the STOP
                 decision, so the iterates are unchanged; the solver stops at
                 the first CHECKED iteration satisfying the criterion, i.e. up
                 to `norm_stride-1` iterations later (slightly more converged)
                 than `norm_stride=1`. `residual_history` holds one entry per
                 CHECK. Default 25. `norm_stride=1` = check every iteration.
  `mg_cycles`    when > 0, run every inner MG solve with a FIXED number of
                 V-cycles and no residual checks / zero-mean projections (zero
                 inner host syncs; static launch sequence): the PRESSURE solve
                 runs `mg_cycles` V-cycles (default 3 — matches the legacy
                 1e-3 inner tolerance from a zero start) and each MOMENTUM
                 solve runs `mom_mg_cycles` V-cycles (default 1 — the warm-
                 started, strongly σ-dominant momentum solve needs ~1 cycle at
                 the legacy tolerance). `mg_tol`, `mg_maxcycles`, `mom_mg_tol`,
                 `mom_mg_maxcycles` are then ignored. `mg_cycles=0` = legacy
                 tolerance-driven inner solves (validated against Ghia Re=100).
  `mom_mg_cycles` fixed V-cycles per momentum solve when `mg_cycles > 0`
                 (ignored when `mg_cycles=0`). Default 1.

GPU-efficiency opt-ins (ALL default OFF — defaults are bit-identical to the
previous revision):
  `mg_mixed_precision`     run the inner PRESSURE MG solves in mixed precision
                 (F64 defect correction wrapping F32 V-cycles; see
                 `solve_poisson_mg`'s `mixed_precision`). The outer SIMPLE
                 iterate stays Float64.
  `mom_mg_mixed_precision` same for the MOMENTUM MG solves. Exposed separately
                 from the pressure flag because the two operators differ
                 (σ-shifted Dirichlet vs singular Neumann) and may tolerate
                 F32 differently.
  `static_gauge` replace the library `sum!` in the fixed-cycles zero-mean gauge
                 with a fixed two-kernel reduction (allocation-free, static
                 launch count — required for CUDA-graph capture; last-ulp
                 summation-order difference only). Ignored when `mg_cycles=0`.
  `offstride_executor` injectable `f(S)` executing one full off-stride outer
                 iteration (default `_cavity_mg_offstride_step!`). Seam for the
                 CUDA-graph wrapper in cavity_mg_cuda.jl; on-stride (checked)
                 iterations always run the regular uncaptured path.

Returns a NamedTuple with `u, v, p` (nx x ny host `Array`s for convenience),
`residual_history`, `iters`, `converged`, grid metrics, and `checkerboard`.

Validation & performance receipts
  * Ghia, Ghia & Shin (1982) centreline profiles: max deviation 0.689% of
    `U_lid` at Re=100 (128², gate <=5%) and 2.31% at Re=1000 (512², improving
    with grid) — `test/analytical/incns_cavity_mg_ghia.jl` (set
    `INCNS_MG_GHIA_SKIP_RE1000=1` to skip the long Re=1000 case) and
    `benchmarks/results/cavity_gpu_aqua_a100.md` (issue #7).
  * GPU↔CPU BIT-EXACT parity (‖Δ‖∞ ~1e-16): the same source run with
    `backend_ka=CUDABackend(), atype=CuArray{Float64}` reproduces the CPU
    fields — `benchmarks/krk/inc_ns/cavity_gpu_bench.jl`.
  * Fast path (`norm_stride=25, mg_cycles=3, mom_mg_cycles=1`, the defaults):
    converged fields deviate <=4.3e-5 RELATIVE from the stride-1
    tolerance-driven solve and stop 11 iterations later (i.e. slightly MORE
    converged) — `test/scratch/incns_cavity_mg_fastpath_driver.jl`.
  * Mixed precision (`mg_mixed_precision`/`mom_mg_mixed_precision`): converged
    Ghia Re=100 fields deviate <=8.2e-10 from the all-Float64 path —
    `test/scratch/incns_cavity_mg_mixed_precision_driver.jl`.
  * CUDA-graph front-end: `solve_incns_cavity_mg_cuda_graph`
    (`cavity_mg_cuda.jl`, loaded only under `using CUDA`) replays each
    off-stride iteration as one graph launch through the `offstride_executor`
    seam (issue #8).

Registered in `src/KrakeNN.jl` and exported (also reachable as
`solve(params, IncNS(:cavity_mg))`); the file remains standalone-include-able
(its guard pulls in `src/solve/poisson_mg.jl`).
"""
function solve_incns_cavity_mg(; nx::Integer=128, ny::Integer=128,
                               U_lid::Real=1.0, Re::Real=100.0,
                               relax=(u=0.7, p=0.3),
                               tol::Real=1e-7, vel_tol::Real=1e-6,
                               maxiter::Integer=8000,
                               L::Real=1.0,
                               backend_ka=KernelAbstractions.CPU(),
                               atype::Type{<:AbstractArray}=Array{Float64},
                               mg_tol::Real=1e-3, mg_maxcycles::Integer=20,
                               mom_mg_tol::Real=1e-3, mom_mg_maxcycles::Integer=20,
                               norm_stride::Integer=25,
                               mg_cycles::Integer=3,
                               mom_mg_cycles::Integer=1,
                               mg_mixed_precision::Bool=false,
                               mom_mg_mixed_precision::Bool=false,
                               static_gauge::Bool=false,
                               offstride_executor=nothing,
                               verbose::Bool=false)
    nx == ny || throw(ArgumentError("cavity_mg requires a square grid (nx == ny)"))
    N = Int(nx)
    U_lid = Float64(U_lid); Re = Float64(Re); L = Float64(L)
    αu = Float64(relax.u); αp = Float64(relax.p)
    nu = U_lid * L / Re                 # rho = 1 -> mu = nu
    dx = L / N; dy = L / N
    h = dx
    invdx = 1.0 / dx; invdy = 1.0 / dy
    invh2 = 1.0 / (h * h)
    kab = backend_ka
    stride = max(Int(norm_stride), 1)
    nfixed = max(Int(mg_cycles), 0)                       # pressure fixed cycles
    use_fixed = nfixed > 0
    nfixed_mom = use_fixed ? max(Int(mom_mg_cycles), 1) : 0  # momentum fixed cycles
    maxit = Int(maxiter)

    # ----- SIMPLE response coefficient d = αu / a_p (relaxed) -----
    # Momentum under-relaxation is folded into the operator DIAGONAL via the
    # Helmholtz shift σ (below), so the effective momentum diagonal is the relaxed
    # a_p = a_visc/αu with a_visc = ν·4/h². The SIMPLE velocity-pressure response
    # is then d = αu / a_p = αu² / a_visc — identical to cavity.jl, which
    # reproduces Ghia's v-profile. Uniform d (collocated regular grid).
    a_visc = nu * 4.0 * invh2
    a_p = a_visc / αu
    dval = αu / a_p                     # = αu² / a_visc
    # Pressure-correction operator is dscalar*(-∇²); on the unit-square-scaled MG
    # the d-coefficient is folded into the RHS scaling (see below), so MG solves
    # the PLAIN Neumann Laplacian and we rescale.

    # ----- momentum under-relaxation as a Helmholtz shift (matches cavity.jl) ----
    # cavity.jl folds SIMPLE momentum under-relaxation into the operator diagonal:
    #   A_relaxed = A_visc + (1/αu - 1) Diag(a_visc),  rhs += (1/αu-1) a_visc φ_old.
    # Dividing by ν gives exactly the MG Helmholtz form (-∇² + σ) with
    #   σ = (1/αu - 1) · 4/h²   and   rhs += σ φ_old   (interior a_visc/ν = 4/h²).
    # The predictor IS the under-relaxed velocity, so we copy it straight to u,v
    # (no second, explicit relaxation). At the fixed point φ=φ_old the σ term
    # cancels and the converged field satisfies the un-relaxed steady momentum
    # equation — reproducing Ghia. σ also gives strong diagonal dominance, which
    # stabilises the deferred-correction advection at high Re and makes the MG
    # momentum solve converge in very few V-cycles.
    sigma_mom = (1.0 / αu - 1.0) * 4.0 * invh2

    # MG solves on the unit square (spacing 1/N). The physical operator uses
    # spacing h=L/N; on a square domain L scales (-∇²) by 1/L², so we pass the
    # RHS scaled by L² (and σ by L²). With L=1 these are identities.
    L2 = L * L
    prhs_scale = L2 / dval
    invNN = 1.0 / (N * N)

    # ----- backend fields (all via atype) -----
    mk() = (a = atype(undef, N, N); fill!(a, 0.0); a)
    u = mk(); v = mk(); p = mk()
    u_old = mk(); v_old = mk()
    gpx = mk(); gpy = mk()
    uf = mk(); vf = mk()
    divstar = mk(); pcorr = mk()
    rhs_u = mk(); rhs_v = mk(); rhs_p = mk()
    d_u = mk(); d_v = mk()
    delta = mk()
    osc = mk(); dev2 = mk()
    gbuf = atype(undef, 1, 1); fill!(gbuf, 0.0)   # device-resident gauge mean
    # per-cell d = αu² h² / (ν (4 + n_wall)); dnum = αu² h² / ν = dval * 4
    # (dval = αu²/(ν·4/h²) is the interior value).
    dnum = αu * αu / (nu * invh2)
    cav_fill_d_kernel!(kab)(d_u, dnum, N; ndrange=(N, N))
    cav_fill_d_kernel!(kab)(d_v, dnum, N; ndrange=(N, N))

    # ONE multigrid hierarchy shared by all three inner solves (momentum u,
    # momentum v, pressure): every level array is fully (re)initialized per
    # solve, so reuse is value-identical and removes 3 full level-stack
    # allocations per outer iteration.
    mg_hier = build_mg_hierarchy(N, atype)
    # Shared Float32 hierarchy for the mixed-precision inner solves (only
    # allocated when an MP flag is on; fully reinitialized per solve like
    # mg_hier, so sharing it across the three solves is value-identical).
    mg_hier_f32 = (mg_mixed_precision || mom_mg_mixed_precision) ?
        build_mg_hierarchy(N, _mg_eltype_variant(atype, Float32)) : nothing
    # 1xN column-sum buffer for the static gauge reduction.
    csum = atype(undef, 1, N); fill!(csum, 0.0)
    # Inner-cycle counters [momentum; pressure], host-side (see phase functions).
    counters = zeros(Int, 2)

    # Shared per-iteration state for the phase functions / injectable executor.
    # EVERY device buffer the iteration touches lives here and in mg_hier(.f32);
    # nothing below reallocates them, so the off-stride step is allocation-
    # stable across iterations (CUDA-graph requirement).
    S = (; kab, N, atype,
         u, v, p, gpx, gpy, uf, vf, divstar, pcorr, rhs_u, rhs_v, rhs_p,
         d_u, d_v, gbuf, csum,
         invdx, invdy, invnu=1.0 / nu, sigma_mom, sigma_mom_mg=sigma_mom * L2,
         invh2, U_lid, L2, prhs_scale, invNN, alpha_p=αp,
         mg_tol, mg_maxcycles=Int(mg_maxcycles),
         mom_mg_tol, mom_mg_maxcycles=Int(mom_mg_maxcycles),
         nfixed, nfixed_mom, use_fixed, static_gauge,
         mg_mixed_precision, mom_mg_mixed_precision,
         mg_hier, mg_hier_f32, counters)
    offexec = offstride_executor === nothing ? _cavity_mg_offstride_step! :
              offstride_executor

    xcenters = [(i - 0.5) * dx for i in 1:N]
    ycenters = [(j - 0.5) * dy for j in 1:N]

    residual_history = Float64[]
    converged = false
    iters = 0
    vel_change = Inf
    res = Inf
    ref_flux = U_lid

    for it in 1:maxit
        iters = it
        # Convergence norms only every `stride` iterations (+ first and last):
        # they gate the STOP decision only, so skipping them does not change the
        # iterates. With stride=1 this is the legacy every-iteration behavior.
        do_check = (it == 1) || (it % stride == 0) || (it == maxit)

        if !do_check
            # OFF-STRIDE iteration: one static device launch sequence (no norms,
            # no host syncs), via the injectable executor (CUDA-graph seam).
            offexec(S)
            continue
        end

        # ON-STRIDE (checked) iteration: the SAME phase sequence with the norm
        # computations and the convergence break interleaved, exactly as before.

        # 1+2. fused: cell pressure gradient + Rhie-Chow face velocities,
        #      then deferred-correction convection + momentum RHS (×L²).
        _cav_phase1!(S)

        # snapshot old velocity for the settle metric (device-to-device copy,
        # stream-ordered, no host sync)
        copyto!(u_old, u); copyto!(v_old, v)

        # momentum predictor: Helmholtz MG solves, warm-started from the current
        # velocity. The σ-shifted predictor IS the under-relaxed velocity
        # (relaxation lives in the operator/RHS): copied straight in.
        _cav_phase2!(S)

        # settle metric (device reductions; host scalar reads = the only syncs)
        cav_delta_kernel!(kab)(delta, u, u_old, v, v_old, N; ndrange=(N, N))
        umax_prev = max(_field_absmax(u_old), _field_absmax(v_old), U_lid * eps())
        vel_change = _field_absmax(delta) / umax_prev

        # 5+6. Rhie-Chow faces from the predictor + continuity residual
        #      divstar = div(u*_face) AND the scaled pressure RHS.
        _cav_phase3!(S)

        res = sqrt(sum(abs2, divstar) / (N * N)) * dx / max(ref_flux, eps())
        push!(residual_history, res)
        if verbose && (it <= 5 || it % 100 == 0)
            @info "cavity_mg SIMPLE" it res vel_change
        end
        # Convergence: continuity residual below `tol` AND the velocity field
        # essentially steady (settle below `vel_tol`). The velocity-settle
        # gate is looser because explicit momentum under-relaxation settles
        # more slowly than the continuity residual drops.
        if res < tol && vel_change < vel_tol
            converged = true
            break
        end

        # 7-10. pressure-correction Poisson + zero-mean gauge + fused
        #       face/cell-velocity corrections and pressure update.
        _cav_phase4!(S)
    end

    # One final device barrier before host post-processing.
    KernelAbstractions.synchronize(kab)

    # checkerboard metric via device reductions
    pbar = sum(p) / (N * N)
    cav_checker_kernel!(kab)(osc, dev2, p, pbar, N; ndrange=(N, N))
    checkerboard = sqrt(sum(osc) / max(sum(dev2), eps()))

    # Return host copies for convenience (interpolation/plotting on the host).
    return (; u=Array(u), v=Array(v), p=Array(p),
            residual_history, iters, converged, vel_change,
            dx, dy, xcenters, ycenters, nx=N, ny=N, U_lid, Re, mu=nu, L,
            checkerboard, sigma_mom,
            norm_stride=stride, mg_cycles=nfixed, mom_mg_cycles=nfixed_mom,
            mg_mixed_precision, mom_mg_mixed_precision, static_gauge,
            mg_cycles_mom_total=counters[1], mg_cycles_p_total=counters[2])
end

# Device reduction for the max abs value (no host scalar indexing of device arrays).
_field_absmax(a) = maximum(abs, a)
