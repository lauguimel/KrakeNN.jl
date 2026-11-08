# =============================================================================
# Matrix-free geometric multigrid (V-cycle) Poisson solver, GPU-native via
# KernelAbstractions (same source CPU + CUDA).
#
# WHY THIS EXISTS
# ---------------
# cuDSS sparse-direct under-uses the GPU (~9% occupancy on the pressure-Poisson
# solve). A matrix-free geometric multigrid whose smoothers are LBM-like 5-point
# stencils saturates the GPU and is O(N) per V-cycle, with a V-cycle count that
# is asymptotically INDEPENDENT of N (the multigrid hallmark). This module is the
# GPU-performance path; it is written KA-generic so a CUDA backend drops in with
# NO source change (every elementwise op is an `@kernel` launched on `backend`).
#
# OPERATOR CONVENTION (must match src/solve/poisson.jl exactly)
# -------------------------------------------------------------
# We solve  -∇²u = f  on the unit square, cell-centred regular grid, cell centres
# at ((i-0.5)h, (j-0.5)h), h = 1/N. The 5-point operator is
#
#     (-∇²u)_{ij} = (4 u_{ij} - u_{i-1,j} - u_{i+1,j} - u_{i,j-1} - u_{i,j+1}) / h²
#
# with the DIRICHLET convention of `assemble_poisson_dirichlet`: a missing
# neighbour across a Dirichlet boundary face adds +1/h² to the diagonal and the
# (zero) boundary value contributes nothing to the off-diagonal — i.e. the ghost
# value is taken as 0 at the ghost CELL CENTRE (the "+1/h² per face", NOT "+2/h²"
# half-spacing-corrected convention). Inhomogeneous Dirichlet data is folded into
# the RHS by the caller (as poisson.jl does via `b`).
#
# NEUMANN convention matches `assemble_poisson_neumann_unpinned`: a missing
# neighbour across a Neumann face SUBTRACTS 1/h² from the diagonal (zero-flux
# mirror), giving a singular all-Neumann operator. The singular nullspace
# (constants) is removed by pinning one DOF (`pin_k0`), matching poisson.jl.
#
# COMPONENTS (all `@kernel` on `backend`)
# ---------------------------------------
#   * residual         r = f - (-∇²)u   (matrix-free 5-point apply)
#   * smoother         weighted Jacobi (ω=2/3) OR red-black Gauss-Seidel.
#                      DEFAULT = RBGS: it smooths high-frequency error far better
#                      per sweep (smoothing factor ~0.25 vs ~0.6 for Jacobi) so
#                      the V-cycle count is lower and flatter. RBGS is two
#                      colour-kernel launches per sweep; fully data-parallel, so
#                      it is just as GPU-friendly as Jacobi.
#   * restriction      full-weighting fine -> coarse
#   * prolongation     bilinear coarse -> fine, with correction add
#   * coarsest grid    heavy RBGS smoothing (grid is ~2x2 .. 8x8)
#   * V-cycle driver   iterate to a relative-residual tolerance
#
# Only KernelAbstractions + LinearAlgebra (norm) are used. No CUDA-specific code,
# no scalar indexing of device arrays in host loops.
# =============================================================================

using KernelAbstractions
using LinearAlgebra: norm

# Boundary-condition tags for the matrix-free operator. Distinct from the FVFD
# UInt8 BC consts (those are for the velocity operators); these select the
# diagonal correction at boundary faces.
if !isdefined(@__MODULE__, :MG_BC_DIRICHLET)
    const MG_BC_DIRICHLET = UInt8(1)   # missing neighbour -> diag += 1/h² (ghost 0)
    const MG_BC_NEUMANN   = UInt8(2)   # missing neighbour -> diag -= 1/h² (mirror)
end

# -----------------------------------------------------------------------------
# Backend tag reuse. linear_solve.jl defines CPUBackendTag/CUDABackendTag and the
# factorize-once seam. We accept those tags at the API boundary and translate to
# a KernelAbstractions backend for the kernels. Pull it in if not present so this
# file is standalone-include-able.
# -----------------------------------------------------------------------------
if !isdefined(@__MODULE__, :CPUBackendTag)
    include(joinpath(@__DIR__, "linear_solve.jl"))
end

"""
    _mg_ka_backend(tag) -> KernelAbstractions backend

Translate a `LinearSolveBackend` tag (CPU/CUDA) into the concrete KA backend used
to launch kernels. The CUDA branch is intentionally written so it only resolves
`CUDABackend` when CUDA is loaded (inside a GPU job); on a CPU-only box only the
`CPUBackendTag` method is ever called.
"""
_mg_ka_backend(::CPUBackendTag) = KernelAbstractions.CPU()
# CUDA path: resolved at call time inside a `using CUDA` job. Kept as a generic
# fallback so this file compiles CUDA-free.
function _mg_ka_backend(::CUDABackendTag)
    error("CUDABackendTag requires `using CUDA` (provides CUDABackend()). " *
          "Pass the live CUDA backend via `backend_ka=CUDABackend()` instead.")
end

# =============================================================================
# Matrix-free 5-point operator  L u = (-∇²)u   and residual  r = f - L u
# =============================================================================

@kernel function mg_laplacian_apply_kernel!(
    Lu, @Const(u), invh2, sigma, west_bc, east_bc, south_bc, north_bc, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            T = eltype(Lu)
            uc = u[i, j]
            acc = T(4) * uc

            # West face (i-1)
            if i > 1
                acc -= u[i - 1, j]
            elseif west_bc == MG_BC_DIRICHLET
                acc += uc                 # diag += 1/h² (ghost value 0)
            else                          # Neumann: zero-flux mirror -> diag -= 1/h²
                acc -= uc
            end

            # East face (i+1)
            if i < Nx
                acc -= u[i + 1, j]
            elseif east_bc == MG_BC_DIRICHLET
                acc += uc
            else
                acc -= uc
            end

            # South face (j-1)
            if j > 1
                acc -= u[i, j - 1]
            elseif south_bc == MG_BC_DIRICHLET
                acc += uc
            else
                acc -= uc
            end

            # North face (j+1)
            if j < Ny
                acc -= u[i, j + 1]
            elseif north_bc == MG_BC_DIRICHLET
                acc += uc
            else
                acc -= uc
            end

            # Helmholtz shift: (σI - ∇²)u. σ=0 recovers the plain Poisson operator.
            Lu[i, j] = acc * invh2 + sigma * uc
        end
    end
end

@kernel function mg_residual_kernel!(
    r, @Const(u), @Const(f), invh2, sigma, west_bc, east_bc, south_bc, north_bc, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            T = eltype(r)
            uc = u[i, j]
            acc = T(4) * uc

            if i > 1
                acc -= u[i - 1, j]
            elseif west_bc == MG_BC_DIRICHLET
                acc += uc
            else
                acc -= uc
            end
            if i < Nx
                acc -= u[i + 1, j]
            elseif east_bc == MG_BC_DIRICHLET
                acc += uc
            else
                acc -= uc
            end
            if j > 1
                acc -= u[i, j - 1]
            elseif south_bc == MG_BC_DIRICHLET
                acc += uc
            else
                acc -= uc
            end
            if j < Ny
                acc -= u[i, j + 1]
            elseif north_bc == MG_BC_DIRICHLET
                acc += uc
            else
                acc -= uc
            end

            r[i, j] = f[i, j] - (acc * invh2 + sigma * uc)
        end
    end
end

# =============================================================================
# Smoothers
# =============================================================================
# The diagonal of L is  d_{ij} = (4 + nbnd_dir - nbnd_neu) / h²  where nbnd_dir /
# nbnd_neu are the counts of missing Dirichlet / Neumann boundary faces at (i,j).
# We compute it inline in the smoother kernels (cheap, branch on boundary).

@inline function _mg_diag_count(i, j, Nx, Ny, west_bc, east_bc, south_bc, north_bc)
    # returns (4 + Σ_dir - Σ_neu) as an Int
    d = 4
    if i == 1
        d += west_bc == MG_BC_DIRICHLET ? 1 : -1
    end
    if i == Nx
        d += east_bc == MG_BC_DIRICHLET ? 1 : -1
    end
    if j == 1
        d += south_bc == MG_BC_DIRICHLET ? 1 : -1
    end
    if j == Ny
        d += north_bc == MG_BC_DIRICHLET ? 1 : -1
    end
    return d
end

@inline function _mg_offdiag_sum(u, i, j, Nx, Ny)
    # Σ of off-diagonal neighbour contributions (with the +1 sign of -∇², i.e.
    # the value that, divided by h², equals -(off-diag * u_nb)). Missing
    # neighbours contribute 0 (their effect is already folded into the diagonal).
    @inbounds begin
        s = zero(eltype(u))
        if i > 1;  s += u[i - 1, j]; end
        if i < Nx; s += u[i + 1, j]; end
        if j > 1;  s += u[i, j - 1]; end
        if j < Ny; s += u[i, j + 1]; end
        return s
    end
end

# --- Weighted Jacobi --------------------------------------------------------
# u_new = u + ω D⁻¹ (f - L u). With L = (diag*I - offdiag)/h², D = diag/h², so
#   u_new = (1-ω) u + ω (h² f + Σ_nb u_nb) / diag.
@kernel function mg_jacobi_kernel!(
    unew, @Const(u), @Const(f), h2, omega, sigma,
    west_bc, east_bc, south_bc, north_bc, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            T = eltype(unew)
            # Helmholtz diagonal: (diag_count + σ h²) since D = diag_count/h² + σ.
            d = T(_mg_diag_count(i, j, Nx, Ny, west_bc, east_bc, south_bc, north_bc)) + sigma * h2
            s = _mg_offdiag_sum(u, i, j, Nx, Ny)
            gs = (h2 * f[i, j] + s) / d
            unew[i, j] = (one(T) - omega) * u[i, j] + omega * gs
        end
    end
end

# --- Red-black Gauss-Seidel -------------------------------------------------
# colour = mod(i+j, 2). Updates cells of one colour in place using the latest
# neighbour values (neighbours are all the other colour, so this is a true GS
# sweep and fully data-parallel within a colour).
@kernel function mg_rbgs_kernel!(
    u, @Const(f), h2, colour, sigma,
    west_bc, east_bc, south_bc, north_bc, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny && ((i + j) % 2 == colour)
            T = eltype(u)
            d = T(_mg_diag_count(i, j, Nx, Ny, west_bc, east_bc, south_bc, north_bc)) + sigma * h2
            s = _mg_offdiag_sum(u, i, j, Nx, Ny)
            u[i, j] = (h2 * f[i, j] + s) / d
        end
    end
end

# =============================================================================
# Restriction (full-weighting fine -> coarse) and prolongation (bilinear)
# =============================================================================
# Cell-centred grids: a coarse cell (I,J) covers fine cells (2I-1:2I, 2J-1:2J).
# Full-weighting here is the cell-centred 2x2 average of the fine residual
# (the natural restriction whose transpose is the bilinear prolongation up to a
# constant; for cell-centred MG the simple 2x2 average + bilinear interpolation
# pair is standard and gives textbook V-cycle convergence).
@kernel function mg_restrict_kernel!(rc, @Const(rf), Nxc, Nyc)
    I, J = @index(Global, NTuple)
    @inbounds begin
        if I <= Nxc && J <= Nyc
            T = eltype(rc)
            i = 2 * I - 1
            j = 2 * J - 1
            rc[I, J] = (rf[i, j] + rf[i + 1, j] + rf[i, j + 1] + rf[i + 1, j + 1]) / T(4)
        end
    end
end

# Bilinear prolongation with correction add: uf += P ec. For cell-centred grids
# each fine cell interpolates from the four nearest coarse cell centres with
# weights (9/16, 3/16, 3/16, 1/16). We clamp coarse indices at the boundary
# (constant extrapolation), which is the standard cell-centred treatment.
@kernel function mg_prolong_add_kernel!(uf, @Const(ec), Nxf, Nyf, Nxc, Nyc)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nxf && j <= Nyf
            T = eltype(uf)
            # Coarse cell directly containing this fine cell.
            Ic = (i + 1) >> 1     # = cld(i,2)
            Jc = (j + 1) >> 1
            # Direction to the neighbouring coarse cell (depends on fine parity).
            di = (i % 2 == 1) ? -1 : 1     # odd fine -> left coarse neighbour
            dj = (j % 2 == 1) ? -1 : 1
            Ic2 = Ic + di
            Jc2 = Jc + dj
            # Clamp (constant extrapolation at boundaries).
            Ic2 = Ic2 < 1 ? 1 : (Ic2 > Nxc ? Nxc : Ic2)
            Jc2 = Jc2 < 1 ? 1 : (Jc2 > Nyc ? Nyc : Jc2)
            w  = T(9) / T(16)
            wx = T(3) / T(16)
            wy = T(3) / T(16)
            wc = T(1) / T(16)
            val = w  * ec[Ic,  Jc] +
                  wx * ec[Ic2, Jc] +
                  wy * ec[Ic,  Jc2] +
                  wc * ec[Ic2, Jc2]
            uf[i, j] += val
        end
    end
end

@kernel function mg_zero_kernel!(a, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            a[i, j] = zero(eltype(a))
        end
    end
end

# Mixed-precision transfer kernels: elementwise eltype conversion (F64 residual
# -> F32 RHS) and converted-add (F32 correction -> F64 solution). Backend-generic
# (no copyto! between mismatched eltypes, which is not guaranteed on every
# device array type) and allocation-free, so they are CUDA-graph-capturable.
@kernel function mg_convert_kernel!(dst, @Const(src), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            dst[i, j] = convert(eltype(dst), src[i, j])
        end
    end
end

@kernel function mg_add_converted_kernel!(dst, @Const(src), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            dst[i, j] += convert(eltype(dst), src[i, j])
        end
    end
end

# Pin one DOF to remove the constant nullspace of the all-Neumann operator.
# We pin cell (1,1) to zero by projecting the constant out of the correction /
# subtracting its mean. For the V-cycle we instead enforce a zero-mean RHS and
# zero-mean solution at the finest level (the discrete compatibility condition).
@kernel function mg_shift_kernel!(u, shift, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            u[i, j] -= shift
        end
    end
end

# =============================================================================
# Host-side launch helpers (backend-generic).
#
# NO per-launch `synchronize`: KA CPU kernels execute synchronously (the call
# returns only after completion; `synchronize(::CPU)` is a no-op), and on CUDA
# all launches go to the task-local stream IN ORDER, so device-side data
# dependencies between consecutive kernels need no host sync. Host syncs happen
# implicitly only where a host scalar is read (reductions: `norm`, `sum`).
# Removing them eliminates the per-launch host<->device round trip that
# dominated the latency-bound SIMPLE cavity solve.
# =============================================================================

# Grid/operator scalars (h², 1/h², σ) are converted to the FIELD eltype before
# the launch: for the default Float64 hierarchy this is an identity conversion
# (bit-identical results, unchanged code path); for the mixed-precision Float32
# hierarchy it keeps ALL kernel arithmetic in Float32 (no silent F64 promotion
# inside the kernels, which would forfeit the F32 bandwidth/FLOP gain on GPU).

function _mg_apply!(Lu, u, invh2, sigma, bc, Nx, Ny, kab)
    T = eltype(Lu)
    mg_laplacian_apply_kernel!(kab)(Lu, u, T(invh2), T(sigma), bc[1], bc[2], bc[3], bc[4],
                                    Nx, Ny; ndrange = (Nx, Ny))
    return Lu
end

function _mg_residual!(r, u, f, invh2, sigma, bc, Nx, Ny, kab)
    T = eltype(r)
    mg_residual_kernel!(kab)(r, u, f, T(invh2), T(sigma), bc[1], bc[2], bc[3], bc[4],
                             Nx, Ny; ndrange = (Nx, Ny))
    return r
end

function _mg_smooth!(u, f, h2, sigma, nsweeps, bc, Nx, Ny, kab, smoother, scratch)
    T = eltype(u)
    h2c = T(h2)
    sigc = T(sigma)
    if smoother === :rbgs
        for _ in 1:nsweeps
            mg_rbgs_kernel!(kab)(u, f, h2c, 0, sigc, bc[1], bc[2], bc[3], bc[4],
                                 Nx, Ny; ndrange = (Nx, Ny))
            mg_rbgs_kernel!(kab)(u, f, h2c, 1, sigc, bc[1], bc[2], bc[3], bc[4],
                                 Nx, Ny; ndrange = (Nx, Ny))
        end
    else # :jacobi (weighted, ω=2/3), needs a scratch buffer (ping-pong)
        omega = T(2 // 3)
        for _ in 1:nsweeps
            mg_jacobi_kernel!(kab)(scratch, u, f, h2c, omega, sigc,
                                   bc[1], bc[2], bc[3], bc[4],
                                   Nx, Ny; ndrange = (Nx, Ny))
            copyto!(u, scratch)   # device-to-device, stream-ordered
        end
    end
    return u
end

function _mg_convert!(dst, src, Nx, Ny, kab)
    mg_convert_kernel!(kab)(dst, src, Nx, Ny; ndrange = (Nx, Ny))
    return dst
end

function _mg_add_converted!(dst, src, Nx, Ny, kab)
    mg_add_converted_kernel!(kab)(dst, src, Nx, Ny; ndrange = (Nx, Ny))
    return dst
end

function _mg_zero!(a, Nx, Ny, kab)
    mg_zero_kernel!(kab)(a, Nx, Ny; ndrange = (Nx, Ny))
    return a
end

function _mg_restrict!(rc, rf, Nxc, Nyc, kab)
    mg_restrict_kernel!(kab)(rc, rf, Nxc, Nyc; ndrange = (Nxc, Nyc))
    return rc
end

function _mg_prolong_add!(uf, ec, Nxf, Nyf, Nxc, Nyc, kab)
    mg_prolong_add_kernel!(kab)(uf, ec, Nxf, Nyf, Nxc, Nyc; ndrange = (Nxf, Nyf))
    return uf
end

# =============================================================================
# Multigrid hierarchy + V-cycle
# =============================================================================

"""
    MGHierarchy

Pre-allocated work arrays for every level of a `N x N` regular-grid V-cycle.
Level 1 is the finest. Each level stores the solution `u`, RHS `f`, residual `r`
and a scratch buffer (for Jacobi ping-pong / general temporaries). All arrays
live on `backend` (CPU now, CUDA later) and are allocated ONCE.
"""
struct MGHierarchy{A}
    sizes::Vector{Int}        # N at each level (square grids: Nx == Ny == N)
    h::Vector{Float64}        # mesh spacing at each level
    u::Vector{A}
    f::Vector{A}
    r::Vector{A}
    scratch::Vector{A}
end

"""
    _mg_eltype_variant(atype, T) -> array type with element type `T`

Derive the same backend array family with a different element type, e.g.
`Array{Float64} -> Array{Float32}` or `CuArray{Float64} -> CuArray{Float32}`.
Used to allocate the Float32 hierarchy for mixed-precision multigrid without
any backend-specific code (`Base.typename(...).wrapper` strips ALL type
parameters, so trailing parameters like `CuArray`'s memory kind fall back to
their constructor defaults).
"""
_mg_eltype_variant(::Type{A}, ::Type{T}) where {A <: AbstractArray, T} =
    Base.typename(A).wrapper{T}

"""
    build_mg_hierarchy(N, backend_array_template; min_size=4) -> MGHierarchy

Allocate the multigrid hierarchy for a square `N x N` grid by repeated
coarsening (N -> N/2) down to `min_size`. `backend_array_template` is a 0-length
array of the device array type (e.g. `Array{Float64}` on CPU) used to allocate
matching device arrays via `similar`. Requires `N` to be a power-of-two multiple
of a grid no smaller than `min_size` so every level halves cleanly.
"""
function build_mg_hierarchy(N::Integer, atype::Type{<:AbstractArray};
                            min_size::Integer = 4)
    N = Int(N)
    N > 0 || throw(ArgumentError("N must be positive"))
    sizes = Int[]
    n = N
    while true
        push!(sizes, n)
        (n <= min_size || isodd(n)) && break
        n = n ÷ 2
    end
    nl = length(sizes)
    mk(m) = (a = atype(undef, m, m); fill!(a, 0.0); a)
    A = typeof(mk(sizes[1]))
    u = A[mk(sizes[l]) for l in 1:nl]
    f = A[mk(sizes[l]) for l in 1:nl]
    r = A[mk(sizes[l]) for l in 1:nl]
    scratch = A[mk(sizes[l]) for l in 1:nl]
    h = [1.0 / sizes[l] for l in 1:nl]
    return MGHierarchy{A}(sizes, h, u, f, r, scratch)
end

"""
    vcycle!(hier, level, bc, kab; nu1, nu2, ncoarse, smoother, neumann_pin)

One recursive V-cycle starting at `level` (1 = finest). Pre-smooth `nu1` sweeps,
restrict the residual, recurse on the coarse correction (with zero initial
guess), prolong-and-correct, post-smooth `nu2` sweeps. On the coarsest level do
`ncoarse` smoothing sweeps (the grid is tiny so this is an effective direct
solve). For the singular all-Neumann case (`neumann_pin=true`) the coarse RHS and
solution are projected to zero mean to stay in the range of the operator.
"""
function vcycle!(hier::MGHierarchy, level::Int, bc, kab;
                 nu1::Int, nu2::Int, ncoarse::Int, smoother::Symbol,
                 neumann_pin::Bool, sigma::Float64 = 0.0)
    N = hier.sizes[level]
    h2 = hier.h[level]^2
    invh2 = 1.0 / h2
    u = hier.u[level]
    f = hier.f[level]
    r = hier.r[level]

    if level == length(hier.sizes)
        # Coarsest grid: smooth heavily (effective direct solve on a tiny grid).
        _mg_smooth!(u, f, h2, sigma, ncoarse, bc, N, N, kab, smoother, hier.scratch[level])
        neumann_pin && _project_zero_mean!(u, N, N, kab, hier.scratch[level])
        return nothing
    end

    # Pre-smooth.
    _mg_smooth!(u, f, h2, sigma, nu1, bc, N, N, kab, smoother, hier.scratch[level])

    # Residual r = f - L u.
    _mg_residual!(r, u, f, invh2, sigma, bc, N, N, kab)

    # Restrict residual to coarse RHS.
    Nc = hier.sizes[level + 1]
    fc = hier.f[level + 1]
    _mg_restrict!(fc, r, Nc, Nc, kab)
    neumann_pin && _project_zero_mean!(fc, Nc, Nc, kab, hier.scratch[level + 1])

    # Coarse correction starts at zero.
    _mg_zero!(hier.u[level + 1], Nc, Nc, kab)

    vcycle!(hier, level + 1, bc, kab; nu1 = nu1, nu2 = nu2, ncoarse = ncoarse,
            smoother = smoother, neumann_pin = neumann_pin, sigma = sigma)

    # Prolong coarse correction and add to fine solution.
    _mg_prolong_add!(u, hier.u[level + 1], N, N, Nc, Nc, kab)

    # Post-smooth.
    _mg_smooth!(u, f, h2, sigma, nu2, bc, N, N, kab, smoother, hier.scratch[level])

    neumann_pin && _project_zero_mean!(u, N, N, kab, hier.scratch[level])
    return nothing
end

# Remove the constant component (project onto the zero-mean subspace) to handle
# the singular all-Neumann operator. Uses a reduction (sum) which on GPU is a
# library call; here we use the generic `sum`, which dispatches to the device
# implementation for CUDA arrays (no scalar host indexing).
function _project_zero_mean!(a, Nx, Ny, kab, scratch)
    m = sum(a) / (Nx * Ny)   # device reduction; the host read is the only sync
    mg_shift_kernel!(kab)(a, m, Nx, Ny; ndrange = (Nx, Ny))
    return a
end

# -----------------------------------------------------------------------------
# Mixed-precision defect-correction cycle. The OUTER iterate `hier.u[1]` and the
# residual stay in Float64; the V-cycle that approximately solves the error
# equation `A e = r` runs entirely on the Float32 hierarchy `hier32` (half the
# bandwidth per smoothing sweep — the MG cost driver). One cycle =
#   r64 = f - A x          (F64 residual kernel)
#   f32 = convert(r64)     (conversion kernel)
#   e32 ~ A^{-1} f32       (one F32 V-cycle, zero start)
#   x  += convert(e32)     (converted-add kernel)
# Standard iterative-refinement convergence: as long as cond(A)*eps(Float32) < 1
# (true here: cond ~ (2N/pi)^2 ~ 1e5 at N=512, eps32 ~ 6e-8), the F64 residual
# keeps contracting at the MG rate down to F64 tolerances. The launch sequence
# is STATIC and reduction-free when `project=false` (fixed-cycles fast path).
# -----------------------------------------------------------------------------
function _mg_mixed_cycle!(hier::MGHierarchy, hier32::MGHierarchy, bcs, kab;
                          nu1::Int, nu2::Int, ncoarse::Int, smoother::Symbol,
                          sigma::Float64, project::Bool)
    N = hier.sizes[1]
    invh2 = 1.0 / hier.h[1]^2
    _mg_residual!(hier.r[1], hier.u[1], hier.f[1], invh2, sigma, bcs, N, N, kab)
    _mg_convert!(hier32.f[1], hier.r[1], N, N, kab)
    project && _project_zero_mean!(hier32.f[1], N, N, kab, hier32.scratch[1])
    _mg_zero!(hier32.u[1], N, N, kab)
    vcycle!(hier32, 1, bcs, kab; nu1 = nu1, nu2 = nu2, ncoarse = ncoarse,
            smoother = smoother, neumann_pin = project, sigma = sigma)
    _mg_add_converted!(hier.u[1], hier32.u[1], N, N, kab)
    project && _project_zero_mean!(hier.u[1], N, N, kab, hier.scratch[1])
    return nothing
end

# Relative residual norm (L2, scaled by h to be a grid-function norm). Uses the
# device-side residual kernel + a reduction; no host scalar indexing.
function _mg_relresid(hier::MGHierarchy, bc, kab; fnorm, sigma::Float64 = 0.0)
    N = hier.sizes[1]
    invh2 = 1.0 / hier.h[1]^2
    _mg_residual!(hier.r[1], hier.u[1], hier.f[1], invh2, sigma, bc, N, N, kab)
    rn = norm(hier.r[1])
    return rn / fnorm, rn
end

# =============================================================================
# Public driver
# =============================================================================

"""
    solve_poisson_mg(f, N; bc=:dirichlet, backend=CPUBackendTag(), tol=1e-10,
                     maxcycles=50, nu1=2, nu2=2, ncoarse=50, smoother=:rbgs,
                     verbose=false)
        -> (u, ncycles, resid_history)

Solve `-∇²u = f` on the unit square with a regular `N x N` cell-centred grid via
a matrix-free geometric multigrid V-cycle.

Arguments
  `f`  either a `Function (x,y) -> f(x,y)` sampled at cell centres, or an
       `N x N` array already holding the RHS at cell centres.
  `N`  grid size (square). Should be a power-of-two multiple of the coarsest grid
       (default coarsest >= 4) so the hierarchy halves cleanly.

Keywords
  `bc`        `:dirichlet` (homogeneous; inhomogeneous data must be folded into
              `f` by the caller, as in poisson.jl) or `:neumann` (all-Neumann,
              singular — handled by zero-mean projection / pin).
  `backend`   `CPUBackendTag()` (default) or `CUDABackendTag()`.
  `backend_ka` optional explicit KA backend (e.g. `CUDABackend()`); overrides the
              tag translation. Lets a GPU job pass its live backend directly.
  `atype`     device array type for allocation (default `Array{Float64}`; pass
              `CuArray{Float64}` for GPU).
  `tol`       relative-residual stopping tolerance.
  `maxcycles` cap on V-cycles.
  `nu1,nu2`   pre/post smoothing sweeps.
  `ncoarse`   coarsest-grid smoothing sweeps.
  `smoother`  `:rbgs` (default, red-black Gauss-Seidel) or `:jacobi` (weighted).
  `sigma`     Helmholtz shift: solves `(σI - ∇²)u = f` on the unit square
              (σ has units 1/length² in unit-square coordinates). Default 0
              recovers the plain Poisson operator. A positive σ makes the
              operator non-singular and strongly diagonally dominant — used by
              the cavity momentum solve (under-relaxed/pseudo-transient
              predictor). With σ>0 the Neumann case is NOT projected to zero
              mean (the shift removes the constant nullspace).
  `u0`        optional `N x N` initial guess (warm start); default zero start.
              Lets a SIMPLE loop reuse the previous solution to cut V-cycles.
  `fixed_cycles`  when > 0, run EXACTLY this many V-cycles with NO residual-norm
              computation at all (no `fnorm`, no per-cycle relative residual):
              zero host reductions/syncs and a STATIC kernel-launch sequence
              (CUDA-graph-capturable later). `tol`/`maxcycles` are ignored.
              For `:neumann` the RHS is assumed discretely zero-mean (true for
              a conservative face divergence) and the per-level zero-mean
              projections are SKIPPED: the V-cycle commutes with constant
              shifts, so the result only differs by an additive constant,
              which the caller must gauge away. Default 0 = tolerance-driven
              behavior (unchanged).
  `hier`      optional pre-built `MGHierarchy` (from `build_mg_hierarchy(N,
              atype)`) to reuse across repeated solves of the same size —
              avoids re-allocating the full level stack per call (a per-
              iteration allocation in SIMPLE loops). All level arrays are
              fully (re)initialized here, so reuse is value-identical.
  `mixed_precision`  when true, each V-cycle becomes a Float64 defect-correction
              step whose inner V-cycle runs entirely in Float32 (see
              `_mg_mixed_cycle!`): residual in F64, error solve in F32,
              correction added back in F64. Opt-in; default `false` keeps the
              all-Float64 path bit-identical. Composes with `fixed_cycles`
              (the mixed cycle is also a static, reduction-free launch
              sequence). The returned solution stays Float64.
  `hier_f32`  optional pre-built Float32 `MGHierarchy` (from
              `build_mg_hierarchy(N, _mg_eltype_variant(atype, Float32))`) for
              `mixed_precision=true`; reuse it across repeated solves exactly
              like `hier`. Ignored when `mixed_precision=false`.

Returns the solution `u` (N x N device array), the number of V-cycles performed,
and the relative-residual history (one entry per cycle, on the host; EMPTY when
`fixed_cycles > 0`).

Validation & performance receipts
  * MMS second order, V-cycle count flat vs N (the multigrid hallmark), parity
    vs the assembled CHOLMOD solver, and the singular Neumann variant:
    `test/analytical/poisson_mg_mms.jl`.
  * GPU (issue #8, Aqua A100): V-cycle counts stay flat at 10–13 from 16k to
    16M DOF (O(N) per cycle, O(1) cycles) and MG-GPU reaches ~43x MG-CPU at
    16M DOF, saturating the device (peak 99% util) where cuDSS idles —
    `benchmarks/krk/inc_ns/poisson_mg_gpu_bench.jl`,
    `benchmarks/results/poisson_mg_gpu_aqua_a100.md`. For moderate-N 2D steady
    problems the factorize-once cuDSS path still wins per-solve (different
    regimes; see the results note).
  * GPU↔CPU parity at machine epsilon (‖Δ‖∞ ~1e-14, same source via the KA
    backend swap).

Registered in `src/KrakeNN.jl` (exported by `using KrakeNN`); the file remains
standalone-include-able (its guard pulls in `linear_solve.jl` for the backend
tags when needed).
"""
function solve_poisson_mg(f, N::Integer;
                          bc::Symbol = :dirichlet,
                          backend::LinearSolveBackend = CPUBackendTag(),
                          backend_ka = _mg_ka_backend(backend),
                          atype::Type{<:AbstractArray} = Array{Float64},
                          tol::Real = 1e-10,
                          maxcycles::Integer = 50,
                          nu1::Integer = 2,
                          nu2::Integer = 2,
                          ncoarse::Integer = 50,
                          smoother::Symbol = :rbgs,
                          min_size::Integer = 4,
                          sigma::Real = 0.0,
                          u0 = nothing,
                          fixed_cycles::Integer = 0,
                          hier::Union{Nothing, MGHierarchy} = nothing,
                          mixed_precision::Bool = false,
                          hier_f32::Union{Nothing, MGHierarchy} = nothing,
                          verbose::Bool = false)
    N = Int(N)
    kab = backend_ka
    sig = Float64(sigma)
    nfixed = Int(fixed_cycles)

    bctag = bc === :dirichlet ? MG_BC_DIRICHLET :
            bc === :neumann   ? MG_BC_NEUMANN   :
            throw(ArgumentError("bc must be :dirichlet or :neumann"))
    bcs = (bctag, bctag, bctag, bctag)  # (west, east, south, north)
    neumann_pin = bc === :neumann

    if hier === nothing
        hier = build_mg_hierarchy(N, atype; min_size = min_size)
    else
        hier.sizes[1] == N || throw(ArgumentError("reused hierarchy is sized $(hier.sizes[1]), need $N"))
    end

    # Mixed precision: the F32 hierarchy hosts the inner error-equation V-cycle.
    # Allocated here only for standalone calls; repeated solvers (SIMPLE loops)
    # should pass a shared `hier_f32` exactly like `hier`.
    if mixed_precision
        if hier_f32 === nothing
            hier_f32 = build_mg_hierarchy(N, _mg_eltype_variant(atype, Float32);
                                          min_size = min_size)
        else
            hier_f32.sizes[1] == N ||
                throw(ArgumentError("reused F32 hierarchy is sized $(hier_f32.sizes[1]), need $N"))
            eltype(hier_f32.u[1]) === Float32 ||
                throw(ArgumentError("hier_f32 must hold Float32 arrays"))
        end
    end

    # Fill the finest RHS at cell centres. Sampling a Function is a host loop into
    # a host staging array, then copied to the device array (no device scalar
    # indexing). If `f` is already an array, just copy it.
    if f isa Function
        host = Matrix{Float64}(undef, N, N)
        h = 1.0 / N
        @inbounds for j in 1:N, i in 1:N
            x = (i - 0.5) * h
            y = (j - 0.5) * h
            host[i, j] = Float64(f(x, y))
        end
        copyto!(hier.f[1], host)
    else
        size(f) == (N, N) || throw(ArgumentError("f array must be N x N"))
        copyto!(hier.f[1], f)
    end

    # For the singular Neumann problem the RHS must be in the range of L
    # (zero mean). Project it. A Helmholtz shift (sig>0) makes the operator
    # non-singular, so zero-mean handling is only applied for the pure-Neumann,
    # sig=0 case (neumann_pin is only set for :neumann bc, which the cavity uses
    # with sig=0 for the pressure solve). In fixed-cycles mode the projection
    # is SKIPPED entirely (it is a host-sync reduction at every level of every
    # cycle): the V-cycle commutes with constant shifts and the caller's RHS is
    # discretely zero-mean, so the answer differs by a constant only (gauged by
    # the caller).
    do_project = neumann_pin && sig == 0.0 && nfixed == 0
    if do_project
        _project_zero_mean!(hier.f[1], N, N, kab, hier.scratch[1])
    end

    # Initial guess: zero, or a caller-supplied warm start (e.g. the previous
    # outer-iteration solution, which cuts V-cycles markedly in SIMPLE loops).
    if u0 === nothing
        _mg_zero!(hier.u[1], N, N, kab)
    else
        size(u0) == (N, N) || throw(ArgumentError("u0 must be N x N"))
        copyto!(hier.u[1], u0)
    end

    # ---- fixed-cycle mode: static launch sequence, ZERO host syncs ----------
    if nfixed > 0
        if mixed_precision
            # Each fixed "cycle" is one F64 defect-correction pass wrapping one
            # F32 V-cycle — still a static, reduction-free launch sequence
            # (CUDA-graph-capturable). Zero-mean projections stay skipped (the
            # defect correction commutes with constant shifts the same way the
            # plain Neumann V-cycle does; the caller gauges the constant).
            for _ in 1:nfixed
                _mg_mixed_cycle!(hier, hier_f32, bcs, kab;
                                 nu1 = Int(nu1), nu2 = Int(nu2),
                                 ncoarse = Int(ncoarse), smoother = smoother,
                                 sigma = sig, project = false)
            end
        else
            for _ in 1:nfixed
                vcycle!(hier, 1, bcs, kab; nu1 = Int(nu1), nu2 = Int(nu2),
                        ncoarse = Int(ncoarse), smoother = smoother,
                        neumann_pin = false, sigma = sig)
            end
        end
        return hier.u[1], nfixed, Float64[]
    end

    # ---- tolerance-driven mode (legacy, unchanged) --------------------------
    fnorm = norm(hier.f[1])
    fnorm == 0 && (fnorm = 1.0)

    resid_history = Float64[]
    relres, _ = _mg_relresid(hier, bcs, kab; fnorm = fnorm, sigma = sig)
    push!(resid_history, relres)
    verbose && @info "MG start" N relres

    ncycles = 0
    for cyc in 1:Int(maxcycles)
        if mixed_precision
            _mg_mixed_cycle!(hier, hier_f32, bcs, kab;
                             nu1 = Int(nu1), nu2 = Int(nu2),
                             ncoarse = Int(ncoarse), smoother = smoother,
                             sigma = sig, project = do_project)
        else
            vcycle!(hier, 1, bcs, kab; nu1 = Int(nu1), nu2 = Int(nu2),
                    ncoarse = Int(ncoarse), smoother = smoother,
                    neumann_pin = do_project, sigma = sig)
        end
        ncycles += 1
        relres, _ = _mg_relresid(hier, bcs, kab; fnorm = fnorm, sigma = sig)
        push!(resid_history, relres)
        verbose && @info "MG cycle" cyc relres
        relres <= tol && break
    end

    return hier.u[1], ncycles, resid_history
end

# =============================================================================
# Optional: MG-preconditioned Conjugate Gradient
# =============================================================================
# A single V-cycle is an excellent SPD preconditioner for the Dirichlet operator.
# This wrapper runs CG with one V-cycle as the preconditioner. All vector ops are
# elementwise kernels / device reductions (dot, axpy via broadcast), so it is
# GPU-ready. Provided as a bonus; the standalone V-cycle driver above is the
# primary path.

@kernel function mg_axpy_kernel!(y, a, @Const(x), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            y[i, j] += a * x[i, j]
        end
    end
end

@kernel function mg_xpay_kernel!(y, @Const(x), a, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            y[i, j] = x[i, j] + a * y[i, j]
        end
    end
end

"""
    solve_poisson_mgcg(f, N; bc=:dirichlet, ...) -> (u, niters, resid_history)

Conjugate gradient on `-∇²u = f`, preconditioned by ONE multigrid V-cycle per
iteration. Dirichlet only (SPD). Bonus path; same KA-generic structure.
"""
function solve_poisson_mgcg(f, N::Integer;
                            bc::Symbol = :dirichlet,
                            backend::LinearSolveBackend = CPUBackendTag(),
                            backend_ka = _mg_ka_backend(backend),
                            atype::Type{<:AbstractArray} = Array{Float64},
                            tol::Real = 1e-10,
                            maxiters::Integer = 100,
                            nu1::Integer = 2,
                            nu2::Integer = 2,
                            ncoarse::Integer = 50,
                            smoother::Symbol = :rbgs,
                            min_size::Integer = 4,
                            verbose::Bool = false)
    bc === :dirichlet || throw(ArgumentError("MG-CG wrapper is Dirichlet-only (SPD)"))
    N = Int(N)
    kab = backend_ka
    bcs = (MG_BC_DIRICHLET, MG_BC_DIRICHLET, MG_BC_DIRICHLET, MG_BC_DIRICHLET)
    invh2 = N^2  # 1/h^2 at finest

    # Hierarchy reused as the preconditioner workspace.
    hier = build_mg_hierarchy(N, atype; min_size = min_size)

    # CG vectors.
    x  = atype(undef, N, N); fill!(x, 0.0)
    b  = atype(undef, N, N)
    r  = atype(undef, N, N)
    z  = atype(undef, N, N)
    p  = atype(undef, N, N)
    Ap = atype(undef, N, N)

    if f isa Function
        host = Matrix{Float64}(undef, N, N)
        h = 1.0 / N
        @inbounds for j in 1:N, i in 1:N
            host[i, j] = Float64(f((i - 0.5) * h, (j - 0.5) * h))
        end
        copyto!(b, host)
    else
        copyto!(b, f)
    end

    # r = b - A x = b (x=0)
    copyto!(r, b)
    bnorm = norm(b); bnorm == 0 && (bnorm = 1.0)

    apply_M!(zout, rin) = begin
        # One V-cycle preconditioner: solve A z ≈ rin with z0 = 0.
        copyto!(hier.f[1], rin)
        _mg_zero!(hier.u[1], N, N, kab)
        vcycle!(hier, 1, bcs, kab; nu1 = Int(nu1), nu2 = Int(nu2),
                ncoarse = Int(ncoarse), smoother = smoother, neumann_pin = false,
                sigma = 0.0)
        copyto!(zout, hier.u[1])
    end

    apply_M!(z, r)
    copyto!(p, z)
    rz = _mg_dot(r, z)

    resid_history = Float64[norm(r) / bnorm]
    verbose && @info "MG-CG start" resid_history[end]

    niters = 0
    for it in 1:Int(maxiters)
        _mg_apply!(Ap, p, invh2, 0.0, bcs, N, N, kab)
        pAp = _mg_dot(p, Ap)
        alpha = rz / pAp
        mg_axpy_kernel!(kab)(x, alpha, p, N, N; ndrange = (N, N))
        KernelAbstractions.synchronize(kab)
        mg_axpy_kernel!(kab)(r, -alpha, Ap, N, N; ndrange = (N, N))
        KernelAbstractions.synchronize(kab)
        niters += 1
        relres = norm(r) / bnorm
        push!(resid_history, relres)
        verbose && @info "MG-CG it" it relres
        relres <= tol && break
        apply_M!(z, r)
        rz_new = _mg_dot(r, z)
        beta = rz_new / rz
        mg_xpay_kernel!(kab)(p, z, beta, N, N; ndrange = (N, N))
        KernelAbstractions.synchronize(kab)
        rz = rz_new
    end

    return x, niters, resid_history
end

# Device-friendly dot product (sum of elementwise product). For CUDA arrays this
# dispatches to the GPU reduction; no host scalar indexing.
_mg_dot(a, b) = sum(a .* b)
