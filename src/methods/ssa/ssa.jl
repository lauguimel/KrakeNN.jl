# Shallow-shelf / shallow-stream approximation (SSA) of glacier ice flow:
# steady 2D depth-averaged velocity (u, v) on a cell-centred FV grid, Newton
# with the coloured dual Jacobian of the FV machinery, learned basal sliding
# law, and the gradient of a misfit w.r.t. the law weights by one discrete
# adjoint solve.
#
# Units (all quantities below are in these units, nothing is rescaled):
#   length m, time yr, velocity m/yr, stress kPa, strain rate 1/yr,
#   rate factor A in kPa^-n yr^-1 (2.4e-24 Pa^-3 s^-1 = 7.57e-8 kPa^-3 yr^-1),
#   rho_i g = 9.0 kPa/m  (SSA_RHOG), effective pressure N in kPa.
# The residual is per unit area (kPa), so its size does not depend on the grid.
#
#   d_x[2 eta H (2 u_x + v_y)] + d_y[eta H (u_y + v_x)] - tau_bx = rho g H s_x
#   d_y[2 eta H (2 v_y + u_x)] + d_x[eta H (u_y + v_x)] - tau_by = rho g H s_y
#   eta = B (eps_e^2 + eps_reg^2)^((1-n)/(2n)),  B = A^(-1/n)/2
#
# Unknown layout: interleaved (u, v) per cell, dof = 2 (cell - 1) + k,
# cell = i + (j - 1) Nx.  Boundaries: per side (1 E, 2 W, 3 N, 4 S) either
# Dirichlet velocity (ghost cell 2 b - u_P) or zero normal gradient (ghost
# = u_P).  Tangential derivatives on a boundary face come from the interior
# cell alone.  Everything is a plain KernelAbstractions kernel on isbits
# arguments (CPU and GPU).

import ForwardDiff
using KernelAbstractions

"""Ice weight rho_i g in kPa/m (917 kg/m^3, 9.81 m/s^2)."""
const SSA_RHOG = 917.0 * 9.81 / 1000.0

# ------------------------------------------------------------------ drag laws
"""
    SSAWeertman(C, m = 1/3, u_reg = 1e-2)

Weertman sliding `tau_b = C (|u|^2 + u_reg^2)^((m-1)/2) u` (C in kPa (m/yr)^-m).
"""
struct SSAWeertman{T}
    C::T
    m::T
    u_reg::T
end
SSAWeertman(C::Real, m::Real = 1 / 3, u_reg::Real = 1e-2) =
    SSAWeertman{Float64}(Float64(C), Float64(m), Float64(u_reg))

"""
    SSALearnedDrag(mlp::TinyMLP{2}, theta; F0 = 30.0, u_reg = 1e-2)

Learned sliding `tau_b = F u / sqrt(|u|^2 + u_reg^2)`, `F = F0 softplus(mlp(|u|_reg, N; theta))`
(kPa, positive), inputs `(|u|_reg, N)` normalised by the shift/scale of `mlp`.
`theta` is a flat vector (device array on the GPU).
"""
struct SSALearnedDrag{M<:TinyMLP,T,A}
    mlp::M
    F0::T
    u_reg::T
    theta::A
end
SSALearnedDrag(mlp::TinyMLP, theta::AbstractVector; F0::Real = 30.0, u_reg::Real = 1e-2) =
    (length(theta) == law_nparams(mlp) || throw(DimensionMismatch("theta has wrong length"));
     law_ninputs(mlp) == 2 || throw(ArgumentError("learned drag needs a 2-input TinyMLP"));
     SSALearnedDrag{typeof(mlp),Float64,typeof(theta)}(mlp, Float64(F0), Float64(u_reg), theta))
KernelAbstractions.Adapt.adapt_structure(to, d::SSALearnedDrag) =
    SSALearnedDrag(d.mlp, d.F0, d.u_reg, KernelAbstractions.Adapt.adapt(to, d.theta))

@inline function ssa_drag(d::SSAWeertman, ux, uy, N)
    f = d.C * (ux * ux + uy * uy + d.u_reg * d.u_reg)^((d.m - 1) / 2)
    return f * ux, f * uy
end
@inline function ssa_drag(d::SSALearnedDrag, ux, uy, N)
    sp = sqrt(ux * ux + uy * uy + d.u_reg * d.u_reg)
    z = law_value(d.mlp, (sp, N + zero(sp)), d.theta)
    F = d.F0 * _law_softplus(z)
    return F * ux / sp, F * uy / sp
end

# ------------------------------------------------------------------ problem
"""
    SSAProblem2D

Steady SSA problem on an `Nx × Ny` cell-centred grid (build with
[`ssa_problem`](@ref)). Fields: sizes and spacings, ice thickness `H`, driving
stress components `fx = rho g H s_x`, `fy` (kPa), effective pressure `Neff`,
Glen constants `Bn = A^(-1/n)/2`, `expo = (1-n)/(2n)`, `eps2 = eps_reg^2`, the
boundary velocities `bvel[component, l, side]`, the Dirichlet flags `bdir` and
the basal drag law.
"""
struct SSAProblem2D{M,B,D}
    Nx::Int
    Ny::Int
    dx::Float64
    dy::Float64
    H::M
    fx::M
    fy::M
    Neff::M
    Bn::Float64
    expo::Float64
    eps2::Float64
    bvel::B
    bdir::NTuple{4,Bool}
    drag::D
    n_glen::Float64
end

KernelAbstractions.Adapt.adapt_structure(to, p::SSAProblem2D) =
    SSAProblem2D(p.Nx, p.Ny, p.dx, p.dy, map(a -> KernelAbstractions.Adapt.adapt(to, a), (p.H, p.fx, p.fy, p.Neff))...,
                 p.Bn, p.expo, p.eps2, KernelAbstractions.Adapt.adapt(to, p.bvel), p.bdir,
                 KernelAbstractions.Adapt.adapt(to, p.drag), p.n_glen)

"""
    ssa_adapt(to, prob) -> SSAProblem2D

Move all arrays of `prob` with `to` (`Array`, `CuArray`).
"""
ssa_adapt(to, p::SSAProblem2D) =
    SSAProblem2D(p.Nx, p.Ny, p.dx, p.dy, to(p.H), to(p.fx), to(p.fy), to(p.Neff),
                 p.Bn, p.expo, p.eps2, to(p.bvel), p.bdir, _ssa_adapt_drag(to, p.drag),
                 p.n_glen)
_ssa_adapt_drag(to, d::SSAWeertman) = d
_ssa_adapt_drag(to, d::SSALearnedDrag) = SSALearnedDrag(d.mlp, d.F0, d.u_reg, to(d.theta))

"""Same problem with a new drag law (e.g. other weights) or regularisation."""
ssa_with(p::SSAProblem2D; drag = p.drag, eps_reg = sqrt(p.eps2)) =
    SSAProblem2D(p.Nx, p.Ny, p.dx, p.dy, p.H, p.fx, p.fy, p.Neff, p.Bn, p.expo,
                 Float64(eps_reg)^2, p.bvel, p.bdir, drag, p.n_glen)

ssa_nunknowns(p::SSAProblem2D) = 2 * p.Nx * p.Ny

# surface-slope gradient: central differences inside, one-sided at the edges
function _ssa_gradient(s::AbstractMatrix, dx, dy)
    Nx, Ny = size(s)
    sx = zeros(Nx, Ny); sy = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        i0 = max(i - 1, 1); i1 = min(i + 1, Nx)
        j0 = max(j - 1, 1); j1 = min(j + 1, Ny)
        sx[i, j] = (s[i1, j] - s[i0, j]) / ((i1 - i0) * dx)
        sy[i, j] = (s[i, j1] - s[i, j0]) / ((j1 - j0) * dy)
    end
    return sx, sy
end

"""
    ssa_problem(Nx, Ny; Lx, Ly, H, s, A, n = 3, drag, N = nothing, eps_reg = 1e-5,
                boundary_velocity = nothing, dirichlet = (true,true,true,true))

`H`, `s` are numbers, `(x, y) -> value` functions (cell centres, origin at the
lower-left corner) or `Nx × Ny` matrices (m). `A` is the rate factor in
kPa^-n yr^-1. The driving stress is `rho g H ∇s` with the surface gradient by
central differences. `N` (effective pressure, kPa) defaults to `rho g H`.
`boundary_velocity(x, y) -> (u, v)` is evaluated at the wall midpoints of the
Dirichlet sides (default zero). Sides are ordered E, W, N, S.
"""
function ssa_problem(Nx::Int, Ny::Int; Lx::Real, Ly::Real, H, s, A::Real, n::Real = 3,
                     drag, N = nothing, eps_reg::Real = 1e-5, boundary_velocity = nothing,
                     dirichlet::NTuple{4,Bool} = (true, true, true, true))
    dx = Float64(Lx) / Nx; dy = Float64(Ly) / Ny
    cellfield(f::Number) = fill(Float64(f), Nx, Ny)
    cellfield(f::AbstractMatrix) = Float64.(f)
    cellfield(f::Function) = [Float64(f((i - 0.5) * dx, (j - 0.5) * dy)) for i in 1:Nx, j in 1:Ny]
    Hm = cellfield(H); sm = cellfield(s)
    all(>(0), Hm) || throw(ArgumentError("ice thickness must be positive"))
    sx, sy = _ssa_gradient(sm, dx, dy)
    Nm = N === nothing ? SSA_RHOG .* Hm : cellfield(N)
    bvel = zeros(2, max(Nx, Ny), 4)
    if boundary_velocity !== nothing
        for l in 1:max(Nx, Ny)
            pts = ((Lx, (l - 0.5) * dy), (0.0, (l - 0.5) * dy), ((l - 0.5) * dx, Ly), ((l - 0.5) * dx, 0.0))
            for side in 1:4
                (side <= 2 ? l <= Ny : l <= Nx) || continue
                bvel[1, l, side], bvel[2, l, side] = boundary_velocity(pts[side]...)
            end
        end
    end
    nn = Float64(n)
    return SSAProblem2D(Nx, Ny, dx, dy, Hm, SSA_RHOG .* Hm .* sx, SSA_RHOG .* Hm .* sy, Nm,
                        0.5 * Float64(A)^(-1 / nn), (1 - nn) / (2nn), Float64(eps_reg)^2,
                        bvel, dirichlet, drag, nn)
end

# ------------------------------------------------------------------ residual
# velocity of cell (i, j), or of the ghost cell across a boundary face
@inline function _ssa_cell(U, p, i, j)
    Nx = p.Nx; Ny = p.Ny
    if i > Nx
        side = 1; l = j; ii = Nx; jj = j
    elseif i < 1
        side = 2; l = j; ii = 1; jj = j
    elseif j > Ny
        side = 3; l = i; ii = i; jj = Ny
    elseif j < 1
        side = 4; l = i; ii = i; jj = 1
    else
        c = 2 * ((i - 1) + (j - 1) * Nx)
        @inbounds return U[c + 1], U[c + 2]
    end
    c = 2 * ((ii - 1) + (jj - 1) * Nx)
    @inbounds up = U[c + 1]; @inbounds vp = U[c + 2]
    if p.bdir[side]
        @inbounds return 2 * p.bvel[1, l, side] - up, 2 * p.bvel[2, l, side] - vp
    end
    return up, vp
end

# central derivatives at a real cell: (u_x, v_x) and (u_y, v_y)
@inline function _ssa_ddx(U, p, i, j)
    ue, ve = _ssa_cell(U, p, i + 1, j); uw, vw = _ssa_cell(U, p, i - 1, j)
    return (ue - uw) / (2 * p.dx), (ve - vw) / (2 * p.dx)
end
@inline function _ssa_ddy(U, p, i, j)
    un, vn = _ssa_cell(U, p, i, j + 1); us, vs = _ssa_cell(U, p, i, j - 1)
    return (un - us) / (2 * p.dy), (vn - vs) / (2 * p.dy)
end

@inline function _ssa_stress(p, ux, vx, uy, vy, Hf)
    e2 = ux * ux + vy * vy + ux * vy + (uy + vx) * (uy + vx) / 4 + p.eps2
    eta = p.Bn * e2^p.expo
    return 2 * eta * Hf * (2 * ux + vy), eta * Hf * (uy + vx), 2 * eta * Hf * (2 * vy + ux)
end

# face between (i, j) and (i + 1, j): (F_xx, F_xy, F_yy)
@inline function _ssa_xface(U, p, i, j)
    uA, vA = _ssa_cell(U, p, i, j); uB, vB = _ssa_cell(U, p, i + 1, j)
    ux = (uB - uA) / p.dx; vx = (vB - vA) / p.dx
    if i < 1
        uy, vy = _ssa_ddy(U, p, i + 1, j); Hf = @inbounds p.H[i + 1, j]
    elseif i + 1 > p.Nx
        uy, vy = _ssa_ddy(U, p, i, j); Hf = @inbounds p.H[i, j]
    else
        uyA, vyA = _ssa_ddy(U, p, i, j); uyB, vyB = _ssa_ddy(U, p, i + 1, j)
        uy = (uyA + uyB) / 2; vy = (vyA + vyB) / 2
        Hf = @inbounds (p.H[i, j] + p.H[i + 1, j]) / 2
    end
    return _ssa_stress(p, ux, vx, uy, vy, Hf)
end
# face between (i, j) and (i, j + 1)
@inline function _ssa_yface(U, p, i, j)
    uA, vA = _ssa_cell(U, p, i, j); uB, vB = _ssa_cell(U, p, i, j + 1)
    uy = (uB - uA) / p.dy; vy = (vB - vA) / p.dy
    if j < 1
        ux, vx = _ssa_ddx(U, p, i, j + 1); Hf = @inbounds p.H[i, j + 1]
    elseif j + 1 > p.Ny
        ux, vx = _ssa_ddx(U, p, i, j); Hf = @inbounds p.H[i, j]
    else
        uxA, vxA = _ssa_ddx(U, p, i, j); uxB, vxB = _ssa_ddx(U, p, i, j + 1)
        ux = (uxA + uxB) / 2; vx = (vxA + vxB) / 2
        Hf = @inbounds (p.H[i, j] + p.H[i, j + 1]) / 2
    end
    return _ssa_stress(p, ux, vx, uy, vy, Hf)
end

@inline function ssa_cell_residual(U, p, i, j)
    Fxx_e, Fxy_e, _ = _ssa_xface(U, p, i, j)
    Fxx_w, Fxy_w, _ = _ssa_xface(U, p, i - 1, j)
    _, Fxy_n, Fyy_n = _ssa_yface(U, p, i, j)
    _, Fxy_s, Fyy_s = _ssa_yface(U, p, i, j - 1)
    u, v = _ssa_cell(U, p, i, j)
    tbx, tby = ssa_drag(p.drag, u, v, @inbounds p.Neff[i, j])
    ru = (Fxx_e - Fxx_w) / p.dx + (Fxy_n - Fxy_s) / p.dy - tbx - @inbounds(p.fx[i, j])
    rv = (Fxy_e - Fxy_w) / p.dx + (Fyy_n - Fyy_s) / p.dy - tby - @inbounds(p.fy[i, j])
    return ru, rv
end

@kernel function _ssa_residual_kernel!(R, @Const(U), p)
    i, j = @index(Global, NTuple)
    ru, rv = ssa_cell_residual(U, p, i, j)
    c = 2 * ((i - 1) + (j - 1) * p.Nx)
    @inbounds R[c + 1] = ru
    @inbounds R[c + 2] = rv
end

"""
    ssa_residual!(R, U, prob; sync = true) -> R

SSA residual (kPa, per unit area) for the interleaved state `U`, on the device of
`U` (any element type, e.g. ForwardDiff duals). `prob` must be on that device.
"""
function ssa_residual!(R, U, p::SSAProblem2D; sync::Bool = true)
    length(U) == ssa_nunknowns(p) || throw(DimensionMismatch("wrong state length"))
    backend = KernelAbstractions.get_backend(U)
    _ssa_residual_kernel!(backend, 64)(R, U, p; ndrange = (p.Nx, p.Ny))
    sync && KernelAbstractions.synchronize(backend)
    return R
end
ssa_residual(p::SSAProblem2D, U::AbstractVector) = ssa_residual!(similar(U), U, p)

# ------------------------------------------------------------------ Jacobian
"""
    ssa_jacobian_pattern(prob) -> SparseMatrixCSC{Bool,Int}

3 × 3 cell block, both components (≤ 18 entries per row), clipped to the grid.
"""
function ssa_jacobian_pattern(p::SSAProblem2D)
    Nx, Ny = p.Nx, p.Ny
    I = Int[]; J = Int[]
    for j in 1:Ny, i in 1:Nx, k in 1:2
        row = 2 * ((i - 1) + (j - 1) * Nx) + k
        for jj in max(j - 1, 1):min(j + 1, Ny), ii in max(i - 1, 1):min(i + 1, Nx), kk in 1:2
            push!(I, row); push!(J, 2 * ((ii - 1) + (jj - 1) * Nx) + kk)
        end
    end
    n = ssa_nunknowns(p)
    return sparse(I, J, trues(length(I)), n, n, |)
end

function ssa_assemble_jacobian!(nzval, asm::FVJacobianAssembler{N}, U, p::SSAProblem2D;
                                sync::Bool = true) where {N}
    backend = KernelAbstractions.get_backend(U)
    seed! = _fv_seed_kernel!(backend, 256)
    dec! = _fv_decompress_kernel!(backend, 256)
    for offset in 0:N:(asm.ncolors - 1)
        seed!(asm.Ud, U, asm.colors, offset, Val(N); ndrange = asm.n)
        ssa_residual!(asm.Rd, asm.Ud, p; sync = false)
        dec!(nzval, asm.nzrow, asm.nzcol, asm.colors, asm.Rd, offset, Val(N), asm.n; ndrange = asm.nnz)
    end
    sync && KernelAbstractions.synchronize(backend)
    return nzval
end

# ------------------------------------------------------------------ Newton
"""
    SSANewtonSolver

Newton solver state (build with [`ssa_newton_solver`](@ref)): device problem,
coloured Jacobian assembler, matrix `J`, linear-solve cache and work vectors.
`prob` may be replaced (e.g. by [`ssa_with`](@ref)) as long as the grid is the same.
"""
mutable struct SSANewtonSolver{P,A,M,V,B}
    prob::P
    asm::A
    J::M
    nzval::V
    linsolve::B
    cache::Union{Nothing,LinearSolveCache}
    R::V
    dU::V
    Ut::V
    n_factorize::Int
    n_refactorize::Int
end

"""
    ssa_newton_solver(prob; to = Array, linsolve = CPUBackendTag(), chunk = 4)

`prob` is the HOST problem (`to = CuArray` with `linsolve = CUDABackendTag()` for
cuDSS, which needs `using CUDA, CUDSS`).
"""
function ssa_newton_solver(prob::SSAProblem2D; to = Array,
                           linsolve::LinearSolveBackend = CPUBackendTag(), chunk::Int = 4)
    S = ssa_jacobian_pattern(prob)
    colors, ncol = fv_jacobian_coloring(S)
    n = size(S, 1)
    if linsolve isa CPUBackendTag
        J = SparseMatrixCSC{Float64,Int}(n, n, copy(S.colptr), copy(rowvals(S)), zeros(Float64, nnz(S)))
        nzval = nonzeros(J); order = :csc
    else
        _, _, rowptr, colval = _fv_storage_order(S, :csr)
        J, nzval = _fv_device_csr(linsolve, rowptr, colval, n)
        order = :csr
    end
    asm = fv_jacobian_assembler(S, colors, ncol; to, order, chunk)
    vec0() = to(zeros(Float64, n))
    return SSANewtonSolver(ssa_adapt(to, prob), asm, J, nzval, linsolve, nothing,
                           vec0(), vec0(), vec0(), 0, 0)
end

function _ssa_factor!(s::SSANewtonSolver)
    if s.cache === nothing
        s.cache = lin_factorize(s.J; backend = s.linsolve, spd = false)
        s.n_factorize += 1
    else
        lin_refactorize!(s.cache)
        s.n_refactorize += 1
    end
    return s.cache
end

"""
    ssa_jacobian!(solver, U) -> solver.J

Assemble `∂R/∂U` at `U` (values only).
"""
ssa_jacobian!(s::SSANewtonSolver, U) = (ssa_assemble_jacobian!(s.nzval, s.asm, U, s.prob); s.J)

"""
    ssa_initial_state(prob) -> Vector{Float64}

Local-balance guess: for Weertman drag the speed solving `tau_b(|u|) = |f|`
along the driving stress; for any other drag, zero velocity.
"""
function ssa_initial_state(p::SSAProblem2D)
    U = zeros(ssa_nunknowns(p))
    d = p.drag
    d isa SSAWeertman || return U
    for j in 1:p.Ny, i in 1:p.Nx
        f = hypot(p.fx[i, j], p.fy[i, j])
        f > 0 || continue
        sp = (f / d.C)^(1 / d.m)               # |tau_b| = C |u|^m = |f|; flow opposes f
        c = 2 * ((i - 1) + (j - 1) * p.Nx)
        U[c + 1] = -sp * p.fx[i, j] / f
        U[c + 2] = -sp * p.fy[i, j] / f
    end
    return U
end

"""
    ssa_newton!(solver, U; atol = 1e-9, rtol = 1e-12, maxit = 40, linesearch = true,
                armijo = 1e-4, alpha_min = 2^-10, verbose = false) -> NamedTuple

Newton with Armijo backtracking on `‖R‖₂`, `U` on the solver's device, updated
in place. Stops when `‖R‖∞ ≤ atol` (kPa) or `‖R‖₂ ≤ rtol ‖R₀‖₂`. Returns
`(converged, iterations, history, inf, alpha)`.
"""
function ssa_newton!(s::SSANewtonSolver, U; atol::Real = 1e-9, rtol::Real = 1e-12,
                     maxit::Int = 40, linesearch::Bool = true, armijo::Real = 1e-4,
                     alpha_min::Real = 2.0^-10, verbose::Bool = false)
    R = s.R; dU = s.dU; Ut = s.Ut
    ssa_residual!(R, U, s.prob)
    r = norm(R); ri = maximum(abs, R); r0 = r
    hist = [r]; inf = [ri]; alphas = Float64[]
    converged = ri <= atol || r <= rtol * r0
    it = 0; nfail = 0
    while !converged && it < maxit
        it += 1
        ssa_jacobian!(s, U)
        _ssa_factor!(s)
        lin_solve!(dU, s.cache, R)
        alpha = 1.0
        local rt, rit
        while true
            Ut .= U .- alpha .* dU
            ssa_residual!(R, Ut, s.prob)
            rt = norm(R); rit = maximum(abs, R)
            (!linesearch || rt <= (1 - armijo * alpha) * r || alpha <= alpha_min) && break
            alpha /= 2
        end
        nfail = (linesearch && rt > (1 - armijo * alpha) * r) ? nfail + 1 : 0
        U .= Ut
        r = rt
        push!(hist, rt); push!(inf, rit); push!(alphas, alpha)
        verbose && @info "ssa_newton" it res = rt inf = rit alpha
        converged = rit <= atol || rt <= rtol * r0
        nfail >= 2 && break
    end
    return (; converged, iterations = it, history = hist, inf, alpha = alphas)
end

"""
    ssa_newton_continuation!(solver, U, eps_regs; newton...) -> Vector{NamedTuple}

Solve for the decreasing list of regularisations `eps_regs` (1/yr), each starting
from the previous solution; the solver's problem ends with the last value.
"""
function ssa_newton_continuation!(s::SSANewtonSolver, U, eps_regs; newton...)
    out = NamedTuple[]
    for e in eps_regs
        s.prob = ssa_with(s.prob; eps_reg = e)
        push!(out, ssa_newton!(s, U; newton...))
    end
    return out
end

"""
    ssa_solve(prob; U0 = nothing, to = Array, linsolve = CPUBackendTag(), newton...)

Build a solver, start from `U0` (default [`ssa_initial_state`](@ref)) and run Newton.
Returns `(u, v, U, converged, iterations, stats, solver)` with host `u`, `v`.
"""
function ssa_solve(prob::SSAProblem2D; U0 = nothing, to = Array,
                   linsolve::LinearSolveBackend = CPUBackendTag(), chunk::Int = 4, newton...)
    s = ssa_newton_solver(prob; to, linsolve, chunk)
    U = to(U0 === nothing ? ssa_initial_state(prob) : Vector{Float64}(U0))
    stats = ssa_newton!(s, U; newton...)
    u, v = ssa_fields(prob, U)
    return (; u, v, U, converged = stats.converged, iterations = stats.iterations, stats, solver = s)
end

"""Host matrices `(u, v)` of an interleaved state."""
function ssa_fields(p::SSAProblem2D, U)
    h = Array(U)
    return reshape(h[1:2:end], p.Nx, p.Ny), reshape(h[2:2:end], p.Nx, p.Ny)
end

# ------------------------------------------------------------------ adjoint
# dL/dtheta = - lambda' R_theta, J' lambda = dL/dU. Only the learned drag depends on
# theta: R_k = ... - F u_k/|u|_reg, F = F0 softplus(z), z = mlp(|u|_reg, N; theta).
@kernel function _ssa_theta_capture_kernel!(inputs, coef, @Const(U), @Const(lam), p)
    i, j = @index(Global, NTuple)
    c = i + (j - 1) * p.Nx
    d = p.drag
    @inbounds u = U[2c - 1]; @inbounds v = U[2c]
    sp = sqrt(u * u + v * v + d.u_reg * d.u_reg)
    N = @inbounds p.Neff[i, j]
    z = law_value(d.mlp, (sp, N), d.theta)
    @inbounds coef[c] = (lam[2c - 1] * u + lam[2c] * v) / sp * d.F0 * _law_logistic(z)
    @inbounds inputs[1, c] = sp
    @inbounds inputs[2, c] = N
end

@kernel function _ssa_theta_pullback_kernel!(partials, @Const(inputs), @Const(coef), mlp, @Const(theta), nc, per)
    tile = @index(Global)
    g = _FVLawPartialColumn{eltype(partials),typeof(partials)}(partials, tile)
    @inbounds for q in 1:law_nparams(mlp)
        g[q] = 0.0
    end
    @inbounds for c in ((tile - 1) * per + 1):min(tile * per, nc)
        law_pullback!(g, mlp, (inputs[1, c], inputs[2, c]), theta, coef[c])
    end
end

@kernel function _ssa_theta_reduce_kernel!(gradient, @Const(partials))
    q = @index(Global)
    acc = 0.0
    @inbounds for tile in axes(partials, 1)
        acc += partials[tile, q]
    end
    @inbounds gradient[q] = acc
end

"""
    ssa_theta_gradient(solver, U, g) -> Vector{Float64}

Gradient w.r.t. the weights of the learned drag of `dL/dθ` where `g = ∂L/∂U`
(device vector, same layout as `U`), by ONE transposed solve `Jᵀ λ = g` with the
Newton factors re-assembled at `U` (no continuation shift), then a per-cell
contraction `-λᵀ ∂R/∂θ` through [`law_pullback!`](@ref).
"""
function ssa_theta_gradient(s::SSANewtonSolver, U, g)
    p = s.prob
    p.drag isa SSALearnedDrag || throw(ArgumentError("the problem has no learned drag"))
    ssa_jacobian!(s, U)
    _ssa_factor!(s)
    lam = similar(U)
    lin_solve!(lam, s.cache, g; transpose = true)
    nc = p.Nx * p.Ny; per = 64; np = length(p.drag.theta)
    inputs = similar(U, 2, nc); coef = similar(U, nc)
    partials = similar(U, cld(nc, per), np); grad = similar(U, np)
    backend = KernelAbstractions.get_backend(U)
    _ssa_theta_capture_kernel!(backend, 64)(inputs, coef, U, lam, p; ndrange = (p.Nx, p.Ny))
    _ssa_theta_pullback_kernel!(backend, 64)(partials, inputs, coef, p.drag.mlp, p.drag.theta, nc, per;
                                             ndrange = size(partials, 1))
    _ssa_theta_reduce_kernel!(backend, 64)(grad, partials; ndrange = np)
    KernelAbstractions.synchronize(backend)
    return Array(grad)
end

"""
    ssa_misfit(U, Uobs) -> (L, g)

`L = ½ Σ (U - Uobs)²` and `g = ∂L/∂U = U - Uobs`.
"""
function ssa_misfit(U, Uobs)
    g = U .- Uobs
    return sum(abs2, g) / 2, g
end

# replace the learned-drag weights of a (host or device) problem
"""
    ssa_set_theta!(solver, theta)

Overwrite the weights of the learned drag held by the solver's device problem
(same pattern and symbolic analysis kept).
"""
function ssa_set_theta!(s::SSANewtonSolver, theta)
    copyto!(s.prob.drag.theta, theta)
    return s
end
