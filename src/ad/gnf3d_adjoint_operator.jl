# --- Steady adjoint of the 3D D3Q19 GNF step (part 2/2): operator, solve, gradients ---
#
# Matrix-free products of the linearised 3D step (src/ad/gnf3d_adjoint.jl) as
# KernelAbstractions kernels (CPU now, same kernels on GPU), the gauge-bordered
# adjoint solved by the shared solvers of the 2D adjoint (`gnf_adjoint_solve`,
# `gnf_adjoint_kstep_solve`, `gnf_adjoint_fixed_point`: the operator is an
# `AbstractGNFStepOperator` with `gnf_jvp!` / `gnf_vjp!` methods), and the
# law-parameter gradient.

"""
    GNF3DStepOperator(lin::GNF3DLinearisation; backend=CPU(), T=Float64)

Device copy of the 20 × 20 blocks, solid mask, active mask (`active`, in T),
all-ones mask (`ones`, unprojected products) and gauge vector `chat` (Float64,
ĉ = c/‖c‖, c = 1 on fluid populations). `T` is the arithmetic type of the
products. An `AbstractGNFStepOperator`: `gnf_jvp!`, `gnf_vjp!` and the adjoint
solvers of the 2D step accept it.
"""
struct GNF3DStepOperator{T, TB<:AbstractArray{T,5}, TS<:AbstractArray{Bool,3},
                         TV<:AbstractVector{T}, TC<:AbstractVector{Float64}} <: AbstractGNFStepOperator
    nx::Int
    ny::Int
    nz::Int
    n::Int
    B::TB
    solid::TS
    active::TV
    ones::TV
    chat::TC
end

function GNF3DStepOperator(lin::GNF3DLinearisation; backend=KernelAbstractions.CPU(),
                           T::Type=Float64)
    nx, ny, nz = lin.nx, lin.ny, lin.nz
    N = nx * ny * nz; n = 20N
    B = KernelAbstractions.allocate(backend, T, nx, ny, nz, 20, 20)
    copyto!(B, T.(lin.B[:, :, :, :, 1:20]))
    solid = KernelAbstractions.allocate(backend, Bool, nx, ny, nz)
    copyto!(solid, lin.is_solid)
    act = KernelAbstractions.allocate(backend, T, n)
    copyto!(act, T.(lin.active))
    on = KernelAbstractions.allocate(backend, T, n)
    fill!(on, one(T))
    c = Float64.(lin.active); c[19N+1:20N] .= 0
    c ./= norm(c)
    chat = KernelAbstractions.allocate(backend, Float64, n)
    copyto!(chat, c)
    return GNF3DStepOperator(nx, ny, nz, n, B, solid, act, on, chat)
end

Base.eltype(::GNF3DStepOperator{T}) where {T} = T

# Projected values of the 20 components read by fluid node (i, j, k) in the
# JVP: the 19 arrival slots (stream sources) and its own τ⁺.
@inline function _gnf3d_gather_arrivals(x, pin, solid, ::Type{T}, i, j, k, nx, ny, nz) where {T}
    return ntuple(Val(20)) do s
        kk = s <= 19 ? _gnf3d_src(i, j, k, s, nx, ny, nz, solid) : _gnf3d_lin(i, j, k, 20, nx, ny, nz)
        @inbounds T(pin[kk]) * T(x[kk])
    end
end

# Projected values of the 20 stored components of node (i, j, k).
@inline function _gnf3d_gather_local(x, pin, ::Type{T}, i, j, k, nx, ny, nz) where {T}
    return ntuple(Val(20)) do r
        kk = _gnf3d_lin(i, j, k, r, nx, ny, nz)
        @inbounds T(pin[kk]) * T(x[kk])
    end
end

@inline function _gnf3d_block_mul(B, a::NTuple{20,T}, i, j, k) where {T}
    return ntuple(Val(20)) do r
        acc = zero(T)
        for s in 1:20
            @inbounds acc += B[i, j, k, r, s] * a[s]
        end
        acc
    end
end

@inline function _gnf3d_block_mul_t(B, λ::NTuple{20,T}, i, j, k) where {T}
    return ntuple(Val(20)) do s
        acc = zero(T)
        for r in 1:20
            @inbounds acc += B[i, j, k, r, s] * λ[r]
        end
        acc
    end
end

@kernel function gnf3d_jvp_kernel!(y, @Const(x), @Const(B), @Const(solid),
                                   @Const(pin), @Const(pout), nx, ny, nz)
    i, j, k = @index(Global, NTuple)
    if @inbounds solid[i, j, k]
        # solid components are constants of the step: zero rows of A
        for r in 1:20
            kk = _gnf3d_lin(i, j, k, r, nx, ny, nz)
            @inbounds y[kk] = zero(eltype(y))
        end
    else
        a = _gnf3d_gather_arrivals(x, pin, solid, eltype(B), i, j, k, nx, ny, nz)
        out = _gnf3d_block_mul(B, a, i, j, k)
        for r in 1:20
            kk = _gnf3d_lin(i, j, k, r, nx, ny, nz)
            @inbounds y[kk] = pout[kk] * out[r]
        end
    end
end

@kernel function gnf3d_vjp_kernel!(y, @Const(x), @Const(B), @Const(solid),
                                   @Const(pin), @Const(pout), nx, ny, nz)
    i, j, k = @index(Global, NTuple)
    if @inbounds solid[i, j, k]
        # never a stream source of a fluid node: zero columns of A
        for r in 1:20
            kk = _gnf3d_lin(i, j, k, r, nx, ny, nz)
            @inbounds y[kk] = zero(eltype(y))
        end
    else
        λ = _gnf3d_gather_local(x, pin, eltype(B), i, j, k, nx, ny, nz)
        z = _gnf3d_block_mul_t(B, λ, i, j, k)
        # transposed stream: arrival slot s goes back to its (unique) source component
        for s in 1:19
            kk = _gnf3d_src(i, j, k, s, nx, ny, nz, solid)
            @inbounds y[kk] = pout[kk] * z[s]
        end
        kk = _gnf3d_lin(i, j, k, 20, nx, ny, nz)
        @inbounds y[kk] = pout[kk] * z[20]
    end
end

"""
    gnf3d_jvp!(y, op, x; project=true) -> y

Tangent product y = P A P x of the linearised 3D step (A = ∂G/∂w). `x`, `y`:
distinct flat vectors of length 20 nx ny nz on the device of `op`.
`project=false` gives the unprojected A x.
"""
function gnf3d_jvp!(y, op::GNF3DStepOperator, x; project::Bool=true)
    m = project ? op.active : op.ones
    backend = KernelAbstractions.get_backend(op.B)
    gnf3d_jvp_kernel!(backend)(y, x, op.B, op.solid, m, m, op.nx, op.ny, op.nz;
                               ndrange=(op.nx, op.ny, op.nz))
    return y
end

"""
    gnf3d_vjp!(y, op, x; project=true) -> y

Adjoint product y = P Aᵀ P x (transposed blocks, then the transposed D3Q19
stream, halfway bounce-back included). `project=false` gives the unprojected Aᵀ x.
"""
function gnf3d_vjp!(y, op::GNF3DStepOperator, x; project::Bool=true)
    m = project ? op.active : op.ones
    backend = KernelAbstractions.get_backend(op.B)
    gnf3d_vjp_kernel!(backend)(y, x, op.B, op.solid, m, m, op.nx, op.ny, op.nz;
                               ndrange=(op.nx, op.ny, op.nz))
    return y
end

# The shared adjoint solvers call the generic product names.
gnf_jvp!(y, op::GNF3DStepOperator, x; project::Bool=true) = gnf3d_jvp!(y, op, x; project)
gnf_vjp!(y, op::GNF3DStepOperator, x; project::Bool=true) = gnf3d_vjp!(y, op, x; project)

"""
    gnf3d_adjoint_solve(op, b; solver=:kstep, kwargs...) -> (λ, η, info)

Steady adjoint of the 3D step for the right-hand side `b` = ∂J/∂w (device
vector, length n; projected on the active space by the solver), with the
solvers of the 2D adjoint: `solver = :kstep` (default, `gnf_adjoint_kstep_solve`),
`:gmres` (gauge-bordered GMRES, `gnf_adjoint_solve`) or `:fixed_point`
(`gnf_adjoint_fixed_point`); `kwargs` passed on (tol, restart, maxiter, K, λ0, ...).
"""
gnf3d_adjoint_solve(op::GNF3DStepOperator, b; solver::Symbol=:kstep, kwargs...) =
    _gnf_solve_adjoint(op, b, solver; kwargs...)

_gnf3d_split_gu(gu::NTuple{3,AbstractArray{<:Real,3}}) = gu
_gnf3d_split_gu(gu::NamedTuple{(:ux, :uy, :uz)}) = (gu.ux, gu.uy, gu.uz)
function _gnf3d_split_gu(gu::AbstractArray{<:Real,4})
    size(gu, 4) == 3 || throw(DimensionMismatch("g_u must be nx×ny×nz×3, got $(size(gu))"))
    g = Array(gu)
    return (g[:, :, :, 1], g[:, :, :, 2], g[:, :, :, 3])
end

"""
    gnf3d_velocity_rhs(lin, gu) -> b

Adjoint right-hand side b = (∂u/∂w)ᵀ g_u for a cotangent `gu` on the
post-collision lattice velocity (`gnf3d_post_moments`): a tuple
`(gux, guy, guz)`, a named tuple `(; ux, uy, uz)` or an nx × ny × nz × 3 array.
∂u_α/∂f_q = (c_αq − u_α)/ρ; host vector of length 20 N, nonzero on fluid
populations only (values of g_u at solid nodes are ignored).
"""
function gnf3d_velocity_rhs(lin::GNF3DLinearisation, gu)
    gx, gy, gz = map(a -> Float64.(Array(a)), _gnf3d_split_gu(gu))
    nx, ny, nz = lin.nx, lin.ny, lin.nz; N = nx * ny * nz
    all(size(g) == (nx, ny, nz) for g in (gx, gy, gz)) ||
        throw(DimensionMismatch("g_u components must be $(nx)×$(ny)×$(nz)"))
    b = zeros(20N)
    for c in 1:N
        lin.is_solid[c] && continue
        (gx[c] == 0 && gy[c] == 0 && gz[c] == 0) && continue
        ρ = lin.rho[c]; ux = lin.ux[c]; uy = lin.uy[c]; uz = lin.uz[c]
        for q in 1:19
            b[c + N * (q - 1)] = (gx[c] * (_G3_CX[q] - ux) + gy[c] * (_G3_CY[q] - uy) +
                                  gz[c] * (_G3_CZ[q] - uz)) / ρ
        end
    end
    return b
end

"""
    gnf3d_nu_sensitivity(lin, λ) -> s

Per-node cotangent of ν_lat, s_c = (∂(f', τ⁺')_c/∂ν)ᵀ λ_c (column 21 of the
blocks), host nx × ny × nz, zero at solid nodes; `λ` any backend.
"""
function gnf3d_nu_sensitivity(lin::GNF3DLinearisation, λ)
    λh = Float64.(Array(λ))
    nx, ny, nz = lin.nx, lin.ny, lin.nz
    s = zeros(nx, ny, nz)
    for k in 1:nz, j in 1:ny, i in 1:nx
        lin.is_solid[i, j, k] && continue
        acc = 0.0
        for r in 1:20
            acc += lin.B[i, j, k, r, 21] * λh[_gnf3d_lin(i, j, k, r, nx, ny, nz)]
        end
        s[i, j, k] = acc
    end
    return s
end

"""
    gnf3d_parameter_gradient(lin, s, θ, closure_of) -> grad

Σ_c s_c ∂ν_c/∂θ with ∂ν_c/∂θ the ForwardDiff gradient of
`gnf_viscosity(closure_of(θ), I1_c)` (any closure built from analytic parameters;
lattice units).
"""
function gnf3d_parameter_gradient(lin::GNF3DLinearisation, s, θ, closure_of)
    P = length(θ)
    θs = SVector{P,Float64}(Tuple(θ))
    grad = zeros(P)
    for c in eachindex(lin.is_solid)
        lin.is_solid[c] && continue
        I1c = lin.I1[c]
        gθ = ForwardDiff.gradient(t -> gnf_viscosity(closure_of(t), I1c), θs)
        grad .+= s[c] .* gθ
    end
    return grad
end

"""
    gnf3d_law_gradient(lin::GNF3DLinearisation, s, closure::LBMLawViscosity) -> Vector{Float64}

Parameter gradient Σ_c s_c ∂ν_lat,c/∂θ of a law closure in the 3D step, as
[`gnf_law_gradient`](@ref): `law_pullback!` at the regularised shear rate of
each fluid node, the state held fixed.
"""
function gnf3d_law_gradient(lin::GNF3DLinearisation, s::AbstractArray{<:Real,3},
                            closure::LBMLawViscosity)
    c = _gnf_host_closure(gnf_convert_closure(Float64, closure))
    size(s) == size(lin.is_solid) || throw(DimensionMismatch("s must match the lattice"))
    g = zeros(law_nparams(c.law))
    @inbounds for idx in eachindex(lin.is_solid)
        lin.is_solid[idx] && continue
        sc = s[idx]
        iszero(sc) && continue
        x = (gnf_shear_rate(_gnf_law_invariant(c, lin.I1[idx])),)
        law_pullback!(g, c.law, x, c.theta, c.nu_scale * sc)
    end
    return g
end

"""
    gnf3d_steady_law_gradient(f, tau, gu; is_solid, closure::LBMLawViscosity, F, Λ=3/16,
                              backend=CPU(), T=Float64, kwargs...) -> NamedTuple

Gradient with respect to every law parameter θ of an objective L of the
post-collision lattice velocity at the fixed point (f, τ⁺) of the 3D GNF step
on the reduced periodic lattice (`run_gnf3d_periodic_steady`), with
`gu = ∂L/∂u_lat` (`gnf3d_velocity_rhs` forms). Linearises
([`gnf3d_linearise`](@ref)), solves the steady adjoint
([`gnf3d_adjoint_solve`](@ref), `kwargs` passed on, default solver `:kstep`
as in 2D; `solver=:gmres` for the bordered GMRES), then contracts with
[`gnf3d_law_gradient`](@ref). Returns `(; grad, s, lin, op, stats, t_lin,
t_adjoint, t_grad)`, `stats = (; converged, iters, relres, λ, eta, info)`.
"""
function gnf3d_steady_law_gradient(f, tau, gu; is_solid, closure::LBMLawViscosity, F,
                                   Λ::Real=3 / 16, backend=KernelAbstractions.CPU(),
                                   T::Type=Float64, kwargs...)
    t0 = time()
    lin = gnf3d_linearise(f, tau, is_solid, closure, F; Λ)
    op = GNF3DStepOperator(lin; backend, T)
    t1 = time()
    bh = gnf3d_velocity_rhs(lin, gu)
    b = similar(op.chat, op.n); copyto!(b, bh)
    λ, η, info = gnf3d_adjoint_solve(op, b; kwargs...)
    s = gnf3d_nu_sensitivity(lin, λ)
    t2 = time()
    grad = gnf3d_law_gradient(lin, s, closure)
    t3 = time()
    stats = (; converged=info.converged, iters=info.iters, relres=info.relres, λ, eta=η, info)
    return (; grad, s, lin, op, stats, t_lin=t1 - t0, t_adjoint=t2 - t1, t_grad=t3 - t2)
end
