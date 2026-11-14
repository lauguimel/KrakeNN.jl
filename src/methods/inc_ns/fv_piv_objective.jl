# Composite PIV objective of the 3D steady FV adjoint (plan "solveur FV
# stationnaire 3D", step 4, increment I2): misfit of plane-PIV samples
# (src/platform/piv_observation.jl) plus weighted state functionals.
#
# Hot path (fv_steady_gradient with an FVPIVObjective): per plane and measured
# component, one kernel contracts the state along the plane normal with the
# light-sheet weights (the gather of u, v or w straight from U, no field
# copies), two matrix products give the samples, two more the cotangent, and
# one kernel scatters it into ∂𝒥/∂U. Linear terms are dense weight vectors
# (one 1 × 1 product and one axpy). No dual numbers, no host copy of the state, and no
# device allocation once the objective is built. Nonlinear terms (e.g. a wall
# force) fall back to host duals on their support.

struct _FVPIVPlaneData{Ax,NC,OP,M}
    op::OP                      # PIVOperator on the solver backend
    comps::NTuple{NC,Int}       # velocity component of each sample slot
    data::NTuple{NC,M}
    mask::NTuple{NC,M}
    C::M                        # normal contraction (Na × Nb)
    T1::M                       # Wa C (nsa × Nb)
    S::M                        # samples (nsa × nsb)
    r::M                        # masked residual
    gs::M                       # sample cotangent m .* r
    T2::M                       # Waᵀ gs (Na × nsb)
    G::M                        # contraction cotangent (Na × Nb)
    acc::M                      # 1 × 1 device result of rᵀ gs (no reduction temporaries)
    hacc::Vector{Float64}       # host landing slot of acc
end

struct _FVPIVTerm{O,W,B,A}
    obj::O
    weight::Float64
    target::Float64             # NaN: linear term weight · 𝒥_t
    w::W                        # dense device weights of a linear functional, or nothing
    buf::B                      # device buffer for a host-evaluated state gradient, or nothing
    acc::A                      # 1 × 1 device result of wᵀ U (linear terms), or nothing
    hacc::Vector{Float64}
end

@inline _fv_term_value(t::_FVPIVTerm, x) = isnan(t.target) ? t.weight * x : t.weight * (x - t.target)^2 / 2
@inline _fv_term_slope(t::_FVPIVTerm, x) = isnan(t.target) ? t.weight : t.weight * (x - t.target)

"""
    FVPIVObjective(prob::FVSteadyProblem3D, ops, data; mask = nothing, sigma_u = 1.0, terms = ())

Composite objective of the 3D steady FV state for calibration against plane
PIV:

    𝒥 = Σ_p L_p / σ_u²  +  Σ_t c_t(𝒥_t)

- `ops`: one [`PIVOperator`](@ref) per plane, built with
  `piv_operator(plane, template; h = (h, h, h))` on an `Nx × Ny × Nz` Float64
  template of the solver backend (`Array` or `CuArray`); FV cell centres are
  the operator's sample grid.
- `data[p]`: tuple of sample matrices of plane `p` (in-plane components in
  ascending axis order, then the out-of-plane one for a stereo plane), as
  returned by [`piv_apply`](@ref). `L_p` is the [`piv_misfit`](@ref) of plane
  `p`: `½ Σ m (sample − data)²`.
- `mask`: `nothing`, or one entry per plane (`nothing`, one matrix for all
  components, or a tuple of matrices); `m = 0` drops a vector (its data may be
  `NaN`).
- `sigma_u`: velocity noise scale of the PIV data.
- `terms`: tuple of `(obj, weight)` (adds `weight · 𝒥_t`) or
  `(obj, weight, target)` (adds `weight (𝒥_t − target)² / 2`), with `obj` an
  [`FVObjective`](@ref) of the problem: [`fv_flow_rate`](@ref),
  [`fv_flow_force_objective`](@ref) (the solved force `Fx` on its index) or
  any [`FVLinearFunctional`](@ref) — evaluated on the device as dense weight
  vectors — or a nonlinear objective such as [`FVWallForce`](@ref), evaluated
  with host duals (state copied to the host once per gradient) and whose
  explicit parameter dependence enters `∂𝒥/∂θ`.

[`fv_objective`](@ref)`(obj, U, prob)` evaluates the same value on the host
with the [`piv_misfit`](@ref) formula on [`piv_apply`](@ref) samples (generic
element type; the reference of the tests); [`fv_steady_gradient`](@ref) uses the allocation-free device path.
"""
struct FVPIVObjective{G,P,H,T,V} <: FVObjective
    grid::G
    n::Int
    planes::P                   # Tuple of _FVPIVPlaneData
    host::H                     # Tuple of (host operator, host data, host mask)
    sigma_u::Float64
    terms::T                    # Tuple of _FVPIVTerm
    gU::V                       # ∂𝒥/∂U on the solver backend
end

_fv_piv_host_op(op::PIVOperator{Ax,S}) where {Ax,S} =
    PIVOperator{Ax,S,Float64,Matrix{Float64},Vector{Float64}}(Array(op.Wa), Array(op.Wb), Array(op.wn))

function _fv_piv_plane(op::PIVOperator{Ax,S,T}, data, mask, g) where {Ax,S,T}
    T === Float64 || throw(ArgumentError("FVPIVObjective needs Float64 PIV operators"))
    dims = (g.Nx, g.Ny, g.Nz)
    ia, ib = Ax == 1 ? (2, 3) : Ax == 2 ? (1, 3) : (1, 2)
    (size(op.Wa, 2), size(op.Wb, 2), length(op.wn)) == (dims[ia], dims[ib], dims[Ax]) ||
        throw(DimensionMismatch("PIV operator built for another grid than $(dims)"))
    comps = _components(op)[1:(S ? 3 : 2)]
    NC = length(comps)
    data isa Tuple && length(data) == NC ||
        throw(ArgumentError("plane data must be a tuple of $NC sample matrices"))
    nsa, nsb = size(op.Wa, 1), size(op.Wb, 1)
    all(d -> size(d) == (nsa, nsb), data) ||
        throw(DimensionMismatch("sample matrices must be $nsa × $nsb"))
    masks = mask === nothing ? ntuple(_ -> ones(nsa, nsb), NC) :
            mask isa Tuple ? mask : ntuple(_ -> mask, NC)
    length(masks) == NC && all(m -> size(m) == (nsa, nsb), masks) ||
        throw(DimensionMismatch("mask must be $nsa × $nsb (one matrix or one per component)"))
    dev(A) = copyto!(similar(op.Wa, Float64, size(A)), Matrix{Float64}(A))
    buf(a, b) = fill!(similar(op.Wa, Float64, a, b), 0)
    Na, Nb = dims[ia], dims[ib]
    pl = _FVPIVPlaneData{Ax,NC,typeof(op),typeof(op.Wa)}(op, comps, map(dev, data), map(dev, masks),
        buf(Na, Nb), buf(nsa, Nb), buf(nsa, nsb), buf(nsa, nsb), buf(nsa, nsb), buf(Na, nsb), buf(Na, Nb),
        buf(1, 1), zeros(1))
    host = (_fv_piv_host_op(op), map(d -> Matrix{Float64}(Array(d)), data),
            map(m -> Matrix{Float64}(Array(m)), masks))
    return pl, host
end

function _fv_piv_term(term, proto, n)
    length(term) in (2, 3) || throw(ArgumentError("a term is (objective, weight) or (objective, weight, target)"))
    obj = term[1]
    obj isa FVObjective || throw(ArgumentError("term objective must be an FVObjective"))
    weight = Float64(term[2])
    target = length(term) == 3 ? Float64(term[3]) : NaN
    length(term) == 3 && isnan(target) && throw(ArgumentError("term target must be a number"))
    if obj isa FVLinearFunctional
        wh = zeros(Float64, n)
        for k in eachindex(obj.index)
            1 <= obj.index[k] <= n || throw(BoundsError(wh, obj.index[k]))
            wh[obj.index[k]] += obj.weight[k]
        end
        w = copyto!(similar(proto, Float64, n), wh)
        return _FVPIVTerm(obj, weight, target, w, nothing, reshape(fill!(similar(proto, Float64, 1), 0), 1, 1),
                          zeros(1))
    end
    return _FVPIVTerm(obj, weight, target, nothing, fill!(similar(proto, Float64, n), 0), nothing, zeros(1))
end

function FVPIVObjective(prob::FVSteadyProblem3D, ops, data; mask = nothing, sigma_u::Real = 1.0,
                        terms = ())
    g = prob.grid
    isempty(ops) && throw(ArgumentError("FVPIVObjective needs at least one PIV operator"))
    length(data) == length(ops) || throw(DimensionMismatch("one data tuple per PIV operator"))
    mask === nothing || length(mask) == length(ops) ||
        throw(DimensionMismatch("one mask entry per PIV operator"))
    sigma_u > 0 || throw(ArgumentError("sigma_u must be positive"))
    n = fv_nunknowns(prob)
    built = ntuple(p -> _fv_piv_plane(ops[p], data[p], mask === nothing ? nothing : mask[p], g),
                   length(ops))
    proto = first(ops).wn
    tterms = Tuple(_fv_piv_term(t, proto, n) for t in terms)
    gU = fill!(similar(proto, Float64, n), 0)
    return FVPIVObjective(g, n, map(first, built), map(last, built), Float64(sigma_u), tterms, gU)
end

fv_objective_explicit(obj::FVPIVObjective) = any(t -> fv_objective_explicit(t.obj), obj.terms)

# value of `piv_misfit` for any element type (piv_misfit itself also returns the Float64
# adjoint, which rejects dual numbers): same masking and 1/2 sum m r^2
function _fv_piv_misfit_value(op, u, data, mask)
    s = piv_apply(op, u)
    T = eltype(s[1])
    L = zero(T)
    for c in eachindex(s)
        m = mask === nothing ? nothing : (mask isa Tuple ? mask[c] : mask)
        for q in eachindex(s[c])
            mq = m === nothing ? 1.0 : m[q]
            mq > 0 && (L += mq * (s[c][q] - data[c][q])^2)
        end
    end
    return L / 2
end

# host reference (generic element type): fields from U, then the piv_misfit value per plane
function fv_objective(obj::FVPIVObjective, U, prob)
    g = obj.grid
    Uh = U isa Array ? U : Array(U)
    A = reshape(view(Uh, 1:fv_nunknowns(g)), _fv_nfields(g), g.Nx, g.Ny, g.Nz)
    u = (A[1, :, :, :], A[2, :, :, :], A[3, :, :, :])
    J = zero(eltype(Uh))
    for (op, data, mask) in obj.host
        J += _fv_piv_misfit_value(op, u, data, mask) / obj.sigma_u^2
    end
    for t in obj.terms
        J += _fv_term_value(t, fv_objective(t.obj, Uh, prob))
    end
    return J
end

fv_objective_support(obj::FVPIVObjective, prob) = collect(1:obj.n)

# C[a, b] = Σ_n wn[n] U[component comp of the cell (a, b, n) of the plane axes]
@kernel function _fv_piv_contract_kernel!(C, @Const(U), @Const(wn), g, comp, ::Val{Ax}) where {Ax}
    a, b = @index(Global, NTuple)
    acc = 0.0
    @inbounds for n in 1:length(wn)
        w = wn[n]
        if !iszero(w)
            i, j, k = Ax == 1 ? (n, a, b) : Ax == 2 ? (a, n, b) : (a, b, n)
            acc += w * U[fv_index(g, i, j, k, comp)]
        end
    end
    @inbounds C[a, b] = acc
end

# gU[...] += scale wn[n] G[a, b]; each state entry is written by one work item
@kernel function _fv_piv_scatter_kernel!(gU, @Const(G), @Const(wn), g, comp, scale, ::Val{Ax}) where {Ax}
    a, b = @index(Global, NTuple)
    @inbounds gv = scale * G[a, b]
    @inbounds for n in 1:length(wn)
        w = wn[n]
        if !iszero(w)
            i, j, k = Ax == 1 ? (n, a, b) : Ax == 2 ? (a, n, b) : (a, b, n)
            gU[fv_index(g, i, j, k, comp)] += w * gv
        end
    end
end

# aᵀ b as a 1 × 1 matrix product into a preallocated device slot, then one
# element copied to the host: LinearAlgebra.dot on CuArrays allocated a small
# device result per call (512 bytes per gradient measured on the GPU)
function _fv_piv_inner!(acc, hacc, a, b)
    mul!(acc, reshape(a, 1, length(a)), reshape(b, length(b), 1))
    copyto!(hacc, 1, acc, 1, 1)
    return hacc[1]
end

function _fv_piv_plane!(gU, pl::_FVPIVPlaneData{Ax,NC}, U, g, scale, backend) where {Ax,NC}
    op = pl.op
    nd = (size(op.Wa, 2), size(op.Wb, 2))
    contract! = _fv_piv_contract_kernel!(backend, 64)
    scatter! = _fv_piv_scatter_kernel!(backend, 64)
    L = 0.0
    for c in 1:NC
        comp = pl.comps[c]
        contract!(pl.C, U, op.wn, g, comp, Val(Ax); ndrange = nd)
        KernelAbstractions.synchronize(backend)
        mul!(pl.T1, op.Wa, pl.C)
        mul!(pl.S, pl.T1, transpose(op.Wb))
        d = pl.data[c]; m = pl.mask[c]
        pl.r .= ifelse.(m .> 0, pl.S .- d, 0.0)
        pl.gs .= m .* pl.r
        L += _fv_piv_inner!(pl.acc, pl.hacc, pl.r, pl.gs) / 2
        mul!(pl.T2, transpose(op.Wa), pl.gs)
        mul!(pl.G, pl.T2, op.Wb)
        scatter!(gU, pl.G, op.wn, g, comp, scale, Val(Ax); ndrange = nd)
        KernelAbstractions.synchronize(backend)
    end
    return L
end

"""
    fv_objective_gradient!(obj::FVPIVObjective, U, prob) -> (J, gU)

Value of `obj` and its state gradient `∂𝒥/∂U` (the vector `obj.gU`, on the
backend of `U`, overwritten) by the device path: kernels and matrix products
only for the PIV planes and linear terms; host duals for nonlinear terms.
"""
fv_objective_gradient!(obj::FVPIVObjective, U, prob) = (r = _fv_piv_value_grad!(obj, U, prob); (r.J, obj.gU))

function _fv_piv_value_grad!(obj::FVPIVObjective, U, prob)
    length(U) == obj.n || throw(DimensionMismatch("state length $(length(U)) ≠ $(obj.n)"))
    backend = KernelAbstractions.get_backend(U)
    typeof(KernelAbstractions.get_backend(obj.gU)) === typeof(backend) ||
        throw(ArgumentError("FVPIVObjective built on another backend than the state"))
    gU = obj.gU
    fill!(gU, 0)
    scale = 1 / obj.sigma_u^2
    J = 0.0
    for pl in obj.planes
        J += scale * _fv_piv_plane!(gU, pl, U, obj.grid, scale, backend)
    end
    slopes = zeros(Float64, length(obj.terms))
    host = nothing
    for (q, t) in enumerate(obj.terms)
        if t.w !== nothing
            x = _fv_piv_inner!(t.acc, t.hacc, t.w, U)
            s = _fv_term_slope(t, x)
            gU .+= s .* t.w
        else
            host === nothing && (host = (Vector{Float64}(Array(U)), _fv_host_problem(prob)))
            x, gx = _fv_objective_grad_u(t.obj, host[1], host[2])
            s = _fv_term_slope(t, x)
            copyto!(t.buf, gx)
            gU .+= s .* t.buf
        end
        J += _fv_term_value(t, x)
        slopes[q] = s
    end
    KernelAbstractions.synchronize(backend)
    return (; J, slopes, host)
end

"""
    fv_steady_gradient(solver, U, θ, obj::FVPIVObjective; params, ws = nothing,
                       finalize = true, law_gradient = :contraction) -> NamedTuple

Method of [`fv_steady_gradient`](@ref) for the composite PIV objective: the
state gradient comes from the device path of [`fv_objective_gradient!`](@ref)
(`gU` in the result is `obj.gU`, on the solver backend), the adjoint and the
parameter cotangent are those of [`fv_steady_adjoint`](@ref), and `explicit`
sums the explicit parameter derivatives of the nonlinear terms (zero for PIV
planes and linear terms). Same return fields as the generic method.
"""
function fv_steady_gradient(s::FVNewtonSolver, U, θ, obj::FVPIVObjective;
                            params::FVParameterMap, ws = nothing, finalize::Bool = true,
                            law_gradient = :contraction)
    _fv_require_adjoint_map(s.prob, params)
    _fv_check_law_gradient(law_gradient)
    t0 = time_ns()
    val = _fv_piv_value_grad!(obj, U, s.prob)
    t1 = time_ns()
    cot, st = fv_steady_adjoint(s, U, obj.gU; params, θ, ws, finalize, law_gradient)
    t2 = time_ns()
    explicit = zeros(Float64, fv_nparams(params, s.prob))
    if fv_objective_explicit(obj)
        Uh, hprob = val.host === nothing ? (Vector{Float64}(Array(U)), _fv_host_problem(s.prob)) : val.host
        θh = params isa FVViscosityField ? vec(Array(hprob.nu)) : Vector{Float64}(Array(θ))
        for (q, t) in enumerate(obj.terms)
            fv_objective_explicit(t.obj) || continue
            explicit .+= val.slopes[q] .* _fv_objective_grad_theta(t.obj, params, Uh, hprob, θh)
        end
    end
    grad = vec(Array(cot)) .+ explicit
    t3 = time_ns()
    return (; J = val.J, grad, gU = obj.gU, explicit, s = cot, λ = st.λ, residual_inf = st.residual_inf,
            t_objective = (t1 - t0) / 1e9, t_finalize = st.t_finalize, t_solve = st.t_solve,
            t_param = st.t_param, t_explicit = (t3 - t2) / 1e9, t_total = (t3 - t0) / 1e9)
end
