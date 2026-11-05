# Small quasi-Newton minimiser used by `calibrate_carreau.jl`: BFGS with a dense
# inverse-Hessian update and a backtracking Armijo line search. No package
# dependency beyond LinearAlgebra. Not part of the package API.

using LinearAlgebra

"""
    gc_bfgs(fval, fgrad, x0; maxiter=100, ftarget=-Inf, gtol_rel=1e-8, xtol=1e-9,
            ftol=1e-12, max_step=0.2, c1=1e-4, max_backtracks=30, callback=nothing)
        -> NamedTuple

Minimise f from `x0`.

- `fval(x)` returns f(x), or `Inf` when x cannot be evaluated (the line search
  then shortens the step).
- `fgrad(x)` returns ∇f(x). It is only called at a point where the last `fval`
  call returned a finite value, so an objective may reuse work done there.

Iteration k: direction d = −H g (H = inverse-Hessian estimate, identity at the
start, rescaled by sᵀy/yᵀy before the first update, reset to the identity if d is
not a descent direction); d is scaled down so that |d_i| ≤ `max_step` (a number
or one value per component); step α = 1, then backtracking until the Armijo
condition f(x + αd) ≤ f(x) + c1 α gᵀd holds (quadratic interpolation clamped to
[0.1α, 0.5α] after a finite trial, α/4 after a non-finite one). The BFGS update
H ← (I − ρ s yᵀ) H (I − ρ y sᵀ) + ρ s sᵀ, ρ = 1/sᵀy, is skipped when
sᵀy ≤ 1e-12 ‖s‖‖y‖.

Stopping rule, checked in this order after each accepted step:

1. `:target`: f ≤ `ftarget` (also checked at the start);
2. `:gradient`: ‖g‖∞ ≤ `gtol_rel` · ‖g(x0)‖∞;
3. `:step`: ‖x_k − x_{k−1}‖∞ ≤ `xtol`;
4. `:objective`: f_{k−1} − f_k ≤ `ftol` · |f_{k−1}|;
5. `:linesearch`: no Armijo point after `max_backtracks` reductions (x is the
   last accepted point);
6. `:maxiter`: k = `maxiter`.

`callback(rec)` is called with each history record. Returns `(; x, f, g, stop,
iterations, n_fval, n_fgrad, n_fail, n_skip, n_reset, history)`; `history` holds
one record per accepted point `(; iter, f, gnorm, step, alpha, backtracks, x)`.
"""
function gc_bfgs(fval, fgrad, x0::AbstractVector{<:Real}; maxiter::Integer=100,
                 ftarget::Real=-Inf, gtol_rel::Real=1e-8, xtol::Real=1e-9, ftol::Real=1e-12,
                 max_step=0.2, c1::Real=1e-4, max_backtracks::Integer=30, callback=nothing)
    n = length(x0)
    x = Float64.(collect(x0))
    f = Float64(fval(x))
    isfinite(f) || error("gc_bfgs: objective is not finite at the starting point")
    g = Float64.(fgrad(x))
    g0 = norm(g, Inf)
    H = Matrix{Float64}(I, n, n)
    scaled = false
    n_fval = 1; n_fgrad = 1; n_fail = 0; n_skip = 0; n_reset = 0
    cap = max_step isa Real ? fill(Float64(max_step), n) : Float64.(collect(max_step))
    length(cap) == n || throw(DimensionMismatch("max_step must have $n entries"))
    history = NamedTuple[]
    rec = (; iter=0, f, gnorm=g0, step=0.0, alpha=NaN, backtracks=0, x=copy(x))
    push!(history, rec); callback === nothing || callback(rec)
    stop = :maxiter
    k = 0
    if f <= ftarget
        stop = :target
    elseif g0 == 0
        stop = :gradient
    else
        while k < maxiter
            k += 1
            d = -(H * g)
            slope = dot(g, d)
            if !(slope < 0)
                H = Matrix{Float64}(I, n, n); scaled = false; n_reset += 1
                d = -copy(g); slope = dot(g, d)
            end
            r = maximum(abs.(d) ./ cap)
            if r > 1
                d ./= r; slope /= r
            end
            α = 1.0; nb = 0; ft = Inf; xt = x
            found = false
            while true
                xt = x .+ α .* d
                ft = Float64(fval(xt)); n_fval += 1
                isfinite(ft) || (n_fail += 1)
                if isfinite(ft) && ft <= f + c1 * α * slope
                    found = true
                    break
                end
                nb += 1
                nb > max_backtracks && break
                if isfinite(ft)
                    den = 2 * (ft - f - slope * α)
                    αq = den > 0 ? -slope * α^2 / den : 0.5α
                    α = clamp(αq, 0.1α, 0.5α)
                else
                    α *= 0.25
                end
            end
            if !found
                k -= 1
                stop = :linesearch
                break
            end
            gt = Float64.(fgrad(xt)); n_fgrad += 1
            s = xt .- x; y = gt .- g; sy = dot(s, y)
            if sy > 1e-12 * norm(s) * norm(y)
                if !scaled
                    H = (sy / dot(y, y)) * Matrix{Float64}(I, n, n)
                    scaled = true
                end
                ρ = 1 / sy
                A = I - ρ * s * y'
                H = A * H * A' + ρ * (s * s')
                H = (H + H') / 2
            else
                n_skip += 1
            end
            fprev = f
            x, f, g = xt, ft, gt
            rec = (; iter=k, f, gnorm=norm(g, Inf), step=norm(s, Inf), alpha=α, backtracks=nb,
                   x=copy(x))
            push!(history, rec); callback === nothing || callback(rec)
            if f <= ftarget
                stop = :target; break
            elseif norm(g, Inf) <= gtol_rel * g0
                stop = :gradient; break
            elseif norm(s, Inf) <= xtol
                stop = :step; break
            elseif fprev - f <= ftol * abs(fprev)
                stop = :objective; break
            end
        end
    end
    return (; x, f, g, stop, iterations=k, n_fval, n_fgrad, n_fail, n_skip, n_reset, history)
end
