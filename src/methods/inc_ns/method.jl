# Platform-contract wrapper for the IncNS solver stack (mirrors the LBM
# precedent in src/platform/solution.jl + sample.jl). Thin, behaviour-preserving:
# `solve` forwards the params NamedTuple verbatim (as keyword arguments) to the
# matching `solve_incns_*` driver and wraps its NamedTuple result. No solver
# math lives here.

const _INCNS_DRIVERS = (:simple, :cavity, :cavity_mg, :projection, :manifold, :fv_newton)

"""
    IncNS <: AbstractMethod

The incompressible Navier-Stokes (FV/FD) solver stack as a contract-level
[`AbstractMethod`](@ref). Constructed with the driver to dispatch to:

    IncNS(driver::Symbol)

with `driver` one of:

- `:simple`     → [`solve_incns_simple`](@ref) (steady SIMPLE, body-force channel)
- `:cavity`     → [`solve_incns_cavity`](@ref) (steady SIMPLE lid-driven cavity)
- `:cavity_mg`  → [`solve_incns_cavity_mg`](@ref) (matrix-free multigrid cavity)
- `:projection` → [`solve_incns_projection`](@ref) (unsteady fractional-step)
- `:manifold`   → [`solve_incns_manifold`](@ref) (inlet/outlet manifold SIMPLE)
- `:fv_newton`  → [`solve_incns_fv_newton`](@ref) (steady colocated FV, Newton
  on the assembled coloured-dual Jacobian; C2 of the FV Newton roadmap)

[`capabilities`](@ref) depend on the driver. Every driver declares
`ForwardSolve` (the CPU forward solve is the validated path; the CUDA seam
methods are opt-in, so `GPUExecution` is deliberately NOT claimed). Only
`:fv_newton` also declares `SteadyResidual`: the steady colocated FV residual
of `fv_steady_residual.jl` (C1), reached as
`residual(prob::FVSteadyProblem2D, IncNS(:fv_newton), U)`, is the residual that
driver drives to zero. The SIMPLE drivers converge to a state that depends on
their relaxation factor (roadmap F16) and expose no residual of their own.
"""
struct IncNS <: AbstractMethod
    driver::Symbol
    function IncNS(driver::Symbol)
        driver in _INCNS_DRIVERS ||
            throw(ArgumentError("unknown IncNS driver :$driver — expected one of " *
                                join((":" * string(d) for d in _INCNS_DRIVERS), ", ")))
        return new(driver)
    end
end

capabilities(m::IncNS) =
    m.driver === :fv_newton ? Set((ForwardSolve, SteadyResidual)) : Set((ForwardSolve,))

"""
    IncNSSolution{R} <: AbstractSolution

Thin, behaviour-preserving wrapper around the `NamedTuple` an IncNS driver
returns (`u`, `v`, `p`, `residual_history`, …). Gives the result a nominal type
in the [`AbstractSolution`](@ref) hierarchy so the platform can dispatch
`sample`/`observe` on it without coupling to the internal representation —
the exact analogue of `LBMSolution`. The wrapped result is stored verbatim
(parity test: `test/platform/incns_contract_test.jl`).
"""
struct IncNSSolution{R} <: AbstractSolution
    result::R
end

"""
    solve(params::NamedTuple, m::IncNS) -> IncNSSolution

Run the IncNS driver selected by `m.driver` with `params` splatted verbatim as
keyword arguments, and wrap the returned `NamedTuple` in an
[`IncNSSolution`](@ref). Behaviour-preserving: bit-identical to calling the
`solve_incns_*` driver directly with the same keywords.
"""
function solve(params::NamedTuple, m::IncNS)
    res = if m.driver === :simple
        solve_incns_simple(; params...)
    elseif m.driver === :cavity
        solve_incns_cavity(; params...)
    elseif m.driver === :cavity_mg
        solve_incns_cavity_mg(; params...)
    elseif m.driver === :projection
        solve_incns_projection(; params...)
    elseif m.driver === :fv_newton
        solve_incns_fv_newton(; params...)
    else # :manifold (constructor-validated)
        solve_incns_manifold(; params...)
    end
    return IncNSSolution(res)
end

"""
    sample(sol::IncNSSolution, field::Symbol)
    sample(sol::IncNSSolution, field::Symbol, query)

Query an IncNS solution by field name (and optional location `query`) on the
platform interface — a faithful pass-through to the wrapped driver result,
mirroring the `LBMSolution` methods.

- `sample(sol, :u)` / `sample(sol, :u, :)` → the whole field.
- `sample(sol, :u, (i, j))` → the value at index `(i, j)`.
"""
sample(sol::IncNSSolution, field::Symbol) = getproperty(sol.result, field)
sample(sol::IncNSSolution, field::Symbol, ::Colon) = sample(sol, field)
sample(sol::IncNSSolution, field::Symbol, idx::Tuple) = sample(sol, field)[idx...]
