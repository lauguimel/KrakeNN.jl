# Platform-contract wrapper for the IncNS solver stack (mirrors the LBM
# precedent in src/platform/solution.jl + sample.jl). Thin, behaviour-preserving:
# `solve` forwards the params NamedTuple verbatim (as keyword arguments) to the
# matching `solve_incns_*` driver and wraps its NamedTuple result. No solver
# math lives here.

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

Declares [`capabilities`](@ref) `Set((ForwardSolve, SteadyResidual))`: the CPU
forward solve is the validated path (the CUDA seam methods are manual-load, so
`GPUExecution` is deliberately NOT claimed). `SteadyResidual` is the steady
colocated FV residual of `fv_steady_residual.jl` (C1 of the FV Newton roadmap),
reached as `residual(prob::FVSteadyProblem2D, ::IncNS, U)`; it is the residual
of that coupled discretisation, not of the SIMPLE drivers.
"""
struct IncNS <: AbstractMethod
    driver::Symbol
    function IncNS(driver::Symbol)
        driver in (:simple, :cavity, :cavity_mg, :projection, :manifold) ||
            throw(ArgumentError("unknown IncNS driver :$driver — expected one of " *
                                ":simple, :cavity, :cavity_mg, :projection, :manifold"))
        return new(driver)
    end
end

capabilities(::IncNS) = Set((ForwardSolve, SteadyResidual))

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
