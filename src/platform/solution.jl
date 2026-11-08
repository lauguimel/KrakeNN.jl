"""
    LBMSolution{R} <: AbstractSolution

Thin, behaviour-preserving wrapper around the `NamedTuple` that `run_simulation`
returns (`ρ`, `ux`, `uy`, `setup`, …). Gives the result a nominal type in the
[`AbstractSolution`](@ref) hierarchy so the platform can dispatch `sample`/`observe`
on it without coupling to the internal representation. Phase 0: it stores the result
verbatim — see the parity test in `test/platform/contract_parity_test.jl`.
"""
struct LBMSolution{R} <: AbstractSolution
    result::R
end

"""
    LBM <: AbstractMethod

The lattice-Boltzmann method as a contract-level [`AbstractMethod`](@ref). Phase 0:
a marker type whose `solve` forwards to the existing `run_simulation`; it declares its
[`capabilities`](@ref) for introspection.
"""
struct LBM <: AbstractMethod end

capabilities(::LBM) = Set((ForwardSolve, GPUExecution, SteadyAdjoint, SteadyResidual))

"""
    solve(problem, method::AbstractMethod; kwargs...) -> AbstractSolution

Run `problem` with `method` and return an [`AbstractSolution`](@ref). The caller names
the method explicitly (no automatic selection). Phase 0: `LBM` forwards verbatim to
`run_simulation` (no behaviour change) and parameters pass through as `kwargs`. The
`solve(problem, method, p)` form driven by a `ParameterSpace` arrives in Phase 2.

(When SciML enters in Phase 4, `solve` will be switched to extend `CommonSolve.solve`;
today it is KrakeNN's own generic function — no such dependency yet.)
"""
solve(problem, ::LBM; kwargs...) = LBMSolution(run_simulation(problem; kwargs...))
