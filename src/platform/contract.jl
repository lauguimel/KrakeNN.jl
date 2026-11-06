"""
Platform contract — the stable interface every method/physics enters behind.

Phase 0: types and capability introspection ONLY. No behaviour, no dependencies,
fully additive. The verbs (`solve`, `sample`, `observe`, `residual`, `fit`) are
introduced in later phases; see `docs/platform/`.
"""

"""
    AbstractProblem

A method-agnostic problem description (domain, boundary conditions, physics).
What is solved — independent of how it is discretized.
"""
abstract type AbstractProblem end

"""
    AbstractMethod

A discretization/solve method (LBM, FV/FD, VoF, …). The caller names the method;
there is no automatic method selection. Declares what it can do via [`capabilities`](@ref).
"""
abstract type AbstractMethod end

"""
    AbstractSolution

A queryable solution. Internal storage (grid, populations, conformation) is private;
the only contract is to be queryable by field + query (via `sample`, Phase 1).
"""
abstract type AbstractSolution end

"""
    AbstractObservable

A quantity comparable to data (a drag coefficient, a probe, a profile). Defined via
`sample` on an [`AbstractSolution`](@ref), never via a method's internal storage.
"""
abstract type AbstractObservable end

"""
    AbstractClosure

An injectable term — analytic now, learned later — evaluated INSIDE the residual.
The single injection point for a parameter/field/model that may be calibrated or
inferred from data (same API whether analytic or a neural closure).
"""
abstract type AbstractClosure end

"""
    Capability

A capability a method may declare, introspected by the platform (e.g. `fit` chooses
a gradient vs finite-difference path; an NL/agent layer checks what is feasible
before dispatch).

Values: `ForwardSolve`, `GPUExecution`, `SteadyAdjoint`, `TransientAdjoint`,
`FiniteDiff`, `NeuralClosure`, `SteadyResidual`, `AssembledJacobian`.

`AssembledJacobian`: the method assembles the exact sparse Jacobian `∂R/∂U` of
its steady residual and keeps its factorization, so a steady adjoint is one
transposed solve with those factors (declared by `IncNS(:fv_newton)` only;
the LBM `SteadyAdjoint` is matrix-free).
"""
@enum Capability ForwardSolve GPUExecution SteadyAdjoint TransientAdjoint FiniteDiff NeuralClosure SteadyResidual AssembledJacobian

"""
    capabilities(m::AbstractMethod) -> Set{Capability}

The set of [`Capability`](@ref) a method supports. Default: empty (a method opts in
by adding a method to `capabilities` returning its set).
"""
capabilities(::AbstractMethod) = Set{Capability}()
