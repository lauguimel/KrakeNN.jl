# Platform state & checkpoints API

The resumable simulation-state contract (`AbstractSimulationState`,
`StateSnapshot`, the generic verbs) and its HDF5 checkpoint layer. See the
[Simulation state and checkpoints](../users/simulation-state-checkpoints.md)
guide for a narrative introduction and worked examples.

```@autodocs
Modules = [KrakeNN]
Pages = [
    "platform/state.jl",
    "io/checkpoint_hdf5.jl",
]
Order = [:constant, :type, :function]
```

## Contract types the state verbs refer to

`solution(state)` returns an `AbstractSolution`, and `solve` stays the one-shot
verb (create, advance, produce the result).

```@docs
AbstractSolution
LBMSolution
IncNSSolution
solve
```

## Electroconvection state (worked example client)

`ECState`/`ECSolution` are the first client of the contract above: the 2D
electroconvection driver split into `init_state`/`advance!`/`solution`.

```@autodocs
Modules = [KrakeNN]
Pages = [
    "drivers/ehd_ec_state.jl",
]
Order = [:type, :function]
```

## Plane-PIV observation operator (3D)

`piv_operator` builds the linear map from a 3D cell-centred velocity field
(`u[i,j,k]` at `((i-1/2)hx, (j-1/2)hy, (k-1/2)hz)`, the D3Q19 layout) to what a PIV
plane measures: the two in-plane components (plus the out-of-plane one for
stereo-PIV) on a user-chosen grid of vectors. The plane is axis-aligned at any
physical position. Each sample is (i) the box average over the interrogation
window of the trilinear interpolant (midpoint rule, sub-point spacing at most
`h/2`), (ii) weighted along the normal by a Gaussian light sheet of standard
deviation `thickness` (truncated at 3 sigma, discretised on equispaced symmetric
points, normalised to 1); `thickness = 0` is pure trilinear interpolation.
Positions outside the first/last cell centre are clamped (constant
extrapolation), so linear fields are exact only in the interior.

The map is separable (normal contraction, then two small dense matrices), built
on the host and stored with the array type of a template field, so
`piv_apply`, `piv_apply_adjoint` (exact transpose) and `piv_misfit`
(`L = 1/2 sum m (sample - data)^2` with a mask for missing vectors, returning
`L` and `dL/d(ux,uy,uz)`) run on CPU and CUDA arrays with broadcasts and
matrix products only.

```@autodocs
Modules = [KrakeNN]
Pages = ["platform/piv_observation.jl"]
Order = [:type, :function]
```
