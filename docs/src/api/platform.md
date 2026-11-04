# Platform state & checkpoints API

The resumable simulation-state contract (`AbstractSimulationState`,
`StateSnapshot`, the generic verbs) and its HDF5 checkpoint layer. See the
[Simulation state and checkpoints](../users/simulation-state-checkpoints.md)
guide for a narrative introduction and worked examples.

```@autodocs
Modules = [Kraken]
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
Modules = [Kraken]
Pages = [
    "drivers/ehd_ec_state.jl",
]
Order = [:type, :function]
```
