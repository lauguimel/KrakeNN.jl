# Simulation state and checkpoints

Kraken has a **solver-agnostic contract for resumable simulation state**: a
common way, shared by every solver that adopts it, to carry a run forward in
memory, save it to a file, and load it back. This page explains what that
contract is for, what it lets you do today, and what it does not do yet.

A **state** is everything a solver needs to keep from one cycle to the next —
population arrays, running histories, counters — as opposed to values that can
always be recomputed from it. A **checkpoint** is a state written to a file on
disk, so it survives after the Julia process that produced it has ended.

## What this is for

**Jobs bounded by an HPC walltime.** A compute cluster gives you a job for a
fixed wall-clock time (a *walltime*, e.g. 24 hours), after which the job is
killed whether your simulation is finished or not. Without a checkpoint, a run
that does not finish inside one walltime window has to restart from cycle zero
every time. With one, the last segment writes a checkpoint just before the
walltime runs out, and the next job picks it up exactly where the first one
stopped.

**Crash recovery.** A run can fail mid-way — a diverged solve, a node failure,
a killed process — and rebuilding the lost cycles by hand is wasted compute.
A checkpoint written at a cycle boundary is a safe point to resume from, and
the write itself is designed so that a crash during the write never destroys
the previous good checkpoint (see [rotation](#the-checkpoint-file) below).

**Following a solution branch while a parameter changes.** Some studies need
to change one physical parameter partway through a run and keep going from the
current state, rather than restarting from the initial condition at the new
parameter value — for example, slowly ramping an electric field and watching
how the flow responds, instead of running one independent simulation per
field value. The contract has a dedicated verb for this,
[`update_parameter!`](@ref), separate from resuming after an interruption.

## What works today, and what does not

**Works today:**

- The in-memory contract and its generic verbs — [`init_state`](@ref),
  [`advance!`](@ref), [`solution`](@ref) — for any solver that implements them
  (source: `src/platform/state.jl`).
- The HDF5 checkpoint file format: writing ([`write_checkpoint`](@ref)),
  reading ([`read_checkpoint`](@ref)), and inspecting a file without reading
  its arrays ([`checkpoint_info`](@ref)) (source: `src/io/checkpoint_hdf5.jl`).
  A damaged or truncated file is refused rather than silently misread. Writing
  a new checkpoint keeps the previous one (`path * ".prev"`), so one good
  generation always survives a crash mid-write.
- Running the **electroconvection** solver (`ECState`, 2D) in segments, in
  memory: `init_state(ECState; ...)`, `advance!`, `solution`. Splitting a run
  into segments gives **bit-for-bit the same result** as running it
  continuously, on the same backend and the same floating-point precision —
  see the worked example below.

**Not yet:**

- **Saving or restoring the electroconvection state to a file, or changing a
  parameter on a restored `ECState`.** The client hooks a solver must add to
  support this — `Kraken.snapshot`, `restore_state`, `update_parameter!` — are
  not implemented for `ECState` yet.
  Calling [`export_state`](@ref) or [`save_checkpoint`](@ref) on an `ECState`
  throws.
- **Restarting a run from a `.krk` configuration file.** The `.krk` runner does
  not know about checkpoints yet; resuming is only reachable from Julia code
  today.
- **Any solver other than electroconvection.** No other Kraken solver
  implements the state contract yet.
- **Restoring a checkpoint onto a different grid, a different floating-point
  precision, or a different backend (CPU ↔ GPU) than the one it was written
  from.** This has not been built or measured.

## Run a simulation in segments

This example runs the electroconvection solver for 11 cycles two ways — once
continuously, once split into segments of 4 and 7 cycles — and checks that the
results are identical. `history_interval=3` means a history entry is recorded
every 3rd cycle; the split point (cycle 4) is deliberately **not** a multiple
of 3, to make the point below concrete.

```julia
using Kraken

cfg = (Nx=8, Ny=12, T=175.0, C=10.0, M=10.0, Ma_E=1e-2, alpha=1e-4,
       history_interval=3)

continuous = init_state(ECState; cfg...)
advance!(continuous, 11; sample_final=true)
sol_continuous = solution(continuous)

segmented = init_state(ECState; cfg...)
advance!(segmented, 4)
advance!(segmented, 7; sample_final=true)
sol_segmented = solution(segmented)

sol_continuous.result.cycle_history == sol_segmented.result.cycle_history  # true
sol_continuous.result.ux == sol_segmented.result.ux                        # true
```

Both runs print the same history:

```
cycle_history: [3, 6, 9, 11]
umax_history:  [0.0012303241839259102, 0.00222354892427558,
                0.0026139779313496406, 0.0022204371645721914]
```

A history entry is recorded when the cycle count is a multiple of
`history_interval`, **or** when `sample_final=true` and the cycle is the last
one of the current `advance!` call. `sample_final` is opt-in per call, which
is why the one-shot driver `run_electroconvection_2d` — and the example above
— pass it only on the very last segment: it forces a sample at the horizon
even when the horizon does not fall on a multiple of `history_interval`.

Sampling is decided on `cycle` — the **global** count of cycles since
`init_state`, not the local count within one `advance!` call — precisely so
that splitting a run into segments (for a walltime, or to inspect it midway)
never changes which cycles get sampled, and therefore never changes the
result.

## The checkpoint file

A checkpoint is one HDF5 file. HDF5 is a binary file format for large
numerical data organized into a tree of *groups* (folders) and *datasets*
(arrays), each of which can carry *attributes* (small named values, like a
scalar or a short string).

| Group | Holds |
|---|---|
| `/` (root attributes) | `container_version`, `schema_version`, `solver`, `cycle`, `kraken_version`, `julia_version`, `created_unix` |
| `/fields` | Arrays whose shape is part of the contract (checked on restore) |
| `/series` | Append-only histories of free length (e.g. `umax_history`) |
| `/scalars` | Carried scalar state (one attribute per entry) |
| `/identity` | Configuration that defines *which* simulation this is |
| `/run_control` | Configuration that may differ between two segments of a run |
| `/parameters` | Physical parameters, changeable only through `update_parameter!` |
| `/derived` | Values recomputed from the configuration, stored to detect drift |

The three configuration classes are compared differently when a checkpoint is
restored:

- **`identity`** — compared key by key, must match exactly (same type, same
  value). This is the grid size, the floating-point precision, the numerical
  schemes selected, the physical non-dimensional numbers that define the
  problem: things that make it *this* simulation and not a different one.
- **`run_control`** — never compared. This is the run's horizon (how many more
  cycles to run), the backend, and similar settings that are allowed to differ
  between two segments of the same run — for example, extending a run past its
  original planned length.
- **`parameters`** — never compared. This is exactly the set of values
  [`update_parameter!`](@ref) is allowed to change; restoring a checkpoint
  after a parameter update must work, by design.

There are **two independent version numbers**: `container_version` is the
version of the *file layout* (the groups and how attributes are encoded), and
is a property of the checkpoint format itself, currently
[`CHECKPOINT_CONTAINER_VERSION`](@ref) `= 1`. `schema_version` is the version
of *one solver's* set of keys inside a checkpoint, and is chosen by that
solver. A file layout change bumps the first; a change to what one solver
stores bumps the second.

### Inspect a file

[`checkpoint_info`](@ref) reads the root attributes only — solver name,
schema and container versions, cycle, the Kraken and Julia versions that wrote
the file, and the write timestamp — without touching any array:

```julia
using Kraken

snap = StateSnapshot(
    solver="toy_doc_example", schema_version=1, cycle=42,
    fields=Dict("a" => [1.0 2.0 5.0; 3.0 4.0 6.0]),   # 2 x 3 in Julia
    series=Dict("umax" => [0.01, 0.02, 0.015]),
    scalars=Dict("last_sum" => 2.5),
    identity=Dict("nx" => 2, "ny" => 3, "scheme" => "bgk"),
    run_control=Dict("max_cycles" => 100),
    parameters=Dict("gain" => 0.3),
    derived=Dict("coef" => 0.25),
)
path = write_checkpoint(joinpath(mktempdir(), "toy.h5"), snap)
checkpoint_info(path)
```

which prints:

```
(container_version = 1, solver = "toy_doc_example", schema_version = 1,
 cycle = 42, kraken_version = "0.3.0", julia_version = "1.12.5",
 created_unix = 1.7896405512886837e9)
```

### Read it from Python

`h5py` opens a Kraken checkpoint directly — no Kraken or Julia installation
needed. Two things differ from what a Julia array of the same shape shows:

- **Index order is reversed.** Julia stores arrays column-major (the first
  index varies fastest); `h5py`/NumPy are row-major. A Julia array of shape
  `(Nx, Ny)` therefore appears in Python with shape `(Ny, Nx)`.
- **Strings come back as bytes.** Decode them (`.decode()`), and split the
  `keys` attribute of a scalar group on `"\n"` to get the list of keys stored
  in that group (a key listed there with no matching attribute holds
  `nothing`/`None`).

```python
import h5py

with h5py.File("toy.h5", "r") as f:
    print(dict(f.attrs))                       # root: solver, cycle, versions...
    print(f["fields"]["a"].shape)               # (3, 2): it was (2, 3) in Julia
    keys = f["identity"].attrs["keys"].decode().split("\n")
    identity = {k: f["identity"].attrs[k] for k in keys if k in f["identity"].attrs}
    print(identity)
```

which prints:

```
{'container_version': np.int64(1), 'created_unix': np.float64(1789640891.265115),
 'cycle': np.int64(42), 'julia_version': np.bytes_(b'1.12.5'),
 'kraken_version': np.bytes_(b'0.3.0'), 'schema_version': np.int64(1),
 'solver': np.bytes_(b'toy_doc_example')}
(3, 2)
{'nx': np.int64(2), 'ny': np.int64(3), 'scheme': np.bytes_(b'bgk')}
```

## What gets refused, and what the message looks like

Every refusal is a [`CheckpointError`](@ref), and its message always names the
offending key or object. Three real examples, produced from the checkpoint
built above:

**Mismatched identity key** — restoring into a simulation configured with a
different scheme:

```
CheckpointError: identity/scheme: expected "mrt"::String, found "bgk"::String
```

**Damaged file** — one bit flipped inside the file (this file was written
with the metadata-checksum option turned on, so libhdf5 itself refuses to open
the corrupted group rather than silently returning the wrong bytes):

```
CheckpointError: <path>: cannot read group /identity (damaged, truncated or
foreign file): HDF5.API.H5Error: Error opening object //identity
libhdf5 Stacktrace:
  [1] H5C__load_entry: Object cache/Read failed
      incorrect metadata checksum after all read attempts
   ⋮ (truncated)
```

**Non-finite value** — a `NaN` snuck into a field. The `StateSnapshot`
constructor only checks types and names; the finiteness check runs at
`write_checkpoint`/`export_state`, so that a diverged state never overwrites a
good checkpoint. Only arrays (`fields`, `series`) are checked: a carried scalar
may legitimately be `Inf`, for example a convergence indicator before its first
sample.

```
CheckpointError: fields/a: non-finite value NaN at index (2,); snapshot refused
```

## Make your own solver resumable

To add checkpoint support to a solver, subtype `AbstractSimulationState` and
implement the following hooks. The toy client in
`test/platform/state_contract_test.jl` (`ToyState`) is a complete, minimal
worked example of every one of them — read it alongside this checklist.

1. **`init_state(::Type{YourState}; kwargs...)`** — build a fresh state at
   cycle 0 from your solver's configuration keywords.
2. **`advance!(state, n; sample_final=false)`** — run `n` cycles in place.
   Decide whether to sample a history entry using the state's **global**
   cycle counter, never a per-call counter, so that any split of a run gives
   the same histories as running it continuously.
3. **`solution(state)`** — return the queryable result as an
   `AbstractSolution`, without mutating `state`.
4. **`Kraken.snapshot(state)`** — copy everything needed to resume into a
   `StateSnapshot`, sorting your state into `fields`, `series`, `scalars`,
   and the three configuration classes `identity`/`run_control`/`parameters`.
5. **`restore_state(::Type{YourState}, snap; backend, kwargs...)`** — rebuild
   a state from a snapshot. Call `check_compatible` and (if you wrote one)
   `Kraken.validate_snapshot` **before allocating anything**, so a refused
   snapshot costs nothing.
6. **`Kraken.at_boundary(state)`** — override this only if `advance!` can be
   interrupted mid-cycle (e.g. by a caught exception): return `true` exactly
   when `state` sits between two cycles. [`export_state`](@ref) refuses to
   checkpoint a state that is not at a boundary, so a checkpoint is never
   written from a half-finished cycle. Your `advance!` (even for `n == 0`)
   and `solution` must refuse such a state too, by calling
   `Kraken.require_boundary(state, "advance!")` / `Kraken.require_boundary(state, "solution")`
   as their first statement: it throws an `ArgumentError` and leaves the
   state untouched, so the only way forward is a fresh state or a restore
   from a known-good checkpoint. The default is `true`, which is
   correct for a solver whose `advance!` cannot fail partway through a cycle.
7. If your solver has parameters that may change between two segments of a
   run, declare them with **`updatable_parameters(::Type{YourState})`**
   (a `ParameterSpace`, the same object `fit` consumes) and implement
   **`update_parameter!(state, name, value)`**, starting with
   `Kraken.check_updatable`.

Once these are in place, the rule is: **your client must pass
`run_state_contract_suite`** (`test/platform/state_contract_suite.jl`). It
checks, generically and for any solver, that a split run matches a continuous
one bit for bit, that a round trip through memory and through disk reproduces
the state exactly, that every stored field actually matters (a negative
control perturbs each one and checks the final result changes), and that
mismatched identity, schema, or an interrupted cycle are all refused with the
right message.

## Limits and guarantees

- A restart is checked **bit for bit**, not to a tolerance: the contract suite
  treats a "close enough" restart that is missing a piece of state as a
  failure, not a pass.
- Bit-for-bit equality of a restart has been established on CPU. It has not
  been measured on GPU (CUDA/Metal), where kernel scheduling could in
  principle reorder floating-point sums.
- An export holds one full host-side copy of the state in memory; there is no
  streaming writer yet, which matters for very large 3D states.
- There is no `fsync` after a checkpoint write; keeping the previous
  generation (`.prev`) reduces, but does not eliminate, the risk of losing
  both files to a node failure on a networked filesystem.
- A checkpoint written with one `schema_version` cannot be read by code
  expecting a different one: the `migrate` hook exists for this but currently
  always refuses. The first schema change will need a real `migrate` method.
