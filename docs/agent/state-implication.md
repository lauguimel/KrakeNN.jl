---
module: platform/state
path: src/platform/state.jl
owner_concern: resumable-simulation-state
status: contract-and-disk-layer (first client: drivers/ehd_ec_state.jl, in-memory verbs only)
last_verified: 2026-09-18
depends_on: [platform/contract.jl, platform/calibration.jl, io/checkpoint_hdf5.jl]
---

# platform/state — implication map

Solver-agnostic contract for a resumable simulation state (issue #26), plus its
on-disk layer. `src/platform/state.jl` defines the in-memory contract and never
mentions a file format; `src/io/checkpoint_hdf5.jl` is the only file of Kraken that
talks to HDF5. Design record: `docs/platform/07-STATE-CONTRACT.md` (ADR-02).

Positioning: a state is the transient counterpart of the `u` of
`platform/residual.jl`; `advance!(state, n)` is `n` applications of the step `G`;
`solve` stays the one-shot verb (= `init_state` + `advance!` + `solution`).

## Public surface

Defined in `src/platform/state.jl`; export status verified against
`src/Kraken.jl` (drift found and fixed here on 2026-09-17: five client hooks
are **unexported**, not re-exported as an earlier version of this map claimed):

- `AbstractSimulationState` — supertype of every client state. **Exported.**
- `StateSnapshot(; solver, schema_version, cycle, fields, series, scalars, identity,
  run_control, parameters, derived)` — plain host-side container. **Exported.**
  Scalars: `String`/`Bool`/`Int`/`Float32`/`Float64` (+ `nothing` in `identity`,
  `run_control`, `parameters` only). Array eltypes: `Float32`/`Float64`/`Int32`/
  `Int64`/`Bool`. Field and series names may contain `/`; scalar keys may not, and
  `keys` is reserved. The constructor rejects anything else.
- `CheckpointError` — every refusal; the message names the key or object. **Exported.**
- Client verbs, **exported**: `init_state(::Type{S}; kwargs...)`,
  `advance!(state, n; sample_final=false)`, `solution(state)`,
  `restore_state(::Type{S}, snap; backend, kwargs...)`,
  `update_parameter!(state, name, value)`, `updatable_parameters(::Type{S})`
  (default: empty `ParameterSpace`).
- Client verbs, **unexported** — call as `Kraken.<name>`: `snapshot(state)`,
  `validate_snapshot(::Type{S}, snap)` (default: no check), `at_boundary(state)`
  (default: `true`), `migrate(::Type{S}, snap, from_version)` (default: throws
  `CheckpointError`).
- Platform-owned, **exported**: `export_state(state)`, `check_compatible([S,] snap;
  solver, schema_version, identity, identity_defaults)`.
- Platform-owned, **unexported** — call as `Kraken.<name>`:
  `check_updatable(S, name, value)`, `require_boundary(state, what)` (throws
  `ArgumentError` when `!at_boundary(state)`). Internal companions (unexported, not
  part of the contract surface): `validate_content`, `check_finite`.

Invariant every client must hold (checked by the contract suite): once an
`advance!` has thrown mid-cycle (`at_boundary(state) == false`), the client's
`advance!` (any `n`, including 0) and `solution` refuse the state through
`require_boundary` without mutating it, and `export_state` refuses it with a
`CheckpointError`. There is no repair verb: recovery is a fresh `init_state` or a
`restore_state` / `load_checkpoint` from a known-good snapshot.

Defined in `src/io/checkpoint_hdf5.jl`, exported by `Kraken`:

- `CHECKPOINT_CONTAINER_VERSION` (= 1), `write_checkpoint(path, snap; keep_previous=true)`,
  `read_checkpoint(path)`, `checkpoint_info(path)`, `save_checkpoint(path, state)`,
  `load_checkpoint(::Type{S}, path; kwargs...)`.

## Call graph

```
save_checkpoint(path, state)
  ├─ export_state(state)                          [platform-owned]
  │   ├─ at_boundary(state) or CheckpointError    [client hook]
  │   ├─ snapshot(state) → StateSnapshot          [client hook, raw]
  │   └─ validate_content + check_finite          [refuse NaN/Inf in fields/series; scalars may carry Inf]
  └─ write_checkpoint(path, snap)
      ├─ validate_content + check_finite          [again: the dicts are mutable]
      ├─ h5open(path.tmp, "w"; libver_bounds=(v"1.10", v"1.10"))
      ├─ close, rename(path → path.prev) if keep_previous, rename(path.tmp → path)

load_checkpoint(S, path; kwargs...)
  ├─ read_checkpoint(path) → StateSnapshot        [any library error → CheckpointError]
  └─ restore_state(S, snap; kwargs...)            [client hook]
      ├─ check_compatible(S, snap; …)             [solver, schema (→ migrate), identity]
      ├─ validate_snapshot(S, snap)               [client hook: shapes, eltypes]
      └─ allocate on the backend, copy            [only after both checks passed]
```

## File layout (container version 1)

Root attributes `container_version`, `schema_version`, `solver`, `cycle`,
`kraken_version`, `julia_version`, `created_unix`. Groups `/fields` and `/series`
hold one chunked Fletcher32 dataset per entry (`/` in a name → nested groups).
Groups `/scalars`, `/identity`, `/run_control`, `/parameters`, `/derived` hold one
attribute per entry plus `keys`, a newline-joined scalar string listing every key;
a listed key without an attribute is `nothing`. Chunks: one slab along the last
dimension for ≥3D, whole columns up to ~1 MiB for 2D, ≤65 536 elements for 1D,
never above 1 GiB.

## Reads from

- `src/platform/contract.jl`: `AbstractSolution` (return type of `solution`).
- `src/platform/calibration.jl`: `ParameterSpace`, referenced in function bodies only
  (`calibration.jl` is included after `state.jl`; no signature may mention it).
- `HDF5.jl` (direct dependency since #26; libhdf5 1.14.x shared with `gmsh_jll`).

## Writes to

- `path`, `path * ".tmp"` (transient, same directory), `path * ".prev"` (one older
  generation). Nothing else; no global state. `write_checkpoint` never touches
  `path` when it refuses a snapshot or when the write fails.

## Backend constraints

`StateSnapshot` holds host `Array`s only: a GPU client copies device arrays to the
host in `snapshot` and uploads them in `restore_state(...; backend)`. No kernel is
launched from these two files. The host copy of a full 3D state is held in memory
during an export (no streaming writer yet). Bitwise equality of a restart is
established on CPU; it has not been measured on CUDA.

## Failure modes

- Without `libver_bounds=(v"1.10", v"1.10")`, libhdf5 1.14 writes superblock v0 with
  no metadata checksum: a flipped bit in an attribute (`cycle`, a physical
  parameter) reads back silently. `test/platform/state_contract_test.jl`
  ("attribute bit flip rejected") fails if the keyword is removed.
- Variable-length strings (any `Vector{String}` attribute or dataset) live in the
  HDF5 global heap, which is not checksummed even in the 1.10 format. Only scalar
  strings are written; do not add a string array to the layout.
- Fletcher32 needs an explicit chunk layout, and a chunk is capped at 4 GiB.
- `mv(tmp, path; force=true)` is not atomic on Julia 1.11 (it removes the
  destination first). Use `Base.rename`, with the temporary file in the same directory.
- `findfirst(!isfinite, a)` throws on an empty N-d array; `check_finite` loops instead.
- The dictionaries of a snapshot are abstractly typed (`Dict{String,Array}`): any
  per-element work on them needs a function barrier (see `_first_nonfinite`).
- A tolerance-based restart test passes with a piece of state missing; the contract
  suite compares bit for bit and perturbs every stored field as a negative control.

## Touch order

1. `docs/platform/07-STATE-CONTRACT.md` — the decision record, if the contract or
   the layout changes (a layout change bumps `CHECKPOINT_CONTAINER_VERSION`).
2. `src/platform/state.jl` — container, verbs, platform-owned checks.
3. `src/io/checkpoint_hdf5.jl` — layout, integrity, rotation.
4. `src/Kraken.jl` — include/export lines (choke file; edits serialized on `dev/platform`).
5. `test/platform/state_contract_suite.jl` — the reusable suite every client runs;
   `test/platform/state_contract_test.jl` — toy client and disk-layer cases.
6. A new client: its own state file under `src/drivers/`, its `schema_version`, and
   a test file that calls `run_state_contract_suite`.

## Clients

- `src/drivers/ehd_ec_state.jl` — `ECState` / `ECSolution` (2D electroconvection).
  Implements `init_state`, `advance!`, `solution`, `at_boundary`; `advance!` and
  `solution` open with `require_boundary`.
  `run_electroconvection_2d` (`src/drivers/ehd_ec.jl`) is the one-shot wrapper
  (`advance!(s, max_cycles; sample_final=true)`). Not implemented yet: `snapshot`,
  `restore_state`, `validate_snapshot`, `update_parameter!`, `updatable_parameters`,
  so `export_state` / `save_checkpoint` still throw for this client.
  `solution(::ECState)` recomputes derived buffers in place (`qfield`, `phi`, `Ex`,
  `Ey` to the values they already hold; `rho`, `ux`, `uy` are rewritten by the next
  cycle before being read). Bit-for-bit parity with the former monolithic driver and
  segment independence: `test/analytical/ehd_ec_split_parity_2d.jl` (reference copy
  in `test/reference/ehd_ec_legacy.jl`, test-only, never edited).
