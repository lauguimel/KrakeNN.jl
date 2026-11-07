# Branch Contract — dev/platform

Date: 2026-07-22
Branch: `dev/platform`
Worktree: `/Users/guillaume/Documents/Recherche/kraken/Kraken.jl-platform`

## Branch role

`dev/platform` is the **canonical integration base** for the Kraken platform
layer (post-pivot 2026-06). It owns:

- the platform contract — `src/platform/{contract,solution,sample,observe,residual,calibration}.jl`
  (6 nouns, `Capability` enum, verbs `solve/sample/observe/predict/residual/adjoint_vjp/fit`);
- the FVFD co-solver — `src/fvfd/` + `src/methods/inc_ns/` (IncNS, SIMPLE/SIMPLEC,
  Poisson MG in `src/solve/poisson_mg.jl`) + `src/methods/scalar_transport/`;
- the AD↔platform seam — `src/platform/residual.jl` delegating to `src/ad/`;
- calibration (`ParameterSpace`/`loss`/`fit`) + `ext/KrakeNNOptimExt.jl` (Optim weakdep).

Integration model: **merge-on-green**. Feature work happens on short-lived
`feat/platform-*` branches (own worktrees, deleted on merge); `dev/platform`
only ever receives branches whose validation gate is green. Rebase the feature
branch on `dev/platform` before merging; no long divergence.

## Choke zones (serialized lane)

The following files are contention hotspots. Edits to them are **serialized**:
one branch/session at a time, registered in the mandate before touching them.

```text
src/KrakeNN.jl        # module includes + exports
src/io/krk/          # .krk DSL parser / fixtures semantics
Project.toml         # deps, weakdeps, compat
```

New files (bricks) are parallel-safe; choke files are not.

## LOC budget

- Standard mission: **<= 500 lines** changed (added + modified) per merge.
- Hard cap: **700 lines** — beyond that, split the mission before merging.
- Generated files, test fixtures, and docs pages do not count toward the budget,
  but must be listed in the merge summary.

## GPU rules

- No dynamic allocation inside kernels (no `push!`, no array construction,
  no closures capturing host state that allocates).
- Kernels are backend-generic (KernelAbstractions); no `CUDA.`-specific calls
  in shared kernel code — backend-specific code goes in the backend layer or an ext.
- Any kernel change needs at least a CPU test; GPU parity (CUDA) validated
  before merge when the touched path has a GPU route.

## Validation gate (merge-on-green)

A branch merges into `dev/platform` only when ALL of:

1. **Tests** — `Pkg.test` (or the targeted `test/test_*.jl` set for the touched
   area) exits green.
2. **Parity** — platform-path results match the legacy/reference path on the
   touched physics (existing parity tests stay green; new seams add one).
3. **Doc-lint** — the docs build's implication-map lint gate passes: every
   public module touched has a linting `docs/agent/<mod>-implication.md`.

## Forbidden

- **No `git push`** from any agent session — pushing is a human gate
  (see kraken-git-ops for the verified push command when the human asks).
- **No `git branch -D`** (force-delete) — only `-d` after a confirmed merge;
  force-deletion requires explicit user confirmation.
- No history rewrite (`rebase -i`, `commit --amend` on merged commits,
  `push --force`) on `dev/platform`.
- No merging with a red or skipped validation gate ("it's just docs" included —
  doc-lint is part of the gate).

## Commit rules

- Conventional commits, English, no AI/Claude mentions, no `Co-Authored-By`.
- The Boss (top-level session) commits; executors (Codex/subagents) never commit.
- Stage only files inside the mission's declared edit zones; never stage
  `Manifest.toml` or unrelated dirty files.
