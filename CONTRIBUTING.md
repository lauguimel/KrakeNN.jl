# Contributing to Kraken.jl

## Quick Start

```bash
# Install / update dependencies
julia --project -e 'using Pkg; Pkg.instantiate()'

# Run tests
julia --project test/runtests.jl

# Build docs locally
julia --project=docs docs/make.jl
```

## Project Structure

```
src/
  Kraken.jl              # Main module (exports)
  simulation.jl          # High-level simulation drivers
  lattice/               # D2Q9, D3Q19 lattice definitions
  kernels/               # GPU kernels (stream, collide, BCs, thermal, multiphase)
  rheology/              # Non-Newtonian rheology models
  refinement/            # Patch-based grid refinement
  io/                    # VTK output, .krk config parser
test/
  runtests.jl            # Test entry point
docs/
  make.jl                # Documenter + Literate.jl build
  refs.bib               # BibTeX references (DOI-verified)
  src/
    theory/              # Literate theory pages (.jl → .md)
    examples/            # Literate validation examples (.jl → .md)
    benchmarks/          # Literate benchmark pages (.jl → .md)
```

## How we work together on the repository

### The two permanent branches

| Branch | Role | Moves when |
|---|---|---|
| `main` | published versions only | a release is cut |
| `dev/platform` | integration line, where everything lands | continuously |

Clone and you land on `main`, which is always a released, tested state. **Work
starts from `dev/platform`**, not from `main`.

Everything else is a short-lived branch: one unit of work, named after it
(`fix/...`, `feat/...`, `val/...`), merged into `dev/platform`, then deleted.

### Merge early, merge often — the cost is not linear

The effort a merge costs grows much faster than the time the branch spent
apart, because each new commit on one side can conflict with several on the
other. Measured on this repository:

| Branch | Divergence | Cost when merged |
|---|---|---|
| `fix/krk-public-entry-points` | 3 commits, same day | no conflict |
| `chore/merge-main-into-platform` | 36 commits | 1 conflict |
| `dev/ve-3d-sphere` | 357 commits | 17 conflicts, a dedicated session |
| `dev-viscoelastic` | 238 commits | never attempted; branch frozen instead |

A branch should live days, not months.

### Bring the trunk into your branch before bringing your branch into the trunk

Merge `dev/platform` into your branch first, resolve conflicts there, run the
tests, and only then open the pull request. Conflicts get resolved in isolation
where breaking things costs nothing, and the final integration becomes trivial.

### Where conflicts actually happen

Separate modules almost never collide — two people working in `src/kernels/`
and `test/analytical/` will not meet. Conflicts concentrate in the few files
everyone must touch:

- `src/Kraken.jl` (the export list)
- `src/simulation_runner.jl`
- `src/io/krk/`
- `test/runtests.jl`
- `Project.toml`

Edit those one person at a time, in a small dedicated commit, merged quickly.
Do not let changes to them accumulate on a long-lived branch.

### Daily routine

1. Pull `dev/platform` before starting.
2. One branch per issue, named after it.
3. Push your branch daily, even unfinished — it is a backup and it makes the
   work visible.
4. Before opening a pull request: merge `dev/platform` in, run
   `julia --project test/runtests.jl`, state the result in the pull request.
5. Merge within days.

### Retiring a branch

A branch that will not be finished is frozen, not deleted: annotate a tag
`archive/<name>-<YYYY-MM>` describing what it held and why it stopped, then
delete the branch. The commits stay reachable forever and the branch list stays
readable. A branch left lying around reads as active work; a tag reads as
deliberately closed.

## Conventions

- **Code and comments**: English
- **Commit style**: conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`)
- **Docstrings**: Google-style adapted for Julia
- **Documentation**: literate (Literate.jl) — `.jl` files with `# ` markdown comments
- **Equations**: LaTeX in ` ```math ` blocks
- **References**: all with DOI in `docs/refs.bib`
- **Examples**: self-contained, runnable, with validation against analytical/reference solutions

## Adding New Physics

1. Create kernel(s) in `src/kernels/`
2. Export functions in `src/Kraken.jl`
3. Add a simulation driver in `src/simulation.jl` (or appropriate `src/drivers/` file)
4. Write tests in `test/`
5. Add a theory page in `docs/src/theory/`
6. Add a validation example in `docs/src/examples/`
