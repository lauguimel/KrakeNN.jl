# AGENTS.md — working rules for Kraken.jl

This file is read automatically by coding agents (Codex, Claude Code) at the
repository root. It defines how work is done here. Follow it literally.

Maintainer: Guillaume Maîtrejean (@lauguimel).

---

## 1. Ownership and review — read this before editing anything

Everything enters through a pull request, including the maintainer's own work.
`main` and `dev/platform` are protected: no direct push, at least one approving
review, and no force-push. This is enforced by GitHub, not by goodwill.

Contributors may propose changes anywhere in the repository, `src/` and `ext/`
included. What is reserved is not the right to write the code, it is the right
to merge it.

| Path | Who must approve | Before you start |
|---|---|---|
| `src/`, `ext/` | maintainer, required by `.github/CODEOWNERS` | open an issue and get the approach agreed |
| `docs/src/theory/`, `docs/agent/`, `docs/spec/` | maintainer | open an issue |
| `test/`, `benchmarks/`, `docs/src/users/benchmarks/` | any reviewer | go ahead |

GitHub applies the `src/`/`ext/` rule automatically: a pull request touching
those paths cannot be merged without the maintainer's explicit approval, and
that approval is dismissed if new commits are pushed afterwards.

### Before writing code that touches `src/`

Open an issue first and wait for the approach to be agreed. Not for permission
to be useful — to avoid building something that duplicates an existing seam or
introduces a second way of doing what the platform already does. `src/` carries
the platform contract, the automatic-differentiation seam and the GPU kernels;
a fix that works on one case routinely breaks three others that are not visible
from where the bug appeared. That risk is handled by review and by tests, not
by keeping contributors out of the directory.

### One pull request, one subject

A pull request implements one issue. If it fixes a bug and also renames three
functions and also adds a benchmark, it will be sent back to be split — not
because the extra work is bad, but because a change nobody can finish reading
is a change nobody can review honestly. Prefer several small pull requests over
one large one; they merge faster.

### Reporting a bug you are not fixing

A failing test is a valid deliverable on its own. Write it, put it under
`test/analytical/`, and open an issue pointing at it.

Mark it `@test_broken`, not `@test`. Julia treats `@test_broken` as a known,
documented failure: the suite stays green, so the test can be merged
immediately instead of living in a stale branch, and the day the bug is fixed
it reports `Unexpected Pass` — which forces whoever fixed it to promote the
test back to `@test`. A deliberately failing `@test` cannot satisfy the
green-suite gate below, so it has nowhere to go.

---

## 2. Branches and submissions

- Never commit to `main`. Never commit to `dev/platform`. Both are protected.
- Integration branch is `dev/platform`. Everything targets it.
- One branch per unit of work, named after what it does:
  - validation case: `val/<case-id>` — e.g. `val/TH-001R`
  - bug report with a failing test: `bug/<short-name>`
  - bug fix: `fix/<short-name>`
  - new capability: `feat/<short-name>`
  - build, CI or tooling: `ci/<short-name>`
- Open a pull request into `dev/platform`. The maintainer reviews and merges.
- Commit messages in English, conventional style: `test:`, `fix:`, `docs:`,
  `feat:`, `chore:`.
- Never mention an AI assistant, a model name, or a co-author trailer in a
  commit message, pull request, issue, or comment.

---

## 3. Running the test suite

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project test/runtests.jl
```

Environment gates (all default to off):

| Variable | Effect |
|---|---|
| `KRAKEN_TEST_HEAVY=true` | adds the long validations (Ghia cavities, multigrid up to 512²) |
| `KRAKEN_INCNS_ONLY=true` | runs only the incompressible Navier–Stokes and solver-services tier |
| `KRAKEN_AD_ONLY=true` | runs only the automatic-differentiation tier |

On Linux x86_64, run the five Enzyme-driven files (`platform/calibration_test.jl`,
`platform/calibration_nufield_test.jl`, `ad/test_ad_sensitivity.jl`,
`ad/test_ad_ve_sensitivity.jl`, `ad/test_ad_ve_fd_check.jl`) with production bounds checking,
`Pkg.test(julia_args=["--check-bounds=auto"])`, one file at a time through
`KRAKEN_ONLY`. Under the `Pkg.test` default `--check-bounds=yes`, Enzyme's
reverse mode miscompiles them: segfault or LLVM crash on Linux, wrong gradient
on macOS (#41). CI does exactly this in the `full` job.

A pull request is not reviewable until the full suite passes on the final head
of its branch. Pull requests into `main` or `dev/platform` automatically run
only the `quick` tier: Julia 1.11 and 1.12, with the automatic-differentiation
and 3D viscoelastic tiers skipped. Start the full suite yourself after your
last push, in either of two ways:

- on GitHub: `gh workflow run CI.yml --ref <your-branch>`, leaving the `only`
  and `check_bounds` inputs at their defaults (this needs write access to the
  repository; otherwise ask the maintainer to start it). The `full` job applies
  the bounds-checking rule above to the Enzyme-driven files by itself;
- locally, with the commands above and the same rule.

A green dispatched run is enough; a local run is not required. A run started
before your last push does not count for the new head. Post the run link in
the pull request. The full suite also runs on every merge into `dev/platform`.

State in the pull request body which gates you ran, on which backend and which
precision. A known failure carried deliberately must be `@test_broken` and named
as such in the body — see section 1.

---

## 4. Adding a validation case

This is the main way to contribute. The shape is fixed:

1. **One file per case**, in `test/analytical/`, named after the case
   identifier: `test/analytical/<case-id>.jl` (e.g. `AXI-001.jl`). Keep your
   own identifier scheme — it becomes the project's.
2. **Register it** by adding an `include(...)` line in `test/runtests.jl`,
   inside the matching `@testset`.
3. **Frozen gates.** Every quantitative claim is asserted against a threshold
   that is written down in the file and does not move. State next to each gate
   where the number comes from (analytical solution, published reference with a
   citation, or a measured baseline with the date and the machine). A gate that
   is loosened to make a test pass is a regression, not a fix.
4. **Reference data** (published values, tables digitised from a paper,
   reference fields produced by another code) goes in `test/reference/`, as
   plain CSV with a header line, plus a short `README.md` in the same directory
   stating the provenance: who produced it, with what code, what parameters,
   what date.
5. **Figures** go in `benchmarks/results/<topic>/`, with a `README.md` giving
   the command that regenerates them.
6. **Both precisions, both backends** where the case allows it: CPU Float64,
   CPU Float32, GPU Float64, GPU Float32. Parity between CPU and GPU is itself
   a gate.

---

## 5. GPU

- Long runs are GPU runs. Use `CUDABackend()`. Do not benchmark on CPU.
- CPU is acceptable for short unit tests (under about thirty seconds).
- On the QUT Aqua cluster, always export `JULIA_CUDA_USE_COMPAT=false` before
  launching Julia — the compute nodes need the system CUDA driver. Request GPUs
  with `gpu_id=H100` or `gpu_id=A100`. The scheduler is PBS Pro, not Slurm;
  containers are Apptainer, not Docker.

---

## 6. Configuration files (`.krk`)

Kraken simulations can be driven by a `.krk` configuration file parsed by
`src/io/kraken_parser.jl`, run with
`run_simulation("path/to/case.krk")`. When a validation case can be expressed
as a `.krk` file, prefer that over a hand-written Julia driver: it is what
users will actually run, so it is what should be tested.

Fixture naming convention, which carries meaning:

- `<flow>_<layout>.krk` — production configuration, default settings.
- `<flow>_<layout>_debug.krk` — sets a non-default internal mode. A failure
  here may be intentional behaviour of that mode, not a bug.

Do not change the semantics of an existing `.krk` keyword. That is a breaking
change for every user script and needs a decision from the maintainer first.

---

## 7. Conventions

- Code, comments, docstrings, commit messages, documentation: **English**.
- Docstrings: Google style adapted to Julia.
- Equations in documentation: LaTeX inside ` ```math ` blocks.
- Every bibliographic reference carries a DOI and lives in `docs/refs.bib`.
- Documentation pages are written with Literate.jl: a `.jl` file whose comments
  starting with `# ` become Markdown.

---

## 8. Things that waste days here — verify before assuming

- **Check which code path actually runs** before forming a hypothesis about a
  kernel bug. Several past investigations correctly analysed a function that
  the failing case never calls. Add a log line at the entry of every candidate
  kernel, run the failing case for five steps, and read the trace first.
- **Check the backend.** The same logical operation lives in different files
  for CPU and for GPU. State the backend explicitly in any bug report.
- **Check the mode.** A `.krk` fixture may select a strict internal mode whose
  observable behaviour differs from the default on purpose.

---

## 9. Asking

If a task requires touching an owned directory, or changes a `.krk` keyword, or
loosens a frozen gate: stop and open an issue instead. Describe what you wanted
to do and why the current structure blocks it.
