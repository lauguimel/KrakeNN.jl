# Reproducing the PR79 moment-injector verification

These are two different verification suites, not one T=190 validation run.
They were prepared by Jiachen Zhao for #73/#79. This directory publishes the
actual frozen numerical scripts and gates plus portable execution wrappers.
No private MATLAB/Fortran implementation, paper, unpublished reference fields,
credentials, checkpoints or long-trajectory output is included.

## Evidence identity

| Suite | Original local harness commit | Aqua job | What it verifies |
|---|---|---|---|
| `restart/` | `490c53a544a3ff32440b22b91b8961abaa9743d7` | 27042625 | Both closures; CPU/disk exactness; independent GPU continuous-repeat calibration and 7+13 restart verification |
| `scalar/` | `7affbefe72d72c5fc0f32c999390ef0bf378cecd` | 26985556 | Actual integrated kernels; periodic/box analytical transients; ordered NEE control; low-tau and CPU1/CPU2/repeated-CUDA corner checks |

Both original runs loaded solver **41f5209d19f8cbb798a3027166d6022867675f6c**.
The original harness hashes belong to local, standalone run repositories, not
Kraken's Git history. They will not resolve as commits in this repository.
Instead, `ORIGINAL_SHA256SUMS` maps the published, byte-identical numerical
scripts, frozen protocols and synthetic reference tables to those run manifests:

```bash
cd benchmarks/validation/ehd_moment
sha256sum -c ORIGINAL_SHA256SUMS
```

The new packaging commit identifies this public copy. Historical protocols retain
their dates and then-current statements; they are not new claims about PR status.
Unchanged original files: restart `common.jl`, `calibrate.jl`, `check.jl`, protocol;
scalar `verify.jl`, `reference_kernels.jl`, both protocols and six CSV tables.
The reference module supplies equilibrium/reference algebra; `verify.jl` calls
the actual `Kraken._apply_charge_moment_box_2d!` for every boundary application.
It does not replace the production seam with the reference module's wall kernels.

Packaging changes only: replace site-specific donor-copy/PBS wrappers with the
scripts here; replace the Manifest's sole absolute Kraken path with `../source`;
make the independent scalar audit accept paths and explicitly check completeness
and the already-frozen periodic gates. No numerical acceptance threshold changed.
The original Manifest SHA256 was
`36693983a7e1c05a9eb5b2406567ec6cdd0d74a0a5ea7a5198abf614a8498764`;
all dependency versions/tree hashes are preserved. The scalar run used the same
dependency graph with a different absolute source path.

## Reproduce on Linux with an NVIDIA GPU

Requires Git, Bash/GNU coreutils, Julia **1.11.9**, a compatible NVIDIA driver,
and the pinned Julia environment (CUDA5.11.3, CUDSS0.6.5, KernelAbstractions0.9.42).
Original hardware was an H100. Calibration is specific to the runtime/hardware;
do not substitute the historical tolerance file into a different environment.
Use allocated compute nodes for Julia imports/JIT or numerical work on Aqua.

From a **clean committed** Kraken checkout containing this directory:

```bash
PACKAGE="$PWD/benchmarks/validation/ehd_moment"
# New directory, whose parent already exists. Never reuse an old run directory.
bash "$PACKAGE/prepare.sh" restart /absolute/path/new-restart-run \
    41f5209d19f8cbb798a3027166d6022867675f6c
cd /absolute/path/new-restart-run
export JULIA_CUDA_USE_COMPAT=false JULIA_PKG_PRECOMPILE_AUTO=0 JULIA_CPU_TARGET=generic
export JULIA=/absolute/path/to/julia-1.11.9/bin/julia
# Prepare on an allocated compute node; use a task-appropriate depot if desired.
"$JULIA" --startup-file=no --project=environment \
    -e 'using Pkg; Pkg.instantiate(; allow_autoprecomp=false)'
bash seal.sh
# On an existing suitable allocation:
bash run.sh
# OR from the login node, after returning from dependency preparation:
# qsub -v JULIA restart.sub
```

For scalar verification, substitute `scalar`, another new directory and
`scalar.sub`. The scalar job requests two CPUs; restart requests one. Each uses
one H100, 3GB, at most20min. `run.sh` starts CPU1/CPU2 scalar processes separately,
or calibration/verification restart processes separately. Default depot selection
is Julia's normal behavior; if using `JULIA_DEPOT_PATH`, also export it to PBS.
No package resolution/installation occurs inside the sealed numerical run.
Preparation records `SOURCE_SHA` and `HARNESS_SHA`; pre/post source/input hashes
and runtime `pathof(Kraken)` checks prevent silently using another checkout.
Omit the optional solver SHA only when deliberately testing the current committed
source; that produces a new run, not historical evidence for41f5209.

## Interpretation and output

Restart: `raw/tolerances.toml` is generated and sealed **before** verification.
There is no epsilon floor, retrospective enlargement or automatic retry.
`raw/status.toml`, `checks.toml`, `comparisons.toml` and `stages.toml` distinguish
acceptance from Poisson-seam diagnostics. Formal tolerance is2x the calibration
envelope with the fixed safety ceiling; all 306 acceptance checks must pass.
The original result also fitted the unmultiplied envelope. That stronger observed
property is not a new general acceptance gate. `audit_historical.py` audits that
specific historical finding; it is intentionally not part of the generic runner
and assumes the nonzero envelopes seen in27042625.

Scalar: `raw/result.toml` must report `PASS_INTEGRATED_SCALAR_ONLY` after both
processes succeed. The original two processes totalled754 passing assertions.
The six `reference/*.csv` tables are deterministic NumPy-generated scalar fields
and exact PDE values, **not private literature/MATLAB data**. Their producer and
original protocol are in `reference_generator/` (Python3 and NumPy). To regenerate,
copy that directory to a fresh scratch folder, then run `screen.py` followed by
`export_reference.py`; retain the frozen30s CPU cap. Floating-point/platform
changes may change last bits; Julia compares values at the frozen tolerances,
while the published original CSVs retain their exact hashes.

Audit scalar output independently with Python>=3.11 and NumPy:

```bash
python "$PACKAGE/scalar/audit.py" /absolute/path/scalar-run /absolute/path/new-audit
```

The low-tau variable-shear check is stability/parity only. Exact transient
ladders use prescribed constant drift. `moment=false` means an ordered NEE
control with charge recomputed before corner reads, **not bitwise equivalence
to the old concurrent production NEE schedule**. Original labels saying
"same-schedule legacy" mean the same ordered schedule between controls.

Job26985556's later, separate coupled exact-GPU restart stage had six failures;
its conditional pilot did not run. Those failures remain historical failures.
The scalar runner deliberately does not launch that obsolete coupled stage or
any pilot. Numerical restart acceptance instead uses the separately frozen
27042625 protocol above. Neither suite alone establishes saturated T=190,
Float32, rigid-wall, onset/hysteresis or whole-platform physical validity.
