# Viscoelastic V&V inventory — 2026-05-23

Scope: phase-A inventory only. No commitment to a hierarchy yet.
Sources: `test/`, `bench/viscoelastic_logfv/`, `bench/viscoelastic_audit/`,
`bench/rheotool/`, `bench/scratch/`, plus external trees
`/Users/guillaume/Documents/Recherche/Codes CFD/rheotool/` and
`/Users/guillaume/Documents/Recherche/Codes CFD/basilisk/`.

Robustness rubric used below:
- HIGH: deterministic analytic reference, machine-precision or O(dx²) tolerance, runs in unit-test suite, no GPU-only dependency.
- MED: analytic or canonical-numerical reference but with mixed concerns (multiple kernels exercised) or coarse tolerance (~1e-3).
- LOW: ad-hoc script, manual eyeballing, undocumented pass criterion, or known-flaky reference.

---

## 1. Existing Kraken tests (viscoelastic)

| Path | What it tests | Reference | Pass/fail criterion | Robustness |
|------|---------------|-----------|---------------------|------------|
| `test/test_viscoelastic.jl` | `evolve_stress_2d!` in pure shear (imposed `ux=γ̇·y`) | Analytic Oldroyd-B steady: τ_xy=ν_p·γ̇, τ_xx=2λν_p·γ̇² | `rtol=1e-3` at γ̇∈{1e-3,1e-2,5e-2} | HIGH |
| `test/test_viscoelastic_equations.jl` | OldroydB stress closure τ=G(C−I); CDE source incl. C·div(u) | algebraic identity | exact / 1e-12 | HIGH |
| `test/test_viscoelastic_equation_patch_ladder.jl` | direct-C source vs log-conformation source (Loewner) consistency | self-consistency w/ analytic SPD log | atol 1e-12 | HIGH |
| `test/test_viscoelastic_coupling.jl` | Hermite stress source coupling on prescribed τ_p in body-force Poiseuille | analytic shear profile w/ prescribed τ_p_xy(y) | velocity match (rtol untracked) | MED (prescribed τ_p ≠ self-consistent) |
| `test/test_viscoelastic_force_accounting.jl` | Pressure-tensor / 2nd-moment increment from Hermite source kernel | analytic ∇·τ_p projection on D2Q9 | atol ~1e-12 | HIGH |
| `test/test_viscoelastic_patch_tests.jl` | Local LBM patches: CNEBB Couette wall, momentum closure | analytic Couette + momentum conservation | atol ~1e-12 | HIGH |
| `test/test_logconformation.jl` | C↔Ψ round-trip; Ψ=0⇒C=I; low-Wi direct-C vs log-conf agreement on cylinder | self-consistency + tiny cylinder run | atol 1e-12 + run match | MED (cylinder leg is full pipeline) |
| `test/test_logconformation_3d.jl` | 3D Ψ↔C exp/log round-trip on six independent components | analytic exp/log identity | atol/rtol 2e-12 | HIGH |
| `test/test_logfv_frozen_channel_cde.jl` | Log-FV CDE driver on frozen velocity (Couette+Poiseuille), prescribed steady ∇u | analytic ∇u; checks min_c_eig & gradient errors | 1e-12 on grads, min_c_eig>0.8 | HIGH (this IS the "imposed velocity" path) |
| `test/test_viscoelastic_logfv_patch_ladder.jl` | Log-FV primitives: sym2 min-eig, exp/log, mat exp, SPD reconstruction | algebraic / Mathematica reference | atol/rtol 1e-12 | HIGH |
| `test/test_viscoelastic_logfv_gpu_smoke.jl` | Log-FV cylinder driver runs on Metal/CUDA backend (smoke only) | none — just non-NaN | no NaN, monotone steps | LOW (smoke) |
| `test/test_fvfd_operators_2d.jl` | FVFD operator library: BC aliases, divergence, gradient, stress accumulator | analytic discretisation identities | atol 1e-12 | HIGH |
| `test/test_polymeric_drag_geometry.jl` | `precompute_q_wall_cylinder` cut links analytical per D2Q9 direction | analytic q at each cut | exact (==) on counts + closed form | HIGH |

Reading: 7/13 HIGH, 4/13 MED, 1/13 LOW (smoke), 1/13 MED-but-pipeline. The 0D + frozen-velocity unit tests are already strong; what is missing is a **driver-level** cheap pass-criterion test (everything pipeline-flavoured lives in `bench/`, not `test/`).

---

## 2. Existing Kraken benches (viscoelastic)

`bench/viscoelastic_logfv/*.jl` (production log-FV driver benches):

| Path | What it does | Reference | Pass criterion | Robustness |
|------|--------------|-----------|----------------|------------|
| `analyse_cavity_guo_vs_fd_2d.jl` | 1×1 lid-driven cavity Oldroyd-B, Guo Hermite vs FD stress | rheoTool cavity Wi=1 β=0.5 | gap to rT u, τ_xx peak | MED |
| `analyse_cavity_remismatch.jl` | Same cavity, Re mismatch audit | rT cavity | manual | LOW |
| `analyze_logfv_field_dump.jl` | Post-mortem reader of log-FV field dumps | n/a | n/a | LOW (analysis tool) |
| `analyze_logfv_gradient_dump.jl` | Post-mortem reader of gradient dumps | n/a | n/a | LOW |
| `cylinder_cd_convergence.jl` | Confined cylinder Cd convergence sweep | Liu 2025 (now suspect — see §8) | Cd table | LOW (reference suspect) |
| `logfv_cylinder_cd_convergence.jl` | Same but log-FV driver | rheoTool cylinder + Liu | Cd convergence + R sweep | MED |
| `run_bsd_dual_path_diagnostic_2d.jl` | BSD on/off split path diagnostic | self-comparison | dual-path equality | MED |
| `run_bsd_kinetic_audit_2d.jl` | Kinetic 2nd-moment audit BSD vs no-BSD | analytic moment | residual table | MED |
| `run_cavity_corner_artifact_2d.jl` | Corner-singularity probe in cavity | n/a | manual | LOW |
| `run_cavity_oldroydb_vs_rheotool.jl` | Full lid-cavity vs rheoTool | rT cavity | L2 vs rT field | MED |
| `run_constitutive_0d_vs_rheotest.jl` | **0D constitutive vs analytic + rheoTestFoam** | analytic Oldroyd-B start-up + rheoTestFoam | trajectory match | HIGH (closest to a clean L0+ test) |
| `run_contraction41_oldroydb_vs_rheotool.jl` | 4:1 contraction vs rT | rT Contraction41 Oldroyd-BLog | corner-vortex length | MED |
| `run_cyl_bigsweep_v2_2d.jl` | Wi×R cylinder DOE sweep | rT + Liu (suspect) | Cd table | MED |
| `run_cyl_cd_convergence_baseline_2d.jl` | Cd grid convergence baseline | Liu | order p ≥ 1.8 | LOW (Liu suspect) |
| `run_cyl_cd_convergence_bsd_off_2d.jl` | BSD-off variant | self | dual-path | MED |
| `run_cyl_cd_convergence_v2_bsd{0,0p5,1}_2d.jl` | BSD fraction sweep | self + rT | Cd vs BSD frac | MED |
| `run_poiseuille_imposed_stress_2d.jl` | **Body-force Poiseuille with prescribed τ_p** | analytic body-force balance | velocity match | HIGH |
| `run_poiseuille_polymer_analytical_2d.jl` | **Body-force Poiseuille, full pipeline** | Waters & King analytic | u(y), τ_xy(y), N1(y) | HIGH |
| `run_quantitative_simple_ladder.jl` | "Simple" L0→Lx convergence ladder, multi-step | analytic | L2 vs analytic | MED |
| `run_rheotool_frozen_replay_2d.jl` | Replay rT velocity field, evolve only polymer | rT τ field | τ L2 vs rT | HIGH (closest to "imposed-velocity → recover stress") |
| `run_rheotool_frozen_replay_cavity_2d.jl` | Same on cavity | rT cavity τ | τ L2 | HIGH |
| `run_simple_validation_outputs.jl` | Bundled dashboard producer | n/a | n/a | LOW |
| `run_wall_stencil_audit_2d.jl` | Wall-stencil unit on confined geometry | analytic | atol | MED |

`bench/viscoelastic_audit/*.jl` (staircase / step-by-step isolation):

| Path | What | Reference | Notes |
|------|------|-----------|-------|
| `bsd_analytical_ladder_2d.jl` | BSD analytical ladder | analytic | step diagnostic |
| `common.jl` | Shared analytic helpers (Waters & King, Liu Eq.62) | papers | reused across steps |
| `run_cyl_embedded_drag_newtonian_diag_2d.jl` | Newtonian cylinder embedded drag diag | Schaefer-Turek | sanity baseline |
| `run_kraken_vs_rheotool_tau_compare.jl` | Direct τ comparison Kraken vs rT | rT | L2 per component |
| `run_m29c_postmortem_scalar_2d.jl` | Per-cell scalar postmortem on M29c | self | trace tool |
| `run_poiseuille_bsd_pathmatrix_2d.jl` | Path matrix for BSD on Poiseuille | analytic | matrix table |
| `run_poiseuille_bsd_trace_2d.jl` | BSD on/off trace | self | path |
| `smoke_sphere_metal.jl`, `step4_sphere_metal.jl` | 3D sphere Metal smoke | none/n/a | smoke |
| `step1_bgk_guo.jl` … `step2_trt_hermite.jl` | Minimal-baseline → TRT staircase (see audit README) | analytic Poiseuille Oldroyd-B (Liu Eq.62) | order p ≥ 1.8 |
| `step1b_profile_dump.jl`, `step1c_analytic_wall.jl`, `step1d_cnebb_wall.jl` | Profile + wall variants | analytic | order |

`bench/scratch/m*` — ad-hoc forensic dirs and CSVs (M22/M23/M26b/M28/M29/M30/M32) tied to past sessions; no canonical pass criterion, useful only as evidence trail. Listed but flagged LOW for V&V.

---

## 3. rheoTool available cases

Path: `/Users/guillaume/Documents/Recherche/Codes CFD/rheotool/rheoTool/of90/tutorials/`

`rheoFoam` (full N-S + constitutive) tutorials:
- `Aneurysm/Channel2D_VE`, `Channel3D_VE`, `tube_GNF`
- `Cavity/Oldroyd-BLog`
- `Channel/Oldroyd-BLog` — planar Poiseuille viscoelastic
- `Contraction41/Oldroyd-BLog`
- `CrossSlot/{Oldroyd-BLog, Oldroyd-BRootk, Oldroyd-BSqrt, PTTLog}`
- `Cylinder/Oldroyd-BLog`
- `fluidDamper/`
- `OtherTests/HerschelBulkley` (GNF, not VE)

`rheoTestFoam` tutorials (one-cell, prescribed gradient, constitutive-only):
- `FENE-CR`
- `HerschelBulkley`
- `multimode_XPomPom`

"Imposed velocity → constitutive only" capability: **YES, dedicated solver `rheoTestFoam`** (`of90/src/solvers/rheoTestFoam/rheoTestFoam.C`). Description from header: "*returns the principal components of the extra-stress tensor, given the velocity gradient tensor, which is defined by the user. Two modes: ramp (several shear-rates up to steady-state) and non-ramp (single shear-rate over time). Mesh is a one-cell unitary cube.*" The solver manipulates BCs internally to produce the requested γ̇. Project has a sister case `bench/rheotool/rheotest_oldroydb/` that we already pair with `bench/viscoelastic_logfv/run_constitutive_0d_vs_rheotest.jl`.

Cases especially relevant for Kraken V&V (ranked by reuse cost vs information):
1. `rheoTestFoam/FENE-CR` (template; clone with Oldroyd-B) — **L0+ constitutive ground truth**.
2. `rheoFoam/Channel/Oldroyd-BLog` — planar Poiseuille; closed-form Waters & King; ideal L1.
3. `rheoFoam/Cavity/Oldroyd-BLog` — lid-driven, Fattal-Kupferman geometry; ideal L3.
4. `rheoFoam/Contraction41/Oldroyd-BLog` — corner-vortex length canonical case; L4.
5. `rheoFoam/Cylinder/Oldroyd-BLog` — confined-cylinder Cd; L4-bis (Kraken's flagship benchmark — must keep alive).
6. `rheoFoam/CrossSlot/{Oldroyd-BLog,PTTLog}` — extensional/elastic-instability — L4+.

Bench tree `bench/rheotool/` already contains pre-computed cases used as Kraken reference (cavity, contraction41, cylinder Wi sweep, rheotest_oldroydb), plus a shrunk-15R variant for confined-cylinder convergence.

---

## 4. Basilisk available cases

Path: `/Users/guillaume/Documents/Recherche/Codes CFD/basilisk/src/`

Viscoelastic source: `log-conform.h`, `fene-p.h` (FENE-P), plus `viscosity.h`/`viscosity-embed.h`.

Validation cases in `src/test/`:
| File | Geometry | Analytic reference (in source comments) |
|------|----------|------------------------------------------|
| `poiseuille-oldroydb.c` | Transient planar Poiseuille, Oldroyd-B (and FENE-P via sister `.ref`) | **Waters & King (1970)** — closed-form U(Y,T) with complex β_n |
| `lid-oldroydb.c` | Lid-driven cavity, time-ramped wall vel. (8(1+tanh(8(t−0.5)))x²(1−x)²), β=0.5, Wi=1 | **Fattal & Kupferman (2005)** — `.kinetic` + `.ux` + `.uy` reference dumps |
| `viscodrop.c` | Two-phase visco drop | Figueiredo `.figueiredo` reference + `.interface` |
| `couette.c`, `couette-gotm.c` | Simple Couette (Newtonian sanity baseline) | analytic linear |
| `poiseuille.c`, `poiseuille45.c`, `poiseuille-periodic.c`, `poiseuille-axi.c` | Newtonian Poiseuille variants (sanity) | analytic parabolic |

Each test ships a `.ref` file (reproducible regression target) and many include analytic comparison plots (`.plot` gnuplot scripts). Basilisk's policy: every solver in `src/` has at least one analytic-comparison test in `src/test/`. This is the discipline pattern Kraken should mirror.

---

## 5. Cheap-test design candidates (≤ a few cells per direction)

Trade-offs flagged inline.

### 5.1 Planar Poiseuille, periodic in x, body-force driven, thin slice
- Proposed grid: **Nx=4, Ny=32–64** (periodic x kills inlet noise; ≥32 in y to resolve the parabola + Hermite source gradient).
- Reference: Waters & King (1970) transient + steady-state parabola.
- Walls: HWBB or LI-BB at y=1, y=Ny (Kraken already has `evolve_stress_2d!` + Guo).
- Existing artefact: `run_poiseuille_polymer_analytical_2d.jl` already implements this — promote it to a unit test by capping Ny=32 and walltime budget ~10 s on CPU.
- Pros: cheap, analytic ref, exercises full Kraken pipeline (collision + stream + wall BC + Hermite source + CDE).
- Cons: a steady-state-only variant masks transient sign errors that Waters & King catches.

### 5.2 Symmetric half-channel (only top half + symmetry plane at y=0)
- Trade-off: cost halved. **Pitfall**: symmetry BC for the polymer stress is non-trivial. The condition is τ_xy=0 and ∂y τ_xx=∂y τ_yy=0 at the plane. LBM ghost-cell mirror does NOT impose this automatically for the polymer kernel; the conformation C must be mirrored as (C_xx, −C_xy, C_yy). Flag: open question whether `evolve_stress_2d!` reads ghost cells, and whether the FV gradient stencil supports symmetry.
- Recommendation: **do NOT use symmetry for L0+ tests**. Saves 2× but introduces a class of bugs the cheap-test should be detecting.

### 5.3 Imposed-velocity (constitutive-only) unit test
- **Status: hook EXISTS for the log-FV path.** `run_viscoelastic_logfv_frozen_channel_cde_2d` (used by `test/test_logfv_frozen_channel_cde.jl`) takes `flow=:couette|:poiseuille`, fills u/v from the analytic profile, freezes them, and evolves only the conformation. This is exactly the desired hook.
- Hook also EXISTS implicitly for the LBM path via `test/test_viscoelastic.jl` (caller fills `ux` from `γ̇·(j-0.5)`, calls `evolve_stress_2d!` 50 000 times). Less abstracted than the log-FV one but functional.
- What's MISSING: a clean public API like `frozen_velocity_constitutive!(τ_p, ux, uy, model; substeps, tol)` that returns the steady-state τ_p for an arbitrary user-supplied (ux,uy) on an arbitrary grid. Would centralise the pattern.
- Proposed minimal: 4×32 cell grid, prescribe ux = γ̇·y, run 5 000 substeps, compare τ_xx/τ_xy/τ_yy to analytic — same pattern as `test_viscoelastic.jl` but parametrised by `BackendSpec`.

### 5.4 Inverse / back-force test (impose τ_p, recover ∇·τ_p)
- **Status: hook EXISTS** via `test/test_viscoelastic_force_accounting.jl` which prescribes τ_p and checks the 2nd-moment increment of the LBM source. The FV analogue would call `fvfd_div_stress!` (in `src/fvfd/operators_2d.jl`) on a prescribed τ_p field and compare to analytic ∇·τ_p. The FVFD test (`test_fvfd_operators_2d.jl`) already covers divergence at 1e-12 for generic fields but does NOT yet have a polymer-stress-named convenience entrypoint. Wrap-up: low effort.

### 5.5 Couette start-up (transient)
- Grid: 4 × 32, walls top/bottom, top wall starts moving at t=0.
- Reference: Oldroyd-B transient analytic (closed form with damped oscillation).
- Pros: catches transient sign errors that steady-state Poiseuille misses; cheap.
- Cons: needs a transient analytic reference (Liu 2025 Eq. ~63 or Bird-Armstrong-Hassager textbook).

### 5.6 4:1 contraction MVP
- Minimum cells for a sane qualitative test (corner-vortex length): ~120×80 (channel sides 4H + 4·H_c) — NOT cheap.
- Reference: rheoTool/Oldroyd-BLog Contraction41 case (already in `bench/rheotool/contraction41_oldroydb_log/`) + Alves et al. (2003) corner-vortex length tables.
- Not an L0+ candidate; this is L4.

---

## 6. Recommended L0 → L4 hierarchy (proposed, NOT committed)

Cost = wall-clock budget on a single Metal M3 Max (Float32) GPU or single CPU core. Order is ascending complexity.

| Level | Test | Reference | Cost target | Builds on |
|-------|------|-----------|-------------|-----------|
| **L0** | sym2 exp/log + Ψ↔C round-trip + eigen | algebraic | < 1 s | already in `test_logconformation*.jl`, `test_viscoelastic_logfv_patch_ladder.jl` |
| **L0** | OldroydB stress closure τ=G(C−I) | algebraic | < 1 s | `test_viscoelastic_equations.jl` |
| **L1** | Constitutive-only (frozen u): pure shear, steady-state | analytic τ_xy, τ_xx | < 2 s | `test_viscoelastic.jl`, `test_logfv_frozen_channel_cde.jl` |
| **L1** | Constitutive-only: Poiseuille frozen u, steady-state | analytic τ_xy(y), N1(y) | < 5 s | `test_logfv_frozen_channel_cde.jl` (already there for couette + poiseuille) |
| **L1** | Inverse: prescribed τ_p, divergence operator check | analytic ∇·τ_p | < 2 s | `test_viscoelastic_force_accounting.jl` + new FVFD wrapper |
| **L2** | Planar Poiseuille body-force, steady-state, full pipeline | Waters & King steady (parabola + τ_xy linear + N1 quadratic) | < 30 s | `run_poiseuille_polymer_analytical_2d.jl` (promote to test) |
| **L2** | Couette start-up, full pipeline | Oldroyd-B transient analytic | < 30 s | new |
| **L3** | Planar Poiseuille body-force, **transient**, full pipeline | Waters & King transient (oscillation envelope) | < 2 min | new |
| **L3** | Lid-driven cavity, β=0.5 Wi=1 Fattal-Kupferman | rheoTool cavity dump + Basilisk `lid-oldroydb` `.ref` | ~5 min (32²) | `run_cavity_oldroydb_vs_rheotool.jl` |
| **L4** | Confined cylinder Cd, Re=1 Wi sweep | rheoTool cylinder Wi sweep (Liu suspect; cross-check Alves-Oliveira 2003) | 30 min–1 h | `logfv_cylinder_cd_convergence.jl` |
| **L4** | 4:1 contraction corner-vortex length | rheoTool contraction41 + Alves 2003 | 1 h | `run_contraction41_oldroydb_vs_rheotool.jl` |

Notes:
- L0/L1 belong in `test/runtests.jl` (CI-cheap).
- L2 should be runnable via `julia bench/viscoelastic_validation/run_L2.jl` from a single command in ≤ 1 minute.
- L3+ stays in `bench/`, with explicit pass criteria and reference data committed in `bench/viscoelastic_validation/ref/`.

---

## 7. Gaps and risks

- **Missing public API for "frozen velocity → constitutive only".** The LBM-side pattern (test_viscoelastic.jl) and log-FV-side pattern (test_logfv_frozen_channel_cde.jl) coexist but are not surfaced as a single helper. Building this is cheap (single dispatching function) and would massively de-risk constitutive regressions.
- **Symmetric-BC trap for polymer stress.** No current test exercises a symmetry plane; if a future bench uses half-channel symmetry, expect false-positive errors. Either add a symmetric-BC test or forbid symmetry in V&V cases.
- **Transient validation absent.** Every steady-state test passes does NOT prove the time integrator. Add at least one Waters & King transient test (Basilisk has exactly this; replicate it).
- **No regression-tracked reference field dumps.** Basilisk commits `.ref`, `.kinetic`, `.ux`, `.uy` artefacts. Kraken should commit minimal versions (e.g. 32² τ_xy, N1 dumps) at known parameters.
- **Convergence verification absent for most benches.** Many `run_*` scripts report a number; few verify p≥1.8 order on grid halving. Convergence order is the cheapest false-positive killer (cf. Liu non-converged comparison incident).
- **Cd reference is unstable.** See §8.
- **Smoke tests masquerading as validation.** `test_viscoelastic_logfv_gpu_smoke.jl` runs but only checks for NaN. The next-mission upgrade: have GPU smoke also produce a one-norm against the CPU run, fail if drift > 1e-5.

---

## 8. References that need cross-check

- **Liu 2025 Eq.62 (Poiseuille Oldroyd-B analytic)** — used in `bench/viscoelastic_audit/common.jl` and several Cd convergence benches. Independent literature on Oldroyd-B planar Poiseuille (Waters & King 1970, Bird-Armstrong-Hassager Vol.1) should match Liu within transcription tolerance — but per `[Cylinder benchmark conventions]` in user memory, "Liu Re=1 gives Cd~131" was used as a target while we now suspect Liu non-converged. Cross-check Liu Eq.62 against Waters & King BEFORE trusting any planar-Poiseuille audit verdict.
- **Liu 2025 confined-cylinder Cd** — flagged twice in memory as "suspect" (M28 audit verdicts, M32 phase 3 closure). The new ground truth should be (a) rheoTool Oldroyd-BLog Cylinder canonical case at converged mesh + (b) Alves & Oliveira 2003 / Hulsen 2005 confined-cylinder published Cd tables. Until cross-checked: gate cylinder benches on rheoTool only, not Liu.
- **Hulsen 2005 K=132 plug flow** — used as historical reference for the Wi=0 limit. Memory note says this is correct; cross-check is cheap (analytic K=145 for Poiseuille inlet vs K=132 for plug).
- **Fattal-Kupferman 2005 lid-cavity** — Basilisk and rheoTool both ship this. Cross-checking these two against each other (without Kraken in the loop) would give us a second-opinion bound.
- **Alves-Oliveira 2003 / Oliveira-Pinho 1999 contraction 4:1** — corner-vortex length is the canonical pass criterion. We do not yet have a committed reference table; build one from rheoTool + at least one literature source.

---

## TOP-3 recommended patches (Phase B kickoff)

1. **Promote the existing `evolve_stress_2d!` + `run_viscoelastic_logfv_frozen_channel_cde_2d` patterns to a single public `frozen_velocity_constitutive!(...)` API**, with an L1 test in `test/runtests.jl` covering pure shear + Poiseuille at γ̇ ∈ {1e-3, 1e-2, 1e-1}, with analytic pass criterion. Cost: < 1 d, high payoff (prevents future "Liu non-converged" class of incidents).
2. **Commit a `bench/viscoelastic_validation/ref/` directory** seeded with: (a) Waters & King transient u(y,t) at 4 (Wi, β) points (analytic), (b) Fattal-Kupferman cavity centreline u dump from Basilisk `lid-oldroydb.ux/.uy/.kinetic`, (c) rheoTool Oldroyd-BLog Channel τ_xy(y) at converged mesh. Single source of truth, version-controlled.
3. **Convert `run_poiseuille_polymer_analytical_2d.jl` into `test/test_viscoelastic_L2_poiseuille.jl`** with a strict pass criterion (L2(u) < 1e-3, L2(τ_xy) < 5e-3, plus a grid-halving order check p ≥ 1.8 on a 16²→32²→64² ladder). Cost: < 0.5 d. This converts an existing artefact into a regression sentinel and exercises the FULL pipeline at L2 in CI.

These three together would close the largest current V&V gap (no canonical, cheap, full-pipeline regression test with an analytic reference) without committing to the full L0→L4 hierarchy until the Boss approves it.
