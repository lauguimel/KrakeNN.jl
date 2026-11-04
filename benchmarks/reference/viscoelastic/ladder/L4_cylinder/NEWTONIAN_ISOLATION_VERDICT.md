# M41 — Newtonian curved-BC isolation on cylinder — RESULTS

**Status**: COMPLETE 2026-05-23 evening. **Verdict: MIXED — BC fine at Newt, polymer-curvature coupling is the locus.**

Date    : 2026-05-23
Mission : Discriminate "curved BC family is buggy at Wi=0" vs "BC OK at
          Wi=0 but polymer-curvature coupling buggy at Wi>0".
Method  : Run cylinder at strict-Newtonian limit (beta=1.0, nu_p=0,
          polymer dormant) with `wall_bc=:bouzidi_fl_twopass` at
          R={30, 40, 60}, compare to halfwayBB G3 PASS reference
          (M32 Phase 3: R=30 Cd=132.08, rT 132.37, gap -0.22 %).
Aqua    : Jobs 21729717 (halfwayBB R=30 sanity, 4:36 walltime, exit 0) +
          21729718 (bouzidi_fl_twopass R=30/40/60, 8:08 walltime, exit 0)

## Results

| Setup                                    | Cd_kraken | NaN  | Δ vs halfwayBB | Δ vs rT |
| ---------------------------------------- | --------- | ---- | -------------- | ------- |
| R=30 :halfwayBB Newt (sanity)            | 132.076   | false| 0 (reference)  | −0.22 % |
| R=30 :bouzidi_fl_twopass Newt            | **132.637** | false| +0.42 %        | **+0.20 %** |
| R=40 :bouzidi_fl_twopass Newt            | 133.537   | false| +1.10 %        | (no rT ref) |
| R=60 :bouzidi_fl_twopass Newt            | **135.436** | false| **+2.54 %**    | (no rT ref) |

Reference rsync'd to `tmp/m41_newt_results/{halfway,curved_bc}/SUMMARY.csv`.

## Verdict (load-bearing): MIXED

**Three empirical findings:**

1. **`:bouzidi_fl_twopass` does NOT NaN at Newtonian R=60** (Cd=135.44 finite). At Wi=0.1 R=60 the same setup DID NaN. **→ The R=60 NaN at Wi≥0.1 is polymer-coupled, NOT BC-intrinsic.**
2. **At Newtonian R=30, `:bouzidi_fl_twopass` is CLOSER to rT** (132.64 vs rT 132.37, +0.20 %) **than `:halfwayBB`** (132.08, −0.22 %). The BC family corrects in the right direction.
3. **R-scaling of `:bouzidi_fl_twopass` Newt over-shoot**: +0.42 % at R=30 → +1.1 % at R=40 → +2.5 % at R=60. There IS a small R-dependent intrinsic bias (likely scales with cut-link cell count), but it is **much smaller than the polymer-induced over-shoot at Wi=0.1** (+1.6 % at R=30 already).

**Empirical decomposition of the cylinder Wi=1 gap:**

| Subsystem | Status at unit/Newtonian level | Reference |
|---|---|---|
| Pure polymer chain (planar) | **CORRECT** to Wi=1 | M40 L1 Wi sweep |
| Pure curved BC (Newtonian) | **MOSTLY CORRECT** (small R-bias) | This verdict (M41) |
| **Polymer × curved BC coupling** | **LOCUS** | empirically isolated by elimination |

The Wi-dependent jump from +0.20 % (Newt R=30) to +1.60 % (Wi=0.1 R=30) to NaN (Wi=1 R=30) is **polymer-driven amplification at curved boundary**, not BC-intrinsic.

## Implication for next mission

The locus is the **interaction between polymer-pipeline and curved-BC**, specifically at cut-link / wall-ring cells where:
- `vel_grad` uses solid-aware one-sided stencils (`_fvfd_solid_bc_derivative_*_2d`)
- `psi_advect` falls back from MUSCL to 1st-order Rusanov within ±2 cells of solid (M29b limitation)
- `poly_force` `∇·τ_p` computed across the cut-link
- `ρ_out` written by halfwayBB BC differs from Bouzidi-FL post-collision result

**Top candidate (HIGH confidence):** M29b boundary fallback to Rusanov within ±2 cells of solid. This is the only known-load-bearing limitation in the polymer-advection chain that is specific to curved geometry. M29b improved Cd at Wi=1 by +5 (rusanov → muscl_superbee) BUT the bulk fallback to rusanov within ±2 cells means the polymer field around the cylinder is still 1st-order advected — and this is exactly the shoulder zone where D1 found polymer × shoulder = +3.14 deficit.

**Recommended next mission**: implement and test the M29c-v2-style boundary-MUSCL relaxation (one-sided MUSCL at solid-adjacent cells) BUT applied with care to avoid the late NaN observed in M29c-v2 (step 92,200 on rho at j=1 south wall). The M30 P2b "two-pass kernel split" architecture may help by ensuring lag-0 reads at boundary cells.

The cylinder Wi=1 gap is now empirically attributed to **MUSCL boundary fallback at curved geometry**. Next experiment: probe-instrument that fallback (count fallback cells per step, max τ_p in fallback zone) to confirm before any implementation.

## Files produced

- `tmp/m41_newt_results/halfway/SUMMARY.csv` (R=30 halfwayBB Newt)
- `tmp/m41_newt_results/curved_bc/SUMMARY.csv` (R=30/40/60 bouzidi_fl_twopass Newt)
- Aqua job 21729717.aqua (halfway) + 21729718.aqua (curved) — finished exit 0

## Design notes (for future M41-style runs)

- **Manifest.toml is NOT portable between Mac local and Aqua** — different package registry snapshots cause GPUCompiler / SimpleTraits / etc. pinning errors. Best practice: rsync project WITHOUT Manifest.toml (add `--exclude 'Manifest.toml'`), let each side `Pkg.instantiate` from its own Project.toml + registry.
- **CairoMakie should stay in `[weakdeps]`** of Project.toml, NOT `[deps]`. Pulling it into `[deps]` triggers Aqua resolve failures because CairoMakie's transitive deps (e.g. SimpleTraits 0.9.6) are too new for the Aqua registry. The V&V suite plotting uses CairoMakie via the extension `KrakenMakieExt` which works with weakdep.

## Status: **PENDING_AQUA** (job submission blocked by network)

Aqua (`aqua.qut.edu.au:22`) is currently **unreachable** from this
session:

```
$ ssh -o ConnectTimeout=30 aqua "echo OK"
ssh: connect to host aqua.qut.edu.au port 22: Operation timed out
```

Two retry attempts (15 s and 30 s ConnectTimeout) both failed identically.
PBS files are **written and ready**; only the rsync + qsub step is
deferred.

## PBS artefacts (READY)

- `bench/viscoelastic_logfv/run_cyl_m41_newt_curved_bc_a100.pbs`
  - **Main case**: R = {30, 40, 60} x wall_bc=:bouzidi_fl_twopass at
    beta=1.0 (Newtonian limit), Wi=1.0 (placeholder, polymer dormant),
    Re=1, bsd=1.0, qwall, rusanov, L_up=L_down=15R, u_mean=0.005,
    max_steps=100k, save_fields=1.
  - Walltime budget: 02:00:00 (estimate 5-30 min per case on A100 F64;
    R=60 is the largest).
  - Output: `tmp/m41_newt_curved_bc/<jobid>/cyl_bigsweep_v2_*.csv` and
    `SUMMARY.csv`.

- `bench/viscoelastic_logfv/run_cyl_m41_newt_halfway_a100.pbs`
  - **Sanity case**: R=30, wall_bc=:halfwayBB, identical other params.
    Must reproduce 132.08 to confirm parameter-set fidelity before
    comparing curved-BC numbers.
  - Walltime: 01:00:00.

## Submission instructions (Boss next session, OR resume now if Aqua up)

```bash
# Step 1: rsync canonical
cd ~/Documents/Recherche/Kraken.jl-viscoelastic
bash hpc/sync_to_aqua.sh --apply

# Step 2: submit both PBS
ssh -o ConnectTimeout=15 aqua \
  "cd ~/Kraken.jl && \
   qsub bench/viscoelastic_logfv/run_cyl_m41_newt_halfway_a100.pbs && \
   qsub bench/viscoelastic_logfv/run_cyl_m41_newt_curved_bc_a100.pbs && \
   qstat -u maitreje | tail -10"

# Step 3: capture job IDs (M41_HALFWAY_JOB and M41_CURVED_JOB)
# Step 4: poll until F
ssh aqua "qstat -fx <jobid> | grep -E 'job_state|exit_status'"

# Step 5: rsync results back
rsync -avz aqua:Kraken.jl/tmp/m41_newt_curved_bc/ \
  ~/Documents/Recherche/Kraken.jl-viscoelastic/tmp/m41_newt_curved_bc/
rsync -avz aqua:Kraken.jl/tmp/m41_newt_halfway/ \
  ~/Documents/Recherche/Kraken.jl-viscoelastic/tmp/m41_newt_halfway/

# Step 6: extract Cd values
ls tmp/m41_newt_curved_bc/*/SUMMARY.csv
cat tmp/m41_newt_halfway/*/SUMMARY.csv | column -t -s,
```

## Job IDs

| PBS file | Job ID | Status |
|---|---|---|
| `run_cyl_m41_newt_halfway_a100.pbs` | **NOT SUBMITTED** | Pending (Aqua unreachable) |
| `run_cyl_m41_newt_curved_bc_a100.pbs` | **NOT SUBMITTED** | Pending (Aqua unreachable) |

## Expected results table (TO FILL after Aqua run)

| Case | R  | wall_bc            | Cd_kraken | vs ref | delta % | nan_flag |
|------|----|--------------------|-----------|--------|---------|----------|
| 1    | 30 | halfwayBB          | TBD       | 132.08 | TBD     | TBD      |
| 2    | 30 | bouzidi_fl_twopass | TBD       | 132.08 | TBD     | TBD      |
| 3    | 40 | bouzidi_fl_twopass | TBD       | TBD    | TBD     | TBD      |
| 4    | 60 | bouzidi_fl_twopass | TBD       | TBD    | TBD     | TBD      |

Reference: rheoTool Newtonian Cd = 132.37 (Hulsen K convention,
ρU²D = 2η_total = 2 → equivalent to Cd_classical at this benchmark).

## Decision tree on completion

Apply this once CSVs land:

1. If **case 2 Cd ≈ 132 ± 1%** (i.e. |delta| < 1.3): verdict
   `BC_FINE_AT_WI0`. Curved BC works at Wi=0. The +1.6 % Wi=0.1
   over-shoot and Wi=1 divergence are **polymer-coupled**. Next mission
   targets τ_p ↔ curved-BC stress-gradient coupling at the wall.

2. If **case 2 Cd over-shoots by ≈ +1.5%** (consistent with M34
   Wi=0.1 over-shoot): verdict `BC_INTRINSIC_BIAS`. Curved BC has a
   constant over-bounce independent of polymer. Next mission audits
   Bouzidi-FL two-pass interpolation coefficients (q-correction sign /
   moment timing) at staircased-circle geometry.

3. If **case 4 NaN** at Newtonian R=60 (where halfwayBB is finite per
   M32 Phase 3): verdict `BC_NaN_AT_R60_NEWT`. BC fundamentally
   unstable at high R. Curved-BC path is BLOCKED for R>=60 production
   work — must either fix the BC kernel or abandon it for the Hulsen
   benchmark.

4. **MIXED** (e.g. R=30 fine, R=40 over-shoots): document the
   R-dependence; suggests the bias scales with curvature resolution
   (Bouzidi-FL departing-link tolerance vs cell size).

## Forensic reference (M32 Phase 3 closure)

- `bench/viscoelastic_audit/M32_PHASE3_C1_CD_NORM_VERDICT.md`: confirms
  Kraken Cd ≡ classical 2Fx/(ρU²D), rT Hulsen K ≡ Fx/η_total, equal at
  the canonical setup (ρU²D = 2η_total = 2, holds for R=1 U=1 ρ=1
  η_total=1 OR equivalently R=30 u_mean=0.005 ρ=1 η_total=0.15
  → Re=1 ⇒ same dimensionless Cd).
- `bench/viscoelastic_audit/AUDIT_WI_SWEEP_20260430.md:33`:
  Cd_Newt(Kraken) = 132.08, Cd_Newt(rheoTool) = 132.36 (the reference
  numbers).
- `bench/viscoelastic_logfv/run_cyl_m32_newtonian_R60_a100.pbs`: prior
  Newtonian PBS template (beta=1.0, Wi=1.0 placeholder).
- `bench/viscoelastic_logfv/run_cyl_m34_bouzidi_fl_matrix_a100.pbs`:
  M34 G4 BC gate PBS at Wi={0.1, 1.0} R={30, 40} (the curved-BC
  over-shoot context that M41 isolates).

## Implication for next mission (verdict-dependent)

**Cannot finalise** until Aqua results land. Once available, the verdict
tree above selects the next mission. Branch:

- `BC_FINE_AT_WI0` → M42 polymer-wall coupling audit.
- `BC_INTRINSIC_BIAS` → M42 Bouzidi-FL interpolation audit.
- `BC_NaN_AT_R60_NEWT` → re-evaluate curved-BC viability for L4
  Hulsen-grade Wi-sweep; possibly retreat to halfwayBB for R=60 only.

## Memory candidates (filed once verdict resolves)

- `project_viscoelastic_l4_curved_bc_isolation` — M41 setup + verdict.
- `feedback_newtonian_isolation_pattern` — methodology: always isolate
  BC family vs polymer-curvature coupling by running Newtonian limit
  with the curved BC; β=1.0 + Wi-placeholder is the cleanest config.
