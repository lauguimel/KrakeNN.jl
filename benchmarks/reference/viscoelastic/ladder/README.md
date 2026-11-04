# Viscoelastic V&V suite — discoverable index

Canonical home for Kraken's viscoelastic verification & validation tests.
Created 2026-05-23 (mission M38-vv-architecture).

This suite is the **single source of truth** for "does Kraken's viscoelastic
stack still produce physically and numerically correct results?". It supersedes
the ad-hoc, per-session benches accumulated in `bench/viscoelastic_logfv/` and
`bench/viscoelastic_audit/` (which remain as evidence trails — see
[INVENTORY.md](INVENTORY.md)).

---

## 1. How to run the full suite

Each level is a directory with a self-contained runner.

```bash
# L1: planar Poiseuille, full pipeline, analytic Oldroyd-B reference, < 2 min CPU F64
julia --project=. bench/viscoelastic_validation/L1_poiseuille_oldb/run.jl

# L1 comparison & PASS/FAIL report
julia --project=. bench/viscoelastic_validation/L1_poiseuille_oldb/compare.jl
```

L0 / L2 / L3a / L3b / L4 are currently **stubs** (see each `STUB.md`); they will
be promoted to implemented tests in follow-up missions.

---

## 2. Hierarchy at a glance

| Level | Test                            | Reference                                | Status     | Cost target |
|-------|---------------------------------|------------------------------------------|------------|-------------|
| L0    | unit operators (exp/log, ∇·τ)   | algebraic / analytic                     | STUB       | < 1 s       |
| L1    | planar Poiseuille Oldroyd-B     | Bird-Armstrong-Hassager §3.4 (analytic)  | **READY**  | < 2 min     |
| L2    | Couette start-up Oldroyd-B      | Waters & King 1970 (transient series)    | STUB       | < 30 s      |
| L3a   | backward-facing step Oldroyd-B  | rheoTool BFS / Alves 2008                | STUB       | ~ 30 min    |
| L3b   | 4:1 contraction Oldroyd-B       | Alves-Pinho-Oliveira 2003 JNNFM          | STUB       | ~ 1 h       |
| L4    | confined cylinder Wi sweep      | rheoTool Cylinder + Hulsen 2005          | STUB       | 30 min–1 h  |

Each directory contains:

- `setup.md`: geometry, parameters, scheme choices, expected wall-clock
- `reference.json`: machine-readable reference values with provenance and tolerances
- `run.jl`: self-contained runner (no external configuration)
- `compare.jl`: assertions against `reference.json`; prints a per-quantity PASS/FAIL table
- `EXPECTED.md`: what PASS looks like, what each FAIL mode means

L4 also points to the existing M28-M32 audit trail in `bench/viscoelastic_audit/`
which is the longest-running real-world stress test of this stack.

---

## 3. How to interpret PASS / FAIL

`compare.jl` asserts MULTIPLE independent quantities; reporting is per-quantity
plus an overall verdict. The intent is to make a failure diagnose itself.

| Quantity                | What it tests                                     | Threshold (L1) |
|-------------------------|---------------------------------------------------|----------------|
| `u_centerline`          | momentum balance, BSD correction, BCs             | rel < 5e-3     |
| `tau_xy_wall`           | constitutive law + ∇u stencil at wall             | rel < 5e-2     |
| `tau_xx_wall`           | first normal stress (catches sign / Wi² errors)   | rel < 5e-2     |
| `max(rho - 1)`          | incompressibility / LBM density drift             | abs < 1e-3     |
| `max(div u)`            | continuity                                        | abs < 1e-8     |
| `min eig(C)`            | conformation tensor SPD-ness                      | > 0.8          |
| `no NaN / no Inf`       | sentinel                                          | hard fail      |

Tolerance choices reflect the LBM half-cell wall offset and the body-force
discretisation order. They are NOT "tight to machine precision" — they are
tight enough to catch the historical failure modes:

- Liu-non-converged: Cd off by 2-5%, but Cd alone is unreliable; here we catch
  it through the τ_xy ratio not matching analytic.
- Hermite CE-correction sign error: gives wrong τ_xx by factor (1-s/2);
  caught by `tau_xx_wall` relative threshold.
- Symmetric-BC polymer trap (see §4): mirrored τ_xy at symmetry would violate
  the analytic τ_xy(0) = 0; we avoid this by using full top-bottom walls.

---

## 4. Liu non-converged trap and other false-positive history

(Carry-forward from auto-memory; this section MUST stay current — it is the
project's institutional memory on viscoelastic V&V false positives.)

- **Liu 2025 confined cylinder Cd ≈ 131.** Used as ground truth in M22-M28;
  *suspect* since M28 audit, formally retired as a reference at M32 closure.
  Cross-check: rheoTool cylinder converged + Hulsen 2005 K=132 plug-flow.
  Until cross-checked, **never gate on Liu alone**.
- **Liu Eq. 62 planar Poiseuille analytic.** Algebraically equivalent to
  Waters & King 1970 in the steady limit; transient form has not been
  cross-checked. We use Bird-Armstrong-Hassager Vol.1 §3.4 (Oldroyd-B steady
  planar Poiseuille) here.
- **HWNP / "8-17 % at low Wi" false alarm (2026-04-21 audit).** Not physics;
  was λ < cell-substep numerical stiffness at Wi = 1e-3. Documented in
  `MEMORY.md → project_viscoelastic_audit`.
- **Post-source MEA != surface quadrature.** `test_viscoelastic_force_accounting.jl`
  proves that the post-Hermite MEA increment is NOT a physical surface stress;
  ratio to analytic is 1/(1-s/2) ≠ 1. **Don't trust MEA-only τ_p drag.**
- **Symmetry BC polymer trap.** A symmetry plane at y = 0 requires τ_xy = 0
  and ∂y τ_xx = ∂y τ_yy = 0. LBM ghost-cell mirror does NOT enforce this
  for the polymer kernel. **All V&V cases here use full top-bottom walls.**
- **Wall-ring indexing trap.** `:phys` frame is 1 LU off and biases Cd_polymer
  by +24 % (M31 adversarial finding). Use `:idx` frame: `dx = (i-1) - cx_phys`.

When you find a new false positive: add a bullet here AND log it in
`MEMORY.md`.

---

## 5. Update protocol: when references change

References live in `ref/` as JSON with full provenance. Each entry carries:

- `type`: `"analytic"`, `"published"`, `"rheotool_run"`, `"basilisk_dump"`
- `source`: human-readable citation
- `provenance`: DOI, commit hash, formula, or path
- `convergence_verified`: bool + explanation
- `values`: numerical content (with units)
- `tolerance_pass_fail`: thresholds the compare scripts use

**To add a new reference**: drop the JSON in `ref/`, document its origin in
`ref/PROVENANCE.md`, cite it in `REFERENCES.md`, and add a row to
`reference.json` of the level that consumes it.

**To revise a reference**: never overwrite silently. Bump a version field in
the JSON, keep the previous one as `_v1`, document the change in
`ref/PROVENANCE.md`. The compare scripts will then need to be updated to
point at the new key.

**Never**: hard-code a reference value inline in `compare.jl`. Always go
through `reference.json`.

---

## 6. Provenance & inventory

- [INVENTORY.md](INVENTORY.md) — Phase A audit (2026-05-23) of all existing
  viscoelastic tests and benches (13 tests + 25 benches catalogued).
- [REFERENCES.md](REFERENCES.md) — bibliography with per-source verification
  status.
- [ref/PROVENANCE.md](ref/PROVENANCE.md) — per-JSON detailed sources.

---

## 7. Out of scope (this mission, M38)

- No `src/` edits; the public API hook for "imposed-velocity constitutive-only"
  is a separate follow-up mission.
- No promotion of existing `test/test_*.jl` files in place; they are
  referenced from each level's `STUB.md`.
- No Aqua HPC submission; everything here is CPU F64 and must run on a
  laptop.
- L0 / L2 / L3a / L3b / L4 implementation is deferred (stubs only).
