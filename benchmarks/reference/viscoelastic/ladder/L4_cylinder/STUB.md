# L4 — confined cylinder Oldroyd-B (STUB)

Status: **STUB**. This is Kraken's *flagship* viscoelastic benchmark and
has the longest live audit trail of any case in the repo.

## Why this level

The confined-cylinder drag coefficient `Cd` at fixed Re with a polymer
Wi-sweep is the de-facto industry benchmark for viscoelastic simulators.
Hulsen 2005's K = 132 plug-flow reference and the Liu 2025 Wi-sweep
table are the two most cited references.

## Reference

Primary candidate (post-Liu retirement): **rheoTool Cylinder
Oldroyd-BLog** + **Hulsen 2005** for the Wi → 0 limit.

- rheoTool canonical case:
  `/Users/guillaume/Documents/Recherche/Codes CFD/rheotool/rheoTool/of90/tutorials/rheoFoam/Cylinder/Oldroyd-BLog`
- Pre-computed Wi sweep in `bench/rheotool/cylinder_wi{0.05,0.1,0.2,0.3,0.5,1.0}/`
- Newtonian baseline: `bench/rheotool/cylinder_newtonian_re1/`,
  `cylinder_newtonian_re1_shrunk15R/`

**Liu 2025 Cd reference is SUSPECT** (flagged in MEMORY.md / M28 audit /
M32 closure). Do not gate L4 on Liu.

## Existing audit trail

Mission cluster **M22 → M32** documented at:
- `bench/viscoelastic_audit/README.md`
- `bench/viscoelastic_audit/AUDIT_SUMMARY.md`
- Multiple `bench/scratch/m*` directories
- Auto-memory `project_viscoelastic_audit` entry

M32 Phase 3 closure (commit `fabaf7e8`): G3 PASS, gap at Wi=1
R-invariant = polymer scheme locus.

This is far more empirical evidence than for any other Kraken viscoelastic
case. **L4's job is to consolidate that evidence into a maintained
test**, not to re-derive it.

## Promotion plan

L4 should be the **last** level promoted to a maintained test, because:
1. The Wi = 1 polymer-scheme-locus question is not yet closed.
2. The reference (rT cylinder converged) is more expensive to ship as a
   JSON than the analytic L1 / L2 references.
3. Validation lives at the boundary of current Kraken capability —
   stabilising the reference comparison BEFORE locking it as a regression
   gate is essential.

## Design sketch

- Geometry: 2D confined cylinder, blockage ratio B = 0.5
  (cylinder D / channel H = 0.5).
- Inlet: developed Poiseuille from L1.
- Outlet: zero-gradient or convective.
- Walls: top/bottom no-slip; cylinder surface no-slip via Bouzidi /
  cut-link.
- Re = 1, Wi ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0}.

## Existing Kraken assets

- `src/drivers/viscoelastic_logfv_2d.jl` — `run_viscoelastic_logfv_cylinder_coupled_2d`
- `bench/viscoelastic_logfv/logfv_cylinder_cd_convergence.jl`
- `bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl`
- `bench/viscoelastic_logfv/run_cyl_cd_convergence_*.jl` (multiple BSD
  variants)
- `bench/viscoelastic_audit/` (live audit trail)

## Cost target

30 min – 1 h on a single CPU core per Wi point; sweep needs GPU.

## Out of scope until L4 promotion

- 3D cylinder (sphere).
- Different blockage ratios.
- FENE-P / PTT.
