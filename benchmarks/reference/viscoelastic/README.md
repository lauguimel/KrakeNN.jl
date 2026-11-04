# Viscoelastic reference material

Ground truth for the viscoelastic validation work, recovered from the dormant
`dev-viscoelastic` branch before it was frozen. Two kinds of material live
here, and a third is deliberately kept outside the trunk.

## `ladder/` — the L0–L4 validation ladder

Five levels of increasing difficulty, each with its reference values and a
diagnostic purpose. `ladder/REFERENCES.md` is the entry point: it carries the
bibliography with an explicit provenance tag per entry —

- `[ANALYTIC]` closed form, derived from first principles
- `[PUBLISHED]` values from a peer-reviewed publication
- `[CODE-RUN]` output of a reference code
- `[VERIFIED]` cross-checked against two or more independent sources
- `[UNVERIFIED]` single source
- `[SUSPECT]` known or suspected to disagree with consensus

Two closed-form solutions are written out in full: steady planar Poiseuille of
an Oldroyd-B fluid (Bird, Armstrong & Hassager 1987, §3.4) and the Waters &
King (1970) transient series. The first is the sharper diagnostic of the two:
the Oldroyd-B velocity profile in steady planar Poiseuille is identical to the
Newtonian one, so a wrong velocity means wrong momentum coupling while a
correct velocity with a wrong `tau_xx` means a wrong constitutive law.

## `derivations/` — Chapman–Enskog moment analyses

Two worked derivations produced while diagnosing a force-coupling defect.
Kept because they are reasoning, not logs: regenerating them means redoing the
algebra.

## Raw RheoTool / OpenFOAM runs — NOT in the trunk

447 files, about 96 MB: full field dumps at roughly 15 timesteps for an
Oldroyd-B lid-driven cavity (two variants) and three cylinder cases
(Newtonian, Wi = 0.1, Wi = 1.0), plus the `m8_refs/` drag tables. They are
results from another code and cannot be cheaply regenerated, but nobody reads
them day to day, so they are preserved by tag rather than checked out in every
clone.

Retrieve them with:

```bash
git checkout archive/dev-viscoelastic-2026-09 -- bench/rheotool
```

The distilled comparison that the trunk *does* carry is
`benchmarks/results/rheotool_compare/viscoelastic/rheotool_cd_vs_wi.csv` — a
different mesh and radius from the archived raw runs, so the two are not
interchangeable.
