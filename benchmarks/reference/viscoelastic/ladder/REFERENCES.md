# References — viscoelastic V&V suite

Bibliography for all references used in `bench/viscoelastic_validation/`.
Each entry includes the verification status indicating whether the
underlying data has been cross-checked against an independent source.

Legend:
- `[ANALYTIC]` — closed-form expression, derived from first principles
- `[PUBLISHED]` — values from a peer-reviewed publication
- `[CODE-RUN]` — output of a reference code (Basilisk, rheoTool, …)
- `[VERIFIED]` — cross-checked against ≥ 2 independent sources
- `[UNVERIFIED]` — single source; future cross-check warranted
- `[SUSPECT]` — known or suspected to disagree with consensus; carry until
  retired by an audit mission

---

## 1. Planar Poiseuille Oldroyd-B (steady)

### Bird, Armstrong, Hassager (1987), *Dynamics of Polymeric Liquids, Vol 1: Fluid Mechanics*, 2nd ed., Wiley, §3.4
**[ANALYTIC] [VERIFIED]**

For pressure-driven planar Poiseuille of an Oldroyd-B fluid at steady state,
with channel half-height `h`, body force `Fx` per unit volume, total
viscosity `η = η_s + η_p`, polymer relaxation `λ`, polymer viscosity `η_p`:

```
γ̇(y)   = Fx · y / η                      (linear in y, zero at centreline y=0)
u(y)   = Fx · (h² - y²) / (2 · η)         (parabolic, Newtonian-like in u)
τ_xy(y) = η_p · γ̇(y)                      (linear in y)
τ_xx(y) = 2 · λ · η_p · γ̇(y)²             (quadratic in y; first normal stress)
τ_yy(y) = 0
N1(y)  = τ_xx - τ_yy = 2 · λ · η_p · γ̇²   (always non-negative)
```

Key point: in steady planar Poiseuille of Oldroyd-B, the **velocity profile
is identical to Newtonian** (no shear thinning); the polymer signature lives
entirely in `τ_xx`. This makes the test highly diagnostic: a wrong velocity
implies wrong momentum coupling; a correct velocity but wrong `τ_xx`
implies a wrong constitutive law.

Used at: L1 (`L1_poiseuille_oldb/`).

Cross-check: Waters & King (1970) reduces to these formulas in the limit
`t → ∞`; Basilisk `poiseuille-oldroydb.ref` converges to `u_centerline ≈
1.5` (in their non-dimensional units where `u_avg = 1`) as required.

### Waters & King (1970), *Rheologica Acta* 9 (3), 345-355
**[ANALYTIC] [PUBLISHED] [VERIFIED]**

Transient start-up planar Poiseuille of Oldroyd-B. Closed-form series solution
involving complex coefficients (β_n square-rooted from α_n² - E·n²); reduces
to the Bird-Armstrong-Hassager steady solution as `t → ∞`.

```
U(Y, T) = 1.5 (1 - Y²) - 48 Σ_{k=1}^∞ sin(n(1+Y)/2) / n³ · exp(-α_n T / 2) · G(T)
```

with `n = (2k-1)π`, `α_n = 1 + β E n² / 4`, `β_n = √(α_n² - E n²)`,
`γ_n = 1 - (2 - β) E n² / 4`, `G(T) = sinh(β_n T/2) + γ_n/β_n cosh(β_n T/2)`,
`E = λ μ₀ / (ρ h²)`, `T = t / λ`.

Used at: L2 (`L2_couette_startup_oldb/` for inspiration; L2 itself uses the
Couette start-up analogue), and as a cross-check against L1's steady reference.

Cross-check: implemented and validated in Basilisk
`src/test/poiseuille-oldroydb.c` and `.ref` (16² → 64² convergence shown).
Stored at `ref/waters_king_1970_couette.json` (formula coefficients only).

### Liu (2025) Eq. 62
**[PUBLISHED] [UNVERIFIED]**

Planar Poiseuille Oldroyd-B analytic; algebraically equivalent to
Bird-Armstrong-Hassager in the steady limit. Used historically in
`bench/viscoelastic_audit/common.jl`. **Carry as a cross-check only, not as
ground truth**, pending cross-validation against Waters & King transient at
finite t.

---

## 2. Couette start-up Oldroyd-B (transient)

### Waters & King (1970), *Rheologica Acta* 9 (3), 345-355
**[ANALYTIC] [PUBLISHED]**

Transient series for sudden start-up of one wall (Couette analogue). The
Poiseuille and Couette transient series share the same eigenvalue structure
(α_n, β_n) and differ only in the inhomogeneous part.

Used at: L2 (`L2_couette_startup_oldb/`, currently STUB).

---

## 3. Confined cylinder Oldroyd-B

### Hulsen (2005), JNNFM
**[PUBLISHED] [VERIFIED]** (Wi = 0 plug-flow K = 132 cross-checked against
analytic K = 145 Poiseuille inlet K = 132 plug inlet)

Confined-cylinder drag coefficient for Oldroyd-B. Wi = 0 limit serves as the
Newtonian sanity baseline.

### rheoTool Cylinder Oldroyd-BLog (canonical case)
**[CODE-RUN]**

Path: `/Users/guillaume/Documents/Recherche/Codes CFD/rheotool/rheoTool/of90/tutorials/rheoFoam/Cylinder/Oldroyd-BLog`.
Pre-computed Wi sweep stored at `bench/rheotool/cylinder_wi{0.05, 0.1, 0.2, 0.3, 0.5, 1.0}/`.

Used at: L4 (`L4_cylinder/`, STUB; see also `bench/viscoelastic_audit/` for
the live M28-M32 audit trail).

### Liu (2025) cylinder Cd
**[SUSPECT]** — flagged twice in `MEMORY.md` (M28 audit, M32 closure). Do
not use as primary reference; cross-check against rheoTool + Hulsen 2005.

---

## 4. Lid-driven cavity Oldroyd-B

### Fattal & Kupferman (2005), *JNNFM* 126 (1), 23-37
**[PUBLISHED]**

Time-dependent simulation of viscoelastic flows at high Wi using the
log-conformation representation. The cavity case at β = 0.5, Wi = 1 with
time-ramped lid is the canonical benchmark.

### Basilisk `lid-oldroydb.{c,ref,ux,uy,kinetic}`
**[CODE-RUN]**

Path: `/Users/guillaume/Documents/Recherche/Codes CFD/basilisk/src/test/`.
- `lid-oldroydb.kinetic` (52 pts): Fattal-Kupferman 2005 digitised kinetic
  energy time series
- `lid-oldroydb.ux` (49 pts): Fattal-Kupferman 2005 digitised u_x(x=0.5, y)
  profile at t = 8
- `lid-oldroydb.ref` (321 pts): Basilisk regression dump of energy time
  series
- `lid-oldroydb.c`: source with parameters DT_MAX = 5e-4, MU0 = 1, β = 0.5,
  Wi = 1, N = 64 multigrid, lid `8(1+tanh(8(t-0.5))) x² (1-x)²`

Not directly consumed at L1; cited for future L3 cavity case.

---

## 5. 4:1 contraction Oldroyd-B

### Alves, Pinho, Oliveira (2003), *JNNFM* 110, 45-75
**[PUBLISHED]**

Corner-vortex length tables for 4:1 planar contraction at multiple De.
Canonical pass criterion for the contraction benchmark.

### rheoTool Contraction41 Oldroyd-BLog
**[CODE-RUN]**

Path: `bench/rheotool/contraction41_oldroydb_log/` (pre-computed in Kraken
repo). Source rT tutorial at
`/Users/guillaume/Documents/Recherche/Codes CFD/rheotool/rheoTool/of90/tutorials/rheoFoam/Contraction41/Oldroyd-BLog`.

Used at: L3b (`L3b_4to1_contraction/`, STUB).

---

## 6. rheoTool Channel Oldroyd-BLog (canonical planar Poiseuille setup)

**[CODE-RUN]** (setup committed; output NOT precomputed in this repo)

Path: `/Users/guillaume/Documents/Recherche/Codes CFD/rheotool/rheoTool/of90/tutorials/rheoFoam/Channel/Oldroyd-BLog`.

Parameters from `constant/constitutiveProperties` + `system/blockMeshDict`:
- Domain: L × 2H × 1 = 40 × 2 × 1, mesh 50×30 per half (Ny = 60 total)
- Type: Oldroyd-BLog, stabilisation = coupling
- ρ = 1, η_s = 0.01, η_p = 0.99, λ = 1
- Inlet velocity: uniform U = 1 (developing flow); outlet zeroGradient
- BC: walls fixedValue U = 0; tau linearExtrapolation
- Re = ρ U H / (η_s + η_p) = 1, Wi = λ U / H = 1, β = 0.01

**Pre-computed output is NOT present** in
`/Users/guillaume/Documents/Recherche/Codes CFD/rheotool/.../Channel/Oldroyd-BLog/`
(only `0/` initial condition, `system/`, `constant/`; no time directories).
To produce a reference τ-profile dump, run `Allrun` in that case directory
and sample at `x = 30, y ∈ [-1, 1]` (sampleDict line set).

Not directly consumed at L1 (L1 uses analytic Bird-Armstrong-Hassager
instead). Documented here so that L1 can later be cross-checked against a
rheoTool dump if needed.

---

## 7. Cross-reference matrix

| Reference | L0 | L1  | L2  | L3a | L3b | L4  |
|-----------|----|-----|-----|-----|-----|-----|
| Bird-Armstrong-Hassager §3.4 |  | yes |     |     |     |     |
| Waters & King 1970           |  | aux | yes |     |     |     |
| Hulsen 2005                  |  |     |     |     |     | yes |
| rheoTool Cylinder            |  |     |     |     |     | yes |
| Fattal-Kupferman 2005        |  |     |     |     |     |     |
| Basilisk lid-oldroydb        |  |     |     |     |     |     |
| Alves-Pinho-Oliveira 2003    |  |     |     |     | yes |     |
| rheoTool Channel             |  | aux |     |     |     |     |
| rheoTool Contraction41       |  |     |     |     | yes |     |

(L3a BFS is included in the hierarchy as a stub but does not yet have a
selected primary reference. Candidates: Alves 2008, rheoTool BFS.)
