# Provenance of `ref/*.json` reference data

For each JSON, document where the data originated and how to reproduce it.

---

## `basilisk_lid_oldroydb.json`

Source: `/Users/guillaume/Documents/Recherche/Codes CFD/basilisk/src/test/lid-oldroydb.{c,ref,ux,uy,kinetic}`.

Content extracted:
- `kinetic_fattal_kupferman_2005`: 52-pt digitised kinetic energy time series
  from Fattal & Kupferman (2005), JNNFM 126 (1), 23-37, as committed by
  Basilisk under filename `lid-oldroydb.kinetic`. First column is `t`,
  second is `KE(t)`.
- `ux_centerline_fattal_kupferman_2005`: 49-pt digitised u_x(x=0.5, y)
  profile at t = 8, also Fattal-Kupferman 2005, committed by Basilisk as
  `lid-oldroydb.ux`. First column is `u_x`, second is `y` (note column
  order in the dump).
- `basilisk_regression_ref`: a pointer ONLY to `lid-oldroydb.ref` (321 pts);
  not inlined here because it is the Basilisk-self regression target
  (`t, KE`), not the physical reference. Use the Fattal-Kupferman columns
  above for cross-checking Kraken.

Parameters of the canonical run: domain `L0 = 1`, multigrid `N = 64`,
β = 0.5, Wi = λ = 1, `DT_MAX = 5e-4`, lid `8 (1 + tanh(8(t - 0.5))) x² (1 - x)²`.

Convergence verified: Basilisk publishes a single mesh (64²). Status
`convergence_verified = "single_mesh_published"`.

To reproduce: clone Basilisk, `cd src/test`, `make lid-oldroydb.tst`.

---

## `basilisk_poiseuille_oldb.json`

Source: `/Users/guillaume/Documents/Recherche/Codes CFD/basilisk/src/test/poiseuille-oldroydb.{c,ref}`.

Content extracted: the full `.ref` (153 rows × 5 columns: t/λ,
u_centerline/u_avg from Basilisk, u_centerline/u_avg analytic Waters & King,
mesh level lev ∈ {4, 5, 6} corresponding to 16² / 32² / 64², and `L²` flag
= 1 for the Oldroyd-B variant).

Parameters: μ₀ = 1, β = 1/9, λ = 1, periodic in x with ∇p = 1 between
left p = 1 and right p = 0; bottom Neumann, top Dirichlet u = 0 (i.e.
the simulation uses symmetry at y = 0, full wall at y = 1). Time grid
`t += 0.2`, `t ≤ 10`. `KF = 8` modes summed in the Waters & King series.

Convergence verified: 16² → 32² → 64² triplet present in the `.ref`,
showing convergence to the analytic curve. Status
`convergence_verified = "grid_converged_16_32_64"`.

To reproduce: as above, `make poiseuille-oldroydb.tst`.

---

## `rheotool_channel_tau.json`

Source: setup at
`/Users/guillaume/Documents/Recherche/Codes CFD/rheotool/rheoTool/of90/tutorials/rheoFoam/Channel/Oldroyd-BLog`.

Content extracted: **setup parameters only**. No precomputed τ profile is
committed in the rheoTool tutorial directory (only `0/` initial fields and
`system/sampleDict`); the case has not been pre-run.

The JSON documents the rT setup so that whoever runs `Allrun` can store the
resulting τ-profile dump (line `lineVert` at x = 30, y ∈ [-1.2, 1.2]) back
into this directory under `rheotool_channel_tau_<commit>.dat`. The expected
file pattern is `postProcessing/sets/<time>/lineVert_tau.xy`.

Parameters (verbatim):
- Domain: L × 2H × 1 = 40 × 2 × 1
- Mesh: 2 blocks of (50, 30, 1) cells (total Nx × Ny = 50 × 60 in 2D plane)
- ρ = 1, η_s = 0.01, η_p = 0.99, λ = 1
- Inlet: U = (1, 0, 0) uniform fixedValue; outlet zeroGradient
- Walls: U = 0 fixedValue; τ linearExtrapolation
- endTime = 30, deltaT = 0.01, writeInterval = 500
- Constitutive: Oldroyd-BLog, stabilisation = coupling

Convergence verified: NOT verified at the level of the τ-profile (case not
yet run in this project). Status `convergence_verified = false`.

Re = 1, Wi = 1, β = 0.01 (these are non-dimensional; rT carries dimensional
form with H = 1, U = 1).

To reproduce: `cd <path>`; `./Allrun`; then sample with `sample` from the
`postProcessing` directory.

---

## `waters_king_1970_couette.json`

Source: analytic formula transcribed from Waters & King (1970), as
implemented in Basilisk `poiseuille-oldroydb.c` (lines 106-119).

Content extracted: the closed-form coefficients (α_n, β_n, γ_n) and the
series structure. Implementation is in the JSON so that L2's runner can
reuse it; *not* a numerical table.

Parameters needed: β (solvent ratio), E (= λ μ₀ / (ρ h²)), KF (number of
modes — Basilisk uses 8, which is converged at single-precision tolerance
for E ≲ 1; for higher E, more modes may be required).

Convergence verified: against Basilisk's `.ref` Oldroyd-B sweep at
β = 1/9, E = λ μ₀ = 1, on multiple grids. Status
`convergence_verified = "cross_checked_against_basilisk_ref"`.

To reproduce: any implementation of the series in any language using
`complex` arithmetic; Basilisk source serves as a reference implementation.
