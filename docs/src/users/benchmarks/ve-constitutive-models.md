# Viscoelastic constitutive models — FVFD log-conformation (3D)

Kraken ships **four** 3D viscoelastic constitutive closures, all evolved in the
log-conformation variable `Ψ = log C` (Fattal & Kupferman 2004) by the same
finite-volume / finite-difference (**FVFD**) polymer transport that the
[Viscoelastic Poiseuille](ve3d-poiseuille-convergence.md) and
[Viscoelastic cylinder](viscoelastic-cylinder.md) benchmarks use. They differ
only in how the conformation relaxes — a single change in the constitutive
source term — so they share one validated transport, stress-reconstruction and
wall-BC machinery and are selected by passing a different model spec
(`LogConfOldroydB`, `LogConfFENEP`, `LogConfGiesekus`, `LogConfPTT`).

This page introduces the four models and compares them **analytically**: each is
driven single-cell through steady simple shear and steady planar extension by
the *real* Kraken constitutive solver (no closed-form shortcut), so the figure
below shows exactly what Kraken computes.

## The four models

All four advance `dC/dt = ∇uᵀ·C + C·∇u − R(C)/λ`, with the same upper-convected
stretch term and an Oldroyd-B stress reconstruction `τ_p = G·(C − I)` (FENE-P
adds the Peterlin factor). Only the relaxation `R(C)` changes:

| Model | Spec | Relaxation `R(C)` | Parameter | Equilibrium / OB limit |
|-------|------|-------------------|-----------|------------------------|
| **Oldroyd-B** | `LogConfOldroydB(G, λ)` | `C − I` | — | — |
| **FENE-P** | `LogConfFENEP(G, λ, L²)` | `f·C − I`, `f = (L²−3)/(L²−tr C)` | `L² > 3` | `f→1` as `L²→∞` |
| **Giesekus** | `LogConfGiesekus(G, λ, α)` | `(C−I) + α·(C−I)²` | `α ∈ [0, 0.5]` | `α = 0` |
| **PTT** | `LogConfPTT(G, λ, ε, variant)` | `Y(tr C)·(C − I)` | `ε ≥ 0` | `ε = 0` |

For PTT the scalar trace multiplier is `Y(tr C) = 1 + ε(tr C − 3)` (`:linear`,
Phan-Thien & Tanner 1977) or `Y(tr C) = exp(ε(tr C − 3))` (`:exponential`,
Phan-Thien 1978). For Giesekus the quadratic mobility acts per-eigenvalue,
`(c_i − 1)·(1 + α(c_i − 1))`, enhancing relaxation of stretched modes; for PTT
the *same* scalar `Y` multiplies every eigenvalue.

### Physical behaviour

| | constant viscosity | finite extensibility (`tr C < L²`) | shear-thinning |
|-|:-:|:-:|:-:|
| **Oldroyd-B** | ✓ | ✗ (unbounded) | ✗ |
| **FENE-P** | ✗ | ✓ | ✓ |
| **Giesekus** | ✗ | ✓ (bounded extension) | ✓ |
| **PTT** | ✗ | ✓ (bounded extension) | ✓ |

Oldroyd-B is the linear reference: a constant total viscosity, a first
normal-stress difference `N1 ∝ Wi²` that grows without bound, and a planar
extension that **diverges** at the coil-stretch number `λε̇ = 0.5`
(`C_xx = 1/(1−2λε̇)`). The other three are nonlinear: they shear-thin and bound
the stretch, so they remain finite in strong extension — the physically
essential behaviour for real polymer solutions.

## Analytical comparison

![Viscoelastic constitutive models — shear & extension](ve-constitutive-models.png)

Left: steady **simple shear**, first normal-stress difference
`N1 = G(C_xx − C_yy)` vs `Wi = λγ̇` (log-log). All four collapse onto the
Oldroyd-B quadratic `N1 = 2 Wi²` at low `Wi` (the linear-viscoelastic limit),
then separate: Oldroyd-B stays on the `Wi²` line, while FENE-P, Giesekus and PTT
**shear-thin** and fall below it, each with a distinct slope set by its
nonlinearity. Right: steady **planar extension**, streamwise conformation
`C_xx` vs `λε̇ ∈ [0, 0.49]`. Oldroyd-B blows up toward the pole at `λε̇ = 0.5`;
the three bounded models saturate at distinct finite plateaus.

### Key numbers (from the real solver, `G = λ = 1`)

At `Wi = 5` in simple shear the models are clearly separated by their
shear-thinning, even though they agree to `< 1 %` at `Wi ≤ 0.1`:

| Model | `N1` (`Wi = 5`) | `η_p,app/η_p,0` (`Wi = 5`) | `C_xx` (`λε̇ = 0.49`) |
|-------|----------------:|---------------------------:|----------------------:|
| **Oldroyd-B** | 50.0 (`= 2 Wi²`) | 1.000 (no thinning) | **49.98** (→ pole at 0.5) |
| **FENE-P** (`L²=50`) | 15.88 | 0.466 | **6.88** |
| **Giesekus** (`α=0.2`) | 6.47 | 0.275 | **3.16** |
| **PTT** (`ε=0.25`) | 6.83 | 0.369 | **3.15** |

The contrast is the headline: Oldroyd-B's extensional `C_xx` runs away by an
order of magnitude as `λε̇ → 0.5`, while the bounded models settle at finite
plateaus — `≈ 6.9` (FENE-P, set by `L²`), `≈ 3.2` (Giesekus), `≈ 3.1` (PTT).
The two trace-based closures (Giesekus, PTT) plateau close together here; FENE-P
plateaus higher because `L² = 50` permits more stretch than the `α = 0.2` /
`ε = 0.25` nonlinearities. This is *exactly* the qualitative behaviour a
viscoelastic solver must reproduce to be useful beyond the linear regime.

## In a channel flow (Poiseuille)

The analytical comparison above imposes the velocity gradient. The real test is
**coupled**: drive each model through the same planar-Poiseuille channel with
the full FVFD solver — D3Q19 solvent at `ν_s` plus the log-conformation polymer
transport, two-way coupled through the Guo body force `F_poly = ∇·τ_p` — and
let the flow set its own shear rate `γ̇(y)`. Only the constitutive
`polymer_model` spec changes between runs, so the wall-normal profiles differ
*only* by the closure.

The channel matches the constitutive-coupling canaries
(`test/test_fvfd_{fenep,giesekus,ptt}_coupled_3d.jl`): periodic `x`/`z`,
half-way bounce-back no-slip `y` walls, a constant streamwise body force `Fx`,
solvent/polymer split `β = ν_s/ν_total = 0.5`, and `λ` set so the wall
Weissenberg number `Wi_wall = λγ̇_wall ≈ 1`. In Poiseuille the shear rate is
**zero at the centre and peaks at the walls**, so the near-wall band is the
high-`Wi` region where the closures separate, while the low-shear core relaxes
to the equilibrium `tr C = 3` for every model.

![Viscoelastic constitutive models — coupled Poiseuille trace(C)](ve-constitutive-models-poiseuille.png)

Left: the polymer stretch `tr C = C_xx + C_yy + C_zz` across the channel
half-width (wall → centre). All four collapse onto `tr C = 3` at the low-shear
core, then **fan out in the near-wall high-shear band** — exactly the
differentiation the homogeneous figure predicts, now emerging from the flow
itself. Right: the coupled velocity profile `u(y)`; the shear-thinning closures
carry less near-wall polymer stress, so the solvent shear moves a little faster
and their profiles are slightly fuller.

### Key numbers (coupled, `Wi_wall ≈ 1`, `β = 0.5`, `Ny = 32`)

Ranked by near-wall stretch (`tr C` at the wall plane `j = 1`, the highest-shear
row), the models order exactly as the constitutive nonlinearity predicts:

| Model | `tr C` (wall) | `tr C` (centre) | peak `u` |
|-------|--------------:|----------------:|---------:|
| **Oldroyd-B** | **5.00** (unbounded stretch) | 3.002 | 0.01916 |
| **FENE-P** (`L²=50`) | **4.74** | 3.002 | 0.01936 |
| **PTT** (`ε=0.25`) | **4.44** | 3.002 | 0.02093 |
| **Giesekus** (`α=0.2`) | **4.27** | 3.001 | 0.02124 |

Oldroyd-B sits highest because nothing caps its stretch; FENE-P / PTT / Giesekus
each thin the near-wall conformation, in that order, and the same ordering holds
all the way through the high-shear band (e.g. at `j = 4`: `4.30 / 4.16 / 4.02 /
3.89`). The velocity peak ranks inversely — Giesekus and PTT, which shed the most
near-wall polymer stress, flow fastest. All four runs are NaN-free and reach a
steady, symmetric profile in `10 000` steps. This is the same near-wall
separation the analytical figure predicts, now produced by a fully coupled
solver — the practical signature a viscoelastic code must show in a real flow.

## Validation status

Each constitutive closure is validated independently before it appears here:

- **Oldroyd-B / FENE-P / Giesekus / PTT** each match their closed-form steady
  simple-shear fixed point in a dedicated single-cell canary
  (`test/test_fvfd_{logconf,fenep,giesekus,ptt}_3d.jl`): Giesekus and PTT to a
  steady-equation residual `≤ 1e-6`, FENE-P and the Oldroyd-B limit to `≤ 1e-3`.
- **Oldroyd-B limit.** Setting `α = 0` (Giesekus) or `ε = 0` (PTT, both
  variants) makes the relaxation factor exactly `1`, so the integrator follows
  the same floating-point trajectory as the dedicated Oldroyd-B kernel. FENE-P
  only tends to that limit as `L² → ∞` (the factor is `(L²−3)/(L²−tr C)`, never
  exactly `1`): the FENE-P test uses `L² = 1e8` and accepts an error of `1e-3`
  on `C_xy` and `C_xx` against the Oldroyd-B fixed point, so it is not a
  byte-identical check.
- **External cross-validation (RheoTool).** The Oldroyd-B planar-extension
  conformation matches RheoTool's `rheoTestFoam` to machine precision in the
  Hookean (`L² → ∞`) limit. For finite `L²`, Kraken's FENE-P and RheoTool's
  FENE-P differ by a documented `~11 %` in `C_xx` — a genuine **closure-variant**
  difference (Peterlin argument `tr C` vs `tr A`), *not* a bug: Kraken's own
  1000-step canary matches its own steady-state transcendental solution to
  `1e-5` relative. See
  `benchmarks/results/rheotool_compare/viscoelastic_extensional_fenep_3d/`
  and the [Viscoelastic cylinder](viscoelastic-cylinder.md) page (Oldroyd-B,
  `< 1 %` on `C_d`).

## Reproduce

The four constitutive closures are exercised in CI by the single-cell canaries
above (CPU Float64):

```julia
# from the repo root
import Pkg
Pkg.test("Kraken"; test_args=["test_fvfd_giesekus_3d.jl"])
Pkg.test("Kraken"; test_args=["test_fvfd_ptt_3d.jl"])
Pkg.test("Kraken"; test_args=["test_fvfd_fenep_3d.jl"])
```

The analytical comparison data and the figure generator are tracked under
`benchmarks/results/repro/ve-constitutive-models/`. The data is **fully
CI-reproducible** (single-cell, CPU Float64, ~1 min) — regenerate the CSV with
the real Kraken solver:

```bash
# from the repo root — drives all 4 models, both flows, writes the CSV
julia --project=. benchmarks/results/repro/ve-constitutive-models/make_csv.jl
```

then regenerate the figure (env `kraken-v0-3-figures`):

```bash
conda run -n kraken-v0-3-figures python \
  benchmarks/results/repro/ve-constitutive-models/plot.py
```

The **coupled Poiseuille** comparison is a sibling pair in the same bundle. It
drives the real FVFD Poiseuille solver for each model and writes the
wall-normal profiles (CPU Float64, `Ny = 32`, ~1–2 min, CI-reproducible — no
GPU):

```bash
# from the repo root — runs the 4 coupled drivers, writes the profile CSV
julia --project=. \
  benchmarks/results/repro/ve-constitutive-models/make_poiseuille_csv.jl

# then the dark figure (env kraken-v0-3-figures)
conda run -n kraken-v0-3-figures python \
  benchmarks/results/repro/ve-constitutive-models/plot_poiseuille.py
```

The **cross-slot strand** comparison is a third pair in the same bundle. It
drives the real FVFD planar-extension solver
(`run_viscoelastic_fvfd_extensional_3d`, `velocity_mode=:imposed`) for each
model and writes the `z`-mid `tr C(x, y)` field (CPU Float64, `Nx = Ny = 32`,
`λε̇ = 0.35`, ~a few minutes, CI-reproducible — no GPU):

```bash
# from the repo root — runs the 4 imposed-stagnation drivers, writes the field CSV
julia --project=. \
  benchmarks/results/repro/ve-constitutive-models/make_crossslot_csv.jl

# then the dark figure (env kraken-v0-3-figures)
conda run -n kraken-v0-3-figures python \
  benchmarks/results/repro/ve-constitutive-models/plot_crossslot.py
```

The shipped viscoelastic `.krk` presets (sphere, extensional, cylinder) live in
`benchmarks/krk/viscoelastic/`.

## References

- R. Fattal, R. Kupferman (2004), *Constitutive laws for the matrix-logarithm of
  the conformation tensor*, J. Non-Newtonian Fluid Mech. **123**, 281–285 — the
  log-conformation representation shared by all four models.
- H. Giesekus (1982), *A simple constitutive equation for polymer fluids based on
  the concept of deformation-dependent tensorial mobility*, J. Non-Newtonian
  Fluid Mech. **11**, 69–109.
- N. Phan-Thien, R.I. Tanner (1977), *A new constitutive equation derived from
  network theory*, J. Non-Newtonian Fluid Mech. **2**, 353–365.
- F. Pimenta, M.A. Alves (2017), *Stabilization of an open-source finite-volume
  solver for viscoelastic fluid flows*, J. Non-Newtonian Fluid Mech. **239**,
  85–104 — the RheoTool reference.

## Planar stagnation-point flow (cross-slot strand)

The Poiseuille channel above separates the closures through wall **shear**. The
complementary — and for polymer solutions far more demanding — test is strong
**extension**. The cleanest extensional flow is the **planar stagnation point**:
the kinematic core of a cross-slot device, where two opposed inlets feed two
opposed outlets and the streamlines cross at a central stagnation point. We
idealise that core directly with the imposed straining field

```
u = (ε̇·x,  −ε̇·y,  0)
```

— **outflow along ±x**, **inflow along ±y**, with a stagnation point at the
centre. A fluid element entering along the inflow (`y`) axis decelerates toward
the centre, then accelerates out along the outflow (`x`) axis: it spends a long
residence time in the extensional region, so the conformation tensor `C` builds
a high-stretch **strand** along the outflow axis while relaxing back toward
equilibrium `tr C = 3` along the inflow axis. This is the lattice-scale origin
of the cross-slot's iconic central **birefringent strand**. Crucially, it is
driven by conformation **advection** (the `∇uᵀ·C + C·∇u` transport): even though
the imposed strain rate is spatially *uniform*, the resulting `tr C` field is
strongly **non-uniform** — exactly the strand. (A literal four-arm cross-slot
geometry, with the stagnation point produced by the flow rather than imposed, is
a natural future extension; here we isolate its essential kinematics.)

We drive the same straining field through all four closures with the real FVFD
log-conformation pipeline (`run_viscoelastic_fvfd_extensional_3d`,
`velocity_mode=:imposed`: the conformation is advected + relaxed by the actual
solver while the velocity is held at the analytical stagnation field). The
operating point is matched to the validated extensional canary — a small strain
rate `ε̇` in lattice units (so the coupled D3Q19 solvent step stays stable) with
a large `λ` giving an extension number `λε̇ = 0.35`, a strong Oldroyd-B stretch
safely below the coil-stretch pole `λε̇ = 0.5`.

![Viscoelastic constitutive models — cross-slot strand trace(C)](ve-constitutive-models-crossslot.png)

Top: the `tr C(x, y)` field in the `z`-mid plane for the four models on a shared
colour scale. Each shows the bright **strand along the outflow (`x`) axis**
through the stagnation point, fading to `tr C = 3` along the inflow (`y`) axis —
the signature stagnation-point anisotropy. Bottom: `tr C` along the outflow axis,
which cleanly ranks the four closures by strand intensity.

### Key numbers (`λε̇ = 0.35`, `β = 0.5`, `Nx = Ny = 32`)

Ranked by strand intensity (peak `tr C` on the outflow axis), the models order
exactly as the constitutive nonlinearity predicts:

| Model | strand peak `tr C` | inflow `tr C` (`r = 8`) | outflow/inflow anisotropy |
|-------|-------------------:|------------------------:|--------------------------:|
| **Oldroyd-B** | **4.71** (unbounded stretch) | 3.63 | **1.30×** |
| **FENE-P** (`L²=50`) | **4.45** | 3.60 | 1.23× |
| **PTT** (`ε=0.25`) | **3.94** | 3.56 | 1.11× |
| **Giesekus** (`α=0.2`) | **3.83** | 3.49 | 1.10× |

Oldroyd-B builds the most intense strand because nothing caps its extensional
stretch; FENE-P sits below it (the `L² = 50` extensibility limit), and the two
trace-based closures shed the most — Giesekus and PTT cap extension hardest, so
their strands are the weakest. This is the **same ordering** as the homogeneous
extension figure (`C_xx` plateaus) and the coupled Poiseuille near-wall stretch,
now produced by a stagnation-point flow with a spatially varying field — the
practical signature a viscoelastic solver must reproduce to predict cross-slot
behaviour. All four runs are NaN-free and reach a steady strand.
