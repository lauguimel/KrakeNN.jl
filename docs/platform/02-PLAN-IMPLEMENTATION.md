# PLAN D'IMPLÉMENTATION — premières PRs, ordre des fichiers

> Concret. Poser le contrat **autour** du solveur LBM existant sans le casser. Petits incréments
> validés, aucune feature sans test/benchmark (principe #8). Aucune PR ne touche les kernels
> LBM en Phase 0-2.

---

## Arborescence cible (nouveau module `src/platform/`)

```
src/platform/
  contract.jl        # abstract types + capabilities() (le contrat, ~80 LOC)
  solution.jl        # AbstractSolution + LBMSolution (wrappe le NamedTuple)
  sample.jl          # sample(sol, field, query)  (déplace extract_line/probe)
  observe.jl         # AbstractObservable, Prediction, observe(), predict()
  residual.jl        # residual(problem, method, u, p) + VJP via l'adjoint existant
  calibration.jl     # ParameterSpace, loss, fit
  closure.jl         # AbstractClosure, evaluate()  (Phase 3)
  schema/            # (Phase axes) JSON inverse-envelope + .krk-as-config
test/platform/
  contract_parity_test.jl   # solve(...) == run_simulation(...) bit-à-bit
  benchmarks/               # un dossier par méthode×physique : ref analytique + ordre + gradient réf
```

`src/platform/` est **additif** : `Kraken.jl` (module top) ajoute `include("platform/...")`
et exporte les nouveaux verbes. Rien n'est retiré.

---

## PR 1 — Le contrat nu (Phase 0a)

**Fichiers :** `src/platform/contract.jl` (nouveau), `src/KrakeNN.jl` (include+export).
**Contenu :** `abstract type AbstractProblem/AbstractMethod/AbstractSolution/AbstractObservable/
AbstractClosure end` ; enum/Set `Capability` (`ForwardSolve, GPUExecution, SteadyAdjoint,
TransientAdjoint, FiniteDiff, NeuralClosure`) ; `capabilities(::AbstractMethod)=Set()`.
**Test :** compile + `capabilities` par défaut vide. **Aucune** logique métier.
**Risque :** nul (types vides). **Taille :** ~80 LOC.

## PR 2 — `LBMSolution` + `solve` wrapper (Phase 0b)

**Fichiers :** `src/platform/solution.jl`, `src/platform/sample.jl`.
**Contenu :** `struct LBMSolution <: AbstractSolution` qui contient le NamedTuple
`(ρ,ux,uy,setup)` actuel ; `struct LBM <: AbstractMethod` ; `capabilities(::LBM)=
{ForwardSolve,GPUExecution,SteadyAdjoint}` ; `solve(problem, ::LBM, p) = LBMSolution(run_simulation(...))`.
`sample(::LBMSolution, field, query)` = `extract_line`/`probe` redirigés.
**Test (bloquant) :** `contract_parity_test.jl` — pour 3 cas existants, `solve(...)` rend des
champs **bit-identiques** à `run_simulation(...)`.
**Risque :** faible. **Garde :** ne PAS modifier `simulation_runner.jl` (juste l'appeler).

## PR 3 — Observables & predict (Phase 1)

**Fichiers :** `src/platform/observe.jl`.
**Contenu :** `AbstractObservable`, `struct Prediction`, `observe(sol,obs)` via `sample` ;
observables `DragCoefficient`, `FieldProbe`, `LineProfile`, `NusseltNumber` ;
`predict(problem,method,p)=observe(solve(problem,method,p),obs)`.
**Test :** `observe` reproduit `value` de `steady_shape_sensitivity` à tol machine.
**Garde :** revue — interdiction d'accès direct aux arrays internes.

## PR 4 — Seam résidu + VJP (Phase 2a)

**Fichiers :** `src/platform/residual.jl`, branchement sur `ext/KrakeNNADExt.jl` (existant).
**Contenu :** `residual(problem, ::LBM, u, p) = u .- G(u,p)` (expose l'implicite,
BILAN §A1) ; si `SteadyAdjoint ∈ capabilities`, exposer `(∂R/∂u)ᵀv`, `(∂R/∂p)ᵀv` en réutilisant
`apply_GtT` / la pile Richardson→GMRES de `ad_adjoint.jl`.
**Test :** les VJP exposés == ceux utilisés en interne par `steady_shape_sensitivity` (égalité).
**Risque :** confondre résidu point-fixe vs PDE. **Garde :** `residual` est **par-méthode**, pas symbolique.

## PR 5 — ParameterSpace + loss + fit (Phase 2b, tête de pont)

**Fichiers :** `src/platform/calibration.jl`.
**Contenu :** `struct ParameterSpace` (named↔flat, log-scale, bornes, fixed/free) ;
`loss(problem,method,p,data)` ; `fit(problem,method,data,p0;pspace)` qui consulte
`capabilities()` → gradient AD si `SteadyAdjoint`, sinon FiniteDiff.
**Milestone/benchmark (bloquant) :** `test/platform/benchmarks/inverse_geom_newtonian/` —
inverse géométrique Newtonien, gradient AD, convergence + gradient de réf documentés. Puis
**rejouer `fit` sur thermique et VE** (mêmes fichiers, méthode/physique différentes) pour prouver
la généralité (cf. `01-PLAN-DE-MATCH.md` Voie G).
**Risque :** VE instable. **Garde :** VE est cas de validation dur, **hors chemin critique**.

## PR 6+ — Méthodes forward existantes derrière le contrat (parallélisable)

Envelopper, sans les rendre différentiables : `VoFMethod`, `PhaseFieldMethod`, `FVFD_VE` comme
`AbstractMethod` forward avec `capabilities = {ForwardSolve, GPUExecution}`. Une PR par méthode,
chacune avec son test de parité vs le driver actuel. **C'est ce qui rend la plateforme générale**
dès le départ — plusieurs méthodes derrière un seul contrat.

---

## Ordre de refactor (dette, après contrat posé — pas avant)

1. Scinder `simulation_runner.jl` : extraire le dispatch en **registre de méthodes** typé
   (remplace le matching de chaîne sur `setup.name`) → débloque `capabilities()` réel.
2. Reloger `io/kraken_parser.jl` → `io/krk/` (cohérence module).
3. Réécrire `docs/src/architecture.md` (périmé).

**Ne PAS** faire 1-3 avant que PR1-5 soient vertes (principe : contrat d'abord, refactor ensuite).

---

## Invariants de chaque PR (gate, à encoder en skill — cf. hygiène)

1. Compile + suite existante verte (aucune régression).
2. Test de **parité** vs chemin actuel (le contrat n'altère pas les nombres).
3. Si la PR ajoute une méthode/physique : **benchmark** = cas canonique + réf analytique +
   ordre de convergence + (si différentiable) gradient de référence.
4. Aucune fuite d'implémentation à travers `sample`/`observe`.
5. `.krk`-doable si code-doable (le `.krk` reste canonique).
