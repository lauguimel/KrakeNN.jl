# BILAN — état réel du code vs. cadrage plateforme

> Audit adversarial, lecture seule, ancré dans le code (`fichier:ligne`). Daté 2026-06-09.
> Objectif : vérifier/réfuter chaque hypothèse du brainstorm avant d'écrire la moindre
> ligne d'infrastructure. **On construit la plateforme GÉNÉRALE ; le VE est un exemplaire,
> pas le centre de gravité.**

---

## A. Corrections au brief (adversarial)

Le brainstorm contient plusieurs hypothèses que le code **réfute**. Elles changent le plan.

| # | Affirmation du brief | Verdict | Réalité du code |
|---|----------------------|---------|-----------------|
| A1 | « La primitive partagée est le **résidu `R(u,p)`** » | **ASPIRATIONNEL** | Aucun `R(u,p)` nommé n'existe. La primitive réelle est le **pas de temps `G`** (collide-stream) et son VJP Enzyme. L'adjoint steady résout le point fixe `(I−Gᵀ)λ = ∂J/∂w` (`ad_adjoint.jl:18-22`, Richardson→GMRES). Bonne nouvelle : `R(u)=u−G(u)` est **déjà calculé implicitement** (`flat_w .- vec(apply_GtT(w))`) → exposer le seam coûte peu, mais ce n'est **pas** déjà fait. |
| A2 | « Calibrer un **paramètre matériau** VE/thermique avec l'AD existant » (milestone 1) | **RÉFUTÉ — bloquant** | L'AD ne différencie **QUE la géométrie** (`:radius`, `:wall_position`). λ, η, Wi, β, ν, Ra, Pr sont passés en `Const` et **explicitement rejetés** comme `wrt` (`ad_api.jl:356-361`). Calibrer un paramètre matériau exige `∂J/∂(param)` qui **n'existe pas**. → Milestone 1 doit être un **inverse géométrique**, ou bien étendre l'adjoint aux paramètres (vrai travail). |
| A3 | « LBM + FV/FD VE différentiable en steady » | **PARTIEL/trompeur** | Différentiable = **LBM D2Q9 seul**, CPU Float64, 3 physiques. Le « FV/FD VE » différencié est en fait l'advection ψ sur le masque LBM (`ad_ve_ops.jl:393`) — **aucune** discrétisation FV/FD n'est différenciée. |
| A4 | « quelques **briques** FV/FD » | **SOUS-ESTIMÉ** | `src/fvfd/` est une vraie bibliothèque d'opérateurs cell-centered (dérivées 1er/2e ordre solid/BC-aware, moyennage cell→face, advection upwind/MUSCL, divergence tensorielle + traction murale) — `operators_2d*.jl`. 3D dans la branche ve-3d. |
| A5 | implicite : « l'analogue Basilisk (VoF/interface) reste à faire » | **RÉFUTÉ** | Stack interface **déjà substantiel** : PLIC VoF (`kernels/vof_2d.jl`), Allen-Cahn (`phasefield_2d.jl`), MRT diphasique + tension de surface (`collide_twophase_rheology_2d.jl`), ghost-fluid, VoF dual-grid ; drivers `run_twophase_vof.jl`, `multiphase.jl` ; dispatch `simulation_runner.jl:21-22`. |
| A6 | implicite : « l'actif AD vit sur la branche `dev/ad-ve` » | **RÉFUTÉ** | `src/ad/` et `ext/KrakeNNADExt.jl` sont **octet-identiques** entre `dev/ad-ve` et le trunk `dev/v0.3-campaign` (`diff -rq` propre). **L'AD est déjà sur le trunk.** Le port n'est plus un sujet. |
| A7 | « FV/FD = fournisseur de résidus différentiables » | **FAUX aujourd'hui** | Le chemin FVFD est un **stepper explicite ségrégé** (faces→advect→subcycle→stress→force→stream→collide→swap, `viscoelastic_logfv_obstacle_bfs_2d.jl:147-202`). Aucune forme résidu. |
| A8 | FEM | **CONFIRMÉ absent** | Zéro `FEM`/`assemble`/`stiffness`/`galerkin` dans `src/`. |

**Conséquence générale (répond à ta correction « rester général ») :** la capacité différentiable
n'est **pas** spécifique au VE — elle est **géométrique et déjà transverse à 3 physiques**
(Newtonien, thermique, VE) via le **même mécanisme** (VJP du pas LBM). Le VE est seulement
le cas le plus *dur* (annulation catastrophique ~20×, `fwd_tol=1e-13`). Donc la tête de pont
naturelle est **« sensibilité/inversion géométrique steady, agnostique de la physique »**,
démontrée d'abord sur le **Newtonien** (le plus simple), VE/thermique venant « gratuitement »
par le même seam. Voir `01-PLAN-DE-MATCH.md`.

---

## B. Matrice de capacités (fondée sur le code)

Méthode × physique × type de sensibilité. `SA`=SteadyAdjoint, `FS`=ForwardSolve,
`FD`=FiniteDiff (vérif only), `TA`=TransientAdjoint.

| Méthode | Physique | Forward (prod) | SA | FS | FD | TA |
|---------|----------|----------------|----|----|----|----|
| LBM D2Q9 | Newtonien (Cd / `:radius`) | ✅ GPU+CPU | ✅ `ad_api.jl:121` | — | vérif | ❌ |
| LBM D2Q9 | Thermique nat-conv (Nu / `:wall_position`) | ✅ | ✅ mass-gauged `:229` | — | vérif | ❌ |
| LBM D2Q9 | Oldroyd-B VE (Fx / `:radius`) | ✅ | ✅ `:373` | JVP dG/dR `:269` | vérif | ❌ |
| LBM (MRT) | Diphasique VoF / phase-field | ✅ (transient) | ❌ | — | — | ❌ |
| FV/FD | Transport log-conf VE (2D ; 3D branche) | ✅ | ❌ | — | — | ❌ |
| FV/FD | Thermique | opérateurs présents mais le thermique tourne sur **kernels LBM fusionnés** | ❌ | — | — | ❌ |
| FEM | — | ❌ | ❌ | — | — | ❌ |

**Différentiable aujourd'hui = strictement : LBM steady, QoI géométrique, CPU Float64.**
Tout le reste (matériau, transitoire, FV/FD, VoF) est **non** différentiable.

---

## C. Le contrat vs. le code actuel

Aucun des 5 verbes proposés n'existe ; le solveur renvoie un **NamedTuple brut** `(ρ,ux,uy,setup)`
(`simulation_runner.jl:339`) ; la « méthode » est un enchevêtrement de symboles `modules` +
matching de chaîne sur `setup.name` dans un arbre de dispatch de ~200 lignes (`:151-178`).

| Verbe contrat | Aujourd'hui | Effort | Note |
|---------------|-------------|--------|------|
| `solve(problem, method, p)::AbstractSolution` | `run_simulation(setup)` → NamedTuple | **moyen** | pas de type `AbstractMethod`/`AbstractProblem`/`AbstractSolution` (grep vide) |
| `sample(sol, field, query)` | `extract_line`, `probe` (`postprocess.jl:22,130`) sur le NamedTuple | **facile** | la logique existe, manque le type `Solution` pour dispatcher |
| `observe(sol, obs)::Prediction` | `field_error`, `domain_stats`, `steady_shape_sensitivity` | **moyen** | concepts éparpillés, pas de `Prediction` |
| `capabilities(m)` | **rien** (grep vide) — support vérifié réactivement par `throw(ArgumentError)` | **moyen** | pas d'introspection → bloque la couche NL |
| `residual(problem, method, u, p)` | seulement `ad_relative_step_residual` (delta de convergence) | **moyen** | `R(u)=u−G(u)` implicite dans l'adjoint, à exposer (cf. A1) |
| `ParameterSpace` / `fit` / `loss` / `predict` | **n'existent pas** (grep vide) | **moyen** | params = scalaires keyword plats, géométrie seule, un seul gradient (pas de boucle d'optim) |

**Verrou structurel #1 :** le solveur rend des tableaux bruts, pas un objet Solution.
Introduire `AbstractSolution` (wrapper non-cassant autour du NamedTuple existant) est la
**première brique** : elle débloque `sample`/`observe` sans toucher aux kernels.

---

## D. JSON / NL-readiness

- **Zéro JSON/sérialisation** dans `src/` ou `Project.toml` (grep vide).
- `SimulationSetup` (20 champs, `parser.jl:183-214`) est **partiellement** sérialisable :
  scalaires/symboles OK, mais embarque des **AST `KrakenExpr`** (`body_force`, `condition`,
  `bc_values`, `InitialSetup.fields`) et un champ `mesh::Any`. Pas de handles device/closures
  *dans* le struct (les tableaux sont alloués dans le runner) → bon point.
- **Conséquence (principe #7 du brief, confirmé faisable) :** l'enveloppe problème-inverse
  (data/observables/paramètres/loss/résultat = nombres plats) est **directement sérialisable** ;
  le modèle forward reste **par-méthode** — et le **texte `.krk` EST le blob `config`** naturel,
  pas besoin d'un IR universel.
- **Erreurs : bare exceptions**, pas de codes, pas d'introspection. Les seuls types custom
  (`PlanValidationError`, `NotImplementedError`) vivent dans le sous-module isolé `Units`
  (`Units.jl:17,173`) et ne sont pas sur le chemin solveur. Une couche NL devrait aujourd'hui
  parser du texte libre → cf. axe NL (`03-EXPLORATION-AXES.md`).

---

## E. Inventaire des bloqueurs transient-AD (classés)

Le transient-AD est un **piège** ; il faut l'isoler (principe #4 du brief, correct). Sources
réelles, par sévérité :

| # | Sév | Site | Pourquoi ça bloque le reverse-through-time |
|---|-----|------|--------------------------------------------|
| 1 | CRIT | sous-cyclage VE + swap 3-voies (`viscoelastic_logfv_obstacle_bfs_2d.jl:170-184`) | **adaptatif jusqu'à 64×** sous-pas/pas LBM (`logconformation_fv_2d.jl:220-269`) → 64× le tape ; alias-swap mute les tableaux liés |
| 2 | CRIT | stream/collide in-place + ping-pong `f_in,f_out=f_out,f_in` (`:198-201`) | boucle temporelle non pure |
| 3 | HIGH | kernels `fvfd/*` `@kernel … !` écrivant en place | règles reverse manuelles ou historique complet de buffers |
| 4 | HIGH | réutilisation de buffers persistants (`:106-143`) | aliase les valeurs du pas précédent que l'adjoint doit reconstruire |
| 5 | MED | swaps `w_in,w_out` dans `ad_*_forward` | déjà mitigé en steady (GMRES, pas de tape) ; rebloque si réutilisé en transient |

**Atténuant clé :** les kernels FVFD sont **gather/par-cellule, aucun `@atomic`/`Atomix`/scatter
dans `src/`** → propriété GPU race-free préservée. Le blocage est **mutation/ping-pong/sous-cyclage**,
pas le scatter — ce qui rend viable la route « array/XLA façon XLB/Reactant » mentionnée au brief.

---

## F. Actifs généraux sous-estimés (utiles pour rester général)

- **Capture d'interface (analogue Basilisk) déjà là** : VoF PLIC, phase-field, diphasique MRT +
  tension de surface. Forward seulement, transitoire, non différentiable — **à exposer derrière
  le contrat comme méthode forward**, sans tenter de le rendre différentiable tôt.
- **Bibliothèque d'opérateurs FV/FD réelle** (2D + 3D) : matière première pour un **fournisseur
  de résidus** (A7) — mais il faut écrire la *forme résidu*, elle n'existe pas.
- **AD géométrique transverse 3 physiques** via un seul mécanisme → généralité déjà prouvée
  côté sensibilité (≠ VE-spécifique).
- **`.krk` DSL mature** (`src/io/krk/`) = surface canonique sur laquelle le JSON inverse se greffe.

---

## G. Dette technique repérée (à traiter, cf. `04-HYGIENE-PROCESS.md`)

- `simulation_runner.jl` ~1843 LOC, dispatch par matching de chaîne sur `setup.name` → fragile,
  empêche `capabilities()`/typed-method.
- `io/kraken_parser.jl` ~2010 LOC, mal placé (devrait être sous `io/krk/`).
- `docs/src/architecture.md` **périmé** (décrit l'ancien layout LBM-only 8 modules).
- AMR conservative-tree / multiblock-exchange : tests **rouges** sur v0.3 (dette release).
- `dev-viscoelastic` HEAD **non-buildable** (includes non trackés).
- ~132 clés `refs.bib` pendantes avant tout tag release.
- Worktrees désynchronisés (déplacement sous `kraken/` sans `git worktree repair`).
