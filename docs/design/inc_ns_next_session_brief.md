# Brief de reprise — IncNS : efficacité GPU + documentation (humain & LLM)

> Prompt autonome pour une nouvelle session Claude Code. Coller tel quel, ou pointer la session dessus.
> Projet : `Kraken.jl`, branche `dev/platform`, worktree `Kraken.jl-platform`. Dépôt PUBLIC (technique only, jamais d'IA/LLM/JEI dans commits/issues).

## 0. Démarrage (à faire en premier)

1. Lire la mémoire `kraken-ns-solver-design.md` (point de reprise complet : décisions, résultats, recette Aqua, gotchas).
2. Lire `docs/design/inc_ns_design_spike.md` (design + reframe GPU + analyse honnête elliptique/LBM/Fluent) et `benchmarks/results/{poisson_gpu_aqua_a100,poisson_mg_gpu_aqua_a100,cavity_gpu_aqua_a100}.md` (chiffres GPU).
3. Charger les skills : `kraken-git-ops` (boucle issue + discipline), `kraken-codebase-map`, `orchestrator` (exécution déléguée), `kraken-doc` (doc humaine + LLM + figures dark), `sci-style` (figures publi), `pbs` + `hpc-watch` (Aqua).
4. Workflow imposé : 1 seule validation humaine = LE PLAN ; ensuite autonome jusqu'au merge-on-green. **JAMAIS push.** Choke files (`src/KrakeNN.jl`, `Project.toml`, `io/`, `simulation_runner.jl`) sérialisés. **Missions solveur = Claude direct (PAS Codex `run-engineer.sh` : il a hangé).**

## 1. État actuel (ce qui existe, validé, committé sur dev/platform)

Solveur Navier–Stokes incompressible **stationnaire**, newtonien, laminaire, autonome (PAS encore `AbstractMethod`) :
- `src/solve/poisson.jl` (Poisson cartésien assemblé + CHOLMOD), `poisson_embedded.jl` (cut-cell), `poisson_embedded_fvfd.jl` (adaptateur `FVFDEmbeddedBoundary2D`).
- `src/solve/linear_solve.jl` (seam **factorize-once** backend-paramétrique : `lin_factorize`/`lin_solve!`, CPU CHOLMOD), `linear_solve_cuda.jl` (méthode cuDSS F64).
- `src/solve/poisson_mg.jl` (**multigrille géométrique matrix-free**, KA, red-black GS, Dirichlet+Neumann+pin+Helmholtz σ, O(N) V-cycles).
- `src/fvfd/operators_2d_grad_div_laplacian.jl` (opérateurs ∇·u, ∇p, ∇²u matrix-free KA).
- `src/methods/inc_ns/simple.jl` (Poiseuille), `cavity_mg.jl` (**cavité backend-paramétrique CPU/CUDA**, MG pour pression ET momentum).
- Validations : Poiseuille 0.03% ; cavité Ghia Re=100 **0.69%**, Re=1000 **2.31%** ; portage GPU **parité bit-exacte 1e-16**.

Chiffres GPU mesurés (Aqua A100, F64) — ancrages à NE PAS re-mesurer :
- cuDSS Poisson : solve **30× vs CPU** à 1M DDL, parité 1e-12 ; factorize 3× plus lent mais 1 fois.
- Multigrille matrix-free : **43× vs CPU à 16M DDL** (croît avec N), **sature l'A100 (99% pic)**, V-cycles plats [10-13].
- Cavité Re=1000 512² : GPU 865 s vs CPU 3687 s = **4,3×**, **mais GPU ~20% utilisé pendant le solve** (mesuré ; le 4,5% job-moyen incluait la phase CPU).
- Diagnostic : 31 218 itérations SIMPLE, chacune avec réductions globales (normes) → sync hôte + plein de petits kernels → latence-bound, GPU affamé.

Recette Aqua (env `~/kraken_poisson_gpu_bench` : CUDA+CUDSS+KernelAbstractions déjà installés) : rsync `src/solve/*.jl` + `src/methods/inc_ns/cavity_mg.jl` + bench ; PBS `select=1:ncpus=4:ngpus=1:mem=32GB:gpu_id=A100`, `JULIA_CUDA_USE_COMPAT=false`, `WORKDIR=$HOME/kraken_poisson_gpu_bench` ; `qsub` ; poller `~/kraken_poisson_gpu_bench/<job>.o*`. Gotcha world-age : charger CUDA/CUDSS en statements top-level SÉPARÉS + `Base.invokelatest`.

---

## 2. MISSION A — Efficacité GPU du solveur (« Levier 1 », gain sûr ~15×, croît avec la grille)

But : passer la cavité GPU de ~20% à ~70% d'utilisation A100 → de 4,3× à **~15× vs CPU** à 512² (et 25-40× sur grandes grilles), SANS changer la physique. Garder parité vs CPU + Ghia.

Changements, par ordre de rentabilité (mesurer la contribution de CHACUN) :
1. **Espacer/grouper les normes de convergence** (résidu + velocity-change). Aujourd'hui calculées à chaque itération → sync GPU→hôte 31 000 fois. Les calculer toutes les K itérations (K≈25-50), async/on-device. *Plus gros gain unitaire.*
2. **Fusion de kernels** : fusionner les petits kernels par itération (prédicteur, advection compacte, Rhie-Chow, corrections) en moins/plus gros.
3. **CUDA Graphs** : capturer toute l'itération SIMPLE en un graphe → amortir le coût de lancement.
4. **Précision mixte** (plus risqué, ×2-3) : V-cycles multigrille en FP32 + raffinement itératif FP64. Garder la parité finale en FP64.

Validation : re-run `benchmarks/krk/inc_ns/cavity_gpu_bench.jl` sur Aqua A100 (Re=1000, 512² et 1024² via `CAVITY_BENCH_1024=1`) ; cibler util ↑ et ~15× ; tableau avant/après par changement ; parité GPU↔CPU < 1e-3 et Ghia inchangé.

Suites possibles (noter, ne pas tout faire) : enregistrer `IncNS<:AbstractMethod` + seam `.krk solver=` (choke) ; schéma **couplé/JFNK** (moins d'itérations, ×2-4) ; **driver instationnaire** (projection/PISO) ; **branche viscoélastique** (closure pluggable, réutilise opérateurs log-conf FVFD existants).

NB cap honnête : viser l'**ordre de grandeur Fluent par-GPU** (~30-100× vs 1 cœur) est atteignable en empilant Levier 1 + grande grille + précision mixte (+ couplé) ; le multi-GPU, la robustesse toutes-physiques et la précision mixte robuste restent hors-scope. Détails dans le design note.

---

## 3. MISSION B — Documentation (humain + LLM + benchmarks), via `kraken-doc`

Produire la doc end-to-end du solveur incompressible, **deux volets**.

### Volet HUMAIN (Track A, DocumenterVitepress, figures dark `krakendark`/`sci-style`)
- Page « Incompressible steady Navier–Stokes (FVFD/SIMPLE) » : ce que c'est, **quand l'utiliser** (régime elliptique/steady vs LBM unsteady — réutiliser l'explication « elliptique = couplage global instantané, incompressible = son infini » de cette session), comment lancer.
- Section validation : Poiseuille (profil parabolique) + cavité Ghia Re=100/1000 — **figures** profils centreline vs Ghia 1982.
- Section **benchmarks GPU** : figures (1) speedup GPU/CPU vs taille de grille, (2) V-cycles multigrille plats = O(N), (3) cuDSS vs multigrille vs cavité, (4) utilisation GPU. Inclure l'**analyse de perf honnête** : compromis few-global-vs-many-local, Levier 1 vs compressibilité artificielle, comparaison ordre-de-grandeur Fluent. Source des chiffres : `benchmarks/results/*.md`.

### Volet LLM/TECHNIQUE (Track C + Track B, lint-bloquant)
- **Cartes d'implication (Track C)** pour chaque module : `src/solve/{poisson,poisson_embedded,poisson_embedded_fvfd,linear_solve,poisson_mg}.jl` et `src/methods/inc_ns/{simple,cavity_mg}.jl`. Respecter le gate de lint (`julia make.jl`).
- **Docstrings (Track B)** sur les fonctions publiques (`solve_poisson*`, `lin_factorize`/`lin_solve!`, `solve_poisson_mg`, `solve_incns_cavity_mg`, opérateurs grad/div/lap).
- Capturer le **récit de perf** (explication elliptique, calcul de gain robuste, analyse Fluent) dans un doc benchmark — précieux pour humains ET futures sessions LLM.

Figures : worktree canonique doc = `Kraken-release-v0.2` (release/v0.2) ; mais le code IncNS vit sur `dev/platform`. Décider avec le plan où les pages atterrissent (probable : rédiger sur dev/platform, intégrer à la doc lors d'un merge ultérieur).

---

## 4. Critères de succès / livrables

- **Mission A** : cavité GPU 512² à ~15× vs CPU (util ↑ vers ~70%), parité + Ghia préservés, contribution de chaque changement chiffrée, bench Aqua re-run et committé avec artefact résultats.
- **Mission B** : page(s) humaine(s) (install/usage/validation/benchmarks avec figures dark), cartes d'implication + docstrings passant le lint, note benchmark avec l'analyse de perf honnête.
- Tout sur `dev/platform`, commits conventionnels référençant `#7`/`#8`, **jamais push**, choke files sérialisés. Mémoire `kraken-ns-solver-design.md` mise à jour.

## 5. Ordre suggéré

Mission A d'abord (le solveur efficace donne les bons chiffres à documenter), puis Mission B (qui documente l'état final). Présenter UN plan combiné, attendre la validation, puis exécuter en autonomie via l'orchestrator + Claude direct, valider sur Aqua, commit-on-green.
