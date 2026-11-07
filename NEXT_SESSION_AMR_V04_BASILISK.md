# Prompt - AMR v0.4 Basilisk-inspired, efficace, leger, test-first

## Mission

Reprendre le chantier AMR de Kraken.jl dans:

```text
/Users/guillaume/Documents/Recherche/Kraken.jl
branche: slbm-paper
```

Objectif: construire une v0.4 AMR inspiree de Basilisk, mais sans big rewrite
aveugle. La priorite n'est pas d'ecrire une architecture abstraite enorme; la
priorite est de fermer les invariants physiques et numeriques qui bloquent
encore l'AMR actuel:

- subcycling coarse/fine correct;
- conservation stricte;
- rest-state preservation;
- obstacle square/cylinder proche de l'oracle leaf;
- compatibilite AD pour optimisation de forme et sensibilites;
- extensibilite multi-level et GPU sans casser le chemin CPU valide.

La v0.4 doit cohabiter avec la voie D actuelle jusqu'a parite. Ne jamais
supprimer `src/refinement/conservative_tree_*.jl` pendant le developpement.

## Etat reel actuel a respecter

Lire d'abord:

```text
docs/design/amr_route_native_progress.md
docs/design/amr_complete_project_plan.md
src/refinement/conservative_tree_topology_2d.jl
src/refinement/conservative_tree_streaming_2d.jl
test/test_conservative_tree_obstacle_interface_2d.jl
test/test_conservative_tree_open_boundary_2d.jl
test/test_conservative_tree_subcycling_2d.jl
```

Etat valide:

- open-channel/BFS 2D est maintenant solide avec
  `coarse_route_mode=:leaf_equivalent`;
- le canal ouvert au repos reste au repos;
- BFS AMR suit l'oracle leaf sur aqua;
- square/cylinder conservent la masse au roundoff;
- square/cylinder restent encore trop eloignes de l'oracle physique;
- le diagnostic obstacle indique que l'ecart vient surtout du transport
  coarse/fine non-subcycle accumule, pas du drag MEA ni d'un bug solide isole.
- le repo contient deja `ForwardDiff`, `Enzyme`, `EnzymeCore` et
  `src/kernels/enzyme_rules.jl` pour les derivees analytiques de geometrie
  type `dq_wall/dR` couplees a Enzyme.

Derniers commits importants:

```text
39caf6b Record AMR obstacle leaf-equivalent aqua ladder
4e2eb87 Record AMR obstacle leaf-equivalent ladder
15da099 Use leaf-equivalent AMR routes for obstacles
245f6ec Record AMR leaf-equivalent aqua benchmarks
78149fb Record AMR BFS leaf-equivalent canary
2a4c9a4 Fix AMR open-channel rest-state routes
87d6290 Add AMR open-channel rest-state canary
```

## Regles de travail

- Toujours commencer par `git status --short`.
- Le worktree peut etre sale hors AMR. Ne jamais revert ce qui n'a pas ete
  touche par cette session.
- `src/KrakeNN.jl` peut contenir des changements hors perimetre: ne pas le stage
  sans inspection explicite.
- Commits petits, en anglais, conventional si possible:
  `feat:`, `fix:`, `test:`, `docs:`, `refactor:`.
- Avant chaque macro-flow, ajouter ou relancer un patch test chirurgical.
- Si un macro-flow diverge, redescendre a un test 1-step/rest-state/route packet.
- Ne pas chercher a tout generaliser en 2D/3D/GPU/multiphysique des le premier
  commit. Generaliser seulement apres invariants CPU 2D verts.
- Ne pas melanger adaptation discrete et gradient AD dans le meme premier
  chemin. L'AD v04 doit d'abord fonctionner a topologie gelee.

Validation de base voie D avant gros changement:

```bash
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_topology_2d.jl"); include("test/test_conservative_tree_streaming_2d.jl"); include("test/test_conservative_tree_open_boundary_2d.jl"); include("test/test_conservative_tree_obstacle_interface_2d.jl"); include("test/test_conservative_tree_subcycling_2d.jl")'
```

Validation partagee avant commit structurel:

```bash
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_gpu_pack_2d.jl"); include("test/test_conservative_tree_adaptation_2d.jl"); include("test/test_conservative_tree_multipatch_2d.jl"); include("test/test_conservative_tree_2d.jl")'
```

## Decision d'architecture

### Ce que "Basilisk-like" veut dire ici

On veut les idees utiles de Basilisk:

- cellules cell-centered;
- parent/enfants explicites;
- niveaux explicites;
- 2:1 balance;
- adaptation locale;
- voisinage topologique fiable.

On ne veut pas copier litteralement Basilisk si cela rend Kraken lourd ou lent.
Le design doit rester compatible Julia/Kraken, testable par etapes, et portable
CPU/GPU.

### Ne pas faire un dense all-level pool global en production

Un pool dense complet par niveau est acceptable pour un prototype 2D et des
tests, mais il n'est pas viable comme seule solution 3D Lmax>=5. En 3D, avec
D3Q19, double buffers, champs multiples et Lmax eleve, la memoire explose.

Approche retenue:

1. **Phase CPU 2D minimale**: representation simple, claire, compatible avec
   les tests et les primitives actuelles.
2. **Phase production**: block-pool sparse par niveau, capacity preallouee,
   free stack preallouee pour adaptation. Pas d'allocation dans le hot loop.
3. **Phase GPU**: layout packed actif, route tables packees, masks GPU-friendly.

Calendrier explicite:

- Phases 2-6, Lmax=2: dense per-level ou arrays simples autorises pour fermer
  vite les invariants.
- Debut Phase 7, Lmax>=3: migration vers block-pool obligatoire avant
  d'ajouter de nouveaux macro-flows. Rejouer les tests Phases 2-6 sur le
  block-pool avant de continuer.
- Phase 7+: documenter le budget memoire 2D et 3D au constructeur, incluant
  populations, metadata, route tables, ledger subcycling et buffers AD/GPU si
  presents.

### Ne pas supposer que `F[i,j,q]` est AoS contigu

Julia est column-major. Dans `F[i,j,q]`, les populations `q` d'une cellule sont
stridees, pas contigues. Il faut mesurer le layout avant GPU.

Layouts a benchmarker avant de figer:

- `F[i,j,q]`: compatible avec l'existant, simple CPU;
- `F[q, cell]`: q contigu par cellule, bon si un thread traite toutes les
  populations d'une cellule;
- `F[cell, q]` ou SoA packed: potentiellement meilleur si les threads traitent
  le meme q sur cellules voisines;
- buffers flat `Vector{T}` avec indexeurs inlines.

Regle: le design expose une API de layout, pas un layout grave dans le marbre.
Le layout GPU final doit etre choisi par microbench dans le repo.
Le choix CPU de Phase 0 est revisitable en Phase 11 si les benchmarks device
Metal/CUDA contredisent le compromis CPU; toute migration de layout doit avoir
un ADR court et un test de roundtrip.

### Reutiliser les lattices existants

Le repo a deja:

```text
src/lattice/d2q9.jl: struct D2Q9 <: AbstractLattice{2,9}
```

Ne pas recreer un type concurrent `D2Q9`. La v0.4 doit fournir des adapters et
helpers generiques autour des lattices existants (`AbstractLattice{D,Q}`).

### Integrer AD sans alourdir l'AMR

AD veut dire ici automatic differentiation pour sensitivites, calibration et
optimisation de forme. Ce n'est pas un pretexte pour rendre tout l'AMR
differentiable des le depart.

Approche retenue:

1. **Frozen-tree AD**: on gele la topologie, les masks, les routes et les
   ownerships. On differencie seulement les valeurs continues transportees par
   cette topologie.
2. **Design-loop adapt**: l'adaptation reste hors tape AD. Une iteration
   optimise un design sur arbre gele, puis regrid si necessaire, puis relance
   une passe AD.
3. **Discrete relax later**: les seuils de raffinement et decisions refine /
   coarsen ne sont pas differentiables au debut. Si besoin, ajouter plus tard
   des indicateurs lisses, mais sans bloquer l'AMR hydro.

Parametres AD autorises en premier:

- rayon/centre de cylindre via `q_wall`;
- position d'obstacle via SDF ou fractions de cut-link;
- viscosite, forcing, inlet amplitude;
- vitesse de paroi type Couette/Taylor-Couette;
- coefficients de criteres continus, si la topologie reste gelee.

Parametres a exclure du premier chemin AD:

- creation/destruction de cellules;
- changement de niveau;
- `Bool` masks depends de Dual/adjoints;
- route table construite sous AD;
- seuils de raffinement discontinus.

Implementation style:

- `ForwardDiff` pour derivees locales de geometrie/metrics et verifications
  finite-difference;
- `Enzyme` pour reverse-mode sur boucles de temps in-place quand les kernels
  sont stabilises;
- chain rule explicite pour geometrie non lisse ou discontinue, en reutilisant
  l'esprit de `src/kernels/enzyme_rules.jl`;
- pas de dependance Zygote dans le hot path AMR.

Contrats code AD-friendly:

- fonctions numeriques parametrees par `T<:Real`, pas de `Float64` force;
- constantes ecrites `T(0)`, `T(1)`, etc. dans les kernels generiques;
- separation stricte entre tables entieres/topologie et valeurs physiques;
- objectifs scalaires purs: drag, lift, debit, perte de masse, energie;
- reductions deterministes, testables CPU d'abord.

Representation obstacle pour AD:

- un `Bool` solid mask est une decision topologique, pas une variable
  differentiable;
- le cylindre AD doit passer par une representation continue: fractions de
  cut-link, fraction fluide par cellule, ou SDF lissee;
- le chemin v04 doit fournir un helper de drag compatible poids continus
  (`solid_weight`/`fluid_fraction`) au lieu d'imposer uniquement un mask `Bool`;
- `compute_drag_mea_solid_F_2d` peut rester le chemin voie D, mais la v04 AD
  doit ajouter un equivalent pondere avant tout gradient rayon/centre.

## Architecture cible minimale

### Module v04

Creer un module parallele:

```text
src/refinement/v04/
  AMRv04.jl
  layout.jl
  tree.jl
  field.jl
  topology_pull.jl
  streaming_cpu.jl
  subcycling.jl
  adaptation.jl
  ad.jl
  diagnostics.jl
  gpu_pack.jl
```

Tests:

```text
test/v04_test_layout.jl
test/v04_test_tree_2d.jl
test/v04_test_topology_pull_2d.jl
test/v04_test_streaming_cpu_2d.jl
test/v04_test_subcycling_2d.jl
test/v04_test_obstacles_2d.jl
test/v04_test_ad_geometry_2d.jl
test/v04_test_ad_frozen_tree_2d.jl
test/v04_test_gpu_pack_2d.jl
```

Ne pas exporter largement depuis `src/KrakeNN.jl` au debut si ce fichier est
dirty hors perimetre. Les tests peuvent inclure le module v04 directement ou
utiliser un export minimal dans un commit dedie apres inspection.

### Types coeur

#### Cell id compacte

Utiliser un id stable, compact et directement indexable:

```julia
struct AMRCellId
    value::Int32
end
```

Decision:

- `Int32` est le format interne production tant que la capacity reste sous
  `typemax(Int32)`;
- `level`, `block id`, `local id` vivent dans les metadata du block-pool;
- `UInt64` n'est a introduire que plus tard pour serialization stable ou
  simulations depassant explicitement la borne `Int32`.

#### Block-tree field

Production cible:

```julia
struct AMRBlockTreeField{T,D,L,K,Layout}
    lattice::L
    kind::K
    layout::Layout
    lmax::Int
    block_size::NTuple{D,Int}

    # metadata preallouee
    level::Vector{Int8}
    parent::Vector{Int32}
    children::Matrix{Int32}      # nchildren x nblocks, 0 si absent
    origin::Matrix{Int32}        # D x nblocks, coord coarse/fine selon level
    active_leaf::Vector{UInt8}   # GPU-friendly, pas BitArray pour backend GPU
    free_stack::Vector{Int32}    # preallouee, utilisee seulement pendant adapt

    # population storage backend-specific
    F_in
    F_out
end
```

Pour la premiere phase 2D CPU, autoriser une implementation plus simple:

- un seul block par level ou quelques blocks explicites;
- `Array` CPU;
- pas de GPU;
- invariants et topology d'abord.

Ne pas promettre "zero allocation" partout. Promettre plutot:

- zero allocation dans `stream!` et `collide!`;
- allocations autorisees dans `build_topology!` au debut;
- puis preallocation/capacity pour `rebuild_topology!` avant GPU.

### Pull route table

Le pull streaming doit etre la direction cible GPU. Une route est definie pour
chaque destination active cell et chaque population `q_dst`.

CSR correct:

```julia
struct PullRouteTable{T}
    # offsets length = n_active_cells * Q + 1
    offsets::Vector{Int32}

    # entries length = nnz routes
    src_cell::Vector{Int32}
    src_q::Vector{Int16}
    weight::Vector{T}
    kind::Vector{UInt8}
end
```

Pourquoi `src_q` est necessaire:

- normal streaming: `src_q == q_dst`;
- bounce-back wall/solid: `src_q == opposite(q_dst)`;
- certaines BCs peuvent reconstruire une population depuis plusieurs q.

Ne pas utiliser un pseudo `F_in_levels[gid]` ambigu. Il faut une fonction
inline:

```julia
@inline getpop(field, cell::Int32, q::Int)
@inline setpop!(field, cell::Int32, q::Int, value)
```

Le backend CPU peut utiliser cette abstraction; le backend GPU recevra des
arrays packes.

### 2:1 balance

Implementer deux niveaux de check:

1. `check_2_to_1(field)` pur diagnostic, retourne rapport structuré;
2. `enforce_2_to_1_marks!(marks, field)` propage les refinements nécessaires.

Ne pas muter silencieusement pendant un check. Les throws doivent inclure:

- level;
- coord;
- voisin fautif;
- difference de level.

### Subcycling

C'est le verrou principal. Ne pas repousser subcycling apres GPU.

Contrat minimal ratio 2:

- coarse level avance 1 pas;
- fine level avance 2 demi-pas;
- coarse-to-fine: injection/prolongation temporelle cohérente;
- fine-to-coarse: reflux/ledger conservatif;
- rest-state preserve;
- masse et momentum preserve a roundoff sur interface isolee;
- obstacles/cylinder re-evalues seulement apres ces gates.
- AD gradients autorises seulement apres ces gates, sinon on differencie un
  bug de transport.

Reutiliser le diagnostic existant:

```text
src/refinement/conservative_tree_subcycling_2d.jl
test/test_conservative_tree_subcycling_2d.jl
```

Ne pas se contenter d'appeler le kernel pull plusieurs fois par level. Il faut
un vrai echange temporel coarse/fine et un ledger/reflux.

## Phases d'implementation

Chaque phase doit avoir tests verts et commit separe. Ne pas lancer la phase
suivante si la precedente ne passe pas.

### Phase 0 - ADR et microbench layout

Livrables:

- `docs/design/amr_v04_status.md`
- `docs/design/amr_v04_layout_adr.md`
- `benchmarks/amr_v04_layout_microbench.jl`

Comparer au minimum:

- `F[i,j,q]`;
- `F[q,cell]`;
- flat packed active cells;
- CPU local;
- CUDA H100 si disponible;
- Metal local si disponible.

Gate:

- au moins 3 layouts mesures sur le meme hot loop BGK + stream same-level;
- baseline = LBM cartesien dense Kraken existant;
- viser >= 70% du baseline MLUPS sur CPU local pour le layout retenu;
- si aucun layout ne passe 70%, documenter le bottleneck et ajuster la cible
  avant GPU;
- documenter le layout choisi pour CPU phase 1;
- documenter le layout cible GPU;
- ne pas pretendre AoS/SoA sans mesure.

### Phase 1 - Lattice adapters et helpers generiques

Livrables:

- `src/refinement/v04/lattice_adapters.jl`
- tests `test/v04_test_lattice_adapters.jl`

Contraintes:

- reutiliser `AbstractLattice{D,Q}`, `D2Q9`, `D3Q19` existants;
- ajouter accessors generiques `amr_dim`, `amr_nq`, `amr_c`, `amr_weight`,
  `amr_opposite`;
- pas de nouveau `D2Q9` global.

Gate:

- `opposite(opposite(q)) == q`;
- weights sum;
- type-stability smoke;
- zero allocation accessors.

### Phase 2 - Tree 2D minimal et 2:1 balance

Livrables:

- `src/refinement/v04/tree_2d.jl` ou generic avec specialisation 2D claire;
- tests `test/v04_test_tree_2d.jl`.

Support initial:

- base grid;
- refine/coarsen manuel;
- parent/children;
- leaf active mask;
- 2:1 check et mark propagation.

Gate:

- refine/coarsen conserve leaf ownership;
- neighbor levels corrects;
- violation 2:1 detectee avec coords;
- pas de macro-flow.

### Phase 3 - PullTopology 2D Lmax=2

Livrables:

- `src/refinement/v04/topology_pull_2d.jl`
- tests `test/v04_test_topology_pull_2d.jl`

Cas obligatoires:

- domaine pur level 1;
- une region raffinee level 2;
- same-level;
- coarse -> fine;
- fine -> coarse;
- periodic x;
- wall y;
- solid bounce-back;
- open boundary plus tard, apres rest-state.

Gate:

- pour chaque destination active cell et q: routes non vides;
- somme des poids correcte pour les cas de conservation;
- rest-state topology localisable;
- route table deterministe.

### Phase 4 - Pull streaming CPU 2D, sans subcycling

Livrables:

- `src/refinement/v04/streaming_cpu_2d.jl`
- tests `test/v04_test_streaming_cpu_2d.jl`

Tests chirurgicaux:

- one-step rest-state, no obstacle;
- one-step rest-state, obstacle inside fine;
- one-step rest-state, obstacle straddling interface;
- route-vs-leaf oracle localizer;
- mass/momentum ledger.

Gate:

- rest-state preserve a roundoff;
- domaine level 1 bit-identique au chemin leaf;
- kernels parametrises par `T<:Real` des l'ecriture initiale;
- constantes physiques ecrites `T(0.5)`, `T(1)`, etc., aucun `Float64`
  hardcode dans les boucles physiques;
- smoke `Float64` et `Float32` passent le rest-state;
- pas encore de claim obstacle convergence.

### Phase 5 - Subcycling 2D ratio 2

Livrables:

- `src/refinement/v04/subcycling_2d.jl`
- tests `test/v04_test_subcycling_2d.jl`

Tests obligatoires:

- coarse-to-fine packet consomme exactement une fois sur deux demi-pas;
- fine-to-coarse reflux accumule sans double count;
- rest-state preserve apres 1/10/100 coarse steps;
- Guo forcing faible ne cree pas de drift interface non physique;
- comparaison voie D / oracle sur patch simple.

Gate:

- Phase 5 valide seulement `omega_coarse == omega_fine`;
- le rescaling Filippova-Hanel de `tau` devient obligatoire en Phase 7 quand
  le multi-level introduit `omega_l = f(level)`;
- kernels parametrises par `T<:Real`, aucun `Float64` hardcode dans les
  boucles physiques;
- smoke `Float64` et `Float32` passent le rest-state subcycle;
- pas de regression open-channel/BFS;
- reduction claire du diff route-vs-oracle sur le diagnostic obstacle existant.

### Phase 6 - Obstacle 2D et BFS avec v04

Livrables:

- `test/v04_test_obstacles_2d.jl`
- benchmark local v04 square/cylinder/BFS.

Ordre:

1. BFS court vs oracle;
2. square obstacle scale 1;
3. cylinder scale 1;
4. ladder scales 1,2;
5. aqua seulement apres local vert.

Gate cible initiale:

- masse roundoff;
- BFS proche oracle comme voie D corrigee;
- aucun NaN sur 5000 steps courts locaux;
- cylinder Cd ratio scale 2 explicitement `<= 1.30x` vs oracle leaf, en
  prenant la reference voie D actuelle autour de `1.86x`;
- si la reference voie D est recalculee, garder une cible numerique explicite
  dans `docs/design/amr_v04_status.md`, pas seulement "meilleur";
- ne pas exiger `<1.10x` avant l'etude multi-level de Phase 7.

### Phase 7 - Multi-level Lmax>2 et 2:1 balance production

Livrables:

- migration block-pool sparse/capacity/free-stack;
- tree multi-level;
- topology multi-level;
- tests Lmax=3 puis Lmax=4.

Gate:

- tests Phases 2-6 rejoues sur block-pool avant nouveaux macro-flows;
- no adjacent level jump > 1;
- rest-state multi-level;
- subcycling nested correct;
- rescaling Filippova-Hanel `tau` si `omega` varie par niveau;
- ladder formelle square/cylinder scale `{1,2,4}`;
- ratio obstacle `<= 1.10x` sur au moins 2/3 des scales ou regression
  d'erreur d'ordre `>= 1.5` vs maillage;
- memory budget documente.

Lmax=5 est un objectif de support, pas une obligation pour le premier runner
physique.

### Phase 8 - Multi-patch et adaptation dynamique

Livrables:

- patch/block ownership;
- adaptation marks;
- regrid conservative;
- hysteresis;
- balance propagation.

Tests:

- disjoint patches;
- adjacent patches;
- refine near obstacle;
- coarsen wake;
- conservation during adapt;
- no double ownership.

Regle temporelle:

- `adapt!` s'execute uniquement entre deux coarse steps complets;
- jamais de refine/coarsen entre deux demi-pas fine;
- ne jamais reconstruire `PullTopology` mid-cycle.

### Phase 9 - DSL `.krk`

Ne pas construire le DSL avant que l'API Julia soit stable.

Livrables:

- parser minimal `AMR { ... }` ou extension `Refine { ... }`;
- criteria:
  - `gradient(ux)`;
  - `vorticity`;
  - `solid_distance`;
  - `wake_box`;
- tests parser + smoke 200 steps.

Le DSL doit appeler les helpers v04; il ne doit pas dupliquer la logique.

### Phase 10 - AD frozen-tree CPU

Livrables:

- `src/refinement/v04/ad.jl`;
- `test/v04_test_ad_geometry_2d.jl`;
- `test/v04_test_ad_frozen_tree_2d.jl`;
- doc `docs/design/amr_v04_ad.md`.

But:

- rendre l'AMR v04 compatible avec l'optimisation de forme sans rendre
  l'adaptation discrete differentiable;
- fournir une route claire pour Enzyme/ForwardDiff sur les champs continus.

API minimale:

```julia
struct AMRADConfig
    mode::Symbol          # :frozen_tree au debut
    backend::Symbol       # :forwarddiff, :enzyme, :finite_difference
    parameters::Vector{Symbol}
    objective::Symbol
end
```

Tests chirurgicaux:

- representation obstacle differentiable pour cylindre:
  `fluid_fraction`/`solid_weight` ou SDF lissee;
- helper de drag pondere compatible avec `compute_drag_mea_solid_F_2d`, mais
  sans imposer un `Bool` mask dans le chemin AD;
- `dq_wall/dR` cylindre: analytique vs finite difference;
- `q_wall` et wall velocity type Couette/Taylor-Couette type-stables pour
  `ForwardDiff.Dual`;
- objectif drag scalaire varie avec rayon sur arbre gele;
- AD gradient rayon vs finite difference sur 5-20 steps courts;
- route table gelee: aucune allocation ni rebuild pendant la passe gradient;
- rest-state donne gradient nul pour forcing/geometry neutres.

Regles:

- ne pas differencier `build_topology!`, `adapt!`, `refine!`, `coarsen!`;
- ne pas mettre de Dual dans les indices, masks ou route kinds;
- pour obstacles non lisses type square, utiliser d'abord finite difference
  et/ou SDF lisse. Ne pas promettre un gradient exact aux coins;
- garder Enzyme optionnel dans les tests longs si compilation trop couteuse
  localement, mais garder un smoke ForwardDiff/finite-difference rapide.

DSL `.krk` futur:

```text
AD {
  mode = frozen_tree
  backend = enzyme
  parameter = cylinder.radius
  objective = drag_x
}
```

Gate:

- un gradient geometrique cylinder passe analytique vs finite difference;
- un objectif hydrodynamique court passe AD vs finite difference;
- aucune regression des tests subcycling/obstacle.

### Phase 11 - GPU pack et KernelAbstractions

Livrables:

- pack route table vers GPU;
- pack active cells;
- pull kernel KA;
- CPU/GPU equality tests.

Tests:

- topology pack roundtrip;
- one-step rest-state CPU/GPU;
- level 1 bit/near equality;
- FP32 drift bounded;
- benchmark MLUPS.

Gate:

- pas de GPU avant subcycling CPU stable;
- pas de GPU avant que l'API layout/route soit compatible AD frozen-tree;
- pas de BitArray dans kernel GPU;
- pas d'allocations dans streaming kernel.

Priorite device:

- Metal local d'abord pour iteration rapide;
- CUDA H100 aqua ensuite pour production;
- garder la meme API KernelAbstractions pour les deux.

AD GPU:

- hors scope du premier GPU pack;
- possible ensuite via Enzyme GPU seulement apres egalite CPU/GPU et gradients
  CPU valides;
- ne jamais bloquer le runner GPU hydrodynamique sur AD.

### Phase 12 - 3D

Apres 2D subcycled stable.

Livrables:

- D3Q19 topology pull;
- face/edge/corner interface tests;
- rest-state 3D;
- Couette/Poiseuille 3D smoke;
- sphere/cylinder 3D seulement ensuite.

Regle:

- mutualiser les concepts;
- accepter des kernels 2D/3D specialises si cela garde le code clair et rapide.

### Phase 13 - Multi-physique

Apres hydro AMR stable.

Scope v0.4:

- Hydro + Thermal.

Hors scope v0.4:

- viscoelastic;
- phase-field;
- CIJ;
- multiphase.

Ces couplages passent en v0.5+ et doivent etre documentes dans
`docs/design/amr_v05_roadmap.md`. Ne pas bloquer l'AMR hydro sur eux. Ajouter
`FieldKind` progressivement quand l'interface streaming/topology est stable.

## Quality gates globaux

Une phase est terminee seulement si:

1. tests v04 de la phase verts;
2. baseline AMR voie D verte;
3. pas de changements hors perimetre stages;
4. doc `docs/design/amr_v04_status.md` mise a jour;
5. commit separe.

La voie D peut etre marquee "legacy" apres Phase 6 verte, mais elle n'est
supprimable qu'apres succes v0.4 complet, parite avec les runners aqua
existants, et au moins un mois sans regression AMR observee.

## Ce qu'il faut supprimer du prompt original

Ne pas suivre ces idees telles quelles:

- "tout en un seul jet de design";
- "q derniere dimension = AoS contigu" sans benchmark;
- "dense all-level pool obligatoire";
- "zero allocation runtime" sans accepter capacity/free stack pour adaptation;
- "un seul code 2D/3D sans aucune specialisation";
- "GPU avant subcycling";
- "AD a travers refine/coarsen/topology rebuild au premier jet";
- "multi-physique avant hydro AMR stable";
- "supprimer voie D en Phase P10 si tous les objectifs papier ne sont pas
  encore stabilises".

## Critere de succes v0.4

La v0.4 est acceptable quand:

- 2D hydro subcyclee est stable;
- rest-state est preserve avec interfaces, obstacles, boundaries;
- BFS suit l'oracle leaf au moins aussi bien que la voie D corrigee;
- square/cylinder reduisent l'ecart Cd/ux de maniere reproductible;
- multi-level et 2:1 balance sont testes;
- AD frozen-tree CPU valide au moins une sensibilite geometrique courte;
- GPU pack existe et passe CPU/GPU one-step;
- `.krk` sait declarer un cas AMR sans code utilisateur;
- la voie D cohabite jusqu'a parite, puis seulement alors une migration peut
  etre discutee.

Si le cylinder Cd reste eloigne, ne pas masquer le probleme par abstraction:
redescendre au diagnostic subcycling/interface route-vs-oracle.
