# Next Session - Voie D AMR Architecture

## Vision

Construire dans Kraken.jl un AMR LBM GPU-compatible, inspire de la logique
Basilisk mais sans arbre pointeur.

Objectif:

```text
Basilisk-like dans la logique des levels,
PowerFLOW/Rohde-like dans la conservation volumetrique,
GPU-like dans le stockage et le chemin chaud.
```

Non-objectif immediat: ne pas pretendre que la voie D actuelle est deja un
vrai solveur AMR a dx local.

Contrainte long terme: l'architecture doit rester generalisable a plusieurs
types de maillage et a la 3D. La voie D 2D actuelle est un prototype
cartesien/quadtree-like, pas le modele final unique.

## Etat actuel

La voie D actuelle est une representation conservative tree compressee:

```text
composite coarse/fine
    -> projection conservative vers leaf grid uniforme
    -> streaming/collision sur leaf grid uniforme
    -> restriction conservative vers composite
```

Elle sert d'oracle de validation. Elle valide surtout:

- stockage integre `F_i = f_i * volume_cellule`;
- cellules coarse actives hors patch;
- cellules fine actives dans patch;
- cellules coarse inactives sous patch;
- diagnostics actifs masse/momentum;
- projection/restriction conservatives;
- canaries Couette, Poiseuille, obstacle carre, BFS, cylindre.

Tests voie D actuels:

```text
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_2d.jl")'
Conservative tree 2D | 897 / 897
```

Plots:

```text
tmp/voie_d_debug_plots/poiseuille_bands_debug.png
tmp/voie_d_debug_plots/square_obstacle_drag_debug.png
tmp/voie_d_debug_plots/summary.txt
```

Correction importante deja faite:

- `explode_uniform_F_2d!` reste une primitive conservative bas niveau.
- `composite_to_leaf_F_2d!` utilise maintenant une reconstruction lineaire
  limitee par population, via `_explode_limited_linear_composite_F_2d!`.
- Cela evite de casser les gradients locaux et le stress non-equilibre.
- Poiseuille bandes:
  - bande verticale x=L/2: `L2 = 1.198e-3`;
  - bande horizontale y=L/2: `L2 = 1.126e-3`.
- Obstacle carre avec patch englobant obstacle:
  - `Fx_refined / Fx_coarse = 0.986`.

Limite importante:

Le pas chaud est encore fait sur une leaf grid uniforme. Le maillage actif
coarse/fine est une compression conservative, pas encore un streaming AMR natif.

## Lecons apprises dans la branche D

Point critique:

```text
Conserver masse/momentum/populations globales est necessaire,
mais pas suffisant pour un LBM correct.
```

Le bug observe:

- l'explosion uniforme parent -> 4 enfants conservait les populations globales;
- mais elle effacait les gradients locaux de vitesse;
- elle detruisait le stress non-equilibre local;
- Poiseuille avec bande raffinee ne devenait pas transparent.

Correctif actuel:

- conserver `explode_uniform_F_2d!` comme test/primitif bas niveau;
- ne pas l'utiliser comme reconstruction physique dans le chemin composite;
- utiliser une reconstruction lineaire limitee par population;
- verifier Poiseuille bande verticale/horizontale et drag obstacle carre.

Conclusion pour la suite:

```text
Toute future interface coarse/fine doit tester conservation ET transparence
sur stress/Poiseuille. Un test de masse seule est insuffisant.
```

## Contrats de non-regression

Avant d'ecrire le streaming composite natif, ne pas casser:

```text
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_2d.jl")'
```

Canaris importants:

- `uniform projection loses local velocity and stress moments`;
- `macroflow Poiseuille analytic with full x and y refinement bands`;
- `square obstacle drag with enclosing refinement vs coarse Cartesian`.

Seuils actuels a conserver ou ameliorer:

```text
Poiseuille bande verticale x=L/2:   L2 ~ 1.2e-3
Poiseuille bande horizontale y=L/2: L2 ~ 1.1e-3
Obstacle carre refined/coarse Fx:   ratio ~ 0.986
```

Si une modification conserve la masse mais degrade ces seuils, elle est
probablement mauvaise.

## Architecture cible

Separer strictement quatre couches.

### 0. Abstraction mesh/topologie

Ne pas rendre les operateurs dependants d'un patch rectangulaire 2D code en dur.

La topologie doit exposer des cellules actives, volumes, voisins et routes de
paquets. Le type de maillage qui a produit ces cellules doit rester derriere
un adaptateur.

Familles visees a terme:

```text
cartesien dense 2D/3D
patchs rectangulaires fixes ratio 2
quadtree/octree AMR
blocs actifs prealloues
eventuellement multiblock/bodyfit plus tard, via un autre adaptateur
```

Le noyau numerique doit consommer:

```text
cells + metrics + routes
```

et non:

```text
un type concret de mesh
```

En 3D, la meme separation doit permettre de remplacer:

```text
D2Q9 + quadtree -> D3Q19/D3Q27 + octree
```

sans reecrire la conservation, les diagnostics actifs, ni le principe des
tables compactes.

Implication 3D concrete:

```text
2D: coarse/fine split face ou corner
3D: coarse/fine split face, edge ou corner
```

Donc la Phase 1 doit eviter les noms et structures qui supposent uniquement
`:west/:east/:south/:north`. Pour la 2D on peut garder ces helpers existants,
mais la nouvelle topologie doit deja penser en termes generiques:

```text
direction q
offset lattice c_q
dimension dim
children touched by the packet route
weights per child
```

En 3D, un paquet axial coarse->fine sur une face peut toucher 4 enfants, un
paquet d'arete peut toucher 2 enfants, et un paquet diagonal/corner peut
toucher 1 enfant. La table de routes ponderees doit pouvoir representer ces
cas sans changer le hot loop.

## Generalisation multiphysique

La topologie AMR ne doit pas etre specifique au fluide newtonien.

Elle doit fournir une information geometrique reusable:

```text
cellules actives
volumes
levels
voisins
routes ponderees
boundary tags
```

Les champs physiques doivent ensuite brancher leurs propres operateurs de
transfert sur ces routes.

Champs a garder en tete:

```text
hydrodynamique LBM: populations F_i = f_i * volume
temperature: populations thermiques ou scalaire conservatif
especes: populations/scalaires conservatifs
viscoelasticite: tenseurs de contrainte et/ou populations polymeriques
phase-field/VOF: champs scalaires avec bornes et conservation stricte
```

Regle d'architecture:

```text
la topologie connait les routes;
le champ connait comment ses donnees traversent ces routes.
```

Exemples:

- hydrodynamique: conserver masse, momentum et limiter les moments non-equilibre;
- temperature/especes: conserver le scalaire extensif `phi * volume`;
- viscoelasticite: transferer les tenseurs avec positivite/stabilite, pas avec
  une interpolation arbitraire qui casse SPD ou cree des contraintes negatives;
- phase-field/VOF: conserver la quantite extensive et limiter pour rester dans
  les bornes physiques.

Ne pas implementer toute la multiphysique maintenant. Mais ne pas concevoir
une API qui force `F[:,:,q]` hydrodynamique partout. Preferer un schema:

```text
routes geometriques communes
transfer_operator[field_kind]
collision_or_update_operator[field_kind]
diagnostics[field_kind]
```

### 1. Topologie AMR statique/dynamique

Responsabilites:

- decrire les levels;
- enumerer les cellules actives;
- marquer les coarse masquees par du fine;
- garantir la balance 2:1;
- classifier les liens D2Q9, puis plus tard D3Q19.

Distinction importante:

```text
lien logique D2Q9 = "depuis cette cellule active, dans cette direction q"
route de paquet   = transfert effectif d'une fraction de F_q vers une cible
```

Un lien same-level donne une route unique. Un lien coarse_to_fine peut donner
deux routes vers deux enfants pour une face axiale, ou une route vers un enfant
pour un coin diagonal. Un lien fine_to_coarse peut etre une contribution parmi
plusieurs vers une meme cellule coarse.

Classification minimale des liens logiques:

```text
same_level
coarse_to_fine
fine_to_coarse
boundary
```

Classification minimale des routes:

```text
direct
split_face
split_corner
coalesce_face
coalesce_corner
boundary
```

Cette couche peut etre construite CPU d'abord, puis copiee GPU sous forme de
tables compactes.

### 2. Etat conservatif composite

Responsabilites:

- stocker `F_i` sur les cellules actives;
- calculer masse/momentum/diagnostics uniquement sur cellules actives;
- fournir projection/restriction conservatives;
- ne jamais compter les coarse inactives sous patch.

### 3. Operateurs numeriques

Responsabilites:

- collision locale;
- streaming composite;
- restriction/prolongation conservative;
- traitement interfaces coarse/fine;
- plus tard, sous-cycling temporel par level.

La collision peut rester level-agnostic:

```text
collide(cell, metrics[level])
```

Le streaming et les interfaces ne le sont pas.

## Conventions f vs F

Principe central:

```text
F_i = f_i * volume_cellule
```

Mais la collision BGK/Guo agit physiquement sur les populations densitaires
`f_i`, pas directement sur les populations extensives `F_i`.

Convention obligatoire par operateur:

```text
collision locale:
    consomme F
    convertit localement f = F / volume
    applique BGK/Guo sur f
    restocke F = f * volume

streaming / routes / restriction / prolongation:
    consomment F
    transferent des paquets extensifs F_q
    doivent conserver les sommes actives de F_q

diagnostics hydrodynamiques:
    masse    = sum(F_q)
    momentum = sum(c_q * F_q)
    rho      = masse / volume
    u        = momentum / masse
```

Ne pas melanger les deux conventions dans un meme kernel sans nom explicite.
Les fonctions qui manipulent des paquets extensifs doivent garder un suffixe ou
une doc claire indiquant `F`. Les fonctions qui manipulent des densites doivent
indiquer `f`.

Canari Phase 2 a ajouter:

```text
C-T1: round-trip f -> F -> f bit-identique sur cellules actives
```

Ce canari doit couvrir au moins deux levels avec volumes differents. C'est un
piege classique: oublier une conversion `F/V` ou `f*V` peut produire un champ
"presque correct" pendant quelques pas, puis casser conservation ou stress.

### 4. Execution GPU

Principe:

```text
bulk same-level -> kernel proche du cartesien
interface       -> kernel specialise
boundary        -> kernel specialise
```

Interdits dans le hot loop bulk:

- test `if level`;
- test `if neighbor_type`;
- hash lookup Morton;
- logique coarse/fine;
- allocation dynamique.

Stockage vise a terme:

```text
F[Q, cells_per_block, max_blocks]
level[max_blocks]
morton_key[max_blocks]
active_bulk[level]
active_interface[level]
active_boundary[level]
free_stack
```

## Layout transformation: AoS topology -> block-packed storage

La Phase 1 peut construire une topologie logique cell-granulaire:

```text
Vector{ConservativeTreeCell2D}
Vector{ConservativeTreeLink2D}
Vector{ConservativeTreeRoute2D}
```

Ce n'est pas le stockage chaud final. Il faut une etape explicite de packing:

```text
logical topology -> sorted active cells -> blocks -> remapped routes
```

Transformation minimale:

1. Construire les cellules actives et les routes en indices logiques `cell_id`.
2. Trier les cellules actives par `(level, morton_key)` pour garder proximite
   spatiale et separer les levels.
3. Decouper la liste triee en blocs de taille fixe `cells_per_block`.
4. Construire une table:

```text
logical_cell_id -> (block_id, local_cell)
```

5. Reecrire chaque route:

```text
src::cell_id, dst::cell_id
    -> src_block, src_local, dst_block, dst_local
```

6. Construire les listes compactes:

```text
bulk same-level routes
interface routes
boundary routes
```

Cette transformation a lieu:

- au build initial de la topologie statique;
- apres chaque adaptation dynamique future;
- jamais dans le hot loop de streaming.

Choix de packing initial recommande:

```text
tri primaire   : level
tri secondaire : Morton key
bloc           : taille fixe, preallouee
```

Raison:

- le bulk same-level reste proche du cartesian dense;
- les interfaces sont isolees dans des listes specialisees;
- les routes survivent au packing via une table de remapping explicite;
- le futur GPU peut stocker `F[Q, cells_per_block, max_blocks]` sans imposer
  cette structure a la topologie logique.

Ne pas faire en Phase 1:

- pas de hash lookup Morton dans les kernels;
- pas de packing dynamique dans le hot loop;
- pas de dependance du code numerique a `Vector{Cell}` comme stockage final.

## Conventions AoS vs SoA en Julia

Convention pour ce projet:

```text
Topologie logique: AoS
LBM dense hot arrays: garder le layout mesure localement
Packing GPU final: decider par benchmark, pas par intuition C++
```

En Julia, l'intuition C++ "SoA est toujours plus rapide" est fausse trop
souvent pour etre une regle. Sur petits structs isbits ou quand plusieurs
champs sont lus ensemble, `Vector{SmallStruct}` peut etre plus simple et plus
rapide qu'un SoA manuel.

Decisions pour la Phase 1:

- garder `Vector{ConservativeTreeCell2D}` en AoS;
- garder `Vector{ConservativeTreeLink2D}` en AoS;
- garder `Vector{ConservativeTreeRoute2D}` en AoS;
- utiliser des `@enum ... :: UInt8`, pas des `Symbol`, pour rester isbits/GPU;
- eviter `StructArrays.jl` par defaut.

Quand envisager SoA:

- si un kernel lit un seul champ sur beaucoup d'elements;
- si un benchmark local montre un gain net;
- pour les tableaux de populations LBM denses, selon les resultats reels du
  backend cible.

Precedent experimental a garder en tete:

```text
sur la branche lbm, le passage f[q,i,j] vers f[i,j,q] a montre que SoA pouvait
ralentir Julia, contrairement a l'intuition C++.
```

Conclusion:

```text
ne pas migrer prematurement la topologie vers StructArrays;
mesurer avant de changer le layout.
```

## Cibles de performance

Les performances doivent etre mesurees contre un cartesian dense equivalent
sur le meme backend, meme precision et meme domaine physique.

Cible Phase 2, streaming composite FP32 GPU:

```text
bulk same-level  >= 70% MLUPS du cartesien equivalent a dx_min
composite global >= 50% MLUPS du cartesien equivalent a dx_min
```

Cible long terme, avec sous-cycling Phase 3+:

```text
bulk same-level  >= 80% MLUPS du cartesien equivalent a dx_min
composite global >= 60% MLUPS du cartesien equivalent a dx_min
cellules economisees >= 3x
speedup absolu >= 1.8x vs cartesien dense
```

Signal d'alarme:

```text
si bulk same-level < 60%, revenir a la conception du hot loop
```

Cela indique probablement:

- indirection non necessaire;
- branche AMR qui a fui dans le kernel bulk;
- layout defavorable;
- routes mal compactees.

## Roadmap immediate

Ne pas essayer de tout faire dans la prochaine session.

### Phase P cible prioritaire: patch refinement voie D publiable

Avant de construire un AMR dynamique ou un streaming composite complet, le
raffinement par patch fixe doit etre propre, robuste et publiable.

Objectif Phase P:

```text
fixed patch refinement ratio 2
representation composite conservative
oracle leaf-grid explicite
validation macroflows et obstacles
figures/debug reproductibles
```

Ce jalon est prioritaire sur l'AMR general. L'AMR dynamique ne doit pas servir
a masquer un patch refinement pas encore solide.

Criteres de validation Phase P:

1. Conservation exacte ou roundoff:
   - populations par orientation;
   - masse;
   - momentum;
   - volume actif.
2. Transparence coarse/fine:
   - Poiseuille bande verticale `x=L/2`;
   - Poiseuille bande horizontale `y=L/2`;
   - Couette avec patch traversant le cisaillement.
3. Stress non-equilibre:
   - test explicite montrant que l'explosion uniforme est insuffisante;
   - test explicite validant la reconstruction utilisee par le chemin composite.
4. Obstacles:
   - obstacle carre entierement englobe par patch;
   - drag compare au coarse cartesian;
   - lift quasi nul sur cas symetrique.
5. Cylinder/BFS seulement apres les canaris precedents:
   - stable;
   - mass drift bornee;
   - comparaison cartesian ou oracle leaf-grid.
6. Backend:
   - CPU comme reference;
   - smoke GPU projection/restriction si possible;
   - ne pas retarder Phase P sur un kernel AMR GPU complet.
7. Reproductibilite:
   - tests dans `test/test_conservative_tree_2d.jl`;
   - scripts de plots dans `tmp/voie_d_debug_plots.jl`;
   - figures regenerables sans etat cache.

Livrable publiable minimal:

```text
Methode:
  conservative fixed-patch refinement for LBM using integrated populations F_i

Validation:
  patch tests orientationnels
  Couette
  Poiseuille bandes verticales/horizontales
  obstacle carre drag/lift
  BFS/cylindre comme demonstrations si assez stables

Discussion:
  ce n'est pas encore un AMR dynamique
  ce n'est pas encore un streaming dx-local natif
  le leaf-grid oracle sert a isoler conservation et projection
```

Non-goals Phase P:

- adaptation dynamique;
- multi-level > 2;
- D3Q19;
- sous-cycling;
- kernels GPU optimises;
- integration SLBM/bodyfit/multiblock.

Condition pour passer a Phase 1:

```text
Phase P verte, stable, documentee, avec figures publiables.
```

### Phase 1 cible: brique topologique AMR statique

Fichiers probables:

```text
src/refinement/conservative_tree_topology_2d.jl
test/test_conservative_tree_topology_2d.jl
src/KrakeNN.jl
test/runtests.jl
```

API minimale proposee:

```julia
@enum LinkKind::UInt8 SAME_LEVEL=0 COARSE_TO_FINE=1 FINE_TO_COARSE=2 BOUNDARY=3
@enum RouteKind::UInt8 DIRECT=0 SPLIT_FACE=1 SPLIT_CORNER=2 COALESCE_FACE=3 COALESCE_CORNER=4 ROUTE_BOUNDARY=5

abstract type AbstractCellMetrics end

struct CartesianMetrics2D <: AbstractCellMetrics
    volume::Float64
end

# Plus tard, sans refactor de la topologie:
# struct CurvilinearMetrics2D <: AbstractCellMetrics
#     jacobian::SMatrix{2,2,Float64}
#     volume::Float64
# end

struct ConservativeTreeCell2D
    level::Int
    i::Int
    j::Int
    active::Bool
    metrics::CartesianMetrics2D
    parent::Int
end

struct ConservativeTreeLink2D
    src::Int
    q::Int
    kind::LinkKind
end

struct ConservativeTreeRoute2D
    src::Int
    dst::Int
    q::Int
    weight::Float64
    kind::RouteKind
end

struct ConservativeTreeTopology2D
    cells::Vector{ConservativeTreeCell2D}
    links::Vector{ConservativeTreeLink2D}
    routes::Vector{ConservativeTreeRoute2D}
    active_cells::Vector{Int}
    same_level_links::Vector{Int}
    coarse_to_fine_links::Vector{Int}
    fine_to_coarse_links::Vector{Int}
    boundary_links::Vector{Int}
    direct_routes::Vector{Int}
    interface_routes::Vector{Int}
    boundary_routes::Vector{Int}
end
```

Notes de typage:

- ne pas utiliser `Symbol` dans les structs destines a aller sur GPU;
- utiliser des `@enum ... :: UInt8` pour les kinds;
- ne pas stocker `metrics::AbstractCellMetrics` dans un hot struct, car un
  champ abstrait casserait l'isbits et la performance;
- en Phase 1, garder `metrics::CartesianMetrics2D`;
- plus tard, passer a un `ConservativeTreeCell{M<:AbstractCellMetrics}` si un
  meme code doit specialiser cartesian/curvilinear sans refactor logique.

But du champ `metrics`:

```text
cartesien aujourd'hui: volume
curviligne plus tard: jacobian + volume + normales/metriques si necessaire
```

La topologie doit donc eviter d'exposer directement `cell.volume` partout. Les
operateurs doivent passer par les metrics de cellule/level.

Tests canaris a ecrire avant tout streaming natif:

1. Volume actif total egal au volume domaine.
2. Coarse sous patch inactif.
3. Chaque cellule active a 9 liens logiques D2Q9 classes.
4. Les liens same-level sont majoritaires et separables.
5. Les liens coarse/fine apparaissent uniquement sur bord de patch.
6. Les liens boundary apparaissent uniquement au bord physique.
7. Les routes conservent chaque paquet logique:

```text
sum(weight des routes d'un lien interieur) == 1
```

8. Tables compactes coherentes:

```text
length(same_level_links) +
length(coarse_to_fine_links) +
length(fine_to_coarse_links) +
length(boundary_links) == length(links)
```

9. Les routes d'interface reproduisent exactement les operateurs existants:
   - `split_coarse_to_fine_face_F_2d!`;
   - `split_coarse_to_fine_corner_F_2d!`;
   - `coalesce_fine_to_coarse_face_F`;
   - `coalesce_fine_to_coarse_corner_F`.

### Phase 2 cible: streaming composite sans collision

Seulement apres Phase 1 verte.

Objectif:

```text
transport direct sur cellules actives
comparaison stricte avec oracle leaf-grid
conservation population par population
```

Ordre de montee:

1. same-level only, patch absent;
2. patch present mais aucun paquet ne traverse interface;
3. crossing coarse -> fine;
4. crossing fine -> coarse;
5. all orientations D2Q9;
6. periodic x / wall y;
7. collision BGK;
8. collision Guo;
9. Couette/Poiseuille;
10. obstacle carre.

Canaries Phase 2 supplementaires:

```text
C-T1: round-trip f -> F -> f bit-identique sur cellules actives
```

Canaries stabilite Phase 2:

```text
C-S1: Couette long-run 50000 steps, drift de masse < 1e-10
C-S2: Cylindre Re=100 sur composite, St = 0.165 +/- 0.5% vs cartesien
C-S3: Channel Re=1000, profil de vitesse stable >= 100000 steps
C-S4: Bit-identique CPU/GPU sur C-S1 a >= 10000 steps
```

Justification:

Un AMR LBM peut etre parfaitement conservatif a court terme et exploser vers
`t = 10000` a cause d'un defaut d'interface, d'un mauvais traitement
Filippova-Hanel-like, d'une conversion `f/F` manquante, ou d'une instabilite
numerique locale. Conservation sans stabilite long-run est un piege classique.

## Ne pas faire maintenant

- AMR dynamique;
- D3Q19;
- mixed precision;
- AA AMR complet;
- persistent kernels;
- SLBM/bodyfit/multiblock;
- refactor global de Kraken.

## Critere de succes de la prochaine etape

On doit pouvoir dire:

```text
Nous avons une representation composite conservative testee.
Nous avons un oracle leaf-grid pour validation.
Nous avons une topologie AMR statique qui classe les liens.
Nous avons des tables compactes CPU/GPU-ready.
Nous sommes prets a ecrire le streaming composite natif.
```
