# AMR Conservative Kraken.jl - Plan De Reference Jusqu'au Projet Complet

## Objectif Final

Construire un AMR LBM conservatif, 2D puis 3D, compatible CPU/GPU, fonde sur
des populations integrees:

```text
F_i = f_i * volume_cellule
```

La cible finale est:

```text
AMR dynamique multi-niveaux
streaming composite natif
interfaces coarse/fine conservatives et transparentes
packing GPU-ready
validations 2D et 3D
benchmarks contre grille dense dx_min
```

Le projet doit avancer par paliers testables. Aucun macro-flow ne doit etre
interprete comme preuve si les patch tests chirurgicaux sous-jacents ne sont
pas verts.

## Regle De Developpement

Chaque avancee doit avoir:

- un patch de code limite;
- des tests chirurgicaux;
- si necessaire, un macro-flow de validation;
- un commit separe.

Ordre obligatoire:

```text
patch tests chirurgicaux
  -> tests de conservation/route
  -> comparaison oracle leaf-grid
  -> macro-flow court
  -> macro-flow long si risque de stabilite
  -> commit
```

Si une incoherence apparait:

```text
macro-flow casse
  -> revenir au test oracle
  -> revenir au test de route
  -> revenir au test de primitive split/coalesce
  -> corriger au plus bas niveau qui reproduit le bug
```

Ne pas corriger un macro-flow en ajustant des seuils tant que les invariants
de transport ne sont pas expliques.

## Etat De Depart

Deja disponible:

- voie D conservative tree 2D;
- Phase P fixed-patch publiee en interne;
- oracle composite -> leaf -> composite;
- topologie 2D avec liens/routes;
- packing 2D `(level, Morton) -> block/local`;
- gate Phase P:
  - Couette;
  - Poiseuille bandes verticale/horizontale;
  - square obstacle;
  - BFS/VFS;
  - pas de cylinder dans Phase P.

Commandes actuelles:

```bash
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_2d.jl")'
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_topology_2d.jl")'
julia --project=. scripts/figures/plot_voie_d_phase_p.jl
```

## Milestone 1 - Streaming Composite Natif 2D Sans Collision

But:

```text
remplacer le transport leaf-grid oracle par un streaming direct sur cellules
actives, sans collision.
```

Fichiers probables:

```text
src/refinement/conservative_tree_streaming_2d.jl
test/test_conservative_tree_streaming_2d.jl
src/KrakeNN.jl
test/runtests.jl
```

Patch tests:

1. Route directe same-level:
   - une population traceur part d'une cellule active coarse;
   - arrive dans la cellule active voisine;
   - conservation par orientation.
2. Coarse -> fine face:
   - paquet axial entrant dans patch;
   - split vers 2 enfants;
   - poids `1/2 + 1/2`.
3. Coarse -> fine corner:
   - paquet diagonal entrant dans patch;
   - route vers 1 enfant;
   - poids `1`.
4. Fine -> coarse face:
   - deux enfants de bord sortent vers meme coarse voisin;
   - somme conservative.
5. Fine -> coarse corner:
   - enfant de coin sort vers coarse diagonal;
   - paquet conserve.
6. Toutes orientations D2Q9:
   - somme des poids par lien interieur = 1;
   - sommes globales par `q` conservees hors boundary.
7. Boundary explicite:
   - periodic x;
   - wall y;
   - pas de paquet perdu non documente.

Validation oracle:

```text
stream_composite_routes_F_2d!
  vs
stream_composite_*_leaf_F_2d!
```

Cas de montee:

```text
patch absent
patch present sans crossing
crossing coarse -> fine
crossing fine -> coarse
all q
periodic x
wall y
```

Critere de sortie:

```text
population-wise conservation a roundoff
equivalence oracle leaf-grid sur les cas sans collision
commit: "Add route-driven conservative tree streaming 2D"
```

## Milestone 2 - Collision Locale Active 2D

But:

```text
BGK/Guo collision sur cellules actives, puis streaming composite natif.
```

Regles:

- coarse active: collision avec `volume=1`;
- fine active: collision avec `volume=0.25`;
- coarse inactif sous patch: jamais collisionne;
- collision en `f = F / volume`, stockage en `F`.

Patch tests:

1. BGK conserve masse/momentum par cellule active.
2. Guo conserve masse et ajoute le momentum attendu.
3. Parent inactif sous patch reste ignore.
4. Round-trip `f -> F -> f` sur deux niveaux.

Macro-flow courts:

```text
Couette route-native vs oracle
Poiseuille route-native vs oracle
```

Critere:

```text
mass drift < 1e-10
profils proches de Phase P
commit: "Add active-cell collision for route-native AMR 2D"
```

## Milestone 3 - Boundaries Natives 2D

But:

```text
enlever la dependance aux boundaries leaf-grid dans le chemin natif.
```

Boundary kernels/logique:

- fully periodic;
- periodic x + wall y;
- moving wall Couette;
- bounce-back solid mask;
- Zou-He inlet/outlet pour BFS/VFS.

Patch tests:

1. Periodic wrap par orientation.
2. Wall bounce-back stationnaire conserve masse.
3. Moving wall injecte momentum attendu.
4. Solid mask au repos reste invariant.
5. Zou-He west/east impose les moments attendus.

Macro-flow:

```text
Couette natif
Poiseuille natif
square obstacle natif
BFS/VFS natif
```

Critere:

```text
Phase P reproduite sans projection leaf-grid dans le hot path
commit: "Add native boundary routes for AMR 2D"
```

## Milestone 4 - Multi-Patch Statique 2D

But:

```text
plusieurs patchs ratio 2 sans adaptation dynamique.
```

Ordre:

1. Patchs disjoints.
2. Patchs adjacents.
3. Patchs proches mais separes par coarse.
4. Patchs imbriques seulement apres decision explicite.

Patch tests:

- volume actif total;
- coarse inactive sous chaque patch;
- pas de double ownership;
- routes patch-patch;
- routes patch-boundary;
- balance 2:1.

Macro-flow:

```text
Poiseuille avec deux bandes
Couette avec patchs multiples
square obstacle avec patch local
```

Critere:

```text
multi-patch statique stable et conservatif
commit: "Add static multi-patch AMR topology 2D"
```

## Milestone 5 - Adaptation Dynamique CPU 2D

But:

```text
raffinement/derefinement en temps de simulation, CPU d'abord.
```

Critere de raffinement initial:

- gradient de vitesse;
- vorticite;
- proximite obstacle;
- seuils avec hysteresis.

Patch tests:

1. Rebuild topologie conserve volume actif.
2. Ancien composite -> nouveau composite conserve populations par orientation.
3. Raffinement puis derefinement round-trip.
4. Pas d'oscillation si hysteresis active.
5. Balance 2:1 garantie.

Macro-flow:

```text
Poiseuille adaptatif
square obstacle adaptatif
BFS/VFS adaptatif
```

Critere:

```text
adaptation dynamique conservative sans explosion de masse/momentum
commit: "Add dynamic AMR adaptation 2D CPU"
```

## Milestone 6 - Sous-Cycling Temporel 2D

But:

```text
fine fait deux pas pour un pas coarse.
```

Patch tests:

- sequence coarse/fine temporelle conservative;
- interface coarse/fine apres demi-pas;
- pas de double streaming;
- conservation par orientation sur cycle complet.

Macro-flow long:

```text
Couette 50k steps
Poiseuille 50k steps
BFS/VFS stabilite
```

Critere:

```text
stabilite long-run et transparence interface
commit: "Add conservative AMR subcycling 2D"
```

## Milestone 7 - GPU 2D

But:

```text
executer le chemin natif route-based sur GPU.
```

Architecture kernels:

- bulk same-level;
- interface routes;
- boundary routes;
- collision active-cell;
- reductions diagnostics.

Interdits dans bulk hot loop:

- hash lookup;
- branche coarse/fine;
- allocation;
- logique Morton.

Patch tests:

- bitwise ou near-bitwise CPU/GPU sur petits cas;
- conservation route GPU;
- boundary GPU;
- macro-flow court CPU/GPU.

Benchmarks:

```text
bulk same-level >= 70% cartesian dense dx_min
composite global >= 50% cartesian dense dx_min
```

Critere:

```text
AMR 2D GPU demonstrable
commit: "Add GPU route kernels for AMR 2D"
```

## Milestone 8 - Topologie Et Primitives 3D

But:

```text
porter la logique conservative vers D3Q19, sans macro-flow complique.
```

Fichiers probables:

```text
src/refinement/conservative_tree_3d.jl
src/refinement/conservative_tree_topology_3d.jl
test/test_conservative_tree_3d.jl
test/test_conservative_tree_topology_3d.jl
```

Primitives:

- `coalesce_F_3d!`: 8 enfants -> parent;
- `explode_uniform_F_3d!`;
- split face: 4 enfants;
- split edge: 2 enfants;
- split corner: 1 enfant;
- coalesce face/edge/corner.

Patch tests:

- D3Q19 directions/opposites;
- volume coarse `1`, fine `1/8`;
- conservation par orientation;
- momentum 3D;
- route weights = 1;
- face/edge/corner touch counts.

Critere:

```text
topologie 3D statique et primitives conservatives vertes
commit: "Add conservative tree topology 3D"
```

## Milestone 9 - Oracle Composite 3D

But:

```text
composite 3D -> leaf grid 3D -> composite 3D.
```

Tests:

- round-trip projection/restriction;
- active mass/momentum skip inactive ledgers;
- uniform explosion detruit gradient/stress;
- reconstruction limitee restaure transparence.

Macro-flow:

```text
3D Couette
3D Poiseuille channel
cube obstacle
```

Critere:

```text
Phase P-3D fixed patch validable
commit: "Add fixed-patch conservative AMR oracle 3D"
```

## Milestone 10 - Streaming Natif 3D

But:

```text
stream_composite_routes_F_3d!
```

Ordre:

1. Same-level.
2. Coarse -> fine face.
3. Coarse -> fine edge.
4. Coarse -> fine corner.
5. Fine -> coarse face.
6. Fine -> coarse edge.
7. Fine -> coarse corner.
8. Periodic.
9. Wall.
10. Collision.

Critere:

```text
streaming 3D route-native compare a l'oracle
commit: "Add route-driven conservative streaming 3D"
```

## Milestone 11 - AMR 3D GPU Et Benchmarks

But:

```text
D3Q19 route-native sur GPU.
```

Benchmarks:

- Couette/Poiseuille 3D;
- cube obstacle;
- sphere seulement apres cube stable;
- comparaison dense dx_min.

Critere:

```text
AMR 3D GPU fonctionnel et mesure
commit: "Add GPU AMR 3D benchmark path"
```

## Definition De Done AMR Complet

Le projet AMR complet est termine quand:

- 2D route-native fonctionne sans oracle leaf-grid dans le hot path;
- adaptation dynamique 2D est conservative;
- GPU 2D est benchmarke;
- 3D D3Q19 fixed-patch est validable;
- 3D route-native fonctionne;
- GPU 3D a au moins un benchmark stable;
- chaque milestone a ses tests chirurgicaux;
- chaque macro-flow a une explication si un seuil change;
- la documentation distingue clairement:
  - ce qui est oracle;
  - ce qui est natif;
  - ce qui est CPU;
  - ce qui est GPU;
  - ce qui est 2D;
  - ce qui est 3D.

