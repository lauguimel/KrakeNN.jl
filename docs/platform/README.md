# Kraken → Plateforme : dossier de cadrage

> Cadrage d'évolution de Kraken (noyau LBM GPU performant) vers une **plateforme générale,
> multi-méthodes, à opérateurs différentiables quand possible**, de résolution ET de calibration
> de problèmes physiques — sans accumuler de dette technique. Cadre arrêté dans `brainstorm.md`.
> **Cible générale** : le VE est un cas de validation dur, **pas** le centre de gravité.

## Lecture

| Doc | Contenu |
|-----|---------|
| [`00-BILAN.md`](00-BILAN.md) | État réel du code vs. contrat. Matrice de capacités. **Corrections adversariales au brainstorm.** Bloqueurs transient-AD. |
| [`01-PLAN-DE-MATCH.md`](01-PLAN-DE-MATCH.md) | Plan phasé par seams. Reframe de la tête de pont (géométrique/générale, pas VE). Milestones + risques. |
| [`02-PLAN-IMPLEMENTATION.md`](02-PLAN-IMPLEMENTATION.md) | Premières PRs, arborescence `src/platform/`, ordre des fichiers, gates. |
| [`03-EXPLORATION-AXES.md`](03-EXPLORATION-AXES.md) | Options (pas une voie) : data, IA/ONNX, feedback, **NL→JSON+API**. |
| [`04-HYGIENE-PROCESS.md`](04-HYGIENE-PROCESS.md) | Worktrees, branches/bugs, nettoyage, skills, **red-team du plan**. |
| [`05-DOF-LIBRES.md`](05-DOF-LIBRES.md) | **Le cœur conceptuel** : libérer n'importe quelle quantité figée (param → champ → modèle constitutif appris IA) via le même `fit`. 3 axes orthogonaux, échelle de coût, exemples cibles. |
| [`06-WORKFLOW.md`](06-WORKFLOW.md) | **Modèle opératoire** : éviter les collisions git (choke files vs briques), branches courtes, doc humaine+LLM dans la PR (gate), mandat vivant, cycle de vie d'une modif. |
| [`07-STATE-CONTRACT.md`](07-STATE-CONTRACT.md) | **ADR-02** : état de simulation reprenable (contrat indépendant du solveur) et format de checkpoint HDF5. Trois classes de configuration, deux entiers de version, reprise `.krk` différée. |

## Les 3 prises de tête à valider AVANT de coder

1. **La cible est l'axe des DOF libres, pas la géométrie.** Libérer n'importe quelle quantité figée
   de `R` (paramètre → champ → modèle constitutif appris depuis données type PIV) via le **même**
   `fit`. La géométrie est juste le seul DOF câblé aujourd'hui (sert de parité) ; la tête de pont
   réelle est **« libérer un paramètre scalaire et l'inférer depuis des données »**. Loi : le coût
   est dans *rendre `R` différentiable vis-à-vis du DOF*, jamais dans `fit`. (`05-DOF-LIBRES.md`)
2. **La primitive partagée est le pas `G`, pas le résidu `R`.** `R(u)=u−G(u)` est implicite et à
   **exposer**, pas à inventer comme assembleur PDE. (`00` §A1)
3. **Plusieurs actifs « généraux » existent déjà** (VoF/diphasique, opérateurs FV/FD 2D/3D, AD
   transverse 3 physiques) → la plateforme est générale dès P0 en mettant ces forwards derrière le
   contrat. (`00` §F)

## Décisions verrouillées (ADR)

- **ADR-01 — Formats I/O** (`03-EXPLORATION-AXES.md`) : **VTKHDF** (viz/archivage, natif ParaView)
  + **Zarr** (datasets ML/cloud). Métadonnées JSON séparées des champs lourds. NetCDF différé au
  vertical géo. Pas de writer maison (`WriteVTK.jl`/`HDF5.jl`/`Zarr.jl`).

## Statut

Cadrage **lecture seule** — aucun code modifié, aucun refactor lancé. Prochaine étape : valider
le plan, puis ouvrir PR 1 (`src/platform/contract.jl`).
