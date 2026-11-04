# État de simulation reprenable — contrat et format de checkpoint

> Kraken n'avait aucune notion d'état reprenable : un run interrompu (walltime HPC, divergence,
> changement de paramètre en cours de route) repartait de zéro. Ce document fixe le contrat
> **indépendant du solveur** et le format sur disque (issue #26). Code : `src/platform/state.jl`,
> `src/io/checkpoint_hdf5.jl`. Carte d'implication : `docs/agent/state-implication.md`.

---

## ADR-02 — État reprenable & checkpoint (VERROUILLÉ 2026-09-17)

**Décision :** un contrat en mémoire (`AbstractSimulationState` + `StateSnapshot` + verbes
génériques) et, séparément, une couche disque HDF5. `state.jl` ne connaît aucun format de fichier ;
`checkpoint_hdf5.jl` est le seul fichier de Kraken qui parle à HDF5.

### 1. Position dans la plateforme

- **L'état est le pendant transitoire du `u`** de `src/platform/residual.jl` : tout ce que le
  solveur doit porter d'un cycle au suivant, et rien de ce qui s'en reconstruit.
- **`advance!(state, n)` = n applications du pas `G`**, celui dont `residual.jl` expose le point
  fixe `R(u,p) = u − G(u,p)`. Le contrat ne crée pas un second pas de temps.
- **`solve` reste le verbe one-shot** : `solve` = `init_state` + `advance!` + `solution`.
  `solution(state)` renvoie un `AbstractSolution`, donc `sample`/`observe` s'appliquent tels quels.
- **Les paramètres modifiables en cours de run se déclarent avec le `ParameterSpace` existant**
  (`updatable_parameters(S)`), pas avec une liste parallèle : mêmes noms, mêmes bornes que ce que
  `fit` consomme. `check_updatable` refuse un nom non déclaré ou une valeur hors bornes.
- **Pas d'unification avec le chemin AD `(u, p)`** : celui-ci est CPU-Float64 stationnaire
  uniquement. Les deux sont positionnés l'un par rapport à l'autre, pas fusionnés.
- **Verbes génériques, aucun verbe par solveur** : `init_state`, `advance!`, `solution`,
  `snapshot`, `restore_state`, `validate_snapshot`, `at_boundary`, `update_parameter!`,
  `updatable_parameters`, `migrate`. C'est ce qui rend écrivable, plus tard, une boucle
  « checkpoint tous les N cycles » qui ignore quel solveur elle fait tourner.

### 2. Format sur disque

- **HDF5 via `HDF5.jl`, un fichier `.h5` par checkpoint, schéma plat, aucune métadonnée de type
  Julia.** Cohérent avec ADR-01 (aucun writer maison) ; libhdf5 est déjà chargée par Gmsh, le coût
  est une entrée de Manifest. Lisible par `h5py` (ordre des indices inversé, chaînes en `bytes`).
- **Intégrité, trois faits mesurés :** (a) charge utile : datasets chunkés + filtre Fletcher32, un
  bit retourné → erreur de lecture ; (b) métadonnées : ouverture avec
  `libver_bounds=(v"1.10", v"1.10")` (superblock v3, en-têtes d'objets avec checksum) — sans ce
  mot-clé, libhdf5 1.14 écrit un superblock v0 et un bit retourné dans un attribut (`cycle`, un
  paramètre physique) est **relu sans erreur** ; (c) les chaînes de longueur variable vont dans le
  *global heap*, qui n'a pas de checksum même en format 1.10 → aucune liste de chaînes dans le
  fichier, la liste des clés est une seule chaîne scalaire jointe par `\n`.
- **Écriture atomique :** `path.tmp` dans le même dossier, fermeture, puis `Base.rename` — jamais
  `mv(...; force=true)`, non atomique sous Julia 1.11 (la destination est supprimée avant le
  renommage). Une génération `.prev` est conservée. Un snapshot dont un tableau (`fields`, `series`) contient
  `NaN`/`Inf` est refusé **avant** toute écriture : un état divergé ne remplace jamais le dernier
  bon point de reprise. Les scalaires portés ne sont pas contrôlés : un indicateur de convergence
  vaut légitimement `Inf` avant son premier échantillon.

Les trois faits sont verrouillés par `test/platform/state_contract_test.jl` (le cas « attribute bit
flip rejected » échoue si le mot-clé `libver_bounds` est retiré).

**Layout (container version 1) :** attributs racine `container_version`, `schema_version`,
`solver`, `cycle`, `kraken_version`, `julia_version`, `created_unix` ; groupes `/fields` et
`/series` (un dataset par entrée, `/` dans un nom → sous-groupes, p. ex. `block003/f`) ; groupes
`/scalars`, `/identity`, `/run_control`, `/parameters`, `/derived` (un attribut par entrée + `keys`).
`nothing` = clé listée dans `keys`, attribut absent — distinguable d'une clé absente parce que le
fichier est ancien. Chunks : une tranche le long de la dernière dimension (`(Nx,Ny,1)`,
`(Nx,Ny,Nz,1)`) ; un chunk unique par tableau buterait sur la limite HDF5 de 4 Gio en 3D.

### 3. Trois classes de configuration

| Classe | Comparée à la reprise ? | Contenu type | Pourquoi |
|---|---|---|---|
| `identity` | **oui**, clé par clé (type et valeur) | grille, précision, schémas, nombres sans dimension, tolérances, intervalle d'historique | définit *quelle* simulation c'est |
| `run_control` | non | horizon (`max_cycles`, temps cible), backend, provenance de la perturbation initiale | prolonger un run doit marcher |
| `parameters` | non | paramètres physiques gouvernés par `update_parameter!` | reprendre après une mise à jour doit marcher |

Une égalité globale sur toute la configuration rejetterait les deux usages que le checkpoint doit
servir. Clé d'identité absente du fichier → valeur par défaut **déclarée par le client**
(`identity_defaults`), sinon erreur ; clé en trop ou différente → erreur nommant la clé, l'attendu
et le trouvé. `derived` stocke des valeurs recalculables depuis la configuration : le client les
recalcule à la reprise et échoue bruyamment si elles diffèrent (dérive du code de conversion
d'unités). Encodage figé : `Symbol` → `String` côté client ; valeurs admises `String`, `Bool`,
`Int`, `Float32`, `Float64`, `nothing` (classes de configuration seulement).

`fields` (formes vérifiées par le hook `validate_snapshot` du client — avec raffinement de maillage,
les formes font partie de l'état, d'où un hook et non une comparaison avec un `init_state` frais)
est séparé de `series` (historiques en ajout seul, longueur libre). Le compteur `cycle` est
**global** : l'échantillonnage se décide sur lui seul, jamais sur le cycle du segment, sinon
`7 + 13` cycles et `20` cycles ne donnent pas le même historique. `sample_final=false` par défaut.

### 4. Deux entiers de version, migration par rejet

- `container_version` : le layout du fichier (groupes, encodage des attributs). Propriété de
  `checkpoint_hdf5.jl`.
- `schema_version` : l'ensemble des clés d'**un** client. Propriété du client.

Dans cet incrément, toute version différente est **rejetée**. Le hook
`migrate(::Type{S}, snap, from_version)` existe et lève une erreur par défaut : le premier
changement de schéma sera une méthode à ajouter, pas une refonte.

### 5. Reprise depuis un `.krk` : différée, exception déclarée

L'invariant « tout ce que fait la plateforme est faisable depuis un `.krk` » n'est **pas** tenu
pour la reprise dans cet incrément. C'est une exception déclarée, pas un oubli. Forme visée : un
`Sweep` à état porté (chaque point part de l'état du précédent, avec `update_parameter!` entre
deux). Prérequis identifié : un run repris ne doit pas tronquer ses sorties de diagnostic — les CSV
et le PVD sont aujourd'hui ouverts en mode `"w"` ; ils devront être ouverts en ajout et recalés sur
le cycle du checkpoint.

### 6. Clients

- **Premier client :** le pilote d'électroconvection 2D (incrément suivant ; état et mapping dans
  leur propre fichier sous `src/drivers/`).
- **Second client nommé :** la boucle générique du runner (« avancer, checkpoint tous les N
  cycles, reprendre »), qui n'utilise que les verbes ci-dessus.
- **Garde pour tous les suivants :** `run_state_contract_suite` dans
  `test/platform/state_contract_suite.jl` — aller-retour à zéro pas en mémoire et via disque
  (égalité bit à bit), avance fractionnée = avance continue, contrôle négatif par champ stocké
  (un test à tolérance passe avec un morceau d'état manquant), export refusé hors frontière de
  cycle, rejets d'identité/schéma, différence de `run_control` acceptée.

**Raisons :** ces choix sortent d'une revue de conception contradictoire, expériences à l'appui
(redémarrage émulé comparé bit à bit à un run continu ; corruption de fichiers, kill en cours
d'écriture, renommage sous charge). Chacun coûte peu maintenant et serait une rupture de format ou
d'API plus tard.

**Conséquences :**
- `HDF5.jl` est une dépendance directe. `state.jl` ne l'importe pas : le choix dépendance dure /
  dépendance faible reste réversible sans toucher au contrat.
- Un client convertit ses tableaux device → hôte dans `snapshot` et hôte → device dans
  `restore_state(...; backend)` ; `restore_state` appelle `check_compatible` puis
  `validate_snapshot` **avant** toute allocation.
- `export_state` (propriété de la plateforme) refuse un état laissé en milieu de cycle par une
  exception (`at_boundary`) et toute valeur non finie.

**Hors périmètre (dette déclarée) :** migration de schéma ; reprise `.krk` et sorties en ajout ;
les autres pilotes ; writer en flux (l'export tient une copie hôte complète de l'état, à surveiller
en 3D) ; durabilité après crash de nœud sur Lustre (pas de `fsync` ; `.prev` atténue) ; déterminisme
bit à bit de la reprise sur CUDA (mêmes kernels, même ordre — à mesurer, pas à supposer) ;
unification avec le chemin AD `(u, p)`.
