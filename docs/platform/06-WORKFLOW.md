# MODÈLE OPÉRATOIRE — modifs, worktrees, git, docs, mandat

> Comment construire la plateforme sans chaos : éviter les collisions git, garder la double doc
> (humaine + LLM) à jour, faire vivre le mandat. Complète `04-HYGIENE-PROCESS.md` (stratégie
> worktrees/branches) avec le **processus de développement** lui-même.

## Le problème en une phrase

Une collision git = **deux missions qui éditent le même fichier**. La doc et le mandat dérivent
quand ils sont mis à jour *après* la PR au lieu de *dans* la PR. Tout le modèle découle de là.

---

## Règle 1 — Fichiers-goulots en voie UNIQUE, briques en voies PARALLÈLES

**Choke files de Kraken** (édités par presque tout → source des collisions) :

| Choke file | Pourquoi | Mitigation |
|------------|----------|------------|
| `src/KrakeNN.jl` | exports | n'ajouter qu'**une ligne d'export** par brique ; sérialiser |
| `simulation_runner.jl` (~1843 LOC) | dispatch par matching de chaîne | **casser en registre typé tôt** (plan 02) → disparaît comme choke file |
| `io/kraken_parser.jl` (~2010 LOC) | parser DSL | reloger sous `io/krk/` + dispatch par directive |
| `Project.toml` | deps | une mission « deps » sérialisée, pas dans chaque PR brique |
| nav de doc (code-générée) | sommaire | générée, pas éditée à la main (kraken-doc) |

**Règle :** toute modif touchant un choke file passe **sérialisée sur le trunk, une à la fois**.
Toute **brique** (= un *fichier neuf* sous `src/platform/`, `src/physics/`, `src/methods/`) part
en **worktree parallèle** → collision structurellement impossible.

**Corollaire prioritaire :** le refactor `simulation_runner → registre de méthodes typé` est le
plus gros réducteur de collisions. Après lui, ajouter une méthode = **un fichier + une ligne
d'enregistrement**, jamais une édition du monolithe.

---

## Règle 2 — Branches courtes, rebase quotidien, divergence bornée

Anti-pattern présent dans le dépôt : `dev-viscoelastic` à **238 commits** de divergence = merge
infernal. → mission = `feat/platform-<x>` | `fix/<bug>`, vivant **≤ jours**, **rebasé sur trunk
chaque jour** (petits conflits fréquents > gros conflit final), mergé puis **supprimé** (worktree
+ branche). Plafond **≤6 worktrees vivants** (cf. `04`). Audit périodique : `git-audit` skill.

---

## Règle 3 — « Fini » = code + test parité + benchmark + doc (un seul gate)

Une PR est GREEN seulement si **tout** est vert :

1. **Tests** : suite existante + **test de parité** (le contrat n'altère pas les nombres).
2. **Benchmark** (si nouvelle méthode/physique) : cas canonique + réf analytique + ordre de
   convergence + (si différentiable) gradient de réf. (principe #8 du brief)
3. **Doc dans la PR, jamais après** :
   - **Track B** — docstring API sur tout symbole public ;
   - **Track C** — **implication map LLM** ; le **lint implication-map est bloquant** (kraken-doc)
     → la doc LLM ne *peut pas* prendre de retard ;
   - **Track A** — doc humaine seulement si user-facing (tuto, page d'install, showcase).

→ **« Rien n'entre sans son benchmark NI sa doc. »** C'est la réponse au « update doc humaine+LLM
en cours de route » : ce n'est pas une étape séparée, c'est une condition de merge.

---

## Règle 4 — Mandat vivant, écrivain unique = Boss, ADR append-only

**Deux artefacts à ne JAMAIS confondre :**

| Artefact | Rôle | Visibilité | Qui écrit |
|----------|------|-----------|-----------|
| `docs/platform/*` | **design partageable** : plan, ADR de conception (ADR-01…) | versionné, peut devenir public | Boss |
| `.orchestrator/mandate.md` | **état d'orchestration vivant** : carte branches, statut missions, file d'attente | **local-only, jamais push** (verrouillé) | Boss seul |
| `.orchestrator/memory/boss.md` | mémoire narrative (décisions, pistes abandonnées) | local-only | Boss seul |

**Quand mettre à jour :** à chaque décision actée OU merge qui change l'architecture →
**ajouter** un ADR (jamais réécrire un ADR existant) + **mettre à jour** la carte branches/missions.
**Écrivain unique = Boss** (règle orchestrateur) ; les couches basses *suggèrent*, le Boss persiste.

**Dette à résorber :** `boss.md` = 272 Ko (trop gros). Créer le `.orchestrator/mandate.md` compact
que `kraken-architect` attend (et qui n'existe pas) ; dégonfler boss.md en mémoire narrative.

---

## Règle 5 — Le cycle de vie d'une modif (le loop unique)

```
0. Boss lit mandate.md + le plan (docs/platform) → choisit la prochaine mission
1. Check collision : la mission touche-t-elle un choke file ?
      ├─ OUI → voie TRUNK sérialisée (pas de worktree parallèle)
      └─ NON → branche feat/platform-<x> + worktree dédié
2. Engineer (Codex/Claude) code DANS ses zones autorisées
      + test parité + benchmark + docstring + implication map
3. 3 gates : tests ✓  benchmark ✓  doc-lint ✓
4. Boss review → MERGE (seul à committer) → delete worktree+branche
5. Boss : append ADR / update carte branches+missions (mandate.md)
6. Rebase les autres branches en vol sur le nouveau trunk
```

C'est la discipline orchestrateur **à la lettre** : chaque mission = un brief avec ses **zones
d'édition autorisées** (= la carte anti-collision) ; **seul le Boss committe** (= écrivain unique
code ET mandat). Rien d'exotique — on applique ce qui existe déjà.

---

## Règle 6 — La carte des zones (anti-collision, par mission)

Avant de lancer des missions en parallèle, le Boss tient une **carte zone→mission** : si deux
missions déclarent des zones qui se recoupent → on les **sérialise** ou on les **fusionne** en une.
Cette carte vit dans `mandate.md`. Le contrat aide structurellement : les briques étant des
fichiers neufs, leurs zones sont **disjointes par construction**.

---

## Règle 7 — Déterminiser via orchestrator + skills

- **`kraken-platform-contract`** (skill proposé, à valider) : encode le gate « brique = contrat +
  benchmark + doc » → tout ajout suit le même protocole.
- **`git-audit`** (existe) : hygiène périodique worktrees/branches.
- **`kraken-doc`** (existe) : lint implication-map bloquant + build/deploy Vitepress.
- **`orchestrator`** : Boss/Department/Engineer = la machine d'exécution (zones autorisées,
  écrivain unique).

---

## Décisions VERROUILLÉES (2026-06-09)

1. **Branche de base = `dev/platform` dédié**, coupé depuis `dev/v0.3-campaign` (qui porte l'AD).
   La fondation (Phases 0-2, choke files) y vit **sérialisée** ; les briques partent en worktrees
   enfants. Isole la fondation, garde la release v0.3 propre. → ADR-02.
2. **Mandat = `.orchestrator/mandate.md` compact** (carte branches + missions + index ADR), `boss.md`
   reste mémoire narrative. → ADR-03.
3. **Doc-gate strict dès la PR 1** : lint implication-map bloquant + docstring exigés sur toute PR
   touchant un symbole public. → ADR-04.
