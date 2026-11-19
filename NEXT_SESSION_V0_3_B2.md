# Reprise v0.3 — approach (E) Multi-block O-grid cylinder Re=100

Branche : `slbm-paper`.

**Session 2 update (2026-04-21, end)** : tous les blockers d'infra B.2 sont
levés (commits `4451b7f`, `fe7242f`, `8eb6fd0`). Total suite
multiblock : 683 tests verts. La prochaine session peut commencer
directement par le driver + .geo. Voir section "Status B.2" et
"Driver à écrire" ci-dessous.

## Contexte paper — état 2026-04-21

La matrice comparative WP-MESH-6 vise 4 approches sur **même nombre de
cellules** pour le cylindre Re=100 (Williamson 1996, Schäfer-Turek 2D-2) :

| # | Approche | Mesh | Status |
|---|---|---|---|
| (A) | Cartésien uniforme + halfway-BB | `cartesian_mesh` | ✅ Aqua H100, D=40 Cd=1.652 |
| (B) | Cartésien uniforme + LI-BB v2 | `cartesian_mesh` | ✅ Aqua H100, D=40 Cd=1.649 |
| (C) | gmsh Bump (clustering sur parois rect) + SLBM + LI-BB | gmsh Transfinite Bump | ✅ Aqua D=40 Cd=1.630 ; NaN D=20/80 |
| **(D)** | **cylinder_focused_mesh (clustering sur cylindre) + SLBM + LI-BB** | `cylinder_focused_mesh` | **🟡 driver committé `2facd3d`, à submit sur Aqua** |
| **(E)** | **Multi-block O-grid + extensions rect + SLBM + LI-BB per block** | gmsh multi-surface | **🔴 infrastructure B.2.1 ✓, reste B.2.2 + B.2.3 + driver** |

## Livré cette session (3 commits)

1. **`9d08e74`** — v0.3 Phase B.1 gmsh multi-surface loader + 44 tests
   - `load_gmsh_multiblock_2d(path; layout=:auto)` in
     [src/multiblock/mesh_gmsh_multiblock.jl](src/multiblock/mesh_gmsh_multiblock.jl)
   - Canal 2-block + O-grid 4-block cylindre validés
   - Sanity colocation check flip-aware
2. **`2facd3d`** — approche (D) cylinder_focused single-block
   - Driver production CUDA : [hpc/wp_mesh_6_focus_aqua.jl](hpc/wp_mesh_6_focus_aqua.jl)
   - PBS : [hpc/wp_mesh_6_focus_aqua.pbs](hpc/wp_mesh_6_focus_aqua.pbs)
   - Smoke Metal : [tmp/smoke_focus_local_metal.jl](tmp/smoke_focus_local_metal.jl)
     (F32 instable à D=20/40 — même symptôme que bump smoke, instabilité
     F32 pas F64, à valider sur Aqua)
   - Pipeline : `cylinder_focused_mesh` + `local_cfl=true` +
     `compute_local_omega_2d(:quadratic)` + `slbm_trt_libb_step_local_2d!`
   - Sweep strength ∈ (1.0, 2.0) sur 3 résolutions D ∈ (20, 40, 80)
3. **`4451b7f`** — v0.3 Phase B.2.1 shared-node ghost exchange + 363 tests
4. **`fe7242f`** — v0.3 Phase B.2.2 block reorientation + BFS autoreorient + 42 tests
   - `reorient_block(block; flip_ξ, flip_η)` transforms mesh + tags
   - `autoreorient_blocks(mbm)` BFS across the interface graph with
     per-block cumulative flip_state so multi-hop chains propagate
     correctly
5. **`8eb6fd0`** — v0.3 Phase B.2.3 extended-grid mesh for SLBM + 30 tests
   - `extend_mesh_2d(mesh; n_ghost)` linearly extrapolates + refits
     spline metric
   - `build_block_slbm_geometry_extended(block; n_ghost, local_cfl)`
     one-shot helper for per-block SLBM on ghost-layer arrays
   - `extend_interior_field_2d(field, n_ghost)` lifts `is_solid`,
     `q_wall`, `uw_x`, `uw_y` onto the extended grid
6. **`439e4db`** — approach (E3) 3-block Cartesian multi-block cylinder
   driver + PBS + Metal smoke — first multi-block Re=100 cylinder
   driver running end-to-end (pipeline: exchange_ghost + wall_ghost +
   per-block step + per-block BCSpec with HalfwayBB-as-noop for
   interface sides + drag on block-C interior view).  Rewrote
   `fill_physical_wall_ghost_2d!` as KernelAbstractions kernels for
   GPU compatibility.
7. **`2935a26`** — `reorient_block` + `autoreorient_blocks` extended
   with TRANSPOSE (ξ↔η swap) for the dihedral-8 group. Needed for
   the 8-block O-grid topology where the walker produces orthogonal
   ξ/η orientations between neighbour blocks. `tmp/gen_ogrid_rect_8block.jl`
   writes the 8-block .geo programmatically; `tmp/validate_ogrid_rect_8block.jl`
   loads + reorients + sanity-checks it → 0 errors, 8 warnings
   (expected shared-node), topology READY for (E-full) driver.

**Aqua jobs submitted** :
- `20185587.aqua` — wp_mesh_6_focus (approach D cylinder-focused)
- `20186108.aqua` — wp_mesh_6_multi (approach E3 3-block Cartesian)

**Total multiblock v0.3 : 683 tests verts** (31 + 163 + 10 + 44 + 363 + 42 + 30).

## À faire en début de prochaine session

### Priorité 1 — Submit (D) sur Aqua pour résultats paper

```bash
cd ~/Documents/Recherche/Kraken.jl
./hpc/sync_to_aqua.sh         # rsync branche slbm-paper
ssh <user>@aqua.qut.edu.au
cd Kraken.jl && qsub hpc/wp_mesh_6_focus_aqua.pbs
```

Attendre ~6h, puis `./hpc/pull_results_from_aqua.sh`.

**Résultat attendu** : Cd ≈ 1.64 à D=40 pour strength=1 et strength=2,
Cl_RMS ≈ 0.28, MLUPS meilleur que Bump (par meilleure condition du τ).

### Priorité 2 — ~~B.2.2 block reorientation~~ ✅ LIVRÉ `fe7242f`

`reorient_block(block; flip_ξ, flip_η)` + `autoreorient_blocks(mbm)`
(BFS avec cumulative flip_state).

### Priorité 3 — ~~B.2.3 SLBM extended~~ ✅ LIVRÉ `8eb6fd0`

`extend_mesh_2d(mesh; n_ghost)`,
`build_block_slbm_geometry_extended(block; n_ghost, local_cfl)`,
`extend_interior_field_2d(field, n_ghost; pad_value)`.

### Priorité 4 — Driver approach (E-full) — 2 bugs infra à lever avant Aqua

**Status** : driver squelette écrit (`hpc/wp_mesh_6_ogrid_aqua.jl`,
commit à venir) mais 2 issues d'infra bloquent la soumission Aqua :

**(I1) ~~Loader topological attribue les mauvais tags physiques~~ ✅ FIXÉ `76728c8`**

Midpoint check ajouté à `_match_curve_to_edge` : chaque bord candidat
est scoré par distance entre midpoint du bord et midpoint de la courbe
(via gmsh `getParametrizationBounds` + `getValue`). L'arc et son spoke
voisin partagent un endpoint mais divergent au milieu → désambiguïsation
propre. Validation sur 8-block : toutes les `:west` edges à mean radius
= R_in, `:east` au bord extérieur, match physique.

**(I2) extend_mesh_2d folde le Jacobien pour arcs — STILL OPEN**

Avec `radial_progression=0.8` dans le .geo, l'extension radiale
(ξ direction) près du cylindre est OK (cells clustered). Mais
l'extension ANGULAIRE (η direction) folde toujours sur les arcs
serrés : les spokes du ring bloc ont des longueurs différentes
selon le rayon, donc extrapoler en η aux petits rayons produit des
cellules qui croisent le ring bloc voisin. Fold au bord j=1 ou j=Nη.

**Fix proposé** (priorité 1 prochaine session) :
- **Option 1** : Extension géométrique basée sur la bissectrice. Pour
  le ghost η, utiliser une rotation (autour du centre cylindre) des
  points intérieurs plutôt qu'une extrapolation linéaire. Pour les
  extensions non-wall (interfaces), fetch depuis le voisin via
  exchange à l'init.
- **Option 2** : Construire `SLBMGeometry` sur le mesh NON-étendu
  `(Nξ, Nη)` et écrire un kernel variant qui accepte state arrays
  étendus `(Nξ+2Ng, Nη+2Ng)` mais stencils interior-sized, en
  dispatchant sur la bordure via ghost-layer lookup (requiert
  modification du kernel SLBM — plus invasif).

Simplification alternative : pour les blocs n'ayant PAS de bord courbé
sur l'axe à étendre, utiliser l'extension linéaire existante. Pour les
blocs O-grid avec arc, implémenter une variante `extend_mesh_2d_arc`
qui connaît le centre du cylindre et rotationne les points intérieurs
pour générer un ghost cohérent.

**Driver stagé** :

- `hpc/wp_mesh_6_ogrid_aqua.jl` — full driver skeleton (n_blocks=8,
  hybrid step loop, drag aggregation). Tous les appels composent
  l'infra livrée (`load_gmsh_multiblock_2d` + `autoreorient_blocks` +
  `build_block_slbm_geometry_extended` + `exchange_ghost_shared_node_2d!`
  + per-block bcspec + `compute_drag_libb_mei_2d` sum).
- `hpc/wp_mesh_6_ogrid_aqua.pbs` — PBS (ne PAS soumettre avant fix).
- `tmp/smoke_ogrid_local_metal.jl` — Metal smoke qui reproduit I1/I2.
- `tmp/gen_ogrid_rect_8block.jl` — .geo generator (déjà validé
  structurellement, commit `2935a26`).

**Étapes** (répétition pour référence rapide, une fois I1+I2 résolus) :

**Infrastructure complète, reste l'assemblage** : écrire
[hpc/wp_mesh_6_multi_aqua.jl](hpc/wp_mesh_6_multi_aqua.jl).

**Topologie 8-block proposée** (inspirée de rheoTool mais adaptée au
domaine Schäfer-Turek 1.0 × 0.5) :
- Centre : 4 blocs O-grid annulaires de `R_in=0.025` à `R_out=0.1`,
  quadrants NE / NW / SW / SE
- 4 blocs extension "+" forme : N, E, S, W, chacun un rectangle
  trapézoïdal entre le quart-arc de `R_out` et la frontière du
  rectangle. PAS de blocs de coin dans cette topologie : les 4 blocs
  extension se rejoignent au point médian de chaque bord du rectangle
  → 4 triangles manquants (coins du rectangle). Pour couvrir ces coins
  sans rendre la topologie trop complexe, 2 options :
  - **Option A** : étendre les 4 blocs extension jusqu'au coin via
    un 5ème sommet — mais 5 sommets cassent Transfinite 4-corner.
    Nécessite 4 blocs de coin additionnels (total 12).
  - **Option B (simpler MVP)** : limiter le domaine aux blocs
    directement autour du cylindre (pas de coin). Domaine efficace
    est un octogone plutôt qu'un rectangle. Comparer à (A)(B)(D) sur
    même cell count via un domaine Cartésien-uniforme équivalent.

**Recommandé pour paper-MVP** : **option B** (8 blocs, octogone).
Message paper : "multi-block O-grid résout body-fitted ; la forme
octogonale résultante n'impacte pas Cd/Cl dans la plage de référence
car les coins rectangulaires sont hors zone de sillage".

**Étapes** :
1. `.geo` intégré dans le driver (8 surfaces, Transfinite, Recombine)
   avec Physical Curves `cylinder`, `inlet`, `outlet`, `wall_top`,
   `wall_bot` (ou `farfield` en MVP octogone).
2. `mbm, _ = load_gmsh_multiblock_2d(geo_path; FT=T, layout=:topological)`
   puis `mbm = autoreorient_blocks(mbm)` + `sanity_check_multiblock(mbm)`.
3. Par bloc, identifier si c'est un bloc O-grid (touche le cylindre)
   ou un bloc extension :
   ```julia
   is_cylinder_block = [any(e -> b.boundary_tags[e] === :cylinder,
                             EDGE_SYMBOLS_2D) for b in mbm.blocks]
   ```
4. Allouer `BlockState2D` par bloc, `n_ghost=1`.
5. Pour chaque bloc O-grid :
   - `mesh_ext, geom_ext = build_block_slbm_geometry_extended(blk;
                                                                n_ghost=1,
                                                                local_cfl=true)`
   - Construire `is_solid_ext`, `q_wall_ext`, `uw_x_ext`, `uw_y_ext`
     via `extend_interior_field_2d` sur les arrays interior calculés
     par `precompute_q_wall_slbm_cylinder_2d(blk.mesh, …)`.
   - `sp, sm = compute_local_omega_2d(mesh_ext; ν, :quadratic, τ_floor=0.51)`.
6. Pour chaque bloc extension : arrays extended de zéros
   (`is_solid`, `q_wall`).
7. Time loop :
   ```julia
   for step in 1:N
       exchange_ghost_shared_node_2d!(mbm, states)   # B.2.1
       fill_physical_wall_ghost_2d!(mbm, states)     # wall sides
       for (k, blk) in enumerate(mbm.blocks)
           Nx_ext, Ny_ext = ext_dims(states[k])
           if is_cylinder_block[k]
               slbm_trt_libb_step_local_2d!(f_out[k], states[k].f, …,
                                              q_wall_ext[k], …,
                                              geom_ext[k], sp[k], sm[k])
           else
               fused_trt_libb_v2_step!(f_out[k], states[k].f, …,
                                         zeros_q_wall[k], zeros_uw_x[k],
                                         zeros_uw_y[k], Nx_ext, Ny_ext, ν)
           end
           # Apply physical-BC only on the block's physical-BC sides;
           # use NoBC (or skip) for interface sides.
           apply_bc_rebuild_2d!(interior_f(f_out[k]), interior_f(states[k].f),
                                  bcspec[k], ν, blk.mesh.Nξ, blk.mesh.Nη)
       end
       for k in eachindex(states); states[k].f, f_out[k] = f_out[k], states[k].f; end
   end
   ```
8. **Aggregate drag** : sur les blocs O-grid (ceux avec cylinder sur
   un bord), sommer `Fx, Fy` de `compute_drag_libb_mei_2d(f_interior,
   q_wall_interior, uw_x_interior, uw_y_interior, Nx, Nη)` par bloc,
   puis `Cd = 2 Σ Fx / (u² · D)`.

**Pièges critiques à surveiller** :
- **BC per-block** : `apply_bc_rebuild_2d!` écrase la colonne boundary.
  Pour les bords interface, utiliser une `NoBC()` ou équivalent (à
  définir — pas sûr que Kraken en ait). Alternativement, appeler
  `apply_bc_rebuild_2d!` SEULEMENT sur les blocs qui ont un bord
  physique (pas tous), et construire un bcspec par bloc qui
  correspond à ses 4 bords effectifs.
- **Orientation des drivers GPU** : `build_slbm_geometry_extended`
  tourne CPU uniquement ; transférer geom via `transfer_slbm_geometry(geom_h, CUDABackend())`.
- **NaN sur Metal F32** : déjà connu pour (C) et (D) à D=20/40. La
  validation est faite sur Aqua H100 F64.

### Priorité 5 — Paper matrix update

Mettre à jour [paper/wp_mesh_6.md](paper/wp_mesh_6.md) avec lignes (D)
et (E). Killer chiffre attendu : (E) atteint Cd + Cl_RMS de référence
à budget cellules < (A) et (B).

## Handoff B.2 cheat-sheet

```
Pipeline approach (E) step :
  exchange_ghost_shared_node_2d!(mbm, states)   # NEW B.2.1 ✓
  fill_physical_wall_ghost_2d!(mbm, states)     # Phase A.5c ✓
  for (k, blk) in enumerate(mbm.blocks)
      Nx_ext, Ny_ext = ext_dims(states[k])
      if is_cylinder_block[k]
          slbm_trt_libb_step_local_2d!(…, geom_ext[k], sp[k], sm[k])  # needs B.2.3
      else
          fused_trt_libb_v2_step!(…, Nx_ext, Ny_ext, ν)
      end
      apply_bc_rebuild_2d!(interior_f(f_out[k]), interior_f(state[k].f),
                             bcspec[k], ν, blk.mesh.Nξ, blk.mesh.Nη)
  end
  swap(states[k].f, f_out[k]) for each k
```

Points critiques :
- **Orientation ξ/η** : doit être cohérente OU exchange extended pour
  handler tous les pairs. Voir Priorité 2.
- **Interface + LI-BB corner cells** : coexistent sans interaction.
  LI-BB est in-kernel (brick `ApplyLiBBPrePhase`); interface ghost
  est pre-step.
- **Drag sum** : `compute_drag_libb_mei_2d` par bloc qui touche
  le cylindre, sum Fx/Fy puis normaliser par `u² × D`.

## Mémoire utile

- [project_multiblock_v03.md](~/.claude/projects/-Users-guillaume-Documents-Recherche-Kraken-jl/memory/project_multiblock_v03.md)
  — état phase A + B.1 + B.2.1
- [project_slbm_local_cfl.md] — local-CFL 1.7% err
- [project_wp_mesh_6_bump.md] — résultats (C) et instabilité τ→0.5
- [project_kernel_dsl.md] — LI-BB V2 + GPU drag patterns
