# E-full O-grid — Session 2026-04-22

Branche : `slbm-paper`

## Bug résolu : NaN explosion en 50 steps

### Root cause identifiée

**`fill_physical_wall_ghost_2d!` ne remplit que 3/9 populations par arête ghost.**

Sur un maillage Cartésien, le streaming standard (pull ±1 en i ou j) ne lit que
3 populations spécifiques depuis le ghost mur (cqx>0 pour west: q={2,6,9}).
Les 6 autres ne sont jamais lues → les laisser à 0 est inoffensif.

Sur un maillage **curvilinéaire avec SLBM**, les departure points sont obliques
(mappés via la métrique). Une cellule intérieure au bord du cylindre (west)
peut avoir un departure pour q=3 (cqy=1 physique) qui passe dans le ghost
west à cause de la composante ξy ≠ 0 de la métrique. Le bilinéaire lit
`f_in[ghost, j, 3]` qui vaut 0 → ρ s'effondre → NaN en ~50 steps.

**Preuve quantitative** (script `diag_efull_ghost_pops.jl`) :
- Block ring_3 : 49 lectures "bad" au ghost west (q={3,4,7,8})
- Block ring_4 : 49 lectures "bad" au ghost west (q={4,5,7,8})
- Chaque bloc a 80-238 lectures de populations non-remplies

### Fix implémenté

1. **`fill_slbm_wall_ghost_2d!(mbm, states)`** — nouvelle fonction
   (src/multiblock/wall_ghost.jl) qui copie les 9 populations de la
   rangée frontière vers le ghost pour chaque arête physique.
   Appelée AVANT `fill_physical_wall_ghost_2d!` qui écrase ensuite
   les 3 populations réfléchies avec les valeurs BB plus précises.

2. **`dx_ref` kwarg dans `build_slbm_geometry` et
   `build_block_slbm_geometry_extended`** — permet de passer le
   dx_ref global en multi-bloc au lieu d'utiliser le dx_ref per-bloc
   (qui peut différer de 2.16× entre blocs sur le O-grid).

### Résultat

| Config | Step 1 u_max | Step 50 | Step 500 |
|--------|-------------|---------|----------|
| **Avant** (original) | 0.35 | NaN 💥 | — |
| **Après** (fix) | 0.04 ✓ | 0.04 ✓ | 0.04 (inner) |

Re=100, D_lu=20 tourne 500 steps sans NaN sur CPU F64.

### Problème résiduel : drift lent aux outlets (ring_0/7)

- Step 100 : u_max=0.11 aux blocs outlet
- Step 500 : u_max=0.33 aux blocs outlet
- Les blocs intérieurs restent stables (u_max ≈ 0.04)
- Présent aussi à Re=20 (drift plus lent)
- **PAS causé par τ→0.5** (Re=20 donne τ=1.15, même drift)

Hypothèses pour le drift outlet :
1. Le ghost boundary-copy (zeroth-order extrapolation) n'est pas assez
   précis pour le SLBM aux cellules grossières de l'outlet
2. L'interaction ZouHePressure + SLBM sur maillage curvilinéaire
3. La conservation de masse du SLBM bilinéaire aux interfaces bloc

### Call sequence de production

```julia
exchange_ghost_shared_node_2d!(mbm, states)
fill_slbm_wall_ghost_2d!(mbm, states)       # NEW: full pre-fill
fill_physical_wall_ghost_2d!(mbm, states)    # overwrites 3 reflected
for k in 1:n_blocks
    slbm_trt_libb_step_local_2d!(...)
end
```

Pour le dx_ref global :
```julia
mesh_ext, geom = build_block_slbm_geometry_extended(blk;
                    n_ghost=ng, local_cfl=false, dx_ref=dx_ref_global)
```

## Fichiers modifiés

| Fichier | Modification |
|---|---|
| `src/multiblock/wall_ghost.jl` | `fill_slbm_wall_ghost_2d!` + kernels `_copy_col/row_kernel_2d!` |
| `src/curvilinear/slbm.jl` | `dx_ref` kwarg dans `build_slbm_geometry` |
| `src/multiblock/mesh_extend.jl` | `dx_ref` kwarg dans `build_block_slbm_geometry_extended` |
| `src/KrakeNN.jl` | Export `fill_slbm_wall_ghost_2d!` |

## Scripts diagnostiques (tmp/)

| Script | But |
|---|---|
| `diag_efull_inspect.jl` | Inspecte blocs post-autoreorient, J, departures, τ |
| `diag_efull_ghost_pops.jl` | Identifie les lectures ghost non-remplies par population |
| `diag_efull_prefill.jl` | Vérifie que le pre-fill equilibrium stabilise |
| `diag_efull_both_fixes.jl` | Test les 2 fixes ensemble (ghost + dx_ref) |
| `diag_efull_production.jl` | Validation production avec l'API propre |

## Prochaines étapes

1. **Investiguer le drift outlet** — essayer :
   - Extrapolation linéaire du ghost au lieu de copy (ghost = 2*boundary - interior)
   - Augmenter le n_ghost à 2 pour les blocs outlet
   - Tester avec outlet Neumann (zero-gradient) au lieu de ZouHePressure
   - Tester en single-block SLBM sur un ring seul (isoler multi-block vs BC)

2. **Valider sur Aqua H100 FP64** — D=20/40/80 sweep pour convergence Cd

3. **Optimiser** — le `fill_slbm_wall_ghost_2d!` fait un kernel launch
   par arête × bloc. Fusionner en un seul kernel si perf critique.
