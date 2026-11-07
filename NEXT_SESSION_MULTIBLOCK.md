# Multi-block Poiseuille ladder — debug session

Branche : `slbm-paper`

## Bug identifié

Le kernel LBM (`fused_bgk_step!` ET `slbm_bgk_step!`) applique **halfway-BB
à TOUS les bords du domaine étendu**, y compris les bords qui sont des
interfaces multi-bloc. L'exchange ghost remplit les ghosts correctement,
mais le kernel les IGNORE et fait BB → velocity = 0 à l'interface.

**Preuve** : un Poiseuille 2-bloc nord-sud donne ux=0.0002 à l'interface
(attendu: 0.04), que ce soit avec `fused_bgk_step!`, `slbm_bgk_step!`,
shared-node ou non-overlap exchange. Les interfaces est-ouest fonctionnent
car le ZouHe BC écrase le BB aux bords inlet/outlet.

## Ce qui fonctionne

- 1-bloc Poiseuille : ux matche l'analytique (décalage ~2.5% + asymétrie
  ~1.8% aux coins, à investiguer séparément)
- 2-blocs est-ouest : ux matche le 1-bloc à 0.47% — l'interface est-ouest
  est transparente PARCE QUE le ZouHe inlet/outlet écrase le BB du kernel
- L'exchange ghost (shared-node et non-overlap) remplit les ghosts
  correctement (vérifié par print des valeurs)
- Le `apply_bc_rebuild_2d!` fonctionne (vérifié step par step)

## Ce qui ne fonctionne PAS

- 2-blocs nord-sud : l'interface se comporte comme un mur. Deux Poiseuilles
  séparés se développent dans chaque demi-canal.
- 4-blocs (2×2) : même problème à l'interface horizontale
- 8-blocs O-grid : problème masqué par les BCs incorrectes mais même cause

## Fix nécessaire

Le kernel `fused_bgk_step!` (et `slbm_bgk_step!`, et le DSL `PullHalfwayBB`)
applique BB aux bords i=1, i=Nx, j=1, j=Ny du domaine étendu. Pour le
multi-bloc, les bords qui correspondent à des interfaces ne doivent PAS
avoir de BB — le ghost exchange fournit déjà les bonnes valeurs.

Options :
1. **Passer un masque `is_interface`** au kernel pour sauter le BB aux
   interfaces. Nécessite de modifier tous les kernels.
2. **Post-step rebuild** : après le kernel (qui fait BB partout), écraser
   les rangées interface avec les valeurs correctes recalculées depuis les
   ghosts. Similaire à `apply_bc_rebuild_2d!` mais pour les interfaces.
3. **Séparer streaming et BB** : le kernel fait le pull streaming sans BB,
   une passe séparée fait le BB uniquement aux vrais murs. Plus propre mais
   refactor important.
4. **Ne pas lancer le kernel sur les rangées de bord** : réduire le
   `ndrange` pour exclure les bords, puis laisser le ghost+exchange
   gérer les populations aux bords. Risque de perdre les BCs.

## Approche recommandée : ladder progressive

**Avant de fixer le kernel**, valider chaque étape sur un Poiseuille
analytique. À chaque niveau, vérifier :

### Checklist par niveau

1. **Profil ux(y) au centre du canal** : matche l'analytique ?
   - Poiseuille exact : `u(y) = u_max × 4y(H-y)/H²` où H = hauteur effective
   - Avec halfway-BB : le mur effectif est à y = -dy/2 et y = Ly + dy/2
   - Vérifier que le max est au bon endroit et à la bonne valeur
2. **Symétrie** : `ux(y) = ux(H-y)` ? Asymétrie = bug de coin
3. **Densité** : ρ ≈ 1.0 partout ? Gradient ρ(x) = gradient de pression
4. **Coins** : ρ aux 4 coins identiques 2 à 2 ? Sinon = bug BC corner
5. **Convergence** : la solution change-t-elle encore au dernier step ?

### Niveaux

| Niveau | Config | Ce qu'on valide |
|--------|--------|-----------------|
| L0 | 1-bloc, Nx=80, Ny=40 | Référence Poiseuille |
| L1-EW | 2-blocs est-ouest (split x=Lx/2) | Interface ξ transparente |
| L1-NS | 2-blocs nord-sud (split y=Ly/2) | Interface η transparente |
| L2 | 4-blocs (2×2) | Les deux interfaces simultanées |
| L3 | 8-blocs O-grid sans cylindre | Topologie annulaire |
| L4 | 8-blocs O-grid avec cylindre | E-full complet |

**Ne PAS passer au niveau suivant tant que le précédent n'est pas validé
bit-exact (ou à <0.1% près) par rapport au niveau 0.**

## Autres bugs trouvés cette session (à corriger)

### Bug moments post-BC (FIXÉ)
`apply_bc_rebuild_2d!` modifie f_out aux BCs mais ne met pas à jour
ρ/ux/uy → les diagnostics affichent les moments SLBM (faux au bord).
**Fix** : ajouté `ρ_out, ux_out, uy_out` kwargs + `_update_bc_moments_2d!`.

### Bug ZouHe direction sur O-grid
Le ZouHe east assume que la normale à la face est en +x. Sur le O-grid,
la direction ξ à l'east varie (dX/dξ peut être négatif, dY/dξ non-nul).
Le profil inlet doit être signé selon `sign(dX/dξ)` à l'east edge.
**Fix nécessaire** : soit signer le profil, soit implémenter un ZouHe
qui projette sur la normale locale.

### Collision régularisée dans le BC kernel
`_trt_collide_local` dans `boundary_rebuild.jl` a été modifié pour
utiliser la collision régularisée (filtre les ghost modes). Ça stabilise
le ZouHe à bas τ. **À tester** : est-ce que ça change les résultats
Poiseuille ? Vérifier bit-exactness vs la version TRT standard.

## Fichiers modifiés cette session

| Fichier | Modification |
|---------|-------------|
| `src/multiblock/wall_ghost.jl` | `fill_slbm_wall_ghost_2d!` avec extrapolation inlet/outlet |
| `src/curvilinear/slbm.jl` | `PullSLBMBiquad`, `CollideRegularizedTRTLocal`, biquad/reg specs |
| `src/kernels/dsl/bricks.jl` | `PullSLBMBiquad`, `CollideRegularizedTRTLocal` bricks |
| `src/kernels/boundary_rebuild.jl` | `_trt_collide_local` régularisé + moments update |
| `src/kernels/enzyme_rules.jl` | `dq_wall_dR_cylinder` (dérivée analytique géométrie) |
| `src/KrakeNN.jl` | Exports + include enzyme_rules |
| `docs/src/theory/10_limitations.md` | Section τ > 1/2 détaillée |

## Scripts diagnostiques utiles

| Script | But |
|--------|-----|
| `tmp/ladder_multiblock_plot.jl` | Plot ladder L0/L1/L2 |
| `tmp/ladder_ns_plot.jl` | Plot 2-bloc N-S |
| `tmp/audit_outlet_drift.jl` | Audit ZouHe stabilité |
| `tmp/diag_efull_ghost_pops.jl` | Diagnostic ghost pops lues par SLBM |
| `tmp/plot_efull_flow.jl` | Visualisation O-grid |

## Résumé pour la prochaine session

1. **Fixer le BB aux interfaces** dans le kernel (le cœur du problème)
2. **Valider L1-NS** : le Poiseuille doit traverser l'interface nord-sud
3. **Valider L2** : 4-blocs identique au 1-bloc
4. **Valider L3** : O-grid sans cylindre, flow traverse les 8 blocs
5. **Valider L4** : O-grid + cylindre, Cd convergé
6. Ensuite seulement : étude 3D sphère
