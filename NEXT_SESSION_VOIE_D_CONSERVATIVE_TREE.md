# Next Session: Voie D - Conservative Cell-Centered Tree Refinement

Branche: `slbm-paper`

## But de la session

Implementer une voie D separee du raffinement actuel:

```text
cartesien coarse + patch cartesien fine ratio 2
coarse inactif dans la zone raffinee
fine actif dans la zone raffinee
coarse actif hors patch
transferts coarse/fine par populations integrees F_i = f_i * volume
```

Le but de la premiere session n'est PAS de faire un cas Poiseuille ou cylindre.
Le but est de prouver que les operateurs coarse/fine conservent exactement les
populations orientees, puis masse et momentum.

## Pourquoi cette voie

La branche actuelle a deja une voie patch + ghost + Filippova-Hanel:

- `src/refinement/refinement.jl`
- `src/kernels/refinement_exchange_2d.jl`
- `src/refinement/time_stepping.jl`

Cette voie garde des etats coarse et fine actifs dans/pres de l'overlap, puis
essaie de recoller par ghost fill, restriction, interpolation temporelle et
reflux. La voie D doit etre isolee pour ne pas melanger les raisonnements.

La voie D suit plutot l'idee:

```text
fine -> coarse: coalescence conservative
coarse -> fine: explosion conservative
```

Noms litterature utiles:

- cell-centered grid refinement
- volumetric LBM refinement
- conservative coarse/fine coupling
- coalescence / explosion

References a garder en tete, sans refaire la revue:

- Chen et al. 2006, volumetric formulation / PowerFLOW.
- Rohde et al. 2006, mass-conservative local grid refinement.
- Coreixas & Latt 2025, conservative cell-centered grid refinement.

## Regles anti-rabbit-hole

Interdits pendant cette session:

- pas de SLBM;
- pas de maillage deforme;
- pas de multiblock;
- pas de mur;
- pas de cylindre;
- pas de GPU;
- pas de performance;
- pas d'interpolation libre de `u`;
- pas de moyenne de vitesse `u` pour le momentum;
- pas de comparaison de profil fluide avant les tests d'invariants.

Autorise:

- D2Q9 uniquement;
- CPU `Array`;
- ratio 2 uniquement;
- patch rectangulaire fixe;
- tests unitaires algebraiques;
- tests de transport sans collision;
- collision BGK seulement apres invariants de transport.

Critere de progression:

```text
Si masse/momentum ne ferment pas a l'erreur machine, ne rien ajouter.
```

## Fichiers a creer

Creer:

```text
src/refinement/conservative_tree_2d.jl
test/test_conservative_tree_2d.jl
```

Modifier:

```text
src/KrakeNN.jl
test/runtests.jl
```

Emplacement dans `src/KrakeNN.jl`:

```julia
# --- Grid refinement ---
include("refinement/refinement.jl")
include("refinement/conservative_tree_2d.jl")
include("kernels/refinement_exchange_2d.jl")
```

Ajouter les exports pres des exports refinement existants si necessaire:

```julia
export d2q9_cx, d2q9_cy
export coalesce_F_2d!, explode_uniform_F_2d!
export mass_F, momentum_F, moments_F
export conservative_tree_parent_index
```

Dans `test/runtests.jl`, ajouter juste apres `test_refinement.jl`:

```julia
include("test_conservative_tree_2d.jl")
```

Commande de test focalisee:

```bash
julia --project=. -e 'using Test; using KrakeNN; include("test/test_conservative_tree_2d.jl")'
```

## Convention de donnees

Pour cette voie, utiliser des populations integrees:

```text
F_i = f_i * volume_cellule
```

Ne pas commencer par les densites `f_i`. Les densites reviennent seulement
quand on ajoute une collision locale:

```text
f_i = F_i / volume
F_i = f_i * volume
```

Pour le ratio 2 en 2D:

```text
volume_coarse = 4 * volume_fine
```

On peut prendre:

```julia
vol_fine = 1.0
vol_coarse = 4.0
```

ou bien:

```julia
vol_coarse = 1.0
vol_fine = 0.25
```

Choisir une convention et ne pas la changer dans les tests. Pour les premiers
tests, le plus simple est de manipuler directement `F`, donc aucun volume n'est
necessaire.

## API minimale a implementer

Dans `src/refinement/conservative_tree_2d.jl`:

```julia
const D2Q9_CX_INT = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const D2Q9_CY_INT = (0, 0, 1, 0, -1, 1, 1, -1, -1)

@inline d2q9_cx(q::Int) = D2Q9_CX_INT[q]
@inline d2q9_cy(q::Int) = D2Q9_CY_INT[q]
```

Fonctions algebraiques:

```julia
function coalesce_F_2d!(Fp::AbstractVector{T},
                        Fc::AbstractArray{T,3}) where T
    # Fc has size (2, 2, 9), integrated child populations.
    # Fp has length 9, integrated parent populations.
end
```

Semantique exacte:

```text
Fp[q] = Fc[1,1,q] + Fc[2,1,q] + Fc[1,2,q] + Fc[2,2,q]
```

Attention a l'ordre des indices: utiliser toujours `(ix, iy, q)`.

```julia
function explode_uniform_F_2d!(Fc::AbstractArray{T,3},
                               Fp::AbstractVector{T}) where T
    # Fc[ix, iy, q] = Fp[q] / 4
end
```

Diagnostics:

```julia
mass_F(F::AbstractVector) = sum(F)

function momentum_F(F::AbstractVector{T}) where T
    mx = zero(T)
    my = zero(T)
    for q in 1:9
        mx += T(d2q9_cx(q)) * F[q]
        my += T(d2q9_cy(q)) * F[q]
    end
    return mx, my
end

function moments_F(F::AbstractVector{T}) where T
    m = mass_F(F)
    mx, my = momentum_F(F)
    return m, mx, my
end
```

Helper parent/enfant:

```julia
@inline function conservative_tree_parent_index(i_f::Int, j_f::Int)
    # Fine indices are 1-based.
    # Return parent cell index and child-local index.
    i_parent = (i_f + 1) >>> 1
    j_parent = (j_f + 1) >>> 1
    i_child = isodd(i_f) ? 1 : 2
    j_child = isodd(j_f) ? 1 : 2
    return i_parent, j_parent, i_child, j_child
end
```

Ne pas ajouter de type lourd au debut. Les fonctions ci-dessus suffisent pour
les premiers tests.

## Tests obligatoires - phase A

Dans `test/test_conservative_tree_2d.jl`:

```julia
using Test
using KrakeNN

@testset "Conservative tree 2D" begin
    @testset "D2Q9 directions" begin
        @test d2q9_cx(1) == 0
        @test d2q9_cy(1) == 0
        @test d2q9_cx(2) == 1
        @test d2q9_cy(3) == 1
        @test d2q9_cx(4) == -1
        @test d2q9_cy(5) == -1
    end
end
```

Ajouter ensuite:

```julia
@testset "coalesce preserves per-population sums" begin
    Fc = reshape(collect(1.0:36.0), 2, 2, 9)
    Fp = zeros(Float64, 9)
    coalesce_F_2d!(Fp, Fc)
    for q in 1:9
        @test Fp[q] == sum(Fc[:, :, q])
    end
end
```

Test masse/momentum:

```julia
@testset "coalesce preserves mass and momentum" begin
    Fc = rand(2, 2, 9)
    Fp = zeros(Float64, 9)
    coalesce_F_2d!(Fp, Fc)

    child_mass = sum(Fc)
    child_mx = sum(KrakeNN.d2q9_cx(q) * Fc[ix, iy, q]
                   for ix in 1:2, iy in 1:2, q in 1:9)
    child_my = sum(KrakeNN.d2q9_cy(q) * Fc[ix, iy, q]
                   for ix in 1:2, iy in 1:2, q in 1:9)

    m, mx, my = moments_F(Fp)
    @test isapprox(m, child_mass; atol=1e-14, rtol=0)
    @test isapprox(mx, child_mx; atol=1e-14, rtol=0)
    @test isapprox(my, child_my; atol=1e-14, rtol=0)
end
```

Test explosion:

```julia
@testset "uniform explosion conserves parent" begin
    Fp = rand(9)
    Fc = zeros(Float64, 2, 2, 9)
    explode_uniform_F_2d!(Fc, Fp)

    Fback = zeros(Float64, 9)
    coalesce_F_2d!(Fback, Fc)

    @test isapprox(Fback, Fp; atol=1e-14, rtol=0)
    @test moments_F(Fback) == moments_F(Fp)
end
```

Important: ne PAS tester:

```text
explode(coalesce(children)) == children
```

C'est faux avec explosion uniforme et ce serait un piege.

## Phase B - mapping parent/enfant

Ajouter seulement des tests de mapping, sans streaming:

```julia
@testset "parent-child mapping" begin
    @test conservative_tree_parent_index(1, 1) == (1, 1, 1, 1)
    @test conservative_tree_parent_index(2, 1) == (1, 1, 2, 1)
    @test conservative_tree_parent_index(1, 2) == (1, 1, 1, 2)
    @test conservative_tree_parent_index(2, 2) == (1, 1, 2, 2)
    @test conservative_tree_parent_index(3, 4) == (2, 2, 1, 2)
end
```

Si ces tests echouent, ne pas coder le streaming.

## Phase C - transport sans collision, cas 1D dans D2Q9

Objectif: prouver qu'un paquet oriente qui traverse une interface conserve `F`.

Implementer d'abord un cas vertical minimal:

```text
coarse cell C | refined parent P = [a b]
              |                    [c d]
```

Ne tester que la direction est/ouest au debut:

```text
q = 2 : east
q = 4 : west
```

Fonction a ajouter:

```julia
function split_coarse_to_fine_vertical_F_2d!(Fc_dest, Fq::T, q::Int) where T
    # Conservative uniform split for a packet crossing from coarse into a
    # refined parent through a vertical interface.
    # For phase C, put half into the two fine cells adjacent to the interface.
end
```

Pour une interface verticale coarse -> fine vers l'est:

```text
F_q coarse exits into the west side of the refined parent.
Put F_q/2 into child (1,1,q) and F_q/2 into child (1,2,q).
```

Test attendu:

```julia
Fc = zeros(Float64, 2, 2, 9)
split_coarse_to_fine_vertical_F_2d!(Fc, 3.7, 2)
@test isapprox(sum(Fc[:, :, 2]), 3.7; atol=1e-14, rtol=0)
@test isapprox(sum(Fc), 3.7; atol=1e-14, rtol=0)
```

Fine -> coarse:

```julia
function coalesce_fine_to_coarse_vertical_F(Fc_src, q::Int)
    # Return integrated packet crossing from the two fine interface children
    # into the coarse neighbor.
end
```

Pour fine -> coarse vers l'ouest:

```text
read children (1,1,q) and (1,2,q)
return their sum
```

Ne pas coder diagonales tant que east/west ne ferme pas.

## Phase D - diagonales D2Q9

Apres phase C seulement, ajouter les diagonales:

```text
q=6 NE, q=7 NW, q=8 SW, q=9 SE
```

Regle simple pour prototype:

```text
coarse -> fine:
split sur les enfants touches par la face d'entree.
Les poids doivent sommer exactement a 1.

fine -> coarse:
additionner toutes les contributions fines qui sortent vers la meme cellule
coarse destination.
```

Ne pas chercher l'ordre 2 ici. La cible est conservation exacte.

## Phase E - collision locale, seulement apres transport conservatif

Ajouter une fonction locale optionnelle:

```julia
function collide_BGK_integrated_D2Q9!(Fcell::AbstractVector{T},
                                      volume::T,
                                      omega::T) where T
    # 1. f_i = F_i / volume
    # 2. rho = sum(f)
    # 3. ux = sum(cx_i * f_i) / rho
    # 4. uy = sum(cy_i * f_i) / rho
    # 5. f_i_new = f_i - omega * (f_i - feq_i)
    # 6. F_i_new = f_i_new * volume
end
```

Test obligatoire:

```julia
F0 = rand(9)
F1 = copy(F0)
collide_BGK_integrated_D2Q9!(F1, 1.0, 1.2)
@test isapprox(mass_F(F1), mass_F(F0); atol=1e-14, rtol=0)
@test isapprox(momentum_F(F1)[1], momentum_F(F0)[1]; atol=1e-14, rtol=0)
@test isapprox(momentum_F(F1)[2], momentum_F(F0)[2]; atol=1e-14, rtol=0)
```

Si ce test ne ferme pas, le bug est dans la conversion `F <-> f` ou dans
`feq`.

## Phase F - integration minimale dans Kraken

Ne pas remplacer le raffinement existant.

Ajouter une API experimentalement nommee:

```julia
ConservativeTreePatch2D
create_conservative_tree_patch_2d
```

Mais seulement apres phases A-E.

Proposition de struct:

```julia
struct ConservativeTreePatch2D{T}
    parent_i_range::UnitRange{Int}
    parent_j_range::UnitRange{Int}
    ratio::Int
    fine_F::Array{T,3}     # (Nx_fine, Ny_fine, 9), integrated populations
    coarse_shadow_F::Array{T,3}  # (Nx_parent, Ny_parent, 9), ledger only
end
```

`coarse_shadow_F` est un ledger/agregat, pas un etat actif fluide.

Ne pas appeler cette API depuis `simulation_runner.jl` dans la premiere session.

## Pieges connus

1. Conserver `rho` et `u` separement ne conserve pas le momentum si `rho` varie.
   Il faut conserver `rho*u`, ou mieux les populations integrees `F_i`.

2. Conserver seulement masse + momentum ne suffit pas forcement.
   Conserver les 9 populations orientees `F_i` est plus strict et evite de
   casser les modes non-equilibre.

3. Une cellule mere raffinee ne doit pas etre active fluide.
   Elle est un ledger:

```text
M_i = sum(children_i)
```

4. Ne pas tester la precision physique trop tot.
   L'ordre de precision de l'explosion uniforme est faible pres interface,
   mais ce n'est pas le sujet de la premiere session.

5. Ne pas utiliser de ghost cells dans la phase A-E.
   Si des ghost cells apparaissent, la session repart vers l'ancienne voie.

## Definition of Done pour la premiere session

La session est reussie si:

- `src/refinement/conservative_tree_2d.jl` existe;
- `test/test_conservative_tree_2d.jl` existe;
- `src/KrakeNN.jl` inclut le nouveau fichier;
- `test/runtests.jl` inclut le nouveau test;
- les tests phase A et B passent;
- idealement phase C east/west passe;
- aucune modification de `simulation_runner.jl`;
- aucune modification de la voie `RefinementPatch` existante;
- aucun benchmark fluide ajoute.

Ne pas aller au-dela si les invariants ne ferment pas.
