---
module: solve-poisson-embedded-fvfd
path: src/solve/poisson_embedded_fvfd.jl
owner_concern: elliptic-solve
status: implemented
last_verified: 2026-06-11
depends_on:
  - solve-poisson-embedded
---

# solve-poisson-embedded-fvfd — module implication map

The thin BRIDGE between the FVFD geometry layer's per-cell embedded-boundary
convention and the aperture-array convention of `solve-poisson-embedded`. It owns
exactly one concern: the fraction-layout translation (per-cell
`west/east/south/north_fraction` -> staggered `face_frac_x/face_frac_y` +
`vol_frac`), with validation. **Registered in `src/KrakeNN.jl`**
(`assemble_poisson_embedded_from_fvfd` and `fractions_from_fvfd` exported).

## Public surface

- `fractions_from_fvfd(eb) -> (vol_frac, face_frac_x, face_frac_y)` — convert a
  duck-typed FVFD embedded-boundary object (`cell_fraction`, `west_fraction`,
  `east_fraction`, `south_fraction`, `north_fraction`, all `(Nx, Ny)`).
  Tolerant clamp into `[0,1]` (`sqrt(eps)` via `_FVFD_POISSON_FRACTION_TOL`),
  and a shared-face consistency check
  (`east_fraction[i,j] == west_fraction[i+1,j]`, same for north/south).
- `assemble_poisson_embedded_from_fvfd(eb, f; outer_bc, embedded_bc,
  outer_dirichlet, embedded_dirichlet) -> (A, b)` — convert + call
  `assemble_poisson_embedded`. Requires a SQUARE grid (`Nx == Ny`).

## Reads from

`solve-poisson-embedded` (`src/solve/poisson_embedded.jl`, included under an
`isdefined(:assemble_poisson_embedded)` guard — transitively `solve-poisson` and
`solve-linear`). The `eb` argument is duck-typed: any object with the five
fraction fields works (the FVFD embedded-boundary structs in `src/fvfd/` —
which has no implication map yet — but also plain NamedTuples in tests).

## Writes to

Nothing persistent. Allocates and returns the three fraction arrays / `(A, b)`;
`eb` is never mutated. Validation throws `DimensionMismatch` / `ArgumentError`
instead of fixing inputs silently (except the eps-level clamp).

## Backend constraints

CPU-only host translation + assembly (O(N²)). The fraction conversion is cheap,
geometry-only work meant to run ONCE per geometry — keep it out of iteration
loops. No KA kernels, no device arrays.

## Failure modes

- **Shared-face mismatch throws**: if the FVFD object's `east_fraction[i,j]`
  and `west_fraction[i+1,j]` differ beyond `sqrt(eps)`, conversion aborts. This
  is the designed tripwire for inconsistent cut-cell geometry — do NOT loosen
  the tolerance to "make it pass"; fix the geometry producer.
- **Boundary semantics**: `west_fraction[1,j]` / `east_fraction[Nx,j]` (and the
  y analogues) BECOME the domain-box face apertures `face_frac_x[1,j]` /
  `face_frac_x[Nx+1,j]` — there is no separate outer-boundary input. An FVFD
  object that encodes box walls differently silently changes the outer BC.
- **Square-grid restriction**: `assemble_poisson_embedded_from_fvfd` throws on
  `Nx != Ny` (the downstream assembly is `N`-square). Rectangular support means
  touching `solve-poisson-embedded` first, not relaxing the check here.
- Receipt: `test/analytical/poisson_embedded_fvfd_mms.jl` (FVFD-built fractions
  reproduce the hand-built tilted half-plane assembly; ~2nd-order MMS); manual
  driver `test/scratch/poisson_embedded_fvfd_driver.jl`.

## Touch order

1. `src/solve/poisson_embedded_fvfd.jl` — the translation + checks (look here
   for any "fractions look transposed/shifted" symptom).
2. `test/analytical/poisson_embedded_fvfd_mms.jl` — the parity + MMS gates.
3. `src/solve/poisson_embedded.jl` — if the failure is past the translation
   (assembly itself).
4. The FVFD geometry producer in `src/fvfd/` (e.g. `specs.jl`,
   `lowering_2d.jl`) — if the input fractions themselves are wrong.
