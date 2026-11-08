---
module: solve-poisson-embedded
path: src/solve/poisson_embedded.jl
owner_concern: elliptic-solve
status: implemented
last_verified: 2026-06-11
depends_on:
  - solve-poisson
---

# solve-poisson-embedded — module implication map

Cut-cell (embedded-boundary) Poisson assembly on the unit square: the regular
`N x N` grid is kept, geometry enters ONLY through aperture fractions
(`face_frac_x :: (N+1,N)`, `face_frac_y :: (N,N+1)`, `vol_frac :: (N,N)`).
Fully solid cells stay in the system as identity rows (`b=0`) so the global `N²`
indexing of `solve-poisson` is preserved. **Registered in
`src/KrakeNN.jl`** (after `poisson.jl`; `assemble_poisson_embedded` and
`solve_poisson_embedded` exported). Include guards keep the file
standalone-include-able.

## Public surface

- `assemble_poisson_embedded(N, face_frac_x, face_frac_y, vol_frac, f;
  outer_bc=:neumann, embedded_bc=:neumann, outer_dirichlet, embedded_dirichlet)
  -> (A, b)` — the core assembly. Interior fluid-fluid face: symmetric `α/h²`
  conductance. `:dirichlet` outer walls: `2α/h²` half-spacing terms; embedded
  `:dirichlet`: the BLOCKED part `β = 1-α` of each cut-cell face becomes a wall
  (`2β/h²` diag + RHS).
- `solve_poisson_embedded(N, ...; kwargs...)` — assemble + `solve_poisson`
  (only safe when a Dirichlet face makes the operator non-singular).
- `first_fluid_dof(vol_frac, N)` — pick the pin DOF for the singular all-Neumann
  case.
- Geometry generator for tests: `tilted_half_plane_fractions(N; normal, point)`
  (exact polygon clipping; also exposes `_polygon_area`,
  `_clip_polygon_half_plane`, `_segment_fraction_in_half_plane` as de-facto
  internals).
- Fluid-aware MMS metrics: `fluid_l2_error` (vol-weighted), `fluid_row_sum_max`
  (conservation check), `fluid_constant_deviation`.
- `EMBEDDED_POISSON_BCS = (:neumann, :dirichlet)` — the accepted BC symbols.

## Reads from

`solve-poisson` (`src/solve/poisson.jl`, included at the top under an
`isdefined(:linear_index)` guard): indexing helpers (`linear_index`,
`cell_center`, `cell_coordinates`), `_push_entry!`, `_check_grid_size`, and
`solve_poisson` itself (hence transitively the `solve-linear` seam). Only
LinearAlgebra + SparseArrays beyond that. Fraction arrays are caller-provided
(duck-typed `AbstractMatrix`-likes indexed `[i,j]`).

## Writes to

Nothing persistent. Assembly returns fresh `(A, b)`; helpers return scalars or
new arrays. Fraction inputs are validated (`[0,1]` bounds via
`_check_fraction_value`, size via `_check_fraction_arrays`) but never mutated.

## Backend constraints

CPU-only host assembly (O(N²) loops, `SparseMatrixCSC`). Not KA, no device
arrays — the GPU story for embedded operators is the matrix-free apply in
`src/fvfd/operators_2d_grad_div_laplacian.jl`
(`gdl_laplacian_apply_embedded_2d!` reproduces these matrix rows kernel-side,
WITHOUT the cell-fraction division). Keep both in sync when touching the flux
form.

## Failure modes

- **Singular all-Neumann default**: with `outer_bc=embedded_bc=:neumann` the
  fluid block is singular — `solve_poisson_embedded` throws. Pin via
  `first_fluid_dof` + `pin_reference_dof` and call `solve_poisson`, as
  `test/analytical/poisson_embedded_mms.jl` does.
- **Fraction-array convention**: `face_frac_x[i,j]` is the face at `x=(i-1)h`
  (i=1 and N+1 are the box faces). Off-by-one against the FVFD per-cell
  west/east convention is the classic bug — that translation lives ONLY in
  `solve-poisson-embedded-fvfd` (`fractions_from_fvfd`); don't re-derive it.
- **Strict `[0,1]` bounds**: raw `Float64` fraction arithmetic can exceed the
  bounds by eps and `_check_fraction_value` throws (no tolerance HERE; the
  tolerant clamp is in the FVFD bridge). Feed exact or pre-clamped fractions.
- **Conservation regression**: fluid row sums must vanish in the all-Neumann
  case (`fluid_row_sum_max` ~ 1e-10 gate). A nonzero row sum after an edit means
  the symmetric-face/diagonal bookkeeping (`_add_symmetric_face!` vs `diag`)
  was broken. Receipt: `test/analytical/poisson_embedded_mms.jl`.
- Regular fractions (all ones) must reproduce `solve-poisson` exactly — the
  rung-compatibility gate in the same testset.

## Touch order

1. `src/solve/poisson_embedded.jl` — assembly (`assemble_poisson_embedded` +
   `_add_*_faces!` helpers) for any operator/BC/fraction question.
2. `test/analytical/poisson_embedded_mms.jl` — gates: regular-fraction parity,
   tilted half-plane ~2nd-order, conservation, constant-mode.
3. `src/solve/poisson_embedded_fvfd.jl` — if the bug smells like a fraction
   convention/translation issue from FVFD inputs.
4. `src/fvfd/operators_2d_grad_div_laplacian.jl` — if assembled vs matrix-free
   embedded operators disagree.
5. `test/scratch/poisson_embedded_driver.jl` — manual driver.
