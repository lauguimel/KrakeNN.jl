---
module: bc
path: src/bc/
owner_concern: boundary-condition
status: implemented
last_verified: 2026-06-02
depends_on:
  - lbm
  - io-krk
  - backend
---

# bc — module implication map

The `bc` module owns every **boundary closure** Kraken applies to a D2Q9/D3Q19
distribution: no-slip walls (halfway bounce-back), interpolated bounce-back
(LI-BB / Bouzidi), Zou-He velocity/pressure inflow & outflow, periodic edges,
moving walls, and the zero-gradient (Neumann) outlet. It spans two regimes:
**standalone post-stream kernels** (`src/kernels/boundary_2d.jl`,
`boundary_spatial_2d.jl`, `boundary_3d.jl`) called by the legacy generic runner,
and the **fused / per-face dispatch** path where the BC is woven into the
collide kernel via the `BCSpec2D`/`BCSpec3D` type-dispatch system. As of the
KRK-SHIP-002 S8 refactor the per-face dispatch layer was carved out of
`kernels/boundary_rebuild.jl` into a dedicated `src/bc/` module:

- `src/bc/specs.jl` — the `AbstractBC` tag types (`HalfwayBB`, `InterfaceBC`, `ZouHeVelocity`, `ZouHeTangentialVelocity`, `ZouHePressure`) and the `BCSpec2D`/`BCSpec3D` per-face spec structs.
- `src/bc/rebuild_2d.jl` — `apply_bc_rebuild_2d!` per-face dispatch + the scalar/spatial Zou-He launches for 2D.
- `src/bc/rebuild_3d.jl` — `apply_bc_rebuild_3d!` and the 3D analogues.
- `src/bc/moments.jl` — `_update_bc_moments_2d!` / boundary-row moment recompute kernels.
- `src/bc/handlers.jl` — the runner-facing `BoundaryHandler` struct, `_build_boundary_handlers`, and `_apply_boundary_conditions!`.

The fused cut-link kernels themselves (`li_bb_2d.jl`, `li_bb_2d_v2.jl`,
`li_bb_3d_v2.jl`) remain in `src/kernels/`, as does the body-fitted outflow path
(`_mesh_drag_*` now in `src/kernels/mesh_drag_2d.jl`). High-level inflow/outflow
orchestration lives in `src/simulation_runner.jl` (carved to ~410 LOC in the
S1 refactor), which calls into `src/bc/handlers.jl`.

## Public surface

Standalone post-stream BC kernels (all exported, called as `KrakeNN.apply_*`):

- `apply_bounce_back_walls_2d!(f, Nx, Ny)` — no-slip BB on south/west/east of a
  2D cavity (north left for a velocity lid). `apply_bounce_back_wall_2d!(f, Nx, Ny, side)` does ONE face (`:south/:west/:east/:north`), used by refinement patches.
- `apply_zou_he_north_2d!`, `apply_zou_he_south_2d!`, `apply_zou_he_west_2d!` — Zou-He velocity (lid/Couette/inlet) per face; `apply_zou_he_pressure_east_2d!(f, Nx, Ny; ρ_out)` — Zou-He pressure outlet; `apply_extrapolate_east_2d!(f, Nx, Ny)` — zero-gradient (Neumann) outflow (copies column `Nx-1 → Nx`).
- Per-node spatial variants in `boundary_spatial_2d.jl`: `apply_zou_he_north_spatial_2d!`, `apply_zou_he_south_spatial_2d!`, `apply_zou_he_west_spatial_2d!`, `apply_zou_he_pressure_east_spatial_2d!` (take `ux_arr`/`uy_arr`/`rho_arr` device arrays; skip the two corner nodes `i∈{1,Nx}`).
- 3D analogues exported from `boundary_3d.jl`: `apply_bounce_back_walls_3d!`, `apply_bounce_back_wall_3d!`, `apply_zou_he_{top,bottom,west,east,south,north}_3d!`, `apply_zou_he_pressure_{east,top}_3d!`.

Fused LI-BB / per-face dispatch surface (fused kernels in `li_bb_2d.jl`,
`li_bb_2d_v2.jl`; spec + dispatch + handlers in `src/bc/`):

- `fused_trt_libb_step!(f_out, f_in, ρ, ux, uy, is_solid, q_wall, uw_x, uw_y, Nx, Ny, ν; Λ=3/16)` — single-launch TRT collide + pull-stream + interpolated bounce-back on cut links flagged by `q_wall`. `fused_trt_libb_v2_step!(...)` — same signature, DSL-assembled spec that fixes the double-BC bug (see Failure modes). `fused_trt_libb_v2_step_3d!` — 3D analogue.
- `precompute_q_wall_cylinder(Nx, Ny, …)`, `precompute_q_wall_annulus`, `precompute_q_wall_sphere_3d`, `dq_wall_dR_cylinder` — build the `q_wall[Nx,Ny,9]` cut-fraction array (sentinel 0 = uncut, value in (0,1] = cut link). `wall_velocity_rotating_cylinder`, `wall_velocity_rotating_inner` — per-link moving-wall velocity arrays.
- BC tag types: `AbstractBC` and concrete tags `HalfwayBB` (no-op fallback), `InterfaceBC` (multiblock ghost-owned edge), `ZouHeVelocity(profile, physical_dir)`, `ZouHeTangentialVelocity(profile)` (moving south/north wall), `ZouHePressure(ρ_out, physical_dir)`, plus the runner-local `MeshDragOutflow` (zero-gradient body-fit outlet).
- `BCSpec2D(; west, east, south, north)` / `BCSpec3D(; west, east, south, north, bottom, top)` — per-face spec, all faces default `HalfwayBB`. `apply_bc_rebuild_2d!(f_out, f_in, bcspec, ν, Nx, Ny; sp_field, sm_field, ρ_out, ux_out, uy_out)` / `apply_bc_rebuild_3d!` — dispatch per face, launch the matching kernel, optionally recompute boundary-row moments. `rebuild_inlet_outlet_libb_2d!`, `rebuild_inlet_outlet_libb_3d!` — legacy hard-coded (ZouHe-vel west + ZouHe-pressure east) variant.
- Runner-level: `BoundaryHandler` struct (one per face; holds the compiled `ux_fn/uy_fn/rho_fn`, spatial/time-dep flags, and pre-allocated device arrays) is built by `_build_boundary_handlers` and applied each step by `_apply_boundary_conditions!`, both now in `src/bc/handlers.jl` (driven from `simulation_runner.jl`).

## Reads from

- `lbm` (`src/lattice/` + the fused-kernel helpers `moments_2d`, `feq_2d`,
  `trt_rates`) — D2Q9/D3Q19 opposite-direction pairs (the explicit `(2,4)/(3,5)/(6,8)/(7,9)` index reflections in `bounce_back_walls_2d_kernel!` and the `fp*c` TRT-collided populations), lattice weights `w_q`, `cs²=1/3`, and the TRT magic rate split `s_plus,s_minus = trt_rates(ν; Λ=3/16)`. Consumed read-only.
- `io-krk` (`src/io/`) — the `.krk` `Boundary`/`Region` blocks. The runner reads `setup.boundaries[*].face/.type/.values` and `bc.values[:ux/:uy/:rho].func` to build each `BoundaryHandler`; `block.boundary_tags` (`:inlet/:outlet/:wall/INTERFACE_TAG`) drive `_mesh_drag_bcspec` → `BCSpec2D`.
- `backend` (`KernelAbstractions`) — every kernel calls `KernelAbstractions.get_backend(f)` to pick CPU/CUDA/Metal at launch; spatial-BC arrays are allocated via `KernelAbstractions.zeros(backend, T, face_size)`.
- Geometry-side input (read-only, produced upstream): the `is_solid` mask and the `q_wall`/`uw_link_*` cut-fraction & wall-velocity arrays consumed by the fused LI-BB kernels.

## Writes to

- **Mutates the distribution array `f` (or `f_out`) in place** at boundary/halo
  cells: the standalone `apply_*` kernels overwrite the unknown incoming
  populations on one face; the fused `fused_trt_libb_*_step!` writes the full
  `f_out` interior plus the LI-BB cut-link overwrite; `apply_bc_rebuild_2d!`
  rewrites only the boundary rows/columns of `f_out`.
- **Optionally writes the macroscopic moment fields** `ρ_out, ux_out, uy_out`:
  `apply_bc_rebuild_2d!` calls `_update_bc_moments_2d!` (kernels
  `_recompute_moments_row_2d!` / `_recompute_moments_col_2d!`) when `ρ_out` is
  passed, recomputing density/velocity on the touched boundary line.
- **Solid cells**: in the fused kernel an interior solid cell is set to plain
  bounce-back and its moments forced to `ρ=1, u=0`.
- **No global registry, no files.** BCs touch only the simulation arrays passed
  in. `BoundaryHandler` / `BCSpec2D` are immutable value objects built once
  before the time loop; spatial arrays are filled once (static) or refilled per
  step only for time-dependent BCs (`_fill_bc_arrays!` with `t=step`).

## Backend constraints

- **GPU-safe / KernelAbstractions-clean**: every BC is a `@kernel` launched on
  `get_backend(f)`; no dynamic allocation inside the hot kernels. The per-face
  dispatch in `apply_bc_rebuild_2d!` resolves BC type at Julia method level
  (compiled on first call) — no `eval`, no runtime branch on a symbol inside the
  kernel.
- **Per-step cost is O(edge cells)** for the standalone/per-face kernels
  (negligible vs collide); the fused `fused_trt_libb_*_step!` is the collide
  itself, so the LI-BB overwrite adds only a handful of FLOPs per cut link.
- **Spatial-array allocation is host-side and one-shot** (`_build_boundary_handlers`); time-dependent BCs re-fill via a CPU loop + `copyto!` each step (`_fill_bc_arrays!`), which IS a per-step host round-trip — keep BCs constant when you can.
- **Float32 caveat**: all kernels are `T=eltype(f)`-generic and run under Metal
  F32; the Zou-He density divisions (`/(1±u)`, `/ρ_out`) and the LI-BB `1/(2q_w)`
  branch lose precision near `q_w→0` and at high `u`, compounding the documented
  Metal-F32 R-drift. The local-τ `sp_field/sm_field` path keeps per-cell rates
  in-field for refined/stretched grids.

## Failure modes

This module concentrates Kraken's worst BC rabbit-holes; cite the receipt before
re-deriving:

- **Double-BC over-bounce** — the legacy `fused_trt_libb_step!` applied BB twice
  (SolidSwapBB on solids + post-collision LI-BB on fluids), giving L2 ≈ 2.2 %.
  `fused_trt_libb_v2_step!` (`li_bb_2d_v2.jl`, spec `_TRT_LIBB_V2_SPEC`) is
  **PRE-PHASE ONLY**: substitute the bounced pop before collision, no post
  overwrite → L2 = 0.06 %. Do NOT add a post-phase LI-BB on top of v2.
- **Double Bouzidi two-pass trap (M34 v1)** — reusing a BC-containing spec
  (`_TRT_LIBB_V2_GUO_FIELD_SPEC` with `ApplyLiBBPrePhase`) for a first pass fires
  the BC twice per step → 3/4 NaN on the Aqua matrix. Write a fresh RAW
  (BC-free) spec for any extra pass.
- **Smoke that hides cut-link bugs** — a closed box with `q_wall=0.5` collapses
  Bouzidi-FL to plain halfway-BB algebraically and passes while the cut-link path
  is broken. Always smoke with a cylinder R=4–8 that exercises real `q_w∈(0,1)`.
- **NaN triage fingerprint** — ≥90 % NaN domain = BC over-bounce (wall closure);
  bilateral front-shoulder arcs (θ≈±45°, r−R∈[0,7]) = polymer back-force, NOT a
  BC bug. 1-minute classification before touching this module.
- **Corner ownership / double-application** — the standalone `bounce_back_walls_2d!` deliberately leaves the north wall to Zou-He, and the *spatial* kernels skip `i∈{1,Nx}` (corners) assuming Zou-He/streaming owns them. In a multiblock interface-wall corner this skip caused a 1.4e-3 bit-exact error/step → absurd Cd (multiblock corner BC bug, ebf0867); `InterfaceBC` widens the loop bounds (`j_lo/j_hi`, `i_lo/i_hi`) to reclaim those corners.
- **Multiblock N-S interface blocked by BB** — kernels that BB at every domain
  edge wrongly block N-S interface edges (E-W survives only because Zou-He
  overwrites); the fused/refined path must skip BB on `InterfaceBC` edges.
- **`physical_dir = :auto`** on `ZouHeVelocity`/`ZouHePressure` must be resolved
  from the geometry tag (`_physical_normal_from_tag`) before launch — a wrong
  normal silently injects momentum on the wrong axis.
- **`_apply_wall_bc!` is intentionally a no-op** — pure walls are handled inside
  the streaming kernel's PullHalfwayBB fallback, NOT here. Editing it to "fix" a
  wall bug is the classic edited-dead-code anti-pattern.

## Touch order

For a suspected BC bug (wrong wall profile, spurious momentum, drag off, NaN at a
face), inspect in this order:

1. `src/bc/handlers.jl` (driven from `src/simulation_runner.jl`) — the
   orchestration: `_build_boundary_handlers` / `_apply_boundary_conditions!`
   (which face gets which `apply_*`), the `_mesh_drag_bc` → `BCSpec2D` mapping
   (`src/kernels/mesh_drag_2d.jl`), and `_apply_wall_bc!` (no-op — confirm the
   wall is really streaming-handled). 80 % of "BC not applied / wrong face" bugs
   are a wiring issue here, not in a kernel.
2. `src/bc/specs.jl` + `src/bc/rebuild_2d.jl` — the `AbstractBC`/`BCSpec2D` type
   defs (specs) and `apply_bc_rebuild_2d!` per-face dispatch (rebuild_2d); check
   `InterfaceBC` loop bounds for corner/interface bugs and `_update_bc_moments_2d!`
   (`src/bc/moments.jl`) for a wrong boundary ρ/u.
3. `src/kernels/boundary_2d.jl` — the scalar Zou-He closures and the BB index
   reflections; verify the opposite-direction pairing and the `±ρ·u` momentum
   terms for a uniform-BC bug.
4. `src/kernels/boundary_spatial_2d.jl` — per-node profile bugs; check the
   `i∈{1,Nx}` corner skip and the `ux_arr/uy_arr` fill (`_fill_bc_arrays!`).
5. `src/kernels/li_bb_2d_v2.jl` then `li_bb_2d.jl` — for fused-kernel / cut-link
   bugs: v2 spec (pre-phase-only) first, then `_libb_branch` (the q_w≤½ vs >½
   formula) and `precompute_q_wall_cylinder` (cut-fraction & sentinel logic).
6. `src/kernels/boundary_3d.jl` / `li_bb_3d_v2.jl` — only for the 3D analogues;
   the 2D mechanism almost always reproduces the bug more cheaply first.
