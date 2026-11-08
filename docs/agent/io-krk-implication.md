---
module: io-krk
path: src/io/krk/
owner_concern: krk-parsing
status: implemented
last_verified: 2026-06-02
depends_on:
  - geometry
  - physics-viscoelastic
---

# io-krk — module implication map

The `io-krk` module is the **single owner** of the `.krk` declarative simulation
DSL: tokenizing the text, evaluating the sandboxed expression language, and
lowering it into a plain-old-data `SimulationSetup` struct that the runner
consumes. As of the KRK-SHIP-002 S2 refactor the formerly-monolithic
`kraken_parser.jl` (~2010 LOC) was fissioned into `src/io/krk/`; the live
source files are:

- `src/io/expression.jl` — the sandboxed math mini-language (`KrakenExpr`).
- `src/io/krk/parser.jl` — the dispatch loop and tokenizer (`_parse_kraken_internal_single`, `_preprocess_lines`, `_strip_comment`, `_first_word`, `_resolve_setup_paths`, `_copy_setup_with_mesh`).
- `src/io/krk/directives.jl` — the individual `_parse_<directive>` extractors (`_parse_domain`, `_parse_physics`, `_parse_boundary`, `_parse_refine`, `_parse_output`, `_parse_mesh`, `_parse_setup`, `_apply_setup_helpers!`, `_probe_U_ref`).
- `src/io/krk/rheology.jl` — `build_rheology_model` (model `Symbol` → constructor + thermal coupling).
- `src/io/krk/setup_lbm.jl` — the offline LBM advisor (`lbm_params`, `lbm_params_table`) and the `SimulationSetup` lowering.
- `src/io/krk/diagnostics.jl` — `sanity_check` + the seven `_check_*!` helpers + `_emit_issues` and `_print_parameter_summary`.
- `src/io/krk/units_bridge.jl` — the physical-unit ↔ lattice-unit bridge for `Setup`-derived parameters.

All are included at package top level (`Kraken.jl` lines 104–110) and export
their public names directly into `Kraken` — there is no `KrakeNN.IO` submodule.

## Public surface

Symbols below are exported into `Kraken` (re-exported, not namespaced).

- `load_kraken(filename; kwargs...) -> SimulationSetup` — read a `.krk` file, parse it, and resolve relative mesh paths against the file's directory. Single-setup; throws if a `Sweep` is present.
- `parse_kraken(text; kwargs...) -> SimulationSetup` — parse `.krk` text (no file I/O). `kwargs` override `Define` defaults / `Physics` params / `Domain` dims / `max_steps` for parametric studies.
- `parse_kraken_sweep(text; kwargs...) -> Vector{SimulationSetup}` / `load_kraken_sweep(filename; kwargs...)` — expand zero-or-more `Sweep p = [a,b,c]` directives into one setup per Cartesian-product combination.
- `parse_kraken_expr(source, user_vars=Dict()) -> KrakenExpr` — parse + AST-validate + compile a math string into a callable. `evaluate(ke; kwargs...) -> Float64` runs it (via `Base.invokelatest`). `has_variable`, `is_time_dependent`, `is_spatial` introspect the variable set.
- `build_rheology_model(rs::RheologySetup; FT=Float64) -> AbstractRheology` — instantiate a concrete rheology (Newtonian / PowerLaw / CarreauYasuda / Cross / Bingham / HerschelBulkley / OldroydB / FENEP / Saramito) plus its thermal coupling from a parsed `RheologySetup`.
- `sanity_check(setup; verbose=true) -> Vector{SanityIssue}` — run the 7 check families (relaxation, compressibility, resolution, thermal, two-phase, rheology, refinement); `@warn` soft issues, `throw` on `:error`. `sanity_check_sweep(setups; verbose)` is the non-throwing batch variant.
- `lbm_params(; Re, N, U_ref=0.01, L_ref=N) -> LBMParams` (also `lbm_params(setup)`) — the offline LBM advisor: derives ν/τ/ω/Ma, classifies the regime, and recommends N / U_ref. `lbm_params_table(; Re, N_range, U_ref)` prints a comparison table.
- Public POD types (all exported): `SimulationSetup`, `DomainSetup`, `PhysicsSetup`, `MeshSetup`, `GeometryRegion`, `BoundarySetup`, `RheologySetup`, `InitialSetup`, `OutputSetup`, `DiagnosticsSetup`, `STLSource`, `RefineSetup`, `RefineCriterionSetup`, plus `KrakenExpr`, `LBMParams`, `SanityIssue`.
- DSL directives recognized by the parser (de-facto public API surface for `.krk` authors): `Simulation`, `Domain`, `Physics`, `Define`, `Obstacle`, `Fluid`, `Boundary`, `Refine`, `Initial`, `Velocity`, `Mesh`, `Module`, `Run`, `Output`, `Diagnostics`, `Rheology`, `Setup`, `Preset`, `Sweep`.

## Reads from

- `geometry` (`src/io/stl_reader.jl`, `stl_libb.jl`, `voxelizer.jl`) — `STLSource` records an STL file + transform; `_parse_stl_params` only parses the directive, the actual STL load/voxelize happens downstream in geometry. The `Mesh gmsh(...)` directive (`MeshSetup`) likewise just records the path for the multi-block mesh loader.
- `physics-viscoelastic` (and the Newtonian/GNF rheology types) — `build_rheology_model` constructs `Newtonian`, `PowerLaw`, `OldroydB`, `FENEP`, `Saramito`, … plus `ArrheniusCoupling`/`WLFCoupling`/`IsothermalCoupling` and `StressFormulation`/`LogConfFormulation`, all owned by the rheology module; this file only maps `Symbol` → constructor.
- The module reads NO live LBM/backend state. The LBM advisor math (`cs = 1/√3`, `τ = 3ν+0.5`) is hard-coded locally, not pulled from a lattice module — it is duplicated here intentionally so the advisor runs before any simulation object exists.

## Writes to

- **Returns** a freshly built `SimulationSetup` (immutable struct of POD + `KrakenExpr` closures). `load_kraken`/`_resolve_setup_paths` rebuild the struct with an absolute mesh path via `_copy_setup_with_mesh`.
- **Mutates only locals during parsing** — `physics_params`, `body_force`, `user_vars`, `boundaries`, etc., are accumulated in `_parse_kraken_internal_single` and `_apply_setup_helpers!` (which back-fills `:nu`/`:alpha`/`:gbeta_DT`/`:Re`/`:Ra`/`:Pr` from `Setup reynolds/rayleigh`). No global/module registry is mutated.
- **Compiles fresh anonymous `Module()`s** — `_compile_expr` evaluates each expression in a throwaway sandbox `Module` via `Core.eval`. This is the one durable side effect: each `KrakenExpr` carries a closure compiled into a one-shot module.
- **Side effects / logging**: `sanity_check` emits `@warn`/`@error`/`@info` and `_print_parameter_summary` logs a parameter banner (`verbose=true`). `parse_kraken` calls `sanity_check(...; verbose=false)` so parse-time warnings surface but the summary is deferred to run time. No files are written by this module.

## Backend constraints

- **Parse-time / host-only — GPU-irrelevant.** Everything runs once on the CPU before the time loop. `KrakenExpr` closures are evaluated on the host (init/BC field setup, `_probe_U_ref`); they are NOT called inside GPU kernels and `evaluate` goes through `Base.invokelatest`, which is incompatible with a GPU launch anyway.
- **`Base.invokelatest` is mandatory**, not optional: expression functions are built by `Core.eval` at parse time (world-age newer than the caller), so a direct call would hit a world-age error. This is the single hardest constraint — see [KRK design decisions / invokelatest required].
- **No per-step allocation in the hot loop** because the module never runs in the hot loop. Expression compilation allocates a `Module` per expression (parse-time cost only).
- **`Float32` caveat is advisory, not enforced here**: the LBM advisor and `_check_compressibility!` *warn* when `U_ref < 1e-5`/`1e-6` (Float32 round-off floor) but `io-krk` does not pick `T` — it parses everything as `Float64` (`DomainSetup`, `PhysicsSetup` are `Float64`-typed); the actual `T` is chosen downstream.

## Failure modes

This module is the front door for human-authored configs, so most footguns are
malformed-input rabbit-holes. Receipts:

- **Brace-depth tokenizer is line-oriented and string-naive** — `_preprocess_lines` counts `{`/`}` to join multi-line blocks and `_strip_comment` strips everything after the first `#`. A `#` or unbalanced brace *inside a string literal* (e.g. an STL path) will mis-tokenize; an unclosed brace throws `ArgumentError("Unclosed brace in .krk file")`. Keep one block per `{ ... }`.
- **`Define` is `Float64`-only** — `_parse_define` does `parse(Float64, ...)`, so a `Define` RHS that is itself an expression (`Define A = 2*pi`) throws; only `Domain`/`Physics`/`Setup`/`Rheology` values go through the expression evaluator. Mismatch surfaces as a bare `ArgumentError` from `parse`.
- **`Setup reynolds`/`rayleigh` conflicts with explicit `Physics nu`** — `_apply_setup_helpers!` throws `ArgumentError("Setup reynolds conflicts with Physics nu ...")` if both are set; the planner-owns-derived-ν guardrail in DSL form (cf. the units-module `:planner_override` philosophy). `U_ref` is probed from a velocity BC (`_probe_U_ref`) or defaults to 0.1 — a silently-wrong `U_ref` yields a silently-wrong ν.
- **Acoustic-vs-diffusive scaling is NOT handled here** — the refinement check (`_check_refinement!`) assumes Ma is preserved across levels (acoustic scaling) and only rescales τ via Filippova-Hanel. The M59–M61 acoustic U-shape and the diffusive-scaling rule live in the units module / sweep driver, not in `.krk` parsing; do not expect this file to auto-select scaling.
- **Sanity `:error` aborts the parse** — `parse_kraken` → `sanity_check(verbose=false)` → `_emit_issues` throws `ErrorException` on any `:error` (τ<0.5, τ>100, U_ref>0.3, thermal τ<0.5). A config that "won't even parse" is often a hard sanity failure, not a syntax error — read the `[category]` tag.
- **Expression whitelist rejects unknown symbols/functions** — `validate_ast!` throws on non-whitelisted calls (anything outside `EXPR_WHITELIST`), qualified calls (`Mod.f`), `using`/`import`/`ccall`/macros, and any symbol not in `EXPR_BUILTIN_VARS ∪ user_vars ∪ constants`. A typo'd variable in an `Initial`/`Boundary` expression fails loudly with the available-variable list — this is the sandbox, not a bug.
- **Sweep grid is a full Cartesian product** — `_parse_kraken_internal` expands every `Sweep` combination; N sweeps of length k each produce kᴺ setups. No de-duplication, no cap.
- **`Float64`-only `Define`/`Domain` rounding** — `Domain N = ...` is `round(Int, ...)`; a fractional grid expression silently rounds.
- **Unknown directives / BC types / faces get Levenshtein "did you mean?"** — `_suggest_name` (distance ≤ max(2, len÷3)); a 2D config using `top`/`bottom` faces is rejected post-parse by `_validate_faces_vs_lattice`, not at the `Boundary` line.

## Touch order

For a `.krk` parse bug (wrong value, spurious/missing error, expression glitch),
inspect in this order:

1. `src/io/krk/parser.jl` → `_parse_kraken_internal_single` — the dispatch loop over directives; confirm the keyword is reaching the right `_parse_*` and that the first/second-pass `Define`/`Module` pre-scans ran. Most "directive ignored" bugs are here.
2. `src/io/krk/directives.jl` → the specific `_parse_<directive>` (`_parse_domain`, `_parse_physics`, `_parse_boundary`, `_parse_refine`, `_parse_output`, `_parse_mesh`, `_parse_setup`) — regex-level extraction bugs (a value not matching, a missing capture group).
3. `src/io/krk/parser.jl` → `_preprocess_lines` / `_strip_comment` / `_first_word` — for "the whole line is wrong" / tokenization bugs (brace joining, comment stripping, multi-line blocks).
4. `src/io/expression.jl` → `parse_kraken_expr` / `validate_ast!` / `_substitute_constants` / `_compile_expr` — for "expression rejected" or "expression evaluates wrong"; check the whitelist and that user vars made it into `all_allowed`.
5. `src/io/krk/directives.jl` → `_apply_setup_helpers!` / `_probe_U_ref` (and `src/io/krk/units_bridge.jl`) — for wrong auto-derived `ν`/`alpha`/`gbeta_DT` from `Setup reynolds`/`rayleigh` and the U_ref probe.
6. `src/io/krk/diagnostics.jl` → `sanity_check` + the seven `_check_*!` helpers + `_emit_issues` — for a spurious/missing/throwing sanity issue (read the `:category`).
7. `src/io/krk/rheology.jl` → `build_rheology_model` — only for `Rheology`-sourced bugs (model symbol → constructor, thermal coupling selection).
8. `src/io/krk/parser.jl` → `_resolve_setup_paths` / `_copy_setup_with_mesh` — for relative-path / mesh-source resolution (file-vs-text entry difference).
