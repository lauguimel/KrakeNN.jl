# Doc helpers (Phase 4.1A)

Internal helpers used by the KrakeNN "living documentation" build.
These files are **not** served as public doc pages; they are `include`d
from `docs/make.jl` so Literate pages built in Phases 4.2–4.6 can embed
real source code from `src/`.

## Files

- `source_extract.jl` — `extract_function`, `extract_struct`,
  `include_function`. JuliaSyntax-based extraction of top-level
  function/struct source (signature + docstring + body) from any
  `.jl` file. Returns markdown-ready fenced code blocks.

- `krk_download.jl` — `krk_download(path)`. Copies a `.krk` config file
  into `docs/build/assets/` and returns a markdown download badge to
  embed inside a Literate page.

- `api_extract.jl` — `extract_exports`, `categorize_exports`,
  `api_page_data`. Parses `src/KrakeNN.jl`, groups exported symbols into
  categories (collision, streaming, boundary, refinement, io,
  postprocess, drivers, view, lattice, rheology, multiphase, …), and
  resolves each symbol to its on-disk source excerpt. Out-of-scope
  categories (`multiphase`, `rheology`, `viscoelastic`, `species`) are
  filtered by default for the v0.1.0 API pages.

- `_test_helpers.jl` — Literate proof-of-concept page that exercises
  all three helpers at build time. Kept as a hidden page; remove or
  relocate once Phase 4.6 is landed.

## Usage from Phases 4.2–4.6

```julia
# In a Literate page:
src_md = include_function("src/drivers/basic.jl", :run_cavity_2d)
badge  = krk_download("examples/cavity.krk")
data   = api_page_data("src/KrakeNN.jl")
```

The helpers use JuliaSyntax (not regex) so multi-line signatures,
`where` clauses, nested blocks and docstrings are handled robustly.
