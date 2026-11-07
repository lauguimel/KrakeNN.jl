# Installation

## Prerequisites

### Julia

Install Julia via [juliaup](https://github.com/JuliaLang/juliaup) (recommended):

```bash
# macOS / Linux
curl -fsSL https://install.julialang.org | sh

# Windows
winget install julia -s msstore
```

KrakeNN requires **Julia 1.10** or later.

## Installing KrakeNN

KrakeNN is not yet registered in the General registry. Install it directly
from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/lauguimel/Kraken.jl")
```

Or from the Pkg REPL (press `]`):

```
pkg> add https://github.com/lauguimel/Kraken.jl
```

For a development checkout (clone + activate):

```julia
using Pkg
Pkg.activate("/path/to/KrakeNN")
Pkg.instantiate()
```

## GPU Setup

### Metal (Apple Silicon — Mac M1/M2/M3/M4)

Metal support is included by default. No additional setup is required on macOS with Apple Silicon.

Verify Metal is available:

```julia
using Metal
Metal.versioninfo()
```

### CUDA (NVIDIA GPUs)

Install the CUDA toolkit and the Julia CUDA package:

```julia
using Pkg
Pkg.add("CUDA")
```

CUDA.jl will automatically download the appropriate CUDA toolkit. Requires compute capability ≥ 7.5.

Verify CUDA is available:

```julia
using CUDA
CUDA.versioninfo()
```

## Verifying Installation

```julia
using KrakeNN

# Run a tiny lid-driven cavity as a smoke test (CPU)
config = LBMConfig(D2Q9(); Nx=32, Ny=32, ν=0.1, u_lid=0.1, max_steps=100)
result = run_cavity_2d(config)
println("max |ux| = ", maximum(abs.(result.ux)))
```

Real output from this exact snippet (CPU, Julia 1.12):

```
max |ux| = 0.1000000000000001
```

If the cavity simulation completes without errors and prints a value close
to the lid speed (`0.1`), KrakeNN is correctly installed.

To check that a GPU backend actually loads on your machine, `using CUDA;
CUDA.functional()` or `using Metal; Metal.functional()` (see above) — there
is no `available_backends()` helper in KrakeNN itself. To run a `.krk` case
on that backend, either pass it explicitly
(`run_simulation(setup; backend=CUDABackend())` /
`backend=MetalBackend()`), or add a `Backend cuda` / `Backend metal` line to
the `.krk` file itself (see [Directives](krk/directives.md), section
"Backend"). A non-functional GPU backend raises a clear error naming it
rather than silently running on CPU.

From here, head to [Getting started](getting_started.md) for your first real
simulation, or to the [`.krk` overview](krk/overview.md) to describe a run
declaratively.
