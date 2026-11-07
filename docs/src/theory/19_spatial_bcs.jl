# # Spatial Boundary Conditions & KrakenExpr
#
# Many practical LBM simulations require boundary conditions that vary
# in space — parabolic inlet profiles, spatially varying pressure outlets,
# time-dependent forcing.  KrakeNN provides a two-layer system for this:
# spatially varying **Zou-He kernels** that accept per-node arrays, and
# the **KrakenExpr** expression engine that compiles user-defined math
# expressions from `.krk` config files into these arrays.
#
# ## Spatially varying Zou-He BCs
#
# The standard Zou-He boundary conditions impose a **uniform** velocity
# or pressure along a wall.  For non-uniform profiles, KrakeNN provides
# spatial variants that accept per-node arrays instead of scalar values.
#
# ### Velocity BCs
#
# The spatial velocity kernels take arrays `ux_arr[i]` and `uy_arr[i]`
# (or `ux_arr[j]`, `uy_arr[j]` for vertical walls) containing the
# prescribed velocity at each boundary node:
# ```math
# u_x(x_i) = \texttt{ux\_arr}[i], \quad
# u_y(x_i) = \texttt{uy\_arr}[i]
# ```
#
# The Zou-He algebra is identical to the uniform case — only the input
# velocity is read from an array rather than a scalar.
#
# ```julia
# apply_zou_he_north_spatial_2d!(f, ux_arr, uy_arr, Nx, Ny)
# apply_zou_he_south_spatial_2d!(f, ux_arr, uy_arr, Nx)
# apply_zou_he_west_spatial_2d!(f, ux_arr, uy_arr, Nx, Ny)
# ```
#
# ### Pressure BCs
#
# Similarly, spatial pressure BCs accept a per-node density array:
# ```math
# \rho(y_j) = \texttt{rho\_arr}[j]
# ```
#
# ```julia
# apply_zou_he_pressure_east_spatial_2d!(f, rho_arr, Nx, Ny)
# ```
#
# ## Common inlet profiles
#
# ### Parabolic (Poiseuille) profile
#
# The most common spatially varying BC is the fully-developed parabolic
# velocity profile at an inlet:
# ```math
# u_x(y) = \frac{4\,U_{\max}}{H^2}\,y\,(H - y), \quad u_y = 0
# ```
#
# where ``H`` is the channel height and ``U_{\max}`` is the centreline
# velocity.  In a `.krk` config file, this is specified as:
#
# ```
# BC west velocity
#   ux = 4 * U_max * y * (Ly - y) / Ly^2
#   uy = 0.0
# ```
#
# ### Time-dependent ramp
#
# To avoid initial transients, a time ramp can be applied:
# ```math
# u_x(y, t) = u_x^{\text{profile}}(y) \cdot
#     \min\!\left(1, \frac{t}{t_{\text{ramp}}}\right)
# ```
#
# ```
# BC west velocity
#   ux = 4 * U_max * y * (Ly - y) / Ly^2 * min(1.0, t / t_ramp)
#   uy = 0.0
# ```
#
# ## KrakenExpr expression engine
#
# The `KrakenExpr` system parses user-written math strings from `.krk`
# config files, validates them against a whitelist, and compiles them
# into Julia functions — all with **sandboxed evaluation** to prevent
# code injection.
#
# ### Parsing pipeline
#
# 1. **Parse**: `Meta.parse(source)` → Julia AST
# 2. **Validate**: walk the AST, reject unsafe constructs (imports,
#    macros, qualified calls, unknown functions)
# 3. **Collect variables**: identify free symbols
# 4. **Substitute constants**: replace `pi`, `e`, and user-defined
#    `Define` variables with numeric values
# 5. **Compile**: generate a Julia function in a sandboxed `Module`
#
# ```julia
# expr = parse_kraken_expr("sin(2*pi*x/Lx) + U*y")
# result = evaluate(expr; x=0.5, y=1.0, Lx=1.0, U=0.1)
# ```
#
# ### Whitelisted functions
#
# Only mathematical functions are allowed in expressions.  The whitelist
# includes:
#
# | Category | Functions |
# |:---|:---|
# | Arithmetic | `+`, `-`, `*`, `/`, `^`, `mod`, `rem`, `div` |
# | Trigonometric | `sin`, `cos`, `tan`, `asin`, `acos`, `atan` |
# | Hyperbolic | `sinh`, `cosh`, `tanh` |
# | Exponential | `exp`, `log`, `log2`, `log10`, `sqrt`, `cbrt` |
# | Rounding | `abs`, `sign`, `floor`, `ceil`, `round` |
# | Comparison | `min`, `max`, `clamp`, `ifelse`, `>`, `<`, `>=`, `<=` |
#
# Any function not in this list triggers a validation error at parse time.
#
# ### Built-in variables
#
# All expressions have access to spatial and geometric variables:
#
# | Variable | Meaning |
# |:---|:---|
# | `x`, `y`, `z` | Node position |
# | `t` | Current time step |
# | `Lx`, `Ly`, `Lz` | Domain length |
# | `Nx`, `Ny`, `Nz` | Grid dimensions |
# | `dx`, `dy`, `dz` | Grid spacing |
#
# Additional user-defined variables can be set via `Define` blocks in
# the `.krk` config file.
#
# ### Security model
#
# The expression engine is designed for **safe evaluation of
# user-provided strings**:
#
# - **AST validation**: rejects `using`, `import`, `module`, `ccall`,
#   `@eval`, and all macro calls
# - **Function whitelist**: only mathematical functions are callable
# - **Sandboxed module**: compiled functions live in a fresh `Module()`
#   with only math functions imported
# - **`invokelatest`**: ensures correct world-age semantics for
#   dynamically compiled functions
#
# ### Time-dependent expressions
#
# Expressions referencing `t` are flagged as time-dependent and are
# re-evaluated at each time step (the boundary arrays are recomputed).
# Static expressions are evaluated once during initialization.
#
# ```julia
# is_time_dependent(expr)  # true if `t` appears in the expression
# is_spatial(expr)         # true if `x`, `y`, or `z` appears
# ```
#
# ## Implementation flow
#
# The full pipeline from `.krk` file to GPU boundary arrays is:
#
# 1. Parse `.krk` file → extract BC specifications with expression strings
# 2. `parse_kraken_expr()` → compile each expression to a `KrakenExpr`
# 3. At initialization: evaluate expressions at each boundary node
#    position → fill GPU arrays `ux_arr`, `uy_arr`, `rho_arr`
# 4. At each time step:
#    - If time-dependent: re-evaluate and update GPU arrays
#    - Call spatial Zou-He kernels with the arrays
#
# This design keeps the GPU kernels simple (no expression evaluation on
# GPU) while allowing arbitrarily complex spatial/temporal profiles.
#
# ## Real source excerpts
#
# The KrakenExpr parser entry point (`src/io/expression.jl`):
#
# @@EXTRACT src/io/expression.jl parse_kraken_expr@@
#
# The west-boundary spatial Zou-He kernel
# (`src/kernels/boundary_spatial_2d.jl`):
#
# @@EXTRACT src/kernels/boundary_spatial_2d.jl apply_zou_he_west_spatial_2d!@@
#
# ## See in action
#
# - [Poiseuille channel](../examples/01_poiseuille_2d.md) — parabolic inlet
#   as a spatial Zou-He velocity BC.
# - [Flow past a cylinder](../examples/06_cylinder_2d.md) — non-uniform
#   inflow profile from a `.krk` expression.
# - [.krk configuration](../examples/10_krk_config.md) — how the expression
#   strings enter the config language.

nothing  # suppress REPL output
