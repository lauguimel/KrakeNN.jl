```@meta
EditURL = "18_grid_refinement.jl"
```

# Patch-Based Grid Refinement

!!! warning "Under development"
    Grid refinement (patch and conservative AMR) is under development and not validated; it is not part of the supported feature set. This page is unlisted and kept for reference only; see [Limitations](10_limitations.md).

Uniform LBM grids are wasteful when the flow contains localized features
(boundary layers, interfaces, wakes) that require high resolution.
Kraken implements **patch-based static grid refinement** where
rectangular sub-domains at higher resolution are nested within a coarser
base grid.  Each patch is a self-contained uniform LBM grid that reuses
existing stream/collide/BC kernels unchanged.

**References**:
Filippova & Hänel (1998) [filippova1998grid](@cite filippova1998grid),
Dupuis & Chopard (2003) [dupuis2003theory](@cite dupuis2003theory).

## Refinement structure

A `RefinedDomain` consists of:
- A **base grid** at level 0 with spacing ``\Delta x_0``
- One or more **refinement patches** at levels 1, 2, ... with spacing
  ``\Delta x_\ell = \Delta x_0 / r^\ell`` where ``r`` is the
  refinement ratio (typically 2)

Patches can be **nested**: a level-2 patch is contained within a
level-1 patch.  Ghost layers (width 2 for D2Q9) surround each patch
for the streaming stencil.

```julia
patch = create_patch("wake", 1, 2, (x_min, y_min, x_max, y_max),
                      base_Nx, base_Ny, base_dx, base_omega, Float64)
domain = create_refined_domain(base_Nx, base_Ny, base_dx, base_omega, [patch])
```

## Filippova-Hänel rescaling

The key challenge in LBM grid refinement is maintaining **physical
consistency** across resolution levels.  Since the relaxation rate
``\omega`` encodes the viscosity in lattice units, it must be rescaled
to preserve the same physical viscosity ``\nu`` at each level:
```math
\tau_\text{fine} = r\,(\tau_\text{coarse} - \tfrac{1}{2}) + \tfrac{1}{2}
```
```math
\omega_\text{fine} = \frac{1}{\tau_\text{fine}}
```

The non-equilibrium part of the distributions must also be rescaled
when transferring between levels.  The **Filippova-Hänel** rescaling
factor for coarse-to-fine transfer is:
```math
\alpha_{c \to f} = \frac{\tau_f - 1/2}{\tau_c - 1/2}
```

and the inverse for fine-to-coarse:
```math
\alpha_{f \to c} = \frac{\tau_c - 1/2}{\tau_f - 1/2}
```

```julia
rescaled_omega(omega_parent, ratio)           # → omega_fine
rescaling_factor_c2f(omega_c, omega_f, ratio) # → α_c→f
rescaling_factor_f2c(omega_c, omega_f, ratio) # → α_f→c
```

## Prolongation (coarse → fine)

Ghost cells of the fine patch are filled from coarse-grid data using:

1. **Bilinear interpolation** of macroscopic fields (``\rho``,
   ``\mathbf{u}``) from the coarse grid to the fine ghost position
2. Compute fine equilibrium ``f_\alpha^{\mathrm{eq}}(\rho_f, \mathbf{u}_f)``
3. Interpolate coarse non-equilibrium
   ``f_\alpha^{\mathrm{neq}} = f_\alpha - f_\alpha^{\mathrm{eq}}``
   at each bilinear stencil node
4. Assemble: ``f_\alpha^{\mathrm{fine}} = f_\alpha^{\mathrm{eq}}
   + \alpha_{c \to f}\,f_\alpha^{\mathrm{neq,interp}}``

The kernel operates over the full fine grid but only writes to ghost
cells (interior is untouched):

```julia
prolongate_f_rescaled_2d!(f_fine, f_coarse, rho_c, ux_c, uy_c, ...)
```

## Restriction (fine → coarse)

After the fine patch has been advanced, its interior results are
**restricted** back to the coarse overlap region:

1. **Block-average** the ``r \times r`` fine cells covering each
   coarse cell to get averaged macroscopic fields
2. Block-average fine non-equilibrium distributions
3. Assemble: ``f_\alpha^{\mathrm{coarse}} = f_\alpha^{\mathrm{eq}}(\bar\rho, \bar{\mathbf{u}})
   + \alpha_{f \to c}\,\overline{f_\alpha^{\mathrm{neq}}}``

```julia
restrict_f_rescaled_2d!(f_coarse, rho_c, ux_c, uy_c,
                         f_fine, rho_f, ux_f, uy_f, ...)
```

## Temporal sub-cycling

Since ``\Delta t \propto \Delta x^2 / \nu`` in diffusive scaling (or
``\Delta t \propto \Delta x`` in acoustic scaling), the fine grid
requires ``r`` sub-steps per coarse step.  The algorithm per coarse
time step is:

1. Save coarse state at time ``n`` (for temporal interpolation)
2. Advance coarse grid one step
3. For each sub-step ``s = 1, \ldots, r``:
   - Compute temporal interpolation fraction
     ``\alpha_t = (s-1)/r``
   - Linearly interpolate coarse ghost data between time ``n`` and
     ``n+1``: ``\mathbf{q}(t) = (1 - \alpha_t)\,\mathbf{q}^n
     + \alpha_t\,\mathbf{q}^{n+1}``
   - Fill fine ghost layers (prolongation with rescaling)
   - Advance fine patch: stream → BC → collide → macro
4. Restrict fine interior → coarse overlap

```julia
advance_refined_step!(domain, f_in, f_out, rho, ux, uy, is_solid;
                       stream_fn, collide_fn, macro_fn, bc_base_fn)
```

## Two-phase refinement

For multiphase flows, the VOF field is also advected on the fine grid,
providing higher-resolution curvature estimates.  The algorithm extends
the single-phase sub-cycling with additional VOF prolongation and
restriction steps:

- Prolongate ``C`` coarse → fine (bilinear)
- Advect VOF on fine grid during each sub-step
- Compute curvature on fine grid (improved accuracy)
- Restrict fine VOF → coarse (block average)

```julia
advance_twophase_refined_step!(domain, f_in, f_out, rho, ux, uy,
                                is_solid, C, C_new, nx_n, ny_n, κ,
                                Fx_st, Fy_st, σ, ρ_l, ρ_g, ν_l, ν_g,
                                patch_vof; stream_fn, bc_base_fn)
```

## Practical considerations

!!! note "Static patches"
    Kraken V1 uses **static** refinement: patches are defined before
    the simulation and do not move.  Adaptive mesh refinement (AMR)
    with dynamic patch creation is planned for V2.

!!! tip "Ghost width"
    The default ghost width of 2 cells is sufficient for D2Q9/D3Q19
    streaming stencils.  Wider stencils (e.g. for higher-order
    interpolation) may require increasing `n_ghost`.

## Real source excerpts

The Filippova–Hänel rescaling utility (`src/refinement/refinement.jl`):

```julia
"""
    rescaled_omega(omega_parent, ratio) -> Float64

Compute the relaxation parameter at the fine level to preserve physical
viscosity across refinement levels.

    tau_fine = ratio * (tau_parent - 0.5) + 0.5
    omega_fine = 1 / tau_fine
"""
function rescaled_omega(omega_parent::Real, ratio::Int)
    tau_parent = 1.0 / omega_parent
    tau_fine = ratio * (tau_parent - 0.5) + 0.5
    return 1.0 / tau_fine
end
```

Prolongation kernel (`src/kernels/refinement_exchange_2d.jl`):

```julia
"""
    prolongate_f_rescaled_2d!(f_fine, f_coarse, rho_c, ux_c, uy_c, patch)

Fill ghost cells of fine patch from coarse grid data with Filippova-Hanel
non-equilibrium rescaling.
"""
function prolongate_f_rescaled_2d!(f_fine, f_coarse, rho_c, ux_c, uy_c,
                                   ratio::Int, Nx_inner::Int, Ny_inner::Int,
                                   n_ghost::Int, i_c_start::Int, j_c_start::Int,
                                   Nx_c::Int, Ny_c::Int,
                                   omega_coarse::Real, omega_fine::Real)
    backend = KernelAbstractions.get_backend(f_fine)
    T = eltype(f_fine)
    alpha = T(rescaling_factor_c2f(omega_coarse, omega_fine, ratio))
    Nx_f = Nx_inner + 2 * n_ghost
    Ny_f = Ny_inner + 2 * n_ghost
    kernel! = prolongate_f_rescaled_2d_kernel!(backend)
    kernel!(f_fine, f_coarse, rho_c, ux_c, uy_c,
            ratio, Nx_inner, Ny_inner, n_ghost,
            i_c_start, j_c_start, Nx_c, Ny_c, alpha;
            ndrange=(Nx_f, Ny_f))
end
```

## See in action

- [Grid-refined cavity](../examples/20_grid_refinement_cavity.md) — nested
  patches around the top-lid singularity.
- [Lid-driven cavity 2D](../examples/04_cavity_2d.md) — uniform-grid
  baseline to compare against.

```julia
nothing  # suppress REPL output
```

