# Second normal stress difference in the steady 3D FV solver

The steady 3D finite-volume (FV) Newton solver can carry a memoryless
(Reiner–Rivlin) stress

```math
\boldsymbol\tau = 2\,\eta(\dot\gamma)\,\mathbf D + 4\,\Psi_2(\dot\gamma)\,\mathbf D\cdot\mathbf D,
\qquad \dot\gamma = \sqrt{2\,\mathbf D:\mathbf D},
```

with `D` the rate-of-strain tensor. In simple shear it gives no first normal
stress difference (`N1 = 0`) and a second one `N2 = Psi2 gdot^2`. This is the
term that drives the secondary flow of a viscoelastic fluid in a non-circular
duct (eight vortices in a square duct). It is not a viscoelastic model with
memory: it holds where the flow is close to steady shear, as in a fully
developed duct flow.

## Usage

```julia
using KrakeNN
eta  = FVLawViscosity(GiesekusViscosityLaw(), [eta_s, eta_p0, lambda, alpha]; eps = 1e-4)
psi2 = FVLawViscosity(GiesekusPsi2Law(), [eta_p0, lambda, alpha]; eps = 1e-4)
prob = fv_steady_problem(1, N, N; h = 1 / N,
                         bc = (W = :periodic, E = :periodic, S = :wall, N = :wall, B = :wall, T = :wall),
                         viscosity = FVReinerRivlinViscosity(eta, psi2),
                         source = (x, y, z) -> (50.0, 0.0, 0.0, 0.0), u_mean_target = 1.0)
```

- `eta` is an `FVGNFViscosity` or an `FVLawViscosity`; `psi2` is an
  `FVLawViscosity`, so any scalar law (classical, tabulated, `TinyMLP`-based)
  can be used for either. Values are kinematic, as for the viscosity.
- Laws shipped: `GiesekusViscosityLaw` and `GiesekusPsi2Law` (steady-shear
  functions of the Giesekus model in closed form) and
  `TabulatedGiesekusPsi2Law(nodes)` (Giesekus `Psi2` times a piecewise-linear
  correction in `log10 Wi`).
- `FVReinerRivlinParameters(model)` maps `theta = [theta_eta; theta_psi2]`
  for `fv_set_params!`, `fv_steady_gradient` and `fv_steady_tangent` (the
  adjoint uses parameter dual numbers through the residual kernel).
- Isothermal only: no energy model and no state-temperature input.

## Discretisation

The `Psi2` term is added to the viscous traction of each face, interior and
wall, on the same face velocity gradient that sets `eta`, so the Jacobian (dual
numbers) includes it with no hand-written derivative and the Jacobian pattern
is the one of a shear-dependent viscosity. The Rhie–Chow coefficient keeps
`eta` only. Without this model the residual is unchanged.

With `Psi2 = c eta` the continuous forcing is a pure pressure gradient and a
rectilinear duct flow is exact; the FV forcing is not an exact discrete
gradient, so the discrete secondary flow is then a discretisation error that
decreases with refinement.

The driver `benchmarks/ve_duct/slice_forward.jl` runs the duct slice of the
Giesekus reference case on CPU or CUDA.
