# GNF Adjoint Gradient (Steady)

Kraken can compute the gradient of an objective of the converged steady velocity of
a generalised-Newtonian (GNF) lattice Boltzmann flow with respect to the parameters
of its viscosity law. The flow is 2D D2Q9, periodic in `x`, driven by a body force,
with a Carreau–Yasuda viscosity law. The differentiated parameters are

```text
θ = (ln η∞*, ln k*, n, a)
```

where `η∞* = η∞/η0` and `k*` is the time constant of the law in reduced units
(the example below shows the exact mapping to `CarreauYasuda(eta_0, eta_inf,
lambda, a, n)`). The objective is `J = ½ Σ (value − target)²`, where each value is a weighted
sum of lattice-velocity components read at fluid nodes (`GNFVelocityObs`).

The result is the discrete adjoint of Kraken's own step at the fixed point, not a
tape of the transient. It does not use Enzyme: the local derivatives are built with
ForwardDiff and the adjoint kernels are written by hand.

## Method

- The state is `w = (f, τ)`: the nine populations and the relaxation time at every
  node, one flat vector of length `10 Nx Ny`. One step is `w' = G(w)`.
- The step is linearised at the converged fixed point `w*`. Per fluid node, the
  10×10 Jacobian block of the collision, plus one column for the derivative with
  respect to the viscosity, is computed with ForwardDiff on the host in Float64.
  The viscosity law enters through its value and its slope in the strain-rate
  invariant at each node.
- The Jacobian-vector and transposed products are KernelAbstractions kernels.
  The Krylov solve works on the backend of the operator.
- Mass is conserved, so `I − Aᵀ` is singular on the active components. The adjoint
  system is bordered with the mass vector to remove this mode. Components that
  never exchange with the fluid (solid rest populations, solid–solid links, `τ` at
  solid nodes) are left out of the active space.
- The gradient is assembled from the per-node sensitivity to the viscosity,
  `dJ/dθ = Σ_c s_c ∂ν_c/∂θ`, with `∂ν_c/∂θ` from ForwardDiff on the viscosity law.

Solvers, selected by the `solver` keyword of `gnf_steady_adjoint` and
`gnf_steady_gradient`:

- `:kstep` (default): GMRES on the K-step system, `K = 1001` (must be odd),
  `restart = 50`, `orth = :cgs2`.
- `:gmres`: GMRES on the gauge-bordered system.
- `:fixed_point`: fixed-point iteration of the adjoint series.

## What runs where

The tests ran on CPU, Float64 (macOS arm64, Julia 1.12.5, 2026-11-02). The CPU/GPU
parity testset (`G CPU/GPU parity`) is skipped when no CUDA device is available and
was skipped in that run. The same checks were then run on one NVIDIA A100 with the
script `benchmarks/validation/gnf_adjoint/run_gnf_adjoint_gpu.jl`; see "GPU results"
below. The numbers in "Measured agreement" are the CPU ones.

## Example

This block is not executed by the documentation build. It was run once on CPU as
written; the printed values are given below it.

```julia
using Kraken

# Channel periodic in x, walls in y, a half-disk on each wall; 32x16 nodes;
# Carreau-Yasuda viscosity law.
m = 4; Nx, Ny = 8m, 4m
nu0 = 0.17                                   # zero-shear viscosity, lattice units
solid = [(((i - 0.5) / m - 4)^2 + ((j - 0.5) / m)^2 < 1.5^2) ||
         (((i - 0.5) / m - 4)^2 + ((j - 0.5) / m - 4)^2 < 1.5^2) for i in 1:Nx, j in 1:Ny]
g = 0.3 * 5 * nu0^2 / m^3                    # body force, lattice units

# theta = (ln eta_inf*, ln k*, n, a); eta_inf* = eta_inf/eta_0, k* = lambda nu0/m^2
closure_of = θ -> CarreauYasuda(nu0, exp(θ[1]) * nu0, exp(θ[2]) * m^2 / nu0, θ[4], θ[3])
θ0 = [log(0.02), log(5.0), 0.7, 2.0]

# 1. Converge the forward flow to the steady state.
r = run_gnf_periodic_steady(; is_solid=solid, closure=closure_of(θ0), Fx=g,
                            tol=1e-10, check_every=20, fused=true)
@show r.converged r.steps

# 2. Observe both velocity components at every fluid node, targets at 0.8 of
#    the converged velocity, so that the gradient is not zero.
uref = nu0 / m
nodes = findall(vec(.!solid))
M = 2 * length(nodes)
node = zeros(Int, 1, M); weight = ones(1, M); comp = zeros(Int, M); target = zeros(M)
for (k, nd) in enumerate(nodes)
    node[1, 2k-1] = nd; comp[2k-1] = 1; target[2k-1] = 0.8 * r.ux[nd] / uref
    node[1, 2k]   = nd; comp[2k]   = 2; target[2k]   = 0.8 * r.uy[nd] / uref
end
obs = GNFVelocityObs(node, weight, comp, target, 1 / uref)

# 3. Adjoint gradient of J = 1/2 sum (value - target)^2 with respect to theta.
res = gnf_steady_gradient(Array(r.f), Array(r.tau), θ0, obs;
                          is_solid=solid, closure_of=closure_of, Fx=g)
@show res.J res.grad res.info.converged
```

Output of that run:

```text
r.converged = true
r.steps = 1880
res.J = 1.6009928087874041
res.grad = [-0.38363374416914886, 5.189977557137756, -41.19559958659784, -0.17214679584589132]
res.info.converged = true
```

The velocity observed is the post-collision lattice velocity. The force is the
reference body force of the test case multiplied by 0.3 (Mach about 0.108, measured
in the test, see below).

## Measured agreement

All numbers below come from `test/ad/gnf_adjoint_test.jl` and
`test/ad/gnf_adjoint_enzyme_test.jl`, run on CPU Float64 (macOS arm64, Julia 1.12.5,
2026-11-02). The summary of the first file is 55 passed, 1 broken (the skipped GPU
parity testset), 56 total. The test case is a channel periodic in `x` with walls at
`y = 0` and `y = 4` (domain 8×4), obstructed by disks of radius 1.5 centred on each
wall at `x = 4`, so a half-disk on each wall, at 32×16 nodes, with a measured Mach number of 0.108
and a viscosity ratio ν/ν0 between 0.390 and 0.997.

**T5, adjoint gradient against central finite differences of J** (32×16, reference
parameters). The adjoint column is the bordered-GMRES gradient at tolerance 1e-8 on
the third forward level. The finite difference is the one, among the four relative
steps h = 1e-2 … 1e-5, with the smallest difference for that parameter (h = 1e-5
for all four).

| Parameter | Adjoint | Central FD | abs. difference | Gate bound |
|---|---|---|---|---|
| ln η∞ | -3.836337e-01 | -3.836337e-01 | 1.11e-09 | 2.11e-02 |
| ln k  | 5.189978e+00 | 5.189978e+00 | 1.36e-10 | 2.60e-02 |
| n     | -4.119560e+01 | -4.119560e+01 | 7.06e-08 | 6.20e-02 |
| a     | -1.721468e-01 | -1.721468e-01 | 2.20e-09 | 2.09e-02 |

The gate, as written in the test, is `|adjoint − FD| ≤ atol + 1e-3 |FD|` with
`atol = 1e-3 · ‖g‖ / √4`, where `g` is the adjoint gradient vector. The gate is far
looser than the measured agreement: the largest measured difference (7.06e-08) is
more than five orders of magnitude below the smallest bound (2.09e-02). The finite
difference with the best step is chosen per parameter after the fact, which favours
the comparison; the printed relative differences at the other steps are larger
(for `n` at the third level and tolerance 1e-8: 1.8e-03, 1.8e-05, 1.8e-07 and
1.7e-09 relative difference at h = 1e-2, 1e-3, 1e-4, 1e-5).

**Tb, ∂J/∂w against central differences of J.** The worst relative error over three
random directions is 1.12e-10; the gate is 1e-7.

**T4d, Jacobian-vector product against central differences of the real step**
(`gnf_step!`, 32×16, five random directions). The relative error at
h = 1e-3, 1e-4, 1e-5, 1e-6 falls by a factor of about 100 per decade of h (for
direction 1: 7.40e-02, 8.64e-04, 8.66e-06, 8.66e-08). At h = 1e-6 it is between
5.82e-08 and 8.66e-08 over the five directions. The gate on the best step is
1e-6, plus a check that the ratio of errors between the two largest steps is
between 30 and 300 (measured 82.1 to 90.5).

**Enzyme cross-check of the transposed product** (`T4` in
`test/ad/gnf_adjoint_enzyme_test.jl`, 32×16, viscosity contrast 50). The relative
difference between the hand-written transposed product and Enzyme reverse mode is
3.61e-16, 3.43e-16 and 3.32e-16 for three random vectors; the gate is 1e-12. Enzyme
is used only in this test, not in the gradient computation.

Other measured checks in the same file: transposed stream (exact permutation), mass
gauge (`max|Aᵀc − c|` = 9.66e-15), adjointness of the transposed and direct products
(worst relative mismatch 3.19e-15 over 5 pairs), and agreement of the three solvers
on the same adjoint state (relative difference 7.3e-14 for `:fixed_point`,
1.0e-12 for `:kstep` with K = 101, against a bordered-GMRES reference at tolerance
1e-12).

## GPU results

Source: `benchmarks/validation/gnf_adjoint/results/gnf_adjoint_gpu.csv` (and
`gnf_adjoint_gpu.log`), produced on
2026-11-02 by `benchmarks/validation/gnf_adjoint/run_gnf_adjoint_gpu.jl` on
`CUDABackend()`: NVIDIA A100-SXM4-40GB, Float64, Julia 1.12.5, CUDA.jl 5.11.3, driver
610.57.4. The run exited 0 and all gated checks passed. The code that ran is identical
to the branch commit `12fd5d3ba` (checked file by file), which is kept reachable by the
tag `archive/gnf-adjoint-gpu-check`. The branch was later rebased onto `dev/platform`
660b8e871; the only source changes that brought in (the `.krk` `Backend` directive, in
`src/io/krk/directives.jl`, `src/io/krk/parser.jl` and `src/simulation_runner.jl`) are
not used by this script. Provenance details are in the README of that directory. The case is the 32×16 channel of the CPU tests.

- **GPU against CPU operators.** The Jacobian-vector and transposed products on the
  device differ from the CPU ones by a relative 1.2e-16 and 8.7e-17 (bound 1e-14). The
  adjoint solution at tolerance 1e-10 differs by a relative 6.2e-14 (bound 1e-8).
- **Gradient against finite differences.** Absolute differences 9.4e-10 (ln η∞),
  3.7e-11 (ln k), 7.1e-08 (n) and 2.0e-09 (a), against gate bounds between 2.1e-2 and
  6.2e-2. The gate is the same loose one as on CPU, and the best step is again
  chosen per parameter after the fact (h = 1e-5 for all four).
- **GPU gradient against CPU gradient** (informational, no gate): relative differences
  between 5.9e-13 and 9.1e-13.
- **Float32 operator** (informational, no gate): with Float32 operator products the
  solver reached tolerance 1e-5 and the gradient differs from the Float64 one by a
  relative 1.5e-7 to 1.7e-7.

What this does not show: only this one problem size, on one A100, in one run (no H100,
no repeat); no timing or speed-up claim; Float32 has no gate and no Float32 finite
difference check was made; only the default solver was run on the GPU.

## Not yet verified

- GPU scope: one A100 run on the 32×16 case (see "GPU results"). The CPU/GPU parity
  testset in the test suite is still skipped without a CUDA device.
- Float32 is informational only (operator products, no gate).
- The Enzyme cross-check runs in no CI step yet; it runs only when selected with
  `KRAKEN_ONLY=ad/gnf_adjoint_enzyme_test.jl`.
- Only the 32×16 and 16×8 test grids are covered (the GPU run: 32×16).
- Only the Carreau–Yasuda viscosity law is covered.
- Only 2D, `x`-periodic, body-force-driven flow is covered.
- Forward-solve distances in T5. The test labels the three forward levels
  `1e-10`, `1e-12`, `1e-14`, and the third is the one used in the gate. The log line
  `T5 forward levels` prints the Newton distance of each level as 3.7e-10, 1.3e-10
  and 4.9e-11, and the case testset prints the two-step estimate as 8.9e-11, 1.0e-10
  and 4.9e-11. The printed values are not 1e-12 and 1e-14. The test comment says the
  estimate saturates at the Float64 floor on this grid.
