# BSD Analytical Ladder - 2026-05-17

Mission: M17-canary-analytical (L0-L2) + M17-canary-A (L2b Option A test) + M17-nyquist (L4 spectral test)

Params: U_0 = 1.0, nu_p = 0.1, zeta = 0.75, N = (32, 64, 128); L4 uses N=64.

Taylor-Green velocity (L0-L2b): `Ux = U_0 sin(2*pi*x) cos(2*pi*y)`, `Uy = -U_0 cos(2*pi*x) sin(2*pi*y)` on cell centers of `[0, 1]^2`. The references are `lap(U) = -8*pi^2 U`, `tau_p = 2*nu_p*D`, `F_poly = nu_p*lap(U)`, and `F_total = (1 - zeta)*nu_p*lap(U)`.

```text
=========================================================
BSD analytical ladder - Taylor-Green vortex, CPU F64
nu_p = 0.1, zeta = 0.75, U_0 = 1.0
=========================================================

L0 - div(tau_p) vs nu_p * lap(U) (periodic, no walls)
  N=32   rel_L2 = 6.413e-03
  N=64   rel_L2 = 1.606e-03
  N=128  rel_L2 = 4.015e-04    (order ~= 2.00)

L1 - F_total = F_poly - F_BSD vs (1 - zeta) * nu_p * lap(U) (periodic)
  kind=:fd     N=32 rel_L2 = 1.603e-02
               N=64 rel_L2 = 4.014e-03  (order ~= 2.00)
               N=128 rel_L2 = 1.004e-03
  kind=:fd_v2  N=32 rel_L2 = 1.270e-02
               N=64 rel_L2 = 3.203e-03  (order ~= 2.00)
               N=128 rel_L2 = 8.026e-04
  Delta(:fd vs :fd_v2)  N=64 rel_L2 = 7.217e-03

L2 - F_total at walls (closed box, lid profile matches TG)
  kind=:fd     interior rel_L2 = 4.01e-03 | wall rel_L2 = 1.03e-01 |
               max|F_BSD| ratio wall/interior = 9.95e-01
  kind=:fd_v2  interior rel_L2 = 3.20e-03 | wall rel_L2 = 3.03e+02 |
               max|F_BSD| ratio wall/interior = 1.58e+02

L2b - Option A test: BSD reads D_uncorrected (closed box, walls)
  kind=:fd          N=32  interior rel_L2 = 1.603e-02 | wall rel_L2 = 1.969e-01 | ratio = 9.809e-01
               N=64  interior rel_L2 = 4.014e-03 | wall rel_L2 = 1.027e-01 | ratio = 9.952e-01  (interior order ~= 2.00)
               N=128 interior rel_L2 = 1.004e-03 | wall rel_L2 = 5.188e-02 | ratio = 9.988e-01
  kind=:fd_v2       N=32  interior rel_L2 = 1.270e-02 | wall rel_L2 = 7.682e+01 | ratio = 4.079e+01
               N=64  interior rel_L2 = 3.203e-03 | wall rel_L2 = 3.026e+02 | ratio = 1.575e+02  (interior order ~= 2.00)
               N=128 interior rel_L2 = 8.026e-04 | wall rel_L2 = 1.207e+03 | ratio = 6.244e+02
  kind=:fd_v2_unc   N=32  interior rel_L2 = 1.270e-02 | wall rel_L2 = 3.173e-01 | ratio = 9.857e-01
               N=64  interior rel_L2 = 3.203e-03 | wall rel_L2 = 1.631e-01 | ratio = 9.964e-01  (interior order ~= 2.00)
               N=128 interior rel_L2 = 8.026e-04 | wall rel_L2 = 8.212e-02 | ratio = 9.991e-01
  Wall drop factor :fd_v2 -> :fd_v2_unc at N=64 = 1.86e+03

L4 - Spectral test: WIDE F_poly vs NARROW Laplacian (periodic, N=64)
  Velocity uses cell-INDEX sampling; mode m=N/2 is the Nyquist checkerboard.
  Both ratios are normalised by the analytical |lap(U)|_inf at the mode.
  m    k_x*dx       WIDE ratio    NARROW ratio  |lap(U)|_inf_analytical
  1    0.098175     9.9839e-01    9.9920e-01    7.8957e+01
  2    0.196350     9.9359e-01    9.9679e-01    3.1583e+02
  4    0.392699     9.7450e-01    9.8721e-01    1.2633e+03
  8    0.785398     9.0032e-01    9.4964e-01    5.0532e+03
  16   1.570796     6.3662e-01    8.1057e-01    2.0213e+04
  32   3.141593     3.4316e-15    4.0528e-01    8.0852e+04
  Reference: NARROW(Nyquist) = 4/pi^2 = 0.405285

=========================================================
VERDICT (L2):  :fd_v2 gives the tighter periodic cancellation, while :fd has the lower wall-band BSD spike; the M17 implication is to favor Option A D_uncorrected wall handling unless a separate kinetic default is validated.
VERDICT (L2b): [GREEN] implement Option A in cavity_driver_2d.jl as M17
VERDICT (L4):  WIDE Nyquist null-mode hypothesis: CONFIRMED
=========================================================
```

Interpretation (L0-L2): At N=64, the periodic ladder identifies which BSD cancellation strategy matches the analytical Newtonian-limit force, while the closed-box wall band quantifies the penalty from applying cavity wall gradients to the BSD stress. The wall/interior BSD maxima are 9.95e-01 for :fd and 1.58e+02 for :fd_v2, so the M17 architecture should keep the BSD stabilizer on an uncorrected D path at walls unless a separately validated kinetic default replaces it.

## L2b - Option A test (D_uncorrected for BSD)

L2b directly tests Option A: BSD reads D_uncorrected (centered FD only, no wall-correction overwrite) while keeping the wide-stencil :fd_v2 kernel path. At N=64 the :fd_v2_unc row gives interior rel L2 = 3.20e-03 (L1 :fd_v2 baseline 3.20e-03), wall band rel L2 = 1.63e-01 (down from :fd_v2's 3.03e+02 by factor 1.86e+03), and wall/interior |F_BSD| ratio = 9.96e-01 (vs :fd 9.95e-01, :fd_v2 1.58e+02). Verdict: [GREEN] implement Option A in cavity_driver_2d.jl as M17.

## L4 - Spectral test at Nyquist mode

Periodic N=64, CPU F64. Cell-INDEX Fourier modes m in (1, 2, 4, 8, 16, 32). The Nyquist mode is m = N/2 = 32 (k_x*dx = pi). Velocity field:

```
Ux(i,j) = U_0 * cos(2*pi*m*i/N) * cos(2*pi*m*j/N)
Uy(i,j) = -U_0 * sin(2*pi*m*i/N) * sin(2*pi*m*j/N)
```

WIDE applies `logfv_polymer_force_bc_aware_2d!` on the analytical tau = 2*nu_p*D(U); NARROW calls `logfv_bsd_correct_force_bc_aware_2d!` with fx_poly=0, zeta=-1, nu_p=1 to extract lap(U)_narrow alone. Both ratios are normalised by the analytical |lap(U)|_inf = 2*k^2*U_0 at the mode.

| m | k_x*dx | WIDE / |lap U|_anal | NARROW / |lap U|_anal | |lap U|_anal |
|---|--------|---------------------|-----------------------|--------------|
| 1 | 0.098175 | 9.9839e-01 | 9.9920e-01 | 7.8957e+01 |
| 2 | 0.196350 | 9.9359e-01 | 9.9679e-01 | 3.1583e+02 |
| 4 | 0.392699 | 9.7450e-01 | 9.8721e-01 | 1.2633e+03 |
| 8 | 0.785398 | 9.0032e-01 | 9.4964e-01 | 5.0532e+03 |
| 16 | 1.570796 | 6.3662e-01 | 8.1057e-01 | 2.0213e+04 |
| 32 | 3.141593 | 3.4316e-15 | 4.0528e-01 | 8.0852e+04 |

Interpretation: At the Nyquist mode (m = N/2 = 32, k_x*dx = pi), the WIDE F_poly ratio is 3.4316e-15 and the NARROW Laplacian ratio is 4.0528e-01 (analytical NARROW reference 4/pi^2 = 0.4053). The L4 mechanism is that the WIDE divergence stencil uses the 2dx centred difference, whose Fourier symbol sin(k*dx)/dx is identically zero at k*dx = pi; equivalently, the strain rate D built from a checkerboard velocity is zero at every cell centre with cell-INDEX sampling, so tau and its WIDE divergence both vanish. The NARROW 5-point Laplacian on the same checkerboard returns 4*|U|/dx^2 per axis (8*|U|/dx^2 total), giving the 4/pi^2 ratio. Null-mode hypothesis verdict: CONFIRMED.

Implication for the (epsilon) split: the proposed implementation applies a NARROW 5-point Laplacian directly on u (the same operator tested here), so it inherits the >=40 percent Nyquist damping demonstrated in the table above. Replacing F_poly_WIDE by (NARROW Laplacian)*(nu_s + nu_p)*u therefore closes the WIDE null mode in addition to the M10 truncation bias, and gives the (epsilon) split a second, stability-relevant benefit beyond the 3.4 percent bulk-residual reduction.
