# Independent Nahme channel reference

Generated on 2026-11-08 by `benchmarks/fv_thermal/nahme_reference.py`, using
Python/SciPy 1.17.1 on macOS arm64. No Kraken operator, state, discretization,
or E5 driver is called. The script derives a continuum boundary-value problem
from momentum and energy balance and solves it by fourth-order collocation
(`scipy.integrate.solve_bvp`). No experimental or external reference data is
used. All values are dimensionless apart from the arbitrary temperature unit.

Regenerate from the repository root with the available local environment:

```sh
conda run -n kraken-v0-3-figures python benchmarks/fv_thermal/nahme_reference.py
```

The channel height is H=1, nu0=0.3, alpha=0.2, beta=1, Tw=1. The constitutive
law is nu(T)=nu0 exp[-beta(T-Tw)]. With shear stress tau=nu*u' and q=T', the
independent first-order equations are

```math
u'=\tau/\nu(T),\qquad \tau'=-F_x,\qquad
T'=q,\qquad q'=-\sigma\tau^2/[\alpha\nu(T)].
```

The boundaries are u(0)=0, u(1)=Uwall, T(0)=Tw, q(1)=0. These fully developed
solutions are periodic in x, with bottom Dirichlet and top adiabatic thermal
boundaries. In plane shear, 2(S:S)=(u')², so the heat source is exactly the
Brinkman source sigma*nu*(u')².

`poiseuille.csv` uses Fx=0.6, Uwall=0. The independent BVP also imposes
T(1)-Tw=0.7 and solves for sigma=1.9127858635347974. `couette.csv` uses Fx=0,
Uwall=1, sigma=0.8. The center viscosity decreases by 35.8088% and 29.2215%,
respectively. Columns y,u,T,nu sample the cell centers of N=16,32,64,128;
the stored decimal y values are exact binary fractions.

Three adaptive solutions use tolerances 1e-6,1e-8,1e-10. Their final mesh
node counts are 101,441,1977 for Poiseuille and 65,193,686 for Couette.
The medium-to-finest maximum differences over (u,tau,T,q) are 4.37e-11 and
3.86e-12, below the frozen reference error allowance 1e-8. A separate fixed
collocation mesh refinement with 32,64,128,256 intervals demonstrates order
approximately four. The full differences/orders, solver settings and
parameters are stored in `metadata.json`. This is numerical convergence
evidence, not an interval-arithmetic proof.

Before any FV solve, the finest FV error bounds were frozen at
8*C/128², with C=max(|u''|,|T''|,|nu''|) evaluated from the reference ODE on
4097 points. They are 0.002227206179125562 (Poiseuille) and
0.000496048208396679 (Couette). This conservative reference-curvature
acceptance envelope is independent of measured FV errors; second-order
convergence must independently pass order >=1.8 on 32,64,128. The envelope
is an acceptance criterion, not a rigorous stability bound on the FV inverse.

`fold_branch.csv` follows the fixed-Fx BVP by prescribing successive top
temperature rises from 0.05 to 4 and solving for sigma. Maximizing this
independently computed sigma(Ttop) gives the local thermal fold:
sigma_crit=2.611023571651698, Ttop-Tw=1.6661102501192615. Repeating with BVP
tolerances 1e-8/1e-10 changes sigma_crit by 5.13e-12, within the frozen
sigma allowance 1e-7. The actual branch rises then falls across this maximum;
no literature critical value is assumed. The fixed-force Nahme number here
is Na_F=beta*sigma*Fx²*H⁴/(nu0*alpha), giving Na_F,crit=15.666141429910187.
For moving-wall Couette, Na_U=beta*sigma*nu0*Uwall²/alpha=1.2. These are
different control conventions and their numerical critical values cannot be
interchanged. The FV continuation test requests sigma=3 (>sigma_crit) and
reports the last converged step and decreasing singular-value estimate.
