# EC boundary extension: prospective external transient screen

2026-10-19. Scope: a new external two-moment reconstruction; no production
source edit, no Poisson/hydrodynamic coupling, no repeated static campaign.
Parent static evidence is job 26973672. Source algebra pinned at
ba22dcc213674ac8930bec1f8f13879e1a1bc1b9. Approach agreement for src/ is pending
in #73. This standalone verification does not imply that permission or a PR.

## Candidate and exact boundary identities

At the bottom wall q=qb, v=u+K E. Read outgoing post-stream populations before
any overwrite: G=fS+fSW+fSE and D=fSE-fSW. Reconstruct

    jneq_y = 2*(qb*(1-3*vy+3*vy^2)/6-G)
    jneq_x = 6*D-qb*vx*(1-3*vy)
    f_k = feq_k(qb,v)+3*w_k*(cx_k*jneq_x+cy_k*jneq_y).

Thus retain mass qb, outgoing normal aggregate G, and outgoing tangential
difference D. No neighbour extrapolation, coefficient fit or prescribed current.
Top Dirichlet isolation control uses the reflected formula with N/NE/NW.
At the EC collector retain the existing zero-gradient NEE algorithm instead.

Execute electrode interior updates first, recompute scalar charge, then execute
sidewall NEE including corners from those completed values. This defines corner
precedence without same-kernel read-after-write ambiguity. For the external
candidate only, bottom corners inherit qb from the adjacent completed electrode.
The old default source path is untouched. Sidewall and collector discretization
are not upgraded or silently called second order.

## One analytic transient family, two boundary layouts

Solve q_t+U q_y=D0*(q_xx+q_yy), D0=.05, U=.1 on [0,1]^2. Let r=U/(2D0)=1,
q=1+.05*exp(r*y)*sin(k*y)*cos(kx*x)*exp(-lambda*t),
lambda=D0*(kx^2+k^2+r^2). This follows by substitution into the PDE; no forcing.
Bottom q=1 for both layouts.

- Injector isolation: periodic x, kx=2*pi, k=pi, top q=1. Both flat electrodes
  use the reflected two-moment Dirichlet reconstruction.
- EC-box compatibility: kx=pi, insulating sides q_x=0, top q_y=0, so
  k*cos(k)+r*sin(k)=0, k in (pi/2,pi). Use legacy collector and sidewall NEE
  in ordered phases. This is fixed prescribed drift, not a full EC solution.

H16/32/64, tau=.8, dt=((tau-.5)/3)*h^2/D0 (diffusive refinement), final time
the nearest timestep to .2. Compare at actual time; all nodes included.
Initialize leading diffusive nonequilibrium moments to reduce initial kinetic
transients. Errors normalized by the analytic perturbation q_ref-1, not the
large constant background. Report L2 and Linf. No reference-driven BC moments.

## Frozen gates before execution

- Algebraic mass/G/D identities <=2e-14 on deterministic nonzero vx/vy fixtures;
  uniform equilibria invariant, positivity/finiteness throughout all runs.
- Isolation: successive perturbation-L2 order >=1.7; H64 L2 <=2%, Linf <=3%.
- Box: successive perturbation-L2 order >=.8; H64 L2 <=5%, Linf <=8%.
  First-order minimum is deliberate: retained node-copy Neumann conditions are
  not assumed second order. Passing box gates is preliminary compatibility only.
- Identical initial data and deterministic update schedules; reordered sidewall
  execution produces the same result (<=2e-14).
- If any gate fails, retain results and stop before GPU or coupled submission.
  Do not tune the formula/gates and rerun within this evidence record.
- NumPy mirror is an independent bounded development screen, not production
  kernel qualification. Cap CPU execution at 30 seconds and preserve timeout.
  If admitted, qualify actual Julia CPU/CUDA formula and transient seam in one
  bounded GPU allocation; only that later step can support an integration PR.

The deployable low-tau, coupled Poisson, velocity/shear, restart and full-suite
checks remain separate prerequisites. No full T190 job follows automatically.
