# Research prompt — State-of-the-art Curvilinear LBM for Kraken.jl v0.2

## How to use this

Open a **fresh** Claude Code session (or Claude web), paste the prompt
below, and let the agent do the literature survey and comparative
analysis. The output of that session will be the input to the design
doc for Kraken.jl's curvilinear LBM implementation.

---

## Prompt to paste

```
You are helping design a GPU-native curvilinear Lattice Boltzmann
Method (LBM) implementation for a Julia CFD library called Kraken.jl.
The implementation needs to be single-kernel (one GPU dispatch per
timestep, like the existing uniform-grid path), work on CUDA + Metal
via KernelAbstractions.jl, and support D2Q9 (2D) as a first milestone.

Target use cases in order of priority:

  1. Flow past a cylinder with a polar/O-grid around the body
     (Schäfer-Turek 2D-1, Re=20, Cd ≈ 5.58).
  2. Boundary-layer-dominated flows (Poiseuille, channel flow,
     natural convection cavity) with wall-normal stretching.
  3. Generic body-fitted meshes (user supplies X(ξ,η), Y(ξ,η)).

Non-goals for this first pass:

  - 3D (follow-up).
  - Multi-block / unstructured meshes (use uniform-topology
    curvilinear only, i.e. a single logically-structured
    Nξ × Nη grid).
  - Dynamic mesh adaptation (streamline-adapted / Stream-Tube Method
    is parked as a separate v0.3 research track).

Please do a literature survey and deliver the following in one
response:

──────────────────────────────────────────────────────────────────
1. Formulation comparison

  Find and compare the main curvilinear LBM formulations in the
  literature. For each, state:
    - Key reference(s) with year
    - Discretisation family (ISLBM / FD-LBM / FV-LBM / semi-
      Lagrangian / other)
    - Whether BGK is stable or MRT/TRT/regularised is required
    - Demonstrated validation cases and Reynolds / Weissenberg
      range
    - GPU / parallelisation track record
    - Open-source implementations (if any)

  Candidate methods I already know about (validate and extend the
  list):
    - Mei & Shyy (1998) finite-difference curvilinear LBM
    - He, Luo, Dembo (1996) interpolation-supplemented LBM
    - Nannelli & Succi (1992) irregular lattices
    - Imamura, Suzuki et al. (2005) airfoil simulations
    - Budinski (2014) MRT curvilinear
    - Peng et al. FV-LBM
    - Lee & Lin spectral-element LBM
    - Palabos body-fitted; OpenLB curvilinear; waLBerla; Musubi
      (XDG)

2. Recommendation for Kraken.jl

  Given the constraints (single-kernel GPU, Julia +
  KernelAbstractions, BGK+MRT existing, D2Q9 first), pick the
  formulation that gives the best trade-off between:
    - Implementation simplicity (we want a ~3-4 week delivery,
      not a PhD)
    - Accuracy on Taylor-Couette and Schäfer-Turek cylinder
    - Stability at moderate Re (say 100-400)
    - GPU efficiency (no per-step host-device sync, no
      neighbour-of-neighbour look-ups beyond one layer)

  Justify the choice against the next-best alternative.

3. Key pitfalls & design requirements

  List the known pitfalls the literature warns about:
    - Mesh quality constraints (aspect ratio, orthogonality,
      Jacobian positivity, skewness limits)
    - Stability thresholds (what Mach / Re / mesh distortion
      combinations diverge)
    - Boundary condition formulations that must change from
      the Cartesian case (bounce-back on curved walls, Zou-He
      for curvilinear inlet, pressure outlets, periodic in θ)
    - Numerical issues: spurious modes, pressure-velocity
      coupling, grid-induced anisotropy

  For each pitfall, quote the reference that documents it.

4. Validation roadmap

  Propose an ordered list of validation cases, from cheapest to
  most demanding, each with:
    - Reference solution (analytical or published numerical)
    - Expected convergence order
    - Target error at a nominal resolution
    - What it validates (streaming, collision, BCs, metric,
      stability)

  Must include: Taylor-Couette (analytical), Poiseuille on
  stretched grid, Schäfer-Turek 2D-1 cylinder.

5. Mesh generation strategy

  What's the minimum viable mesh generator for v0.2?
    - Polar / O-grid around a point (parametric: cx, cy, R_in,
      R_out, n_r, n_θ, radial stretch parameter)
    - Orthogonal wall-normal stretch (parametric: x range, y
      range, stretch parameter, stretch function)
    - C-grid / H-grid: recommend or defer?

  Should we support an external mesh format (Gmsh .msh) or
  stick to parametric generators for v0.2?

6. API surface for the .krk DSL

  Propose a concrete syntax that integrates with the existing
  Kraken.jl .krk config system (blocks, named keys, expressions).
  The parser is flexible. Example target:

      Mesh polar {
          center = [2.0, 2.0]
          r_inner = 0.5
          r_outer = 5.0
          n_r = 80
          n_theta = 128
          r_stretch = tanh(2.0)
      }

  Compare against alternatives (e.g. separate `Curvilinear {}`
  block, function-based mapping, etc.). Recommend the cleanest.

7. Open questions to resolve before implementation

  List 3-5 concrete questions the team needs to answer before
  writing code.

──────────────────────────────────────────────────────────────────

Output format: one markdown document, under 4000 words, with clear
sections matching the seven headings above. Cite references as
"Author, Year, venue" — no need for BibTeX. Where a reference is
open-access online, include the URL. Be opinionated: we want a
recommendation, not just an inventory.

Context on Kraken.jl you can assume:
  - Julia 1.11+ package, D2Q9 and D3Q19 lattices, BGK+MRT
    collisions, thermal DDF, patch-based grid refinement
    (Filippova-Hänel) already shipped.
  - GPU via KernelAbstractions.jl, validated on CUDA (AQUA H100)
    and Metal (Apple M3 Max); single fused kernel reaches
    ~24000 MLUPS on H100 with AA+f32 optimisations.
  - Uniform-grid refinement ran ~10x slower than uniform on
    Metal due to many small kernel launches; genericity
    constraint rejected case-specific fused kernels. Curvilinear
    is attractive because a single warped mesh keeps the
    uniform-topology stream step — one kernel.
  - Repo: https://github.com/lauguimel/KrakeNN.jl (branch `lbm`).
  - v0.2 publication target: Computer Physics Communications,
    emphasising single-kernel GPU portability across
    CUDA/Metal/ROCm from one source.

I have literature access through my institution for any papers
behind paywalls — if you cite something paywalled, include enough
detail (journal, volume, pages) that I can retrieve it.
```

---

## After the research session

The output of that session goes into:

  `docs/design/curvilinear_lbm_design.md`

which will be the live design doc for implementation. I (Claude in
this workspace) will write that doc based on the research results,
then schedule Weeks 1-4 of the implementation plan.
