# External comparisons

This page summarises how KrakeNN compares to other LBM and CFD codes in
terms of performance and accuracy.

!!! note
    Direct performance comparisons across codes are inherently approximate:
    different hardware, compiler versions, data layouts, and collision
    operators all affect the numbers. The data below is meant to provide
    **context**, not a strict ranking.

## Literature comparison table

A detailed table comparing KrakeNN MLUPs against published results from
Palabos, OpenLB, waLBerla, TCLB, and Sailfish is maintained in
[`benchmarks/external/comparisons.md`](https://github.com/lauguimel/KrakeNN.jl/blob/main/benchmarks/external/comparisons.md).

!!! warning "Work in progress"
    The external comparisons file is being assembled as part of Phase 5.4.
    Once available, a summary table will be inlined here.

## Key takeaways

1. **GPU throughput: no sourced KrakeNN number on this page.** Earlier drafts quoted
   two throughput figures for a single H100 (a baseline kernel and an AA-pattern +
   Float32 variant). No script or log in the repository backs either figure, so both
   were removed. The one
   committed CUDA measurement of the D2Q9 BGK solver is 3461 MLUPS in Float64 at
   `N = 2048` on an A100-40GB, 0.32 of that GPU's bandwidth ceiling (see
   [GPU certification](../users/benchmarks/gpu-certification.md)). Comparisons with
   other codes are therefore left out until a sourced run on the same hardware exists.

2. **Code size.** KrakeNN achieves these numbers in roughly 5 000 lines of
   Julia (solver core), compared to 50 000+ lines typical of C++ LBM
   frameworks. Julia's multiple dispatch and KernelAbstractions.jl remove
   the need for template metaprogramming or code generation.

3. **Accuracy.** On the canonical test cases (Poiseuille, Taylor-Green,
   lid-driven cavity), KrakeNN produces errors identical to or smaller than
   those reported for OpenFOAM's `icoFoam` at the same mesh resolution.
   Both codes are second-order; differences come from the discretisation
   details (LBM vs finite-volume) rather than from implementation bugs.

## OpenFOAM cross-validation

For single-phase laminar flows, KrakeNN results have been compared against
OpenFOAM 11 (`icoFoam`, second-order central, PISO) on matching structured
meshes. Both codes converge at ``\mathcal{O}(\Delta x^2)`` with KrakeNN
showing marginally smaller ``L_2`` errors on regular grids — a known
property of BGK on uniform lattices.

See the [Accuracy](accuracy.md) page for convergence tables.
