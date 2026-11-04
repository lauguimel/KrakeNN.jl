# # Electroconvection restart and controlled continuation
#
# This client implements the shared state contract for Issue #22. It does not
# establish an electroconvection onset or hysteresis benchmark. The scientific
# target is Wu et al., Physical Review E 88, 053018 (2013),
# DOI: 10.1103/PhysRevE.88.053018, Figs. 3--6.
#
# Save only on a completed coupled cycle. The existing HDF5 layer supplies file
# integrity checks and previous-generation recovery; no second file format is
# introduced. Restore on the same backend and precision in this first increment.
# Cross-grid, cross-precision, cross-backend and schema migration are not qualified.
# The caller may extend a run simply by calling `advance!` for more cycles.
#
# ```julia
# using Kraken
# s = init_state(ECState; Nx=10, Ny=16, T=190.0,
#                phi_scheme=:direct, history_interval=3)
# advance!(s, 7)
# save_checkpoint("segment.h5", s)
# r = load_checkpoint(ECState, "segment.h5")
# update_parameter!(r, :T, 185.0)
# advance!(r, 13)
# save_checkpoint("next_segment.h5", r)
# ```
#
# The tiny example verifies an interface, not a resolved convective branch.
# T updates keep the grid, C, M, Ma_E, alpha and voltage scale fixed. Only the
# mapped viscosity, relaxation parameters and T_check may change. Populations,
# force history, clock and sampling cadence do not reset. Both BGK and MRT read
# the updated mapping in the existing stepping loop.
#
# Checkpoints preserve populations, force history, carried convergence scalars,
# sampling histories and a (cycle, T) change log. Direct-Poisson potential is
# retained at its actual staggered time level, not recomputed from newer charge.
# Timing is segment-local instrumentation and is not restart state. The campaign
# must record per-job time, parent checkpoint and source SHA outside the solver.
#
# Scientific workflow: independently verify static injection profiles first,
# then small-amplitude growth and the free-wall critical threshold, and finally
# continue an established nonlinear branch downwards to the fold. Report Vmax*,
# charge/streamfunction fields and electrical current, with grid, Mach-number,
# diffusion and time-window sensitivity. Restart parity is not physical validation.
