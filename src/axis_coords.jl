# =====================================================================
# Wall-aware physical node coordinates
# =====================================================================
#
# LBM cell-centred nodes map to physical coordinates DIFFERENTLY depending
# on the boundary condition at each end of an axis:
#
#   :bb     — halfway bounce-back: the wall sits half a cell OUTSIDE the end
#             node (wall at 0 or 1; node is half a cell inside).
#   :onnode — Zou-He velocity/pressure: the wall coincides WITH the end node.
#
# Hand-coding `(j-0.5)/N` silently assumes BOTH ends are halfway-BB. For a
# MIXED domain — e.g. the lid-driven cavity with a Zou-He moving lid (on-node)
# and halfway-BB side/bottom walls — that mislocates the Zou-He wall by half a
# cell, injecting an O(1/N) (first-order) error into every reference comparison
# (Ghia, analytic, RheoTool). Use this helper everywhere a KrakeNN field is
# mapped to physical coordinates for comparison or output.
#
# Convention: the lo-end wall is placed at 0, the hi-end wall at 1.
#   :bb / :bb        → (j-0.5)/N         (H = N·Δ)      both walls halfway-BB
#   :bb / :onnode    → (j-0.5)/(N-0.5)   (H = (N-0.5)Δ) BB lo, Zou-He hi (cavity lid)
#   :onnode / :onnode→ (j-1)/(N-1)       (H = (N-1)Δ)   both walls on-node
# =====================================================================

"""
    axis_node_coords(N; lo=:bb, hi=:bb) -> Vector{Float64}

Physical coordinates in `[0, 1]` of the `N` cell-centred LBM nodes along one
axis, with the wall at the `lo` end placed at `0` and the `hi` end at `1`,
accounting for the boundary-condition placement:

- `:bb`     — halfway bounce-back (wall half a cell outside the end node);
- `:onnode` — Zou-He velocity/pressure (wall coincides with the end node).

This is the single source of truth for node→physical-coordinate mapping; profile
extraction, reference comparison and coordinate output should call it instead of
hand-coding `(j-0.5)/N`, which is correct only when BOTH ends are `:bb`.
"""
function axis_node_coords(N::Integer; lo::Symbol=:bb, hi::Symbol=:bb)
    (lo === :bb || lo === :onnode) || throw(ArgumentError("lo must be :bb or :onnode, got $lo"))
    (hi === :bb || hi === :onnode) || throw(ArgumentError("hi must be :bb or :onnode, got $hi"))
    off_lo = lo === :bb ? 0.5 : 0.0
    off_hi = hi === :bb ? 0.5 : 0.0
    denom = (N - 1) + off_lo + off_hi
    return [(j - 1 + off_lo) / denom for j in 1:N]
end
