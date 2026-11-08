# Programmatic generator for the 8-block O-grid-in-rectangle topology
# targeting approach (E-full) in the WP-MESH-6 paper matrix.
#
# Layout:
#   - Cylinder at (cx_p, cy_p), radius R_in
#   - 8 cylinder-surface points at 45° intervals: p_0 (0°), p_1 (45°),
#     p_2 (90°), ..., p_7 (315°)
#   - 8 matching outer points on the rectangle boundary (`q_k`):
#         k=0 → (Lx, cy_p)     east-midpoint
#         k=1 → (Lx, Ly)       NE corner
#         k=2 → (cx_p, Ly)     top-midpoint
#         k=3 → (0, Ly)        NW corner
#         k=4 → (0, cy_p)      west-midpoint
#         k=5 → (0, 0)         SW corner
#         k=6 → (cx_p, 0)      bottom-midpoint
#         k=7 → (Lx, 0)        SE corner
#   - 8 Transfinite surfaces (blocks) between consecutive (p_k, q_k)
#     arrangements, each 4-sided: inner cylinder arc p_k→p_{k+1},
#     spoke p_{k+1}→q_{k+1}, outer rectangle segment q_{k+1}→q_k,
#     spoke q_k→p_k.
#
# Resulting topology:
#   - 8 blocks
#   - 16 distinct curves: 8 cylinder arcs + 8 spokes (+ 8 outer segments
#     that together cover the full rectangle perimeter)
#   - 8 shared spokes → 8 interfaces (block k.east ↔ block (k+1).west
#     in a circular chain)
#   - Physical Curves: "cylinder" (8 arcs), "inlet" (q_3→q_4, q_4→q_5),
#     "outlet" (q_7→q_0, q_0→q_1), "wall_top" (q_1→q_2, q_2→q_3),
#     "wall_bot" (q_5→q_6, q_6→q_7).
#   - Physical Surfaces: "ring_0" .. "ring_7".
#
# The generator is exposed as a function `write_ogrid_rect_8block_geo`
# so a driver can use it in-process via `mktempdir`.

using Gmsh

"""
    write_ogrid_rect_8block_geo(path::String;
                                  Lx=1.0, Ly=0.5,
                                  cx_p=0.5, cy_p=0.245,
                                  R_in=0.025,
                                  N_arc=8, N_radial=16,
                                  radial_progression=0.8)

Write an 8-block O-grid-in-rectangle `.geo` file at `path`. Each of the
8 blocks has `N_arc` cells along its cylinder arc (ξ direction) and
`N_radial` cells along its radial spoke (η direction), so the total
cell count is `8 * N_arc * N_radial`.

`radial_progression` (gmsh Transfinite "Progression"): ratio < 1 clusters
cells at the INNER (cylinder) end of each spoke, which is essential for
a body-fitted O-grid to resolve the boundary layer — and, for the KrakeNN
multi-block pipeline, to keep `dx_radial_at_cylinder` below `R_in` so
that `extend_mesh_2d` can linearly extrapolate one ghost row inward
without crossing the cylinder centre. A value of 0.8 gives ≈10× finer
cells at the cylinder than at the rectangle; use 0.7 for aggressive
clustering.

Use the resulting `.geo` with `load_gmsh_multiblock_2d(path)` to
obtain a `MultiBlockMesh2D` with 8 blocks + 8 interfaces.
"""
function write_ogrid_rect_8block_geo(path::AbstractString;
                                       Lx::Real=1.0, Ly::Real=0.5,
                                       cx_p::Real=0.5, cy_p::Real=0.245,
                                       R_in::Real=0.025,
                                       N_arc::Int=8, N_radial::Int=16,
                                       radial_progression::Real=0.8)
    # Cylinder surface points at 45° intervals.
    θ = [k * π / 4 for k in 0:7]     # 0°, 45°, 90°, ..., 315°
    p_xy = [(cx_p + R_in * cos(θk), cy_p + R_in * sin(θk)) for θk in θ]
    # Outer rectangle points:
    q_xy = [
        (Lx,   cy_p),     # k=0
        (Lx,   Ly),       # k=1
        (cx_p, Ly),       # k=2
        (0.0,  Ly),       # k=3
        (0.0,  cy_p),     # k=4
        (0.0,  0.0),      # k=5
        (cx_p, 0.0),      # k=6
        (Lx,   0.0),      # k=7
    ]

    open(path, "w") do io
        println(io, "SetFactory(\"Built-in\");")
        # --- Points ---
        # Cylinder center (for Circle arcs)
        println(io, "Point(99) = {$cx_p, $cy_p, 0};")
        # Cylinder-surface points 1..8 (tag = k+1)
        for (k, (x, y)) in enumerate(p_xy)
            println(io, "Point($k) = {$x, $y, 0};")
        end
        # Outer points 11..18 (tag = 10 + k+1)
        for (k, (x, y)) in enumerate(q_xy)
            println(io, "Point($(10 + k)) = {$x, $y, 0};")
        end

        # --- Curves ---
        # 8 cylinder arcs (curves 101..108): Circle(arc_k) = {p_k, center, p_{k+1}}
        for k in 1:8
            kn = k == 8 ? 1 : k + 1
            println(io, "Circle($(100 + k)) = {$k, 99, $kn};")
        end
        # 8 spokes (curves 201..208): p_k → q_k (k=1..8 → p has tag k, q has tag 10+k)
        for k in 1:8
            println(io, "Line($(200 + k)) = {$k, $(10 + k)};")
        end
        # 8 outer rectangle segments (curves 301..308): q_{k} → q_{k+1}
        for k in 1:8
            kn = k == 8 ? 1 : k + 1
            println(io, "Line($(300 + k)) = {$(10 + k), $(10 + kn)};")
        end

        # --- Surfaces ---
        # Block k (k=1..8) with loop [arc_k, spoke_{k+1}, -outer_k, -spoke_k]
        # i.e. p_k → p_{k+1} (arc) → q_{k+1} (spoke) → q_k (outer reversed)
        # → p_k (spoke reversed). Note: outer is q_k→q_{k+1}, so we need -outer.
        for k in 1:8
            kn = k == 8 ? 1 : k + 1
            arc = 100 + k
            spoke_next = 200 + kn
            outer = 300 + k
            spoke_k = 200 + k
            println(io, "Curve Loop($(400 + k)) = {$arc, $spoke_next, -$outer, -$spoke_k};")
            println(io, "Surface($(500 + k)) = {$(400 + k)};")
        end

        # --- Transfinite ---
        # Cylinder arcs and outer segments: N_arc nodes (tangential direction)
        # Spokes: N_radial nodes (radial direction)
        arc_ids  = join([100 + k for k in 1:8], ", ")
        outer_ids = join([300 + k for k in 1:8], ", ")
        spoke_ids = join([200 + k for k in 1:8], ", ")
        println(io, "Transfinite Curve {$arc_ids} = $N_arc;")
        println(io, "Transfinite Curve {$outer_ids} = $N_arc;")
        # Spokes: Progression <1 clusters nodes at the inner (cylinder)
        # end so the boundary-layer resolution and the KrakeNN mesh
        # extension inward remain valid.
        println(io, "Transfinite Curve {$spoke_ids} = $N_radial Using Progression $radial_progression;")
        for k in 1:8
            kn = k == 8 ? 1 : k + 1
            # Surface corners in canonical order: p_k, p_{k+1}, q_{k+1}, q_k
            println(io, "Transfinite Surface {$(500 + k)} = {$k, $kn, $(10 + kn), $(10 + k)};")
        end
        println(io, "Recombine Surface {$(join([500 + k for k in 1:8], ", "))};")

        # --- Physical groups ---
        # Outer-segment curve map (derived from q_xy coordinates above):
        #   301 = q_1→q_2 = (Lx, cy_p) → (Lx, Ly)   upper east edge
        #   302 = q_2→q_3 = (Lx, Ly)   → (cx_p, Ly) right-half top edge
        #   303 = q_3→q_4 = (cx_p, Ly) → (0, Ly)    left-half top edge
        #   304 = q_4→q_5 = (0, Ly)    → (0, cy_p)  upper west edge
        #   305 = q_5→q_6 = (0, cy_p)  → (0, 0)     lower west edge
        #   306 = q_6→q_7 = (0, 0)     → (cx_p, 0)  left-half bottom edge
        #   307 = q_7→q_8 = (cx_p, 0)  → (Lx, 0)    right-half bottom edge
        #   308 = q_8→q_1 = (Lx, 0)    → (Lx, cy_p) lower east edge
        println(io, "Physical Curve(\"cylinder\") = {$arc_ids};")
        println(io, "Physical Curve(\"outlet\")   = {301, 308};")
        println(io, "Physical Curve(\"wall_top\") = {302, 303};")
        println(io, "Physical Curve(\"inlet\")    = {304, 305};")
        println(io, "Physical Curve(\"wall_bot\") = {306, 307};")

        # Physical surfaces: "ring_0" .. "ring_7"
        for k in 1:8
            println(io, "Physical Surface(\"ring_$(k-1)\") = {$(500 + k)};")
        end
    end
    return path
end
