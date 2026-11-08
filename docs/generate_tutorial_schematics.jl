#!/usr/bin/env julia
# generate_tutorial_schematics.jl — boundary-condition schematics for the 4
# user tutorials. Self-contained (NO KrakeNN, NO simulations): it only includes
# the bc_schematic.jl helper and renders 4 SVGs next to the tutorial .md pages,
# mirroring how the example geometry SVGs sit beside their pages.
#
# Run: julia --project=docs docs/generate_tutorial_schematics.jl

include(joinpath(@__DIR__, "bc_schematic.jl"))

const OUTDIR  = joinpath(@__DIR__, "src", "users", "tutorials")
const PREVIEW = joinpath(@__DIR__, "bc_demo_out", "preview")
isdir(OUTDIR)  || mkpath(OUTDIR)
isdir(PREVIEW) || mkpath(PREVIEW)

function _emit(fig, name)
    save(joinpath(OUTDIR, name * ".svg"), fig)
    save(joinpath(PREVIEW, name * ".png"), fig; px_per_unit=2)
    println("  ✓ ", name, ".svg  (+ preview png)")
end

# ===========================================================================
# 1. Lid-driven cavity — north moving lid, S/E/W no-slip
# ===========================================================================
let
    fig = Figure(size=(500, 560), backgroundcolor=BC_DARK)
    ax = bc_axis(fig[1, 1]; title="Lid-driven cavity — boundary conditions",
                 limits=(0, 5, 0, 5), pad=1.3)
    fluid_region!(ax, (0.0, 0.0), (5.0, 5.0); outline=false)
    moving_wall!(ax, (0.0, 5.0), (5.0, 5.0); side=1, u_label="u_lid (Zou-He)",
                 label="moving lid", labelgap=0.5, ulabelgap=0.6)
    wall!(ax, (0.0, 0.0), (5.0, 0.0); side=-1, label="no-slip, bounce-back",
          labelgap=0.45)
    wall!(ax, (0.0, 0.0), (0.0, 5.0); side=1,  label="no-slip", labelgap=0.45)
    wall!(ax, (5.0, 0.0), (5.0, 5.0); side=-1, label="no-slip", labelgap=0.45)
    _emit(fig, "cartesian-cavity-bc")
end

# ===========================================================================
# 2. Differentially-heated cavity — west hot, east cold, N/S adiabatic, gravity
# ===========================================================================
let
    fig = Figure(size=(520, 560), backgroundcolor=BC_DARK)
    ax = bc_axis(fig[1, 1]; title="Differentially-heated cavity — boundary conditions",
                 limits=(0, 5, 0, 5), pad=1.5)
    fluid_region!(ax, (0.0, 0.0), (5.0, 5.0); outline=false)
    # hot west wall (vertical, solid on the LEFT/outside)
    dirichlet_wall!(ax, (0.0, 0.0), (0.0, 5.0); kind=:hot, side=1,
                    label="T_hot = 1", labelgap=0.55)
    # cold east wall (vertical, solid on the RIGHT/outside)
    dirichlet_wall!(ax, (5.0, 0.0), (5.0, 5.0); kind=:cold, side=-1,
                    label="T_cold = 0", labelgap=0.55)
    # north & south adiabatic no-slip walls
    wall!(ax, (0.0, 5.0), (5.0, 5.0); side=1,  label="adiabatic (∂ₙT=0)",
          labelgap=0.45)
    wall!(ax, (0.0, 0.0), (5.0, 0.0); side=-1, label="adiabatic (∂ₙT=0)",
          labelgap=0.45)
    # gravity cue (secondary grey), placed centre, clear of walls
    gravity!(ax, 2.5, 3.1; label="g", len=1.1)
    _emit(fig, "thermal-natural-convection-bc")
end

# ===========================================================================
# 3. Sphere drag 3D — duct mid-plane: inlet W, pressure outlet E, N/S no-slip,
#    sphere obstacle. Wide duct aspect (120×60).
# ===========================================================================
let
    fig = Figure(size=(720, 380), backgroundcolor=BC_DARK)
    # asymmetric room: a wide left margin so the long "u_in (velocity)" inlet
    # label clears the frame, and enough right margin for the outflow arrowheads.
    ax = bc_axis(fig[1, 1]; title="Sphere drag (3D) — duct mid-plane BCs")
    ax.limits = (-34, 140, -18, 78)
    fluid_region!(ax, (0.0, 0.0), (120.0, 60.0); outline=false)
    # north & south no-slip duct walls
    wall!(ax, (0.0, 60.0), (120.0, 60.0); side=1,  label="no-slip wall",
          labelgap=6.5)
    wall!(ax, (0.0, 0.0), (120.0, 0.0); side=-1, label="no-slip wall",
          labelgap=6.5)
    # inlet west, arrows INTO domain (+x)
    inlet!(ax, (0.0, 6.0), (0.0, 54.0); profile=:uniform, label="u_in (velocity)",
           depth=22.0, n=5, side=1, labelsize=12, labelgap=11.0)
    # pressure outlet east, arrows leaving the domain (+x)
    outlet!(ax, (120.0, 6.0), (120.0, 54.0); label="pressure outlet ρ=1", side=-1,
            n=4, depth=20.0, labelsize=12, labelgap=26.0)
    # sphere obstacle on axis
    obstacle!(ax, (38.0, 30.0), 8.0; label="sphere (STL, libb)", labelside=:below,
              labelgap=2.0)
    _emit(fig, "sphere-drag-3d-bc")
end

# ===========================================================================
# 4. Viscoelastic cylinder — inlet W (parabolic), outlet E, top/bottom no-slip,
#    cylinder obstacle on the channel axis.
# ===========================================================================
let
    fig = Figure(size=(700, 380), backgroundcolor=BC_DARK)
    ax = bc_axis(fig[1, 1]; title="Oldroyd-B flow past a confined cylinder — boundary conditions",
                 limits=(0, 120, 0, 50), pad=18.0)
    fluid_region!(ax, (0.0, 0.0), (120.0, 50.0); outline=false)
    wall!(ax, (0.0, 50.0), (120.0, 50.0); side=1,  label="no-slip, half-way bounce-back",
          labelgap=5.5)
    wall!(ax, (0.0, 0.0), (120.0, 0.0); side=-1, label="no-slip, half-way bounce-back",
          labelgap=5.5)
    # parabolic inlet on the west edge (channel flow), arrows INTO domain (+x)
    inlet!(ax, (0.0, 5.0), (0.0, 45.0); profile=:parabolic, label="u_in",
           depth=24.0, n=7, side=1, labelsize=12, labelgap=11.0)
    outlet!(ax, (120.0, 5.0), (120.0, 45.0); label="outflow", side=-1, n=4,
            depth=18.0, labelsize=12, labelgap=24.0)
    obstacle!(ax, (42.0, 25.0), 8.0; label="cylinder, R", labelside=:below,
              labelgap=2.0)
    _emit(fig, "viscoelastic-cylinder-bc")
end

println("\n=== 4 tutorial BC schematics generated ===")
