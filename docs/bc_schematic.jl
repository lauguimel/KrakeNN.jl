# bc_schematic.jl — reusable boundary-condition / geometry schematic primitives
# =============================================================================
# A pure-CairoMakie drawing toolkit for the KrakeNN docs. NO KrakeNN dependency —
# just `include("bc_schematic.jl")` and assemble any of the example/tutorial
# geometry schematics from composable primitives.
#
# VISUAL LANGUAGE (the contract every schematic shares)
# -----------------------------------------------------
#   • Page / axis background : #1b1b1f  (the live Vitepress dark page colour).
#   • Serif, light text on dark — title gray92, labels gray80.
#   • HATCHING (muted gray) = a solid no-slip wall (fixed boundary). The baseline
#     is the wall face; the short diagonal strokes sit on the SOLID side.
#   • ACCENT  #ff6b6b  = the DRIVING condition: moving wall velocity, body force /
#     pressure gradient. Also the HOT temperature Dirichlet wall.
#   • COOL    #4ea1d3  = fluid in/out & boundary topology: inlet/outlet arrows,
#     periodic dashed edges, symmetry dash-dot axis. Also the COLD Dirichlet wall.
#   • DASHED cool edge + paired chevrons (⇄) = periodic. DASH-DOT cool line = symmetry.
#   • The fluid interior is a very faint gray fill (alpha≈0.04) inside a gray55 outline.
# A compact dark-framed legend maps each style back to its BC name.
# =============================================================================

using CairoMakie

# --- Theme constants --------------------------------------------------------
const BC_DARK    = "#1b1b1f"   # page / axis background (Vitepress --vp-c-bg)
const BC_ACCENT  = "#ff6b6b"   # driving condition: moving wall, body force, HOT wall
const BC_COOL    = "#4ea1d3"   # in/out, periodic, symmetry, COLD wall
const BC_WALL    = "gray65"    # no-slip wall baseline + hatch
const BC_OUTLINE = "gray55"    # fluid-region outline
const BC_FLUID   = (:gray, 0.04)  # very faint fluid fill
const BC_TEXT    = "gray80"    # labels
const BC_TITLE   = "gray92"    # titles

# Serif identity matching the mpl `font.family: serif` contract. Makie only bundles
# the sans "TeX Gyre Heros", so we ask Fontconfig for the generic "serif" family —
# it resolves to a real serif face per platform (PT Serif on macOS, DejaVu/Liberation
# Serif on Linux CI). Portable and always serif, never silently sans.
const BC_SERIF = "serif"

# Small helper: 2D vector length.
_veclen(dx, dy) = sqrt(dx^2 + dy^2)

"""
    bc_axis(fig_or_pos; title="", limits=nothing, pad=0.0)

Return an `Axis` pre-configured for BC schematics: `#1b1b1f` background,
`DataAspect()`, all decorations/spines hidden, light serif title. `limits` is a
`(xmin, xmax, ymin, ymax)` tuple; `pad` (data units) optionally enlarges the
limits symmetrically so labels never clip the frame.
"""
function bc_axis(fig_or_pos; title="", limits=nothing, pad=0.0)
    ax = Axis(fig_or_pos;
        title = title,
        titlefont = BC_SERIF,
        titlesize = 17,
        titlecolor = BC_TITLE,
        backgroundcolor = BC_DARK,
        aspect = DataAspect(),
    )
    if limits !== nothing
        xmin, xmax, ymin, ymax = limits
        ax.limits = (xmin - pad, xmax + pad, ymin - pad, ymax + pad)
    end
    hidespines!(ax)
    hidedecorations!(ax)
    return ax
end

# Internal: draw hatch strokes along baseline p1->p2, offset to the SOLID side.
#   `side` = +1 puts hatching on the left of the p1->p2 direction, -1 on the right.
#   `len` is wall length; hatch spacing/size scale with it so short and long walls
#   look consistent (≈ every 0.45 data units, clamped to 6–24 strokes).
function _hatch!(ax, p1, p2; color=BC_WALL, side=1, depth=nothing, lw=1.2)
    x1, y1 = p1; x2, y2 = p2
    L = _veclen(x2 - x1, y2 - y1)
    L == 0 && return
    # unit tangent & inward-solid normal
    tx, ty = (x2 - x1) / L, (y2 - y1) / L
    nx, ny = -ty * side, tx * side          # normal toward solid side
    d = depth === nothing ? clamp(0.06 * L, 0.12, 0.35) : depth
    spacing = clamp(L / 12, 0.18, 0.5)
    n = clamp(round(Int, L / spacing), 6, 28)
    # diagonal stroke: each tick goes from the baseline back into the solid,
    # slanted by the tangent so it reads as the classic 45°-ish hatch.
    xs = Float64[]; ys = Float64[]
    for i in 0:n
        s = i / n * L
        bx, by = x1 + tx * s, y1 + ty * s              # foot on baseline
        # slant the tip backward along -tangent by ~depth for the 45° look
        tipx = bx + nx * d - tx * d
        tipy = by + ny * d - ty * d
        push!(xs, bx, tipx, NaN)
        push!(ys, by, tipy, NaN)
    end
    lines!(ax, xs, ys; color=color, linewidth=lw)
    return (nx, ny, d)   # outward-solid normal & depth (for label placement)
end

# Internal: place a label clear of a wall, on the chosen side. `pos` is the
# fractional position along p1→p2 (0=start, 0.5=middle, 1=end) so callers can
# slide a label sideways to dodge other annotations.
function _wall_label!(ax, p1, p2, label; color=BC_TEXT, side=1, gap=0.55,
                      fontsize=12, pos=0.5)
    label === nothing && return
    x1, y1 = p1; x2, y2 = p2
    L = _veclen(x2 - x1, y2 - y1)
    tx, ty = (x2 - x1) / L, (y2 - y1) / L
    ax_, ay_ = x1 + tx * L * pos, y1 + ty * L * pos   # anchor along the wall
    nx, ny = -ty * side, tx * side
    # label sits on the OPPOSITE side of the hatching (fluid side) by default
    lx, ly = ax_ - nx * gap, ay_ - ny * gap
    # align so text grows away from the wall
    halign = abs(nx) > abs(ny) ? (nx < 0 ? :right : :left) : :center
    valign = abs(ny) >= abs(nx) ? (ny < 0 ? :top : :bottom) : :center
    text!(ax, lx, ly; text=label, color=color, font=BC_SERIF, fontsize=fontsize,
          align=(halign, valign))
end

"""
    wall!(ax, p1, p2; label=nothing, side=1, labelside=:out, color=BC_WALL)

No-slip wall (bounce-back): a baseline `p1→p2` plus evenly spaced diagonal hatch
strokes on the solid side. `side` (+1/−1) chooses which side the solid (hatching)
is on relative to the `p1→p2` direction. Optional light-serif `label`; `labelside`
`:out` places it on the fluid side (default), `:in` on the solid side; `labelpos`
(0..1) slides the label along the wall to dodge other annotations; `labelgap` is
its perpendicular clearance.
"""
function wall!(ax, p1, p2; label=nothing, side=1, labelside=:out, color=BC_WALL,
               lw=2.0, labelsize=12, labelgap=0.6, labelpos=0.5)
    lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]]; color=color, linewidth=lw)
    _hatch!(ax, p1, p2; color=color, side=side)
    if label !== nothing
        lside = labelside === :out ? side : -side
        _wall_label!(ax, p1, p2, label; color=BC_TEXT, side=lside, gap=labelgap,
                     fontsize=labelsize, pos=labelpos)
    end
    return ax
end

"""
    moving_wall!(ax, p1, p2; u_label="u", label=nothing, side=1, narrows=3)

A MOVING wall: a grey DASHED baseline along `p1→p2` (NO hatching — that's reserved
for fixed no-slip walls) with the DRIVING velocity drawn as ACCENT arrows
superimposed along the face (in the `p1→p2` direction). `u_label` (accent)
annotates the velocity and sits on the fluid side, clear of the arrows; `label`
names the wall and sits on the solid side. Use for Couette moving wall / cavity
lid (Zou-He).
"""
function moving_wall!(ax, p1, p2; u_label="u", label=nothing, side=1, narrows=3,
                      labelsize=12, labelgap=0.7, ulabelgap=0.7)
    # grey DASHED baseline — a moving wall is NOT hatched (hatch == fixed no-slip)
    lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]];
           color="gray60", linewidth=2.0, linestyle=:dash)
    x1, y1 = p1; x2, y2 = p2
    L = _veclen(x2 - x1, y2 - y1)
    tx, ty = (x2 - x1) / L, (y2 - y1) / L
    nx, ny = -ty * side, tx * side                 # toward solid
    # arrows are SUPERIMPOSED directly on the dashed baseline (no hatch to dodge)
    off = 0.0
    xs = Float64[]; ys = Float64[]; us = Float64[]; vs = Float64[]
    alen = clamp(0.5 * L / narrows, 0.5, 1.6)
    for i in 1:narrows
        s = (i - 0.5) / narrows * L - alen / 2
        bx = x1 + tx * s + nx * off
        by = y1 + ty * s + ny * off
        push!(xs, bx); push!(ys, by); push!(us, tx * alen); push!(vs, ty * alen)
    end
    arrows2d!(ax, xs, ys, us, vs; color=BC_ACCENT, shaftwidth=2.5,
              tipwidth=11, tiplength=11)
    # u-label on the FLUID side, pushed clear of the arrow row
    mx, my = x1 + tx * L * 0.5, y1 + ty * L * 0.5
    text!(ax, mx - nx * ulabelgap, my - ny * ulabelgap;
          text=u_label, color=BC_ACCENT, font=BC_SERIF, fontsize=labelsize + 1,
          align=(:center, :center))
    # wall name on the SOLID side (behind the hatch), never under the arrows.
    # `_wall_label!` places on the fluid side for its own `side`, so pass −side to
    # land on the solid side.
    if label !== nothing
        _wall_label!(ax, p1, p2, label; color=BC_TEXT, side=-side, gap=labelgap,
                     fontsize=labelsize)
    end
    return ax
end

"""
    body_force!(ax, x, y; dx=1.0, dy=0.0, label=nothing, n=1, spread=0.0)

ACCENT arrow(s) for a body force / pressure gradient, rooted at `(x, y)`,
pointing `(dx, dy)`. `n>1` stacks `n` parallel arrows `spread` apart (handy for a
column of force arrows). Optional accent `label`.
"""
function body_force!(ax, x, y; dx=1.0, dy=0.0, label=nothing, n=1, spread=0.0,
                     labelside=-1, labelsize=12, labelgap=0.55)
    L = _veclen(dx, dy)
    # perpendicular for stacking
    px, py = L == 0 ? (0.0, 0.0) : (-dy / L, dx / L)
    xs = Float64[]; ys = Float64[]; us = Float64[]; vs = Float64[]
    for i in 1:n
        o = (i - (n + 1) / 2) * spread
        push!(xs, x + px * o); push!(ys, y + py * o)
        push!(us, dx); push!(vs, dy)
    end
    arrows2d!(ax, xs, ys, us, vs; color=BC_ACCENT, shaftwidth=3.0,
              tipwidth=14, tiplength=14)
    if label !== nothing
        # Label sits clear of the WHOLE stack: centred on the arrows along the
        # flow direction, offset half the stack span + a fixed gap on the chosen
        # `labelside` (+1/−1 of the perpendicular; default −1 = "below" a →stack).
        half = spread * (n - 1) / 2
        offp = labelside * (half + labelgap)
        lx = x + dx * 0.5 + px * offp
        ly = y + dy * 0.5 + py * offp
        text!(ax, lx, ly; text=label, color=BC_ACCENT, font=BC_SERIF,
              fontsize=labelsize, align=(:center, :center))
    end
    return ax
end

"""
    inlet!(ax, p1, p2; profile=:uniform, label=nothing, depth=1.4, n=5, side=1)

A row of COOL arrows ENTERING the domain across the edge `p1→p2`, pointing INTO
the interior. `side` selects the interior side (same convention as `wall!`: the
fluid is on the `-side` normal); the arrows point that way, into the box.
`profile` `:uniform` draws equal-length arrows; `:parabolic` shapes them into a
Poiseuille velocity profile (zero at the ends, max at centre). `depth` is the max
arrow length. Optional cool `label`; `labelgap` (data units) is the perpendicular
clearance from the edge to the label — raise it on large domains so the label
clears the inflow arrows.
"""
function inlet!(ax, p1, p2; profile=:uniform, label=nothing, depth=1.4, n=5, side=1,
                labelsize=12, labelgap=0.7)
    x1, y1 = p1; x2, y2 = p2
    L = _veclen(x2 - x1, y2 - y1)
    tx, ty = (x2 - x1) / L, (y2 - y1) / L
    # `-side` normal points INTO the interior (matches wall!'s fluid side).
    nx, ny = ty * side, -tx * side                 # into fluid interior
    xs = Float64[]; ys = Float64[]; us = Float64[]; vs = Float64[]
    for i in 1:n
        f = (i - 0.5) / n            # 0..1 along the edge
        bx, by = x1 + tx * f * L, y1 + ty * f * L
        mag = profile === :parabolic ? depth * 4 * f * (1 - f) : depth
        mag = max(mag, 0.04 * depth)  # keep a nub even at parabola ends
        # arrow tail on the edge, points inward into the interior
        push!(xs, bx); push!(ys, by)
        push!(us, nx * mag); push!(vs, ny * mag)
    end
    arrows2d!(ax, xs, ys, us, vs; color=BC_COOL, shaftwidth=2.0,
              tipwidth=9, tiplength=9)
    # connect the parabola arrow tips with a faint cool envelope curve
    if profile === :parabolic
        ex = Float64[]; ey = Float64[]
        for i in 1:n
            f = (i - 0.5) / n
            bx, by = x1 + tx * f * L, y1 + ty * f * L
            mag = max(depth * 4 * f * (1 - f), 0.04 * depth)
            push!(ex, bx + nx * mag); push!(ey, by + ny * mag)
        end
        lines!(ax, ex, ey; color=(BC_COOL, 0.5), linewidth=1.3, linestyle=:dash)
    end
    if label !== nothing
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        # label on the exterior side (−n), clear of the inward arrows
        text!(ax, mx - nx * labelgap, my - ny * labelgap; text=label, color=BC_COOL,
              font=BC_SERIF, fontsize=labelsize, align=(:center, :center))
    end
    return ax
end

"""
    outlet!(ax, p1, p2; label="outflow ∂ₙu=0", side=1, n=4)

Outflow marker on edge `p1→p2`: a row of COOL arrows pointing OUTWARD, leaving the
domain (mirror of `inlet!`). `side` selects the interior side (same convention as
`wall!`/`inlet!`); arrows point from the interior toward the exterior. Keeps the
cool `outflow ∂ₙu=0` label; `labelgap` (data units) is the perpendicular clearance
from the edge to the label (raise it on large domains). Use for the
cylinder/channel outlet.
"""
function outlet!(ax, p1, p2; label="outflow ∂ₙu=0", side=1, n=4, depth=1.1,
                 labelsize=12, labelgap=0.7)
    x1, y1 = p1; x2, y2 = p2
    L = _veclen(x2 - x1, y2 - y1)
    tx, ty = (x2 - x1) / L, (y2 - y1) / L
    # exterior normal is `+side` (opposite of inlet's interior normal)
    nx, ny = -ty * side, tx * side                 # toward exterior (outward)
    xs = Float64[]; ys = Float64[]; us = Float64[]; vs = Float64[]
    for i in 1:n
        f = (i - 0.5) / n
        bx, by = x1 + tx * f * L, y1 + ty * f * L
        # tail just inside the edge, arrow leaves the domain
        push!(xs, bx - nx * depth * 0.45); push!(ys, by - ny * depth * 0.45)
        push!(us, nx * depth); push!(vs, ny * depth)
    end
    arrows2d!(ax, xs, ys, us, vs; color=BC_COOL, shaftwidth=2.0,
              tipwidth=9, tiplength=9)
    if label !== nothing
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        # label on the interior side (−n), clear of the outgoing arrows
        text!(ax, mx - nx * labelgap, my - ny * labelgap; text=label, color=BC_COOL,
              font=BC_SERIF, fontsize=labelsize, align=(:center, :center))
    end
    return ax
end

"""
    periodic!(ax, p1, p2; label="periodic", labelrot=nothing, side=nothing)

A periodic edge: a COOL DASHED segment `p1→p2` plus a small paired chevron pair
marking the wrap, and a label rotated along the edge. The chevrons point along the
edge's OUTWARD normal — out of the fluid zone, away from the interior — so a left
edge reads `<`, a right edge `>`, a top edge `∧`, a bottom edge `∨`.

`side` selects which side the fluid INTERIOR is on relative to the `p1→p2`
direction (`wall!`/`inlet!` convention): the OUTWARD (exterior) normal the chevrons
follow is `side·(-ty, tx)`, so `side=+1` points the chevrons along `(-ty, tx)` and
`side=-1` along `(ty, -tx)`. When `side === nothing` (default) the outward normal is
INFERRED from the axis limits: the chevrons point from the domain centre toward the
edge (i.e. away from the interior), so a left edge gets `<` and a right edge `>`
automatically — no per-call wiring needed. Pass an explicit `side` to override.
Draw on both matched edges (e.g. left & right) to show periodicity.
"""
# Internal: a small ">" chevron arrowhead centred at (cx,cy) whose tip points
# along the unit direction (dx,dy); `c` is the chevron arm length.
function _chevron!(ax, cx, cy, dx, dy, c; color=BC_COOL, lw=1.8)
    px, py = -dy, dx                       # perpendicular
    tipx, tipy = cx + dx * c, cy + dy * c  # tip ahead of centre
    a1x, a1y = cx + px * c * 0.8, cy + py * c * 0.8     # one back-arm root
    a2x, a2y = cx - px * c * 0.8, cy - py * c * 0.8     # other back-arm root
    lines!(ax, [a1x, tipx, a2x], [a1y, tipy, a2y]; color=color, linewidth=lw)
end

# Infer the outward sign for a periodic edge from the axis limits: the outward
# normal must point AWAY from the domain centre (out of the fluid). Project the
# vector (edge-midpoint − domain-centre) onto the base normal (-ty, tx); the sign of
# that projection is the `side` that makes `side·(-ty, tx)` point outward. Falls back
# to +1 if limits are unavailable or the edge sits on the centreline.
function _periodic_outward_side(ax, mx, my, tx, ty)
    lim = try
        ax.limits[]
    catch
        nothing
    end
    (lim === nothing || any(l -> l === nothing, lim)) && return 1
    xmin, xmax, ymin, ymax = lim
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    bnx, bny = -ty, tx                        # base normal (side = +1)
    proj = (mx - cx) * bnx + (my - cy) * bny  # >0 ⇒ base normal already points out
    return proj >= 0 ? 1 : -1
end

function periodic!(ax, p1, p2; label="periodic", labelrot=nothing, labelsize=11,
                   side=nothing)
    x1, y1 = p1; x2, y2 = p2
    lines!(ax, [x1, x2], [y1, y2]; color=BC_COOL, linewidth=2.0, linestyle=:dash)
    L = _veclen(x2 - x1, y2 - y1)
    tx, ty = (x2 - x1) / L, (y2 - y1) / L          # along edge
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    # OUTWARD normal: exterior side, away from the fluid interior. `side` encodes the
    # interior side (wall!/inlet! convention); when not given, infer it from the axis
    # limits so the chevrons always point OUT of the domain. `side·(-ty, tx)` is the
    # outward normal: left edge `<`, right edge `>`, top edge `∧`, bottom edge `∨`.
    s = side === nothing ? _periodic_outward_side(ax, mx, my, tx, ty) : side
    nx, ny = -ty * s, tx * s
    # paired chevrons along the OUTWARD normal: two arrowheads BOTH pointing OUT of
    # the fluid (the outward-normal direction), stacked along the edge (±g in
    # tangent), so they read as "this edge wraps to its partner" and clearly mark
    # the exterior side.
    c = clamp(0.09 * L, 0.13, 0.28)
    g = 1.6 * c
    _chevron!(ax, mx + tx * g, my + ty * g, nx, ny, c; color=BC_COOL)   # upper, tip outward
    _chevron!(ax, mx - tx * g, my - ty * g, nx, ny, c; color=BC_COOL)   # lower, tip outward
    if label !== nothing
        rot = labelrot === nothing ? atan(ty, tx) : labelrot
        # keep text upright-ish
        if rot > pi/2; rot -= pi; elseif rot < -pi/2; rot += pi; end
        # label sits on the EXTERIOR (outward) side, same as the original render —
        # outside the fluid, along the outward normal, just past the chevrons.
        text!(ax, mx + nx * 0.55, my + ny * 0.55; text=label, color=BC_COOL,
              font=BC_SERIF, fontsize=labelsize, rotation=rot,
              align=(:center, :center))
    end
    return ax
end

"""
    symmetry!(ax, p1, p2; label="symmetry")

A symmetry axis: a COOL DASH-DOT centre line `p1→p2` with a label along it. Use
for the Hagen-Poiseuille r=0 axis.
"""
function symmetry!(ax, p1, p2; label="symmetry", labelsize=11)
    x1, y1 = p1; x2, y2 = p2
    lines!(ax, [x1, x2], [y1, y2]; color=BC_COOL, linewidth=1.8, linestyle=:dashdot)
    if label !== nothing
        L = _veclen(x2 - x1, y2 - y1)
        tx, ty = (x2 - x1) / L, (y2 - y1) / L
        nx, ny = -ty, tx
        rot = atan(ty, tx)
        if rot > pi/2; rot -= pi; elseif rot < -pi/2; rot += pi; end
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        text!(ax, mx + nx * 0.45, my + ny * 0.45; text=label, color=BC_COOL,
              font=BC_SERIF, fontsize=labelsize, rotation=rot,
              align=(:center, :center))
    end
    return ax
end

"""
    dirichlet_wall!(ax, p1, p2; kind=:hot, label=nothing, side=1, band=0.18)

A temperature Dirichlet wall on `p1→p2` — i.e. a NO-SLIP wall held at temperature
T, so it KEEPS the diagonal hatching (hatch == no-slip), coloured by kind:
`:hot` ⇒ ACCENT `#ff6b6b`, `:cold` ⇒ COOL `#4ea1d3`. The hatch + baseline take the
kind colour so it reads as "a no-slip wall held at T". `side` is the solid side;
`band` is the coloured-band half-thickness.
"""
function dirichlet_wall!(ax, p1, p2; kind=:hot, label=nothing, side=1, band=0.16,
                         labelsize=12, labelgap=0.65)
    col = kind === :hot ? BC_ACCENT : BC_COOL
    x1, y1 = p1; x2, y2 = p2
    L = _veclen(x2 - x1, y2 - y1)
    tx, ty = (x2 - x1) / L, (y2 - y1) / L
    nx, ny = -ty * side, tx * side                 # into solid
    # coloured band as a poly (baseline -> offset into solid)
    bx1, by1 = x1, y1
    bx2, by2 = x2, y2
    bx3, by3 = x2 + nx * band, y2 + ny * band
    bx4, by4 = x1 + nx * band, y1 + ny * band
    poly!(ax, Point2f[(bx1, by1), (bx2, by2), (bx3, by3), (bx4, by4)];
          color=(col, 0.55), strokewidth=0)
    lines!(ax, [x1, x2], [y1, y2]; color=col, linewidth=2.2)
    _hatch!(ax, p1, p2; color=col, side=side, depth=band + 0.12, lw=1.1)
    if label !== nothing
        _wall_label!(ax, p1, p2, label; color=col, side=-side, gap=labelgap,
                     fontsize=labelsize)
    end
    return ax
end

"""
    fluid_region!(ax, p1, p2; outline=true)

Subtle fluid interior over the rectangle with opposite corners `p1`, `p2`: a very
faint gray fill (alpha≈0.04) and an optional gray55 outline, so the interior
reads as "fluid". (Alias: `domain!`.)
"""
function fluid_region!(ax, p1, p2; outline=true)
    x1, y1 = p1; x2, y2 = p2
    poly!(ax, Point2f[(x1, y1), (x2, y1), (x2, y2), (x1, y2)];
          color=BC_FLUID, strokewidth=0)
    if outline
        lines!(ax, [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1];
               color=BC_OUTLINE, linewidth=1.0)
    end
    return ax
end
const domain! = fluid_region!

"""
    obstacle!(ax, center, R; label=nothing, labelside=:below, labelgap=0.0)

An immersed solid (cylinder / sphere cross-section) — the bounce-back / STL
obstacle. A filled muted-grey disc (`gray45`) with a thin lighter outline, so it
reads as a clean solid body distinct from the hatched no-slip walls. `center` is
`(cx, cy)`, `R` the radius. Optional serif `label`; `labelside` ∈
`(:below, :above, :right, :left)` places it clear of the disc, `labelgap` adds
extra clearance beyond the radius.
"""
function obstacle!(ax, center, R; label=nothing, labelside=:below, labelgap=0.0,
                   labelsize=12)
    cx, cy = center
    θ = range(0, 2π; length=80)
    poly!(ax, Point2f[(cx + R * cos(t), cy + R * sin(t)) for t in θ];
          color=("gray45", 0.95), strokecolor="gray70", strokewidth=1.4)
    if label !== nothing
        gap = R + 0.45 + labelgap
        lx, ly, ha, va = if labelside === :above
            (cx, cy + gap, :center, :bottom)
        elseif labelside === :right
            (cx + gap, cy, :left, :center)
        elseif labelside === :left
            (cx - gap, cy, :right, :center)
        else  # :below
            (cx, cy - gap, :center, :top)
        end
        text!(ax, lx, ly; text=label, color=BC_TEXT, font=BC_SERIF,
              fontsize=labelsize, align=(ha, va))
    end
    return ax
end

"""
    free_slip!(ax, p1, p2; label="free-slip", labelside=1, labelgap=0.55)

A free-slip (slip / symmetry-type) wall: a PLAIN solid grey line `p1→p2` — NO
hatching, since hatching is reserved for no-slip — so it reads as a distinct
boundary. Optional serif `label`; `labelside` (+1/−1 of the `p1→p2` normal) picks
which side the label sits on, `labelgap` its perpendicular clearance.
"""
function free_slip!(ax, p1, p2; label="free-slip", labelside=1, labelgap=0.55,
                    labelsize=12, lw=2.2)
    lines!(ax, [p1[1], p2[1]], [p1[2], p2[2]]; color="gray75", linewidth=lw)
    if label !== nothing
        _wall_label!(ax, p1, p2, label; color=BC_TEXT, side=labelside, gap=labelgap,
                     fontsize=labelsize)
    end
    return ax
end

"""
    gravity!(ax, x, y; label="g", len=1.0)

A small NEUTRAL-grey downward arrow at `(x, y)` marking the gravity / buoyancy
direction. Kept visually secondary (neither accent nor cool, both of which are
reserved BC cues) so it never competes with the boundary annotations. `len` is the
arrow length; optional grey `label`.
"""
function gravity!(ax, x, y; label="g", len=1.0, labelsize=12)
    arrows2d!(ax, [x], [y], [0.0], [-len]; color="gray70", shaftwidth=2.2,
              tipwidth=10, tiplength=10)
    if label !== nothing
        text!(ax, x + 0.32, y - len * 0.5; text=label, color=BC_TEXT,
              font=BC_SERIF, fontsize=labelsize + 1, align=(:left, :center))
    end
    return ax
end

"""
    bc_legend!(fig_or_pos; entries, title="boundary conditions")

A compact legend mapping styles/colours to BC names. `entries` is a vector of
`(style, label)` pairs where `style` ∈ `(:wall, :moving, :force, :inlet,
:outlet, :periodic, :symmetry, :hot, :cold, :fluid, :obstacle, :free_slip,
:gravity)`. Rendered in its own Axis
(`fig[i, j]` position) with a SINGLE semi-transparent dark frame (one rounded-ish
rectangle, no double border) and light serif text, so it never overlaps the
drawing. Returns the legend Axis.
"""
function bc_legend!(fig_or_pos; entries, title="boundary conditions", framealpha=0.55)
    # Transparent Axis: the frame is drawn ONCE as a single poly! below (no
    # axis-background rectangle stacked under a manual border → no double frame).
    lax = Axis(fig_or_pos; backgroundcolor=:transparent)
    hidedecorations!(lax)
    hidespines!(lax)
    n = length(entries)
    # compact row pitch; tight margins around the single frame
    row = 1.0
    top = n * row + 0.7      # title band
    lax.limits = (0, 10, -0.35, top + 0.35)
    # SINGLE frame: one filled semi-transparent dark rounded rectangle.
    poly!(lax, Point2f[(0.15, -0.2), (9.85, -0.2), (9.85, top), (0.15, top)];
          color=(BC_DARK, framealpha), strokecolor=("gray55", 0.7), strokewidth=1.0)
    text!(lax, 0.6, top - 0.4; text=title, color=BC_TITLE, font=BC_SERIF,
          fontsize=13, align=(:left, :center))
    for (i, (style, label)) in enumerate(entries)
        y = (n - i) * row + 0.45
        _legend_glyph!(lax, 0.55, y, style)
        text!(lax, 2.6, y; text=label, color=BC_TEXT, font=BC_SERIF, fontsize=11.5,
              align=(:left, :center))
    end
    return lax
end

# Internal: a small glyph (x..x+~1.6) at y illustrating each BC style.
function _legend_glyph!(lax, x, y, style)
    x2 = x + 1.6
    xm = (x + x2) / 2
    if style === :wall
        # hatched grey → no-slip wall
        lines!(lax, [x, x2], [y, y]; color=BC_WALL, linewidth=2.0)
        for f in 0:0.25:1
            bx = x + f * 1.6
            lines!(lax, [bx, bx - 0.18], [y, y - 0.26]; color=BC_WALL, linewidth=1.1)
        end
    elseif style === :moving
        # grey dashed baseline + red arrow → moving wall
        lines!(lax, [x, x2], [y - 0.22, y - 0.22]; color="gray60", linewidth=1.8,
               linestyle=:dash)
        arrows2d!(lax, [x], [y + 0.08], [1.3], [0.0]; color=BC_ACCENT,
                  shaftwidth=2.5, tipwidth=10, tiplength=9)
    elseif style === :force
        # red arrow → body force / ∇p
        arrows2d!(lax, [x], [y], [1.5], [0.0]; color=BC_ACCENT,
                  shaftwidth=3.0, tipwidth=12, tiplength=11)
    elseif style === :inlet
        # cool arrows pointing INWARD (→ into the domain)
        arrows2d!(lax, [x], [y + 0.22], [1.4], [0.0]; color=BC_COOL,
                  shaftwidth=2.0, tipwidth=8, tiplength=8)
        arrows2d!(lax, [x], [y - 0.22], [1.4], [0.0]; color=BC_COOL,
                  shaftwidth=2.0, tipwidth=8, tiplength=8)
    elseif style === :outlet
        # cool arrows pointing OUTWARD (mirror of inlet)
        arrows2d!(lax, [x + 0.2], [y + 0.22], [1.4], [0.0]; color=BC_COOL,
                  shaftwidth=2.0, tipwidth=8, tiplength=8)
        arrows2d!(lax, [x + 0.2], [y - 0.22], [1.4], [0.0]; color=BC_COOL,
                  shaftwidth=2.0, tipwidth=8, tiplength=8)
    elseif style === :periodic
        # cool dashed line ALONG + an OUTWARD `<` chevron pair (pointing out of the
        # fluid, away from the interior — left here, matching a left edge's `<`).
        lines!(lax, [x, x2], [y, y]; color=BC_COOL, linewidth=2.0, linestyle=:dash)
        _chevron!(lax, xm + 0.06, y, -1.0, 0.0, 0.22; color=BC_COOL, lw=1.6)
        _chevron!(lax, xm + 0.42, y, -1.0, 0.0, 0.22; color=BC_COOL, lw=1.6)
    elseif style === :symmetry
        # cool dash-dot
        lines!(lax, [x, x2], [y, y]; color=BC_COOL, linewidth=1.8, linestyle=:dashdot)
    elseif style === :hot
        # red hatch → no-slip wall, T_hot
        lines!(lax, [x, x2], [y, y]; color=BC_ACCENT, linewidth=2.2)
        for f in 0:0.25:1
            bx = x + f * 1.6
            lines!(lax, [bx, bx - 0.18], [y, y - 0.26]; color=BC_ACCENT, linewidth=1.1)
        end
    elseif style === :cold
        # blue hatch → no-slip wall, T_cold
        lines!(lax, [x, x2], [y, y]; color=BC_COOL, linewidth=2.2)
        for f in 0:0.25:1
            bx = x + f * 1.6
            lines!(lax, [bx, bx - 0.18], [y, y - 0.26]; color=BC_COOL, linewidth=1.1)
        end
    elseif style === :fluid
        poly!(lax, Point2f[(x, y-0.26), (x2, y-0.26), (x2, y+0.26), (x, y+0.26)];
              color=(:gray, 0.12), strokecolor=BC_OUTLINE, strokewidth=1.0)
    elseif style === :obstacle
        # filled muted-grey disc → immersed solid (cylinder/sphere). Drawn as a
        # pixel-space circle marker so it stays ROUND regardless of the (non-square)
        # legend-axis aspect.
        scatter!(lax, [xm], [y]; color=("gray45", 0.95), strokecolor="gray70",
                 strokewidth=1.2, markersize=20, marker=:circle)
    elseif style === :free_slip
        # plain solid grey line (NO hatch) → free-slip wall
        lines!(lax, [x, x2], [y, y]; color="gray75", linewidth=2.2)
    elseif style === :gravity
        # small neutral-grey downward arrow → gravity / buoyancy
        arrows2d!(lax, [xm], [y + 0.28], [0.0], [-0.56]; color="gray70",
                  shaftwidth=2.0, tipwidth=9, tiplength=9)
    end
end
