"""
    run_viscoelastic_logfv_bfs_coupled_2d(; kwargs...)

Run a coarse coupled log-FV polymer canary on a backward-facing-step geometry.

This is an open-x `StepChannelGeometry2D` path with feedback:

```text
LBM u -> open-x solid-aware log-FV polymer step
      -> tau_p -> div(tau_p) + BSD -> LI-BB V2 Guo-field solvent step
```

The dynamic outlet profiles are copied on device, so no host copy is needed
inside the time loop.
"""
function run_viscoelastic_logfv_bfs_coupled_2d(;
    H_in::Integer=4,
    expansion_ratio::Integer=2,
    L_up::Integer=2,
    L_down::Integer=4,
    kwargs...,
)
    H_in >= 3 || throw(ArgumentError("H_in must be >= 3"))
    expansion_ratio >= 2 || throw(ArgumentError("expansion_ratio must be >= 2"))
    T = get(kwargs, :T, Float64)
    geom_h = backward_facing_step_geometry_2d(;
        H_in=Int(H_in),
        expansion_ratio=Int(expansion_ratio),
        L_up=Int(L_up),
        L_down=Int(L_down),
        FT=T,
    )
    return _run_viscoelastic_logfv_step_channel_coupled_2d(
        geom_h; shear_length=H_in, kwargs...,
    )
end

"""
    run_viscoelastic_logfv_contraction_coupled_2d(; kwargs...)

Run the open-x coupled log-FV polymer path on a symmetric axis-aligned
contraction geometry.

This uses the same `StepChannelGeometry2D` core as the BFS and square-channel
drivers:

```text
LBM u -> open-x solid-aware log-FV polymer step
      -> tau_p -> div(tau_p) + BSD -> LI-BB V2 Guo-field solvent step
```
"""
function run_viscoelastic_logfv_contraction_coupled_2d(;
    H_out::Integer=4,
    β_c::Integer=4,
    L_up::Integer=4,
    L_down::Integer=4,
    kwargs...,
)
    H_out >= 3 || throw(ArgumentError("H_out must be >= 3"))
    β_c >= 2 || throw(ArgumentError("β_c must be >= 2"))
    T = get(kwargs, :T, Float64)
    geom_h = contraction_step_geometry_2d(;
        H_out=Int(H_out),
        β_c=Int(β_c),
        L_up=Int(L_up),
        L_down=Int(L_down),
        FT=T,
    )
    return _run_viscoelastic_logfv_step_channel_coupled_2d(
        geom_h; shear_length=H_out, kwargs...,
    )
end


"""
    run_viscoelastic_logfv_square_channel_coupled_2d(; kwargs...)

Run the same open-x coupled log-FV polymer path on a centered square obstacle
channel. This is the Cartesian-obstacle macro canary between periodic square
tests and curved cylinder validation.
"""
function run_viscoelastic_logfv_square_channel_coupled_2d(;
    H::Integer=12,
    side::Integer=4,
    L_up::Integer=2,
    L_down::Integer=3,
    kwargs...,
)
    H >= side + 4 || throw(ArgumentError("H must leave at least two fluid rows on each side"))
    side >= 2 || throw(ArgumentError("side must be >= 2"))
    T = get(kwargs, :T, Float64)
    geom_h = square_obstacle_channel_geometry_2d(;
        H=Int(H),
        side=Int(side),
        L_up=Int(L_up),
        L_down=Int(L_down),
        FT=T,
    )
    return _run_viscoelastic_logfv_step_channel_coupled_2d(
        geom_h; shear_length=H, kwargs...,
    )
end

function _logfv_cylinder_channel_geometry_2d(;
    radius::Real=6,
    H::Integer=max(ceil(Int, 4 * radius), ceil(Int, 2 * radius + 4)),
    L_up::Real=4,
    L_down::Real=8,
    FT::Type{<:AbstractFloat}=Float64,
)
    radius > 1 || throw(ArgumentError("radius must be > 1"))
    H >= ceil(Int, 2 * radius + 4) ||
        throw(ArgumentError("H must leave at least two fluid rows around the cylinder"))
    L_up > 1 || throw(ArgumentError("L_up must leave upstream clearance"))
    L_down > 1 || throw(ArgumentError("L_down must leave downstream clearance"))

    Nx = ceil(Int, (L_up + L_down) * radius)
    Ny = Int(H)
    cx = FT(L_up * radius)
    cy = FT((Ny - 1) / 2)
    q_wall, is_solid = precompute_q_wall_cylinder(Nx, Ny, cx, cy, radius; FT=FT)
    D = max(1, round(Int, 2 * radius))

    hydro_mask = fill(false, Ny)
    if Ny > 2
        hydro_mask[2:(Ny - 1)] .= true
    end
    conformation_mask = fill(true, Ny)

    return StepChannelGeometry2D{FT,Array{FT,3},Matrix{Bool},Vector{Bool}}(
        :cylinder,
        Nx,
        Ny,
        round(Int, cx) + 1,
        1:Ny,
        1:Ny,
        D,
        Ny,
        Ny,
        q_wall,
        Matrix{Bool}(is_solid),
        hydro_mask,
        copy(hydro_mask),
        conformation_mask,
        copy(conformation_mask),
    )
end

"""
    run_viscoelastic_logfv_cylinder_coupled_2d(; kwargs...)

Run the open-x coupled log-FV polymer path on a circular cylinder with
precomputed cut-link geometry. This is the curved-wall macro canary above BFS
and square-obstacle tests; benchmark Cd convergence still belongs in a
separate harness after the lower ladder is green.

# Stop option

All keywords are forwarded to the shared coupled driver. Off by default.

- `stop=:none`: run exactly `max_steps` steps (default). `stop=:cd_change`: sample the
  total drag coefficient every `check_every` steps and stop when
  `|Cd_k - Cd_(k-1)| <= tol * |Cd_k|`. `max_steps` stays the hard cap.
- `tol=2e-4`: relative tolerance of that rule.
- `check_every=nothing`: steps between samples; default `round(Int, Nx / u_mean)`.
- `step_callback(step, fields)`: a callback that returns exactly `true` stops the run.

The result gains `converged`, `stop_reason` (`:rule`, `:callback` or `:max_steps`),
`stop_step`, `cd_history` (vector of `(step, Cd)`) and `Cd_last`. `Cd` stays the mean over
the last `avg_window` steps; a run stopped before that window reports the last sample.
"""
function run_viscoelastic_logfv_cylinder_coupled_2d(;
    radius::Real=6,
    H::Integer=max(ceil(Int, 4 * radius), ceil(Int, 2 * radius + 4)),
    L_up::Real=4,
    L_down::Real=8,
    kwargs...,
)
    T = get(kwargs, :T, Float64)
    geom_h = _logfv_cylinder_channel_geometry_2d(;
        radius,
        H,
        L_up,
        L_down,
        FT=T,
    )
    return _run_viscoelastic_logfv_step_channel_coupled_2d(
        geom_h;
        shear_length=H,
        drag_cx=Float64(L_up * radius),
        drag_cy=Float64((H - 1) / 2),
        drag_radius=Float64(radius),
        drag_u_ref=Float64(get(kwargs, :u_mean, 0.01)),
        kwargs...,
    )
end

