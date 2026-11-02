# Planar GNF channels: host geometry lowering and backend-generic stepping.

"""
    _gnf_axis_3d(axis) -> Int

Map `:x`, `:y`, `:z` to the array dimension 1, 2, 3. Any other symbol raises
`ArgumentError`.
"""
function _gnf_axis_3d(axis::Symbol)
    d = findfirst(==(axis), (:x, :y, :z))
    isnothing(d) && throw(ArgumentError("axis must be :x, :y, or :z"))
    return d
end

"""
    _gnf_profile_3d(profile, positions, T, name; default=zero(T)) -> Vector{T}

Evaluate a wall-normal profile at the fluid-node `positions`. `profile` may be
`nothing` (use `default`), a scalar, a length-`N` vector, or a function of the
distance from the lower wall. Non-finite values raise `ArgumentError`.
"""
function _gnf_profile_3d(profile, positions, ::Type{T}, name;
                         default=zero(T)) where {T}
    values = if isnothing(profile)
        fill(T(default), length(positions))
    elseif profile isa Real
        fill(T(profile), length(positions))
    elseif profile isa AbstractVector
        length(profile) == length(positions) ||
            throw(ArgumentError("$name must contain N fluid-node values"))
        T.(profile)
    else
        T[profile(y) for y in positions]
    end
    all(isfinite, values) || throw(ArgumentError("$name must be finite"))
    return values
end

"""
    _wrap_gnf_periodic_3d!(a, periodic_axes)

Copy the first/last interior planes into periodic ghosts using backend
broadcasts. Accepts both 3D fields and 4D population arrays. Sequential axis
copies also fill edge ghosts; every periodic dimension includes two ghosts.
"""
function _wrap_gnf_periodic_3d!(a, periodic_axes)
    for d in periodic_axes
        n = size(a, d)
        selectdim(a, d, 1) .= selectdim(a, d, n - 1)
        selectdim(a, d, n) .= selectdim(a, d, 2)
    end
    return a
end

"""
    _gnf_backend_copy_3d(backend, a)

Return a copy of host array `a` allocated on `backend` with the same element
type and size. A `BitArray` is first expanded to a dense `Array{Bool}`: a
bit-packed source has no bulk transfer path to a device array and would fall
back to scalar indexing.
"""
function _gnf_backend_copy_3d(backend, a)
    host = a isa BitArray ? Array{Bool}(a) : a
    b = KernelAbstractions.allocate(backend, eltype(host), size(host)...)
    copyto!(b, host)
    return b
end

"""
    _setup_gnf_channel_3d(; rheology, N, nu_init, kwargs...)

Build the initial state used by `run_gnf_channel_3d`. Arrays remain on the
requested backend; `f_in`/`f_out` can be passed to `_step_gnf_channel_3d!` for
individual steps. Scalar, length-N vector, and `y -> value` profiles use
distance from the lower physical wall. `u_init` denotes physical velocity;
the initial equilibrium momentum subtracts half the body force.
"""
function _setup_gnf_channel_3d(; rheology, N::Int, nu_init,
        q_w::Real=0.5, flow_axis::Symbol=:x, wall_axis::Symbol=:z,
        n_transverse::Int=4, drive::Symbol=:body_force, F::Real=0.0,
        u_inlet=nothing, Nx_open::Int=32, u_init=nothing, u_top::Real=0.0,
        backend=KernelAbstractions.CPU(), FT::Type{<:AbstractFloat}=Float64)
    flow_dim = _gnf_axis_3d(flow_axis)
    wall_dim = _gnf_axis_3d(wall_axis)
    flow_dim != wall_dim || throw(ArgumentError("flow_axis and wall_axis must differ"))
    N >= 2 || throw(ArgumentError("N must be at least 2"))
    n_transverse >= 3 || throw(ArgumentError("n_transverse includes two ghosts and must be at least 3"))
    0 < q_w <= 1 || throw(ArgumentError("q_w must lie in (0, 1]"))
    isfinite(F) && isfinite(u_top) || throw(ArgumentError("F and u_top must be finite"))
    drive in (:body_force, :zou_he) || throw(ArgumentError("drive must be :body_force or :zou_he"))
    if drive === :zou_he
        flow_axis === :x || throw(ArgumentError("Zou-He channels currently require flow_axis=:x"))
        iszero(F) || throw(ArgumentError("Zou-He channels require F=0"))
        isnothing(u_inlet) && throw(ArgumentError("Zou-He channels require u_inlet"))
        Nx_open >= 3 || throw(ArgumentError("Nx_open must be at least 3"))
    end
    model = _gnf3d_rheology(FT, rheology)
    positions = FT(q_w) .+ FT.(0:N-1)
    H = FT(N - 1) + FT(2) * FT(q_w)
    nu_values = _gnf_profile_3d(nu_init, positions, FT, "nu_init")
    all(>(zero(FT)), nu_values) || throw(ArgumentError("nu_init must be positive"))
    u_values = _gnf_profile_3d(u_init, positions, FT, "u_init")
    inlet_values = _gnf_profile_3d(u_inlet, positions, FT, "u_inlet")
    dims = ntuple(d -> d == wall_dim ? N + 2 :
                     (drive === :zou_he && d == flow_dim ? Nx_open : n_transverse), 3)
    Nx, Ny, Nz = dims
    periodic_axes = Tuple(d for d in 1:3 if d != wall_dim &&
                          (drive === :body_force || d != flow_dim))
    interior = ntuple(d -> 2:dims[d]-1, 3)
    is_solid_h = falses(dims)
    q_wall_h = zeros(FT, dims..., 19)
    uw_h = ntuple(_ -> zeros(FT, dims..., 19), 3)
    force_h = ntuple(_ -> zeros(FT, dims), 3)
    velocity_h = ntuple(_ -> zeros(FT, dims), 3)
    sp_h = zeros(FT, dims)
    sm_h = similar(sp_h)
    f_h = zeros(FT, dims..., 19)
    c_wall = (velocities_x(D3Q19()), velocities_y(D3Q19()),
              velocities_z(D3Q19()))[wall_dim]
    for I in CartesianIndices(dims)
        w = I[wall_dim]
        solid = w == 1 || w == N + 2
        is_solid_h[I] = solid
        p = clamp(w - 1, 1, N)
        nu0 = nu_values[p]
        sp_h[I] = inv(FT(3) * nu0 + FT(0.5))
        sm_h[I] = inv(FT(3 / 16) / (FT(3) * nu0) + FT(0.5))
        u0 = solid ? zero(FT) : u_values[p]
        force0 = solid ? zero(FT) : FT(F)
        velocity_h[flow_dim][I] = u0
        force_h[flow_dim][I] = force0
        raw_u = ntuple(d -> d == flow_dim ? u0 - force0 / FT(2) : zero(FT), 3)
        for q in 1:19
            f_h[I, q] = equilibrium(D3Q19(), one(FT), raw_u..., q)
            if (w == 2 && c_wall[q] < 0) || (w == N + 1 && c_wall[q] > 0)
                q_wall_h[I, q] = FT(q_w)
                if w == N + 1
                    uw_h[flow_dim][I, q] = FT(u_top)
                end
            end
        end
    end
    f_in = _gnf_backend_copy_3d(backend, f_h)
    f_out = similar(f_in)
    rho = _gnf_backend_copy_3d(backend, ones(FT, dims))
    ux, uy, uz = map(a -> _gnf_backend_copy_3d(backend, a), velocity_h)
    Fx, Fy, Fz = map(a -> _gnf_backend_copy_3d(backend, a), force_h)
    uw_x, uw_y, uw_z = map(a -> _gnf_backend_copy_3d(backend, a), uw_h)
    sp = _gnf_backend_copy_3d(backend, sp_h)
    sm = _gnf_backend_copy_3d(backend, sm_h)
    is_solid = _gnf_backend_copy_3d(backend, is_solid_h)
    q_wall = _gnf_backend_copy_3d(backend, q_wall_h)
    for a in (f_in, sp, sm)
        _wrap_gnf_periodic_3d!(a, periodic_axes)
    end
    bcspec = if drive === :zou_he
        profile_h = zeros(FT, Ny, Nz)
        for k in 1:Nz, j in 1:Ny
            w = wall_dim == 2 ? j : k
            if 2 <= w <= N + 1
                profile_h[j, k] = inlet_values[w - 1]
            end
        end
        profile = _gnf_backend_copy_3d(backend, profile_h)
        BCSpec3D(; west=ZouHeVelocity(profile), east=ZouHePressure(one(FT)))
    else
        nothing
    end
    return (; f_in, f_out, rho, ux, uy, uz, sp, sm, is_solid, q_wall,
            uw_x, uw_y, uw_z, Fx, Fy, Fz, dims, Nx, Ny, Nz, N, positions, H,
            flow_axis, wall_axis, flow_dim, wall_dim, periodic_axes, interior,
            drive, bcspec, rheology=model, nu_reference=first(nu_values), backend)
end

"""
    _step_gnf_channel_3d!(state, f_out, f_in)

Advance populations once without swapping buffers. The fused collision writes
physical pre-collision velocity, including the Guo half-force shift. Zou-He
rebuilds overwrite only boundary populations; refresh their moments with the
unforced getter when diagnostics are required.
"""
function _step_gnf_channel_3d!(s, f_out, f_in)
    fused_trt_gnf_libb_guo_field_step_3d!(f_out, f_in, s.rho, s.ux, s.uy, s.uz,
        s.is_solid, s.q_wall, s.uw_x, s.uw_y, s.uw_z, s.Fx, s.Fy, s.Fz,
        s.sp, s.sm, s.Nx, s.Ny, s.Nz, s.rheology)
    if s.drive === :zou_he
        # Zero normal viscosity gradient: face shear from the fused pull is
        # incomplete before Zou-He reconstructs its missing populations.
        for rates in (s.sp, s.sm)
            selectdim(rates, 1, 1) .= selectdim(rates, 1, 2)
            selectdim(rates, 1, s.Nx) .= selectdim(rates, 1, s.Nx - 1)
        end
        apply_bc_rebuild_3d!(f_out, f_in, s.bcspec, s.nu_reference,
            s.Nx, s.Ny, s.Nz; sp_field=s.sp, sm_field=s.sm)
    end
    for a in (f_out, s.sp, s.sm)
        _wrap_gnf_periodic_3d!(a, s.periodic_axes)
    end
    return nothing
end

"""
    _gnf_mean_profile_3d(a, interior, wall_dim) -> Vector

Average field `a` over the two transverse directions, restricted to the
`interior` index ranges, leaving one value per wall-normal node.
"""
function _gnf_mean_profile_3d(a, interior, wall_dim)
    transverse = Tuple(d for d in 1:3 if d != wall_dim)
    active = view(a, interior...)
    count = prod(length(interior[d]) for d in transverse)
    return vec(sum(active; dims=transverse)) ./ count
end

"""
    run_gnf_channel_3d(; rheology, N, nu_init, max_steps, kwargs...)

Run an isothermal D3Q19 GNF planar channel with TRT, Guo forcing, and LI-BB
walls. `N` counts fluid nodes across the wall-normal direction; the wall gap
is `H=N-1+2q_w` and returned positions are `q_w .+ (0:N-1)`.

Zou-He uses a zero normal gradient of viscosity at the open faces: both TRT
rates are copied from plane 2 to plane 1 and from plane Nx-1 to plane Nx before
the boundary reconstruction. The face's own GNF shear estimate uses incomplete
pulled populations, so it does not set the boundary viscosity.

Args:
    rheology: Newtonian, PowerLaw, CarreauYasuda, or Cross model.
    N: Number of wall-normal fluid nodes (at least 2).
    nu_init: Positive scalar, length-N vector, or function of wall distance.
    max_steps: Maximum iteration count, including zero for initialization.
    q_w: Cut-link fraction in (0, 1], default 0.5.
    flow_axis, wall_axis: Distinct :x, :y, or :z axes.
    n_transverse: Total extent of each periodic axis INCLUDING two ghosts;
        default 4 gives two physical nodes, as in the LI-BB Couette canary.
    drive: :body_force (periodic flow) or :zou_he (x-flow only, F=0).
    F: Body force density in the positive flow direction, default zero.
    u_inlet: Zou-He profile, as a scalar, length-N vector, or function.
    Nx_open: Total x extent for Zou-He, including inlet and outlet faces.
    u_init: Initial physical velocity profile; defaults to zero.
    u_top: Upper-wall velocity in the flow direction; default zero.
    tol_ss, check_every: Stop when maximum flow-velocity change since the last
        check is at most tol_ss times maximum velocity on physical interior
        cells. Defaults are 1e-10 and 100 steps.
    backend, FT: KernelAbstractions backend and floating-point type.

Returns:
    NamedTuple with positions, transverse-mean u and nu profiles, steps,
    converged, H, N, axes and interior metadata, and backend-resident fields.
    Interior ranges exclude solids, periodic ghosts, and Zou-He end faces.
    Body-force velocities are the fused collision's physical pre-collision
    moments; final f contains post-collision populations. No second half-force
    correction is applied. Zou-He velocities use rebuilt unforced moments.
"""
function run_gnf_channel_3d(; rheology, N::Int, nu_init, max_steps::Int,
        q_w::Real=0.5, flow_axis::Symbol=:x, wall_axis::Symbol=:z,
        n_transverse::Int=4, drive::Symbol=:body_force, F::Real=0.0,
        u_inlet=nothing, Nx_open::Int=32, u_init=nothing, u_top::Real=0.0,
        tol_ss::Real=1e-10, check_every::Int=100,
        backend=KernelAbstractions.CPU(), FT::Type{<:AbstractFloat}=Float64)
    max_steps >= 0 || throw(ArgumentError("max_steps must be nonnegative"))
    check_every > 0 || throw(ArgumentError("check_every must be positive"))
    isfinite(tol_ss) && tol_ss >= 0 || throw(ArgumentError("tol_ss must be finite and nonnegative"))
    s = _setup_gnf_channel_3d(; rheology, N, nu_init, q_w, flow_axis,
        wall_axis, n_transverse, drive, F, u_inlet, Nx_open, u_init, u_top,
        backend, FT)
    f_in, f_out = s.f_in, s.f_out
    velocity = (s.ux, s.uy, s.uz)[s.flow_dim]
    active = view(velocity, s.interior...)
    previous = KernelAbstractions.allocate(backend, FT, size(active)...)
    previous .= active
    difference = similar(previous)
    converged = false
    steps = 0
    for step in 1:max_steps
        _step_gnf_channel_3d!(s, f_out, f_in)
        f_in, f_out = f_out, f_in
        steps = step
        if step % check_every == 0 || step == max_steps
            if drive === :zou_he
                compute_macroscopic_3d!(s.rho, s.ux, s.uy, s.uz, f_in)
            end
            difference .= abs.(active .- previous)
            change = maximum(difference)
            scale = maximum(abs, active)
            converged = change <= FT(tol_ss) * scale
            converged && break
            previous .= active
        end
    end
    for a in (s.rho, s.ux, s.uy, s.uz)
        _wrap_gnf_periodic_3d!(a, s.periodic_axes)
    end
    KernelAbstractions.synchronize(backend)
    u = _gnf_mean_profile_3d(Array(velocity), s.interior, s.wall_dim)
    nu_h = (inv.(Array(s.sp)) .- FT(0.5)) ./ FT(3)
    nu = _gnf_mean_profile_3d(nu_h, s.interior, s.wall_dim)
    fields = (; rho=s.rho, ux=s.ux, uy=s.uy, uz=s.uz, f=f_in, sp=s.sp, sm=s.sm,
               is_solid=s.is_solid, q_wall=s.q_wall, uw_x=s.uw_x, uw_y=s.uw_y,
               uw_z=s.uw_z, Fx=s.Fx, Fy=s.Fy, Fz=s.Fz)
    return (; positions=s.positions, u, nu, steps, converged, fields, N, H=s.H,
            flow_axis, wall_axis, flow_dim=s.flow_dim, wall_dim=s.wall_dim,
            periodic_axes=s.periodic_axes, interior=s.interior, dims=s.dims, drive)
end
