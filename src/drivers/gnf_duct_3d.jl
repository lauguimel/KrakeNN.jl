# GNF duct with a voxel obstacle: D3Q19 TRT + Guo force + LI-BB, periodic in x.
# Reuses `_step_gnf_channel_3d!` and the fused GNF kernel unchanged.

"""
    _gnf_duct_default_nu_3d(rheology) -> Real

Starting lattice viscosity used when `nu_init` is not given: the Newtonian
value, the power-law upper clamp, or the zero-shear viscosity `eta_0`.
"""
_gnf_duct_default_nu_3d(m::Newtonian) = m.nu
_gnf_duct_default_nu_3d(m::PowerLaw) = m.nu_max
_gnf_duct_default_nu_3d(m::Union{CarreauYasuda,Cross}) = m.eta_0
_gnf_duct_default_nu_3d(m) = throw(ArgumentError(
    "3D GNF duct supports Newtonian, PowerLaw, CarreauYasuda, and Cross; got $(typeof(m))"))

"""
    _gnf_duct_mask_3d(mask, Nx, Ny, Nz, q_w) -> Array{Bool,3}

Normalise the obstacle description to a dense `Nx*Ny*Nz` `Bool` array.
`nothing` means no obstacle; an `AbstractArray{Bool,3}` (or `BitArray`) is
copied; a function `(x, y, z) -> Bool` is evaluated at cell centres in units of
the cell size `h`, with `x=a-1/2` (a=1..Nx) and `y=b-1+q_w`, `z=c-1+q_w`
(distance from the lower duct walls, as the channel driver's `positions`).
"""
function _gnf_duct_mask_3d(mask, Nx, Ny, Nz, q_w)
    if isnothing(mask)
        return falses(Nx, Ny, Nz) |> Array{Bool}
    elseif mask isa AbstractArray
        ndims(mask) == 3 && size(mask) == (Nx, Ny, Nz) || throw(ArgumentError(
            "mask must have size (Nx, Ny, Nz) = $((Nx, Ny, Nz)); got $(size(mask))"))
        return Array{Bool}(mask)
    else
        out = Array{Bool}(undef, Nx, Ny, Nz)
        for c in 1:Nz, b in 1:Ny, a in 1:Nx
            out[a, b, c] = Bool(mask(a - 0.5, b - 1 + Float64(q_w), c - 1 + Float64(q_w)))
        end
        return out
    end
end

"""
    _setup_gnf_duct_3d(; rheology, Nx, Ny, Nz, F, mask, nu_init, q_w, u_init,
                       backend, FT)

Build the state used by `run_gnf_duct_obstacle_3d`. Arrays have size
`(Nx+2, Ny+2, Nz+2)`: x carries one periodic ghost plane at each end (wrapped
after every step), y and z carry one solid wall layer at each end. Fluid cell
`(a,b,c)` of the user grid sits at array index `(a+1,b+1,c+1)`.

Every fluid-to-solid link gets a `q_wall` entry (the kernel treats a solid
neighbour without one as a rest-equilibrium reservoir, not as a wall): `q_w`
towards the four duct walls, `0.5` (halfway bounce-back) towards obstacle
voxels. The three wall-velocity arrays are one shared zero array.
"""
function _setup_gnf_duct_3d(; rheology, Nx::Int, Ny::Int, Nz::Int, F::Real,
        mask=nothing, nu_init=nothing, q_w::Real=0.5, u_init::Real=0.0,
        backend=KernelAbstractions.CPU(), FT::Type{<:AbstractFloat}=Float64)
    min(Nx, Ny, Nz) >= 2 || throw(ArgumentError("Nx, Ny and Nz must be at least 2"))
    0 < q_w <= 1 || throw(ArgumentError("q_w must lie in (0, 1]"))
    isfinite(F) && isfinite(u_init) || throw(ArgumentError("F and u_init must be finite"))
    model = _gnf3d_rheology(FT, rheology)
    nu0 = isnothing(nu_init) ? _gnf_duct_default_nu_3d(rheology) : nu_init
    nu0 isa Real && isfinite(nu0) && nu0 > 0 || throw(ArgumentError("nu_init must be a positive number"))
    obstacle = _gnf_duct_mask_3d(mask, Nx, Ny, Nz, q_w)
    any(!, obstacle) || throw(ArgumentError("mask leaves no fluid cell"))
    dims = (Nx + 2, Ny + 2, Nz + 2)
    solid_h = falses(dims)
    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        wall = j == 1 || j == dims[2] || k == 1 || k == dims[3]
        # Ghost planes in x mirror the opposite interior plane.
        ia = i == 1 ? Nx : (i == dims[1] ? 1 : i - 1)
        solid_h[i, j, k] = wall || obstacle[ia, j - 1, k - 1]
    end
    is_wall_layer(j, k) = j == 1 || j == dims[2] || k == 1 || k == dims[3]
    cx, cy, cz = velocities_x(D3Q19()), velocities_y(D3Q19()), velocities_z(D3Q19())
    q_wall_h = zeros(FT, dims..., 19)
    for k in 2:dims[3]-1, j in 2:dims[2]-1, i in 2:dims[1]-1
        solid_h[i, j, k] && continue
        for q in 2:19
            ti, tj, tk = i + Int(cx[q]), j + Int(cy[q]), k + Int(cz[q])
            if solid_h[ti, tj, tk]
                q_wall_h[i, j, k, q] = is_wall_layer(tj, tk) ? FT(q_w) : FT(0.5)
            end
        end
    end
    sp0 = inv(FT(3) * FT(nu0) + FT(0.5))
    sm0 = inv(FT(3 / 16) / (FT(3) * FT(nu0)) + FT(0.5))
    # Equilibria at the physical velocity u_init, shifted by half the force
    # (the same convention as the channel driver).
    raw_fluid = ntuple(q -> equilibrium(D3Q19(), one(FT), FT(u_init) - FT(F) / 2,
                                        zero(FT), zero(FT), q), 19)
    raw_solid = ntuple(q -> equilibrium(D3Q19(), one(FT), zero(FT), zero(FT), zero(FT), q), 19)
    f_h = Array{FT}(undef, dims..., 19)
    for q in 1:19, k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        f_h[i, j, k, q] = solid_h[i, j, k] ? raw_solid[q] : raw_fluid[q]
    end
    fluid_h = .!solid_h
    ux_h = ifelse.(fluid_h, FT(u_init), zero(FT))
    Fx_h = ifelse.(fluid_h, FT(F), zero(FT))
    f_in = _gnf_backend_copy_3d(backend, f_h)
    f_out = similar(f_in)
    rho = _gnf_backend_copy_3d(backend, ones(FT, dims))
    ux = _gnf_backend_copy_3d(backend, ux_h)
    uy = _gnf_backend_copy_3d(backend, zeros(FT, dims))
    uz = _gnf_backend_copy_3d(backend, zeros(FT, dims))
    Fx = _gnf_backend_copy_3d(backend, Fx_h)
    Fy = _gnf_backend_copy_3d(backend, zeros(FT, dims))
    Fz = _gnf_backend_copy_3d(backend, zeros(FT, dims))
    uw = KernelAbstractions.zeros(backend, FT, dims..., 19)
    sp = _gnf_backend_copy_3d(backend, fill(sp0, dims))
    sm = _gnf_backend_copy_3d(backend, fill(sm0, dims))
    is_solid = _gnf_backend_copy_3d(backend, solid_h)
    q_wall = _gnf_backend_copy_3d(backend, q_wall_h)
    periodic_axes = (1,)
    for a in (f_in, sp, sm)
        _wrap_gnf_periodic_3d!(a, periodic_axes)
    end
    Hy = FT(Ny - 1) + 2 * FT(q_w)
    Hz = FT(Nz - 1) + 2 * FT(q_w)
    return (; f_in, f_out, rho, ux, uy, uz, sp, sm, is_solid, q_wall,
            uw_x=uw, uw_y=uw, uw_z=uw, Fx, Fy, Fz, dims,
            Nx=dims[1], Ny=dims[2], Nz=dims[3],
            nx=Nx, ny=Ny, nz=Nz, Hy, Hz, periodic_axes,
            interior=(2:Nx+1, 2:Ny+1, 2:Nz+1), drive=:body_force, bcspec=nothing,
            rheology=model, nu_reference=FT(nu0), obstacle, backend)
end

"""
    run_gnf_duct_obstacle_3d(; rheology, Nx, Ny, Nz, F, max_steps, kwargs...)

Run a steady D3Q19 GNF flow (TRT, Guo forcing, LI-BB) in a straight duct of
`Nx*Ny*Nz` fluid cells, periodic in x and driven by a uniform body force `F`
in +x, with no-slip walls on the four y/z sides and an optional solid obstacle
given as a voxel mask. Newtonian, PowerLaw, CarreauYasuda and Cross laws are
accepted (the 3D GNF kernel's set). Only the body force is imposed: there is
no flow-rate controller, so the flow rate is an output.

Args:
    rheology: viscosity law, parameters in LATTICE units (nu in h^2/dt).
    Nx, Ny, Nz: fluid cells along the periodic axis and the two cross axes.
    F: body force per unit mass (lattice acceleration, h/dt^2).
    mask: `nothing`, an `Nx*Ny*Nz` Bool array (true = solid), or a function
        `(x, y, z) -> Bool` of the cell centre in units of `h`; see
        `_gnf_duct_mask_3d` for the origin. Obstacle links use halfway
        bounce-back (q=0.5); the voxel surface is a staircase.
    nu_init: starting viscosity (default: law-dependent, see
        `_gnf_duct_default_nu_3d`).
    q_w: cut fraction of the four duct walls, default 0.5 (walls on the cell
        faces: cross-section `Ny*h` by `Nz*h`); the duct is `H=N-1+2q_w` wide.
    u_init: uniform initial x-velocity in fluid cells (default 0).
    tol_ss, check_every: stop when, over the last `check_every` steps, the
        maximum change of any velocity component on the fluid cells is at most
        `tol_ss` times the maximum speed. Defaults 1e-10 and 100. The run also
        stops (`converged = false`) at the first check where that change is
        not finite (divergence).
    h, dt: physical cell size [m] and time step [s] attached to the lattice
        (default 1, i.e. lattice units); they only set `scaling`/`physical`.
    backend, FT: KernelAbstractions backend and floating-point type.

Units convention. The solver works with h=1, dt=1, rho=1. A physical
quantity is lattice value times `scaling.<name>`: length h, time dt, velocity
h/dt, kinematic viscosity h^2/dt, acceleration h/dt^2, shear rate 1/dt, flow
rate h^3/dt. To run a physical law `nu_phys(gamma_dot_phys)`, give the
lattice law `nu_lat = nu_phys*dt/h^2` evaluated at `gamma_dot_lat =
gamma_dot_phys*dt` (so a Carreau `lambda` becomes `lambda_phys/dt`). Cell
`(a,b,c)` is centred at `((a-1/2)h, (b-1+q_w)h, (c-1+q_w)h)`.

Returns a NamedTuple: host arrays `ux, uy, uz, nu, rho` and `mask`
(obstacle, `Nx*Ny*Nz`; velocities and `nu` are zero inside it), `flow_rate`
(mean over x-planes of the sum of `ux`, lattice h^3/dt), `flow_rate_planes`,
`mass_flux_planes` (sum of `rho*ux`), `u_bulk` (`flow_rate/(Hy*Hz)`), `F`,
`steps`, `converged`, `convergence` (last relative change, comparable to
`tol_ss`), `u_max`, `Hy`, `Hz`, `scaling`, `physical`. Velocity is the fused
collision's physical pre-collision moment (Guo half-force included); `nu`
comes from the TRT rate of the last step.
"""
function run_gnf_duct_obstacle_3d(; rheology, Nx::Int, Ny::Int, Nz::Int, F::Real,
        max_steps::Int, mask=nothing, nu_init=nothing, q_w::Real=0.5,
        u_init::Real=0.0, tol_ss::Real=1e-10, check_every::Int=100,
        h::Real=1.0, dt::Real=1.0,
        backend=KernelAbstractions.CPU(), FT::Type{<:AbstractFloat}=Float64)
    max_steps >= 0 || throw(ArgumentError("max_steps must be nonnegative"))
    check_every > 0 || throw(ArgumentError("check_every must be positive"))
    isfinite(tol_ss) && tol_ss >= 0 || throw(ArgumentError("tol_ss must be finite and nonnegative"))
    h > 0 && dt > 0 || throw(ArgumentError("h and dt must be positive"))
    s = _setup_gnf_duct_3d(; rheology, Nx, Ny, Nz, F, mask, nu_init, q_w,
                           u_init, backend, FT)
    f_in, f_out = s.f_in, s.f_out
    comps = (view(s.ux, s.interior...), view(s.uy, s.interior...), view(s.uz, s.interior...))
    previous = map(copy, comps)
    converged = false
    change_rel = FT(Inf)
    steps = 0
    for step in 1:max_steps
        _step_gnf_channel_3d!(s, f_out, f_in)
        f_in, f_out = f_out, f_in
        steps = step
        if step % check_every == 0 || step == max_steps
            change = maximum(map((a, b) -> maximum(abs, a .- b), comps, previous))
            scale = maximum(map(a -> maximum(abs, a), comps))
            change_rel = scale > 0 ? change / scale : change
            # stop on divergence (a NaN never passes the test below)
            isfinite(change) && isfinite(scale) || (converged = false; break)
            converged = change <= FT(tol_ss) * scale
            converged && break
            foreach((p, a) -> copyto!(p, a), previous, comps)
        end
    end
    KernelAbstractions.synchronize(backend)
    inner(a) = Array(a)[s.interior...]
    mask_h = copy(s.obstacle)
    fluid = .!mask_h
    ux, uy, uz, rho = inner(s.ux), inner(s.uy), inner(s.uz), inner(s.rho)
    nu = ifelse.(fluid, (inv.(inner(s.sp)) .- FT(0.5)) ./ FT(3), zero(FT))
    ux .*= fluid; uy .*= fluid; uz .*= fluid
    planes = [sum(@view ux[a, :, :]) for a in 1:Nx]
    mass_planes = [sum((@view ux[a, :, :]) .* (@view rho[a, :, :]) .* (@view fluid[a, :, :]))
                   for a in 1:Nx]
    flow_rate = sum(planes) / Nx
    u_max = maximum(sqrt.(ux .^ 2 .+ uy .^ 2 .+ uz .^ 2))
    scaling = (; h=Float64(h), dt=Float64(dt), length=Float64(h), time=Float64(dt),
               velocity=Float64(h / dt), nu=Float64(h^2 / dt),
               acceleration=Float64(h / dt^2), shear_rate=Float64(1 / dt),
               flow_rate=Float64(h^3 / dt))
    physical = (; F=Float64(F) * scaling.acceleration,
                flow_rate=Float64(flow_rate) * scaling.flow_rate,
                u_bulk=Float64(flow_rate / (s.Hy * s.Hz)) * scaling.velocity,
                u_max=Float64(u_max) * scaling.velocity,
                nu_min=Float64(minimum(nu[fluid])) * scaling.nu,
                nu_max=Float64(maximum(nu[fluid])) * scaling.nu,
                Lx=Nx * Float64(h), Ly=Float64(s.Hy * h), Lz=Float64(s.Hz * h))
    return (; ux, uy, uz, nu, rho, mask=mask_h, flow_rate, flow_rate_planes=planes,
            mass_flux_planes=mass_planes, u_bulk=flow_rate / (s.Hy * s.Hz),
            F=Float64(F), steps, converged, convergence=Float64(change_rel),
            u_max=Float64(u_max), Hy=Float64(s.Hy), Hz=Float64(s.Hz), q_w=Float64(q_w),
            Nx, Ny, Nz, scaling, physical)
end
