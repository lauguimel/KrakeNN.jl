# ============================================================================
# src/drivers/ehd_phi_ddf.jl
# The DDF (pseudo-time lattice Boltzmann) solve of the EHD electric potential,
# shared by the hydrostatic driver (`run_ehd_hydrostatic_2d`, periodic sides)
# and the electroconvection state (`advance!(::ECState)`, Neumann box).
#
# One iteration: collide with the charge source, stream, rebuild the plate
# populations by non-equilibrium extrapolation, take the zeroth moment. The
# adaptive solve repeats it until its stopping rule accepts. Not exported.
#
# Issue #23: the potential (zeroth moment) can stop changing while the field,
# rebuilt from the first moment, is still relaxing from a cold start. The
# stopping rule therefore checks both moments.
# ============================================================================

"""
Check cadence of the adaptive potential solve in `advance!(::ECState)`: the
stopping rule is evaluated every `EHD_EC_PHI_CHECK_EVERY` iterations and at
`phi_max_iter`, at the cost of one device-to-host copy per check.
"""
const EHD_EC_PHI_CHECK_EVERY = 8

"""
Check cadence of the adaptive potential solve in `run_ehd_hydrostatic_2d`: the
stopping rule is evaluated after every iteration.
"""
const EHD_HYDROSTATIC_PHI_CHECK_EVERY = 1

function _ehd_phi_ddf_check_xbc(xbc::Symbol)
    xbc in (:neumann, :periodic) ||
        throw(ArgumentError("xbc must be :neumann or :periodic, got :$(xbc)."))
    return nothing
end

# Entry validation of the drivers' `field_tol` keyword: a real in [0, Inf]
# (`NaN` and negative values rejected); `Inf` disables the field check. The EC
# checkpoint restore (ehd_ec_checkpoint.jl) accepts the same range.
function _ehd_phi_ddf_check_field_tol(field_tol)
    (field_tol isa Real && !isnan(field_tol) && field_tol >= 0) ||
        throw(ArgumentError("field_tol must be a real number in [0, Inf], got $(field_tol)."))
    return field_tol
end

# Entry validation of the drivers' `phi_max_iter` keyword on the adaptive DDF path:
# an integral value of at least 1, returned as an `Int` (`10000.0` becomes `10000`).
function _ehd_phi_ddf_max_iter(phi_max_iter)
    (phi_max_iter isa Real && isinteger(phi_max_iter) && phi_max_iter >= 1) ||
        throw(ArgumentError("phi_max_iter must be an integer of at least 1 on the adaptive " *
                            "potential solve, got $(phi_max_iter)."))
    return Int(phi_max_iter)
end

"""
    ehd_phi_ddf_step!(f_in, f_out, phi, qfield, p, xbc; phi_bottom, phi_top) -> (f_in, f_out)

One pseudo-time iteration of the DDF potential solve: collide `f_in` in place with
the source `qfield`, stream into `f_out`, rebuild the plate populations by
non-equilibrium extrapolation (`phi_bottom` at `j = 1`, `phi_top` at `j = Ny`),
and leave the zeroth moment of `f_out` in `phi`.

Returns the swapped pair: the returned `f_in` is the `f_out` just filled. Callers
rebind, `f_in, f_out = ehd_phi_ddf_step!(f_in, f_out, ...)`.

`xbc` selects the lateral treatment:
- `:neumann`: wall streaming on all four sides and extrapolation on the whole box
  (`advance!(::ECState)`);
- `:periodic`: periodic in x, walls in y, extrapolation on the plates only
  (`run_ehd_hydrostatic_2d`).

`p` is any object with the fields `eps`, `omega_U` and `nu_U`.

Inlined so that plate values given as constants by the caller reach the kernel
launch as constants: on the CPU backend a run-time `Float64` kernel argument costs
one small heap allocation per launch.
"""
@inline function ehd_phi_ddf_step!(f_in, f_out, phi, qfield, p, xbc::Symbol; phi_bottom, phi_top)
    _ehd_phi_ddf_check_xbc(xbc)
    Nx, Ny = size(phi)
    collide_electric_potential_2d!(f_in, qfield, p.eps, p.omega_U, p.nu_U)
    if xbc === :neumann
        stream_wall_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        compute_ehd_scalar_2d!(phi, f_out)
        apply_phi_nee_box_2d!(f_out, phi, phi_bottom, phi_top, Nx, Ny)
    else
        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        compute_ehd_scalar_2d!(phi, f_out)
        apply_phi_nee_walls_2d!(f_out, phi, phi_bottom, phi_top, Nx, Ny)
    end
    compute_ehd_scalar_2d!(phi, f_out)
    return f_out, f_in
end

"""
    ehd_phi_ddf_workspace(phi) -> NamedTuple

Scratch buffers of [`ehd_phi_ddf_solve!`](@ref), allocated once on the backend and
with the element type of `phi`: `phi_prev`, `Ex`, `Ey`, `Ex_prev`, `Ey_prev` (the
shape of `phi`), `diag` (length 4, on the device) and `diag_host` (length 4, on the
host). Their contents on entry do not matter: each buffer is overwritten before it
is read.
"""
function ehd_phi_ddf_workspace(phi)
    return (phi_prev=similar(phi), Ex=similar(phi), Ey=similar(phi),
            Ex_prev=similar(phi), Ey_prev=similar(phi),
            diag=similar(phi, 4), diag_host=Vector{eltype(phi)}(undef, 4))
end

"""
    ehd_phi_ddf_solve!(f_in, f_out, phi, qfield, p, xbc, ws; phi_tol, field_tol,
                       max_iter, check_every, phi_bottom, phi_top) -> (f_in, f_out, stats)

Adaptive DDF solve of the electric potential for a frozen charge `qfield`: repeat
[`ehd_phi_ddf_step!`](@ref) until the stopping rule accepts, or until `max_iter`
iterations have run.

The rule is evaluated at iteration `iter` when `iter % check_every == 0` (on
cadence) or `iter == max_iter`, with one device-to-host copy per check. It measures
- `phi_rel = max|phi - phi_prev| / max|phi|`, the relative change of the zeroth
  moment over the last iteration (`ehd_rel_change_2d!`);
- `field_rel = max|E - E_prev| / max(max|E|, E_ref)` (`ehd_field_change_2d!`), for
  the field rebuilt from the first moment (`compute_electric_field_2d!`): the change
  of `E` since the previous check, relative to `max(max|E|, E_ref)`, with
  `E_ref = |phi_bottom − phi_top|/(Ny − 1)` the applied field (`1/(Ny − 1)` in all
  public drivers). The applied field is a minimum scale: it applies while `E` is
  below it everywhere, as at the first checks of a cold start, where `E` starts at
  0. A `floatmin` guard avoids a division by zero when the plates are at the same
  potential and the field is zero. At the first check the previous field is the
  one on entry. Maxima run over all nodes and both components.

  Measured on 2026-10-08 at every accepting check: `max|E| / E_ref` is 1.484 to
  1.495 in the default hydrostatic driver and in 60x96 electroconvection over 20
  cycles (Float64 and Float32), and 0.99999999994 to 1.0043 on the 16-cell
  capacitors of `test/analytical/ES-002-STOP.jl`. At each of these acceptances,
  `max|E - E_prev| / max|E| <= field_tol` held as well, so the floor did not
  change where any of these solves stopped.

It accepts when `phi_rel <= phi_tol`, both moments are finite, the check is on
cadence and `field_rel <= field_tol`. A final check off cadence
(`max_iter % check_every != 0`) measures the field over fewer iterations and
cannot accept. `phi_tol` and `field_tol` are compared as given, without
conversion to the element type.

`field_tol` bounds a change between checks, not the error of the field. The
slowest mode of the iteration, `sin(π y / H)` between the plates with `H = Ny - 1`,
decays by `ρ ≈ 1 - p.nu_U * π^2 / H^2` per iteration; once it dominates, the
relative error of `E` at acceptance is about `κ * field_rel`, with
`κ = ρ^m / (1 - ρ^m) ≈ H^2 / (m * p.nu_U * π^2)` and `m = check_every`. `Inf`
means no field check: the rule is then `phi_rel <= phi_tol` alone, at every check,
which is the pre-#23 rule bit for bit (same exit iteration, same populations, same
`phi_rel`); it is reserved for non-regression comparisons against that rule.

Returns the swapped population pair (the solution is the returned `f_in`; callers
rebind) and `stats = (; iters, phi_rel, field_rel, converged)`:
- `iters`: iterations run (an `Int`);
- `phi_rel`, `field_rel`: their values at the last check;
- `converged`: `true` when a check accepted. `false` after `max_iter` iterations
  without acceptance; no exception is raised, each driver raises its own error.

On exit `phi` holds the zeroth moment of the returned `f_in`, and `ws.Ex`, `ws.Ey`
its field. `ws` is an [`ehd_phi_ddf_workspace`](@ref) or any `NamedTuple` with the
same fields; `p` is any object with the fields `eps`, `omega_U`, `nu_U` and
`tau_U`. `max_iter` and `check_every` are integers of at least 1.

Inlined, like [`ehd_phi_ddf_step!`](@ref), so that the caller's constant plate
values reach the kernel launches.
"""
@inline function ehd_phi_ddf_solve!(f_in, f_out, phi, qfield, p, xbc::Symbol, ws;
                                    phi_tol, field_tol, max_iter, check_every,
                                    phi_bottom, phi_top)
    _ehd_phi_ddf_check_xbc(xbc)
    _ehd_phi_ddf_check_field_tol(field_tol)
    (max_iter isa Integer && max_iter >= 1) ||
        throw(ArgumentError("max_iter must be an integer of at least 1, got $(max_iter)."))
    (check_every isa Integer && check_every >= 1) ||
        throw(ArgumentError("check_every must be an integer of at least 1, got $(check_every)."))
    FT = eltype(phi)
    Nx, Ny = size(phi)
    n_max = Int(max_iter)
    field_check = isfinite(field_tol)
    E_ref = FT(abs(phi_bottom - phi_top) / (Ny - 1))
    (; phi_prev, Ex, Ey, Ex_prev, Ey_prev, diag, diag_host) = ws
    compute_electric_field_2d!(Ex_prev, Ey_prev, f_in, p.tau_U)
    iters = 0
    phi_rel = FT(Inf)
    field_rel = FT(Inf)
    converged = false
    for iter in 1:n_max
        copyto!(phi_prev, phi)
        f_in, f_out = ehd_phi_ddf_step!(f_in, f_out, phi, qfield, p, xbc;
                                        phi_bottom=phi_bottom, phi_top=phi_top)
        iters = iter
        on_cadence = iter % check_every == 0
        if on_cadence || iter == n_max
            ehd_rel_change_2d!(diag, phi, phi_prev, Nx, Ny)
            compute_electric_field_2d!(Ex, Ey, f_in, p.tau_U)
            ehd_field_change_2d!(diag, Ex, Ey, Ex_prev, Ey_prev, E_ref, Nx, Ny, 2)
            copyto!(diag_host, diag)
            phi_rel = diag_host[1]
            field_rel = diag_host[3]
            accept = if field_check
                on_cadence && diag_host[2] != 0 && diag_host[4] != 0 &&
                    phi_rel <= phi_tol && field_rel <= field_tol
            else
                phi_rel <= phi_tol
            end
            if accept
                converged = true
                break
            end
            copyto!(Ex_prev, Ex)
            copyto!(Ey_prev, Ey)
        end
    end
    return f_in, f_out, (; iters, phi_rel, field_rel, converged)
end

# The drivers' non-convergence error. When the potential met `phi_tol` but the
# field did not settle, the message also gives the last relative field change.
function _ehd_phi_ddf_require_converged(stats, max_iter, phi_tol)
    stats.converged && return nothing
    msg = "Electric potential solve did not converge within $(max_iter) iterations. " *
          "Last relative change: $(stats.phi_rel)."
    if stats.phi_rel <= phi_tol
        msg *= " Last relative field change: $(stats.field_rel)."
    end
    error(msg)
end
