using KernelAbstractions

# --- GNF step built from the per-cell pure functions (rheology/gnf_cell.jl) ---
#
# One step of the augmented state (f, τ), periodic in x, half-way bounce-back
# walls in y, arbitrary solid mask (full bounce-back swap at solid nodes), Guo
# body force given per node:
#
#   stream  f_in → f_out (arrived populations)          stream_periodic_x_wall_y_2d!
#   I1[c] = S:S(f_out[c], τ[c])       (τ lagged)        gnf_strain_invariant_2d!
#   ν[c]  = closure(I1[c])                              closure_eval!
#   (f_out[c], τ[c]) ← BGK + Guo at ν[c]                collide_gnf_cell_2d!
#
# The closure takes I1 = S:S, not γ̇ = √(2 I1) (see rheology/gnf_cell.jl);
# `gnf_strain_rate_2d!` gives γ̇ for diagnostics.
#
# The closure is a separate kernel so that an expensive closure is evaluated
# once per cell and per step, and so that the collision
# kernel does not depend on the closure type. `collide_gnf_fused_2d!` does the
# last three stages in one kernel through `gnf_cell_update`; both paths give
# the same numbers as `collide_rheology_guo_2d_kernel!` (test T1).
#
# τ at solid nodes is never written (as in the production kernel). The closure
# should hold numbers of the element type of `f` (`gnf_convert_closure`, done by
# `run_gnf_periodic_steady`), otherwise split and fused paths round ν differently.

@inline _gnf_load_cell(f, i, j) = SVector{9}(f[i,j,1], f[i,j,2], f[i,j,3], f[i,j,4], f[i,j,5],
                                            f[i,j,6], f[i,j,7], f[i,j,8], f[i,j,9])

@inline function _gnf_store_cell!(f, i, j, fo)
    f[i,j,1] = fo[1]; f[i,j,2] = fo[2]; f[i,j,3] = fo[3]
    f[i,j,4] = fo[4]; f[i,j,5] = fo[5]; f[i,j,6] = fo[6]
    f[i,j,7] = fo[7]; f[i,j,8] = fo[8]; f[i,j,9] = fo[9]
    return nothing
end

@kernel function gnf_strain_rate_2d_kernel!(gamma_dot, @Const(f), @Const(is_solid),
                                            @Const(tau), @Const(Fx), @Const(Fy))
    i, j = @index(Global, NTuple)
    @inbounds begin
        if is_solid[i, j]
            gamma_dot[i, j] = zero(eltype(gamma_dot))
        else
            fp = _gnf_load_cell(f, i, j)
            ρ, ux, uy = gnf_moments_cell(fp, Fx[i, j], Fy[i, j])
            gamma_dot[i, j] = gnf_strain_rate_cell(fp, ρ, ux, uy, tau[i, j])
        end
    end
end

@kernel function gnf_strain_invariant_2d_kernel!(I1, @Const(f), @Const(is_solid),
                                                 @Const(tau), @Const(Fx), @Const(Fy))
    i, j = @index(Global, NTuple)
    @inbounds begin
        if is_solid[i, j]
            I1[i, j] = zero(eltype(I1))
        else
            fp = _gnf_load_cell(f, i, j)
            ρ, ux, uy = gnf_moments_cell(fp, Fx[i, j], Fy[i, j])
            I1[i, j] = gnf_strain_invariant_cell(fp, ρ, ux, uy, tau[i, j])
        end
    end
end

@kernel function closure_eval_2d_kernel!(nu, @Const(I1), @Const(is_solid), closure)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if is_solid[i, j]
            nu[i, j] = zero(eltype(nu))
        else
            nu[i, j] = gnf_viscosity(closure, I1[i, j])
        end
    end
end

@kernel function collide_gnf_cell_2d_kernel!(f, @Const(is_solid), @Const(nu), tau,
                                             @Const(Fx), @Const(Fy))
    i, j = @index(Global, NTuple)
    @inbounds begin
        if is_solid[i, j]
            bounce_back_2d!(f, i, j)
        else
            fp = _gnf_load_cell(f, i, j)
            fo, τ_new = gnf_collide_cell(fp, Fx[i, j], Fy[i, j], nu[i, j])
            _gnf_store_cell!(f, i, j, fo)
            tau[i, j] = τ_new
        end
    end
end

@kernel function collide_gnf_fused_2d_kernel!(f, @Const(is_solid), closure, tau,
                                              I1, nu, @Const(Fx), @Const(Fy))
    i, j = @index(Global, NTuple)
    @inbounds begin
        if is_solid[i, j]
            bounce_back_2d!(f, i, j)
            I1[i, j] = zero(eltype(I1))
            nu[i, j] = zero(eltype(nu))
        else
            fp = _gnf_load_cell(f, i, j)
            fo, τ_new, s, ν = gnf_cell_update(fp, tau[i, j], Fx[i, j], Fy[i, j], closure)
            _gnf_store_cell!(f, i, j, fo)
            tau[i, j] = τ_new
            I1[i, j] = s
            nu[i, j] = ν
        end
    end
end

# Macroscopic fields from POST-collision populations. With the Guo scheme the
# collision adds exactly F to Σ c f, so u = (Σ c f_post − F/2)/ρ equals the
# velocity used by that collision, u = (Σ c f_arrived + F/2)/ρ.
@kernel function gnf_macroscopic_post_2d_kernel!(rho, ux, uy, @Const(f), @Const(is_solid),
                                                 @Const(Fx), @Const(Fy))
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)
        f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]; f5=f[i,j,5]
        f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]
        ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
        rho[i, j] = ρ
        if is_solid[i, j]
            ux[i, j] = zero(T)
            uy[i, j] = zero(T)
        else
            ux[i, j] = ((f2-f4+f6-f7-f8+f9) - Fx[i, j]/T(2)) / ρ
            uy[i, j] = ((f3-f5+f6+f7-f8-f9) - Fy[i, j]/T(2)) / ρ
        end
    end
end

# ============================================================
# Public API wrappers
# ============================================================

"""
    gnf_strain_invariant_2d!(I1, f, is_solid, tau, Fx, Fy)

Strain-rate invariant `I1 = S:S` of every fluid node from the ARRIVED
populations `f` and the lagged relaxation time `tau`
(`gnf_strain_invariant_cell`); 0 at solid nodes. This is what the closure reads.
"""
function gnf_strain_invariant_2d!(I1, f, is_solid, tau, Fx, Fy)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    gnf_strain_invariant_2d_kernel!(backend)(I1, f, is_solid, tau, Fx, Fy; ndrange=(Nx, Ny))
end

"""
    gnf_strain_rate_2d!(gamma_dot, f, is_solid, tau, Fx, Fy)

Shear rate `γ̇ = √(2 I1)` of every fluid node from the ARRIVED populations `f`
and the lagged relaxation time `tau` (`gnf_strain_rate_cell`, bit-identical to
the production estimator); 0 at solid nodes. Diagnostic: not used by `gnf_step!`.
"""
function gnf_strain_rate_2d!(gamma_dot, f, is_solid, tau, Fx, Fy)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    gnf_strain_rate_2d_kernel!(backend)(gamma_dot, f, is_solid, tau, Fx, Fy; ndrange=(Nx, Ny))
end

"""
    closure_eval!(nu, I1, is_solid, closure)

Kinematic viscosity (lattice units) of every fluid node, `nu = gnf_viscosity(closure, I1)`
with `I1 = S:S` the strain-rate invariant; 0 at solid nodes. `closure` is any
isbits object with a `gnf_viscosity` method (a `GeneralizedNewtonian` model in
lattice units, for instance).
"""
function closure_eval!(nu, I1, is_solid, closure)
    backend = KernelAbstractions.get_backend(nu)
    Nx, Ny = size(nu, 1), size(nu, 2)
    closure_eval_2d_kernel!(backend)(nu, I1, is_solid, closure; ndrange=(Nx, Ny))
end

"""
    collide_gnf_cell_2d!(f, is_solid, nu, tau, Fx, Fy)

In-place BGK + Guo collision of the arrived populations `f` at the viscosity
field `nu`; stores τ = 1/ω in `tau` at fluid nodes; bounce-back swap at solid nodes.
"""
function collide_gnf_cell_2d!(f, is_solid, nu, tau, Fx, Fy)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    collide_gnf_cell_2d_kernel!(backend)(f, is_solid, nu, tau, Fx, Fy; ndrange=(Nx, Ny))
end

"""
    collide_gnf_fused_2d!(f, is_solid, closure, tau, I1, nu, Fx, Fy)

Strain-rate invariant, closure and collision in one kernel (`gnf_cell_update`);
also writes the `I1` and ν used, so its outputs match the split path.
"""
function collide_gnf_fused_2d!(f, is_solid, closure, tau, I1, nu, Fx, Fy)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    collide_gnf_fused_2d_kernel!(backend)(f, is_solid, closure, tau, I1, nu, Fx, Fy;
                                          ndrange=(Nx, Ny))
end

"""
    gnf_macroscopic_post_2d!(rho, ux, uy, f, is_solid, Fx, Fy)

Density and velocity from post-collision populations (velocity 0 at solid nodes).
"""
function gnf_macroscopic_post_2d!(rho, ux, uy, f, is_solid, Fx, Fy)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    gnf_macroscopic_post_2d_kernel!(backend)(rho, ux, uy, f, is_solid, Fx, Fy; ndrange=(Nx, Ny))
end

"""
    gnf_step!(f_out, f_in, tau, I1, nu, is_solid, closure, Fx, Fy; fused=false)

One step of the periodic-x / wall-y GNF flow: streams `f_in` into `f_out`, then
collides `f_out` in place (split path: strain invariant, closure, collision
kernels; `fused=true`: one kernel). `tau` is read (lagged) and overwritten;
`I1` (strain-rate invariant S:S) and `nu` receive the values used at this step.
The caller swaps `f_in`/`f_out`. `closure` should hold numbers of
`eltype(f_in)` (`gnf_convert_closure`).
"""
function gnf_step!(f_out, f_in, tau, I1, nu, is_solid, closure, Fx, Fy; fused::Bool=false)
    Nx, Ny = size(f_in, 1), size(f_in, 2)
    stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
    if fused
        collide_gnf_fused_2d!(f_out, is_solid, closure, tau, I1, nu, Fx, Fy)
    else
        gnf_strain_invariant_2d!(I1, f_out, is_solid, tau, Fx, Fy)
        closure_eval!(nu, I1, is_solid, closure)
        collide_gnf_cell_2d!(f_out, is_solid, nu, tau, Fx, Fy)
    end
    return f_out
end
