# --- Steady adjoint of the GNF step ---
#
# State: the augmented post-collision state w = (f, τ) of `gnf_step!`, stored
# as one flat vector of length n = 10 N (N = Nx Ny): component (i, j, q) at
# i + Nx (j − 1) + N (q − 1), q = 1..9 the populations, q = 10 the relaxation
# time τ (the vec of an (Nx, Ny, 10) array). One step is w' = G(w):
#
#   arrivals fa = S f (pull stream, a permutation; τ untouched)
#   fluid node c:  (f'_c, τ'_c) = H(fa_c, τ_c),  H = BGK + Guo at ν(I1(fa_c, τ_c))
#   solid node c:  f'_c = swap(fa_c) (bounce-back), τ'_c = τ_c (never written)
#
# so ∂G/∂w = A = Bdiag · S, with Bdiag block diagonal: per fluid node the 10×10
# Jacobian of H, per solid node the swap (9×9) and 1 for τ.
#
# Local blocks. ν depends on the state through I1 only. At the linearisation
# point (I1_c, ν_c) the closure is replaced by its tangent,
# ν = ν_c + (dν/dI1)_c (I1 − I1_c) + δν, which is exact to first order; the
# block of node c is the ForwardDiff Jacobian of that cell map in
# (fa_c, τ_c, δν): 10 × 11, the 11th column ∂(f', τ')/∂ν. (dν/dI1)_c is the
# total derivative of the closure (ForwardDiff on `gnf_viscosity` in I1), so a
# closure only has to provide that scalar per node. Blocks are built
# on the host, once per adjoint solve, and copied to the device (10 × 10 per
# node; the ν column stays on the host for the parameter gradient).
#
# Active space. Components of w that never interact with the fluid are left
# out (they carry eigenvalues ±1 of A and would make I − Aᵀ singular): solid
# rest populations, populations of solid–solid links, τ at solid nodes. Active
# = every component of a fluid node + the solid-side population of each
# fluid–solid link (stored solid component whose arrival slot is at a fluid
# node; it is also the one fed by the fluid). A and Aᵀ map the active space
# into itself and the inactive space into itself (test T4b). P = the 0/1 mask.
#
# Gauge. Mass is conserved: c = 1 on active populations, 0 on τ, satisfies
# Aᵀ c = c, so I − Aᵀ has c in its kernel on the active space. The adjoint
# system is bordered with ĉ = c/‖c‖ (as `ad_gauge_augmented_adjoint`, written
# here for any array type):
#
#   [ I − P Aᵀ P   ĉ ] [λ]   [P ∂J/∂w]
#   [ ĉᵀ           0 ] [η] = [  0    ]
#
# The parameter gradient λᵀ ∂G/∂θ does not depend on the gauge because the
# collision conserves mass for every ν (Σ_q ∂f'_q/∂ν = 0).
#
# Gradient: dJ/dθ = Σ_c s_c ∂ν_c/∂θ, s_c = (∂(f', τ')_c/∂ν)ᵀ λ_c, with ∂ν_c/∂θ
# by ForwardDiff of `gnf_viscosity(closure_of(θ), I1_c)` (lattice units: for
# the starred units of the obstacle channel this is ν0,lat ∂η*/∂θ(I1*_c)).
#
# Krylov solver: restarted GMRES written for any AbstractVector (Array or
# CuArray): Gram–Schmidt with reorthogonalisation (modified, `orth = :mgs2`,
# default; or classical in two passes with matrix–vector products,
# `orth = :cgs2`), Givens rotations, stop on ‖r‖/‖b‖ ≤ tol checked on the
# explicit residual at the end of each cycle, optional deflated restarting
# (GMRES-DR, Morgan 2002: `deflate = k` harmonic Ritz vectors kept).
#
# No automatic differentiation runs inside a device kernel: JVP/VJP kernels
# are hand-written (blocks + `_spxwy_src`, the stream map of
# `kernels/stream_transpose_2d.jl`). Enzyme only appears in test T4.

using LinearAlgebra
import ForwardDiff

const _GNF_CX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const _GNF_CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)

# ============================================================
# State packing
# ============================================================

"""
    gnf_pack_state(f, tau) -> w

Flat state vector `[vec(f); vec(tau)]` (length 10 Nx Ny) on the backend of `f`.
"""
gnf_pack_state(f, tau) = vcat(vec(f), vec(tau))

"""
    gnf_unpack_state(w, Nx, Ny) -> (f, tau)

Copies of the populations (Nx, Ny, 9) and τ (Nx, Ny) held in a flat state vector.
"""
function gnf_unpack_state(w, Nx::Int, Ny::Int)
    N = Nx * Ny
    return reshape(w[1:9N], Nx, Ny, 9), reshape(w[9N+1:10N], Nx, Ny)
end

_gnf_host_field(v::Number, Nx, Ny) = fill(Float64(v), Nx, Ny)
_gnf_host_field(v::AbstractMatrix, Nx, Ny) =
    (size(v) == (Nx, Ny) || throw(DimensionMismatch("force field must be $(Nx)×$(Ny)"));
     Float64.(Array(v)))

# ============================================================
# Linearisation (host)
# ============================================================

# Cell map of the linearised step in x = (fa_1..fa_9, τ_prev, δν).
@inline function _gnf_lin_cell(x::SVector{11}, Fx, Fy, ν_c, κ_c, I1_c)
    fp = SVector{9}(x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9])
    ρ, ux, uy = gnf_moments_cell(fp, Fx, Fy)
    I1 = gnf_strain_invariant_cell(fp, ρ, ux, uy, x[10])
    ν = ν_c + κ_c * (I1 - I1_c) + x[11]
    fo, τn = gnf_collide_cell(fp, Fx, Fy, ν)
    return SVector{10}(fo[1], fo[2], fo[3], fo[4], fo[5], fo[6], fo[7], fo[8], fo[9], τn)
end

"""
    gnf_active_mask(is_solid) -> Vector{Bool}

Active components of the flat state (length 10 Nx Ny): all ten components of
fluid nodes, and the populations of solid nodes whose arrival slot is at a fluid
node (solid side of the fluid–solid links).
"""
function gnf_active_mask(is_solid::AbstractMatrix{Bool})
    Nx, Ny = size(is_solid)
    N = Nx * Ny
    act = fill(false, 10N)
    for j in 1:Ny, i in 1:Nx
        if !is_solid[i, j]
            for q in 1:10
                act[_spxwy_lin(i, j, q, Nx, Ny)] = true
            end
        else
            for q in 1:9
                d = _spxwy_dest(i, j, q, Nx, Ny)
                act[_spxwy_lin(i, j, q, Nx, Ny)] = !is_solid[(d - 1) % N + 1]
            end
        end
    end
    return act
end

"""
    GNFLinearisation

Host data of the step linearised at a state w = (f, τ) (`gnf_linearise`):
blocks `B` (Nx, Ny, 10, 11) (zero at solid nodes), the strain invariant `I1`,
viscosity `nu` and closure slope `dnu_dI1` used by the step from w, the
post-collision moments `rho`, `ux`, `uy` of w (velocity 0 at solid nodes), the
active mask, and counts (`n_small_I1` = fluid nodes with I1 < 1e-12, lattice units).
"""
struct GNFLinearisation
    Nx::Int
    Ny::Int
    is_solid::Matrix{Bool}
    B::Array{Float64,4}
    I1::Matrix{Float64}
    nu::Matrix{Float64}
    dnu_dI1::Matrix{Float64}
    rho::Matrix{Float64}
    ux::Matrix{Float64}
    uy::Matrix{Float64}
    active::Vector{Bool}
    n_fluid::Int
    n_active::Int
    n_small_I1::Int
    I1_min::Float64
    time::Float64
end

"""
    gnf_post_moments(f, is_solid, Fx, Fy) -> (rho, ux, uy)

Host density and velocity of post-collision populations (as
`gnf_macroscopic_post_2d!`: u = (Σ c f − F/2)/ρ, velocity 0 at solid nodes).
"""
function gnf_post_moments(f, is_solid::AbstractMatrix{Bool}, Fx, Fy=0)
    fh = Float64.(Array(f))
    Nx, Ny = size(is_solid)
    Fxh = _gnf_host_field(Fx, Nx, Ny); Fyh = _gnf_host_field(Fy, Nx, Ny)
    rho = zeros(Nx, Ny); ux = zeros(Nx, Ny); uy = zeros(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        f1=fh[i,j,1]; f2=fh[i,j,2]; f3=fh[i,j,3]; f4=fh[i,j,4]; f5=fh[i,j,5]
        f6=fh[i,j,6]; f7=fh[i,j,7]; f8=fh[i,j,8]; f9=fh[i,j,9]
        ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
        rho[i, j] = ρ
        if !is_solid[i, j]
            ux[i, j] = ((f2-f4+f6-f7-f8+f9) - Fxh[i, j]/2) / ρ
            uy[i, j] = ((f3-f5+f6+f7-f8-f9) - Fyh[i, j]/2) / ρ
        end
    end
    return rho, ux, uy
end

"""
    gnf_linearise(f, tau, is_solid, closure, Fx, Fy=0) -> GNFLinearisation

Local blocks of the GNF step at the post-collision state (f, τ) (any backend;
computed on the host in Float64). For every fluid node: arrived populations
fa = S f, (I1_c, ν_c) of the step from w (`gnf_cell_update`), closure slope
κ_c = dν/dI1 (ForwardDiff in I1), and the 10 × 11 Jacobian of the linearised
cell map in (fa_c, τ_c, δν).
"""
function gnf_linearise(f, tau, is_solid::AbstractMatrix{Bool}, closure, Fx, Fy=0)
    t0 = time()
    fh = Float64.(Array(f)); th = Float64.(Array(tau)); solid = Array{Bool}(is_solid)
    Nx, Ny = size(solid)
    size(fh) == (Nx, Ny, 9) || throw(DimensionMismatch("f must be $(Nx)×$(Ny)×9"))
    Fxh = _gnf_host_field(Fx, Nx, Ny); Fyh = _gnf_host_field(Fy, Nx, Ny)
    cl = gnf_convert_closure(Float64, closure)
    fa = similar(fh)
    stream_periodic_x_wall_y_2d!(fa, fh, Nx, Ny)
    B = zeros(Nx, Ny, 10, 11)
    I1 = zeros(Nx, Ny); nu = zeros(Nx, Ny); dnu = zeros(Nx, Ny)
    Threads.@threads for j in 1:Ny
        for i in 1:Nx
            solid[i, j] && continue
            fp = SVector{9,Float64}(fa[i,j,1], fa[i,j,2], fa[i,j,3], fa[i,j,4], fa[i,j,5],
                                    fa[i,j,6], fa[i,j,7], fa[i,j,8], fa[i,j,9])
            τp = th[i, j]; Fxc = Fxh[i, j]; Fyc = Fyh[i, j]
            _, _, s, ν = gnf_cell_update(fp, τp, Fxc, Fyc, cl)
            κ = ForwardDiff.derivative(t -> gnf_viscosity(cl, t), s)
            x0 = SVector{11,Float64}(fp[1], fp[2], fp[3], fp[4], fp[5], fp[6], fp[7], fp[8],
                                     fp[9], τp, 0.0)
            Jc = ForwardDiff.jacobian(x -> _gnf_lin_cell(x, Fxc, Fyc, ν, κ, s), x0)
            for col in 1:11, r in 1:10
                B[i, j, r, col] = Jc[r, col]
            end
            I1[i, j] = s; nu[i, j] = ν; dnu[i, j] = κ
        end
    end
    rho, ux, uy = gnf_post_moments(fh, solid, Fxh, Fyh)
    active = gnf_active_mask(solid)
    fl = .!solid
    n_small = count(fl .& (I1 .< 1e-12))
    I1min = any(fl) ? minimum(I1[fl]) : NaN
    return GNFLinearisation(Nx, Ny, solid, B, I1, nu, dnu, rho, ux, uy, active,
                            count(fl), count(active), n_small, I1min, time() - t0)
end
