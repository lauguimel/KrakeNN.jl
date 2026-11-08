using KernelAbstractions

# Metal silently returns 0 when indexing a runtime tuple inside a
# @kernel. Use explicit branches that the Metal AIR compiler can fold.

@inline function _d2q9_cx(q::Integer)
    q == 1 && return 0
    q == 2 && return 1
    q == 3 && return 0
    q == 4 && return -1
    q == 5 && return 0
    q == 6 && return 1
    q == 7 && return -1
    q == 8 && return -1
    q == 9 && return 1
    return 0
end
@inline function _d2q9_cy(q::Integer)
    q == 1 && return 0
    q == 2 && return 0
    q == 3 && return 1
    q == 4 && return 0
    q == 5 && return -1
    q == 6 && return 1
    q == 7 && return 1
    q == 8 && return -1
    q == 9 && return -1
    return 0
end
@inline function _d2q9_opp(q::Integer)
    q == 1 && return 1
    q == 2 && return 4
    q == 3 && return 5
    q == 4 && return 2
    q == 5 && return 3
    q == 6 && return 8
    q == 7 && return 9
    q == 8 && return 6
    q == 9 && return 7
    return 1
end

@inline _d3q19_cx(q::Integer) =
    q==1 ? 0 : q==2 ? 1 : q==3 ? -1 : q==4 ? 0 : q==5 ? 0 :
    q==6 ? 0 : q==7 ? 0 : q==8 ? 1 : q==9 ? -1 : q==10 ? 1 :
    q==11 ? -1 : q==12 ? 1 : q==13 ? -1 : q==14 ? 1 : q==15 ? -1 :
    q==16 ? 0 : q==17 ? 0 : q==18 ? 0 : q==19 ? 0 : 0
@inline _d3q19_cy(q::Integer) =
    q==1 ? 0 : q==2 ? 0 : q==3 ? 0 : q==4 ? 1 : q==5 ? -1 :
    q==6 ? 0 : q==7 ? 0 : q==8 ? 1 : q==9 ? 1 : q==10 ? -1 :
    q==11 ? -1 : q==12 ? 0 : q==13 ? 0 : q==14 ? 0 : q==15 ? 0 :
    q==16 ? 1 : q==17 ? -1 : q==18 ? 1 : q==19 ? -1 : 0
@inline _d3q19_cz(q::Integer) =
    q==1 ? 0 : q==2 ? 0 : q==3 ? 0 : q==4 ? 0 : q==5 ? 0 :
    q==6 ? 1 : q==7 ? -1 : q==8 ? 0 : q==9 ? 0 : q==10 ? 0 :
    q==11 ? 0 : q==12 ? 1 : q==13 ? 1 : q==14 ? -1 : q==15 ? -1 :
    q==16 ? 1 : q==17 ? 1 : q==18 ? -1 : q==19 ? -1 : 0

# =====================================================================
# GPU-native drag reduction for LI-BB — lightweight cut-link list.
#
# The legacy `compute_drag_libb_mei_2d(Array(f_pre), …)` copies the
# entire f array (41 MB at D=80 Float64) to host at every evaluation
# and computes the Mei MEA loop on CPU. With a per-step drag this
# serialises the GPU pipeline on every PCIe transfer and dominates
# H100 runtime (measured: 52 MLUPS at D=80, vs ~15 k MLUPS peak).
#
# Strategy: a cylinder with D=80 has only ~200 cut links out of the
# 577 k grid cells. Precompute a compact list of these links
# `(i, j, q, q_w)` once on host, upload to device, then at every
# drag evaluation launch a small kernel that reads only those cells
# and writes one float per link into a dense `F_link[1:Nlinks, 2]`
# device array. The CPU reduction — `sum(Array(F_link))` — transfers
# 2 × 4·Nlinks bytes per evaluation (~1-2 KB), negligible vs the
# kernel step.
#
# For the time-resolved ST 2D-2 case, stack results into
# `(Nlinks, n_steps, 2)`. One host transfer at the very end gives the
# full Fx(t), Fy(t) time series for FFT / Strouhal.
# =====================================================================

"""
    CutLinkList{AT}

Compact device-side representation of the cut links flagged by
`q_wall > 0` — `list_i`, `list_j`, `list_q`, `list_qw` are parallel
arrays of length `Nlinks`. Built once via `build_cut_link_list_2d(q_wall_h)`
on host, copied to the target backend.
"""
struct CutLinkList{IT<:AbstractVector{Int32}, WT<:AbstractVector}
    list_i::IT
    list_j::IT
    list_q::IT
    list_qw::WT
    Nlinks::Int
end

"""
    CutLinkList3D

Public type or module in the kernel-level LBM operation.
Construct or dispatch on this type according to the field layout and methods defined below.

```julia
using KrakeNN

KrakeNN.CutLinkList3D
```
"""
struct CutLinkList3D{IT<:AbstractVector{Int32}, WT<:AbstractVector}
    list_i::IT; list_j::IT; list_k::IT; list_q::IT
    list_qw::WT
    Nlinks::Int
end

"""
    PolymericDragSurface2D

Device-side list of polymeric stress drag links. Each slot represents
one `(i, j, q)` contribution from a fluid cell to a neighbouring solid
cell, in the same `j, i, q=2:9` order as `compute_polymeric_drag_2d`.
"""
struct PolymericDragSurface2D{IT<:AbstractVector{Int32}}
    list_i::IT
    list_j::IT
    list_i2::IT
    list_j2::IT
    list_cx::IT
    list_cy::IT
    list_extrapolate::IT
    Nlinks::Int
end

"""
    PolymericDragCache2D

Cached polymeric drag surface plus device/host reduction buffers.
Build once for a static `is_solid` mask with
`build_polymeric_drag_cache_2d`, then reuse each sampled step via
`compute_polymeric_drag_2d_gpu_cached`.
"""
struct PolymericDragCache2D{S, VT<:AbstractVector, HT<:AbstractVector{Float64}}
    surface::S
    Fx_link::VT
    Fy_link::VT
    Fx_buf::HT
    Fy_buf::HT
end

"""
    build_cut_link_list_2d(q_wall_h::Array{T,3}; backend=CPU())
                          -> CutLinkList

Scan a host `q_wall_h[Nx, Ny, 9]` array, collect the coordinates of
every cut link, and upload to `backend`. O(Nx·Ny) once at setup.
"""
function build_cut_link_list_2d(q_wall_h::AbstractArray{T,3}; backend=CPU()) where {T}
    Nx, Ny, _ = size(q_wall_h)
    is_h = Int32[]; js_h = Int32[]; qs_h = Int32[]; qws_h = T[]
    @inbounds for j in 1:Ny, i in 1:Nx, q in 2:9
        if q_wall_h[i,j,q] > 0
            push!(is_h, i); push!(js_h, j); push!(qs_h, q)
            push!(qws_h, q_wall_h[i,j,q])
        end
    end
    N = length(is_h)
    li = KernelAbstractions.allocate(backend, Int32, N)
    lj = KernelAbstractions.allocate(backend, Int32, N)
    lq = KernelAbstractions.allocate(backend, Int32, N)
    lw = KernelAbstractions.allocate(backend, T,      N)
    copyto!(li, is_h); copyto!(lj, js_h)
    copyto!(lq, qs_h); copyto!(lw, qws_h)
    return CutLinkList(li, lj, lq, lw, N)
end

"""
    build_cut_link_list_3d(q_wall_h::AbstractArray{T,4}; backend=CPU()) where {T}

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations.

```julia
using KrakeNN

methods(KrakeNN.build_cut_link_list_3d)
```
"""
function build_cut_link_list_3d(q_wall_h::AbstractArray{T,4}; backend=CPU()) where {T}
    Nx, Ny, Nz, _ = size(q_wall_h)
    is_h = Int32[]; js_h = Int32[]; ks_h = Int32[]; qs_h = Int32[]; qws_h = T[]
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 2:19
        if q_wall_h[i,j,k,q] > 0
            push!(is_h, i); push!(js_h, j); push!(ks_h, k); push!(qs_h, q)
            push!(qws_h, q_wall_h[i,j,k,q])
        end
    end
    N = length(is_h)
    li = KernelAbstractions.allocate(backend, Int32, N)
    lj = KernelAbstractions.allocate(backend, Int32, N)
    lk = KernelAbstractions.allocate(backend, Int32, N)
    lq = KernelAbstractions.allocate(backend, Int32, N)
    lw = KernelAbstractions.allocate(backend, T,      N)
    copyto!(li, is_h); copyto!(lj, js_h); copyto!(lk, ks_h)
    copyto!(lq, qs_h); copyto!(lw, qws_h)
    return CutLinkList3D(li, lj, lk, lq, lw, N)
end

"""
    build_polymeric_drag_surface_2d(is_solid, Nx, Ny; backend=get_backend(is_solid),
                                    extrapolate=true)

Precompute the static fluid-solid links used by `compute_polymeric_drag_2d`.
The uploaded device lists preserve the host loop order and encode whether
the wall extrapolation uses the fluid-side neighbour or falls back to the
current cell.
"""
function build_polymeric_drag_surface_2d(is_solid, Nx::Integer, Ny::Integer;
                                           backend=KernelAbstractions.get_backend(is_solid),
                                           extrapolate::Bool=true)
    solid = Array(is_solid)
    cxv = (0, 1, 0, -1,  0, 1, -1, -1,  1)
    cyv = (0, 0, 1,  0, -1, 1,  1, -1, -1)
    is_h = Int32[]; js_h = Int32[]
    i2s_h = Int32[]; j2s_h = Int32[]
    cxs_h = Int32[]; cys_h = Int32[]
    extrap_h = Int32[]

    @inbounds for j in 1:Ny, i in 1:Nx
        if !solid[i, j]
            for q in 2:9
                cx = cxv[q]
                cy = cyv[q]
                ni = i + cx
                nj = j + cy
                if 1 <= ni <= Nx && 1 <= nj <= Ny && solid[ni, nj]
                    i2 = i - cx
                    j2 = j - cy
                    use_extrapolation =
                        extrapolate && 1 <= i2 <= Nx && 1 <= j2 <= Ny && !solid[i2, j2]
                    push!(is_h, Int32(i)); push!(js_h, Int32(j))
                    push!(i2s_h, Int32(use_extrapolation ? i2 : i))
                    push!(j2s_h, Int32(use_extrapolation ? j2 : j))
                    push!(cxs_h, Int32(cx)); push!(cys_h, Int32(cy))
                    push!(extrap_h, Int32(use_extrapolation ? 1 : 0))
                end
            end
        end
    end

    N = length(is_h)
    li = KernelAbstractions.allocate(backend, Int32, N)
    lj = KernelAbstractions.allocate(backend, Int32, N)
    li2 = KernelAbstractions.allocate(backend, Int32, N)
    lj2 = KernelAbstractions.allocate(backend, Int32, N)
    lcx = KernelAbstractions.allocate(backend, Int32, N)
    lcy = KernelAbstractions.allocate(backend, Int32, N)
    le = KernelAbstractions.allocate(backend, Int32, N)
    copyto!(li, is_h); copyto!(lj, js_h)
    copyto!(li2, i2s_h); copyto!(lj2, j2s_h)
    copyto!(lcx, cxs_h); copyto!(lcy, cys_h)
    copyto!(le, extrap_h)
    return PolymericDragSurface2D(li, lj, li2, lj2, lcx, lcy, le, N)
end

"""
    build_polymeric_drag_cache_2d(tau_template, is_solid, Nx, Ny; extrapolate=true)

Build a GPU/CPU-backend cache for polymeric drag. `tau_template` supplies
the backend; `is_solid` is copied to host once while constructing the
static surface list.
"""
function build_polymeric_drag_cache_2d(tau_template, is_solid,
                                         Nx::Integer, Ny::Integer;
                                         extrapolate::Bool=true)
    backend = KernelAbstractions.get_backend(tau_template)
    surface = build_polymeric_drag_surface_2d(is_solid, Nx, Ny;
                                              backend=backend,
                                              extrapolate=extrapolate)
    Fx_link = KernelAbstractions.allocate(backend, Float64, surface.Nlinks)
    Fy_link = KernelAbstractions.allocate(backend, Float64, surface.Nlinks)
    Fx_buf = zeros(Float64, surface.Nlinks)
    Fy_buf = zeros(Float64, surface.Nlinks)
    return PolymericDragCache2D(surface, Fx_link, Fy_link, Fx_buf, Fy_buf)
end

@kernel function _polymeric_drag_2d_surface_kernel!(Fx_link, Fy_link,
                                                     @Const(list_i), @Const(list_j),
                                                     @Const(list_i2), @Const(list_j2),
                                                     @Const(list_cx), @Const(list_cy),
                                                     @Const(list_extrapolate),
                                                     @Const(tau_p_xx),
                                                     @Const(tau_p_xy),
                                                     @Const(tau_p_yy))
    n = @index(Global)
    T = eltype(Fx_link)
    @inbounds begin
        i = Int(list_i[n]); j = Int(list_j[n])
        cx = T(list_cx[n]); cy = T(list_cy[n])
        if list_extrapolate[n] == 1
            i2 = Int(list_i2[n]); j2 = Int(list_j2[n])
            txx_w = T(1.5) * T(tau_p_xx[i, j]) - T(0.5) * T(tau_p_xx[i2, j2])
            txy_w = T(1.5) * T(tau_p_xy[i, j]) - T(0.5) * T(tau_p_xy[i2, j2])
            tyy_w = T(1.5) * T(tau_p_yy[i, j]) - T(0.5) * T(tau_p_yy[i2, j2])
        else
            txx_w = T(tau_p_xx[i, j])
            txy_w = T(tau_p_xy[i, j])
            tyy_w = T(tau_p_yy[i, j])
        end
        Fx_link[n] = txx_w * cx + txy_w * cy
        Fy_link[n] = txy_w * cx + tyy_w * cy
    end
end

"""
    compute_polymeric_drag_2d_gpu_cached(cache, tau_p_xx, tau_p_xy, tau_p_yy)

Evaluate polymeric drag from current stress fields using a precomputed
surface cache. Only the compact per-link reduction buffers are copied
back to host.
"""
function compute_polymeric_drag_2d_gpu_cached(cache::PolymericDragCache2D,
                                               tau_p_xx, tau_p_xy, tau_p_yy)
    surface = cache.surface
    surface.Nlinks == 0 && return (Fx=0.0, Fy=0.0)
    backend = KernelAbstractions.get_backend(tau_p_xx)
    _polymeric_drag_2d_surface_kernel!(backend)(cache.Fx_link, cache.Fy_link,
                                                 surface.list_i, surface.list_j,
                                                 surface.list_i2, surface.list_j2,
                                                 surface.list_cx, surface.list_cy,
                                                 surface.list_extrapolate,
                                                 tau_p_xx, tau_p_xy, tau_p_yy;
                                                 ndrange=(surface.Nlinks,))
    copyto!(cache.Fx_buf, cache.Fx_link)
    copyto!(cache.Fy_buf, cache.Fy_link)

    Fx = 0.0
    Fy = 0.0
    @inbounds for n in eachindex(cache.Fx_buf)
        Fx -= cache.Fx_buf[n]
        Fy -= cache.Fy_buf[n]
    end
    return (Fx=Fx, Fy=Fy)
end

# --- 2D: Mei-Luo-Shyy MEA on cut-link list ---
@kernel function _drag_mei_2d_list_kernel!(Fx_link, Fy_link,
                                             @Const(list_i), @Const(list_j),
                                             @Const(list_q), @Const(list_qw),
                                             @Const(f),
                                             @Const(uw_x), @Const(uw_y),
                                             Nx, Ny)
    n = @index(Global)
    T = eltype(f)
    @inbounds begin
        i  = Int(list_i[n]); j = Int(list_j[n])
        qi = Int(list_q[n]); qw = list_qw[n]
        qbar = _d2q9_opp(qi)
        cx_int = _d2q9_cx(qi); cy_int = _d2q9_cy(qi)
        cx_q = T(cx_int); cy_q = T(cy_int)
        im = i - cx_int; jm = j - cy_int
        fp_q_back = (1 <= im <= Nx && 1 <= jm <= Ny) ?
                     f[im, jm, qi] : f[i, j, qbar]
        fq_here    = f[i, j, qi]
        fqbar_here = f[i, j, qbar]
        w_q = (qi == 2 || qi == 3 || qi == 4 || qi == 5) ? T(1/9) : T(1/36)
        cu  = cx_q * uw_x[i, j, qi] + cy_q * uw_y[i, j, qi]
        δ   = -T(6) * w_q * cu
        arriving = qw ≤ T(0.5) ?
            T(2)*qw*fq_here + (one(T) - T(2)*qw)*fp_q_back + δ :
            (one(T)/(T(2)*qw))*fq_here + (one(T) - one(T)/(T(2)*qw))*fqbar_here +
                (one(T)/(T(2)*qw))*δ
        F_link = fq_here + arriving
        Fx_link[n] = cx_q * F_link
        Fy_link[n] = cy_q * F_link
    end
end

"""
    compute_drag_libb_mei_2d_gpu!(Fx_link, Fy_link, links::CutLinkList,
                                     f, uw_x, uw_y, Nx, Ny)

Compute each cut link's Fx, Fy contribution into dense device arrays
`Fx_link`, `Fy_link` (length = `links.Nlinks`). No atomic operations;
each thread writes its own slot. The caller sums on host with
`sum(Array(Fx_link))` — transferring at most a few kilobytes.
"""
function compute_drag_libb_mei_2d_gpu!(Fx_link, Fy_link, links::CutLinkList,
                                         f, uw_x, uw_y,
                                         Nx::Integer, Ny::Integer)
    # Skip kernel launch for empty link list — KernelAbstractions on CUDA
    # with ndrange=(0,) triggers an integer division error in the
    # workgroup-size computation. An empty list can arise in multi-block
    # setups where some blocks do not contain any cut cells (the cylinder
    # sits entirely in another block).
    links.Nlinks == 0 && return nothing
    backend = KernelAbstractions.get_backend(f)
    _drag_mei_2d_list_kernel!(backend)(Fx_link, Fy_link,
                                         links.list_i, links.list_j,
                                         links.list_q, links.list_qw,
                                         f, uw_x, uw_y, Nx, Ny;
                                         ndrange=(links.Nlinks,))
    return nothing
end

# --- 3D: halfway-BB Ladd (Mei 3D port = future work) ---
@kernel function _drag_halfway_3d_list_kernel!(Fx_link, Fy_link, Fz_link,
                                                 @Const(list_i), @Const(list_j),
                                                 @Const(list_k), @Const(list_q),
                                                 @Const(f), Nx, Ny, Nz)
    n = @index(Global)
    T = eltype(f)
    @inbounds begin
        i  = Int(list_i[n]); j = Int(list_j[n]); k = Int(list_k[n])
        qi = Int(list_q[n])
        fv = f[i, j, k, qi]
        two_fv = T(2) * fv
        Fx_link[n] = T(_d3q19_cx(qi)) * two_fv
        Fy_link[n] = T(_d3q19_cy(qi)) * two_fv
        Fz_link[n] = T(_d3q19_cz(qi)) * two_fv
    end
end

"""
    compute_drag_libb_3d_gpu!(Fx_link, Fy_link, Fz_link,
                                links::CutLinkList3D, f, Nx, Ny, Nz)

GPU halfway-BB Ladd drag on cut-link list (no atomics). Sum host-side.
"""
function compute_drag_libb_3d_gpu!(Fx_link, Fy_link, Fz_link,
                                     links::CutLinkList3D, f,
                                     Nx::Integer, Ny::Integer, Nz::Integer)
    backend = KernelAbstractions.get_backend(f)
    _drag_halfway_3d_list_kernel!(backend)(Fx_link, Fy_link, Fz_link,
                                             links.list_i, links.list_j,
                                             links.list_k, links.list_q,
                                             f, Nx, Ny, Nz;
                                             ndrange=(links.Nlinks,))
    return nothing
end
