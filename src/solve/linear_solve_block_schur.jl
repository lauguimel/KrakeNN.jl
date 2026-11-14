# Outer block structure and Schur-complement approximations of the block
# preconditioners P2 / P3 of the Krylov branch (src/solve/linear_solve_krylov.jl).
#
# J = [A G; D C] on velocity-like (U) and pressure (P) unknowns, extras (X, the
# flow-rate border) left to GMRES (identity). True Schur complement
# S = C − D A⁻¹ G. Selected by `KrylovBackendTag` options; with every option at
# its default (`block = :simple, schur_approx = :simple, schur_q = :auto,
# schur_shift = 0`) none of this file is used and P2 / P3 are unchanged.
#
# Block structure (`block`), Ã⁻¹ = the velocity-block solver (`ablock`: ILU(0),
# multigrid or exact LU), Ŝ⁻¹ = the Schur approximation below:
#   :simple  SIMPLE:  û = Ã⁻¹ v_U, p = Ŝ⁻¹ (v_P − D û), u = û − Q⁻¹ G p
#   :lower   block lower triangular [A 0; D Ŝ]:  u = Ã⁻¹ v_U, p = Ŝ⁻¹ (v_P − D u)
#   :upper   block upper triangular [A G; 0 Ŝ]:  p = Ŝ⁻¹ v_P, u = Ã⁻¹ (v_U − G p)
# With exact Ã⁻¹ and Ŝ = S the two triangular forms give J M⁻¹ with all
# eigenvalues 1 and a minimal polynomial of degree 2 (GMRES: 2 iterations).
#
# Schur approximation (`schur_approx`); Q is diag(A) (`schur_q = :diag`), the
# l1 row sum sign(aᵢᵢ) Σⱼ|aᵢⱼ| (`:rowsum`), or q I with q = `schur_qscale`
# (`:scalar`; q = cell volume makes Q the velocity mass matrix of the uniform
# grid, the LSC weight of Elman et al.; S̃ still depends on the state through
# C); `:auto` = row sum for SIMPLEC, diag otherwise. The assembled matrix is always
#     S̃ = C − D Q⁻¹ G − σ diag(D Q⁻¹ G)          (σ = `schur_shift`, ≥ 0),
# solved by the `schur` solver (LU / ILU(0) / multigrid):
#   :simple   Ŝ = S̃ with Q = diag(A)
#   :simplec  Ŝ = S̃ with Q = row sum of |A| (SIMPLEC)
#   :lsc      least-squares commutator with the scaled-BFBt weight Q:
#             Ŝ⁻¹ = S̃⁻¹ (α C − D Q⁻¹ A Q⁻¹ G) S̃⁻¹, α = `lsc_alpha`. S̃ plays
#             the stabilized Poisson matrix D Q⁻¹ G − C of the collocated grid
#             (D Q⁻¹ G alone has the checkerboard null space); α C restores
#             the Rhie–Chow part on the modes where D Q⁻¹ G vanishes (there
#             S ≈ C ≈ S̃, so α = 1 gives Ŝ ≈ S). Two S̃ solves per application.
#   :pcd      pressure convection–diffusion, built algebraically (collocated
#             grid: velocity and pressure share the cells): F_p = mean of the
#             diagonal velocity-component blocks of A (u-u, v-v, w-w: the same
#             convection–diffusion stencil; the Newton reaction terms average
#             to ≈ div u / d), Q_p = diag(F_p), and from the commutator
#             A G ≈ G F_p:  S ≈ Q_p S̃ F_p⁻¹, i.e. Ŝ⁻¹ = F_p S̃⁻¹ Q_p⁻¹. One S̃
#             solve per application. F_p carries the velocity boundary
#             conditions (Dirichlet walls), not the Neumann/Robin pressure
#             conditions of the PCD literature: first measure, then adjust.
#             Velocity fields = the fields before the pressure slot.
#   σ > 0 (any approximation) adds a multiple of the diagonal of D Q⁻¹ G,
#   which scales as cell volume / ν: on smooth modes S̃ → 0 like (k h)² while
#   S → V/ν (Stokes), so S̃ alone is h-dependent (κ(S̃⁻¹S) ~ N²) and the shifted
#   S̃ is spectrally equivalent to S for Stokes uniformly in h (σ ≈ 4 matches
#   the smooth modes of the 7-point stencil; Fourier argument, not measured).
#
# Exact velocity block (`ablock = :lu`): sparse LU of A through the Schur LU
# hooks (UMFPACK on CPU, cuDSS on GPU), refactorized with the values; with
# `schur = :lu` every inner solve is exact and the GMRES count measures the
# block approximation alone (benchmarks/fv3d/i5_mg_diagnose.jl).
#
# Transposed solve (adjoint): the transposed preconditioner is built from the
# blocks of Jᵀ (Aᵀ, G ← Dᵀ, D ← Gᵀ, C ← Cᵀ), so S̃ becomes S̃ᵀ (the shift has the
# same diagonal), LSC becomes exactly its transpose, and PCD uses F_pᵀ with the
# same factor order (not the transpose of the forward PCD; GMRES absorbs it).
#
# Device work: KernelAbstractions kernels and broadcasts (Array and CuArray,
# no scalar indexing); the PCD pattern is built once on the host; memory is
# linear in the cells (one Int32 per summed J position of F_p, a few vectors).

using KernelAbstractions

const _BS_BLOCKS = (:simple, :lower, :upper)
const _BS_APPROX = (:simple, :simplec, :lsc, :pcd)

# velocity-block solver actually used (`:auto` resolved per preconditioner)
_krylov_ablock_kind(tag) = tag.ablock === :auto ? (tag.precond === :mg ? :mg : :ilu0) : tag.ablock
_bs_qkind(tag) = tag.schur_q === :auto ? (tag.schur_approx === :simplec ? :rowsum : :diag) : tag.schur_q
_bs_active(tag) = tag.block !== :simple || tag.schur_approx !== :simple || tag.schur_q !== :auto ||
                  tag.schur_shift != 0

function _bs_check_tag(tag)
    tag.block in _BS_BLOCKS ||
        throw(ArgumentError("block must be one of $(_BS_BLOCKS), got :$(tag.block)"))
    tag.schur_approx in _BS_APPROX ||
        throw(ArgumentError("schur_approx must be one of $(_BS_APPROX), got :$(tag.schur_approx)"))
    tag.schur_q in (:auto, :diag, :rowsum, :scalar) ||
        throw(ArgumentError("schur_q must be :auto, :diag, :rowsum or :scalar, got :$(tag.schur_q)"))
    (tag.schur_q !== :scalar || (isfinite(tag.schur_qscale) && tag.schur_qscale > 0)) ||
        throw(ArgumentError("schur_q = :scalar needs schur_qscale > 0 (finite)"))
    tag.ablock in (:auto, :ilu0, :mg, :lu) ||
        throw(ArgumentError("ablock must be :auto, :ilu0, :mg or :lu, got :$(tag.ablock)"))
    (isfinite(tag.schur_shift) && tag.schur_shift >= 0 && isfinite(tag.lsc_alpha) && tag.lsc_alpha >= 0) ||
        throw(ArgumentError("need schur_shift >= 0 and lsc_alpha >= 0 (finite)"))
    return tag
end

# ---- exact velocity block (ablock = :lu) ----

mutable struct _KrylovALU{VI,VV}
    n::Int
    rowptr::VI
    colval::VI
    diag::VI
    A::VV                        # block values (filled by _simple_refresh!)
    lu::Any                      # Schur-LU hook object, built on the first refresh
end

_simple_avals(b::_KrylovALU) = b.A
_simple_ablk_solve!(x, b::_KrylovALU, v) = _simple_schur_solve!(x, b.lu, v)

function _bs_alu_refresh!(b::_KrylovALU)
    if b.lu === nothing
        b.lu = _simple_schur_lu(b.rowptr, b.colval, b.A, b.n)
    else
        _simple_schur_refactor!(b.lu)
    end
    return b
end

# Compact CSR (rowptr, colval, values) of the velocity block, whatever its solver.
_simple_ablk_csr(b::_KrylovALU) = (b.rowptr, b.colval, b.A)
_simple_ablk_csr(b::_KrylovSide) = (b.rowptr, b.colval, b.A)
_simple_ablk_csr(b::_KrylovMG) = (L = b.levels[1]; (L.rowptr, L.colval, L.A))

# ---- block / Schur data ----

mutable struct _KrylovBlockSchur{VI,VV}
    block::Symbol
    approx::Symbol
    qkind::Symbol
    qscale::Float64              # q of Q = q I (qkind = :scalar)
    shift::Float64
    alpha::Float64
    qinv::VV                     # 1 / q of every U row (Q = diag or l1 row sum of A)
    wu1::VV; wu2::VV             # U work vectors
    wp1::VV; wp2::VV; wp3::VV    # P work vectors
    nvel::Int                    # PCD: velocity components averaged into F_p
    Frp::VI; Fcv::VI; Fdiag::VI  # PCD: F_p pattern (pressure rows = cells), diagonal positions
    Fval::VV
    Fgrp::VI; Fgpos::VI          # PCD: J positions summed into each F_p entry
    qp::VV                       # PCD: 1 / diag(F_p)
end

# Host pattern of F_p: entry (c, c') = mean over velocity components m of
# J[(c, m), (c', m)]; Fgrp / Fgpos list the J positions of each entry.
function _bs_pcd_pattern(hrp::Vector{Int32}, hcv::Vector{Int32}, nfields::Int, nvel::Int, nlocal::Int)
    ncell = nlocal ÷ nfields
    Frp = Int32[1]; Fcv = Int32[]; Fdiag = Int32[]; Fgrp = Int32[1]; Fgpos = Int32[]
    pairs = Tuple{Int32,Int32}[]
    for c in 1:ncell
        empty!(pairs)
        for m in 1:nvel
            r = (c - 1) * nfields + m
            for q in hrp[r]:(hrp[r+1]-1)
                col = Int(hcv[q])
                (col <= nlocal && (col - 1) % nfields + 1 == m) || continue
                push!(pairs, (Int32((col - 1) ÷ nfields + 1), Int32(q)))
            end
        end
        sort!(pairs)
        d = Int32(0)
        t = 1
        while t <= length(pairs)
            cc = pairs[t][1]
            push!(Fcv, cc)
            cc == c && (d = Int32(length(Fcv)))
            while t <= length(pairs) && pairs[t][1] == cc
                push!(Fgpos, pairs[t][2])
                t += 1
            end
            push!(Fgrp, Int32(length(Fgpos) + 1))
        end
        d > 0 || throw(ArgumentError("schur_approx = :pcd: cell $c has no velocity diagonal"))
        push!(Fdiag, d)
        push!(Frp, Int32(length(Fcv) + 1))
    end
    return Frp, Fcv, Fdiag, Fgrp, Fgpos
end

function _bs_build(hrp::Vector{Int32}, hcv::Vector{Int32}, nfields::Int, pslot::Int, nlocal::Int,
                   nu::Int, np::Int, J, tag)
    devi(h) = copyto!(similar(J, Int32, length(h)), h)
    vec(m) = fill!(similar(J, m), 0.0)
    approx = tag.schur_approx
    nvel = 0
    Frp = devi(Int32[]); Fcv = devi(Int32[]); Fdiag = devi(Int32[]); Fgrp = devi(Int32[]); Fgpos = devi(Int32[])
    Fval = vec(0); qp = vec(0)
    if approx === :pcd
        nvel = pslot - 1
        nvel >= 1 || throw(ArgumentError("schur_approx = :pcd needs the velocity fields before the pressure slot"))
        hFrp, hFcv, hFdiag, hFgrp, hFgpos = _bs_pcd_pattern(hrp, hcv, nfields, nvel, nlocal)
        length(hFrp) - 1 == np ||
            throw(ArgumentError("schur_approx = :pcd: $(length(hFrp) - 1) cells but $np pressure unknowns"))
        Frp = devi(hFrp); Fcv = devi(hFcv); Fdiag = devi(hFdiag); Fgrp = devi(hFgrp); Fgpos = devi(hFgpos)
        Fval = vec(length(hFcv)); qp = vec(np)
    end
    return _KrylovBlockSchur(tag.block, approx, _bs_qkind(tag), Float64(tag.schur_qscale),
                             Float64(tag.schur_shift),
                             Float64(tag.lsc_alpha), vec(nu), vec(nu), vec(nu), vec(np), vec(np), vec(np),
                             nvel, Frp, Fcv, Fdiag, Fval, Fgrp, Fgpos, qp)
end

# ---- kernels ----

# 1 / q per U row: q = aᵢᵢ (diag) or sign(aᵢᵢ) Σⱼ |aᵢⱼ| (l1 row sum, SIMPLEC)
@kernel function _bs_qinv_kernel!(qinv, @Const(J), @Const(posAd), @Const(Arp), @Const(Avals), rowsum)
    a = @index(Global)
    @inbounds begin
        d = J[posAd[a]]
        if rowsum
            s = 0.0
            for q in Arp[a]:(Arp[a+1]-Int32(1))
                s += abs(Avals[q])
            end
            d = d >= 0 ? s : -s
        end
        qinv[a] = d == 0 ? 0.0 : 1.0 / d
    end
end

# S̃ = C − D Q⁻¹ G − σ diag(D Q⁻¹ G), one thread per pressure row (as
# _simple_schur_kernel!, with Q⁻¹ given and the diagonal of the product kept).
@kernel function _bs_schur_kernel!(Sval, @Const(J), @Const(Srp), @Const(Scv), @Const(posC),
                                   @Const(Drp), @Const(Dcv), @Const(posD),
                                   @Const(Grp), @Const(Gcv), @Const(posG), @Const(qinv), shift)
    i = @index(Global)
    @inbounds begin
        s0 = Srp[i]; s1 = Srp[i+1] - Int32(1)
        for s in s0:s1
            c = posC[s]
            Sval[s] = c > 0 ? J[c] : 0.0
        end
        dsum = 0.0
        dpos = Int32(0)
        for q in Drp[i]:(Drp[i+1]-Int32(1))
            j = Dcv[q]
            dj = J[posD[q]] * qinv[j]
            for qq in Grp[j]:(Grp[j+1]-Int32(1))
                k = Gcv[qq]
                lo = s0; hi = s1
                while lo < hi
                    mid = (lo + hi) >> Int32(1)
                    if Scv[mid] < k
                        lo = mid + Int32(1)
                    else
                        hi = mid
                    end
                end
                t = dj * J[posG[qq]]
                Sval[lo] -= t
                if k == i
                    dsum += t
                    dpos = lo
                end
            end
        end
        if dpos > 0
            Sval[dpos] -= shift * dsum
        end
    end
end

# y = G p
@kernel function _bs_gmul_kernel!(y, @Const(J), @Const(Grp), @Const(Gcv), @Const(posG), @Const(p))
    a = @index(Global)
    @inbounds begin
        s = 0.0
        for q in Grp[a]:(Grp[a+1]-Int32(1))
            s += J[posG[q]] * p[Gcv[q]]
        end
        y[a] = s
    end
end

# y = α C p (C entries located in the S̃ pattern by posC)
@kernel function _bs_cmul_kernel!(y, @Const(J), @Const(Srp), @Const(Scv), @Const(posC), @Const(p), alpha)
    i = @index(Global)
    @inbounds begin
        s = 0.0
        for t in Srp[i]:(Srp[i+1]-Int32(1))
            c = posC[t]
            if c > 0
                s += J[c] * p[Scv[t]]
            end
        end
        y[i] = alpha * s
    end
end

# F_p values: mean of the summed J positions
@kernel function _bs_fgather_kernel!(Fval, @Const(Fgrp), @Const(Fgpos), @Const(J), scale)
    e = @index(Global)
    @inbounds begin
        s = 0.0
        for t in Fgrp[e]:(Fgrp[e+1]-Int32(1))
            s += J[Fgpos[t]]
        end
        Fval[e] = scale * s
    end
end

@kernel function _bs_qp_kernel!(qp, @Const(Fval), @Const(Fdiag))
    i = @index(Global)
    @inbounds begin
        d = Fval[Fdiag[i]]
        qp[i] = d == 0 ? 0.0 : 1.0 / d
    end
end

# ---- host-side wrappers (also used by benchmarks/fv3d/i5_mg_diagnose.jl) ----

_bs_sync(x) = KernelAbstractions.synchronize(KernelAbstractions.get_backend(x))

# w = M z for a CSR triple on the device of w
function _bs_spmv!(w, rp, cv, vals, z)
    be = KernelAbstractions.get_backend(w)
    _krylov_spmv_kernel!(be, 256)(w, rp, cv, vals, z; ndrange = length(w))
    KernelAbstractions.synchronize(be)
    return w
end

function _bs_gmul!(y, s::_KrylovSimple, p)
    be = KernelAbstractions.get_backend(y)
    _bs_gmul_kernel!(be, 256)(y, s.J, s.Grp, s.Gcv, s.posG, p; ndrange = s.nu)
    KernelAbstractions.synchronize(be)
    return y
end

function _bs_cmul!(y, s::_KrylovSimple, p, alpha::Float64)
    be = KernelAbstractions.get_backend(y)
    _bs_cmul_kernel!(be, 256)(y, s.J, s.Srp, s.Scv, s.posC, p, alpha; ndrange = s.np)
    KernelAbstractions.synchronize(be)
    return y
end

# r = v − D u
function _bs_dmul!(r, s::_KrylovSimple, v, u)
    be = KernelAbstractions.get_backend(r)
    _simple_dmul_kernel!(be, 256)(r, v, s.J, s.Drp, s.Dcv, s.posD, u; ndrange = s.np)
    KernelAbstractions.synchronize(be)
    return r
end

# x ≈ S̃⁻¹ b with the configured Schur solver (x and b distinct)
function _bs_ssolve!(x, s::_KrylovSimple, b)
    if s.schur isa _KrylovSide
        _krylov_precond!(x, s.schur, b)
    elseif s.schurkind === :mg
        _mg_apply!(x, s.schur, b)
    else
        _simple_schur_solve!(x, s.schur, b)
    end
    _bs_sync(x)
    return x
end

# New values of J: Q⁻¹, S̃ (Q choice and shift), PCD F_p and Q_p. Called by
# _simple_refresh! after the velocity-block values were gathered.
function _bs_schur!(s::_KrylovSimple)
    bs = s.bs::_KrylovBlockSchur
    be = KernelAbstractions.get_backend(s.J)
    KernelAbstractions.synchronize(be)
    Arp, _, Avals = _simple_ablk_csr(s.Ablk)
    if bs.qkind === :scalar
        fill!(bs.qinv, 1.0 / bs.qscale)
    else
        _bs_qinv_kernel!(be, 256)(bs.qinv, s.J, s.posAd, Arp, Avals, bs.qkind === :rowsum; ndrange = s.nu)
    end
    KernelAbstractions.synchronize(be)
    _bs_schur_kernel!(be, 256)(s.Sval, s.J, s.Srp, s.Scv, s.posC, s.Drp, s.Dcv, s.posD,
                               s.Grp, s.Gcv, s.posG, bs.qinv, bs.shift; ndrange = s.np)
    if bs.approx === :pcd
        _bs_fgather_kernel!(be, 256)(bs.Fval, bs.Fgrp, bs.Fgpos, s.J, 1.0 / bs.nvel; ndrange = length(bs.Fval))
        KernelAbstractions.synchronize(be)
        _bs_qp_kernel!(be, 256)(bs.qp, bs.Fval, bs.Fdiag; ndrange = s.np)
    end
    KernelAbstractions.synchronize(be)
    return s
end

# p = Ŝ⁻¹ r with the configured approximation (p and r distinct; for the
# default SIMPLE preconditioner, s.bs === nothing, this is the S̃ solve).
function _bs_schur_solve!(p, s::_KrylovSimple, r)
    bs = s.bs
    if bs === nothing || bs.approx === :simple || bs.approx === :simplec
        return _bs_ssolve!(p, s, r)
    end
    if bs.approx === :lsc
        y = bs.wp1
        _bs_ssolve!(y, s, r)                             # y = S̃⁻¹ r
        _bs_gmul!(bs.wu1, s, y)
        bs.wu1 .*= bs.qinv                               # Q⁻¹ G y
        Arp, Acv, Avals = _simple_ablk_csr(s.Ablk)
        _bs_spmv!(bs.wu2, Arp, Acv, Avals, bs.wu1)
        bs.wu2 .*= bs.qinv                               # Q⁻¹ A Q⁻¹ G y
        _bs_cmul!(bs.wp3, s, y, bs.alpha)                # α C y
        _bs_dmul!(bs.wp2, s, bs.wp3, bs.wu2)             # α C y − D Q⁻¹ A Q⁻¹ G y
        return _bs_ssolve!(p, s, bs.wp2)                 # p = S̃⁻¹ (…)
    end
    # :pcd
    bs.wp1 .= bs.qp .* r                                 # Q_p⁻¹ r
    _bs_ssolve!(bs.wp2, s, bs.wp1)                       # S̃⁻¹ Q_p⁻¹ r
    return _bs_spmv!(p, bs.Frp, bs.Fcv, bs.Fval, bs.wp2) # F_p S̃⁻¹ Q_p⁻¹ r
end

# w = S v = C v − D A⁻¹ G v with the velocity-block solver (exact when
# ablock = :lu); tu1, tu2 (U) and tp (P) are caller work vectors.
function _bs_schur_true!(w, s::_KrylovSimple, v, tu1, tu2, tp)
    _bs_gmul!(tu1, s, v)
    _simple_ablk_solve!(tu2, s.Ablk, tu1)
    _bs_sync(tu2)
    _bs_cmul!(tp, s, v, 1.0)
    return _bs_dmul!(w, s, tp, tu2)
end

# One application z = M⁻¹ v of the block preconditioner (s.bs !== nothing).
function _bs_apply!(z, s::_KrylovSimple, v)
    bs = s.bs::_KrylovBlockSchur
    be = KernelAbstractions.get_backend(z)
    z .= v                                                   # extras: identity
    _krylov_take_kernel!(be, 256)(s.vu, z, s.uidx; ndrange = s.nu)
    _krylov_take_kernel!(be, 256)(s.vp, z, s.pidx; ndrange = s.np)
    KernelAbstractions.synchronize(be)
    if bs.block === :upper
        _bs_schur_solve!(s.p, s, s.vp)                       # p = Ŝ⁻¹ v_P
        _bs_gmul!(bs.wu1, s, s.p)
        bs.wu1 .= s.vu .- bs.wu1                             # v_U − G p
        _simple_ablk_solve!(s.u, s.Ablk, bs.wu1)             # u = Ã⁻¹ (v_U − G p)
    else
        _simple_ablk_solve!(s.uh, s.Ablk, s.vu)              # û = Ã⁻¹ v_U
        _bs_sync(s.uh)
        _bs_dmul!(s.rp, s, s.vp, s.uh)                       # v_P − D û
        _bs_schur_solve!(s.p, s, s.rp)                       # p = Ŝ⁻¹ (v_P − D û)
        if bs.block === :simple
            _bs_gmul!(bs.wu1, s, s.p)
            s.u .= s.uh .- bs.qinv .* bs.wu1                 # u = û − Q⁻¹ G p
        else
            s.u .= s.uh                                      # :lower
        end
    end
    KernelAbstractions.synchronize(be)
    _krylov_put_kernel!(be, 256)(z, s.u, s.uidx; ndrange = s.nu)
    _krylov_put_kernel!(be, 256)(z, s.p, s.pidx; ndrange = s.np)
    KernelAbstractions.synchronize(be)
    return z
end
