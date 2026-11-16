# Steady GNF adjoint (part 3/5): restarted GMRES.
# Split from src/ad/gnf_adjoint.jl, which holds the overview of the method.


# ============================================================
# GMRES (restarted, optional deflated restarting), any AbstractVector
# ============================================================

_gnf_sync(x) = KernelAbstractions.synchronize(KernelAbstractions.get_backend(x))

# Orthogonalise w against V[:, 1:j]; coefficients added to h[1:j]; returns ‖w‖.
function _gnf_orthogonalise!(h, V, w, j, orth::Symbol, hbuf)
    if orth === :mgs2
        for pass in 1:2
            for i in 1:j
                vi = view(V, :, i)
                hij = dot(vi, w)
                axpy!(-hij, vi, w)
                h[i] += hij
            end
        end
    elseif orth === :cgs2
        Vj = view(V, :, 1:j)
        hj = view(hbuf, 1:j)
        for pass in 1:2
            mul!(hj, transpose(Vj), w)
            mul!(w, Vj, hj, -one(eltype(w)), one(eltype(w)))
            h[1:j] .+= Array(hj)
        end
    else
        throw(ArgumentError("orth must be :mgs2 or :cgs2, got :$orth"))
    end
    return norm(w)
end

# Real basis of the k harmonic Ritz vectors of smallest |θ| (complex pairs kept whole).
function _gnf_harmonic_ritz(Hbar::AbstractMatrix{Float64}, m::Int, k::Int)
    Hm = Hbar[1:m, 1:m]; hm1 = Hbar[m+1, m]
    em = zeros(m); em[m] = 1
    fvec = transpose(Hm) \ em
    M = Hm + hm1^2 * fvec * transpose(em)
    E = eigen(M)
    order = sortperm(abs.(E.values))
    cols = Vector{Vector{Float64}}()
    vals = ComplexF64[]
    used = falses(m)
    for idx in order
        length(cols) >= k && break
        used[idx] && continue
        θ = E.values[idx]; g = E.vectors[:, idx]
        used[idx] = true
        if abs(imag(θ)) <= 1e-12 * max(abs(θ), 1e-300)
            push!(cols, real.(g)); push!(vals, θ)
        else
            push!(cols, real.(g)); push!(cols, imag.(g)); push!(vals, θ); push!(vals, conj(θ))
            # mark the conjugate as used
            for jdx in order
                if !used[jdx] && abs(E.values[jdx] - conj(θ)) <= 1e-10 * max(abs(θ), 1e-300)
                    used[jdx] = true; break
                end
            end
        end
    end
    return reduce(hcat, cols), vals
end

"""
    gnf_gmres!(x, apply!, b; tol=1e-6, restart=1000, maxiter=100_000, orth=:mgs2,
               deflate=0, max_wall=Inf, ritz=false, verbose=false, basis=nothing,
               work=nothing) -> info

Restarted GMRES for `apply!(y, x)` (y = M x), in place on the initial guess `x`.
Works on any AbstractVector type (Array, CuArray): Krylov basis
`similar(b, length(b), restart + 1)`, small dense algebra on the host.

- `orth = :mgs2`: modified Gram–Schmidt, two passes (reorthogonalisation);
  `:cgs2`: classical Gram–Schmidt, two passes, matrix–vector products.
- Givens rotations on the Hessenberg matrix; the implicit residual decides when
  a cycle ends, the explicit residual ‖b − M x‖/‖b‖ ≤ `tol` decides convergence.
- `deflate = k > 0`: GMRES-DR (Morgan 2002), each restart keeps the k harmonic
  Ritz vectors of smallest |θ| of the last cycle.
- Stops at `maxiter` iterations or after `max_wall` seconds.

- `snapshots`: tolerances above `tol`; when the implicit residual first falls
  below one, the iterate at that point is formed (one extra product with the
  basis) and kept, with the iteration count and the elapsed time, so one solve
  gives the cost and the solution at several tolerances.
- `ritz = true`: harmonic Ritz values of every full cycle (the last is kept).
- `basis`: optional preallocated Krylov basis of size `(length(b), restart + 1)`
  and the element/array type of `b` (reused across calls by the Krylov branch
  of the linear-solve seam); by default a fresh one is allocated per call.
- `work`: optional preallocated work vectors `(; r, hbuf, d)` of the array type
  of `b` (`r` of length `length(b)`, `hbuf` and `d` of length ≥ `restart + 1`);
  with `basis` and `work` given (and `deflate = 0`, no snapshots) a solve makes
  no device allocation of its own.

Times exclude the diagnostics (snapshots, Ritz values), reported as `t_diag`.
Returns `(; converged, iters, cycles, relres, relres_cycles, relres_iter, t_iter,
t_total, t_op, t_orth, t_diag, n_apply, restart, orth, deflate, ritz, snaps)`;
`ritz` is empty unless `ritz = true`; `snaps` = vector of `(; tol, iters, t, x)`.
"""
function gnf_gmres!(x, apply!, b; tol::Real=1e-6, restart::Integer=1000,
                    maxiter::Integer=100_000, orth::Symbol=:mgs2, deflate::Integer=0,
                    max_wall::Real=Inf, ritz::Bool=false, verbose::Bool=false,
                    snapshots=(), basis=nothing, work=nothing)
    Tv = eltype(b)
    Nv = length(b)
    m = Int(restart)
    k = Int(deflate)
    0 <= k < m || throw(ArgumentError("need 0 ≤ deflate < restart"))
    t0 = time(); t_op = 0.0; t_orth = 0.0; t_diag = 0.0; n_apply = 0
    if basis === nothing
        V = similar(b, Nv, m + 1)
    else
        size(basis) == (Nv, m + 1) ||
            throw(DimensionMismatch("basis has size $(size(basis)), expected $((Nv, m + 1))"))
        V = basis
    end
    if work === nothing
        hbuf = similar(b, m + 1); r = similar(b); dbuf = nothing
    else
        (length(work.r) == Nv && length(work.hbuf) >= m + 1 && length(work.d) >= m) ||
            throw(DimensionMismatch("work vectors too short for n = $Nv, restart = $m"))
        hbuf = work.hbuf; r = work.r; dbuf = work.d
    end
    bnorm = norm(b)
    relres_cycles = Float64[]; relres_iter = Float64[]; t_iter = Float64[]
    ritz_vals = ComplexF64[]
    snap_tols = sort(collect(Float64, snapshots); rev=true)
    snaps = NamedTuple[]; next_snap = 1
    iters = 0; cycles = 0
    if bnorm == 0
        fill!(x, zero(Tv))
        return (; converged=true, iters=0, cycles=0, relres=0.0, relres_cycles, relres_iter, t_iter,
                t_total=0.0, t_op, t_orth, t_diag, n_apply, restart=m, orth, deflate=k,
                ritz=ritz_vals, snaps)
    end
    ta = time(); apply!(r, x); _gnf_sync(r); t_op += time() - ta; n_apply += 1
    r .= b .- r
    β = norm(r)
    push!(relres_cycles, β / bnorm)
    converged = β <= tol * bnorm

    Hbar = zeros(m + 1, m)       # unrotated (block) Hessenberg of the cycle
    R = zeros(m + 1, m)          # rotated columns
    g = zeros(m + 1)
    cvec = zeros(m + 1)          # residual coordinates in V at the cycle start
    cs = zeros(m); sn = zeros(m)
    kk = 0                       # deflated columns carried into the cycle
    Q1 = ones(1, 1)
    first_cycle = true
    while !converged && iters < maxiter && time() - t0 < max_wall
        cycles += 1
        if first_cycle || k == 0
            kk = 0
            view(V, :, 1) .= r ./ β
            fill!(Hbar, 0); fill!(cvec, 0); cvec[1] = β
            Q1 = ones(1, 1)
        end
        first_cycle = false
        fill!(R, 0); fill!(g, 0)
        g[1:kk+1] .= transpose(Q1) * cvec[1:kk+1]
        if kk > 0
            F = qr(Hbar[1:kk+1, 1:kk])
            R[1:kk, 1:kk] .= F.R
        end
        jlast = kk
        for j in kk+1:m
            vj = view(V, :, j); w = view(V, :, j + 1)
            ta = time(); apply!(w, vj); _gnf_sync(w); t_op += time() - ta; n_apply += 1
            ta = time()
            hcol = zeros(j + 1)
            hn = _gnf_orthogonalise!(hcol, V, w, j, orth, hbuf)
            hcol[j+1] = hn
            hn > 0 && (w ./= hn)
            _gnf_sync(w); t_orth += time() - ta
            Hbar[1:j+1, j] .= hcol
            # rotate: Q1ᵀ on rows 1..kk+1, then the Givens of columns kk+1..j-1
            rc = copy(hcol)
            rc[1:kk+1] .= transpose(Q1) * rc[1:kk+1]
            for i in kk+1:j-1
                tmp = cs[i] * rc[i] + sn[i] * rc[i+1]
                rc[i+1] = -sn[i] * rc[i] + cs[i] * rc[i+1]
                rc[i] = tmp
            end
            ρh = hypot(rc[j], rc[j+1])
            if ρh == 0
                cs[j] = 1.0; sn[j] = 0.0
            else
                cs[j] = rc[j] / ρh; sn[j] = rc[j+1] / ρh
            end
            rc[j] = ρh; rc[j+1] = 0.0
            R[1:j+1, j] .= rc
            g[j+1] = -sn[j] * g[j]; g[j] = cs[j] * g[j]
            iters += 1; jlast = j
            rel_now = abs(g[j+1]) / bnorm
            push!(relres_iter, rel_now); push!(t_iter, time() - t0 - t_diag)
            while next_snap <= length(snap_tols) && rel_now <= snap_tols[next_snap]
                # iterate at this point of the cycle (does not change the iteration;
                # its cost goes to t_diag, not to the solve times)
                td = time()
                ds = similar(b, j); copyto!(ds, UpperTriangular(R[1:j, 1:j]) \ g[1:j])
                xs = copy(x); mul!(xs, view(V, :, 1:j), ds, one(Tv), one(Tv)); _gnf_sync(xs)
                push!(snaps, (; tol=snap_tols[next_snap], iters, t=td - t0 - t_diag, x=xs))
                next_snap += 1
                t_diag += time() - td
            end
            if verbose && iters % 100 == 0
                @info "gnf_gmres" iters relres = abs(g[j+1]) / bnorm t = time() - t0
            end
            (abs(g[j+1]) <= tol * bnorm || iters >= maxiter || hn == 0 ||
             time() - t0 > max_wall) && break
        end
        d = UpperTriangular(R[1:jlast, 1:jlast]) \ g[1:jlast]
        if dbuf === nothing
            ddev = similar(b, jlast); copyto!(ddev, d)
        else
            copyto!(dbuf, 1, d, 1, jlast); ddev = view(dbuf, 1:jlast)
        end
        mul!(x, view(V, :, 1:jlast), ddev, one(Tv), one(Tv))
        ta = time(); apply!(r, x); _gnf_sync(r); t_op += time() - ta; n_apply += 1
        r .= b .- r
        β = norm(r)
        push!(relres_cycles, β / bnorm)
        converged = β <= tol * bnorm
        if ritz && jlast == m
            td = time()
            ritz_vals = eigvals(Hbar[1:m, 1:m] + Hbar[m+1, m]^2 *
                                (transpose(Hbar[1:m, 1:m]) \ [zeros(m - 1); 1.0]) *
                                transpose([zeros(m - 1); 1.0]))
            t_diag += time() - td
        end
        if k > 0 && !converged && jlast == m
            # deflated restart: keep k harmonic Ritz vectors + the residual direction
            G, _ = _gnf_harmonic_ritz(Hbar, m, k)
            kd = size(G, 2)
            kd + 1 <= m || (kd = m - 1; G = G[:, 1:kd])
            Pk = zeros(m + 1, kd); Pk[1:m, :] .= G
            Pk = Matrix(qr(Pk).Q)[:, 1:kd]
            resv = cvec - Hbar * d
            pr = resv - Pk * (transpose(Pk) * resv)
            pr = pr - Pk * (transpose(Pk) * pr)
            pr ./= norm(pr)
            P = hcat(Pk, pr)
            Hnew = transpose(P) * (Hbar * Pk[1:m, :])
            cnew = transpose(P) * resv
            Pdev = similar(b, m + 1, kd + 1); copyto!(Pdev, P)
            Vn = similar(b, Nv, kd + 1)
            mul!(Vn, view(V, :, 1:m+1), Pdev)
            view(V, :, 1:kd+1) .= Vn
            fill!(Hbar, 0); Hbar[1:kd+1, 1:kd] .= Hnew
            fill!(cvec, 0); cvec[1:kd+1] .= cnew
            kk = kd
            Q1 = Matrix(qr(Hnew).Q * Matrix{Float64}(I, kd + 1, kd + 1))
        elseif k > 0
            first_cycle = true      # stopped early: plain restart from the residual
        end
    end
    t_total = time() - t0 - t_diag
    return (; converged, iters, cycles, relres=relres_cycles[end], relres_cycles, relres_iter,
            t_iter, t_total, t_op, t_orth, t_diag, n_apply, restart=m, orth, deflate=k,
            ritz=ritz_vals, snaps)
end
