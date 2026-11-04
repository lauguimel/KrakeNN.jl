# --- GPU-compatible symmetric 3x3 spectral operations ---
#
# Symmetric matrices are stored as (a11, a12, a13, a22, a23, a33).
# The eigensolver uses a fixed-sweep Jacobi diagonalization: no allocation,
# no dynamic stencil, and predictable GPU control flow.

@inline function _jacobi_rotate_sym3(
    a11, a12, a13, a22, a23, a33,
    v11, v21, v31, v12, v22, v32, v13, v23, v33,
    pair::Val{P},
) where {P}
    T = typeof(a11)
    if P == 12
        app = a11; aqq = a22; apq = a12
    elseif P == 13
        app = a11; aqq = a33; apq = a13
    else
        app = a22; aqq = a33; apq = a23
    end

    if abs(apq) <= eps(T) * (abs(app) + abs(aqq) + one(T))
        return a11, a12, a13, a22, a23, a33,
               v11, v21, v31, v12, v22, v32, v13, v23, v33
    end

    tau = (aqq - app) / (T(2) * apq)
    sigma = ifelse(tau >= zero(T), one(T), -one(T))
    t = sigma / (abs(tau) + sqrt(one(T) + tau * tau))
    c = inv(sqrt(one(T) + t * t))
    s = t * c

    app_new = app - t * apq
    aqq_new = aqq + t * apq

    if P == 12
        a13_new = c * a13 - s * a23
        a23_new = s * a13 + c * a23
        a11 = app_new; a12 = zero(T); a13 = a13_new
        a22 = aqq_new; a23 = a23_new

        tv11 = c * v11 - s * v12; tv12 = s * v11 + c * v12
        tv21 = c * v21 - s * v22; tv22 = s * v21 + c * v22
        tv31 = c * v31 - s * v32; tv32 = s * v31 + c * v32
        v11 = tv11; v12 = tv12
        v21 = tv21; v22 = tv22
        v31 = tv31; v32 = tv32
    elseif P == 13
        a12_new = c * a12 - s * a23
        a23_new = s * a12 + c * a23
        a11 = app_new; a12 = a12_new; a13 = zero(T)
        a23 = a23_new; a33 = aqq_new

        tv11 = c * v11 - s * v13; tv13 = s * v11 + c * v13
        tv21 = c * v21 - s * v23; tv23 = s * v21 + c * v23
        tv31 = c * v31 - s * v33; tv33 = s * v31 + c * v33
        v11 = tv11; v13 = tv13
        v21 = tv21; v23 = tv23
        v31 = tv31; v33 = tv33
    else
        a12_new = c * a12 - s * a13
        a13_new = s * a12 + c * a13
        a12 = a12_new; a13 = a13_new
        a22 = app_new; a23 = zero(T); a33 = aqq_new

        tv12 = c * v12 - s * v13; tv13 = s * v12 + c * v13
        tv22 = c * v22 - s * v23; tv23 = s * v22 + c * v23
        tv32 = c * v32 - s * v33; tv33 = s * v32 + c * v33
        v12 = tv12; v13 = tv13
        v22 = tv22; v23 = tv23
        v32 = tv32; v33 = tv33
    end

    return a11, a12, a13, a22, a23, a33,
           v11, v21, v31, v12, v22, v32, v13, v23, v33
end

@inline function eigen_sym3x3(a11, a12, a13, a22, a23, a33)
    T = typeof(a11)
    v11 = one(T);  v21 = zero(T); v31 = zero(T)
    v12 = zero(T); v22 = one(T);  v32 = zero(T)
    v13 = zero(T); v23 = zero(T); v33 = one(T)

    for _ in 1:8
        a11, a12, a13, a22, a23, a33,
        v11, v21, v31, v12, v22, v32, v13, v23, v33 =
            _jacobi_rotate_sym3(
                a11, a12, a13, a22, a23, a33,
                v11, v21, v31, v12, v22, v32, v13, v23, v33,
                Val(12),
            )
        a11, a12, a13, a22, a23, a33,
        v11, v21, v31, v12, v22, v32, v13, v23, v33 =
            _jacobi_rotate_sym3(
                a11, a12, a13, a22, a23, a33,
                v11, v21, v31, v12, v22, v32, v13, v23, v33,
                Val(13),
            )
        a11, a12, a13, a22, a23, a33,
        v11, v21, v31, v12, v22, v32, v13, v23, v33 =
            _jacobi_rotate_sym3(
                a11, a12, a13, a22, a23, a33,
                v11, v21, v31, v12, v22, v32, v13, v23, v33,
                Val(23),
            )
    end

    return a11, a22, a33,
           v11, v21, v31,
           v12, v22, v32,
           v13, v23, v33
end

@inline function _sym3_from_eigenvalues(
    f1, f2, f3,
    v11, v21, v31,
    v12, v22, v32,
    v13, v23, v33,
)
    b11 = f1 * v11 * v11 + f2 * v12 * v12 + f3 * v13 * v13
    b12 = f1 * v11 * v21 + f2 * v12 * v22 + f3 * v13 * v23
    b13 = f1 * v11 * v31 + f2 * v12 * v32 + f3 * v13 * v33
    b22 = f1 * v21 * v21 + f2 * v22 * v22 + f3 * v23 * v23
    b23 = f1 * v21 * v31 + f2 * v22 * v32 + f3 * v23 * v33
    b33 = f1 * v31 * v31 + f2 * v32 * v32 + f3 * v33 * v33
    return b11, b12, b13, b22, b23, b33
end

@inline function mat_exp_sym3x3(a11, a12, a13, a22, a23, a33)
    λ1, λ2, λ3,
    v11, v21, v31,
    v12, v22, v32,
    v13, v23, v33 = eigen_sym3x3(a11, a12, a13, a22, a23, a33)
    return _sym3_from_eigenvalues(
        exp(λ1), exp(λ2), exp(λ3),
        v11, v21, v31, v12, v22, v32, v13, v23, v33,
    )
end

@inline function mat_log_spd_sym3x3(a11, a12, a13, a22, a23, a33)
    λ1, λ2, λ3,
    v11, v21, v31,
    v12, v22, v32,
    v13, v23, v33 = eigen_sym3x3(a11, a12, a13, a22, a23, a33)
    return _sym3_from_eigenvalues(
        log(λ1), log(λ2), log(λ3),
        v11, v21, v31, v12, v22, v32, v13, v23, v33,
    )
end
