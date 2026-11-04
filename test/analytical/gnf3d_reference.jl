# Independent planar GNF references: momentum balance gives |tau| = F*s.
# Velocity is the integral of the inverse stress law, with no quadrature
# and no calls to Kraken's production viscosity/strain-rate implementation.

function _gnf3d_carreau_stress(gamma, eta0, etainf, lambda, n)
    etainf * gamma + (eta0 - etainf) * gamma *
        (1 + (lambda * gamma)^2)^((n - 1) / 2)
end

function _gnf3d_carreau_primitive(gamma, eta0, etainf, lambda, n)
    etainf * gamma^2 / 2 + (eta0 - etainf) *
        (1 + (lambda * gamma)^2)^((n + 1) / 2) / (lambda^2 * (n + 1))
end

function _gnf3d_carreau_inverse(tau, gamma_wall, eta0, etainf, lambda, n)
    iszero(tau) && return zero(tau)
    lo, hi = zero(gamma_wall), gamma_wall
    # Monotone stress for the positive n used here. Stop at adjacent floats.
    for _ in 1:128
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && return mid
        if _gnf3d_carreau_stress(mid, eta0, etainf, lambda, n) < tau
            lo = mid
        else
            hi = mid
        end
    end
    error("Carreau stress bisection did not reach floating-point resolution")
end

function _gnf3d_reference(law::Symbol, N::Int, q_w::Real; smooth=false)
    H = N - 1 + 2q_w
    h, Href = H / 2, 15 + 2q_w
    positions = collect(0:N-1) .+ q_w
    if law in (:power_thin, :power_thick)
        n = law === :power_thin ? 0.7 : 1.5
        gamma_wall = 0.02 * (n + 1) / (n * Href / 2) * (Href / H)^2
        K = 0.1 * gamma_wall^(1 - n)
        rheology = PowerLaw(K, n; nu_min=1e-3, nu_max=1.0)
        tau_wall = K * gamma_wall^n
        if smooth
            # Positive stress throughout removes the centreline singularity:
            # tau(z)=tau_wall-F*z, with tau(H)=tau_wall/2.
            F = tau_wall / (2H)
            u_at(z) = n / (n + 1) / (F * K^(1/n)) *
                (tau_wall^(1+1/n) - (tau_wall-F*z)^(1+1/n))
            gamma = ((tau_wall .- F .* positions) ./ K).^(1/n)
            u, u_top = u_at.(positions), u_at(H)
        else
            F = tau_wall / h
            s = abs.(positions .- h)
            gamma = (F .* s ./ K).^(1/n)
            u = (F / K)^(1/n) * n / (n + 1) .*
                (h^(1+1/n) .- s.^(1+1/n))
            u_top = 0.0
        end
        nu = K .* gamma.^(n - 1)
    elseif law === :carreau
        smooth && error("Smooth-stress fallback is only used for power laws")
        eta0, etainf, n = 0.3, 0.01, 0.5
        gamma_wall = 2.5e-3 * (Href / H)^2
        lambda = 10 / gamma_wall
        rheology = CarreauYasuda(eta0, etainf, lambda, 2.0, n)
        stress(gamma) = _gnf3d_carreau_stress(gamma, eta0, etainf, lambda, n)
        primitive(gamma) = _gnf3d_carreau_primitive(gamma, eta0, etainf, lambda, n)
        tau_wall = stress(gamma_wall)
        F = tau_wall / h
        s = abs.(positions .- h)
        gamma = [_gnf3d_carreau_inverse(F * d, gamma_wall, eta0, etainf, lambda, n)
                 for d in s]
        moment(gamma) = gamma * stress(gamma) - primitive(gamma)
        u = (moment(gamma_wall) .- moment.(gamma)) ./ F
        nu = etainf .+ (eta0 - etainf) .* (1 .+ (lambda .* gamma).^2).^((n-1)/2)
        u_top = 0.0
    else
        throw(ArgumentError("Unknown GNF reference law: $law"))
    end
    return (; rheology, N, q_w, H, positions, F, u, nu, u_top)
end

_gnf3d_rel_l2(actual, expected) = sqrt(sum(abs2, actual .- expected) / sum(abs2, expected))

function _gnf3d_inside_bounds(values, lower, upper)
    T = eltype(values)
    # Exclude an active clamp even when inverse-rate reconstruction moves its
    # value a few ulps inside the interval. This tightens the physical bounds.
    margin(bound) = 32eps(T) * max(abs(T(bound)), one(T))
    return all((lower + margin(lower)) .< values .< (upper - margin(upper)))
end
