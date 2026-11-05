# Shared by the C2 Newton tests (CPU and GPU) and benchmarks/fv_c2: Ghia,
# Ghia & Shin (1982) cavity centreline data (J. Comput. Phys. 48, 387-411,
# doi:10.1016/0021-9991(82)90058-4; same tables as incns_cavity_mg_ghia.jl)
# and the profile error of an FV state against them.

const _FVC2_GHIA_Y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                      0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
                      0.9531, 0.9609, 0.9688, 0.9766, 1.0000]
const _FVC2_GHIA_X = [0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
                      0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
                      0.9453, 0.9531, 0.9609, 0.9688, 1.0000]
const _FVC2_GHIA = Dict(
    100.0 => ([0.0000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
               -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
               0.68717, 0.73722, 0.78871, 0.84123, 1.00000],
              [0.0000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
               0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
               -0.10313, -0.08864, -0.07391, -0.05906, 0.00000]),
    1000.0 => ([0.0000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
                -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304,
                0.46604, 0.51117, 0.57492, 0.65928, 1.00000],
               [0.0000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095,
                0.33075, 0.32235, 0.02526, -0.31966, -0.42665, -0.51550,
                -0.39188, -0.33714, -0.27669, -0.21388, 0.00000]))

_fvc2_lid(s, x, y) = s === :N ? (1.0, 0.0) : (0.0, 0.0)

# piecewise-linear interpolation of cell-centred values with wall values at 0, 1
function _fvc2_interp(vals, q, lo, hi)
    N = length(vals)
    xs = [(k - 0.5) / N for k in 1:N]
    q <= xs[1] && return lo + (vals[1] - lo) * q / xs[1]
    q >= xs[end] && return vals[end] + (hi - vals[end]) * (q - xs[end]) / (1 - xs[end])
    k = findfirst(>=(q), xs)
    t = (q - xs[k-1]) / (xs[k] - xs[k-1])
    return (1 - t) * vals[k-1] + t * vals[k]
end

# max |u - u_Ghia| on x = 1/2 and max |v - v_Ghia| on y = 1/2 (lid velocity 1);
# N even, the centrelines are the means of the two straddling cell columns/rows
function _fvc2_ghia_error(u, v, Re)
    N = size(u, 1)
    uc = (u[N ÷ 2, :] .+ u[N ÷ 2 + 1, :]) ./ 2
    vc = (v[:, N ÷ 2] .+ v[:, N ÷ 2 + 1]) ./ 2
    ug, vg = _FVC2_GHIA[Float64(Re)]
    eu = maximum(abs(_fvc2_interp(uc, y, 0.0, 1.0) - ug[k]) for (k, y) in enumerate(_FVC2_GHIA_Y))
    ev = maximum(abs(_fvc2_interp(vc, x, 0.0, 0.0) - vg[k]) for (k, x) in enumerate(_FVC2_GHIA_X))
    return eu, ev
end
