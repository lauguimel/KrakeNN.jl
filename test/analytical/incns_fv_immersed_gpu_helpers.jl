# Shared small CPU/GPU fixture and GPU-only timing case. No test execution here.
using Kraken

function _c5gpu_problem(N; nu=0.1, mass=:M3)
    ib = fv_immersed_circle(N, N, 0.513, 0.487, 0.16; h=1 / N,
        closure=:quadratic, mass, pressure=:linear)
    return fv_steady_problem(N, N; h=1 / N, nu, conv=false, immersed=ib,
        bc=(W=:inlet, E=:outlet, S=:wall, N=:wall), wall_flux=0.0,
        boundary_velocity=(side, x, y) -> side === :W ? (0.4y * (1-y), 0.0) : (0.0, 0.0))
end

_c5gpu_objectives() = ((:residual_flux, FVWallForce(1; kind=:residual_flux)),
                       (:surface, FVWallForce(1; kind=:surface)))
