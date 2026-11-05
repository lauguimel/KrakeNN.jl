using SparseArrays

"""
    EhdPoissonSetup

Factorize-once cache for the EHD potential equation on the wall-node lattice.
The assembled CPU operator solves `laplacian(phi) = -q / eps` with unit lattice
spacing. Dirichlet plate rows are kept as identity rows; side rows use either
mirror Neumann stencils or cyclic periodic wrap. The mixed-BC matrix is routed
through `lin_factorize(...; spd=false)` because the identity boundary rows make
the full operator non-symmetric. CPU setups use UMFPACK and keep the historical
host `q`/`phi` transfer path. GPU setups use the same assembled host CSC matrix,
factorize once through the CUDSS extension, fill the RHS on device, and solve
with no per-step host transfer.
"""
struct EhdPoissonSetup
    Nx::Int
    Ny::Int
    eps::Float64
    xbc::Symbol
    cache::LinearSolveCache
    q_host::Matrix{Float64}
    phi_host::Matrix{Float64}
    rhs::Vector{Float64}
end

struct EhdPoissonSetupGPU{C<:LinearSolveCache,V<:AbstractVector{Float64}}
    Nx::Int
    Ny::Int
    eps::Float64
    xbc::Symbol
    cache::C
    rhs::V
    sol::V      # preallocated device solution (allocation-free per-step solve)
end

const _EHD_CUDSS_FALLBACK_WARNED = Ref(false)

@inline _ehd_poisson_k(i, j, Nx) = i + (j - 1) * Nx

function _ehd_poisson_matrix(Nx::Int, Ny::Int, xbc::Symbol)
    xbc in (:neumann, :periodic) ||
        throw(ArgumentError("xbc must be :neumann or :periodic."))

    rows = Int[]
    cols = Int[]
    vals = Float64[]
    sizehint!(rows, 5Nx * Ny)
    sizehint!(cols, 5Nx * Ny)
    sizehint!(vals, 5Nx * Ny)

    add!(r, c, v) = (push!(rows, r); push!(cols, c); push!(vals, v))
    for j in 1:Ny, i in 1:Nx
        k = _ehd_poisson_k(i, j, Nx)
        if j == 1 || j == Ny
            add!(k, k, 1.0)
        else
            add!(k, k, -4.0)
            add!(k, _ehd_poisson_k(i, j - 1, Nx), 1.0)
            add!(k, _ehd_poisson_k(i, j + 1, Nx), 1.0)
            if xbc === :periodic
                im = i == 1 ? Nx : i - 1
                ip = i == Nx ? 1 : i + 1
                add!(k, _ehd_poisson_k(im, j, Nx), 1.0)
                add!(k, _ehd_poisson_k(ip, j, Nx), 1.0)
            elseif i == 1
                add!(k, _ehd_poisson_k(2, j, Nx), 2.0)
            elseif i == Nx
                add!(k, _ehd_poisson_k(Nx - 1, j, Nx), 2.0)
            else
                add!(k, _ehd_poisson_k(i - 1, j, Nx), 1.0)
                add!(k, _ehd_poisson_k(i + 1, j, Nx), 1.0)
            end
        end
    end
    return sparse(rows, cols, vals, Nx * Ny, Nx * Ny)
end

"""
    ehd_poisson_setup(Nx, Ny, eps; xbc=:neumann, backend=nothing)

Assemble and factorize the EHD Poisson operator once. `xbc=:neumann` matches
the electroconvection box sidewalls; `xbc=:periodic` matches the hydrostatic
driver. CPU/default backends return the historical UMFPACK setup. GPU backends
route the same operator through the CUDSS extension when `using CUDA, CUDSS`
has loaded it, otherwise they warn once and fall back to the CPU setup.
"""
function ehd_poisson_setup(Nx::Integer, Ny::Integer, eps; xbc::Symbol = :neumann,
                           backend = nothing)
    Nx = Int(Nx)
    Ny = Int(Ny)
    Nx < 3 && throw(ArgumentError("Nx must be at least 3."))
    Ny < 3 && throw(ArgumentError("Ny must be at least 3."))
    eps64 = Float64(eps)
    eps64 > 0 || throw(ArgumentError("eps must be positive."))
    A = _ehd_poisson_matrix(Nx, Ny, xbc)
    if backend isa KernelAbstractions.GPU
        try
            cache = lin_factorize(A; backend = CUDABackendTag(), spd = false,
                                  pin_k0 = 0)
            rhs = KernelAbstractions.zeros(backend, Float64, Nx * Ny)
            sol = KernelAbstractions.zeros(backend, Float64, Nx * Ny)
            return EhdPoissonSetupGPU(Nx, Ny, eps64, xbc, cache, rhs, sol)
        catch err
            if occursin("Load CUDA and CUDSS", sprint(showerror, err))
                if !_EHD_CUDSS_FALLBACK_WARNED[]
                    @warn "CUDSS extension not loaded (`using CUDA, CUDSS`) — EHD :direct phi falls back to CPU UMFPACK with per-step host transfers"
                    _EHD_CUDSS_FALLBACK_WARNED[] = true
                end
            else
                rethrow()
            end
        end
    end
    cache = lin_factorize(A; backend = CPUBackendTag(), spd = false, pin_k0 = 0)
    return EhdPoissonSetup(Nx, Ny, eps64, xbc, cache, zeros(Float64, Nx, Ny),
                           zeros(Float64, Nx, Ny), zeros(Float64, Nx * Ny))
end

function _ehd_poisson_fill_rhs!(b, q, setup::EhdPoissonSetup)
    Nx = setup.Nx
    Ny = setup.Ny
    inv_eps = inv(setup.eps)
    @inbounds for j in 1:Ny, i in 1:Nx
        k = _ehd_poisson_k(i, j, Nx)
        b[k] = j == 1 ? 1.0 : (j == Ny ? 0.0 : -q[i, j] * inv_eps)
    end
    return b
end

@kernel function _ehd_poisson_fill_rhs_gpu_kernel!(b, @Const(q), inv_eps, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        k = i + (j - 1) * Nx
        b[k] = j == 1 ? 1.0 : (j == Ny ? 0.0 : -q[i, j] * inv_eps)
    end
end

function _ehd_poisson_fill_rhs_gpu!(b, q, setup::EhdPoissonSetupGPU)
    backend = KernelAbstractions.get_backend(b)
    kernel! = _ehd_poisson_fill_rhs_gpu_kernel!(backend)
    kernel!(b, q, inv(setup.eps), setup.Nx, setup.Ny;
            ndrange = (setup.Nx, setup.Ny))
    KernelAbstractions.synchronize(backend)
    return b
end

"""
    ehd_poisson_solve!(phi_device, setup, q_device)

Rebuild the RHS for `laplacian(phi) = -q/eps` and reuse the cached
factorization. CPU setups keep the historical host UMFPACK solve. GPU setups
fill the RHS on device, solve in place into the preallocated device solution
`setup.sol` (no device allocation per step), and copy it into `phi_device`.
"""
function ehd_poisson_solve!(phi_device, setup::EhdPoissonSetup, q_device)
    copyto!(setup.q_host, q_device)
    _ehd_poisson_fill_rhs!(setup.rhs, setup.q_host, setup)
    sol = lin_solve!(setup.cache, setup.rhs)
    copyto!(vec(setup.phi_host), sol)
    copyto!(phi_device, setup.phi_host)
    return phi_device
end

function ehd_poisson_solve!(phi_device, setup::EhdPoissonSetupGPU, q_device)
    _ehd_poisson_fill_rhs_gpu!(setup.rhs, q_device, setup)
    lin_solve!(setup.sol, setup.cache, setup.rhs)
    copyto!(phi_device, reshape(setup.sol, setup.Nx, setup.Ny))
    return phi_device
end
