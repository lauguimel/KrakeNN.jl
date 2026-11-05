# Shared operators for the C0 linear-solve seam tests (CPU:
# linear_solve_refactor_transpose.jl, GPU: linear_solve_cudss_c0_gpu.jl, and
# benchmarks/cudss_c0/c0_cudss_bench.jl). Needs SparseArrays and LinearAlgebra
# in scope.

# Non-symmetric 2D convection-diffusion (upwind) on an n x n grid, Dirichlet.
function _c0_convdiff(n::Int; a::Float64 = 1.0, nu::Float64 = 0.05)
    h = 1.0 / (n + 1)
    k(i, j) = i + (j - 1) * n
    I = Int[]; J = Int[]; V = Float64[]
    for j in 1:n, i in 1:n
        r = k(i, j)
        push!(I, r); push!(J, r); push!(V, 4nu / h^2 + 2a / h)
        i > 1 && (push!(I, r); push!(J, k(i - 1, j)); push!(V, -nu / h^2 - a / h))
        i < n && (push!(I, r); push!(J, k(i + 1, j)); push!(V, -nu / h^2))
        j > 1 && (push!(I, r); push!(J, k(i, j - 1)); push!(V, -nu / h^2 - a / h))
        j < n && (push!(I, r); push!(J, k(i, j + 1)); push!(V, -nu / h^2))
    end
    return sparse(I, J, V, n^2, n^2)
end

# Same pattern, new values (what a Newton step does to a Jacobian).
_c0_newvalues(A::SparseMatrixCSC, s::Float64) =
    SparseMatrixCSC(A.m, A.n, copy(A.colptr), copy(rowvals(A)),
                    nonzeros(A) .* (1 .+ s .* sin.(1:nnz(A))))

_c0_spd(n::Int) = (L = _c0_convdiff(n; a = 0.0); L + L' + 1e-3I)

_c0_relres(A, x, b) = norm(A * x - b) / norm(b)

# Singular operator (zero row sums, all-Neumann-like): needs pin_k0 > 0.
_c0_singular(n::Int) = (L = _c0_convdiff(n; a = 0.0); L - spdiagm(0 => vec(sum(L; dims = 2))))
