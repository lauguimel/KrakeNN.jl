using KernelAbstractions

# --- Transpose of the periodic-x / wall-y pull stream (steady GNF adjoint) ---
#
# `stream_periodic_x_wall_y_2d!` is a permutation S of the stored populations:
# the arrival slot (i, j, q) pulls the stored component src(i, j, q) (periodic
# neighbour in x, half-way bounce-back at the walls j = 1 and j = Ny, where a
# population leaving the domain comes back reversed at the same node). Being a
# permutation, Sᵀ = S⁻¹: the stored component s receives the arrival slot
# dest(s) with src(dest(s)) = s, the "push" along the same links.
#
# `_spxwy_src` and `_spxwy_dest` return linear indices into an (Nx, Ny, Q)
# array (Q ≥ 9; the steady adjoint uses Q = 10, component 10 = τ), so that the
# flat-vector kernels of `ad/gnf_adjoint.jl` share them. They are written out
# case by case from the production kernel (`kernels/stream_periodic_2d.jl`);
# test/ad/gnf_adjoint_test.jl checks S·Sᵀ = I and ⟨S a, b⟩ = ⟨a, Sᵀ b⟩ against
# the production stream.

@inline _spxwy_lin(i, j, q, Nx, Ny) = i + Nx * ((j - 1) + Ny * (q - 1))

# Linear index of the stored component that the arrival slot (i, j, q) pulls.
@inline function _spxwy_src(i, j, q, Nx, Ny)
    im = ifelse(i > 1, i - 1, Nx)
    ip = ifelse(i < Nx, i + 1, 1)
    if q == 1
        return _spxwy_lin(i, j, 1, Nx, Ny)
    elseif q == 2
        return _spxwy_lin(im, j, 2, Nx, Ny)
    elseif q == 3
        return j > 1 ? _spxwy_lin(i, j - 1, 3, Nx, Ny) : _spxwy_lin(i, j, 5, Nx, Ny)
    elseif q == 4
        return _spxwy_lin(ip, j, 4, Nx, Ny)
    elseif q == 5
        return j < Ny ? _spxwy_lin(i, j + 1, 5, Nx, Ny) : _spxwy_lin(i, j, 3, Nx, Ny)
    elseif q == 6
        return j > 1 ? _spxwy_lin(im, j - 1, 6, Nx, Ny) : _spxwy_lin(i, j, 8, Nx, Ny)
    elseif q == 7
        return j > 1 ? _spxwy_lin(ip, j - 1, 7, Nx, Ny) : _spxwy_lin(i, j, 9, Nx, Ny)
    elseif q == 8
        return j < Ny ? _spxwy_lin(ip, j + 1, 8, Nx, Ny) : _spxwy_lin(i, j, 6, Nx, Ny)
    else # q == 9
        return j < Ny ? _spxwy_lin(im, j + 1, 9, Nx, Ny) : _spxwy_lin(i, j, 7, Nx, Ny)
    end
end

# Linear index of the arrival slot that the stored component (i, j, q) feeds.
@inline function _spxwy_dest(i, j, q, Nx, Ny)
    im = ifelse(i > 1, i - 1, Nx)
    ip = ifelse(i < Nx, i + 1, 1)
    if q == 1
        return _spxwy_lin(i, j, 1, Nx, Ny)
    elseif q == 2
        return _spxwy_lin(ip, j, 2, Nx, Ny)
    elseif q == 3
        return j < Ny ? _spxwy_lin(i, j + 1, 3, Nx, Ny) : _spxwy_lin(i, j, 5, Nx, Ny)
    elseif q == 4
        return _spxwy_lin(im, j, 4, Nx, Ny)
    elseif q == 5
        return j > 1 ? _spxwy_lin(i, j - 1, 5, Nx, Ny) : _spxwy_lin(i, j, 3, Nx, Ny)
    elseif q == 6
        return j < Ny ? _spxwy_lin(ip, j + 1, 6, Nx, Ny) : _spxwy_lin(i, j, 8, Nx, Ny)
    elseif q == 7
        return j < Ny ? _spxwy_lin(im, j + 1, 7, Nx, Ny) : _spxwy_lin(i, j, 9, Nx, Ny)
    elseif q == 8
        return j > 1 ? _spxwy_lin(im, j - 1, 8, Nx, Ny) : _spxwy_lin(i, j, 6, Nx, Ny)
    else # q == 9
        return j > 1 ? _spxwy_lin(ip, j - 1, 9, Nx, Ny) : _spxwy_lin(i, j, 7, Nx, Ny)
    end
end

@kernel function stream_periodic_x_wall_y_transpose_2d_kernel!(f_out, @Const(f_in), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds for q in 1:9
        f_out[_spxwy_lin(i, j, q, Nx, Ny)] = f_in[_spxwy_dest(i, j, q, Nx, Ny)]
    end
end

"""
    stream_periodic_x_wall_y_transpose_2d!(f_out, f_in, Nx, Ny)

Transpose (= inverse) of `stream_periodic_x_wall_y_2d!`: the stored component
(i, j, q) of `f_out` receives the arrival slot of `f_in` that it feeds in the
forward stream (a "push" along the same periodic / half-way bounce-back links).
`f_in`, `f_out`: (Nx, Ny, 9) arrays on any KernelAbstractions backend, distinct.
Used by the steady GNF adjoint (`ad/gnf_adjoint.jl`).
"""
function stream_periodic_x_wall_y_transpose_2d!(f_out, f_in, Nx, Ny)
    backend = KernelAbstractions.get_backend(f_in)
    stream_periodic_x_wall_y_transpose_2d_kernel!(backend)(f_out, f_in, Nx, Ny; ndrange=(Nx, Ny))
    return f_out
end
