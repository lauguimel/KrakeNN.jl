using KernelAbstractions

# ===========================================================================
# VOF PLIC 2D — Sharp interface tracking for LBM
#
# Volume fraction C ∈ [0,1]: C=1 liquid, C=0 gas
# Interface reconstructed as a line n·x = d in each mixed cell (0 < C < 1)
#
# References:
# - Scardovelli & Zaleski (1999) doi:10.1146/annurev.fluid.31.1.567
# - Rider & Kothe (1998) doi:10.1006/jcph.1998.6029
# ===========================================================================

# ===================================================================
# Part 1: Interface normals (Youngs method, GPU kernel)
# ===================================================================

@kernel function compute_vof_normal_2d_kernel!(nx, ny, @Const(C), Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(C)
        # Central differences with periodic wrapping
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)
        jp = ifelse(j < Ny, j + 1, 1)
        jm = ifelse(j > 1,  j - 1, Ny)

        # Youngs' method: mixed central/corner differences for robustness
        # Convention: n = -∇C (outward from liquid, matching Basilisk geometry.h)
        # ∂C/∂x using Youngs stencil (weighted by y-neighbors)
        dCdx = (C[im,jm] + T(2)*C[im,j] + C[im,jp] -
                C[ip,jm] - T(2)*C[ip,j] - C[ip,jp]) / T(8)
        dCdy = (C[im,jm] + T(2)*C[i,jm] + C[ip,jm] -
                C[im,jp] - T(2)*C[i,jp] - C[ip,jp]) / T(8)

        # Normalize
        mag = sqrt(dCdx^2 + dCdy^2) + T(1e-30)
        nx[i,j] = dCdx / mag
        ny[i,j] = dCdy / mag
    end
end

"""
    compute_vof_normal_2d!(nx, ny, C, Nx, Ny)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.compute_vof_normal_2d!)
```
"""
function compute_vof_normal_2d!(nx, ny, C, Nx, Ny)
    backend = KernelAbstractions.get_backend(C)
    kernel! = compute_vof_normal_2d_kernel!(backend)
    kernel!(nx, ny, C, Nx, Ny; ndrange=(Nx, Ny))
end

# ===================================================================
# Part 2: PLIC geometry (exact Basilisk formulas)
# ===================================================================
#
# Cell coordinates: [-0.5, 0.5]² (cell-centered)
# Normal: n = (nx, ny), pointing from liquid to gas (Youngs)
# Alpha: line parameter such that n·x = alpha defines the interface
#
# References:
# - Scardovelli & Zaleski (2000) doi:10.1006/jcph.2000.6567
# - Weymouth & Yue (2010) doi:10.1016/j.jcp.2010.04.024
# - Basilisk: src/geometry.h, src/vof.h (Popinet 2009)

# --- _line_area: exact Basilisk line_area() ---
# Area of fluid below line n·x = α in unit cell [0,1]²

@inline function _line_area(nx::T, ny::T, alpha::T) where T
    a = alpha + (nx + ny) / T(2)
    n1 = nx; n2 = ny
    if n1 < zero(T); a -= n1; n1 = -n1; end
    if n2 < zero(T); a -= n2; n2 = -n2; end
    if a <= zero(T); return zero(T); end
    if a >= n1 + n2; return one(T); end
    if n1 < T(1e-10); return clamp(a / n2, zero(T), one(T)); end
    if n2 < T(1e-10); return clamp(a / n1, zero(T), one(T)); end
    v = a * a
    a1 = a - n1; if a1 > zero(T); v -= a1 * a1; end
    a2 = a - n2; if a2 > zero(T); v -= a2 * a2; end
    return clamp(v / (T(2) * n1 * n2), zero(T), one(T))
end

# --- _line_alpha: exact Basilisk line_alpha() ---
# Given C and (nx, ny), find α such that area below n·x = α equals C
# in cell [-0.5, 0.5]²

@inline function _line_alpha(c::T, nx::T, ny::T) where T
    n1 = abs(nx); n2 = abs(ny)
    if n1 > n2; n1, n2 = n2, n1; end
    cc = clamp(c, zero(T), one(T))
    v1 = n1 / T(2)
    if n2 < T(1e-10); return zero(T); end  # degenerate
    if cc <= v1 / n2
        alpha = sqrt(T(2) * cc * n1 * n2)
    elseif cc <= one(T) - v1 / n2
        alpha = cc * n2 + v1
    else
        alpha = n1 + n2 - sqrt(T(2) * n1 * n2 * (one(T) - cc))
    end
    if nx < zero(T); alpha += nx; end
    if ny < zero(T); alpha += ny; end
    return alpha - (nx + ny) / T(2)
end

# --- _rectangle_fraction: exact Basilisk rectangle_fraction() ---
# Fraction of fluid in rectangle [ax,bx] × [ay,by] given line n·x = α

@inline function _rectangle_fraction(nx::T, ny::T, alpha::T, ax::T, ay::T, bx::T, by::T) where T
    alpha_p = alpha - nx*(bx+ax)/T(2) - ny*(by+ay)/T(2)
    nx_p = nx * (bx - ax)
    ny_p = ny * (by - ay)
    return _line_area(nx_p, ny_p, alpha_p)
end

# --- plic_line_position: backward-compat wrapper ---

@inline plic_line_position(nx::T, ny::T, C::T) where T = _line_alpha(C, nx, ny)

# ===================================================================
# Part 3: PLIC advection kernels (Basilisk algorithm)
# ===================================================================
#
# Geometric PLIC advection with Weymouth-Yue (2010) correction.
# Follows the exact Basilisk algorithm from src/vof.h:
# - Face velocity = average of cell-centered velocities
# - Donor cell PLIC reconstruction → rectangle_fraction for strip flux
# - Weymouth-Yue: cc = (C > 0.5) ? 1 : 0 for compression term

@kernel function advect_vof_plic_x_2d_kernel!(C_new, @Const(C), @Const(cc_field),
                                                @Const(nx_n), @Const(ny_n),
                                                @Const(ux), Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(C)
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)

        # --- Right face velocity ---
        u_right = (ux[i, j] + ux[ip, j]) / T(2)

        # --- Right face flux (geometric PLIC) ---
        if u_right > zero(T)
            # Donor = cell (i,j), velocity sweeps rightward
            c_donor = C[i,j]
            if c_donor <= zero(T) || c_donor >= one(T)
                cf_right = c_donor
            else
                s = one(T)
                alpha_d = _line_alpha(c_donor, nx_n[i,j], ny_n[i,j])
                cf_right = _rectangle_fraction(-s*nx_n[i,j], ny_n[i,j], alpha_d,
                                               -T(0.5), -T(0.5), s*u_right - T(0.5), T(0.5))
            end
            flux_right = cf_right * u_right
        else
            # Donor = cell (ip,j), velocity sweeps leftward
            c_donor = C[ip,j]
            if c_donor <= zero(T) || c_donor >= one(T)
                cf_right = c_donor
            else
                s = -one(T)
                alpha_d = _line_alpha(c_donor, nx_n[ip,j], ny_n[ip,j])
                cf_right = _rectangle_fraction(-s*nx_n[ip,j], ny_n[ip,j], alpha_d,
                                               -T(0.5), -T(0.5), s*u_right - T(0.5), T(0.5))
            end
            flux_right = cf_right * u_right  # u_right < 0 → negative flux
        end

        # --- Left face velocity ---
        u_left = (ux[im, j] + ux[i, j]) / T(2)

        # --- Left face flux (geometric PLIC) ---
        if u_left > zero(T)
            # Donor = cell (im,j), velocity sweeps rightward
            c_donor = C[im,j]
            if c_donor <= zero(T) || c_donor >= one(T)
                cf_left = c_donor
            else
                s = one(T)
                alpha_d = _line_alpha(c_donor, nx_n[im,j], ny_n[im,j])
                cf_left = _rectangle_fraction(-s*nx_n[im,j], ny_n[im,j], alpha_d,
                                              -T(0.5), -T(0.5), s*u_left - T(0.5), T(0.5))
            end
            flux_left = cf_left * u_left
        else
            # Donor = cell (i,j), velocity sweeps leftward
            c_donor = C[i,j]
            if c_donor <= zero(T) || c_donor >= one(T)
                cf_left = c_donor
            else
                s = -one(T)
                alpha_d = _line_alpha(c_donor, nx_n[i,j], ny_n[i,j])
                cf_left = _rectangle_fraction(-s*nx_n[i,j], ny_n[i,j], alpha_d,
                                              -T(0.5), -T(0.5), s*u_left - T(0.5), T(0.5))
            end
            flux_left = cf_left * u_left  # u_left < 0 → negative flux
        end

        # Weymouth-Yue (2010) correction — cc frozen before all sweeps
        C_new[i, j] = C[i, j] - (flux_right - flux_left) + cc_field[i,j] * (u_right - u_left)
    end
end

@kernel function advect_vof_plic_y_2d_kernel!(C_new, @Const(C), @Const(cc_field),
                                                @Const(nx_n), @Const(ny_n),
                                                @Const(uy), Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(C)
        jp = ifelse(j < Ny, j + 1, 1)
        jm = ifelse(j > 1,  j - 1, Ny)

        # --- Top face velocity ---
        u_top = (uy[i, j] + uy[i, jp]) / T(2)

        # --- Top face flux (geometric PLIC) ---
        # Note: for y-direction, the rectangle extends in y (4th arg pair),
        # and the sign flip is on ny (not nx)
        if u_top > zero(T)
            # Donor = cell (i,j), velocity sweeps upward
            c_donor = C[i,j]
            if c_donor <= zero(T) || c_donor >= one(T)
                cf_top = c_donor
            else
                s = one(T)
                alpha_d = _line_alpha(c_donor, nx_n[i,j], ny_n[i,j])
                cf_top = _rectangle_fraction(nx_n[i,j], -s*ny_n[i,j], alpha_d,
                                             -T(0.5), -T(0.5), T(0.5), s*u_top - T(0.5))
            end
            flux_top = cf_top * u_top
        else
            # Donor = cell (i,jp), velocity sweeps downward
            c_donor = C[i,jp]
            if c_donor <= zero(T) || c_donor >= one(T)
                cf_top = c_donor
            else
                s = -one(T)
                alpha_d = _line_alpha(c_donor, nx_n[i,jp], ny_n[i,jp])
                cf_top = _rectangle_fraction(nx_n[i,jp], -s*ny_n[i,jp], alpha_d,
                                             -T(0.5), -T(0.5), T(0.5), s*u_top - T(0.5))
            end
            flux_top = cf_top * u_top  # u_top < 0 → negative flux
        end

        # --- Bottom face velocity ---
        u_bot = (uy[i, jm] + uy[i, j]) / T(2)

        # --- Bottom face flux (geometric PLIC) ---
        if u_bot > zero(T)
            # Donor = cell (i,jm), velocity sweeps upward
            c_donor = C[i,jm]
            if c_donor <= zero(T) || c_donor >= one(T)
                cf_bot = c_donor
            else
                s = one(T)
                alpha_d = _line_alpha(c_donor, nx_n[i,jm], ny_n[i,jm])
                cf_bot = _rectangle_fraction(nx_n[i,jm], -s*ny_n[i,jm], alpha_d,
                                             -T(0.5), -T(0.5), T(0.5), s*u_bot - T(0.5))
            end
            flux_bot = cf_bot * u_bot
        else
            # Donor = cell (i,j), velocity sweeps downward
            c_donor = C[i,j]
            if c_donor <= zero(T) || c_donor >= one(T)
                cf_bot = c_donor
            else
                s = -one(T)
                alpha_d = _line_alpha(c_donor, nx_n[i,j], ny_n[i,j])
                cf_bot = _rectangle_fraction(nx_n[i,j], -s*ny_n[i,j], alpha_d,
                                             -T(0.5), -T(0.5), T(0.5), s*u_bot - T(0.5))
            end
            flux_bot = cf_bot * u_bot  # u_bot < 0 → negative flux
        end

        # Weymouth-Yue (2010) correction — cc frozen before all sweeps
        C_new[i, j] = C[i, j] - (flux_top - flux_bot) + cc_field[i,j] * (u_top - u_bot)
    end
end

"""
    advect_vof_plic_2d!(C_new, C, nx_n, ny_n, cc_field, ux, uy, Nx, Ny; step=1)

Geometric PLIC advection of volume fraction C using directional splitting
with Basilisk-style `rectangle_fraction` flux computation and Weymouth-Yue
compression correction.

Requires pre-computed interface normals `(nx_n, ny_n)` from `compute_vof_normal_2d!`
and a pre-allocated `cc_field` work array (same size as C).
Strang alternating splitting: x→y on odd steps, y→x on even steps.
"""
function advect_vof_plic_2d!(C_new, C, nx_n, ny_n, cc_field, ux, uy, Nx, Ny; step::Int=1)
    backend = KernelAbstractions.get_backend(C)

    # Weymouth-Yue: freeze cc = (C > 0.5) ONCE before all sweeps
    _compute_cc_field!(cc_field, C, backend, Nx, Ny)

    # Strang alternating splitting: x→y on odd steps, y→x on even steps
    # This reduces directional bias for rotational flows
    if isodd(step)
        # x-sweep first
        kernel_x! = advect_vof_plic_x_2d_kernel!(backend)
        kernel_x!(C_new, C, cc_field, nx_n, ny_n, ux, Nx, Ny; ndrange=(Nx, Ny))
        KernelAbstractions.synchronize(backend)
        copyto!(C, C_new)
        compute_vof_normal_2d!(nx_n, ny_n, C, Nx, Ny)
        # y-sweep second
        kernel_y! = advect_vof_plic_y_2d_kernel!(backend)
        kernel_y!(C_new, C, cc_field, nx_n, ny_n, uy, Nx, Ny; ndrange=(Nx, Ny))
        KernelAbstractions.synchronize(backend)
    else
        # y-sweep first
        kernel_y! = advect_vof_plic_y_2d_kernel!(backend)
        kernel_y!(C_new, C, cc_field, nx_n, ny_n, uy, Nx, Ny; ndrange=(Nx, Ny))
        KernelAbstractions.synchronize(backend)
        copyto!(C, C_new)
        compute_vof_normal_2d!(nx_n, ny_n, C, Nx, Ny)
        # x-sweep second
        kernel_x! = advect_vof_plic_x_2d_kernel!(backend)
        kernel_x!(C_new, C, cc_field, nx_n, ny_n, ux, Nx, Ny; ndrange=(Nx, Ny))
        KernelAbstractions.synchronize(backend)
    end
end

# --- Helper: compute Weymouth-Yue cc field on GPU ---

@kernel function _cc_field_kernel!(cc, @Const(C))
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(C)
        cc[i,j] = ifelse(C[i,j] > T(0.5), one(T), zero(T))
    end
end

function _compute_cc_field!(cc, C, backend, Nx, Ny)
    kernel! = _cc_field_kernel!(backend)
    kernel!(cc, C; ndrange=(Nx, Ny))
end

# ===================================================================
# Part 4: MUSCL-Superbee advection (algebraic VOF advection)
# ===================================================================

# --- Superbee flux limiter (GPU-safe inline) ---

@inline function _superbee(r::T) where T
    # Superbee limiter: max(0, min(2r,1), min(r,2))
    # Most compressive TVD limiter — ideal for sharp VOF interfaces
    ifelse(r <= zero(T), zero(T),
        max(min(T(2) * r, one(T)), min(r, T(2))))
end

# --- MUSCL-Superbee VOF advection (directional splitting) ---
#
# Second-order TVD scheme with superbee flux limiter for sharp interfaces.
# Split advection: first x-direction, then y-direction.
# Conservative: total C preserved to machine precision.

@kernel function advect_vof_x_2d_kernel!(C_new, @Const(C), @Const(ux), Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(C)
        ip  = ifelse(i < Nx, i + 1, 1)
        im  = ifelse(i > 1,  i - 1, Nx)
        ipp = ifelse(ip < Nx, ip + 1, 1)
        imm = ifelse(im > 1,  im - 1, Nx)

        # Face velocities (arithmetic mean)
        u_right = (ux[i, j] + ux[ip, j]) / T(2)
        u_left  = (ux[im, j] + ux[i, j]) / T(2)

        # --- Right face flux (MUSCL-Superbee) ---
        # Upwind value and gradient ratio
        if u_right > zero(T)
            C_up = C[i, j]
            dC_down = C[ip, j] - C[i, j]
            dC_up   = C[i, j] - C[im, j]
        else
            C_up = C[ip, j]
            dC_down = C[i, j] - C[ip, j]
            dC_up   = C[ip, j] - C[ipp, j]
        end
        r_right = ifelse(abs(dC_down) > T(1e-30),
                         dC_up / dC_down, zero(T))
        phi_r = _superbee(r_right)
        flux_right = u_right * (C_up + T(0.5) * phi_r * dC_down * max(zero(T), one(T) - abs(u_right)))

        # --- Left face flux (MUSCL-Superbee) ---
        if u_left > zero(T)
            C_up = C[im, j]
            dC_down = C[i, j] - C[im, j]
            dC_up   = C[im, j] - C[imm, j]
        else
            C_up = C[i, j]
            dC_down = C[im, j] - C[i, j]
            dC_up   = C[i, j] - C[ip, j]
        end
        r_left = ifelse(abs(dC_down) > T(1e-30),
                        dC_up / dC_down, zero(T))
        phi_l = _superbee(r_left)
        flux_left = u_left * (C_up + T(0.5) * phi_l * dC_down * max(zero(T), one(T) - abs(u_left)))

        C_new[i, j] = C[i, j] - (flux_right - flux_left)
    end
end

@kernel function advect_vof_y_2d_kernel!(C_new, @Const(C), @Const(uy), Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(C)
        jp  = ifelse(j < Ny, j + 1, 1)
        jm  = ifelse(j > 1,  j - 1, Ny)
        jpp = ifelse(jp < Ny, jp + 1, 1)
        jmm = ifelse(jm > 1,  jm - 1, Ny)

        u_top = (uy[i, j] + uy[i, jp]) / T(2)
        u_bot = (uy[i, jm] + uy[i, j]) / T(2)

        # --- Top face flux ---
        if u_top > zero(T)
            C_up = C[i, j]
            dC_down = C[i, jp] - C[i, j]
            dC_up   = C[i, j] - C[i, jm]
        else
            C_up = C[i, jp]
            dC_down = C[i, j] - C[i, jp]
            dC_up   = C[i, jp] - C[i, jpp]
        end
        r_top = ifelse(abs(dC_down) > T(1e-30),
                       dC_up / dC_down, zero(T))
        phi_t = _superbee(r_top)
        flux_top = u_top * (C_up + T(0.5) * phi_t * dC_down * max(zero(T), one(T) - abs(u_top)))

        # --- Bottom face flux ---
        if u_bot > zero(T)
            C_up = C[i, jm]
            dC_down = C[i, j] - C[i, jm]
            dC_up   = C[i, jm] - C[i, jmm]
        else
            C_up = C[i, j]
            dC_down = C[i, jm] - C[i, j]
            dC_up   = C[i, j] - C[i, jp]
        end
        r_bot = ifelse(abs(dC_down) > T(1e-30),
                       dC_up / dC_down, zero(T))
        phi_b = _superbee(r_bot)
        flux_bot = u_bot * (C_up + T(0.5) * phi_b * dC_down * max(zero(T), one(T) - abs(u_bot)))

        C_new[i, j] = C[i, j] - (flux_top - flux_bot)
    end
end

"""
    advect_vof_2d!(C_new, C, ux, uy, Nx, Ny)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.advect_vof_2d!)
```
"""
function advect_vof_2d!(C_new, C, ux, uy, Nx, Ny)
    backend = KernelAbstractions.get_backend(C)
    # Strang splitting: x then y
    kernel_x! = advect_vof_x_2d_kernel!(backend)
    kernel_x!(C_new, C, ux, Nx, Ny; ndrange=(Nx, Ny))

    # Copy C_new → C for y-pass
    copyto!(C, C_new)

    kernel_y! = advect_vof_y_2d_kernel!(backend)
    kernel_y!(C_new, C, uy, Nx, Ny; ndrange=(Nx, Ny))
end

# ===================================================================
# Part 5: Height-function curvature
# ===================================================================
#
# For each interface cell, compute curvature from height function:
# κ = -h''/(1 + h'²)^{3/2}
# where h(x) = Σⱼ C(i, j_stencil) for a vertical column (if |ny| > |nx|)
# or h(y) = Σᵢ C(i_stencil, j) for a horizontal column

@kernel function compute_hf_curvature_2d_kernel!(κ, @Const(C), @Const(nx_n), @Const(ny_n),
                                                   Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(C)
        c = C[i,j]

        # Only compute curvature at interface cells
        if c > T(0.01) && c < T(0.99)
            if abs(ny_n[i,j]) >= abs(nx_n[i,j])
                # Interface more horizontal → use vertical columns h(x)
                # h(x) = Σ_j C(x, j) over a stencil of ±3 cells in y
                # h at i-1, i, i+1
                im = ifelse(i > 1, i - 1, Nx)
                ip = ifelse(i < Nx, i + 1, 1)

                h_m = zero(T); h_0 = zero(T); h_p = zero(T)
                for dj in -2:2
                    jj = j + dj
                    jj = ifelse(jj < 1, jj + Ny, ifelse(jj > Ny, jj - Ny, jj))
                    h_m += C[im, jj]
                    h_0 += C[i,  jj]
                    h_p += C[ip, jj]
                end

                # h' ≈ (h_p - h_m) / 2
                hp = (h_p - h_m) / T(2)
                # h'' ≈ h_p - 2h_0 + h_m
                hpp = h_p - T(2)*h_0 + h_m

                κ[i,j] = -hpp / (one(T) + hp^2)^T(1.5)
            else
                # Interface more vertical → use horizontal rows h(y)
                jm = ifelse(j > 1, j - 1, Ny)
                jp = ifelse(j < Ny, j + 1, 1)

                h_m = zero(T); h_0 = zero(T); h_p = zero(T)
                for di in -2:2
                    ii = i + di
                    ii = ifelse(ii < 1, ii + Nx, ifelse(ii > Nx, ii - Nx, ii))
                    h_m += C[ii, jm]
                    h_0 += C[ii, j]
                    h_p += C[ii, jp]
                end

                hp = (h_p - h_m) / T(2)
                hpp = h_p - T(2)*h_0 + h_m

                κ[i,j] = -hpp / (one(T) + hp^2)^T(1.5)
            end
        else
            κ[i,j] = zero(T)
        end
    end
end

"""
    compute_hf_curvature_2d!(κ, C, nx, ny, Nx, Ny)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.compute_hf_curvature_2d!)
```
"""
function compute_hf_curvature_2d!(κ, C, nx, ny, Nx, Ny)
    backend = KernelAbstractions.get_backend(C)
    kernel! = compute_hf_curvature_2d_kernel!(backend)
    kernel!(κ, C, nx, ny, Nx, Ny; ndrange=(Nx, Ny))
end

# ===================================================================
# Part 6: Surface tension, azimuthal curvature, two-phase collision
# ===================================================================

# --- CSF surface tension force: F = σ·κ·∇C ---

@kernel function compute_surface_tension_2d_kernel!(Fx, Fy, @Const(κ), @Const(C),
                                                      σ, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(C)
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)
        jp = ifelse(j < Ny, j + 1, 1)
        jm = ifelse(j > 1,  j - 1, Ny)

        # ∇C (central differences)
        dCdx = (C[ip,j] - C[im,j]) / T(2)
        dCdy = (C[i,jp] - C[i,jm]) / T(2)

        # CSF: F = σ·κ·∇C
        Fx[i,j] = σ * κ[i,j] * dCdx
        Fy[i,j] = σ * κ[i,j] * dCdy
    end
end

# --- Axisymmetric curvature: add azimuthal component κ₂ = n_r / r ---

@kernel function add_azimuthal_curvature_2d_kernel!(κ, @Const(C), @Const(ny_n), Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(κ)
        c = C[i,j]
        if c > T(0.01) && c < T(0.99)
            r = max(T(j) - T(0.5), one(T))  # clamp r ≥ 1 to avoid singularity
            # Azimuthal curvature: κ₂ = n_r / r where n_r = ny_n
            # Normal points outward (Basilisk convention), so n_r > 0 above axis
            κ_axi = ny_n[i,j] / r
            # Clamp total curvature to avoid instability
            κ_total = κ[i,j] + κ_axi
            κ[i,j] = clamp(κ_total, -T(0.5), T(0.5))
        end
    end
end

"""
    add_azimuthal_curvature_2d!(κ, C, ny_n, Ny)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.add_azimuthal_curvature_2d!)
```
"""
function add_azimuthal_curvature_2d!(κ, C, ny_n, Ny)
    backend = KernelAbstractions.get_backend(κ)
    Nx = size(κ, 1)
    kernel! = add_azimuthal_curvature_2d_kernel!(backend)
    kernel!(κ, C, ny_n, Ny; ndrange=(Nx, Ny))
end

"""
    compute_surface_tension_2d!(Fx, Fy, κ, C, σ, Nx, Ny)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.compute_surface_tension_2d!)
```
"""
function compute_surface_tension_2d!(Fx, Fy, κ, C, σ, Nx, Ny)
    backend = KernelAbstractions.get_backend(C)
    T = eltype(C)
    kernel! = compute_surface_tension_2d_kernel!(backend)
    kernel!(Fx, Fy, κ, C, T(σ), Nx, Ny; ndrange=(Nx, Ny))
end

# --- Two-phase collision: variable density and viscosity ---

@kernel function collide_twophase_2d_kernel!(f, @Const(C), @Const(Fx_st), @Const(Fy_st),
                                               @Const(is_solid), ρ_l, ρ_g, ν_l, ν_g)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            bounce_back_2d!(f, i, j)
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            c = C[i,j]
            # Interpolated density and viscosity
            ρ_local = c * ρ_l + (one(T) - c) * ρ_g
            ν_local = c * ν_l + (one(T) - c) * ν_g
            ω_local = one(T) / (T(3) * ν_local + T(0.5))

            # Surface tension force
            fx = Fx_st[i,j]
            fy = Fy_st[i,j]

            ρ_f = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ_f
            ux = ((f2-f4+f6-f7-f8+f9) + fx/T(2)) * inv_ρ
            uy = ((f3-f5+f6+f7-f8-f9) + fy/T(2)) * inv_ρ
            usq = ux*ux + uy*uy

            guo_pref = one(T) - ω_local / T(2)

            Sq=T(4.0/9.0)*((-ux)*fx+(-uy)*fy)*T(3)
            f[i,j,1]=f1-ω_local*(f1-feq_2d(Val(1), ρ_f, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((one(T)-ux)*fx+(-uy)*fy)*T(3)+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,2]=f2-ω_local*(f2-feq_2d(Val(2), ρ_f, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,3]=f3-ω_local*(f3-feq_2d(Val(3), ρ_f, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-one(T)-ux)*fx+(-uy)*fy)*T(3)+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,4]=f4-ω_local*(f4-feq_2d(Val(4), ρ_f, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/9.0)*((-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,5]=f5-ω_local*(f5-feq_2d(Val(5), ρ_f, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(ux+uy)*(fx+fy)*T(9)
            f[i,j,6]=f6-ω_local*(f6-feq_2d(Val(6), ρ_f, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(-ux+uy)*(-fx+fy)*T(9)
            f[i,j,7]=f7-ω_local*(f7-feq_2d(Val(7), ρ_f, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(-ux-uy)*(-fx-fy)*T(9)
            f[i,j,8]=f8-ω_local*(f8-feq_2d(Val(8), ρ_f, ux, uy, usq))+guo_pref*Sq

            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(ux-uy)*(fx-fy)*T(9)
            f[i,j,9]=f9-ω_local*(f9-feq_2d(Val(9), ρ_f, ux, uy, usq))+guo_pref*Sq
        end
    end
end

"""
    collide_twophase_2d!(f, C, Fx_st, Fy_st, is_solid; ρ_l=1.0, ρ_g=0.001, ν_l=0.1, ν_g=0.1)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.collide_twophase_2d!)
```
"""
function collide_twophase_2d!(f, C, Fx_st, Fy_st, is_solid; ρ_l=1.0, ρ_g=0.001, ν_l=0.1, ν_g=0.1)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    T = eltype(f)
    kernel! = collide_twophase_2d_kernel!(backend)
    kernel!(f, C, Fx_st, Fy_st, is_solid, T(ρ_l), T(ρ_g), T(ν_l), T(ν_g); ndrange=(Nx, Ny))
end

# --- Axisymmetric viscous correction: ν/r · ∂u_z/∂r added to axial force ---

@kernel function axisym_viscous_correction_2d_kernel!(Fz, @Const(uz), @Const(C), ν_l, ν_g, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(Fz)
        r = T(j) - T(0.5)

        # Local viscosity from VOF
        c = C[i,j]
        ν_local = c * ν_l + (one(T) - c) * ν_g

        jp = min(j + 1, Ny)
        jm = max(j - 1, 1)

        if j == 1
            # L'Hôpital: ν/r · ∂u/∂r → ν · ∂²u/∂r² at axis
            duz_dr = T(2) * (uz[i, 2] - uz[i, 1])
            Fz[i,j] = Fz[i,j] + ν_local * duz_dr
        else
            duz_dr = (uz[i, jp] - uz[i, jm]) / T(2)
            Fz[i,j] = Fz[i,j] + ν_local / r * duz_dr
        end
    end
end

"""
    add_axisym_viscous_correction_2d!(Fz, uz, C, ν_l, ν_g, Ny)

Add axisymmetric viscous correction ν(C)/r · ∂u_z/∂r to the axial force array.
This accounts for the extra viscous term in cylindrical coordinates that is absent
in the Cartesian D2Q9 collision.
"""
function add_axisym_viscous_correction_2d!(Fz, uz, C, ν_l, ν_g, Ny)
    backend = KernelAbstractions.get_backend(Fz)
    Nx = size(Fz, 1)
    T = eltype(Fz)
    kernel! = axisym_viscous_correction_2d_kernel!(backend)
    kernel!(Fz, uz, C, T(ν_l), T(ν_g), Ny; ndrange=(Nx, Ny))
end

# --- Set VOF at west boundary (i=1) from a 1D profile array ---

@kernel function set_vof_west_2d_kernel!(C, @Const(C_inlet))
    j = @index(Global)
    @inbounds C[1, j] = C_inlet[j]
end

"""
    set_vof_west_2d!(C, C_inlet)

Public function in the kernel-level LBM operation.
See the method definition below for argument requirements, array layout, and backend expectations. The bang suffix indicates that one or more array arguments are updated in-place.

```julia
using KrakeNN

methods(KrakeNN.set_vof_west_2d!)
```
"""
function set_vof_west_2d!(C, C_inlet)
    backend = KernelAbstractions.get_backend(C)
    Ny = length(C_inlet)
    kernel! = set_vof_west_2d_kernel!(backend)
    kernel!(C, C_inlet; ndrange=(Ny,))
end
