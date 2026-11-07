```@meta
EditURL = "17_poiseuille_rheology.jl"
```

# Non-Newtonian Poiseuille Flow

```@raw html
<DownloadMenu :files="[{label:'poiseuille_rheology.csv',href:'/downloads/poiseuille_rheology/poiseuille_rheology.csv'},{label:'poiseuille_rheology.py',href:'/downloads/poiseuille_rheology/poiseuille_rheology.py'}]" />
```

## Problem Statement

Channel flow of a non-Newtonian fluid driven by a uniform body force ``F_x``
between two parallel plates.  Unlike the Newtonian case
([Example 1](@ref)), the viscosity now depends on the local shear rate
``\dot\gamma``, leading to non-parabolic velocity profiles.

We consider two rheology models:

1. **Power-law** (shear-thinning, ``n = 0.7``): ``\eta(\dot\gamma) = K\,\dot\gamma^{\,n-1}``
2. **Bingham** (yield stress): ``\eta(\dot\gamma) = \tau_y\,(1 - e^{-m\,\dot\gamma})\,/\,\dot\gamma + \mu_p``

### Power-law analytical solution

For a power-law fluid in a channel of height ``H`` with body force ``F_x``,
the fully developed velocity profile is:

```math
u_x(y) = \frac{n}{n+1}\left(\frac{F_x}{K}\right)^{1/n}
\left[\left(\frac{H}{2}\right)^{(n+1)/n}
- \left|y - \frac{H}{2}\right|^{(n+1)/n}\right]
```

For ``n < 1`` (shear-thinning), the profile is blunter than a parabola:
the fluid flows faster near the centreline because its effective viscosity
decreases where the shear rate is largest (near the walls).

### Why this test matters

The non-Newtonian Poiseuille test validates:

1. **Strain-rate computation** --- ``\dot\gamma`` must be extracted accurately
   from the non-equilibrium part of the distribution functions.
2. **Local relaxation time** --- Each node has its own ``\omega(x,y)``
   computed from the rheology model; incorrect coupling produces wrong profiles.
3. **Stability of the implicit loop** --- The shear rate and viscosity are
   mutually dependent; the solver stores ``\tau`` from the previous step to
   bootstrap convergence.

For details on the rheology implementation see
[Theory --- Non-Newtonian Rheology](@ref).

---

## LBM Setup

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Lattice   | ---    | D2Q9  |
| Domain    | ``N_x \times N_y`` | ``4 \times 32`` (periodic in ``x``, walls in ``y``) |
| Consistency | ``K`` | 0.1 |
| Power-law index | ``n`` | 0.7 (shear-thinning) |
| Body force | ``F_x`` | ``10^{-4}`` |
| Collision | --- | BGK with local ``\omega(\dot\gamma)`` |
| Forcing   | --- | Guo discrete forcing |
| Wall BCs  | --- | Half-way bounce-back |
| Time steps | --- | 50 000 |

---

## Code --- Power-Law

The kernel [`collide_rheology_guo_2d!`](@ref) performs BGK collision with
Guo forcing and a shear-rate-dependent relaxation time.  The rheology model
is passed as a Julia struct; the compiler specialises the kernel for each
concrete type.

```julia
using KrakeNN

Nx, Ny = 4, 32
K   = 0.1
n   = 0.7
Fx  = 1e-4
max_steps = 50000

rheology = PowerLaw(K, n)

# Initialize LBM arrays
f_in  = zeros(Float64, Nx, Ny, 9)
f_out = zeros(Float64, Nx, Ny, 9)
is_solid = falses(Nx, Ny)
tau_field = fill(3.0 * K + 0.5, Nx, Ny)   # initial guess for τ

for j in 1:Ny, i in 1:Nx, q in 1:9
    f_in[i,j,q] = KrakeNN.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
end
copy!(f_out, f_in)

# Time stepping
Fx_arr = fill(Float64(Fx), Nx, Ny)
Fy_arr = zeros(Float64, Nx, Ny)

for step in 1:max_steps
    stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
    collide_rheology_guo_2d!(f_out, is_solid, Fx_arr, Fy_arr, rheology, tau_field)
    f_in, f_out = f_out, f_in
end
```

---

## Results --- Velocity Profile

The power-law profile is blunter than the Newtonian parabola.  Near the
walls, where ``\dot\gamma`` is large, the effective viscosity drops and the
fluid accelerates; near the centre, where ``\dot\gamma \to 0``, the
viscosity rises and the profile flattens.

```julia
H = Float64(Ny)
u_analytical = zeros(Ny)
for j in 1:Ny
    y = j - 0.5
    dist = abs(y - H / 2)
    u_analytical[j] = n / (n + 1) * (Fx / K)^(1 / n) *
                       ((H / 2)^((n + 1) / n) - dist^((n + 1) / n))
end
```

![Power-law Poiseuille velocity profile.  Blue line: analytical solution for n = 0.7.  Orange dots: LBM with collide_rheology_guo_2d!.  The blunted profile is correctly captured.](poiseuille_rheology_profile.svg)

---

## Code --- Bingham

Switching to a Bingham fluid requires only changing the rheology struct.
The yield stress ``\tau_y`` creates a plug-flow region around the centreline
where ``\dot\gamma = 0``.

```julia
rheology_bingham = Bingham(1e-4, 0.1; m_reg=1000.0)
```

The simulation loop is identical: `collide_rheology_guo_2d!` dispatches on
the new model type at compile time.

---

## References

- [Theory --- Non-Newtonian Rheology](@ref) (page 14)
- [Guo *et al.* (2002)](@cite guo2002discrete) --- Discrete forcing scheme
- [Kruger *et al.* (2017)](@cite kruger2017lattice) --- The Lattice Boltzmann Method

```julia
nothing #hide
```

