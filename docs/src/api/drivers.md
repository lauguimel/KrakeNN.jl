# Drivers

Drivers are the high-level entry points that wire collision,
streaming, boundaries and I/O into a complete simulation. Each one
accepts keyword arguments (grid size, Reynolds, number of steps,
output directory…) and returns the final macroscopic fields plus a
diagnostics table. They are the recommended starting point for
Julia users — the `.krk` DSL ultimately dispatches to the same
functions via `run_simulation`.


## Quick reference

| Symbol | Purpose |
|--------|---------|
| `run_cavity_2d` | Lid-driven cavity driver — 2D |
| `run_cavity_3d` | Lid-driven cavity driver — 3D |
| `run_poiseuille_2d` | Poiseuille channel driver — 2D |
| `run_couette_2d` | Couette flow driver — 2D |
| `initialize_taylor_green_2d` | Taylor-Green analytic initial condition |
| `run_taylor_green_2d` | Taylor-Green vortex driver — 2D |
| `initialize_cylinder_2d` | Cylinder obstacle mask initialiser |
| `run_cylinder_2d` | Flow past a cylinder driver — 2D |
| `compute_drag_mea_2d` | Drag coefficient via momentum-exchange (MEA) |
| `run_rayleigh_benard_2d` | Rayleigh-Bénard convection driver |
| `run_natural_convection_2d` | Natural convection driver — 2D |
| `run_natural_convection_refined_2d` | Natural convection with nested grid refinement (under development, not validated) |
| `fused_natconv_step!` | Fused natconv collide+stream step |
| `fused_natconv_vt_step!` | Fused natconv step (v/T formulation) |
| `fused_bgk_step!` | Fused single-phase BGK collide+stream step |
| `aa_even_step!` | A-A pattern: even sub-step |
| `aa_odd_step!` | A-A pattern: odd sub-step |
| `persistent_fused_bgk!` | Persistent-kernel GPU BGK (experimental) |
| `persistent_aa_bgk!` | Persistent-kernel GPU A-A BGK (experimental) |
| `run_hagen_poiseuille_2d` | Axisymmetric Hagen-Poiseuille pipe driver |
| `benchmark_mlups` | Throughput micro-benchmark (MLUPS) |
| `run_simulation` | Top-level dispatcher (used by the .krk DSL) |

## Details

### `run_cavity_2d`

**Source:** `src/drivers/basic.jl`

```julia
"""
    run_cavity_2d(config::LBMConfig{D2Q9}; backend=CPU(), T=Float64)

Run a 2D lid-driven cavity simulation using D2Q9 BGK-LBM.

Returns a NamedTuple with final fields (ρ, ux, uy) on CPU.
"""
function run_cavity_2d(config::LBMConfig{D2Q9};
                        backend=KernelAbstractions.CPU(), T=Float64)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    Nx, Ny = config.Nx, config.Ny
    ω = T(omega(config))

    for step in 1:config.max_steps
        # 1. Stream (pull scheme, bounce-back at domain edges)
        stream_2d!(f_out, f_in, Nx, Ny)

        # 2. Apply Zou-He velocity BC on lid (post-stream, pre-collision)
        apply_zou_he_north_2d!(f_out, config.u_lid, Nx, Ny)

        # 3. BGK collision (in-place on f_out)
        collide_2d!(f_out, is_solid, ω)

        # 4. Compute macroscopic fields
        compute_macroscopic_2d!(ρ, ux, uy, f_out)

        # 5. Swap populations
        f_in, f_out = f_out, f_in
    end

    # Copy results to CPU
    ρ_cpu  = Array(ρ)
    ux_cpu = Array(ux)
    uy_cpu = Array(uy)

    return (ρ=ρ_cpu, ux=ux_cpu, uy=uy_cpu, config=config)
end
```


### `run_poiseuille_2d`

**Source:** `src/drivers/basic.jl`

```julia
"""
    run_poiseuille_2d(; Nx=4, Ny=32, ν=0.1, Fx=1e-5, max_steps=10000, backend, T)

Channel flow driven by body force Fx. Periodic in x, walls at j=1 and j=Ny.
"""
function run_poiseuille_2d(; Nx=4, Ny=32, ν=0.1, Fx=1e-5, max_steps=10000,
                            backend=KernelAbstractions.CPU(), T=Float64)
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν, u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    ω = T(omega(config))

    for step in 1:max_steps
        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        collide_guo_2d!(f_out, is_solid, ω, T(Fx), T(0))
        compute_macroscopic_forced_2d!(ρ, ux, uy, f_out, T(Fx), T(0))
        f_in, f_out = f_out, f_in
    end

    return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), config=config)
end
```


### `run_cylinder_2d`

**Source:** `src/drivers/basic.jl`

```julia
"""
    run_cylinder_2d(; Nx=200, Ny=50, radius=10, u_in=0.05, ν=0.05,
                     max_steps=20000, backend, T)

Flow around a cylinder at Re = u_in * 2*radius / ν.
Drag computed via momentum exchange on post-stream populations, averaged
over last `avg_window` steps.
"""
function run_cylinder_2d(; Nx=200, Ny=50, cx=nothing, cy=nothing, radius=10,
                          u_in=0.05, ν=0.05, max_steps=20000, avg_window=1000,
                          backend=KernelAbstractions.CPU(), T=Float64)
    state, config = initialize_cylinder_2d(; Nx=Nx, Ny=Ny, cx=cx, cy=cy,
                                            radius=radius, u_in=u_in, ν=ν,
                                            backend=backend, T=T)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    ω = T(omega(config))

    Fx_sum = 0.0
    Fy_sum = 0.0
    n_avg = 0

    for step in 1:max_steps
        # 1. Stream (f_in = pre-stream, f_out = post-stream)
        stream_2d!(f_out, f_in, Nx, Ny)

        # 2. BCs
        apply_zou_he_west_2d!(f_out, u_in, Nx, Ny)
        apply_zou_he_pressure_east_2d!(f_out, Nx, Ny)

        # 3. MEA drag: f_pre=f_in (pre-stream), f_post=f_out (post-stream+BCs)
        if step > max_steps - avg_window
            drag = compute_drag_mea_2d(f_in, f_out, is_solid, Nx, Ny)
            Fx_sum += drag.Fx
            Fy_sum += drag.Fy
            n_avg += 1
        end

        # 4. Collide
        collide_2d!(f_out, is_solid, ω)

        # 5. Macroscopic + swap
        compute_macroscopic_2d!(ρ, ux, uy, f_out)
        f_in, f_out = f_out, f_in
    end

    Fx_avg = Fx_sum / n_avg
    Fy_avg = Fy_sum / n_avg
    D = 2 * radius
    Cd = 2.0 * Fx_avg / (1.0 * u_in^2 * D)

    return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), config=config,
            Cd=Cd, Fx=Fx_avg, Fy=Fy_avg)
end
```


### `run_natural_convection_2d`

**Source:** `src/drivers/thermal.jl`

```julia
"""
    run_natural_convection_2d(; N=128, Ra=1e4, Pr=0.71, Rc=1.0,
                                T_hot=1.0, T_cold=0.0, max_steps=50000,
                                backend, FT)

Natural convection in a square cavity: hot left wall, cold right wall,
adiabatic top/bottom. Boussinesq approximation with optional
temperature-dependent viscosity via modified Arrhenius law.

Ra = β·g·ΔT·H³/(ν·α), Pr = ν/α, Rc = η_max/η_min (rheological contrast).
For Rc=1: constant viscosity. For Rc>1: ν(T) = ν_ref·exp(α_visc·(T - T_ref)).
"""
function run_natural_convection_2d(; N=128, Ra=1e4, Pr=0.71, Rc=1.0,
                                     T_hot=1.0, T_cold=0.0, max_steps=50000,
                                     backend=KernelAbstractions.CPU(), FT=Float64)
    Nx, Ny = N, N
    ΔT = T_hot - T_cold
    H = FT(N)

    # LBM parameters (NatConv convention: TRef=0, η=Pr, κ=1, β=Pr·Ra)
    ν = FT(0.05)
    α_thermal = ν / FT(Pr)
    β_g = FT(Ra) * ν * α_thermal / (FT(ΔT) * H^3)

    ω_f = FT(1.0 / (3.0 * ν + 0.5))
    ω_T = FT(1.0 / (3.0 * α_thermal + 0.5))

    # Variable viscosity: ν(T) = ν_ref · exp(α_visc · (T - T0_visc))
    # Rc = η_cold/η_hot: hot side is Rc times MORE FLUID
    # η_cold(T=0) = ν_ref, η_hot(T=1) = ν_ref/Rc → α = -ln(Rc)
    α_visc = FT(-log(Rc))
    T0_visc = FT(T_cold)
    # Buoyancy reference: F = β_g · (T - T_ref_buoy)
    T_ref_buoy = FT((T_hot + T_cold) / 2)

    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=Float64(ν), u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, FT; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    # Thermal populations: horizontal linear profile T(x)
    g_in  = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_out = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    Temp  = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    w = weights(D2Q9())
    g_cpu = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        T_init = FT(T_hot - ΔT * (i - 1) / (Nx - 1))
        T_init += FT(0.01 * ΔT) * sin(FT(2π * i / Nx)) * sin(FT(π * j / Ny))
        for q in 1:9
            g_cpu[i, j, q] = FT(w[q]) * T_init
        end
    end
    copyto!(g_in, g_cpu)
    copyto!(g_out, g_cpu)

    # Fused kernel: 1 GPU launch per step (stream + BC + macroscopic + collide)
    for step in 1:max_steps
        if Rc ≈ 1.0
            fused_natconv_step!(f_out, f_in, g_out, g_in, Temp, Nx, Ny,
                                 ω_f, ω_T, β_g, T_ref_buoy, FT(T_hot), FT(T_cold))
        else
            fused_natconv_vt_step!(f_out, f_in, g_out, g_in, Temp, Nx, Ny,
                                    ν, T0_visc, α_visc, ω_T, β_g, T_ref_buoy,
                                    FT(T_hot), FT(T_cold))
        end
        f_in, f_out = f_out, f_in
        g_in, g_out = g_out, g_in
    end

    compute_macroscopic_2d!(ρ, ux, uy, f_in)
    compute_temperature_2d!(Temp, g_in)

    # Compute Nusselt number at hot wall (second-order forward FD)
    T_cpu = Array(Temp)
    dx = FT(1.0)  # lattice spacing
    Nu_local = zeros(FT, Ny)
    for j in 2:Ny-1
        Nu_local[j] = -H * (-3*T_cpu[1,j] + 4*T_cpu[2,j] - T_cpu[3,j]) / (2*dx) / FT(ΔT)
    end
    Nu = sum(Nu_local[2:end-1]) / (Ny - 2)

    return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), Temp=T_cpu,
            Nu=Nu, config=config, Ra=Ra, Pr=Pr, Rc=Rc, ν=Float64(ν), α=Float64(α_thermal))
end
```


### `fused_bgk_step!`

**Source:** `src/kernels/fused_bgk_2d.jl`

```julia
"""
    fused_bgk_step!(f_out, f_in, ρ, ux, uy, is_solid, Nx, Ny, ω)

Single fused kernel for isothermal BGK-LBM: stream + bounce-back + collide + macroscopic.
Reduces kernel launches from 3 to 1 per timestep.
"""
function fused_bgk_step!(f_out, f_in, ρ, ux, uy, is_solid, Nx, Ny, ω)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    kernel! = fused_bgk_step_kernel!(backend)
    kernel!(f_out, f_in, ρ, ux, uy, is_solid, Nx, Ny, ET(ω); ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
```


