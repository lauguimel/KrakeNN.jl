# # Performance benchmark: MLUPs
#
# This page measures the throughput of KrakeNN's 2D lid-driven cavity solver in
# **Mega Lattice Updates Per Second** (MLUPs), the standard metric for LBM codes.
#
# ```math
# \text{MLUPs} = \frac{N_x \times N_y \times N_{\text{steps}}}{t_{\text{wall}} \times 10^6}
# ```
#
# We sweep grid sizes from 64 to 512 on CPU, and optionally on GPU when a
# Metal or CUDA backend is available.

# ## Setup

using KrakeNN
using CairoMakie
using Printf

# ### Dark docs theme
#
# Render the throughput bar chart on the live Vitepress page colour
# (``#1b1b1f``) with light text/ticks/spines and a dark-readable legend so it
# sits seamlessly on the dark page.

const KRAKEN_DARK = "#1b1b1f"
set_theme!(Theme(
    backgroundcolor = KRAKEN_DARK,
    textcolor = "gray92",
    Axis = (
        backgroundcolor = KRAKEN_DARK,
        titlecolor = "gray92",
        xlabelcolor = "gray92", ylabelcolor = "gray92",
        xticklabelcolor = "gray85", yticklabelcolor = "gray85",
        xtickcolor = "gray70", ytickcolor = "gray70",
        xgridcolor = ("gray60", 0.18), ygridcolor = ("gray60", 0.18),
        leftspinecolor = "gray55", rightspinecolor = "gray55",
        topspinecolor = "gray55", bottomspinecolor = "gray55",
    ),
    Legend = (
        backgroundcolor = KRAKEN_DARK, labelcolor = "gray92",
        titlecolor = "gray92", framecolor = "gray45",
    ),
))

# Dark-readable series colours (cool blue + kraken accent).
const COLOR_CPU = "#4ea1d3"
const COLOR_GPU = "#ff6b6b"

# ### Benchmark parameters
#
# We use a fixed number of time steps so that the work per run scales only
# with the number of lattice nodes.

grid_sizes = [64, 128, 256, 512]
max_steps  = 500
ν          = 0.1
u_lid      = 0.05

# ## CPU benchmark

mlups_cpu = Float64[]

for N in grid_sizes
    config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=ν, u_lid=u_lid, max_steps=max_steps)
    ## Warm-up run (compile)
    run_cavity_2d(config)
    ## Timed run
    t = @elapsed run_cavity_2d(config)
    m = N * N * max_steps / (t * 1e6)
    push!(mlups_cpu, m)
    @info @sprintf("CPU  N=%3d : %.1f MLUPs  (%.2f s)", N, m, t)
end

# ## GPU benchmark (optional)
#
# KrakeNN uses KernelAbstractions.jl, so the same `run_cavity_2d` works on
# Metal (Apple Silicon) or CUDA (NVIDIA) backends.  If neither is available
# the GPU section is silently skipped.

mlups_gpu = Float64[]
gpu_available = false

try
    import Metal
    gpu_backend = Metal.MetalBackend()
    gpu_available = true
    @info "Metal backend detected"
catch
    try
        import CUDA
        gpu_backend = CUDA.CUDABackend()
        gpu_available = true
        @info "CUDA backend detected"
    catch
        @info "No GPU backend available — skipping GPU benchmark"
    end
end

if gpu_available
    for N in grid_sizes
        config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=ν, u_lid=u_lid, max_steps=max_steps)
        ## Warm-up
        run_cavity_2d(config; backend=gpu_backend)
        ## Timed
        t = @elapsed run_cavity_2d(config; backend=gpu_backend)
        m = N * N * max_steps / (t * 1e6)
        push!(mlups_gpu, m)
        @info @sprintf("GPU  N=%3d : %.1f MLUPs  (%.2f s)", N, m, t)
    end
end

# ## Results table

@info "──────────────────────────────────────"
@info @sprintf("  %5s  %10s  %10s", "N", "CPU MLUPs", gpu_available ? "GPU MLUPs" : "—")
@info "──────────────────────────────────────"
for (i, N) in enumerate(grid_sizes)
    cpu_str = @sprintf("%10.1f", mlups_cpu[i])
    gpu_str = gpu_available ? @sprintf("%10.1f", mlups_gpu[i]) : "        —"
    @info @sprintf("  %5d  %s  %s", N, cpu_str, gpu_str)
end

# ## Bar chart

fig = Figure(size=(700, 400))
ax  = Axis(fig[1, 1];
    title  = "KrakeNN — LBM throughput (D2Q9 lid-driven cavity)",
    xlabel = "Grid size N×N",
    ylabel = "MLUPs",
    xticks = (1:length(grid_sizes), string.(grid_sizes)),
)

barplot!(ax, 1:length(grid_sizes), mlups_cpu; label="CPU", color=COLOR_CPU)

if gpu_available
    barplot!(ax, (1:length(grid_sizes)) .+ 0.35, mlups_gpu;
             label="GPU", color=COLOR_GPU, width=0.3)
end

axislegend(ax; position=:lt)
fig

# ## Discussion
#
# On a modern multi-core CPU, single-threaded Julia LBM typically reaches
# **5–30 MLUPs** for D2Q9 depending on grid size and cache effects.  GPU
# backends can achieve **100–1000+ MLUPs** for grids that fill the device.
#
# These numbers provide a baseline for regression testing: any commit that
# significantly degrades MLUPs on a reference grid should be investigated.
