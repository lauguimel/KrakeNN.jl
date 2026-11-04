# Hardware

All benchmark results in this section are tagged with a hardware identifier
that maps to an entry in
[`benchmarks/hardware.toml`](https://github.com/lauguimel/Kraken.jl/blob/main/benchmarks/hardware.toml).

## Machines used

### `apple_m2` — CPU baseline

| Property | Value |
|:---------|:------|
| Model    | Apple M3 Max |
| Cores    | 8 (4P + 4E) |
| RAM      | 16 GB unified |
| OS       | macOS 14 |
| Julia    | 1.11 |
| Notes    | Single-thread unless stated otherwise |

### `aqua_h100` — GPU target

| Property  | Value |
|:----------|:------|
| Model     | NVIDIA H100 80 GB PCIe |
| Host      | QUT AQUA cluster (`aqua.qut.edu.au`) |
| Scheduler | PBS Pro |
| CUDA      | 12.x |
| Driver    | 550+ |
| Julia     | 1.11 |
| Notes     | Requested via `gpu_id=H100` |

## Adding your own hardware

1. Open `benchmarks/hardware.toml` and add a new TOML section:

   ```toml
   [my_workstation]
   kind   = "cpu"          # or "gpu"
   model  = "AMD Ryzen 9 7950X"
   cores  = 16
   ram_gb = 64
   os     = "Ubuntu 24.04"
   julia  = "1.11"
   notes  = "Description of the setup"
   ```

2. Run the benchmarks with the matching ID:

   ```bash
   julia --project benchmarks/run_all.jl --hardware-id=my_workstation
   ```

3. Results land in `benchmarks/results/` with your hardware ID in the
   filename. Consider opening a PR to add your numbers to the docs.
