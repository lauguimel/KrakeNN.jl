using Test
using Kraken
using KernelAbstractions

# ---------------------------------------------------------------------
# Giesekus 3D log-conformation COUPLED canary — BOTH FVFD drivers.
#
# Scope: the Giesekus coupling wired through the shared constitutive-step
# dispatch (logfv_constitutive_step_dispatch_3d!) inside
#   * run_viscoelastic_fvfd_poiseuille_3d  (planar shear), and
#   * run_viscoelastic_fvfd_extensional_3d (planar extension, :imposed).
# Mirror of the FENE-P coupled canaries, adapted to the Giesekus signature
# (mobility α). NOTE: ν_eff = ν_total is NOT asserted — Giesekus shear-thins,
# so the effective polymer viscosity is rate-dependent; the dispatch is
# pinned instead by the α=0 ≡ OB byte-identity safety net + the finite-α
# signature.
#
# Checks (per driver):
#   P0/E0  α=0 ≡ OB coupled, BYTE-FOR-BYTE (dispatch-correctness net).
#   P1     finite-α shear-thinning: reduced peak N1 vs OB; NaN-free.
#   E1     finite-α bounded extension: center C_xx < OB(≈2) − tol; NaN-free.
#   K      poiseuille_giesekus.krk + extensional_giesekus.krk dispatch+run.
#
# Every binding lives INSIDE the @testset local scope — nothing is declared
# at top level / `const`, so the file leaks no globals into Main.
# ---------------------------------------------------------------------

@testset "FVFD 3D Giesekus coupled (both drivers)" begin

    backend = CPU()
    FT = Float64

    # ---- Poiseuille (planar shear) ----------------------------------
    @testset "Poiseuille driver" begin
        Nx, Ny, Nz = 6, 32, 6
        ν_total = 0.1
        β = 0.5
        ν_s = β * ν_total
        ν_p = (1 - β) * ν_total
        Fx = 1.5e-5
        γ_wall = Fx / (2 * ν_total) * (Ny - 1)
        λ = 1.0 / γ_wall                       # Wi_wall ≈ 1
        max_steps = 10_000
        Gmod = FT(ν_p / λ)

        run_p = model -> Kraken.run_viscoelastic_fvfd_poiseuille_3d(;
            Nx=Nx, Ny=Ny, Nz=Nz, Fx=Fx, ν_s=ν_s, ν_p=nothing, lambda=λ,
            polymer_model=model, max_steps=max_steps, backend=backend, FT=FT,
            advection_scheme=:muscl_superbee,
        )
        peak_N1 = res -> maximum(abs.(res.N1_prof))

        ob_model = LogConfOldroydB(G=Gmod, λ=FT(λ))
        res_ob = run_p(ob_model)

        @testset "P0 α=0 ≡ OB coupled byte-identical" begin
            res0 = run_p(LogConfGiesekus(G=Gmod, λ=FT(λ), α=0.0))
            @test res0.completed_steps == max_steps
            dpsi = maximum(abs.(res0.psi_xx .- res_ob.psi_xx)) +
                   maximum(abs.(res0.psi_xy .- res_ob.psi_xy)) +
                   maximum(abs.(res0.psi_yy .- res_ob.psi_yy)) +
                   maximum(abs.(res0.psi_zz .- res_ob.psi_zz))
            du = maximum(abs.(res0.ux .- res_ob.ux))
            println("P0 Giesekus α=0 vs OB: max|ΔΨ|sum=$(dpsi) max|Δu|=$(du)")
            @test dpsi == 0.0
            @test du == 0.0
        end

        @testset "P1 finite-α shear-thinning (reduced N1 vs OB)" begin
            α = 0.2
            res_gk = run_p(LogConfGiesekus(G=Gmod, λ=FT(λ), α=FT(α)))
            @test res_gk.completed_steps == max_steps
            @test all(isfinite, res_gk.ux)
            @test all(isfinite, res_gk.psi_xx) && all(isfinite, res_gk.psi_xy)
            @test all(isfinite, res_gk.psi_yy) && all(isfinite, res_gk.psi_zz)
            @test all(isfinite, res_gk.tau_p_xx) && all(isfinite, res_gk.tau_p_xy)
            N1_gk = peak_N1(res_gk)
            N1_ob = peak_N1(res_ob)
            println("P1 Giesekus α=$(α): N1=$(N1_gk) < OB N1=$(N1_ob) ?")
            @test N1_ob > 0
            @test N1_gk < N1_ob
        end
    end

    # ---- Extensional (planar extension, :imposed) -------------------
    @testset "Extensional driver" begin
        Nx, Ny, Nz = 24, 24, 6
        ν_total = 0.1
        β = 0.5
        ν_s = β * ν_total
        ν_p = (1 - β) * ν_total
        λ = 50.0
        ε̇ = 0.005
        λε̇ = λ * ε̇                              # = 0.25
        max_steps = 1_000
        Gmod = FT(ν_p / λ)
        Cxx_ref = 1 / (1 - 2 * λε̇)               # OB fixed point = 2.0

        center_mean = function (A)
            nx, ny, nz = size(A)
            i1 = max(1, nx ÷ 2); i2 = min(nx, nx ÷ 2 + 1)
            j1 = max(1, ny ÷ 2); j2 = min(ny, ny ÷ 2 + 1)
            return sum(@view A[i1:i2, j1:j2, :]) / ((i2 - i1 + 1) * (j2 - j1 + 1) * nz)
        end
        finite_all = res ->
            all(isfinite, res.ux) && all(isfinite, res.uy) && all(isfinite, res.uz) &&
            all(isfinite, res.C_xx) && all(isfinite, res.C_yy) && all(isfinite, res.C_zz) &&
            all(isfinite, res.psi_xx) && all(isfinite, res.psi_yy) && all(isfinite, res.psi_zz) &&
            all(isfinite, res.tau_p_xx) && all(isfinite, res.tau_p_yy) && all(isfinite, res.tau_p_zz)

        run_e = model -> Kraken.run_viscoelastic_fvfd_extensional_3d(;
            Nx=Nx, Ny=Ny, Nz=Nz, epsilon_dot=ε̇,
            ν_s=ν_s, ν_p=nothing, lambda=λ,
            polymer_model=model, max_steps=max_steps, backend=backend, FT=FT,
            advection_scheme=:muscl_superbee, velocity_mode=:imposed,
        )

        ob_model = LogConfOldroydB(G=Gmod, λ=FT(λ))
        res_ob = run_e(ob_model)

        @testset "E0 α=0 ≡ OB coupled byte-identical" begin
            res0 = run_e(LogConfGiesekus(G=Gmod, λ=FT(λ), α=0.0))
            @test res0.completed_steps == max_steps
            dc = maximum(abs.(res0.C_xx .- res_ob.C_xx)) +
                 maximum(abs.(res0.C_yy .- res_ob.C_yy)) +
                 maximum(abs.(res0.C_zz .- res_ob.C_zz))
            dpsi = maximum(abs.(res0.psi_xx .- res_ob.psi_xx)) +
                   maximum(abs.(res0.psi_yy .- res_ob.psi_yy))
            println("E0 Giesekus α=0 vs OB: max|ΔC|sum=$(dc) max|ΔΨ|sum=$(dpsi)")
            @test dc == 0.0
            @test dpsi == 0.0
        end

        @testset "E1 finite-α bounded, reduced stretch (< OB)" begin
            α = 0.2
            res_gk = run_e(LogConfGiesekus(G=Gmod, λ=FT(λ), α=FT(α)))
            @test res_gk.completed_steps == max_steps
            @test finite_all(res_gk)
            cxx_gk = center_mean(res_gk.C_xx)
            cxx_ob = center_mean(res_ob.C_xx)
            println("E1 Giesekus α=$(α): Cxx=$(cxx_gk) < OB Cxx=$(cxx_ob)≈$(Cxx_ref) ?")
            @test cxx_ob > 1.5
            @test cxx_gk < cxx_ob
            @test cxx_gk < Cxx_ref - 2.0e-3
            @test cxx_gk > 1.0
        end
    end

    # ---- .krk name-dispatch + run (both drivers) --------------------
    @testset "K .krk Giesekus dispatch+run" begin
        root = normpath(joinpath(@__DIR__, ".."))
        center_mean = function (A)
            nx, ny, nz = size(A)
            i1 = max(1, nx ÷ 2); i2 = min(nx, nx ÷ 2 + 1)
            j1 = max(1, ny ÷ 2); j2 = min(ny, ny ÷ 2 + 1)
            return sum(@view A[i1:i2, j1:j2, :]) / ((i2 - i1 + 1) * (j2 - j1 + 1) * nz)
        end
        cd(root) do
            # Poiseuille
            p_path = joinpath(root, "benchmarks", "krk", "viscoelastic",
                              "poiseuille_giesekus.krk")
            sp = load_kraken(p_path)
            @test sp.lattice == :D3Q19
            @test :viscoelastic in sp.modules
            @test sp.max_steps == 4000
            rp = run_simulation(p_path)
            @test rp.completed_steps == 4000
            @test all(isfinite, rp.ux) && all(isfinite, rp.tau_p_xx)
            @test maximum(abs.(rp.N1_prof)) > 0
            println("K poiseuille_giesekus: peak N1=$(maximum(abs.(rp.N1_prof)))")

            # Extensional
            e_path = joinpath(root, "benchmarks", "krk", "viscoelastic",
                              "extensional_giesekus.krk")
            se = load_kraken(e_path)
            @test se.lattice == :D3Q19
            @test :viscoelastic in se.modules
            @test se.max_steps == 1000
            re = run_simulation(e_path)
            @test re.velocity_mode === :imposed
            @test re.completed_steps == 1000
            @test all(isfinite, re.C_xx) && all(isfinite, re.C_yy)
            cxx = center_mean(re.C_xx)
            # Giesekus signature: bounded, reduced stretch below the OB value 2.0.
            @test 1.0 < cxx < 2.0
            println("K extensional_giesekus: Cxx=$(cxx) (< OB 2.0)")
        end
    end
end

println("EXIT=0")
