# Bit equality of the device-built 3D periodic setup `_gnf3d_periodic_setup`
# (src/ad/gnf3d_adjoint.jl, whole-array broadcasts on the backend) against the
# former host cell loop, kept below verbatim as `_g3s_setup_reference` (only
# the module prefixes added). Every array of the returned state except the
# uninitialised `f_out` must be equal with `==` and `isequal` (which also
# separates −0.0 from +0.0, the sign of Fx at solid nodes when Fx < 0).
# Cases: N³ lattices, N = 8 and 12, Float64 and Float32; masks = fully fluid,
# and a flat wall plane (j = 1) plus random solid voxels; cold start and
# f_init/tau_init start; F with negative components.
# CPU always; CUDA when CUDA.functional() (skipped otherwise, KRAKEN_REQUIRE_GPU=1
# turns the skip into a failure). Command (cluster):
#   KRAKEN_REQUIRE_GPU=1 julia --project=. test/ad/gnf3d_setup_equality_test.jl
using Test, KrakeNN, Random

const _G3S_REQUIRE_GPU = get(ENV, "KRAKEN_REQUIRE_GPU", "0") in ("1", "true")
# CUDA is loaded here, at top level, never inside a function.
const _G3S_GPU_OK = (try
    @eval Main using CUDA
    @eval Main CUDA.functional()
catch
    false
end) === true

# Former implementation (dev/platform 5779fb898), reference only.
function _g3s_setup_reference(is_solid::AbstractArray{Bool,3}, closure, F, nu_init, Λ,
                              f_init, tau_init, backend, FT)
    K = KrakeNN
    nx, ny, nz = size(is_solid)
    dims = (nx + 2, ny + 2, nz + 2)
    Fx, Fy, Fz = K._gnf3d_force3(F)
    model = K._gnf3d_rheology(FT, closure)
    solid_r = Array{Bool}(is_solid)
    solid_p = [solid_r[mod1(i - 1, nx), mod1(j - 1, ny), mod1(k - 1, nz)]
               for i in 1:dims[1], j in 1:dims[2], k in 1:dims[3]]
    q_wall = zeros(FT, dims..., 19)
    for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
        solid_p[i, j, k] && continue
        for q in 2:19
            solid_p[i + K._G3_CX[q], j + K._G3_CY[q], k + K._G3_CZ[q]] && (q_wall[i, j, k, q] = FT(0.5))
        end
    end
    sp0 = inv(FT(3) * FT(nu_init) + FT(0.5))
    sm0 = inv(FT(Λ) / (FT(3) * FT(nu_init)) + FT(0.5))
    sp = fill(sp0, dims); sm = fill(sm0, dims)
    f = zeros(FT, dims..., 19)
    fh = f_init === nothing ? nothing : Array(f_init)
    th = tau_init === nothing ? nothing : Array(tau_init)
    for k in 2:nz+1, j in 2:ny+1, i in 2:nx+1
        ir, jr, kr = i - 1, j - 1, k - 1
        solid = solid_p[i, j, k]
        for q in 1:19
            f[i, j, k, q] = fh !== nothing ? FT(fh[ir, jr, kr, q]) :
                solid ? K.equilibrium(K.D3Q19(), one(FT), zero(FT), zero(FT), zero(FT), q) :
                        K.equilibrium(K.D3Q19(), one(FT), FT(-Fx / 2), FT(-Fy / 2), FT(-Fz / 2), q)
        end
        if !solid && th !== nothing
            τ = FT(th[ir, jr, kr])
            sp[i, j, k] = inv(τ)
            sm[i, j, k] = inv(FT(Λ) / (τ - FT(0.5)) + FT(0.5))
        end
    end
    axes3 = (1, 2, 3)
    for a in (f, sp, sm)
        K._wrap_gnf_periodic_3d!(a, axes3)
    end
    force = ntuple(d -> FT.((d == 1 ? Fx : d == 2 ? Fy : Fz) .* .!solid_p), 3)
    dev(a) = K._gnf_backend_copy_3d(backend, a)
    f_in = dev(f)
    return (; f_in, f_out=similar(f_in), rho=dev(ones(FT, dims)), ux=dev(zeros(FT, dims)),
            uy=dev(zeros(FT, dims)), uz=dev(zeros(FT, dims)), is_solid=dev(solid_p),
            q_wall=dev(q_wall), uw=dev(zeros(FT, dims..., 19)), Fx=dev(force[1]),
            Fy=dev(force[2]), Fz=dev(force[3]), sp=dev(sp), sm=dev(sm), dims, nx, ny, nz,
            model, Λ=FT(Λ), F=(Fx, Fy, Fz), axes3)
end

const _G3S_FIELDS = (:f_in, :rho, :ux, :uy, :uz, :is_solid, :q_wall, :uw, :Fx, :Fy, :Fz,
                     :sp, :sm)

function _g3s_masks(N)
    fluid = falses(N, N, N)
    wall = falses(N, N, N)
    wall[:, 1, :] .= true                       # flat wall plane j = 1
    rng = MersenneTwister(11 + N)
    for k in 1:N, j in 2:N, i in 1:N
        rand(rng) < 0.12 && (wall[i, j, k] = true)  # voxel obstacles
    end
    return (("fluid", Array{Bool}(fluid)), ("wall+voxels", Array{Bool}(wall)))
end

function _g3s_starts(N)
    rng = MersenneTwister(3 + N)
    f0 = 1 / 19 .+ 1e-3 .* randn(rng, N, N, N, 19)
    t0 = 0.55 .+ rand(rng, N, N, N)
    return (("cold", nothing, nothing), ("f_init+tau_init", f0, t0))
end

function _g3s_check(backend, label)
    closure = Newtonian(0.1)
    F = (-1e-3, 2e-4, -5e-4)
    for N in (8, 12), FT in (Float64, Float32)
        for (mname, solid) in _g3s_masks(N), (sname, f0, t0) in _g3s_starts(N)
            @testset "$label N=$N $FT $mname $sname" begin
                ref = _g3s_setup_reference(solid, closure, F, 0.12, 3 / 16, f0, t0,
                                           KrakeNN.KernelAbstractions.CPU(), FT)
                got = KrakeNN._gnf3d_periodic_setup(solid, closure, F, 0.12, 3 / 16, f0, t0,
                                                    backend, FT)
                for name in _G3S_FIELDS
                    a = Array(getproperty(got, name)); b = getproperty(ref, name)
                    @test eltype(a) == eltype(b)
                    @test size(a) == size(b)
                    @test a == b
                    @test isequal(a, b)
                end
                @test size(got.f_out) == size(ref.f_out) && eltype(got.f_out) == FT
                @test got.dims == ref.dims && (got.nx, got.ny, got.nz) == (ref.nx, ref.ny, ref.nz)
                @test got.Λ === ref.Λ && got.F === ref.F && got.axes3 == ref.axes3
                @test got.model == ref.model
            end
        end
    end
end

@testset "3D GNF periodic setup == host reference (bitwise)" begin
    _g3s_check(KrakeNN.KernelAbstractions.CPU(), "CPU")
    if _G3S_GPU_OK
        _g3s_check(Main.CUDA.CUDABackend(), "CUDA")
    elseif _G3S_REQUIRE_GPU
        @test _G3S_GPU_OK
    else
        @test_skip "CUDA part requires CUDA.functional()"
    end
end
