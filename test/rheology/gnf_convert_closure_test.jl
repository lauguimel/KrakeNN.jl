using Test
using Random
using KrakeNN

# gnf_convert_closure(T, ::LBMLawViscosity) converts every numeric constant of
# the law object (network shift/scale, LearnedViscosity reference values,
# ClassicalLaw model and thermal metadata), not only theta and the scales.
#   C1 Float64 -> Float64: closure and law unchanged, viscosity bit for bit
#   C2 Float64 -> Float32: converted law === the law built directly with
#      Float32 constants, viscosity bit for bit
#   C3 no Float64 number left anywhere in the converted closure (recursive walk)
#   C4 size of the Float32 change against the former conversion (law constants
#      left in Float64): Float32 ulp level, reported
# CPU only.

# true when a Float64 number is reachable from x through fields/elements.
function _gcc_has_f64(x)
    x isa Float64 && return true
    x isa AbstractArray && return eltype(x) === Float64 || any(_gcc_has_f64, x)
    (x isa Number || x isa Symbol || x isa Bool) && return false
    return any(i -> _gcc_has_f64(getfield(x, i)), 1:nfields(x))
end

# Each case: name => S -> (law, theta) with every law constant built in type S.
function _gcc_cases()
    net(S) = DenseNet(1, (8, 8); activation=:tanh, output=:exp,
                      shift=(S(0.1),), scale=(S(5.0),))
    mlp(S) = TinyMLP{1,4,2}(; shift=(S(0.3),), scale=(S(2.0),))
    lv(S) = LearnedViscosity(TinyMLP{1,4,1}(; shift=(S(-0.2),), scale=(S(1.5),));
                             nu_ref=S(0.02), epsilon=S(1e-3))
    cy(S) = ClassicalLaw(CarreauYasuda(S(0.17), S(0.0034), S(20.0), S(2.0), S(0.7)))
    pl(S) = ClassicalLaw(PowerLaw(S(0.01), S(0.6); nu_min=S(1e-6), nu_max=S(10.0),
                                  thermal=ArrheniusCoupling(S(300.0), S(2000.0))))
    return [
        "DenseNet" => S -> (net(S), law_init(net(Float64); rng=MersenneTwister(3))),
        "TinyMLP" => S -> (mlp(S), law_init(mlp(Float64); rng=MersenneTwister(5))),
        "LearnedViscosity" => S -> (lv(S), law_init(lv(Float64); rng=MersenneTwister(7))),
        "ClassicalLaw CarreauYasuda" => S -> (cy(S), law_init(cy(Float64))),
        "ClassicalLaw PowerLaw+Arrhenius" => S -> (pl(S), law_init(pl(Float64))),
    ]
end

const _GCC_I1 = [0.0, 1e-8, 3.7e-6, 2.5e-4, 0.011, 0.3, 2.0, 47.0]

_gcc_closure(law, theta) = LBMLawViscosity(law, theta; eps=1e-3, nu_scale=0.7, gamma_scale=3.0)

@testset "gnf_convert_closure: law constants follow T" begin
    for (name, build) in _gcc_cases()
        @testset "$name" begin
            law64, theta = build(Float64)
            law32, _ = build(Float32)
            c = _gcc_closure(law64, theta)

            # C1: Float64 unchanged, bit for bit.
            c64 = KrakeNN.gnf_convert_closure(Float64, c)
            @test c64.law === c.law
            @test c64.theta === c.theta
            @test c64.eps === c.eps && c64.nu_scale === c.nu_scale &&
                  c64.gamma_scale === c.gamma_scale
            @test all(I1 -> KrakeNN.gnf_viscosity(c64, I1) === KrakeNN.gnf_viscosity(c, I1),
                      _GCC_I1)

            # C2: Float32 converted == built directly with Float32 constants.
            c32 = KrakeNN.gnf_convert_closure(Float32, c)
            ref = LBMLawViscosity(law32, Float32.(theta), Float32(1e-3), Float32(0.7), Float32(3.0))
            @test c32.law === ref.law
            @test typeof(c32) === typeof(ref)
            @test c32.theta == ref.theta && eltype(c32.theta) === Float32
            @test c32.eps === ref.eps && c32.nu_scale === ref.nu_scale &&
                  c32.gamma_scale === ref.gamma_scale
            nu32 = [KrakeNN.gnf_viscosity(c32, Float32(I1)) for I1 in _GCC_I1]
            nuref = [KrakeNN.gnf_viscosity(ref, Float32(I1)) for I1 in _GCC_I1]
            @test eltype(nu32) === Float32
            @test all(nu32 .=== nuref)

            # C3: nothing Float64 left (the Float64 closure is the control).
            @test !_gcc_has_f64(c32)
            @test _gcc_has_f64(c)

            # C4: former conversion (law constants kept in Float64): promoted
            # evaluation, rounded to Float32. The change is at Float32 ulp level.
            old = LBMLawViscosity(law64, Float32.(theta), Float32(1e-3), Float32(0.7), Float32(3.0))
            nuold = [Float32(KrakeNN.gnf_viscosity(old, Float32(I1))) for I1 in _GCC_I1]
            ulps = maximum(abs.(Float64.(nu32) .- Float64.(nuold)) ./ eps.(nuold))
            rel = maximum(abs.(Float64.(nu32) .- Float64.(nuold)) ./ abs.(Float64.(nuold)))
            @info "gnf_convert_closure Float32 change vs former conversion" law = name max_ulps = ulps max_rel = rel
            @test rel <= 1e-5
        end
    end

    # Laws without a conversion method are returned unchanged.
    @test KrakeNN._gnf_convert_law(Float32, :not_a_law) === :not_a_law
end
