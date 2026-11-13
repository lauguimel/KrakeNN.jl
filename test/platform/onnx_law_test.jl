# ONNX reader gates (needs `using ProtoBuf`, a weakdep): PyTorch outputs
# reproduced to <= 1e-6 (Float32 export), hand-encoded MatMul+Add / Gemm
# graphs, unsupported ops named in the error, loaded law runs in FVLawViscosity.
using Test, KrakeNN, ProtoBuf, Random, LinearAlgebra

const _ONNX_DIR = joinpath(@__DIR__, "..", "fixtures", "onnx")

# Minimal JSON reader for reference.json: {"name": {"x": [[..],..], "y": [..]}}.
function _onnx_ref(name)
    txt = read(joinpath(_ONNX_DIR, "reference.json"), String)
    sect = split(txt, "\"$(name)\"")[2]
    sect = split(sect, "\"mlp_"; limit=2)[1]
    xs, ys = split(sect, "\"y\"")
    nums(s) = parse.(Float64, [m.match for m in eachmatch(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s)])
    xrows = [nums(m.match) for m in eachmatch(r"\[[^\[\]]+\]", split(xs, "\"x\"")[2])]
    return xrows, nums(ys)
end

# ---- independent protobuf wire encoder (for synthetic graphs) -------------
_varint(n::Integer) = (b = UInt8[]; n = UInt64(n); while n >= 0x80; push!(b, UInt8(n & 0x7f) | 0x80); n >>= 7; end; push!(b, UInt8(n)); b)
_ld(f, bytes) = vcat(_varint(f << 3 | 2), _varint(length(bytes)), bytes)
_str(f, s) = _ld(f, Vector{UInt8}(s))
_int(f, n) = vcat(_varint(f << 3), _varint(n))
_f32(f, x) = vcat(_varint(f << 3 | 5), collect(reinterpret(UInt8, [Float32(x)])))
function _tensor(name, dims, vals)   # vals row-major
    return vcat(map(d -> _int(1, d), dims)..., _int(2, 1), _str(8, name),
                _ld(9, collect(reinterpret(UInt8, Float32.(vals)))))
end
_node(op, ins, outs; attrs=UInt8[]) =
    vcat(map(s -> _str(1, s), ins)..., map(s -> _str(2, s), outs)..., _str(4, op), attrs)
_attr_i(name, v) = _ld(5, vcat(_str(1, name), _int(3, v), _int(20, 2)))
_attr_f(name, v) = _ld(5, vcat(_str(1, name), _f32(2, v), _int(20, 1)))
function _model(nodes, inits, input="x")
    g = vcat(map(n -> _ld(1, n), nodes)..., _str(2, "g"),
             map(t -> _ld(5, t), inits)..., _ld(11, _str(1, input)), _ld(12, _str(1, "y")))
    return vcat(_int(1, 8), _ld(7, g))
end
function _write(bytes)
    path, io = mktemp(); write(io, bytes); close(io); return path
end

@testset "ONNX law reader" begin
    @testset "PyTorch parity: $name" for (name, nin, hid, acts, out) in (
            ("mlp_tanh", 2, (16, 16), (:tanh, :tanh), :identity),
            ("mlp_norm_softplus_exp", 3, (32, 32), (:softplus, :softplus), :exp))
        law, theta = load_onnx_law(joinpath(_ONNX_DIR, name * ".onnx"))
        @test law isa DenseNet
        @test law_ninputs(law) == nin && law_nparams(law) == length(theta)
        @test typeof(law).parameters[2] == (hid..., 1)
        @test typeof(law).parameters[3] == (acts..., out)
        xs, ys = _onnx_ref(name)
        @test length(xs) == 20 && length(ys) == 20
        err = maximum(abs(law_value(law, Tuple(xs[i]), theta) - ys[i]) / max(abs(ys[i]), 1) for i in 1:20)
        @info "ONNX vs PyTorch" name max_rel_err = err
        @test err <= 1e-6
    end

    @testset "MatMul+Add and Gemm(alpha,beta,transB) agree" begin
        rng = MersenneTwister(3)
        W1 = randn(rng, 4, 2); b1 = randn(rng, 4); W2 = randn(rng, 1, 4); b2 = randn(rng, 1)
        f32(a) = Float64.(Float32.(a))
        rm(A) = vec(permutedims(A))   # row-major flattening, as ONNX stores
        # MatMul stores [in,out] row-major; Gemm(transB=1) stores [out,in].
        mm = _model([_node("MatMul", ["x", "W1"], ["a"]), _node("Add", ["a", "b1"], ["b"]),
                     _node("Tanh", ["b"], ["c"]), _node("MatMul", ["c", "W2"], ["d"]),
                     _node("Add", ["d", "b2"], ["y"])],
                    [_tensor("W1", (2, 4), rm(W1')), _tensor("b1", (4,), b1),
                     _tensor("W2", (4, 1), rm(W2')), _tensor("b2", (1,), b2)])
        ge = _model([_node("Gemm", ["x", "W1", "b1"], ["a"];
                           attrs=vcat(_attr_i("transB", 1), _attr_f("alpha", 2.0), _attr_f("beta", 0.5))),
                     _node("Tanh", ["a"], ["c"]),
                     _node("Gemm", ["c", "W2", "b2"], ["y"]; attrs=_attr_i("transB", 1))],
                    [_tensor("W1", (4, 2), rm(W1)), _tensor("b1", (4,), b1),
                     _tensor("W2", (1, 4), rm(W2)), _tensor("b2", (1,), b2)])
        x = (0.3, -0.8)
        ref(W1e, b1e) = (h = tanh.(f32(W1e) * collect(x) .+ f32(b1e)); only(f32(W2) * h .+ f32(b2)))
        l1, t1 = load_onnx_law(_write(mm))
        @test law_value(l1, x, t1) ≈ ref(W1, b1) atol = 1e-12
        l2, t2 = load_onnx_law(_write(ge))
        @test law_value(l2, x, t2) ≈ ref(2 .* W1, 0.5 .* b1) atol = 1e-6
    end

    @testset "unsupported ops are named" begin
        bad(op) = _model([_node("Gemm", ["x", "W", "b"], ["a"]; attrs=_attr_i("transB", 1)),
                          _node(op, ["a"], ["c"]), _node("Gemm", ["c", "V", "d"], ["y"]; attrs=_attr_i("transB", 1))],
                         [_tensor("W", (3, 2), ones(6)), _tensor("b", (3,), ones(3)),
                          _tensor("V", (1, 3), ones(3)), _tensor("d", (1,), ones(1))])
        for op in ("Conv", "LeakyRelu", "Gelu", "Softmax")
            err = try load_onnx_law(_write(bad(op))); nothing catch e; e end
            @test err isa ArgumentError && occursin("'$(op)'", err.msg)
        end
        # Exp on a hidden layer and a width-3 output are rejected.
        @test_throws ArgumentError load_onnx_law(_write(bad("Exp")))
        # Load-time size guard is forwarded.
        law, _ = load_onnx_law(joinpath(_ONNX_DIR, "mlp_tanh.onnx"); max_width=16)
        @test law isa DenseNet
        @test_throws ArgumentError load_onnx_law(joinpath(_ONNX_DIR, "mlp_norm_softplus_exp.onnx"); max_width=16)
    end

    @testset "loaded law runs in FVLawViscosity (CPU)" begin
        law, theta = load_onnx_law(joinpath(_ONNX_DIR, "mlp_norm_softplus_exp.onnx"))
        N = 12
        f1 = [0.2*(i-0.5)+0.1*(j-0.5) for i in 1:N, j in 1:N]
        f2 = [0.1*(i-0.5)-0.2*(j-0.5) for i in 1:N, j in 1:N]
        visc = FVLawViscosity(law, theta, (f1, f2); eps=1e-3)
        p = fv_steady_problem(N, N; h=1.0, bc=(W=:periodic, E=:periodic, S=:wall, N=:wall),
                              source=(x, y) -> (0.02, 0.0, 0.0), viscosity=visc)
        s = fv_newton_solver(p)
        U = zeros(fv_nunknowns(p))
        @test fv_newton!(s, U; atol=1e-12, rtol=0.0, maxit=20).converged
        map = FVLawParameters(length(theta))
        ws = fv_adjoint_workspace(s, map; chunk=4)
        obj = FVLinearFunctional([fv_index(p.grid, 2, 4, 1)], [1.0])
        a = fv_steady_gradient(s, U, theta, obj; params=map, ws)
        b = fv_steady_gradient(s, U, theta, obj; params=map, ws, law_gradient=:duals, finalize=false)
        @test norm(a.grad - b.grad) <= 1e-12 * max(norm(a.grad), norm(b.grad))
        @test all(isfinite, a.grad) && norm(a.grad) > 0
    end
end
