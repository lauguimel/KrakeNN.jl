"""
    load_onnx_law(path; kw...) -> (law::DenseNet, theta::Vector)

Read a feed-forward network exported to ONNX (PyTorch `torch.onnx.export`)
and return it as a [`DenseNet`](@ref) plus its flat parameter vector, ready
for [`FVLawViscosity`](@ref). Requires `using ProtoBuf` (weakdep; activates
`ext/KrakeNNONNXExt.jl`).

Supported: a linear chain of `Gemm` (alpha, beta, transB; transA=0) or
`MatMul`+`Add`, with `Tanh`, `Relu`, `Sigmoid`, `Softplus` between layers;
`Sub`/`Div` by a constant at the input (normalization); `Exp` or `Softplus`
at the output. Any other op is an error naming the op. Keywords are forwarded
to [`DenseNet`](@ref) (`max_hidden_layers`, `max_width`). Weights are read at
the stored precision (Float32 exports give Float32-rounded Float64 values).
"""
function load_onnx_law end
