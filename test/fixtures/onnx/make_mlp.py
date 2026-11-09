"""Export two small MLPs to ONNX plus PyTorch reference outputs (JSON).

Usage: conda run -n torch python test/fixtures/onnx/make_mlp.py
Writes mlp_tanh.onnx, mlp_norm_softplus_exp.onnx and reference.json next to
this script. Float32 export; the Julia test checks agreement to 1e-6.
"""
import json
import pathlib

import numpy as np
import torch
import torch.nn as nn

here = pathlib.Path(__file__).parent
torch.manual_seed(0)

# The legacy TorchScript exporter only needs the `onnx` package to splice in
# onnxscript functions, which plain MLPs never use. Skip that step when
# `onnx` is not installed so the fixture regenerates with torch alone.
try:
    import onnx  # noqa: F401
except ImportError:
    import torch.onnx._internal.onnx_proto_utils as _opu

    _opu._add_onnxscript_fn = lambda proto, custom_opsets: proto


class Normalised(nn.Module):
    """(x - shift) / scale -> MLP -> optional exp (log-viscosity head)."""

    def __init__(self, body, shift, scale, out_exp):
        super().__init__()
        self.body = body
        self.register_buffer("shift", torch.tensor(shift))
        self.register_buffer("scale", torch.tensor(scale))
        self.out_exp = out_exp

    def forward(self, x):
        y = self.body((x - self.shift) / self.scale)
        return torch.exp(y) if self.out_exp else y


def mlp(sizes, act):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


models = {
    "mlp_tanh": Normalised(mlp([2, 16, 16, 1], nn.Tanh), [0.0, 0.0], [1.0, 1.0], False),
    "mlp_norm_softplus_exp": Normalised(
        mlp([3, 32, 32, 1], nn.Softplus), [0.5, -1.0, 2.0], [2.0, 0.5, 4.0], True
    ),
}
rng = np.random.default_rng(1)
ref = {}
for name, m in models.items():
    m.eval()
    nin = m.shift.numel()
    x = rng.uniform(-1.0, 2.0, size=(20, nin)).astype(np.float32)
    with torch.no_grad():
        y = m(torch.from_numpy(x)).numpy()
    torch.onnx.export(
        m, torch.from_numpy(x[:1]), str(here / f"{name}.onnx"),
        input_names=["x"], output_names=["y"], opset_version=13,
        dynamo=False, do_constant_folding=True,
    )
    ref[name] = {"x": x.astype(float).tolist(), "y": y.reshape(-1).astype(float).tolist()}
(here / "reference.json").write_text(json.dumps(ref))
print("wrote", [p.name for p in here.iterdir()])
