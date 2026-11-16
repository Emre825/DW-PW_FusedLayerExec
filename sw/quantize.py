"""Quantization: calibrate activation ranges and derive the integer parameters of every layer.

Weights are int8, symmetric, per output channel. Activations are int8, asymmetric, per tensor:
real = scale * (q - zero). The stem convolution and the classifier stay in float; the integer
chain covers the depthwise and pointwise layers between them.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.fx import GraphModule

from sw.capture import Layer, fold_bn, layer_table, trace
from sw.reference import QLayer, run_layer

Params = dict[str, tuple[torch.Tensor, torch.Tensor]]  # layer name -> (weight, bias), float
QParams = dict[str, tuple[float, int]]                 # layer name -> (scale, zero-point) of its output

DATA_DIR = Path(__file__).resolve().parents[2] / "data"  # next to the repository, not inside it
WEIGHTS = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1


def parameters(folded: GraphModule, layers: list[Layer]) -> Params:
    """Folded float weight and bias of every layer."""
    modules = dict(folded.named_modules())
    convs = {n.name: modules[n.target] for n in folded.graph.nodes if n.op == "call_module"}
    return {l.name: (convs[l.name].weight.detach(), convs[l.name].bias.detach()) for l in layers}


def run_float(layers: list[Layer], params: Params, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Float output tensor of every layer, by layer name, for the image batch `x`."""
    outs = {}
    for l in layers:
        w, b = params[l.name]
        x = F.conv2d(x, w, b, l.stride, l.kernel // 2, groups=l.cin if l.kind == "dw" else 1)
        if l.act:
            x = x.clamp(0, 6)
        if l.residual:
            x = x + outs[l.residual]
        outs[l.name] = x
    return outs


def calibrate(layers: list[Layer], params: Params, images: torch.Tensor) -> QParams:
    """Scale and zero-point of every layer's output, from its value range over `images`."""
    lo, hi = {l.name: 0.0 for l in layers}, {l.name: 0.0 for l in layers}  # every range contains 0
    for batch in images.split(10):
        for name, y in run_float(layers, params, batch).items():
            lo[name], hi[name] = min(lo[name], y.min().item()), max(hi[name], y.max().item())
    qparams = {}
    for name in lo:
        scale = (hi[name] - lo[name]) / 255
        qparams[name] = (scale, round(-128 - lo[name] / scale))
    return qparams


def to_int8(x: torch.Tensor, scale: float, zero: int) -> np.ndarray:
    return np.clip(np.round(x.numpy() / scale) + zero, -128, 127).astype(np.int8)


def to_float(q: np.ndarray, scale: float, zero: int) -> torch.Tensor:
    return torch.from_numpy(scale * (q.astype(np.float32) - zero))


def fixed_point(real):
    """Integers (mult, shift) with mult * 2**-shift ~ real and 2**14 <= mult < 2**15."""
    frac, exp = np.frexp(real)  # real = frac * 2**exp with 0.5 <= frac < 1
    mult = np.minimum(np.round(frac * 2**15), 2**15 - 1).astype(np.int64)
    shift = 15 - exp.astype(np.int64)
    assert np.all((1 <= shift) & (shift <= 47))
    return mult, shift


def quantize(layers: list[Layer], params: Params, qparams: QParams) -> list[QLayer]:
    """Integer parameters of every layer after the stem."""
    qlayers = []
    for prev, l in zip(layers, layers[1:]):
        (s_in, z_in), (s_out, z_out) = qparams[prev.name], qparams[l.name]
        w, b = (t.double().numpy() for t in params[l.name])
        w = w.reshape(l.cout, -1)
        s_w = np.abs(w).max(axis=1) / 127
        w_q = np.round(w / s_w[:, None]).astype(np.int64)
        # sum of w * (q - z_in) = sum of w * q - z_in * sum of w: the second term joins the bias
        bias = np.round(b / (s_in * s_w)).astype(np.int64) - z_in * w_q.sum(axis=1)
        assert np.abs(bias).max() < 2**31
        mult, shift = fixed_point(s_in * s_w / s_out)
        skip = {}
        if l.residual:
            s_skip, z_skip = qparams[l.residual]
            skip_mult, skip_shift = fixed_point(s_skip / s_out)
            skip = dict(skip_mult=int(skip_mult), skip_shift=int(skip_shift), z_skip=z_skip)
        shape = (l.cout, 3, 3) if l.kind == "dw" else (l.cout, l.cin)
        qlayers.append(QLayer(l.kind, l.stride, w_q.astype(np.int8).reshape(shape),
                              bias.astype(np.int32), mult, shift, z_in, z_out, **skip))
    return qlayers


def run_int(layers: list[Layer], qlayers: list[QLayer], x: np.ndarray) -> dict[str, np.ndarray]:
    """int8 output tensor of every layer, by layer name, for the int8 stem output `x` (c, h, w)."""
    outs = {layers[0].name: x}
    for l, q in zip(layers[1:], qlayers):
        outs[l.name] = x = run_layer(q, x, outs.get(l.residual))
    return outs


def imagenette(split: str, count: int) -> torch.Tensor:
    """`count` preprocessed Imagenette images, evenly spread over the classes."""
    dataset = torchvision.datasets.Imagenette(DATA_DIR, split, size="320px", download=True,
                                              transform=WEIGHTS.transforms())
    return torch.stack([dataset[i][0] for i in range(0, len(dataset), len(dataset) // count)][:count])


def mobilenet_v2():
    """Pretrained MobileNetV2, calibrated on 100 training images: the model, its layer table,
    the float parameters, the calibration result and the integer parameters."""
    model = torchvision.models.mobilenet_v2(weights=WEIGHTS).eval()
    folded = fold_bn(trace(model))
    layers = layer_table(folded)
    params = parameters(folded, layers)
    qparams = calibrate(layers, params, imagenette("train", 100))
    return model, layers, params, qparams, quantize(layers, params, qparams)


if __name__ == "__main__":
    model, layers, params, qparams, qlayers = mobilenet_v2()

    shifts = np.concatenate([q.shift for q in qlayers])
    biases = np.concatenate([q.bias for q in qlayers])
    print(f"Calibrated on 100 training images: {len(qlayers)} integer layers, shifts {shifts.min()} to"
          f" {shifts.max()}, largest |bias| {np.abs(biases).max()}")

    stem, last = layers[0].name, layers[-1].name
    held_out = imagenette("val", 100)
    agree = 0
    with torch.no_grad():
        for image in held_out:
            floats = run_float(layers, params, image[None])
            features = to_float(run_int(layers, qlayers, to_int8(floats[stem][0], *qparams[stem]))[last],
                                *qparams[last])
            top1_float = model.classifier(floats[last].mean((2, 3))).argmax()
            top1_int = model.classifier(features.mean((1, 2))[None]).argmax()
            agree += int(top1_float == top1_int)
    print(f"Top-1 agreement with the float model on {len(held_out)} held-out images: {agree}/{len(held_out)}")
