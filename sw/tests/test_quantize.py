import numpy as np
import pytest
import torch

from sw.capture import INPUT_SHAPE, fold_bn, layer_table
from sw.quantize import calibrate, fixed_point, parameters, quantize, run_float, run_int, to_int8
from sw.reference import run_layer


@pytest.fixture(scope="module")
def quantized(traced):
    """Layer table, float parameters, calibration result and integer parameters of the test model."""
    folded = fold_bn(traced)
    layers = layer_table(folded)
    params = parameters(folded, layers)
    torch.manual_seed(0)
    qparams = calibrate(layers, params, torch.randn(4, *INPUT_SHAPE[1:]))
    return layers, params, qparams, quantize(layers, params, qparams)


@pytest.fixture(scope="module")
def tensors(quantized):
    """The float model's output tensors for one image, quantized to int8, by layer name."""
    layers, params, qparams, _ = quantized
    floats = run_float(layers, params, torch.randn(INPUT_SHAPE))
    return {name: to_int8(y[0], *qparams[name]) for name, y in floats.items()}


def test_run_float_matches_model(traced, quantized):
    layers, params, *_ = quantized
    x = torch.randn(INPUT_SHAPE)
    features = run_float(layers, params, x)[layers[-1].name]
    with torch.no_grad():
        ref = traced(x)
        out = traced.get_submodule("classifier.1")(features.mean((2, 3)))
    assert (ref - out).abs().max() <= 1e-4 * ref.abs().max()


def test_fixed_point():
    real = 2.0 ** np.random.default_rng(0).uniform(-30, 10, 1000)
    mult, shift = fixed_point(real)
    assert np.all((2**14 <= mult) & (mult < 2**15))
    assert np.allclose(mult * 2.0**-shift, real, rtol=2**-14, atol=0)


def test_ranges(quantized):
    layers, _, qparams, qlayers = quantized
    for l in layers:
        scale, zero = qparams[l.name]
        assert -128 <= zero <= 127
        if l.act:  # what lets the int8 saturation act as ReLU6
            assert zero == -128 and scale <= 6 / 255
    for l, q in zip(layers[1:], qlayers):  # every channel uses the full weight range, never -128
        assert (np.abs(q.weight.astype(int)).reshape(l.cout, -1).max(axis=1) == 127).all()


def test_each_layer_tracks_float(quantized, tensors):
    """Fed with the float model's tensors, every integer layer reproduces the float layer's output."""
    layers, _, _, qlayers = quantized
    for prev, l, q in zip(layers, layers[1:], qlayers):
        out = run_layer(q, tensors[prev.name], tensors.get(l.residual))
        error = np.abs(out.astype(int) - tensors[l.name])  # in least significant bits of the output
        assert error.mean() < 0.5, l.name


def test_run_int_chains_the_layers(quantized, tensors):
    layers, _, _, qlayers = quantized
    outs = run_int(layers, qlayers, tensors[layers[0].name])
    assert list(outs) == [l.name for l in layers]
    for prev, l, q in zip(layers, layers[1:], qlayers):
        assert np.array_equal(outs[l.name], run_layer(q, outs[prev.name], outs.get(l.residual)))
