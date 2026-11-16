"""Integer reference model: the bit-exact arithmetic that the RTL must reproduce.

For every output channel c and output pixel, a layer computes

    acc = sum over the window of weight * input              (int8 x int8 products)
    y   = requantize(acc + bias[c], mult[c], shift[c]) + z_out
    y  += requantize(skip - z_skip, skip_mult, skip_shift)   (residual layers only)
    out = y saturated to int8

ReLU6 needs no operation of its own: a ReLU6 layer has z_out = -128 and an output scale that
maps 127 to at most 6.0, so the saturation is the activation. Only integer numpy operations
are used here; sw/quantize.py derives the parameters.
"""
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class QLayer:
    """Integer parameters of one depthwise or pointwise layer."""
    kind: str            # "dw" (3x3, padding 1) or "pw"
    stride: int
    weight: np.ndarray   # int8 in [-127, 127]; shape (c, 3, 3) for dw, (cout, cin) for pw
    bias: np.ndarray     # int32 per output channel, with the input zero-point folded in
    mult: np.ndarray     # requantization multiplier per output channel, in [2**14, 2**15)
    shift: np.ndarray    # requantization right shift per output channel, in [1, 47]
    z_in: int            # input zero-point; also the depthwise padding value
    z_out: int           # output zero-point
    skip_mult: int = 0   # residual layers only: multiplier, shift and zero-point of the skip tensor
    skip_shift: int = 0
    z_skip: int = 0


def requantize(v, mult, shift):
    """v * mult / 2**shift, rounded to the nearest integer, ties toward +infinity."""
    return (v * mult + (np.int64(1) << (shift - 1))) >> shift


def accumulate(layer: QLayer, x: np.ndarray) -> np.ndarray:
    """Sum of weight * input for every output channel and pixel of the int8 input (c, h, w)."""
    x, w = x.astype(np.int64), layer.weight.astype(np.int64)
    if layer.kind == "pw":
        return np.einsum("oc,chw->ohw", w, x)
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1)), constant_values=layer.z_in)
    windows = sliding_window_view(padded, (3, 3), axis=(1, 2))[:, ::layer.stride, ::layer.stride]
    return np.einsum("chwij,cij->chw", windows, w)


def postprocess(layer: QLayer, acc: np.ndarray, skip: np.ndarray | None = None) -> np.ndarray:
    """Bias, requantization, residual addition and saturation: accumulators to int8 outputs."""
    acc = acc + layer.bias[:, None, None]
    assert np.abs(acc).max() < 2**31, "accumulator + bias must fit in 32 bits"
    y = requantize(acc, layer.mult[:, None, None], layer.shift[:, None, None]) + layer.z_out
    if skip is not None:
        y += requantize(skip.astype(np.int64) - layer.z_skip, layer.skip_mult, layer.skip_shift)
    return np.clip(y, -128, 127).astype(np.int8)


def run_layer(layer: QLayer, x: np.ndarray, skip: np.ndarray | None = None) -> np.ndarray:
    """int8 output (cout, hout, wout) of one layer; `skip` is the int8 tensor of the residual."""
    return postprocess(layer, accumulate(layer, x), skip)
