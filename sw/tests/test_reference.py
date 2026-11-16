from fractions import Fraction
from math import floor

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sw.reference import QLayer, postprocess, requantize, run_layer


def test_requantize_rounds_half_up():
    # v / 2 for v = -3 .. 3:  -1.5  -1  -0.5  0  0.5  1  1.5
    assert [int(requantize(v, 1, 1)) for v in range(-3, 4)] == [-1, -1, 0, 0, 1, 1, 2]


def test_requantize_is_exact_over_the_full_range():
    rng = np.random.default_rng(0)
    v = rng.integers(-2**31, 2**31, 2000)
    mult = rng.integers(2**14, 2**15, 2000)
    shift = rng.integers(1, 48, 2000)
    v[:2], mult[:2], shift[:2] = (2**31 - 1, -2**31), 2**15 - 1, (47, 1)  # the widest products
    exact = [floor(Fraction(int(a) * int(m), 2**int(s)) + Fraction(1, 2)) for a, m, s in zip(v, mult, shift)]
    assert requantize(v, mult, shift).tolist() == exact


@pytest.mark.parametrize("kind, stride", [("pw", 1), ("dw", 1), ("dw", 2)])
def test_layer_matches_definition(kind, stride):
    """Folding z_in into the bias and padding with z_in equals subtracting z_in from every input."""
    rng = np.random.default_rng(0)
    cin, cout, k = (16, 16, 3) if kind == "dw" else (16, 24, 1)
    x = rng.integers(-128, 128, (cin, 8, 8), dtype=np.int8)
    w = rng.integers(-127, 128, (cout, 3, 3) if kind == "dw" else (cout, cin), dtype=np.int8)
    bias = rng.integers(-2**13, 2**13, cout)
    mult, shift = rng.integers(2**14, 2**15, cout), np.full(cout, 23)
    z_in, z_out = -37, 5
    folded_bias = bias - z_in * w.reshape(cout, -1).sum(axis=1)
    layer = QLayer(kind, stride, w, folded_bias.astype(np.int32), mult, shift, z_in, z_out)

    # The definition, with torch as an independent convolution (float64 is exact at these sizes).
    acc = F.conv2d(torch.from_numpy(x.astype(np.float64) - z_in)[None],
                   torch.from_numpy(w.astype(np.float64)).reshape(cout, -1, k, k),
                   torch.from_numpy(bias.astype(np.float64)),
                   stride, k // 2, groups=cin if kind == "dw" else 1)[0].numpy().astype(np.int64)
    want = np.clip(requantize(acc, mult[:, None, None], shift[:, None, None]) + z_out, -128, 127)

    assert np.array_equal(run_layer(layer, x), want)
    assert (np.abs(want) < 127).mean() > 0.5  # most outputs are not saturated


def test_postprocess_rounds_adds_the_residual_and_saturates():
    layer = QLayer("pw", 1, weight=np.zeros((1, 1), np.int8), bias=np.array([10], np.int32),
                   mult=np.array([2**14]), shift=np.array([15]),  # multiplies by 0.5
                   z_in=0, z_out=3,
                   skip_mult=3 * 2**13, skip_shift=15, z_skip=-5)  # multiplies by 0.75
    acc = np.array([[[-21, -20, 0, 1, 244, 1000, -1000]]])
    # (acc + 10) / 2:           -5.5  -5    5  5.5  127   505  -495
    # rounded, plus z_out = 3:    -2  -2    8    9  130   508  -492
    assert postprocess(layer, acc).tolist() == [[[-2, -2, 8, 9, 127, 127, -128]]]

    skip = np.array([[[-5, -3, -7, -5, -18, -5, -5]]], np.int8)
    # (skip + 5) * 0.75:           0  1.5 -1.5   0 -9.75    0     0
    # rounded:                     0    2   -1   0   -10    0     0
    # The fifth output shows that the saturation comes after the addition: 130 - 10 = 120.
    assert postprocess(layer, acc, skip).tolist() == [[[-2, 0, 7, 9, 120, 127, -128]]]
