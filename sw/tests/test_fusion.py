from dataclasses import replace
from math import inf

import numpy as np
import pytest
import torch

from sw.analytical import Network, Traffic
from sw.capture import INPUT_SHAPE
from sw.fusion import (BLOCKS, DW, FUSED, PW, Job, activation_words, build, find_block, make_job, pack,
                       param_words, weight_words, write)
from sw.quantize import run_float, run_int, to_int8
from sw.reference import QLayer, accumulate


def byte(word, k):
    """Byte k of a 64-bit word, as a signed integer."""
    value = int(word) >> 8 * k & 0xFF
    return value - 256 if value > 127 else value


def qlayer(kind, weight, **fields):
    """A layer for the layout tests, with neutral values in the fields that are not given."""
    cout = len(weight)
    neutral = dict(stride=1, bias=np.zeros(cout, np.int32), mult=np.full(cout, 2**14), shift=np.full(cout, 15),
                   z_in=0, z_out=0)
    return QLayer(kind=kind, weight=weight, **(neutral | fields))


def read_hex(path):
    """DRAM address and words of an image file."""
    address, *words = path.read_text().split()
    return int(address[1:], 16), np.array([int(word, 16) for word in words], np.uint64)


@pytest.fixture(scope="module")
def model(quantized):
    """The test model as the fusion pass sees it: the network, its integer layers by name, and
    the int8 output of every layer for one random image."""
    layers, params, qparams, qlayers = quantized
    stem = layers[0].name
    torch.manual_seed(0)
    x = to_int8(run_float(layers[:1], params, torch.randn(INPUT_SHAPE))[stem][0], *qparams[stem])
    return Network(layers), dict(zip((l.name for l in layers[1:]), qlayers)), run_int(layers, qlayers, x)


@pytest.fixture(scope="module")
def plans(model):
    net = model[0]
    return {"unfused": net.select(inf, "unfused"), "fused": net.select(inf, "rtl")}


def test_pack():
    job = Job(hin=56, win=28, cin=144, cout=24, mode=FUSED, stride=2, residual=1,
              z_pad=-128, z_dw=-1, z_pw=5, z_skip=-7, skip_shift=17, skip_mult=0x5ABC,
              input=0x11111111, output=0x22222222, skip=0x33333333, weights=0x44444444, params=0x55555555)
    assert pack(job) == [0x0018_0090_001C_0038,        # cout, cin, win, hin
                         0x11_F9_05_FF_80_01_02_02,    # skip_shift, z_skip, z_pw, z_dw, z_pad, residual, stride, mode
                         0x11111111_00005ABC,          # input, skip_mult
                         0x33333333_22222222,          # skip, output
                         0x55555555_44444444]          # params, weights
    with pytest.raises(AssertionError):
        pack(replace(job, cin=2**16))


def test_activation_words():
    x = np.random.default_rng(0).integers(-128, 128, (16, 3, 5), dtype=np.int8)
    words = activation_words(x)
    assert len(words) == 3 * 5 * 2
    for channel, row, pixel in np.ndindex(x.shape):
        assert byte(words[(row * 5 + pixel) * 2 + channel // 8], channel % 8) == x[channel, row, pixel]


def test_pointwise_weights_are_in_array_order():
    """Accumulating in the order the systolic array reads the image gives the reference's sums."""
    rng = np.random.default_rng(0)
    layer = qlayer("pw", rng.integers(-127, 128, (16, 24), dtype=np.int8))
    x = rng.integers(-128, 128, (24, 4, 4), dtype=np.int8)
    words = iter(weight_words(layer))
    acc = np.zeros((16, 4, 4), np.int64)
    for out_group in range(2):
        for in_group in range(3):
            for row in range(8):         # one word per array row: an input channel
                word = next(words)
                for column in range(8):  # one byte per array column: an output channel
                    acc[8 * out_group + column] += byte(word, column) * x[8 * in_group + row].astype(np.int64)
    assert next(words, None) is None
    assert np.array_equal(acc, accumulate(layer, x))


def test_depthwise_weights_are_one_word_per_tap():
    rng = np.random.default_rng(0)
    layer = qlayer("dw", rng.integers(-127, 128, (16, 3, 3), dtype=np.int8), z_in=-3)
    x = rng.integers(-128, 128, (16, 5, 6), dtype=np.int8)
    padded = np.pad(x.astype(np.int64), ((0, 0), (1, 1), (1, 1)), constant_values=-3)
    words = iter(weight_words(layer))
    acc = np.zeros((16, 5, 6), np.int64)
    for group in range(2):
        for tap in range(9):             # one word per tap of the 3x3 window
            word = next(words)
            for lane in range(8):        # one byte per lane: a channel
                window = padded[8 * group + lane, tap // 3:tap // 3 + 5, tap % 3:tap % 3 + 6]
                acc[8 * group + lane] += byte(word, lane) * window
    assert next(words, None) is None
    assert np.array_equal(acc, accumulate(layer, x))


def test_param_words():
    bias = np.array([-2**31, 2**31 - 1, -1, 0, 5, -6, 7, 8], np.int32)
    mult = np.array([2**14, 2**15 - 1] * 4)
    shift = np.array([1, 47, 15, 16, 17, 18, 19, 20])
    words = param_words(qlayer("pw", np.zeros((8, 8), np.int8), bias=bias, mult=mult, shift=shift))
    for word, b, m, s in zip(words, bias, mult, shift, strict=True):
        assert int(word) == int(b) % 2**32 | int(m) << 32 | int(s) << 48


def test_jobs_of_the_early_block(model, plans):
    net, qlayers, tensors = model
    start = find_block(net.layers, *BLOCKS["early"])
    images, jobs = build(net, qlayers, tensors, start, plans)
    expand, dw, project = jobs["unfused"]
    same_expand, fused = jobs["fused"]
    q_dw, q_project = (qlayers[l.name] for l in net.layers[start + 1:start + 3])

    assert [(j.mode, j.hin, j.win, j.cin, j.cout, j.stride, j.residual) for j in (expand, dw, project)] == \
        [(PW, 56, 56, 24, 144, 1, 0), (DW, 56, 56, 144, 144, 1, 0), (PW, 56, 56, 144, 24, 1, 1)]
    # Each job reads what the previous one wrote, and the project layer adds the block's input.
    assert expand.input == project.skip == images["input"][0]
    assert expand.output == dw.input == images["expand"][0]
    assert dw.output == project.input == images["depthwise"][0]
    assert project.output == images["project"][0]
    # Both layers around the depthwise layer end in ReLU6, so its zero-points are -128.
    assert (dw.z_pad, dw.z_dw, dw.z_pw) == (q_dw.z_in, q_dw.z_out, 0) == (-128, -128, 0)
    assert (project.z_pad, project.z_dw, project.z_pw) == (0, 0, q_project.z_out)
    assert (project.z_skip, project.skip_shift, project.skip_mult) == \
        (q_project.z_skip, q_project.skip_shift, q_project.skip_mult) and project.skip_mult > 0
    # In DRAM, the project layer's weights and parameters follow the depthwise layer's.
    assert (project.weights, project.params) == (dw.weights + 144 * 9 // 8, dw.params + 144)

    # The fused job is the project job with the depthwise engine in front of it.
    assert same_expand == expand
    assert fused == replace(project, mode=FUSED, z_pad=dw.z_pad, z_dw=dw.z_dw, input=dw.input,
                            weights=dw.weights, params=dw.params)


@pytest.mark.parametrize("label", BLOCKS)
def test_images_fill_dram_and_match_the_analytical_model(model, plans, label):
    net, qlayers, tensors = model
    start = find_block(net.layers, *BLOCKS[label])
    images, jobs = build(net, qlayers, tensors, start, plans)
    assert [len(jobs["unfused"]), len(jobs["fused"])] == [3, 2]
    end = 0
    for address, words in images.values():  # one after another from address 0
        assert address == end
        end += len(words)

    def traffic(plan):
        return sum((g.traffic for g in plan if start <= g.start < start + 3), Traffic())

    size = {file: 8 * len(words) for file, (_, words) in images.items()}
    unfused, fused = traffic(plans["unfused"]), traffic(plans["fused"])
    assert unfused.weights == fused.weights == size["weights"]
    assert unfused.params == fused.params == size["params"]
    assert unfused.residuals == fused.residuals == (size["input"] if jobs["fused"][1].residual else 0)
    # The expand output is written and read back; so is the depthwise output, unless the pair is fused.
    assert unfused.inputs + unfused.outputs == size["input"] + 2 * size["expand"] + 2 * size["depthwise"] + size["project"]
    assert fused.inputs + fused.outputs == size["input"] + 2 * size["expand"] + size["project"]


def test_written_files(model, plans, tmp_path):
    net, qlayers, tensors = model
    start = find_block(net.layers, *BLOCKS["middle"])
    images, jobs = build(net, qlayers, tensors, start, plans)
    write(tmp_path, images, jobs)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [f"{file}.hex" for file in images] + ["jobs_fused.hex", "jobs_unfused.hex"])
    for file, (address, words) in images.items():
        read_address, read_words = read_hex(tmp_path / f"{file}.hex")
        assert read_address == address and np.array_equal(read_words, words)

    # The project file holds the reference model's output of the project layer.
    project = net.layers[start + 2]
    _, words = read_hex(tmp_path / "project.hex")
    pixels = words.astype("<u8").view(np.int8).reshape(project.hout, project.wout, project.cout)
    assert np.array_equal(pixels.transpose(2, 0, 1), tensors[project.name])

    lines = (tmp_path / "jobs_fused.hex").read_text().split()
    assert all(len(line) == 16 for line in lines)
    assert [int(line, 16) for line in lines] == [word for job in jobs["fused"] for word in pack(job)]


def test_unsupported_jobs_are_rejected(model):
    net, qlayers, _ = model
    start = find_block(net.layers, *BLOCKS["early"])
    # The stem convolution, a pointwise->depthwise pair and a pointwise->pointwise pair.
    for first, stop in ((0, 1), (start, start + 2), (start + 2, start + 4)):
        with pytest.raises(AssertionError):
            make_job(net, qlayers, first, stop, {})
