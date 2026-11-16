import pytest
import torch
import torch.nn as nn

from sw.capture import INPUT_SHAPE, fold_bn, layer_table


@pytest.fixture(scope="module")
def layers(traced):
    return layer_table(fold_bn(traced))


def find_block(layers, expand, dw, project):
    """Return the (expand, depthwise, project) rows matching the given (cin, cout) pairs."""
    for a, b, c in zip(layers, layers[1:], layers[2:]):
        if [(a.kind, a.cin, a.cout), (b.kind, b.cin, b.cout), (c.kind, c.cin, c.cout)] == \
                [("pw", *expand), ("dw", *dw), ("pw", *project)]:
            return a, b, c
    raise AssertionError("block not found")


def test_fold_bn_preserves_output(traced):
    folded = fold_bn(traced)
    x, other = torch.randn(INPUT_SHAPE), torch.randn(INPUT_SHAPE)
    with torch.no_grad():
        ref, out, ref_other = traced(x), folded(x), traced(other)
    scale = ref.abs().max()
    assert (ref - ref_other).abs().max() > 0.01 * scale  # the output depends on the input
    assert (ref - out).abs().max() <= 1e-4 * scale


def test_fold_bn_removes_batchnorm(traced):
    folded = fold_bn(traced)
    convs = [m for m in folded.modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 52 and all(c.bias is not None for c in convs)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in folded.modules())
    assert any(isinstance(m, nn.BatchNorm2d) for m in traced.modules())  # input left untouched


def test_layer_counts_and_shapes(layers):
    kinds = [l.kind for l in layers]
    assert (kinds.count("conv"), kinds.count("dw"), kinds.count("pw")) == (1, 17, 34)
    assert layers[0].kind == "conv" and (layers[0].cin, layers[0].hin) == (3, 224)
    assert all(l.hout == l.hin // l.stride and l.wout == l.win // l.stride for l in layers)
    assert all(l.cin % 8 == 0 and l.cout % 8 == 0 for l in layers if l.kind != "conv")


def test_residuals(layers):
    with_residual = [i for i, l in enumerate(layers) if l.residual]
    assert len(with_residual) == 10
    for i in with_residual:
        project, skip = layers[i], layers[i - 3]  # skip = the layer before this block's expand
        assert project.kind == "pw" and project.act is None
        assert project.residual == skip.name
        assert (skip.cout, skip.hout, skip.wout) == (project.cout, project.hout, project.wout)


def test_proposal_blocks(layers):
    expand, dw, project = find_block(layers, (24, 144), (144, 144), (144, 24))  # early block
    assert (expand.hin, dw.hin, project.hout) == (56, 56, 56) and dw.stride == 1
    assert (expand.act, dw.act, project.act) == ("relu6", "relu6", None)
    assert project.residual is not None

    expand, dw, project = find_block(layers, (160, 960), (960, 960), (960, 320))  # late block
    assert (expand.hin, dw.hin, project.hout) == (7, 7, 7) and dw.stride == 1
    assert project.residual is None
