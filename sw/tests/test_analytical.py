from dataclasses import replace
from math import inf

import pytest
import torch.nn as nn
import torchvision

from sw.analytical import BRAM36_BYTES, Network, Traffic, param_bytes, weight_bytes
from sw.capture import fold_bn, layer_table

EARLY = [(24, 144), (144, 144), (144, 24)]   # (cin, cout) of the expand, depthwise and project layers
LATE = [(160, 960), (960, 960), (960, 320)]


@pytest.fixture(scope="module")
def mobilenet(traced):
    return Network(layer_table(fold_bn(traced)))


@pytest.fixture(scope="module")
def resnet(resnet_layers):
    return Network(resnet_layers)


def block(net, channels):
    """Index of the first layer of the block whose layers have the given (cin, cout) pairs."""
    pairs = [(l.cin, l.cout) for l in net.layers]
    return next(i for i in range(len(pairs)) if pairs[i:i + len(channels)] == channels)


def traffic(net, *groups):
    """Total traffic of the given (start, stop) groups."""
    return sum((net.traffic(start, stop) for start, stop in groups), Traffic())


def total(plan):
    return sum((group.traffic for group in plan), Traffic())


def test_worked_example(mobilenet):
    """Table 1 of the proposal: an early block, with the residual connection counted separately."""
    e = block(mobilenet, EARLY)
    unfused = traffic(mobilenet, (e, e + 1), (e + 1, e + 2), (e + 2, e + 3))
    pair = traffic(mobilenet, (e, e + 1), (e + 1, e + 3))
    whole = traffic(mobilenet, (e, e + 3))
    assert [t.inputs + t.outputs for t in (unfused, pair, whole)] == [1_956_864, 1_053_696, 150_528]
    assert [t.residuals for t in (unfused, pair, whole)] == [75_264, 75_264, 0]
    assert unfused.weights == pair.weights == whole.weights == 8_208
    assert unfused.params == pair.params == whole.params == 8 * (144 + 144 + 24)


def test_late_block(mobilenet):
    """Table 2 of the proposal: the expanded intermediate and the weights of a late block."""
    l = block(mobilenet, LATE)
    assert mobilenet.traffic(l, l + 1).outputs == 47_040
    assert mobilenet.traffic(l, l + 3).weights == 469_440


def test_weights_match_the_models(traced, mobilenet, resnet):
    def conv_weights(model):
        return sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Conv2d))
    assert sum(map(weight_bytes, mobilenet.layers)) == conv_weights(traced)
    assert sum(map(weight_bytes, resnet.layers)) == conv_weights(torchvision.models.resnet18(weights=None))
    assert mobilenet.traffic(0, 3, refetch=5).weights == 5 * mobilenet.traffic(0, 3).weights


def test_params_are_one_record_per_output_channel(traced, mobilenet, resnet):
    channels = sum(m.out_channels for m in traced.modules() if isinstance(m, nn.Conv2d))
    assert sum(map(param_bytes, mobilenet.layers)) == 8 * channels
    assert mobilenet.traffic(0, 3, refetch=5).params == 5 * mobilenet.traffic(0, 3).params
    assert [param_bytes(l) for l in resnet.layers if l.kind == "pool"] == [0]


def test_resnet_block_reads_its_input_once_when_fused(resnet):
    """A block with a shortcut convolution: 64 x 56 x 56 in, 128 x 28 x 28 everywhere else."""
    i = next(i for i, l in enumerate(resnet.layers) if l.name == "layer2_0_conv1")
    singles = traffic(resnet, (i, i + 1), (i + 1, i + 2), (i + 2, i + 3))
    assert (singles.inputs, singles.residuals, singles.outputs) == (200_704 + 100_352 + 200_704, 100_352, 301_056)
    fused = resnet.traffic(i, i + 3)
    assert (fused.inputs, fused.residuals, fused.outputs) == (200_704, 0, 100_352)
    # The stride-2 shortcut alone: two input rows per output row, and one row of the main path to add.
    assert resnet.buffers(i + 2, i + 3, tile=1).inputs == 2 * 56 * 64 + 28 * 128


def test_tensor_used_inside_and_outside_a_group_is_still_written(mobilenet):
    e = block(mobilenet, EARLY)
    # Layer e - 1 ends the previous block. Its output feeds layer e and is also this block's skip tensor.
    assert mobilenet.traffic(e - 1, e + 1).outputs == 75_264 + 451_584


def test_pair_buffers(mobilenet):
    """The early depthwise->pointwise pair; the proposal gives 8,064 bytes per bank and 1,792 of accumulators."""
    dw = block(mobilenet, EARLY) + 1
    two, one = mobilenet.buffers(dw, dw + 2, tile=1), mobilenet.buffers(dw, dw + 2, tile=1, banks=1)
    assert (two.intermediates, one.intermediates, two.accumulators) == (2 * 8_064, 8_064, 1_792)
    assert two.inputs == 3 * 56 * 144 + 56 * 24  # three-row line buffer, and one row of the skip tensor
    assert (two.outputs, two.weights, two.params) == (56 * 24, 144 * 9 + 144 * 24, 8 * (144 + 24))
    assert two.total == 50_896
    assert mobilenet.buffers(dw, dw + 2, tile=1, resident=False).total == 50_896 - two.weights - two.params


def test_buffers_grow_with_the_tile_up_to_whole_tensors(mobilenet):
    dw = block(mobilenet, EARLY) + 1
    sizes = [mobilenet.buffers(dw, dw + 2, tile, resident=False) for tile in range(1, 57)]
    assert all(a.total < b.total for a, b in zip(sizes, sizes[1:]))
    whole = sizes[-1]
    assert (whole.inputs, whole.intermediates, whole.outputs) == (451_584 + 75_264, 2 * 451_584, 75_264)


def test_whole_block_buffers(mobilenet):
    e = block(mobilenet, EARLY)
    whole = mobilenet.buffers(e, e + 3, tile=1)
    # One input row for the expand layer, and three rows of the same tensor waiting for the addition.
    assert whole.inputs == (1 + 3) * 56 * 24
    # Depthwise input: two banks of one row plus two kept rows. Project input: two banks of one row.
    assert whole.intermediates == (2 + 2) * 56 * 144 + 2 * 56 * 144
    assert whole.accumulators == 1_792  # the two pointwise layers share the array


def test_stride_two_tiles(mobilenet):
    dw = next(i for i, l in enumerate(mobilenet.layers) if l.kind == "dw" and l.stride == 2)
    expand, l = mobilenet.layers[dw - 1], mobilenet.layers[dw]
    alone = mobilenet.buffers(dw, dw + 1, tile=3)
    assert alone.inputs == (2 * 3 + 1) * l.win * l.cin  # six new rows and one kept row
    assert alone.accumulators == 0                       # the depthwise engine does not accumulate
    # With the expand layer in front, one output row needs two rows from the expand layer.
    both = mobilenet.buffers(dw - 1, dw + 1, tile=1)
    assert both.inputs == 2 * expand.win * expand.cin
    assert both.intermediates == (2 * 2 + 1) * l.win * l.cin


def test_best_trades_tile_height_for_weight_refetches(mobilenet):
    late = block(mobilenet, LATE) + 1  # the late pair: 7 output rows, 315,840 bytes of weights
    weights = 960 * 9 + 960 * 320

    # Room for everything: one tile covers the map, so the weights are read once without a weight buffer.
    roomy = mobilenet.best(late, late + 2, budget=inf)
    assert (roomy.tile, roomy.resident, roomy.traffic.weights) == (7, False, weights)
    # Less room: two tiles of four rows, so the weights are read twice.
    tight = mobilenet.best(late, late + 2, budget=120_000)
    assert (tight.tile, tight.resident, tight.traffic.weights) == (4, False, 2 * weights)
    assert tight.buffers.total <= 120_000 < mobilenet.buffers(late, late + 2, 7, resident=False).total
    # Just enough for one-row tiles, and then one byte less.
    smallest = mobilenet.buffers(late, late + 2, 1, resident=False).total
    assert mobilenet.best(late, late + 2, budget=smallest).traffic.weights == 7 * weights
    assert mobilenet.best(late, late + 2, budget=smallest - 1) is None

    # The early pair has small weights and large rows: keeping the weights on chip is cheapest.
    early = block(mobilenet, EARLY) + 1
    roomy = mobilenet.best(early, early + 2, budget=inf)
    assert (roomy.tile, roomy.resident) == (1, True)
    # Even when re-fetching them would need less memory: traffic decides, memory only breaks ties.
    assert mobilenet.best(early, early + 2, budget=51_000).resident


def test_supported_groups(mobilenet):
    dw = block(mobilenet, EARLY) + 1
    assert mobilenet.supported(dw, dw + 2, "rtl") and not mobilenet.supported(dw, dw + 2, "unfused")
    assert not mobilenet.supported(dw - 1, dw + 1, "rtl")  # pointwise->depthwise
    assert not mobilenet.supported(dw + 1, dw + 3, "rtl")  # pointwise->pointwise
    assert not mobilenet.supported(dw - 1, dw + 2, "rtl") and mobilenet.supported(dw - 1, dw + 2, "any")
    # A depthwise layer followed by something else, or by a pointwise layer that reads another tensor.
    for change in ({"kind": "conv"}, {"src": mobilenet.layers[dw - 1].name}):
        layers = list(mobilenet.layers)
        layers[dw + 1] = replace(layers[dw + 1], **change)
        assert not Network(layers).supported(dw, dw + 2, "rtl")


def test_unfused_selection(mobilenet):
    n = len(mobilenet.layers)
    plan = mobilenet.select(inf, "unfused")
    assert [(g.start, g.stop) for g in plan] == [(i, i + 1) for i in range(n)]
    assert total(plan) == traffic(mobilenet, *((i, i + 1) for i in range(n)))


def test_rtl_selection_fuses_every_pair_when_memory_allows(mobilenet):
    unfused, fused = mobilenet.select(inf, "unfused"), mobilenet.select(inf, "rtl")
    assert sum(g.stop - g.start == 2 for g in fused) == 17 and len(fused) == len(mobilenet.layers) - 17
    # Every fused pair saves one write and one read of the depthwise output.
    dw_outputs = sum(l.cout * l.hout * l.wout for l in mobilenet.layers if l.kind == "dw")
    assert total(unfused).activations - total(fused).activations == 2 * dw_outputs
    assert total(unfused).weights == total(fused).weights


def test_any_selection_with_unlimited_memory_is_one_group(mobilenet, resnet):
    for net in (mobilenet, resnet):
        (group,) = net.select(inf, "any")
        first, last = net.layers[0], net.layers[-1]
        assert group.traffic.activations == first.cin * first.hin * first.win + last.cout * last.hout * last.wout
        assert group.traffic.weights == sum(map(weight_bytes, net.layers))


@pytest.mark.parametrize("groups", ["rtl", "any"])
def test_selection_is_optimal(mobilenet, groups):
    """Compare the dynamic program with an exhaustive search on the first ten layers."""
    net = Network(mobilenet.layers[:10])
    budget = 40_000  # enough for every single layer, too little for some of the pairs

    def cheapest(start):
        if start == len(net.layers):
            return 0
        candidates = (net.best(start, stop, budget) for stop in range(start + 1, len(net.layers) + 1)
                      if net.supported(start, stop, groups))
        return min((g.traffic.total + cheapest(g.stop) for g in candidates if g), default=inf)

    plan = net.select(budget, groups)
    assert total(plan).total == cheapest(0)
    assert any(g.stop - g.start > 1 for g in plan)                              # something is fused
    assert total(plan).total > total(net.select(inf, groups)).total            # and the budget matters


def test_selection_respects_the_budget(mobilenet, resnet):
    for net in (mobilenet, resnet):
        previous = inf
        for blocks in (16, 32, 64, 140):
            plan = net.select(blocks * BRAM36_BYTES, "any")
            assert all(g.buffers.total <= blocks * BRAM36_BYTES for g in plan)
            assert [g.start for g in plan] == [0] + [g.stop for g in plan[:-1]]
            assert plan[-1].stop == len(net.layers)
            assert total(plan).total <= previous  # more memory never costs traffic
            previous = total(plan).total


def test_selection_fails_when_a_layer_cannot_fit(mobilenet):
    with pytest.raises(ValueError):
        mobilenet.select(BRAM36_BYTES, "any")
