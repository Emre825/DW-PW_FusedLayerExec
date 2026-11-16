"""Analytical model: off-chip traffic and on-chip buffers of layer groups, and group selection.

A group is a run of consecutive rows of the layer table, executed together one row tile at a
time. A tensor produced and consumed inside a group stays on chip; everything else crosses the
chip boundary. All sizes are in bytes: int8 activations and weights, one byte each, and one
8-byte parameter record (bias, requantization multiplier and shift) per output channel.

Traffic of a group (Eq. 2-3 of the proposal):
  - a tensor from outside the group is read once, however many of the group's layers use it;
  - a tensor that a layer outside the group needs, or the network's output, is written once;
  - weights and parameters are read once if they are kept on chip, and once per tile otherwise.
Line buffers keep the rows a k x k window shares with the previous tile, so no input is read twice.

Buffers of a group for a tile of `tile` output rows (Eq. 4):
  - a layer's input: the rows of one tile plus the k - s rows kept from the previous tile; if the
    input is produced inside the group, the tile is stored `banks` times (one bank is filled
    while the other is drained);
  - a skip tensor: one tile for every layer of the group that it bypasses;
  - one tile of every tensor that is written out;
  - int32 accumulators for one tile of 8 output channels (pointwise and standard convolutions);
  - the group's weights and parameters, if they are kept on chip.
"""
from dataclasses import astuple, dataclass
from math import ceil, inf

from sw.capture import Layer

BRAM36_BYTES = 4096  # one block RAM of the XC7Z020, which has 140 of them
ARRAY_COLUMNS = 8    # output channels the systolic array accumulates at once, in int32
PARAM_BYTES = 8      # per output channel: int32 bias, multiplier and shift, in one 64-bit word


@dataclass(frozen=True)
class Traffic:
    """Bytes that cross the chip boundary."""
    inputs: int = 0     # activations read as layer inputs
    residuals: int = 0  # activations read only to be added as a skip connection
    outputs: int = 0    # activations written
    weights: int = 0    # weights read, re-fetches included
    params: int = 0     # parameter records read, re-fetches included

    @property
    def activations(self) -> int:
        return self.inputs + self.residuals + self.outputs

    @property
    def total(self) -> int:
        return self.activations + self.weights + self.params

    def __add__(self, other: "Traffic") -> "Traffic":
        return Traffic(*(a + b for a, b in zip(astuple(self), astuple(other))))


@dataclass(frozen=True)
class Buffers:
    """On-chip memory a group needs, by the terms of Eq. 4."""
    inputs: int
    intermediates: int
    weights: int
    params: int
    outputs: int
    accumulators: int

    @property
    def total(self) -> int:
        return sum(astuple(self))


@dataclass(frozen=True)
class Group:
    """The layers[start:stop] of a network and how they are run."""
    start: int
    stop: int
    tile: int        # tile height, in rows of the group's last output
    resident: bool   # weights and parameters kept on chip and read once, rather than re-fetched for every tile
    traffic: Traffic
    buffers: Buffers


def weight_bytes(l: Layer) -> int:
    if l.kind == "pool":
        return 0
    return l.cout * (1 if l.kind == "dw" else l.cin) * l.kernel**2


def param_bytes(l: Layer) -> int:
    return 0 if l.kind == "pool" else PARAM_BYTES * l.cout


class Network:
    """Traffic and buffer model of one layer table."""

    def __init__(self, layers: list[Layer]):
        self.layers = layers
        index = {l.name: i for i, l in enumerate(layers)}
        self.src = [index.get(l.src, -1) for l in layers]   # producer of each layer's input; -1 = network input
        self.skip = [index.get(l.residual) for l in layers]  # producer of each layer's skip tensor, or None
        self.users = [[] for _ in layers]                    # users[p] = (layer that reads p's output, as a skip?)
        for i in range(len(layers)):
            if self.src[i] >= 0:
                self.users[self.src[i]].append((i, False))
            if self.skip[i] is not None:
                self.users[self.skip[i]].append((i, True))

    def _written(self, inside: range) -> list[int]:
        """Layers of the group whose output is needed outside it."""
        return [i for i in inside if not self.users[i] or any(c not in inside for c, _ in self.users[i])]

    def traffic(self, start: int, stop: int, refetch: int = 1) -> Traffic:
        """Traffic of layers[start:stop] run as one group, its weights and parameters being read
        `refetch` times."""
        inside = range(start, stop)
        read, added = {}, {}  # producer -> size of the tensors read from outside, as an input or as a skip
        for i in inside:
            l = self.layers[i]
            if self.src[i] not in inside:
                read[self.src[i]] = l.cin * l.hin * l.win
            if self.skip[i] is not None and self.skip[i] not in inside:
                added[self.skip[i]] = l.cout * l.hout * l.wout
        written = [self.layers[i] for i in self._written(inside)]
        return Traffic(inputs=sum(read.values()),
                       residuals=sum(size for p, size in added.items() if p not in read),
                       outputs=sum(l.cout * l.hout * l.wout for l in written),
                       weights=refetch * sum(weight_bytes(self.layers[i]) for i in inside),
                       params=refetch * sum(param_bytes(self.layers[i]) for i in inside))

    def buffers(self, start: int, stop: int, tile: int, banks: int = 2, resident: bool = True) -> Buffers:
        """Buffers of layers[start:stop] run as one group, `tile` rows of its last output at a time."""
        inside = range(start, stop)
        rows = {}  # rows of each layer's output produced per tile
        for i in reversed(inside):
            needed = [rows[c] * (1 if as_skip else self.layers[c].stride)
                      for c, as_skip in self.users[i] if c in inside]
            rows[i] = min(max(needed, default=tile), self.layers[i].hout)

        inputs = intermediates = accumulators = 0
        for i in inside:
            l = self.layers[i]
            new = min(rows[i] * l.stride, l.hin)
            kept = min(max(l.kernel - l.stride, 0), l.hin - new)
            if self.src[i] in inside:
                intermediates += (banks * new + kept) * l.win * l.cin
            else:
                inputs += (new + kept) * l.win * l.cin
            p = self.skip[i]
            if p is not None:
                # The skip tensor waits from the layer after `origin` until layer i adds it.
                origin = p if p in inside else min((j for j in inside if self.src[j] == p), default=i) - 1
                held = min((i - origin) * rows[i], l.hout) * l.wout * l.cout
                if p in inside:
                    intermediates += held
                else:
                    inputs += held
            if l.kind in ("pw", "conv"):
                accumulators = max(accumulators, rows[i] * l.wout * ARRAY_COLUMNS * 4)
        return Buffers(inputs, intermediates,
                       weights=sum(weight_bytes(self.layers[i]) for i in inside) if resident else 0,
                       params=sum(param_bytes(self.layers[i]) for i in inside) if resident else 0,
                       outputs=sum(rows[i] * self.layers[i].wout * self.layers[i].cout for i in self._written(inside)),
                       accumulators=accumulators)

    def best(self, start: int, stop: int, budget: float, banks: int = 2) -> Group | None:
        """Least-traffic way to run layers[start:stop] within `budget` bytes of buffers, or None."""
        height = self.layers[stop - 1].hout
        options = []
        buffers = self.buffers(start, stop, 1, banks, resident=True)
        if buffers.total <= budget:
            options.append(Group(start, stop, 1, True, self.traffic(start, stop), buffers))
        if self.buffers(start, stop, 1, banks, resident=False).total <= budget:
            # Without the weights on chip, fewer and taller tiles mean fewer re-fetches.
            for tile in sorted({ceil(height / n) for n in range(1, height + 1)}, reverse=True):
                buffers = self.buffers(start, stop, tile, banks, resident=False)
                if buffers.total <= budget:
                    traffic = self.traffic(start, stop, refetch=ceil(height / tile))
                    options.append(Group(start, stop, tile, False, traffic, buffers))
                    break
        return min(options, key=lambda g: (g.traffic.total, g.buffers.total), default=None)

    def supported(self, start: int, stop: int, groups: str) -> bool:
        """Whether layers[start:stop] may form a group: "unfused" allows single layers only,
        "rtl" also depthwise->pointwise pairs, and "any" every run of layers."""
        if groups == "any" or stop - start == 1:
            return True
        pair = (stop - start == 2 and self.layers[start].kind == "dw"
                and self.layers[start + 1].kind == "pw" and self.src[start + 1] == start)
        return groups == "rtl" and pair

    def select(self, budget: float, groups: str = "rtl", banks: int = 2) -> list[Group]:
        """Split the network into groups of least total traffic, each within `budget` bytes (Eq. 5)."""
        n = len(self.layers)
        cost, plan = [0] + [inf] * n, [[]] + [None] * n  # for the first i layers
        for stop in range(1, n + 1):
            for start in range(stop):
                if plan[start] is None or not self.supported(start, stop, groups):
                    continue
                group = self.best(start, stop, budget, banks)
                if group and cost[start] + group.traffic.total < cost[stop]:
                    cost[stop], plan[stop] = cost[start] + group.traffic.total, plan[start] + [group]
        if plan[n] is None:
            raise ValueError(f"no layer grouping fits in {budget} bytes")
        return plan[n]


if __name__ == "__main__":
    import torchvision

    from sw.capture import fold_bn, layer_table, trace

    def network(name: str) -> Network:
        return Network(layer_table(fold_bn(trace(getattr(torchvision.models, name)(weights=None)))))

    def summary(plan: list[Group]) -> str:
        traffic = sum((g.traffic for g in plan), Traffic())
        fused = sum(g.stop - g.start > 1 for g in plan)
        stored = traffic.weights + traffic.params
        return f"{traffic.activations / 1e6:5.2f} + {stored / 1e6:5.2f}  ({fused:>2} fused)"

    for name in ("mobilenet_v2", "resnet18"):
        net = network(name)
        print(f"\n{name}: off-chip traffic in MB, activations + weights and parameters, by on-chip memory budget")
        print(f"{'BRAM36':>6}   {'unfused':<26} {'dw->pw pairs fused':<26} any run of layers fused")
        for blocks in (16, 32, 64, 140):
            plans = (net.select(blocks * BRAM36_BYTES, groups) for groups in ("unfused", "rtl", "any"))
            print(f"{blocks:>6}   " + "  ".join(f"{summary(plan):<24}" for plan in plans))

    net = network("mobilenet_v2")
    print("\nmobilenet_v2: buffer bytes of three dw->pw pairs by tile height, two banks, weights and parameters"
          " on chip")
    print(f"{'pair':<22} {'tile':>4} {'inputs':>8} {'fusion':>8} {'outputs':>8} {'accum.':>7} {'weights':>8}"
          f" {'params':>7} {'total':>8} {'BRAM36':>7}")
    for label, channels, height, cout in (("early", 144, 56, 24), ("middle", 384, 14, 64), ("late", 960, 7, 320)):
        dw = next(i for i, l in enumerate(net.layers) if l.kind == "dw" and (l.cin, l.hin) == (channels, height)
                  and net.layers[i + 1].cout == cout)
        for tile in (1, 2, 4):
            b = net.buffers(dw, dw + 2, tile)
            print(f"{f'{label} {height}x{height}, {channels}->{cout}':<22} {tile:>4} {b.inputs:>8,}"
                  f" {b.intermediates:>8,} {b.outputs:>8,} {b.accumulators:>7,} {b.weights:>8,} {b.params:>7,}"
                  f" {b.total:>8,} {b.total / BRAM36_BYTES:>7.1f}")
