"""Fusion pass back end: job descriptors and the memory images that the accelerator reads.

A job is one pointwise layer, one depthwise layer, or a depthwise->pointwise pair run fused. The
analytical model's group selection decides which; this module turns the groups into what the
hardware needs.

Everything in DRAM is a stream of 64-bit words, and addresses count words. Byte k of a word is
bits 8k+7..8k.

  Activations        One word holds 8 consecutive channels of one pixel: byte k is channel 8g + k.
                     Words are ordered by row, then pixel, then channel group g.
  Pointwise weights  For each group of 8 output channels and each group of 8 input channels, the
                     8x8 tile that the systolic array holds: word i is array row i (an input
                     channel) and its byte j is column j (an output channel).
  Depthwise weights  For each group of 8 channels, nine words: one per tap of the 3x3 window, in
                     row-major order. Byte k of a word is that tap of channel 8g + k.
  Parameters         One word per output channel: the int32 bias in bits 31..0, the
                     requantization multiplier in bits 47..32 and the shift in bits 55..48.

Job descriptors do not go through DRAM: they fill the controller's job queue, five words each.
"""
from dataclasses import astuple, dataclass
from pathlib import Path

import numpy as np

from sw.analytical import BRAM36_BYTES, PARAM_BYTES, Group, Network
from sw.capture import Layer
from sw.reference import QLayer

PW, DW, FUSED = range(3)  # Job.mode
MODES = {("pw",): PW, ("dw",): DW, ("dw", "pw"): FUSED}  # by the kinds of a job's layers

# Input, expanded and output channels of the MobileNetV2 blocks that are run in RTL.
BLOCKS = {"early": (24, 144, 24), "middle": (64, 384, 64), "late": (160, 960, 320)}


@dataclass(frozen=True, kw_only=True)
class Job:
    """Descriptor of one job. The fields of an engine that the job does not use are zero."""
    hin: int             # input height and width, in pixels
    win: int
    cin: int             # input channels; a depthwise layer has as many output channels
    cout: int            # output channels of the job
    mode: int            # PW, DW or FUSED
    stride: int          # 1 or 2 for a depthwise layer; a pointwise layer always has 1
    residual: int = 0    # 1 if the pointwise layer adds the skip tensor
    z_pad: int = 0       # depthwise input zero-point, which is also the padding value
    z_dw: int = 0        # depthwise output zero-point
    z_pw: int = 0        # pointwise output zero-point
    z_skip: int = 0      # skip tensor: zero-point, requantization shift and multiplier
    skip_shift: int = 0
    skip_mult: int = 0
    input: int           # DRAM addresses of the input, output and skip tensors
    output: int
    skip: int = 0
    weights: int         # DRAM addresses of the weights and parameters; in a fused job, those of
    params: int          # the pointwise layer follow those of the depthwise layer


# Width in bits of each Job field, one line per 64-bit word. The first field of a word is in its
# lowest bits, and zero-points are in two's complement.
WIDTHS = (16, 16, 16, 16,
          8, 8, 8, 8, 8, 8, 8, 8,
          32, 32,
          32, 32,
          32, 32)


def pack(job: Job) -> list[int]:
    """The descriptor as five 64-bit words."""
    value = shift = 0
    for field, bits in zip(astuple(job), WIDTHS, strict=True):
        assert -(1 << (bits - 1)) <= field < (1 << bits), f"{field} does not fit in {bits} bits"
        value |= (field & ((1 << bits) - 1)) << shift
        shift += bits
    return [(value >> s) & (2**64 - 1) for s in range(0, shift, 64)]


def to_words(a: np.ndarray) -> np.ndarray:
    """An int8 array whose last axis is the 8 bytes of a word, lowest byte first, as 64-bit words."""
    return np.ascontiguousarray(a).view("<u8").ravel()


def activation_words(x: np.ndarray) -> np.ndarray:
    """Image of the int8 tensor `x` (c, h, w)."""
    c, h, w = x.shape
    return to_words(x.reshape(c // 8, 8, h, w).transpose(2, 3, 0, 1))


def weight_words(q: QLayer) -> np.ndarray:
    """Weight image of one layer, in the order its engine reads it."""
    if q.kind == "dw":
        return to_words(q.weight.reshape(-1, 8, 9).transpose(0, 2, 1))
    cout, cin = q.weight.shape
    return to_words(q.weight.reshape(cout // 8, 8, cin // 8, 8).transpose(0, 2, 3, 1))


def param_words(q: QLayer) -> np.ndarray:
    """Parameter image of one layer."""
    assert PARAM_BYTES == 8  # the analytical model counts one word per record
    return q.bias.astype(np.uint32).astype(np.uint64) | q.mult.astype(np.uint64) << 32 | q.shift.astype(np.uint64) << 48


def make_job(net: Network, qlayers: dict[str, QLayer], start: int, stop: int, at: dict) -> Job:
    """Descriptor of layers[start:stop] run as one job; `at` gives the DRAM address of every image."""
    layers = net.layers[start:stop]
    first, last = layers[0], layers[-1]
    kinds = tuple(l.kind for l in layers)
    assert kinds in MODES and (first is last or last.src == first.name), f"cannot run {kinds} as one job"
    assert first.cin % 8 == 0 and last.cout % 8 == 0, "channel counts must be multiples of 8"
    engines = {}
    if first.kind == "dw":
        assert first.kernel == 3 and first.stride in (1, 2) and first.residual is None
        dw = qlayers[first.name]
        engines.update(z_pad=dw.z_in, z_dw=dw.z_out)
    if last.kind == "pw":
        assert last.stride == 1
        pw = qlayers[last.name]
        engines.update(z_pw=pw.z_out, z_skip=pw.z_skip, skip_shift=pw.skip_shift, skip_mult=pw.skip_mult)
    if last.residual:
        engines.update(residual=1, skip=at["tensor", last.residual])
    return Job(hin=first.hin, win=first.win, cin=first.cin, cout=last.cout, mode=MODES[kinds],
               stride=first.stride, input=at["tensor", first.src], output=at["tensor", last.name],
               weights=at["weights", first.name], params=at["params", first.name], **engines)


def find_block(layers: list[Layer], cin: int, expanded: int, cout: int) -> int:
    """Index of the expand layer of the first block with these channel counts."""
    pairs = [(l.cin, l.cout) for l in layers]
    return next(i for i in range(len(pairs))
                if pairs[i:i + 3] == [(cin, expanded), (expanded, expanded), (expanded, cout)])


def build(net: Network, qlayers: dict[str, QLayer], tensors: dict[str, np.ndarray], start: int,
          plans: dict[str, list[Group]]) -> tuple[dict[str, tuple[int, np.ndarray]], dict[str, list[Job]]]:
    """Everything the testbench needs to run the block layers[start:start + 3].

    Returns the DRAM images as {file name: (address, words)} and, for each plan, the jobs of its
    groups in this block. `tensors` holds the int8 output of every layer, by layer name. The images
    are placed one after another from address 0: the weights of the three layers, their parameters,
    the block's input and the expected output of each layer.
    """
    block = net.layers[start:start + 3]
    assert [l.kind for l in block] == ["pw", "dw", "pw"]
    names = [l.name for l in block]
    parts = {("weights", n): weight_words(qlayers[n]) for n in names}
    parts |= {("params", n): param_words(qlayers[n]) for n in names}
    parts |= {("tensor", n): activation_words(tensors[n]) for n in [block[0].src] + names}
    at = dict(zip(parts, np.cumsum([0] + [len(p) for p in parts.values()]).tolist()))

    images = {what: (at[what, names[0]], np.concatenate([parts[what, n] for n in names]))
              for what in ("weights", "params")}
    for file, n in zip(("input", "expand", "depthwise", "project"), [block[0].src] + names):
        images[file] = (at["tensor", n], parts["tensor", n])
    jobs = {}
    for label, plan in plans.items():
        groups = [g for g in plan if start <= g.start and g.stop <= start + 3]
        assert sum(g.stop - g.start for g in groups) == 3, "a group crosses the block boundary"
        jobs[label] = [make_job(net, qlayers, g.start, g.stop, at) for g in groups]
    return images, jobs


def write_hex(path: Path, words, address: int | None = None) -> None:
    """Write 64-bit words for $readmemh, one per line, after their DRAM address if they have one."""
    lines = [f"@{address:x}"] if address is not None else []
    path.write_text("\n".join(lines + [f"{int(w):016x}" for w in words]) + "\n")


def write(folder: Path, images: dict[str, tuple[int, np.ndarray]], jobs: dict[str, list[Job]]) -> None:
    """Write the result of `build` into `folder`: one file per image and one per job list."""
    folder.mkdir(parents=True, exist_ok=True)
    for file, (address, words) in images.items():
        write_hex(folder / f"{file}.hex", words, address)
    for label, job_list in jobs.items():
        write_hex(folder / f"jobs_{label}.hex", [word for job in job_list for word in pack(job)])


if __name__ == "__main__":
    from sw.quantize import imagenette, mobilenet_v2, run_float, run_int, to_int8

    build_dir = Path(__file__).resolve().parents[1] / "build"
    _, layers, params, qparams, qlayers = mobilenet_v2()
    stem = layers[0].name
    image = imagenette("val", 1)  # the first validation image
    tensors = run_int(layers, qlayers, to_int8(run_float(layers[:1], params, image)[stem][0], *qparams[stem]))

    net = Network(layers)
    budget = 140 * BRAM36_BYTES  # all the block RAM of the XC7Z020
    plans = {"unfused": net.select(budget, "unfused"), "fused": net.select(budget, "rtl")}
    by_name = dict(zip((l.name for l in layers[1:]), qlayers))
    for label, channels in BLOCKS.items():
        start = find_block(layers, *channels)
        images, jobs = build(net, by_name, tensors, start, plans)
        write(build_dir / label, images, jobs)
        l = layers[start]
        print(f"{label}: {l.hin}x{l.win}, {'->'.join(map(str, channels))}, written to {build_dir / label}")
        print("  bytes: " + ", ".join(f"{file} {8 * len(words):,}" for file, (_, words) in images.items()))
        print("  jobs:  " + ", ".join(f"{len(job_list)} {name}" for name, job_list in jobs.items()))
