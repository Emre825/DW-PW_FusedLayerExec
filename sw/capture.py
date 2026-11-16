"""Model capture: trace a CNN with torch.fx, fold batch normalization, build the layer table."""
import copy
import operator
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
from torch.fx import GraphModule, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

INPUT_SHAPE = (1, 3, 224, 224)


@dataclass(frozen=True)
class Layer:
    """One convolution or max pooling, with the residual addition and the activation that follow it."""
    name: str                    # fx node name; also names the layer's output tensor
    kind: str                    # "conv" (standard), "dw" (depthwise), "pw" (pointwise) or "pool"
    cin: int
    cout: int
    kernel: int
    stride: int
    hin: int
    win: int
    hout: int
    wout: int
    act: str | None = None       # "relu6" or "relu", or None for a linear layer
    residual: str | None = None  # layer whose output is added to this one's, or None
    src: str | None = None       # layer whose output this one reads, or None for the network input


def trace(model: nn.Module) -> GraphModule:
    """Trace `model` in eval mode and record every node's output shape."""
    gm = symbolic_trace(model.eval())
    ShapeProp(gm).propagate(torch.zeros(INPUT_SHAPE))
    return gm


def fold_bn(gm: GraphModule) -> GraphModule:
    """Return a copy of `gm` with every BatchNorm2d folded into the Conv2d that feeds it."""
    gm = copy.deepcopy(gm)
    modules = dict(gm.named_modules())
    for bn_node in list(gm.graph.nodes):
        if bn_node.op != "call_module" or not isinstance(modules[bn_node.target], nn.BatchNorm2d):
            continue
        conv_node = bn_node.args[0]
        conv, bn = modules[conv_node.target], modules[bn_node.target]
        assert isinstance(conv, nn.Conv2d) and len(conv_node.users) == 1
        # W' = W * gamma / sigma,   b' = (b - mu) * gamma / sigma + beta
        with torch.no_grad():
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
            conv.weight.mul_(scale.reshape(-1, 1, 1, 1))
            conv.bias = nn.Parameter((bias - bn.running_mean) * scale + bn.bias)
        bn_node.replace_all_uses_with(conv_node)
        gm.graph.erase_node(bn_node)
    gm.delete_all_unused_submodules()
    gm.recompile()
    return gm


def layer_table(gm: GraphModule) -> list[Layer]:
    """List the convolutions and max poolings of a BN-folded graph in execution order."""
    modules = dict(gm.named_modules())
    layers: list[Layer] = []
    owner = {}  # fx node -> name of the layer whose output tensor it carries
    for node in gm.graph.nodes:
        module = modules.get(node.target) if node.op == "call_module" else None
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d)):
            src = node.args[0]
            _, cin, hin, win = src.meta["tensor_meta"].shape
            _, cout, hout, wout = node.meta["tensor_meta"].shape
            kernel, stride = (v if isinstance(v, int) else v[0] for v in (module.kernel_size, module.stride))
            if isinstance(module, nn.MaxPool2d):
                kind = "pool"
            elif module.groups == 1:
                kind = "pw" if kernel == 1 else "conv"
            else:
                assert module.groups == cin == cout
                kind = "dw"
            layers.append(Layer(node.name, kind, cin, cout, kernel, stride, hin, win, hout, wout,
                                src=owner.get(src)))
        elif isinstance(module, (nn.ReLU6, nn.ReLU)):
            assert owner[node.args[0]] == layers[-1].name
            layers[-1] = replace(layers[-1], act=type(module).__name__.lower())
        elif node.op == "call_function" and node.target is operator.add:
            # One operand is the layer just built; the other is the skip connection.
            skip, = (arg for arg in node.args if owner[arg] != layers[-1].name)
            layers[-1] = replace(layers[-1], residual=owner[skip])
        else:
            continue
        owner[node] = layers[-1].name
    return layers


if __name__ == "__main__":
    import torchvision

    weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
    gm = trace(torchvision.models.mobilenet_v2(weights=weights))
    folded = fold_bn(gm)

    x = torch.randn(INPUT_SHAPE)
    with torch.no_grad():
        ref, out = gm(x), folded(x)
    print(f"BN folding: max |original - folded| = {(ref - out).abs().max():.3e}"
          f"  (max |output| = {ref.abs().max():.3f})\n")

    print(f"{'#':>2}  {'name':<22} {'kind':<4} {'cin':>4} {'cout':>4}  k  s  {'in':>7}  {'out':>7}"
          f"  {'act':<5}  residual")
    for i, l in enumerate(layer_table(folded)):
        print(f"{i:>2}  {l.name:<22} {l.kind:<4} {l.cin:>4} {l.cout:>4}  {l.kernel}  {l.stride}"
              f"  {f'{l.hin}x{l.win}':>7}  {f'{l.hout}x{l.wout}':>7}"
              f"  {l.act or '-':<5}  {l.residual or '-'}")
