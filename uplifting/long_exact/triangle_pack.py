"""Prepare RF2 triangle BMM operands while retaining stock projection arithmetic."""

from functools import wraps

import triton
import triton.language as tl

from .weights import _packed_weights


SEPARATE_MODES = ("separate-dense", "separate-view")
EXPERIMENTAL_MODES = ("packed-dense", "packed-view")
MODES = (*SEPARATE_MODES, *EXPERIMENTAL_MODES)
LEGACY_MODES = {"dense": "packed-dense", "view": "packed-view"}
WIDTHS = (32, 64, 128)
EDGE_TILE, CHANNEL_TILE = 64, 32


@triton.jit
def _mul(left, right):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", (left, right),
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _pack_one(projection, gates, packed, reciprocal, length: tl.constexpr, hidden: tl.constexpr,
              pitch: tl.constexpr, right: tl.constexpr, swap: tl.constexpr,
              edge_tile: tl.constexpr, channel_tile: tl.constexpr):
    edges: tl.constexpr = length * length
    edge = tl.program_id(0) * edge_tile + tl.arange(0, edge_tile)
    channel = tl.program_id(1) * channel_tile + tl.arange(0, channel_tile)
    valid = (channel[:, None] < hidden) & (edge[None, :] < edges)
    if swap:
        source_edge = (edge % length) * length + edge // length
    else:
        source_edge = edge
    source = source_edge[None, :] * pitch + channel[:, None]
    value = tl.load(projection + source, valid, other=0.0)
    gate = tl.load(gates + source, valid, other=0.0)
    product = _mul(gate, value)
    if right:
        # Stock CUDA scalar division multiplies by a rounded FP32 reciprocal.
        product = _mul(product, reciprocal)
    dest = channel[:, None] * edges + edge[None, :]
    tl.store(packed + dest, product, valid)


@triton.jit
def _pack_both(left, left_gates, right, right_gates, packed, reciprocal,
               length: tl.constexpr, hidden: tl.constexpr, pitch: tl.constexpr,
               edge_tile: tl.constexpr, channel_tile: tl.constexpr):
    edges: tl.constexpr = length * length
    edge = tl.program_id(0) * edge_tile + tl.arange(0, edge_tile)
    channel = tl.program_id(1) * channel_tile + tl.arange(0, channel_tile)
    valid = (channel[:, None] < hidden) & (edge[None, :] < edges)
    source = edge[None, :] * pitch + channel[:, None]
    left_value = tl.load(left + source, valid, other=0.0)
    left_gate = tl.load(left_gates + source, valid, other=0.0)
    right_value = tl.load(right + source, valid, other=0.0)
    right_gate = tl.load(right_gates + source, valid, other=0.0)
    dest = channel[:, None] * edges + edge[None, :]
    tl.store(packed + dest, _mul(left_gate, left_value), valid)
    tl.store(packed + dest + hidden * edges, _mul(_mul(right_gate, right_value), reciprocal), valid)


def grid_for(length, hidden):
    return triton.cdiv(length**2, EDGE_TILE), triton.cdiv(hidden, CHANNEL_TILE)


def transposed_view(left, right, outgoing):
    if outgoing:
        return left, right.transpose(1, 2)
    return left.transpose(1, 2), right


def separate_products(torch, module, pair, mode):
    """Keep the original four Linear shapes and discard their outputs after packing."""
    length, hidden = pair.shape[1], module.d_hidden
    packed = pair.new_empty((2, hidden, length, length))
    left, right = packed[0], packed[1]
    reciprocal = 1.0 / length
    grid = grid_for(length, hidden)
    outgoing = module.outgoing
    with torch.cuda.device(pair.device):
        left_value = module.left_proj(pair)
        left_gate = module.left_gate(pair).sigmoid_()
        if mode == "separate-dense":
            _pack_one[grid](left_value, left_gate, left, reciprocal, length, hidden, hidden,
                            False, not outgoing, EDGE_TILE, CHANNEL_TILE,
                            num_warps=4, enable_fp_fusion=False)
            del left_value, left_gate
            right_value = module.right_proj(pair)
            right_gate = module.right_gate(pair).sigmoid_()
            _pack_one[grid](right_value, right_gate, right, reciprocal, length, hidden, hidden,
                            True, outgoing, EDGE_TILE, CHANNEL_TILE,
                            num_warps=4, enable_fp_fusion=False)
            return left, right
        right_value = module.right_proj(pair)
        right_gate = module.right_gate(pair).sigmoid_()
        _pack_both[grid](left_value, left_gate, right_value, right_gate, packed, reciprocal,
                         length, hidden, hidden, EDGE_TILE, CHANNEL_TILE,
                         num_warps=4, enable_fp_fusion=False)
        return transposed_view(left, right, outgoing)


def pack_products(torch, projected, hidden, outgoing, mode):
    """Experimental packing from a combined Linear output."""
    mode = LEGACY_MODES.get(mode, mode)
    length = projected.shape[1]
    if (mode not in EXPERIMENTAL_MODES or hidden not in WIDTHS
            or projected.shape != (1, length, length, 4 * hidden)
            or not projected.is_cuda or projected.dtype != torch.float32 or not projected.is_contiguous()
            or projected.numel() >= 2**31 or length <= 1 or torch.is_grad_enabled()
            or torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()
            or torch.backends.cuda.matmul.allow_tf32):
        raise ValueError("Expected a contiguous FP32 singleton RF2 triangle projection")
    left_value, left_gate, right_value, right_gate = projected.split(hidden, dim=-1)
    left_gate.sigmoid_()
    right_gate.sigmoid_()
    packed = projected.new_empty((2, hidden, length, length))
    reciprocal = 1.0 / length
    grid = grid_for(length, hidden)
    left, right = packed[0], packed[1]
    with torch.cuda.device(projected.device):
        if mode == "packed-dense":
            _pack_one[grid](left_value, left_gate, left, reciprocal, length, hidden, 4 * hidden,
                            False, not outgoing, EDGE_TILE, CHANNEL_TILE,
                            num_warps=4, enable_fp_fusion=False)
            _pack_one[grid](right_value, right_gate, right, reciprocal, length, hidden, 4 * hidden,
                            True, outgoing, EDGE_TILE, CHANNEL_TILE,
                            num_warps=4, enable_fp_fusion=False)
            return left, right
        _pack_both[grid](left_value, left_gate, right_value, right_gate, packed, reciprocal,
                         length, hidden, 4 * hidden, EDGE_TILE, CHANNEL_TILE,
                         num_warps=4, enable_fp_fusion=False)
        return transposed_view(left, right, outgoing)


def combined_projection(torch, module, normalized):
    """Experimental P6 projection layout, kept for byte diagnosis."""
    weights, biases = _packed_weights(
        module, "_uplifting_triangle",
        (module.left_proj, module.left_gate, module.right_proj, module.right_gate),
    )
    return torch.nn.functional.linear(normalized, weights, biases)


def final_gate(logits, values):
    return logits.sigmoid_().mul_(values)


def forward(torch, module, pair, mode):
    mode = LEGACY_MODES.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f"Unknown RF2 triangle packing mode: {mode}")
    pair = module.norm(pair)
    if mode in SEPARATE_MODES:
        left, right = separate_products(torch, module, pair, mode)
    else:
        projected = combined_projection(torch, module, pair)
        left, right = pack_products(torch, projected, module.d_hidden, module.outgoing, mode)
        del projected
    product = torch.bmm(left, right)
    del left, right
    result = product.permute(1, 2, 0).unsqueeze(0)
    result = module.out_proj(module.norm_out(result))
    return final_gate(module.gate(pair), result)


def eligible(torch, module, pair, stride):
    if (pair.ndim != 4 or pair.shape[0] != 1 or pair.shape[1] != pair.shape[2]
            or pair.shape[1] <= 1 or module.d_hidden not in WIDTHS
            or not pair.is_cuda or pair.dtype != torch.float32):
        return False
    length = pair.shape[1]
    return (
        4 * module.d_hidden * length**2 < 2**31
        and not torch.is_grad_enabled() and not torch.is_autocast_enabled()
        and not torch.is_autocast_cpu_enabled() and not torch.backends.cuda.matmul.allow_tf32
        and (module.training or stride <= 0 or stride >= length)
        and all(type(layer) is torch.nn.Linear and not layer._forward_hooks and not layer._forward_pre_hooks
                and layer.in_features == pair.shape[-1] and layer.out_features == module.d_hidden
                and layer.weight.device == pair.device and layer.weight.dtype == pair.dtype
                and layer.bias is not None and layer.bias.device == pair.device and layer.bias.dtype == pair.dtype
                for layer in (module.left_proj, module.left_gate, module.right_proj, module.right_gate))
    )


def install(torch, mode):
    from rfantibody.rf2.network.Attention_module import TriangleMultiplication

    mode = LEGACY_MODES.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f"Unknown RF2 triangle packing mode: {mode}")
    original = TriangleMultiplication.forward
    counts = {"packed": 0, "separate_projection": 0, "experimental_projection": 0,
              "outgoing": 0, "incoming": 0, "fallback": 0}

    @wraps(original)
    def invoke(module, pair, stride):
        if not eligible(torch, module, pair, stride):
            counts["fallback"] += 1
            return original(module, pair, stride)
        result = forward(torch, module, pair, mode)
        counts["packed"] += 1
        counts["separate_projection" if mode in SEPARATE_MODES else "experimental_projection"] += 1
        counts["outgoing" if module.outgoing else "incoming"] += 1
        return result

    TriangleMultiplication.forward = invoke
    return counts
