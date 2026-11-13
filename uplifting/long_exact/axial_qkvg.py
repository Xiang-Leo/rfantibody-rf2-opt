"""FP32 QKV and QKVG projections for singleton RFantibody axial attention."""

from functools import wraps
import struct

import torch
from torch.nn import functional as F
import triton
import triton.language as tl


from . import rfd_axial_qk as qk


VARIANTS = ("qkv", "qkvg")
TORCH_VERSION = "2.3.1+cu118"
HEADS = 4
HIDDEN = 32
WIDTH = HEADS * HIDDEN


@triton.jit
def _pack_query(source, dest, scale, length: tl.constexpr, channels: tl.constexpr,
                rows: tl.constexpr, cols: tl.constexpr):
    # P7 query indexing, with the row pitch of the merged projection.
    n = tl.program_id(0) * rows + tl.arange(0, rows)
    col = tl.program_id(1) * cols + tl.arange(0, cols)
    i, component = col // 128, col % 128
    valid = (i[:, None] < length) & (n[None, :] < length)
    src = (n[None, :] * length + i[:, None]) * channels + component[:, None]
    values = tl.load(source + src, valid, other=0) * scale
    head, d = component // 32, component % 32
    row = (head * length + i) * 32 + d
    dst = row[:, None] * length + n[None, :]
    tl.store(dest + dst, values, valid)


@triton.jit
def _pack_key(source, dest, scale, length: tl.constexpr, channels: tl.constexpr,
              rows: tl.constexpr, cols: tl.constexpr):
    nj = tl.program_id(0) * rows + tl.arange(0, rows)
    col = tl.program_id(1) * cols + tl.arange(0, cols)
    square = length * length
    valid = (col[:, None] < 128) & (nj[None, :] < square)
    src = nj[None, :] * channels + 128 + col[:, None]
    values = tl.load(source + src, valid, other=0) * scale
    dst = col[:, None] * square + nj[None, :]
    tl.store(dest + dst, values, valid)


def pack_qk(merged, scale, divisor):
    length, channels = merged.shape[1], merged.shape[-1]
    if (merged.shape != (1, length, length, channels) or channels not in (3 * WIDTH, 4 * WIDTH)
            or not merged.is_cuda or merged.dtype != torch.float32 or not merged.is_contiguous()
            or merged.numel() >= 2**31):
        raise ValueError("Q/K packing requires a dense singleton FP32 CUDA QKV or QKVG projection")
    divisor = struct.unpack("f", struct.pack("f", divisor))[0]
    query = merged.new_empty((HEADS, length, HIDDEN * length))
    key = merged.new_empty((HEADS, HIDDEN * length, length))
    with torch.cuda.device(merged.device):
        _pack_query[(triton.cdiv(length, 32), triton.cdiv(length * WIDTH, 64))](
            merged, query, scale, length, channels, 32, 64,
        )
        _pack_key[(triton.cdiv(length * length, 64), triton.cdiv(WIDTH, 32))](
            merged, key, 1.0 / divisor, length, channels, 64, 32,
        )
    return query, key


def projection(module, variant):
    """Cache the merged read-only parameters by tensor identity and version."""
    layers = (module.to_q, module.to_k, module.to_v, module.to_g)
    layers = layers if variant == "qkvg" else layers[:3]
    parameters = tuple(layer.weight for layer in layers)
    if variant == "qkvg":
        parameters += (module.to_g.bias,)
    cacheable = not any(t.is_inference() for t in parameters)
    key = None
    if cacheable:
        stream = torch.cuda.current_stream(parameters[0].device).cuda_stream
        versions = tuple((t._version, t.device, t.dtype, t.data_ptr()) for t in parameters)
        key = (variant, stream, versions)
    cached = getattr(module, "_uplifting_qkvg_projection", None)
    if (cached is not None and key is not None and cached[0] == key
            and all(current is previous for current, previous in zip(parameters, cached[1]))):
        return cached[2:]
    weight = torch.cat(tuple(layer.weight for layer in layers), dim=0)
    bias = None
    if variant == "qkvg":
        # Same GEMM bias representation as the P8 real-checkpoint screen.
        bias = weight.new_zeros(4 * WIDTH)
        bias[3 * WIDTH:].copy_(module.to_g.bias)
    if cacheable:
        module._uplifting_qkvg_projection = (key, parameters, weight, bias)
    return weight, bias


def eligible(module, pair, bias, stride, model, variant):
    if not qk.eligible(module, pair, bias, stride, model):
        return False
    channels = (4 if variant == "qkvg" else 3) * WIDTH
    return (
        torch.__version__ == TORCH_VERSION and (model == "rf2" or not module.training)
        and type(pair) is torch.Tensor and type(bias) is torch.Tensor
        and bias.ndim == 4 and bias.shape[:3] == pair.shape[:3]
        and not torch.is_autocast_cpu_enabled()
        and torch.get_float32_matmul_precision() == "highest"
        and torch.cuda.get_device_capability(pair.device) == (8, 9)
        and pair.shape[1] ** 2 * channels < 2**31
        and type(module.norm_pair) is torch.nn.LayerNorm
        and not module.norm_pair._forward_hooks and not module.norm_pair._forward_pre_hooks
        and all(type(layer) is torch.nn.Linear and not layer._forward_hooks
                and not layer._forward_pre_hooks and layer.weight.shape == (WIDTH, WIDTH)
                and layer.weight.device == pair.device and layer.weight.dtype == pair.dtype
                and layer.weight.is_contiguous()
                for layer in (module.to_q, module.to_k, module.to_v, module.to_g))
        and module.to_v.bias is None and module.to_g.bias is not None
        and module.to_g.bias.shape == (WIDTH,) and module.to_g.bias.dtype == pair.dtype
        and module.to_g.bias.device == pair.device
    )


def attention(module, pair, bias, model, variant):
    length = pair.shape[1]
    shape = (1, length, length, HEADS, HIDDEN)
    if module.is_row:
        pair, bias = pair.transpose(1, 2), bias.transpose(1, 2)
    pair = module.norm_pair(pair)
    bias = module.to_b(module.norm_bias(bias))
    weight, projection_bias = projection(module, variant)
    merged = F.linear(pair, weight, projection_bias)
    if variant == "qkvg":
        del pair
    query, key = pack_qk(merged, module.scaling, qk.key_divisor(model, length))
    logits = torch.bmm(query, key).permute(1, 2, 0).unsqueeze(0)
    del query, key
    weights = F.softmax(logits + bias, dim=-2)
    del logits, bias

    # Splitting changes the cell pitch. Keep the same einsum equation, so
    # ATen prepares the original AV batch, shape, and reduction dimension.
    value = merged[..., 2 * WIDTH:3 * WIDTH].view(shape)
    weighted = torch.einsum(qk.RFD_AV if model == "rfd" else qk.RF2_AV, weights, value)
    del weights, value
    if variant == "qkvg":
        gate = merged[..., 3 * WIDTH:].view(shape)
        gate.sigmoid_()
        product = gate.new_empty(shape)
        torch.mul(gate, weighted, out=product)
        del gate, merged
    else:
        del merged
        product = module.to_g(pair).view(shape)
        product.sigmoid_()
        torch.mul(product, weighted, out=product)
    del weighted
    output = module.to_out(product.view(1, length, length, WIDTH))
    return output.transpose(1, 2) if module.is_row else output


def replace_qk(model, variant):
    """Replace the attention body under an already-installed P7 Q/K hook."""
    if model not in ("rfd", "rf2") or variant not in VARIANTS:
        raise ValueError(f"unknown axial model or projection variant {model}-{variant}")
    counts = {"fused_qkv": 0, "fused_qkvg": 0, "fallback": 0}
    sequential = qk.attention

    @wraps(sequential)
    def trial(module, pair, bias, mode):
        # The outer P7 hook checked the actual RF2 stride before this call.
        if mode != model or not eligible(module, pair, bias, 0, model, variant):
            counts["fallback"] += 1
            return sequential(module, pair, bias, mode)
        output = attention(module, pair, bias, model, variant)
        counts[f"fused_{variant}"] += 1
        return output

    qk.attention = trial
    return counts
