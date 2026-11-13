"""FP32 axial AV gate kernels for normal and physically transposed outputs."""

import triton
import triton.language as tl

from .rf2_sigmoid_mul import sigmoid_mul


@triton.jit
def _flat_gate(gate, weighted, length: tl.constexpr, row_stride: tl.constexpr,
               column_stride: tl.constexpr, head_stride: tl.constexpr,
               direct: tl.constexpr, block: tl.constexpr):
    idx = tl.program_id(0) * block + tl.arange(0, block)
    valid = idx < length * length * 128
    if direct:
        source = idx
    else:
        cell = idx // 128
        row, column = cell // length, cell % length
        head, d = (idx // 32) % 4, idx % 32
        source = row * row_stride + column * column_stride + head * head_stride + d
    a = tl.load(gate + idx, valid, other=0.0)
    b = tl.load(weighted + source, valid, other=0.0)
    tl.store(gate + idx, sigmoid_mul(a, b), valid)


@triton.jit
def _tiled_reverse_gate(gate, weighted, length: tl.constexpr, inner_axis: tl.constexpr,
                        block_positions: tl.constexpr, block_channels: tl.constexpr):
    inner = tl.program_id(0) * block_positions + tl.arange(0, block_positions)
    channel = tl.program_id(1) * block_channels + tl.arange(0, block_channels)
    outer = tl.program_id(2)
    if inner_axis == 1:
        cell = outer * length + inner
    else:
        cell = inner * length + outer
    target = cell[None, :] * 128 + channel[:, None]
    # The physical AV BMM output is [H, outer*D + d, inner].
    head, d = channel // 32, channel % 32
    source = ((head[:, None] * length + outer) * 32 + d[:, None]) * length + inner[None, :]
    valid = (inner[None, :] < length) & (channel[:, None] < 128)
    a = tl.load(gate + target, valid, other=0.0)
    b = tl.load(weighted + source, valid, other=0.0)
    tl.store(gate + target, sigmoid_mul(a, b), valid)


def launch(gate, weighted, layout, block):
    length = gate.shape[1]
    warps = 8 if block == 2048 else 4
    if layout.startswith("reverse"):
        inner_axis = 1 if layout == "reverse-rf2" else 0
        positions = 32
        channels = block // positions
        _tiled_reverse_gate[(triton.cdiv(length, positions), triton.cdiv(128, channels), length)](
            gate, weighted, length, inner_axis, positions, channels,
            num_warps=warps, enable_fp_fusion=False,
        )
    else:
        row, column, head, _ = weighted.stride()[1:]
        _flat_gate[(triton.cdiv(gate.numel(), block),)](
            gate, weighted, length, row, column, head, layout == "direct", block,
            num_warps=warps, enable_fp_fusion=False,
        )
