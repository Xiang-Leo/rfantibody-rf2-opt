"""Directly pack axial operands in either physical matrix orientation."""

import struct

import torch
import triton
import triton.language as tl


@triton.jit
def _position_major(source, dest, scale, length: tl.constexpr, channels: tl.constexpr,
                    offset: tl.constexpr, transpose_input: tl.constexpr, scale_values: tl.constexpr,
                    rows: tl.constexpr, cols: tl.constexpr):
    residue = tl.program_id(0) * rows + tl.arange(0, rows)
    col = tl.program_id(1) * cols + tl.arange(0, cols)
    position, component = col // 128, col % 128
    valid = (position[:, None] < length) & (residue[None, :] < length)
    if transpose_input:
        source_cell = position[:, None] * length + residue[None, :]
    else:
        source_cell = residue[None, :] * length + position[:, None]
    source_index = source_cell * channels + offset + component[:, None]
    values = tl.load(source + source_index, valid, other=0)
    if scale_values:
        values = values * scale
    head, d = component // 32, component % 32
    row = (head * length + position) * 32 + d
    target = row[:, None] * length + residue[None, :]
    tl.store(dest + target, values, valid)


@triton.jit
def _reduction_major(source, dest, scale, length: tl.constexpr, channels: tl.constexpr,
                     offset: tl.constexpr, rows: tl.constexpr, cols: tl.constexpr):
    cell = tl.program_id(0) * rows + tl.arange(0, rows)
    component = tl.program_id(1) * cols + tl.arange(0, cols)
    square = length * length
    valid = (component[:, None] < 128) & (cell[None, :] < square)
    source_index = cell[None, :] * channels + offset + component[:, None]
    values = tl.load(source + source_index, valid, other=0) * scale
    target = component[:, None] * square + cell[None, :]
    tl.store(dest + target, values, valid)


def pack_qk(merged, scale, divisor, layout):
    """Pack logical [H,i,D*n] and [H,D*n,j] using the P7 D-then-n order."""
    length, channels = merged.shape[1], merged.shape[-1]
    if (merged.shape != (1, length, length, channels) or channels not in (256, 384, 512)
            or not merged.is_contiguous() or not merged.is_cuda or merged.dtype != torch.float32
            or merged.numel() >= 2**31 or layout not in ("nn", "nt", "tn", "tt")):
        raise ValueError("expected a dense FP32 singleton QK/QKV projection")
    divisor = struct.unpack("f", struct.pack("f", divisor))[0]
    position_shape = (4, length, 32 * length)
    reduction_shape = (4, 32 * length, length)
    position_grid = (triton.cdiv(length, 32), triton.cdiv(length * 128, 64))
    reduction_grid = (triton.cdiv(length * length, 64), 4)
    q_t, k_t = layout[0] == "t", layout[1] == "t"
    query = merged.new_empty(reduction_shape if q_t else position_shape)
    key = merged.new_empty(position_shape if k_t else reduction_shape)
    with torch.cuda.device(merged.device):
        if q_t:
            _reduction_major[reduction_grid](merged, query, scale, length, channels, 0, 64, 32)
        else:
            _position_major[position_grid](merged, query, scale, length, channels, 0, False, True, 32, 64)
        if k_t:
            _position_major[position_grid](merged, key, 1.0 / divisor, length, channels, 128, False, True, 32, 64)
        else:
            _reduction_major[reduction_grid](merged, key, 1.0 / divisor, length, channels, 128, 64, 32)
    return (query.transpose(1, 2) if q_t else query, key.transpose(1, 2) if k_t else key)


def pack_transposed_value(value):
    """Materialize [H,n*D,j], then return the original logical [H,j,n*D]."""
    length, pitch = value.shape[1], value.stride(2)
    if (value.shape != (1, length, length, 4, 32) or pitch not in (128, 384, 512)
            or value.stride()[1:] != (length * pitch, pitch, 32, 1)
            or not value.is_cuda or value.dtype != torch.float32 or length**2 * pitch >= 2**31):
        raise ValueError("expected original dense or QKV-strided FP32 V")
    packed = value.new_empty((4, length * 32, length))
    with torch.cuda.device(value.device):
        _position_major[(triton.cdiv(length, 32), triton.cdiv(length * 128, 64))](
            value, packed, 1.0, length, pitch, 0, True, False, 32, 64,
        )
    return packed.transpose(1, 2)
