"""Fuse the rounded RF2 K=1 angular product into the K=64 radial BMM."""

from functools import wraps

import triton
import triton.language as tl


MODES = ("flat", "tile-1", "tile-2", "tile-4")


@triton.jit
def _fma(left, right, accumulator):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", (left, right, accumulator),
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _mul(left, right):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", (left, right),
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _flat(radial, node, basis, output, total, tile: tl.constexpr):
    index = tl.program_id(0) * tile + tl.arange(0, tile)
    edge = index // 96
    row = index // 4 % 24
    column = index % 4
    valid = index < total
    radial_row = (edge * 24 + row) * 64
    basis0 = tl.load(basis + edge * 8 + column, valid, other=0.0)
    basis1 = tl.load(basis + edge * 8 + 4 + column, valid, other=0.0)
    value = tl.full((tile,), 0.0, tl.float32)
    for channel in tl.static_range(32):
        feature = tl.load(node + edge * 32 + channel, valid, other=0.0)
        a0 = tl.load(radial + radial_row + 2 * channel, valid, other=0.0)
        a1 = tl.load(radial + radial_row + 2 * channel + 1, valid, other=0.0)
        # The stock K=1 BMM rounds these products before the radial FMA.
        value = _fma(a0, _mul(feature, basis0), value)
        value = _fma(a1, _mul(feature, basis1), value)
    tl.store(output + index, value, valid)


@triton.jit
def _tiled(radial, node, basis, output, edges, block_edges: tl.constexpr):
    edge = tl.program_id(0) * block_edges + tl.arange(0, block_edges)
    row = tl.arange(0, 32)
    column = tl.arange(0, 4)
    valid_edge = edge < edges
    valid_row = valid_edge[:, None] & (row[None, :] < 24)
    radial_row = (edge[:, None] * 24 + row[None, :]) * 64
    basis_column = edge[:, None] * 8 + column[None, :]
    basis0 = tl.load(basis + basis_column, valid_edge[:, None], other=0.0)
    basis1 = tl.load(basis + basis_column + 4, valid_edge[:, None], other=0.0)
    value = tl.full((block_edges, 32, 4), 0.0, tl.float32)
    for channel in tl.static_range(32):
        feature = tl.load(node + edge * 32 + channel, valid_edge, other=0.0)
        a0 = tl.load(radial + radial_row + 2 * channel, valid_row, other=0.0)
        a1 = tl.load(radial + radial_row + 2 * channel + 1, valid_row, other=0.0)
        b0 = _mul(feature[:, None], basis0)
        b1 = _mul(feature[:, None], basis1)
        value = _fma(a0[:, :, None], b0[:, None, :], value)
        value = _fma(a1[:, :, None], b1[:, None, :], value)
    output_index = (edge[:, None, None] * 24 + row[None, :, None]) * 4 + column[None, None, :]
    tl.store(output + output_index, value, valid_row[:, :, None])


def contract(torch, radial, node, basis, mode):
    """Return the two stock contractions' result with the angular values in registers."""
    if mode not in MODES:
        raise ValueError(f"Unknown fused radial mode: {mode}")
    edges = radial.shape[0]
    if (
        not 0 < edges <= 1_262_252 or radial.shape != (edges, 24, 64)
        or node.shape != (edges, 32, 1) or basis.shape != (edges, 1, 2, 4)
        or not radial.is_cuda or radial.dtype != torch.float32
        or any(tensor.dtype != radial.dtype or tensor.device != radial.device
               or not tensor.is_contiguous() for tensor in (radial, node, basis))
        or torch.is_grad_enabled() or torch.is_autocast_enabled()
        or torch.backends.cuda.matmul.allow_tf32
    ):
        raise ValueError("Expected observed contiguous CUDA FP32 K=1 and K=64 operands")
    output = radial.new_empty((edges, 24, 4))
    if mode == "flat":
        tile = 128
        _flat[(triton.cdiv(output.numel(), tile),)](
            radial, node, basis, output, output.numel(), tile,
            num_warps=4, enable_fp_fusion=False,
        )
    else:
        block_edges = int(mode[-1])
        _tiled[(triton.cdiv(edges, block_edges),)](
            radial, node, basis, output, edges, block_edges,
            num_warps=4, enable_fp_fusion=False,
        )
    return output


def supports_k64(torch, module, features, invariant, basis, edge_counts, full_fuse):
    edges = features.shape[0]
    return (
        edges in edge_counts and features.shape == (edges, 32, 1)
        and basis is not None and basis.shape == (edges, 1, 2, 4)
        and module.training and module.fuse_level == full_fuse
        and module.channels_out == 24 and module.channels_in * module.freq_sum == 64
        and getattr(module, "inference_edge_chunk_size", 0) == 0
        and features.is_cuda and features.dtype == torch.float32
        and features.is_contiguous() and basis.is_contiguous()
        and invariant.device == features.device and invariant.dtype == features.dtype
        and basis.device == features.device and basis.dtype == features.dtype
        and not torch.is_grad_enabled() and not torch.is_autocast_enabled()
        and not torch.is_autocast_cpu_enabled() and not torch.backends.cuda.matmul.allow_tf32
    )


def install(torch, mode):
    """Install the fused K=64 path and keep the validated P6 K=8 path."""
    from se3_transformer.model.layers.convolution import ConvSE3FuseLevel, VersatileConvSE3
    from .rf2_se3_radial_exact import EDGE_COUNTS, install as install_p6

    if mode not in MODES:
        raise ValueError(f"Unknown fused radial mode: {mode}")
    counts = {"k64_fused": 0, "p6": install_p6(torch, "forward")}
    p6_forward = VersatileConvSE3.forward

    @wraps(p6_forward)
    def forward(module, features, invariant, basis):
        edges = features.shape[0]
        if not supports_k64(torch, module, features, invariant, basis, EDGE_COUNTS, ConvSE3FuseLevel.FULL):
            return p6_forward(module, features, invariant, basis)
        radial = module.radial_func(invariant).view(edges, 24, 64)
        result = contract(torch, radial, features, basis, mode)
        counts["k64_fused"] += 1
        return result

    VersatileConvSE3.forward = forward
    return counts
