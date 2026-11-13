"""Stream RF2's last radial Linear into the validated flat contraction."""

from functools import wraps

import triton

from .rf2_radial_fused import _flat, supports_k64


TILE_SIZES = (8192, 16384, 32768, 65536)


def stream(torch, hidden, weight, node, basis, tile_edges, *, inspect_projection=None):
    """Project each edge tile with stock FP32 Linear and write its output slice."""
    edges = hidden.shape[0]
    if (
        tile_edges not in TILE_SIZES or not 0 < edges <= 1_262_252
        or hidden.shape != (edges, 32) or weight.shape != (1536, 32)
        or node.shape != (edges, 32, 1) or basis.shape != (edges, 1, 2, 4)
        or not hidden.is_cuda or hidden.dtype != torch.float32
        or any(tensor.dtype != hidden.dtype or tensor.device != hidden.device
               or not tensor.is_contiguous() for tensor in (hidden, weight, node, basis))
        or torch.is_grad_enabled() or torch.is_autocast_enabled()
        or torch.is_autocast_cpu_enabled() or torch.backends.cuda.matmul.allow_tf32
    ):
        raise ValueError("Expected contiguous CUDA FP32 RF2 hidden, weight and angular tensors")

    output = hidden.new_empty((edges, 24, 4))
    linear = torch.nn.functional.linear
    for start in range(0, edges, tile_edges):
        stop = min(start + tile_edges, edges)
        tile = slice(start, stop)
        radial = linear(hidden[tile], weight)
        if inspect_projection is not None:
            inspect_projection(tile, radial)
        count = (stop - start) * 96
        # A slice of the contiguous output uses the same layout as the
        # standalone flat kernel. No per-tile output copy is needed.
        _flat[(triton.cdiv(count, 128),)](
            radial, node[tile], basis[tile], output[tile], count, 128,
            num_warps=4, enable_fp_fusion=False,
        )
        del radial
    return output


def install(torch, tile_edges, prefix):
    from .rf2_se3_radial_exact import EDGE_COUNTS
    from se3_transformer.model.layers.convolution import ConvSE3FuseLevel, VersatileConvSE3

    if tile_edges not in TILE_SIZES:
        raise ValueError(f"Unsupported RF2 radial tile size: {tile_edges}")
    fallback = VersatileConvSE3.forward
    counts = {"k64_streamed": 0}

    @wraps(fallback)
    def forward(module, features, invariant, basis):
        if not supports_k64(torch, module, features, invariant, basis, EDGE_COUNTS, ConvSE3FuseLevel.FULL):
            return fallback(module, features, invariant, basis)
        profile = module.radial_func
        final = profile.net[-1]
        if (type(final) is not torch.nn.Linear or final.bias is not None
                or final.weight.shape != (1536, 32) or final.weight.dtype != features.dtype
                or final.weight.device != features.device or not final.weight.is_contiguous()
                or any(layer._forward_hooks or layer._forward_pre_hooks for layer in (profile, final))):
            return fallback(module, features, invariant, basis)
        hidden = prefix(profile, invariant)
        if hidden is None:
            return fallback(module, features, invariant, basis)
        result = stream(torch, hidden, final.weight, features, basis, tile_edges)
        counts["k64_streamed"] += 1
        return result

    VersatileConvSE3.forward = forward
    return counts
