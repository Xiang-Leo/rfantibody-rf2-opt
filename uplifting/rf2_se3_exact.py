"""FP32 radial contraction for the measured dense RF2 convolution shapes."""

from functools import wraps
import json
from pathlib import Path
import random
import runpy
import sys

import torch
import triton
import triton.language as tl


EDGE_COUNTS = {157212, 1260006, 1262252}
BASIS_SHAPES = {(32, 1): ((1, 2, 4), 64), (2, 3): ((3, 4, 4), 8)}


@triton.jit
def _radial_bmm(left, right, output, total, inner: tl.constexpr, tile: tl.constexpr):
    index = tl.program_id(0) * tile + tl.arange(0, tile)
    edge = index // 96
    row = index // 4 % 24
    column = index % 4
    valid = index < total
    value = tl.full((tile,), 0.0, tl.float32)
    for k in tl.static_range(inner):
        a = tl.load(left + (edge * 24 + row) * inner + k, valid, other=0.0)
        b = tl.load(right + (edge * inner + k) * 4 + column, valid, other=0.0)
        value = tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", (a, b, value),
            dtype=tl.float32, is_pure=True, pack=1,
        )
    tl.store(output + index, value, valid)


def install():
    from se3_transformer.model.layers.convolution import ConvSE3FuseLevel, VersatileConvSE3

    stock = VersatileConvSE3.forward
    counts = {"k8": 0, "k64": 0}

    @wraps(stock)
    def forward(module, features, invariant, basis):
        edges = features.shape[0]
        dims = BASIS_SHAPES.get(tuple(features.shape[1:]))
        if (edges not in EDGE_COUNTS or dims is None or basis is None
                or not module.training or module.fuse_level != ConvSE3FuseLevel.FULL
                or module.channels_out != 24 or not features.is_cuda or features.dtype != torch.float32
                or not features.is_contiguous() or invariant.device != features.device
                or invariant.dtype != features.dtype or not basis.is_contiguous()
                or basis.dtype != features.dtype or basis.device != features.device
                or torch.is_grad_enabled() or torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()
                or torch.backends.cuda.matmul.allow_tf32):
            return stock(module, features, invariant, basis)
        basis_shape, inner = dims
        if basis.shape != (edges, *basis_shape) or module.channels_in * module.freq_sum != inner:
            return stock(module, features, invariant, basis)

        radial = module.radial_func(invariant).view(edges, 24, inner)
        basis_view = basis.view(edges, features.shape[-1], -1)
        vectors = (features @ basis_view).view(edges, inner, 4)
        output = features.new_empty((edges, 24, 4))
        tile = 128
        # Match the measured cuBLAS accumulation in increasing K order.
        _radial_bmm[(triton.cdiv(output.numel(), tile),)](
            radial, vectors, output, output.numel(), inner, tile,
            num_warps=4, enable_fp_fusion=False,
        )
        counts[f"k{inner}"] += 1
        return output

    VersatileConvSE3.forward = forward
    return counts


def main():
    from triton.runtime import cache

    if torch.__version__ != "2.3.1+cu118" or triton.__version__ != "2.3.1" or cache.random is not random:
        raise RuntimeError("RF2 SE(3) exact mode requires PyTorch 2.3.1+cu118 and Triton 2.3.1")
    entry = Path(sys.argv[1]).resolve(strict=True)
    counts = install()
    cache.random = random.SystemRandom()
    try:
        sys.argv = [str(entry), *sys.argv[2:]]
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        cache.random = random
    print(f"RF2 SE(3) radial contractions {json.dumps(counts, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
