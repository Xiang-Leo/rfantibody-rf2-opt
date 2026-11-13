"""FP32 arithmetic probes for the dense RF2 SE(3) radial BMM."""

import triton
import triton.language as tl


MODES = {
    "forward": (1, False, False, False, False),
    "forward-mul": (1, False, False, False, True),
    "reverse": (1, False, False, True, False),
    "reverse-mul": (1, False, False, True, True),
}
for streams in (2, 4, 8):
    for partition, interleaved in (("block", False), ("interleave", True)):
        MODES[f"{partition}{streams}"] = (streams, interleaved, False, False, False)
        if streams > 2:
            MODES[f"{partition}{streams}-tree"] = (streams, interleaved, True, False, False)


@triton.jit
def _add(left, right):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", "=f,f,f", (left, right),
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _partial(left, right, edge, row, column, valid,
             inner: tl.constexpr, streams: tl.constexpr, group: tl.constexpr,
             interleaved: tl.constexpr, reverse: tl.constexpr,
             first_mul: tl.constexpr, tile: tl.constexpr):
    value = tl.full((tile,), 0.0, tl.float32)
    for index in tl.static_range(inner // streams):
        if interleaved:
            k = index * streams + group
        else:
            k = group * (inner // streams) + index
        if reverse:
            k = inner - 1 - k
        a = tl.load(left + (edge * 24 + row) * inner + k, valid, other=0.0)
        b = tl.load(right + (edge * inner + k) * 4 + column, valid, other=0.0)
        if first_mul and index == 0:
            value = a * b
        else:
            value = tl.inline_asm_elementwise(
                "fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", (a, b, value),
                dtype=tl.float32, is_pure=True, pack=1,
            )
    return value


@triton.jit
def _contract(left, right, output, total,
              inner: tl.constexpr, streams: tl.constexpr, interleaved: tl.constexpr,
              tree: tl.constexpr, reverse: tl.constexpr, first_mul: tl.constexpr,
              tile: tl.constexpr):
    index = tl.program_id(0) * tile + tl.arange(0, tile)
    edge = index // 96
    row = index // 4 % 24
    column = index % 4
    valid = index < total
    value = _partial(left, right, edge, row, column, valid, inner, streams, 0,
                     interleaved, reverse, first_mul, tile)
    if tree:
        p1 = _partial(left, right, edge, row, column, valid, inner, streams, 1,
                      interleaved, reverse, first_mul, tile)
        p2 = _partial(left, right, edge, row, column, valid, inner, streams, 2,
                      interleaved, reverse, first_mul, tile)
        p3 = _partial(left, right, edge, row, column, valid, inner, streams, 3,
                      interleaved, reverse, first_mul, tile)
        value = _add(_add(value, p1), _add(p2, p3))
        if streams == 8:
            p4 = _partial(left, right, edge, row, column, valid, inner, streams, 4,
                          interleaved, reverse, first_mul, tile)
            p5 = _partial(left, right, edge, row, column, valid, inner, streams, 5,
                          interleaved, reverse, first_mul, tile)
            p6 = _partial(left, right, edge, row, column, valid, inner, streams, 6,
                          interleaved, reverse, first_mul, tile)
            p7 = _partial(left, right, edge, row, column, valid, inner, streams, 7,
                          interleaved, reverse, first_mul, tile)
            value = _add(value, _add(_add(p4, p5), _add(p6, p7)))
    else:
        for group in tl.static_range(1, streams):
            partial = _partial(left, right, edge, row, column, valid, inner, streams, group,
                               interleaved, reverse, first_mul, tile)
            value = _add(value, partial)
    tl.store(output + index, value, valid)


def contract(torch, left, right, mode):
    """Test one accumulation order for contiguous [E,24,K] by [E,K,4]."""
    if mode not in MODES:
        raise ValueError(f"Unknown RF2 radial accumulation mode: {mode}")
    if (
        left.ndim != 3 or right.ndim != 3
        or left.shape[1] != 24 or left.shape[2] not in (8, 64)
        or right.shape != (left.shape[0], left.shape[2], 4)
        or not 0 < left.shape[0] <= 1_262_252
        or not left.is_cuda or right.device != left.device
        or left.dtype != torch.float32 or right.dtype != left.dtype
        or not left.is_contiguous() or not right.is_contiguous()
        or torch.is_grad_enabled() or torch.is_autocast_enabled()
        or torch.backends.cuda.matmul.allow_tf32
    ):
        raise ValueError("Expected an observed contiguous RF2 CUDA FP32 radial contraction")
    output = left.new_empty((left.shape[0], 24, 4))
    tile = 128
    _contract[(triton.cdiv(output.numel(), tile),)](
        left, right, output, output.numel(), left.shape[2], *MODES[mode], tile,
        num_warps=4, enable_fp_fusion=False,
    )
    return output
