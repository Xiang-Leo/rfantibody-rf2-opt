"""Fuse RF2 C-alpha radial basis expansion and relative position lookup."""

from functools import wraps

import triton
import triton.language as tl


from .rfd_residual_norm import TORCH_VERSION, verify


BINS = 64
BLOCK_ROWS = 8
MODES = ("fused", "audit", "rbf", "p10-fused", "p10-audit")


@triton.jit
def _ca_features(distance, residue_index, position, raw, positioned,
                 length: tl.constexpr, add_position: tl.constexpr,
                 cyclic: tl.constexpr, block_rows: tl.constexpr):
    pair = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    channel = tl.arange(0, 64)
    valid = pair < length * length
    d = tl.load(distance + pair, valid, other=0.0)
    centers = channel.to(tl.float32) * 0.5
    # The stock scalar division is by exactly 0.5 and scalar pow(2) is a
    # separately rounded square. Libdevice expf keeps Torch's FP32 exp path.
    exponent = tl.inline_asm_elementwise(
        "{ .reg .f32 delta, scaled, square; sub.rn.f32 delta, $1, $2;"
        " mul.rn.f32 scaled, delta, 0f40000000; mul.rn.f32 square, scaled, scaled;"
        " neg.f32 $0, square; }",
        "=f,f,f", (d[:, None], centers[None, :]), dtype=tl.float32, is_pure=True, pack=1,
    )
    radial = tl.math.exp(exponent)
    output_index = pair[:, None] * 64 + channel[None, :]
    tl.store(raw + output_index, radial, valid[:, None])
    if add_position:
        start = tl.load(residue_index + pair // length, valid, other=0)
        end = tl.load(residue_index + pair % length, valid, other=0)
        sep = end - start
        if cyclic:
            wrapped = (sep + length // 2) % length
            sep = tl.where(wrapped < 0, wrapped + length, wrapped) - length // 2
        # bucketize(right=False) with integer boundaries arange(-32, 32).
        bucket = (tl.minimum(tl.maximum(sep, -32), 32) + 32).to(tl.int32)
        offset = bucket[:, None] * 64 + channel[None, :]
        lookup = tl.load(position + offset, valid[:, None], other=0.0)
        adjusted = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;", "=f,f,f", (radial, lookup),
            dtype=tl.float32, is_pure=True, pack=1,
        )
        tl.store(positioned + output_index, adjusted, valid[:, None])


def check_versions(torch):
    if (torch.__version__, triton.__version__) != (TORCH_VERSION, "2.3.1"):
        raise RuntimeError("C-alpha fusion requires Torch 2.3.1+cu118 and Triton 2.3.1")


def eligible(torch, distance, position, idx, length, nc_cycle):
    from rfantibody.rf2.network.Track_module import SeqSep

    if (type(distance) is not torch.Tensor or distance.shape != (1, length, length)
            or distance.dtype != torch.float32 or not distance.is_cuda or not distance.is_contiguous()
            or not 1 < length or length * length * BINS >= 2**31
            or type(idx) is not torch.Tensor or idx.shape != (1, length)
            or idx.device != distance.device or idx.dtype not in (torch.int32, torch.int64)
            or not idx.is_contiguous() or type(nc_cycle) is not bool
            or type(position) is not SeqSep or (position.minpos, position.maxpos) != (-32, 32)
            or type(position.emb) is not torch.nn.Embedding or position.emb.max_norm is not None
            or position.emb.weight.shape != (BINS + 1, BINS)
            or torch.is_grad_enabled() or torch.is_autocast_enabled()
            or torch.is_autocast_cpu_enabled() or torch.backends.cuda.matmul.allow_tf32
            or torch.get_default_dtype() != torch.float32):
        return False
    weight = position.emb.weight
    return (
        weight.device == distance.device and weight.dtype == distance.dtype and weight.is_contiguous()
        and not any(module._forward_pre_hooks or module._forward_hooks for module in (position, position.emb))
        and torch.cuda.get_device_capability(distance.device) == (8, 9)
    )


def features(torch, distance, position, idx, length, nc_cycle, *, add_position=True):
    raw = distance.new_empty((1, length, length, BINS))
    positioned = distance.new_empty(raw.shape) if add_position else raw
    with torch.cuda.device(distance.device):
        _ca_features[(triton.cdiv(length * length, BLOCK_ROWS),)](
            distance, idx, position.emb.weight, raw, positioned,
            length, add_position, nc_cycle, BLOCK_ROWS, num_warps=4, enable_fp_fusion=False,
        )
    if not add_position:
        positioned = raw + position(idx, idx, 1, length, nc_cycle)
    return raw, positioned


def install(torch, mode, base):
    if mode not in MODES:
        raise ValueError(f"Choose one of {', '.join(MODES)}")
    check_versions(torch)
    from rfantibody.rf2.network import Track_module

    if not hasattr(Track_module, "ca_rbf_features"):
        raise RuntimeError("Enable the RF2 C-alpha reuse path before installing the fused kernel")
    stock = Track_module.ca_rbf_features
    add_position = mode != "rbf"
    audit = mode.endswith("audit")
    counts = {"base": base, "fused": 0, "rbf_only": 0, "fallback": 0, "audited": 0,
              "position_elements": 0, "logical_intermediate_bytes_eliminated": 0}

    @wraps(stock)
    def forward(distance, position, idx, length, nc_cycle):
        if not eligible(torch, distance, position, idx, length, nc_cycle):
            counts["fallback"] += 1
            return stock(distance, position, idx, length, nc_cycle)
        result = features(torch, distance, position, idx, length, nc_cycle, add_position=add_position)
        if audit:
            reference = stock(distance, position, idx, length, nc_cycle)
            for name, expected, actual in zip(("raw radial", "positioned radial"), reference, result):
                verify(torch, f"RF2 {name}", expected, actual)
            counts["audited"] += 1
        # Subtract, scale, square and negate each produced a dense temporary.
        # The position path also removes the dense lookup and raw reload.
        counts["logical_intermediate_bytes_eliminated"] += result[0].nbytes * (11 if add_position else 8)
        counts["fused" if add_position else "rbf_only"] += 1
        if add_position:
            counts["position_elements"] += result[0].numel()
        return result

    Track_module.REUSE_CA_RBF = True
    Track_module.ca_rbf_features = forward
    return counts
