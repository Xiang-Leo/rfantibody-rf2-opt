"""Exact-first FP32 dataflow experiments for RF2's unstriped pair gate."""

from functools import wraps

import triton
import triton.language as tl


TILE_ROWS = (32768, 65536, 131072, 262144)
MODES = ("inplace", *(f"{scope}-{rows}" for scope in ("gate", "both") for rows in TILE_ROWS))


@triton.jit
def _outer_tile(left, right, output, first, values, length: tl.constexpr, block: tl.constexpr):
    position = tl.program_id(0) * block + tl.arange(0, block)
    edge = first + position // 256
    row, column = edge // length, edge % length
    left_channel, right_channel = position // 16 % 16, position % 16
    valid = position < values
    a = tl.load(left + row * 16 + left_channel, valid, other=0.0)
    b = tl.load(right + column * 16 + right_channel, valid, other=0.0)
    product = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", (a, b),
        dtype=tl.float32, is_pure=True, pack=1,
    )
    tl.store(output + position, product, valid)


def state_projections(module, state):
    # The input already went through the first normalization in PairStr2Pair.
    state = module.norm_state(state)
    return module.proj_left(state), module.proj_right(state)


def _stock_gate_product(torch, logits, features, output):
    logits.sigmoid_()
    return torch.mul(logits, features, out=output)


_gate_product = _stock_gate_product


def gate_features(torch, module, rbf_feat, state, mode, *, inspect=None):
    """Compute the stock gated feature with reduced intermediate lifetimes."""
    if mode not in MODES:
        raise ValueError(f"Unknown RF2 pair gate mode: {mode}")
    state = module.norm_state(state)
    gate_product = _gate_product if inspect is None else _stock_gate_product
    both = mode.startswith("both-")
    embedded = None if both else module.emb_rbf(rbf_feat)
    left, right = state_projections(module, state)
    length = left.shape[1]
    if mode == "inplace":
        outer = torch.einsum("bli,bmj->blmij", left, right).reshape(1, length, length, -1)
        logits = module.to_gate(outer)
        if inspect is not None:
            inspect(slice(0, length * length), outer.flatten(0, 2), logits.flatten(0, 2),
                    embedded.flatten(0, 2))
        del outer
        gate_product(torch, logits, embedded, embedded)
        return embedded

    tile_rows = int(mode.rsplit("-", 1)[1])
    output = (rbf_feat.new_empty((1, length, length, module.d_pair)) if both else embedded)
    output_rows = output.view(length * length, module.d_pair)
    rbf_rows = rbf_feat.view(length * length, -1) if both else None
    with torch.cuda.device(left.device):
        for start in range(0, length * length, tile_rows):
            stop = min(start + tile_rows, length * length)
            selection = slice(start, stop)
            outer = left.new_empty((stop - start, 256))
            block = 1024
            _outer_tile[(triton.cdiv(outer.numel(), block),)](
                left, right, outer, start, outer.numel(), length, block,
                num_warps=4, enable_fp_fusion=False,
            )
            logits = module.to_gate(outer)
            features = module.emb_rbf(rbf_rows[selection]) if both else output_rows[selection]
            if inspect is not None:
                inspect(selection, outer, logits, features)
            del outer
            gate_product(torch, logits, features, output_rows[selection])
            del logits, features
    return output


def eligible(torch, module, pair, rbf_feat, state, strides, crop):
    if (strides is not None or crop >= 2 or pair.ndim != 4 or pair.shape[0] != 1
            or pair.shape[1] != pair.shape[2] or pair.shape[1] <= 1
            or pair.shape[-1] != module.d_pair or not pair.is_cuda or pair.dtype != torch.float32):
        return False
    length = pair.shape[1]
    return (
        256 * length * length < 2**31
        and rbf_feat.shape == (1, length, length, module.emb_rbf.in_features)
        and state.shape == (1, length, module.proj_left.in_features)
        and module.to_gate.weight.shape == (module.d_pair, 256)
        and module.proj_left.out_features == module.proj_right.out_features == 16
        and all(tensor.dtype == pair.dtype and tensor.device == pair.device and tensor.is_contiguous()
                for tensor in (rbf_feat, state))
        and not torch.is_grad_enabled() and not torch.is_autocast_enabled()
        and not torch.is_autocast_cpu_enabled() and not torch.backends.cuda.matmul.allow_tf32
        and getattr(module, "_uplifting_gate", None) is None
        and all(type(layer) is torch.nn.Linear and not layer._forward_hooks and not layer._forward_pre_hooks
                and layer.weight.dtype == pair.dtype and layer.weight.device == pair.device
                and (layer.bias is None or (layer.bias.dtype == pair.dtype and layer.bias.device == pair.device))
                for layer in (module.to_gate, module.emb_rbf, module.proj_left, module.proj_right))
    )


def install(torch, mode):
    """Use the candidate in the pinned unstriped PairStr2Pair path only."""
    from rfantibody.rf2.network.Track_module import PairStr2Pair

    if mode not in MODES:
        raise ValueError(f"Unknown RF2 pair gate mode: {mode}")
    stock = PairStr2Pair.forward
    counts = {"gate": 0, "fallback": 0}

    @wraps(stock)
    def forward(module, pair, rbf_feat, state, strides, crop=-1):
        if not eligible(torch, module, pair, rbf_feat, state, strides, crop):
            counts["fallback"] += 1
            return stock(module, pair, rbf_feat, state, strides, crop)
        features = gate_features(torch, module, rbf_feat, state, mode)
        counts["gate"] += 1
        if module.training:
            pair = pair + module.drop_row(module.tri_mul_out(pair, -1))
            pair = pair + module.drop_row(module.tri_mul_in(pair, -1))
            pair = pair + module.drop_row(module.row_attn(pair, features, -1))
            pair = pair + module.drop_col(module.col_attn(pair, features, -1))
            pair = pair + module.ff(pair, -1)
        else:
            pair += module.drop_row(module.tri_mul_out(pair, -1))
            pair += module.drop_row(module.tri_mul_in(pair, -1))
            pair += module.drop_row(module.row_attn(pair, features, -1))
            pair += module.drop_col(module.col_attn(pair, features, -1))
            pair += module.ff(pair, -1)
        return pair

    PairStr2Pair.forward = forward
    return counts
