"""Change axial matrix storage without changing either FP32 reduction axis."""
from pathlib import Path


LAYOUTS = ("nn", "nt", "tn", "tt")
QK_MODES = (*LAYOUTS, *(f"{layout}-reverse" for layout in LAYOUTS))
AV_MODES = ("av-nt", "av-reverse", "av-nt-reverse")
HEADS, HIDDEN, WIDTH = 4, 32, 128
TORCH_VERSION = "2.3.1+cu118"


def dependencies():
    from . import rfd_axial_qk, axial_qkvg

    return rfd_axial_qk, axial_qkvg

def contract(torch, left, right, reverse=False):
    if reverse:
        return torch.bmm(right.transpose(1, 2), left.transpose(1, 2)).transpose(1, 2)
    return torch.bmm(left, right)


def projected(torch, qk, qkv, module, normalized, model):
    if model == "rf2":
        weight, bias = qkv.projection(module, "qkv")
        return torch.nn.functional.linear(normalized, weight, bias)
    return torch.nn.functional.linear(normalized, qk.projection_weight(module))


def stock_pack(qk, qkv, merged, scale, divisor):
    if merged.shape[-1] == 2 * WIDTH:
        return qk.pack_combined(merged, HEADS, HIDDEN, scale, divisor)
    return qkv.pack_qk(merged, scale, divisor)


def prepare_qk(qk, qkv, merged, scale, divisor, mode):
    if mode == "stock" or mode in AV_MODES:
        return stock_pack(qk, qkv, merged, scale, divisor)
    from .axial_layout_pack import pack_qk

    return pack_qk(merged, scale, divisor, mode[:2])


def prepare_av(weights, value, mode):
    length = weights.shape[1]
    left = weights.permute(0, 3, 1, 2).reshape(HEADS, length, length)
    if mode in ("av-nt", "av-nt-reverse"):
        from .axial_layout_pack import pack_transposed_value

        right = pack_transposed_value(value)
    else:
        right = value.permute(0, 3, 2, 1, 4).reshape(HEADS, length, length * HIDDEN)
    return left, right


def av_result(bmm, model):
    length = bmm.shape[1]
    view = bmm.view(HEADS, length, length, HIDDEN)
    if model == "rf2":
        return view.permute(2, 1, 0, 3).unsqueeze(0)
    return view.permute(1, 2, 0, 3).unsqueeze(0)


def attention(torch, qk, qkv, module, pair, bias, model, mode, *, gate_product=None):
    length = pair.shape[1]
    shape = (1, length, length, HEADS, HIDDEN)
    if module.is_row:
        pair, bias = pair.transpose(1, 2), bias.transpose(1, 2)
    pair = module.norm_pair(pair)
    bias = module.to_b(module.norm_bias(bias))
    merged = projected(torch, qk, qkv, module, pair, model)
    divisor = qk.key_divisor(model, length)
    query, key = prepare_qk(qk, qkv, merged, module.scaling, divisor, mode)
    if model == "rfd":
        del merged
    logits = contract(torch, query, key, mode in QK_MODES and mode.endswith("-reverse"))
    del query, key
    weights = torch.nn.functional.softmax(logits.permute(1, 2, 0).unsqueeze(0) + bias, dim=-2)
    del logits, bias
    if model == "rf2":
        value = merged[..., 2 * WIDTH:3 * WIDTH].view(shape)
        del merged
    else:
        value = module.to_v(pair).view(shape)
    if mode in AV_MODES:
        left, right = prepare_av(weights, value, mode)
        del weights, value
        weighted = av_result(contract(torch, left, right, mode.endswith("-reverse")), model)
        del left, right
    else:
        weighted = torch.einsum(qk.RF2_AV if model == "rf2" else qk.RFD_AV, weights, value)
        del weights, value
    gate = module.to_g(pair).view(shape)
    if gate_product is None:
        gate.sigmoid_()
        torch.mul(gate, weighted, out=gate)
    else:
        gate_product(gate, weighted)
    result = module.to_out(gate.view(1, length, length, WIDTH))
    return result.transpose(1, 2) if module.is_row else result
