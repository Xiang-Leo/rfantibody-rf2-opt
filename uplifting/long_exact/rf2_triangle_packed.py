"""Use a combined RF2 triangle projection only after its real input passes."""

from functools import wraps


def install(torch, base):
    from rfantibody.rf2.network.Attention_module import TriangleMultiplication
    from . import triangle_pack

    counts = {"base": base, "packed": 0, "projection_audits": 0, "projection_mismatches": 0, "fallback": 0}
    original = TriangleMultiplication.forward
    layers = ("left_proj", "left_gate", "right_proj", "right_gate")
    audited = {}

    @wraps(original)
    def forward(module, pair, stride):
        if (pair.ndim != 4 or pair.shape[1] not in (1123, 1124)
                or not triangle_pack.eligible(torch, module, pair, stride)):
            counts["fallback"] += 1
            return original(module, pair, stride)
        normalized = module.norm(pair)
        tensors = tuple(parameter for name in layers for parameter in getattr(module, name).parameters())
        key = None
        if not any(tensor.is_inference() for tensor in tensors):
            key = (module, tuple(normalized.shape), normalized.stride(), normalized.data_ptr() % 256,
                   tuple((id(tensor), tensor._version) for tensor in tensors))
        exact = audited.get(key) if key is not None else None
        projected = None
        if exact is not False:
            projected = triangle_pack.combined_projection(torch, module, normalized)
        if exact is None:
            exact = True
            for name, trial in zip(layers, projected.split(module.d_hidden, dim=-1)):
                stock = getattr(module, name)(normalized)
                if not torch.equal(stock.view(torch.int32), trial.view(torch.int32)):
                    exact = False
                del stock
            del trial
            counts["projection_audits"] += 1
            counts["projection_mismatches"] += int(not exact)
            if key is not None:
                audited[key] = exact
        if exact:
            left, right = triangle_pack.pack_products(torch, projected, module.d_hidden,
                                                       module.outgoing, "packed-dense")
            counts["packed"] += 1
        else:
            left, right = triangle_pack.separate_products(torch, module, normalized, "separate-dense")
            counts["fallback"] += 1
        del projected
        product = torch.bmm(left, right)
        del left, right
        result = product.permute(1, 2, 0).unsqueeze(0)
        result = module.out_proj(module.norm_out(result))
        return triangle_pack.final_gate(module.gate(normalized), result)

    TriangleMultiplication.forward = forward
    return counts
