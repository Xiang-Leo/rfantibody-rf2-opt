"""Reuse frozen triangle projection weights within each candidate process."""

import torch


def _packed_weights(module, name, layers):
    # Store the cache outside the state_dict. Tensor versions change on a
    # normal parameter update; identities and devices also catch .to()/reload.
    parameters = tuple(tensor for layer in layers for tensor in (layer.weight, layer.bias)
                       if tensor is not None)
    try:
        signature = tuple((id(tensor), tensor._version, tensor.device, tensor.dtype)
                          for tensor in parameters)
    except RuntimeError:
        # Tensors created inside inference_mode do not have a version counter.
        return _concatenate_weights(layers)
    cache = getattr(module, name, None)
    if cache is None or cache[0] != signature:
        cache = (signature, _concatenate_weights(layers))
        setattr(module, name, cache)
    return cache[1]


def _concatenate_weights(layers):
    weight = torch.cat([layer.weight for layer in layers], dim=0)
    bias = None
    if layers[0].bias is not None:
        bias = torch.cat([layer.bias for layer in layers])
    return weight, bias
