"""Replace the observed dense RF2 SE(3) radial contractions in FP32."""

from functools import wraps

from .rf2_se3_radial import contract


EDGE_COUNTS = {157212, 1260006, 1262252}
BASIS_SHAPES = {(32, 1): ((1, 2, 4), 64), (2, 3): ((3, 4, 4), 8)}


def install(torch, mode):
    from se3_transformer.model.layers.convolution import ConvSE3FuseLevel, VersatileConvSE3

    if mode != "forward":
        raise ValueError(f"Unknown RF2 radial mode: {mode}")
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
        result = contract(torch, radial, vectors, mode)
        counts[f"k{inner}"] += 1
        return result

    VersatileConvSE3.forward = forward
    return counts
