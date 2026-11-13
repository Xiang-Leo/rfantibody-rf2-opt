"""FP32 LayerNorm that reads the RF2 triangle BMM channel planes directly."""

import ctypes
from functools import wraps
import math
from pathlib import Path


from .layer_norm import ContiguousLayerNorm, TORCH_VERSION, WIDTHS, audit_norm


class TriangleNorm(ContiguousLayerNorm):
    def __init__(self, torch, library):
        if torch.__version__ != TORCH_VERSION:
            raise RuntimeError(f"triangle LayerNorm requires Torch {TORCH_VERSION}")
        self.torch = torch
        self.library = ctypes.CDLL(str(Path(library).resolve(strict=True)))
        self.launch = self.library.triangle_norm_launch
        self.launch.argtypes = [ctypes.c_void_p] * 6 + [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float, ctypes.c_void_p,
        ]
        self.launch.restype = ctypes.c_int

    def eligible(self, x, norm):
        torch = self.torch
        if type(x) is not torch.Tensor or not x.is_cuda or x.dtype != torch.float32 or x.ndim != 4:
            return False
        batch, length, columns, width = x.shape
        return (
            batch == 1 and length == columns and length > 1 and width in WIDTHS
            and x.numel() < 2**32 and x.stride()[1:] == (length, 1, length**2)
            and norm.normalized_shape == (width,) and math.isfinite(norm.eps) and norm.eps > 0
            and all(t is not None and t.device == x.device and t.dtype == x.dtype
                    and t.shape == (width,) and t.is_contiguous() and t.data_ptr() % 16 == 0
                    for t in (norm.weight, norm.bias))
            and torch.cuda.get_device_capability(x.device) == (8, 9)
        )


def install(torch, mode, library):
    """Attach only to RF2 TriangleMultiplication.norm_out instances."""
    from rfantibody.rf2.network.Attention_module import TriangleMultiplication

    if mode not in ("fused", "audit"):
        raise ValueError(f"Unknown RF2 triangle normalization mode: {mode}")
    runner = TriangleNorm(torch, library)
    counts = {"fused_norms": 0, "fused_rows": 0, "audited_norms": 0, "fallback": 0,
              "outgoing": 0, "incoming": 0, **{f"width{width}": 0 for width in WIDTHS}}
    original = TriangleMultiplication.__init__

    @wraps(original)
    def init(module, *args, **kwargs):
        original(module, *args, **kwargs)
        norm = module.norm_out
        if type(norm) is not torch.nn.LayerNorm or norm.normalized_shape[0] not in WIDTHS:
            return
        stock = norm.forward
        audited = set()
        direction = "outgoing" if module.outgoing else "incoming"

        @wraps(stock)
        def forward(x):
            if (torch.is_grad_enabled() or torch.is_autocast_enabled()
                    or torch.is_autocast_cpu_enabled() or not runner.eligible(x, norm)):
                counts["fallback"] += 1
                return stock(x)
            key = (x.device, tuple(x.shape), x.stride(), norm.eps)
            if mode == "audit" and key not in audited:
                result = audit_norm(torch, runner, x, norm)
                audited.add(key)
                counts["audited_norms"] += 1
            else:
                result = runner(x, norm)
            counts["fused_norms"] += 1
            counts["fused_rows"] += x.shape[1] * x.shape[2]
            counts[f"width{x.shape[-1]}"] += 1
            counts[direction] += 1
            return result

        norm.forward = forward

    TriangleMultiplication.__init__ = init
    return counts
