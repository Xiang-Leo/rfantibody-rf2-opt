"""Fuse the FP32 axial sigmoid and strided AV multiply in place."""
from functools import wraps
from pathlib import Path


BLOCKS = (256, 512, 1024, 2048)
DEFAULT_BLOCK = 1024
TORCH_VERSION = "2.3.1+cu118"


def weighted_layout(weighted):
    length = weighted.shape[1]
    width = 32
    outer_stride = length * width
    head_stride = length * outer_stride
    layouts = {
        (width, outer_stride, head_stride, 1): "normal-rf2",
        (outer_stride, width, head_stride, 1): "normal-rfd",
        (outer_stride, 1, head_stride, length): "reverse-rf2",
        (1, outer_stride, head_stride, length): "reverse-rfd",
        (length * 128, 128, 32, 1): "direct",
    }
    return layouts.get(weighted.stride()[1:])


def stock_gate(torch, gate, weighted):
    gate.sigmoid_()
    torch.mul(gate, weighted, out=gate)


def fused_gate(torch, gate, weighted, block=DEFAULT_BLOCK):
    """Return the applied layout name, or None for an unsupported call."""
    if (type(gate) is not torch.Tensor or type(weighted) is not torch.Tensor
            or gate.ndim != 5 or weighted.shape != gate.shape):
        return None
    length = gate.shape[1]
    if (gate.shape != (1, length, length, 4, 32) or length <= 1 or gate.numel() >= 2**31
            or not gate.is_contiguous() or not gate.is_cuda or gate.dtype != torch.float32
            or gate.device != weighted.device or gate.dtype != weighted.dtype or block not in BLOCKS
            or gate.is_neg() or weighted.is_neg() or gate.data_ptr() % 4 or weighted.data_ptr() % 4
            or gate.untyped_storage().data_ptr() == weighted.untyped_storage().data_ptr()
            or torch.is_grad_enabled() or torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()
            or torch.cuda.get_device_capability(gate.device) != (8, 9)):
        return None
    layout = weighted_layout(weighted)
    if layout is None:
        return None
    from .axial_gate_kernel import launch

    with torch.cuda.device(gate.device):
        launch(gate, weighted, layout, block)
    return layout


def dependencies():
    from . import axial_layout

    return axial_layout, *axial_layout.dependencies()

def replace_qk(torch, mode, *, layouts=None):
    """Apply the gate after P14's selected axial layout, or stock P9 QKV."""
    scope, _, block_value = mode.partition("-")
    block = int(block_value) if block_value else DEFAULT_BLOCK
    if scope not in ("fused", "audit") or block not in BLOCKS:
        raise ValueError("choose fused or audit with optional block 256, 512, 1024 or 2048")
    from .rf2_sigmoid_mul import check_versions

    check_versions(torch)
    axial, qk, qkv = dependencies()
    if layouts is None:
        layouts = {}
    layouts = {int(length): layout for length, layout in layouts.items()}
    if any(layout not in (*axial.QK_MODES, *axial.AV_MODES) for layout in layouts.values()):
        raise ValueError("unknown selected P14 axial layout")
    original = qk.attention
    audited = set()
    counts = {"fused": 0, "normal": 0, "reverse": 0, "direct": 0,
              "audited": 0, "gate_fallback": 0, "module_fallback": 0, "layouts": {}}

    @wraps(original)
    def trial(module, pair, bias, model):
        if (model != "rf2" or torch.__version__ != TORCH_VERSION
                or not qkv.eligible(module, pair, bias, 0, model, "qkv")):
            counts["module_fallback"] += 1
            return original(module, pair, bias, model)
        layout = layouts.get(pair.shape[1], "stock")

        def apply(gate, weighted):
            key = (id(module), gate.shape, weighted.stride())
            check = scope == "audit" and key not in audited
            if check:
                expected = gate.clone()
                stock_gate(torch, expected, weighted)
            actual = fused_gate(torch, gate, weighted, block)
            if actual is None:
                counts["gate_fallback"] += 1
                stock_gate(torch, gate, weighted)
                return
            if check:
                if not torch.equal(gate.view(torch.int32), expected.view(torch.int32)):
                    raise RuntimeError(f"axial gate changed FP32 bytes at L={gate.shape[1]} in {actual}")
                audited.add(key)
                counts["audited"] += 1
            counts["fused"] += 1
            counts[actual.split("-")[0]] += 1
            counts["layouts"][layout] = counts["layouts"].get(layout, 0) + 1

        return axial.attention(torch, qk, qkv, module, pair, bias, model, layout, gate_product=apply)

    qk.attention = trial
    return counts
