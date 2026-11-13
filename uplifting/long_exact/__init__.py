"""The screened FP32 RF2 kernel stack for the pinned Ada runtime."""

import ctypes
import os
from pathlib import Path
import platform
import sys


DEFAULT_NATIVE_DIR = Path(__file__).resolve().parents[1] / "native-build"
CUDA_SOURCES = (
    "radial_norm", "axial_norm", "layer_norm", "triangle_norm",
    "rfd_residual_norm", "rf2_row_residual_norm",
)
TORCH_VERSION = "2.3.1+cu118"
TRITON_VERSION = "2.3.1"
CUBLASLT_VERSION = 111103


def native_path(native_dir=None):
    return Path(native_dir or os.environ.get("UPLIFTING_RF2_NATIVE_DIR", DEFAULT_NATIVE_DIR)).expanduser().resolve()


def unsupported_reason(torch, native_dir):
    if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 10):
        return "requires Linux x86-64 and Python 3.10"
    if torch.__version__ != TORCH_VERSION:
        return f"requires PyTorch {TORCH_VERSION}"
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 9):
        return "requires a CUDA SM89 device"
    if torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != "highest":
        return "requires the original FP32 matmul settings"
    try:
        import triton
        from triton.runtime import cache
    except ImportError:
        return f"requires importable Triton {TRITON_VERSION} and its dependencies"
    if triton.__version__ != TRITON_VERSION or not hasattr(cache, "random"):
        return f"requires Triton {TRITON_VERSION}"
    try:
        for name in CUDA_SOURCES:
            api = ctypes.CDLL(str(native_dir / f"{name}_kernel.so"))
            getattr(api, f"{name}_launch")
        from .linear_api import library

        if library(native_dir / "linear_cublaslt.so").axial_lt_version() != CUBLASLT_VERSION:
            return f"requires cuBLASLt {CUBLASLT_VERSION}"
    except (OSError, AttributeError):
        return "native kernels are unavailable; run uplifting/build_exact.py"
    return None


def install(torch, mode="fused", native_dir=None):
    """Install the measured stack before RF2 constructs its model, or fall back."""
    if mode not in ("fused", "audit"):
        raise ValueError("Choose fused or audit")
    native_dir = native_path(native_dir)
    reason = unsupported_reason(torch, native_dir)
    if reason is not None:
        return {"enabled": False, "fallback_reason": reason}

    from . import (
        axial_gate, axial_norm, axial_qkvg, layer_norm, linear_dispatch, radial_norm,
        rf2_ca_features, rf2_full_graph, rf2_pair_gate, rf2_radial_fused, rf2_radial_stream,
        rf2_residual_norm, rf2_row_residual_norm, rf2_sigmoid_mul, rf2_triangle_packed,
        rfd_axial_qk, triangle_norm, triangle_pack,
    )

    def kernel(name):
        return native_dir / f"{name}_kernel.so"

    axial = rfd_axial_qk.install(torch)
    radial = rf2_radial_fused.install(torch, "flat")
    radial_mlp, prefix = radial_norm.install(torch, "fused", kernel("radial_norm"), return_prefix=True)
    stream = rf2_radial_stream.install(torch, 8192, prefix)
    row_norm = axial_norm.install(torch, "rf2", kernel("axial_norm"))
    norms = layer_norm.install(torch, "fused", kernel("layer_norm"))
    p8 = {"kernels": {"kernels": {"axial": axial, "radial": radial, "stream": stream, "axial_norm": row_norm},
                      "norm": radial_mlp}, "contiguous_norm": norms}
    p9 = {"p8": p8, "graph": rf2_full_graph.install(torch),
          "gate": rf2_pair_gate.install(torch, "gate-65536"),
          "triangle_norm": triangle_norm.install(torch, "fused", kernel("triangle_norm")),
          "axial": axial_qkvg.replace_qk("rf2", "qkv")}
    p10 = {"p9": p9, "triangle": triangle_pack.install(torch, "separate-dense")}
    residual = rf2_residual_norm.install(torch, "fused", kernel("rfd_residual_norm"))
    linear = linear_dispatch.install(torch, mode, native_dir, p10, residual)
    ca = rf2_ca_features.install(torch, "fused", linear)
    triangle = rf2_triangle_packed.install(torch, ca)
    row = rf2_row_residual_norm.install(torch, mode, kernel("rf2_row_residual_norm"),
                                        kernel("rfd_residual_norm"), order="source")
    base = {"base": triangle, "row": row}
    gates = rf2_sigmoid_mul.install_gate(torch, audit=mode == "audit", scopes=("triangle",))
    layouts = {1123: "av-reverse", 1124: "av-reverse"}
    gated_axial = axial_gate.replace_qk(torch, f"{mode}-512", layouts=layouts)
    return {"enabled": True, "base": base, "gates": gates, "axial_gate": gated_axial}
