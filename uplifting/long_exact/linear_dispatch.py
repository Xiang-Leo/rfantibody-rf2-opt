"""Apply screened FP32 RF2 Linear and Linear/ReLU dispatches."""

import atexit
import ctypes as c
from functools import wraps
import json
from pathlib import Path

from . import linear_relu
from .linear_api import info, library
from .rfd_residual_norm import verify


def choices(version):
    data = json.loads(Path(__file__).with_name("dispatch.json").read_text())
    if (data["torch"], data["cublaslt"]) != ("2.3.1+cu118", version):
        raise ValueError("Linear dispatch requires the pinned cuBLASLt runtime")
    return {tuple(row["shape"]): (row["layout"], row["algorithm"]) for row in data["linear"]}


class Dispatcher:
    def __init__(self, torch, native_dir):
        self.torch = torch
        self.stock = torch.nn.functional.linear
        self.api = library(native_dir / "linear_cublaslt.so")
        self.choices = choices(self.api.axial_lt_version())
        self.plans = {}
        self.rejected = set()
        self.packs = {}
        self.workspaces = {}
        atexit.register(self.close)

    def close(self):
        for plan in self.plans.values():
            self.api.axial_lt_destroy(plan)
        self.plans.clear()

    def packed(self, weight, stream):
        if weight.is_inference():
            return weight.t().contiguous()
        key = (id(weight), stream)
        old = self.packs.get(key)
        if old is not None and old[0] is weight and old[1] == weight._version:
            return old[2]
        transposed = weight.t().contiguous()
        self.packs[key] = (weight, weight._version, transposed)
        return transposed

    def choose(self, source, weight, bias, relu):
        torch = self.torch
        if (type(source) is not torch.Tensor or source.ndim < 2 or weight.ndim != 2
                or source.dtype != torch.float32 or not source.is_cuda
                or source.shape[-1] != weight.shape[1] or not source.is_contiguous()
                or torch.is_grad_enabled() or torch.is_autocast_enabled()
                or torch.is_autocast_cpu_enabled() or torch.backends.cuda.matmul.allow_tf32
                or torch.get_float32_matmul_precision() != "highest"):
            return None
        n, k = weight.shape
        key = (source.numel() // k, k, n, bias is not None, relu)
        if (key not in self.choices or (source.device, key) in self.rejected
                or torch.cuda.get_device_capability(source.device) != (8, 9)
                or any(t.device != source.device or t.dtype != source.dtype or not t.is_contiguous()
                       or t.data_ptr() % 16 for t in (source, weight, bias) if t is not None)
                or (bias is not None and bias.shape != (n,))):
            return None
        return key

    def run(self, source, weight, bias, key):
        rows, k, n, with_bias, relu = key
        torch = self.torch
        layout, choice = self.choices[key]
        with torch.cuda.device(source.device):
            stream = torch.cuda.current_stream(source.device).cuda_stream
            packed = self.packed(weight, stream) if layout != "tn" else weight
            if layout == "torch-packed":
                result = self.stock(source, packed.t(), bias)
                return result.relu_() if relu else result
            stream_key = (source.device, stream)
            workspace = self.workspaces.get(stream_key)
            if workspace is None:
                workspace = torch.empty(32 * 2**20, dtype=torch.uint8, device=source.device)
                self.workspaces[stream_key] = workspace
            plan_key = (stream_key, key)
            plan = self.plans.get(plan_key)
            if plan is None:
                status = c.c_int()
                plan = self.api.linear_lt_create(rows, k, n, layout == "nn",
                                                 bias.data_ptr() if with_bias else None, relu,
                                                 workspace.nbytes, c.byref(status))
                if not plan:
                    return None
                if (self.api.axial_lt_count(plan) <= choice["index"]
                        or info(self.api, plan, choice["index"]) !=
                        {name: choice[name] for name in ("index", "id", "tile", "stages", "workspace", "info_status")}):
                    self.api.axial_lt_destroy(plan)
                    return None
                self.plans[plan_key] = plan
            elif with_bias:
                status = self.api.linear_lt_set_bias(plan, bias.data_ptr())
                if status:
                    return None
            output = source.new_empty((*source.shape[:-1], n))
            status = self.api.axial_lt_run(
                plan, choice["index"], source.data_ptr(), packed.data_ptr(), output.data_ptr(),
                workspace.data_ptr(), workspace.nbytes, stream,
            )
            if status:
                return None
            return output


def install(torch, mode, native_dir, base, residual):
    dispatch = Dispatcher(torch, native_dir)
    counts = {"base": base, "residual": residual, "linear": 0, "relu": 0,
              "audited": 0, "fallback": 0, "shapes": {}}
    audited = set()

    def candidate(source, weight, bias, key, category):
        result = dispatch.run(source, weight, bias, key)
        if result is None:
            dispatch.rejected.add((source.device, key))
            counts["fallback"] += 1
            stock = dispatch.stock(source, weight, bias)
            return stock.relu_() if key[-1] else stock
        if mode == "audit" and key not in audited:
            stock = dispatch.stock(source, weight, bias)
            if key[-1]:
                stock.relu_()
            verify(torch, f"RF2 Linear {key}", stock, result)
            audited.add(key)
            counts["audited"] += 1
        counts[category] += 1
        label = str(key)
        counts["shapes"][label] = counts["shapes"].get(label, 0) + 1
        return result

    @wraps(dispatch.stock)
    def linear(input, weight, bias=None):
        key = dispatch.choose(input, weight, bias, False)
        if key is None:
            return dispatch.stock(input, weight, bias)
        return candidate(input, weight, bias, key, "linear")

    cls = linear_relu.feedforward_class("rf2")
    original = cls.forward

    @wraps(original)
    def forward(module, source, *args, **kwargs):
        stride = args[0] if args else kwargs.get("stride", 0)
        if not linear_relu.eligible(torch, module, source, model="rf2", stride=stride):
            return original(module, source, *args, **kwargs)
        normalized = module.norm(source)
        layer = module.linear1
        key = dispatch.choose(normalized, layer.weight, layer.bias, True)
        if key is None:
            hidden = torch.relu(layer(normalized))
        else:
            hidden = candidate(normalized, layer.weight, layer.bias, key, "relu")
        return module.linear2(module.dropout(hidden))

    torch.nn.functional.linear = linear
    cls.forward = forward
    return counts
