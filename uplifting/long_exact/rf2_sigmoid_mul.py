"""Fuse the Torch FP32 sigmoid and product in RF2's contiguous gates."""


import triton
import triton.language as tl


TORCH_VERSION = "2.3.1+cu118"
TRITON_VERSION = "2.3.1"
BLOCK = 1024


@triton.jit
def sigmoid_mul(logits, values):
    negative = tl.inline_asm_elementwise(
        "neg.f32 $0, $1;", "=f,f", (logits,), dtype=tl.float32, is_pure=True, pack=1,
    )
    exponential = tl.math.exp(negative)
    # These instructions leave FTZ disabled, including the reciprocal of a
    # large denominator such as 1 + exp(88), which is a subnormal sigmoid.
    return tl.inline_asm_elementwise(
        "{ .reg .f32 denominator, gate; add.rn.f32 denominator, 0f3f800000, $1;"
        " div.rn.f32 gate, 0f3f800000, denominator; mul.rn.f32 $0, gate, $2; }",
        "=f,f,f", (exponential, values), dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _product(logits, values, out, count, block: tl.constexpr):
    offset = tl.program_id(0) * block + tl.arange(0, block)
    valid = offset < count
    gate = tl.load(logits + offset, valid, other=0.0)
    value = tl.load(values + offset, valid, other=0.0)
    tl.store(out + offset, sigmoid_mul(gate, value), valid)


def check_versions(torch):
    if (torch.__version__, triton.__version__) != (TORCH_VERSION, TRITON_VERSION):
        raise RuntimeError(f"RF2 gate fusion requires Torch {TORCH_VERSION} and Triton {TRITON_VERSION}")


def eligible(torch, logits, values, out):
    if (type(logits) is not torch.Tensor or not logits.is_cuda or logits.dtype != torch.float32
            or not 0 < logits.numel() < 2**31 or torch.is_grad_enabled()
            or torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()):
        return False
    if any(type(tensor) is not torch.Tensor or tensor.device != logits.device
           or tensor.dtype != torch.float32 or tensor.shape != logits.shape
           or not tensor.is_contiguous() or tensor.is_neg() or tensor.data_ptr() % 4
           for tensor in (logits, values, out)):
        return False
    destination, size = out.data_ptr(), out.nbytes
    addresses = logits.data_ptr(), values.data_ptr()
    return (
        abs(addresses[0] - addresses[1]) >= size
        and all(address == destination or abs(address - destination) >= size for address in addresses)
        and torch.cuda.get_device_capability(out.device) == (8, 9)
    )


def launch(torch, logits, values, out, *, block=BLOCK):
    """Run the pointwise product on eligible tensors, including exact aliasing."""
    with torch.cuda.device(out.device):
        _product[(triton.cdiv(out.numel(), block),)](
            logits, values, out, out.numel(), block, num_warps=4, enable_fp_fusion=False,
        )
    return out


def install_gate(torch, *, audit=False, scopes=("triangle", "pair")):
    """Install selected gate products after the chosen RF2 base has loaded."""
    check_versions(torch)
    from . import rf2_pair_gate
    from . import triangle_pack
    from .rfd_residual_norm import verify

    if not scopes or not set(scopes) <= {"triangle", "pair"}:
        raise ValueError("Choose triangle or pair gate fusion")
    counts = {"triangle": 0, "pair": 0, "triangle_audited": 0, "pair_audited": 0,
              "triangle_fallback": 0, "pair_fallback": 0, "logical_bytes_avoided": 0}
    audited = set()
    stock_triangle = triangle_pack.final_gate
    stock_pair = rf2_pair_gate._gate_product

    def product(scope, logits, values, out, fallback):
        if not eligible(torch, logits, values, out):
            counts[f"{scope}_fallback"] += 1
            return fallback()
        key = (scope, logits.device, logits.shape)
        reference = (torch.sigmoid(logits) * values if audit and key not in audited else None)
        result = launch(torch, logits, values, out)
        if reference is not None:
            verify(torch, f"RF2 {scope} sigmoid product", reference, result)
            audited.add(key)
            counts[f"{scope}_audited"] += 1
        counts[scope] += 1
        counts["logical_bytes_avoided"] += 2 * result.nbytes
        return result

    if "triangle" in scopes:
        def triangle(logits, values):
            return product("triangle", logits, values, logits, lambda: stock_triangle(logits, values))

        triangle_pack.final_gate = triangle
    if "pair" in scopes:
        def pair(torch, logits, values, out):
            return product("pair", logits, values, out, lambda: stock_pair(torch, logits, values, out))

        rf2_pair_gate._gate_product = pair
    return counts
