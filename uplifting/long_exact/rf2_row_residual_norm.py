"""Fuse RF2's incoming-triangle residual with its transposed row LayerNorm."""

import ctypes
from functools import wraps
from pathlib import Path


from .rf2_residual_norm import RF2ResidualNorm
from .rfd_residual_norm import TORCH_VERSION, WIDTH, _prepared, verify


MODES = ("fused", "audit", "radial-only", "projection-only", "p12", "p12-audit")


class RowResidualNorm(RF2ResidualNorm):
    def __init__(self, torch, library, order="source"):
        if torch.__version__ != TORCH_VERSION:
            raise RuntimeError(f"Expected Torch {TORCH_VERSION}")
        if order not in ("source", "target"):
            raise ValueError("Choose source or target row order")
        self.torch = torch
        self.source_major = order == "source"
        self.library = ctypes.CDLL(str(Path(library).resolve(strict=True)))
        self.launch = self.library.rf2_row_residual_norm_launch
        self.launch.argtypes = [ctypes.c_void_p] * 8 + [
            ctypes.c_uint32, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
        ]
        self.launch.restype = ctypes.c_int

    def __call__(self, lhs, rhs, norm, *, moments=False):
        torch = self.torch
        length = lhs.shape[1]
        residual = torch.empty_like(lhs)
        normalized = torch.empty_like(lhs)
        mean = lhs.new_empty((1, length, length, 1)) if moments else None
        inverse = lhs.new_empty((1, length, length, 1)) if moments else None
        with torch.cuda.device(lhs.device):
            status = self.launch(
                lhs.data_ptr(), rhs.data_ptr(), norm.weight.data_ptr(), norm.bias.data_ptr(),
                residual.data_ptr(), normalized.data_ptr(),
                None if mean is None else mean.data_ptr(), None if inverse is None else inverse.data_ptr(),
                length, norm.eps, int(not rhs.is_contiguous()), int(self.source_major),
                torch.cuda.current_stream(lhs.device).cuda_stream,
            )
        if status:
            raise RuntimeError(f"CUDA transposed residual LayerNorm launch failed ({status})")
        return (residual, normalized, mean, inverse) if moments else (residual, normalized)


def same_view(lhs, rhs):
    return (lhs.shape == rhs.shape and lhs.stride() == rhs.stride()
            and lhs.dtype == rhs.dtype and lhs.device == rhs.device and lhs.data_ptr() == rhs.data_ptr())


def preserve_row_norm(norm):
    stock = norm.forward

    @wraps(stock)
    def forward(source):
        pending = _prepared.get()
        if pending is not None and pending[0] is norm and same_view(source, pending[1]):
            if pending[3]:
                raise RuntimeError("The prepared transposed row LayerNorm was requested twice")
            pending[3] = True
            return pending[2]
        return stock(source)

    norm.forward = forward


def install(torch, mode, library, residual_library, *, order):
    if mode not in MODES:
        raise ValueError(f"Choose one of {', '.join(MODES)}")
    from rfantibody.rf2.network.Track_module import PairStr2Pair
    from . import rf2_pair_gate

    counts = {name: 0 for name in ("triangle", "incoming", "row", "col", "audited", "fallback_pairs",
                                   "fallback_residuals", "avoided_incoming_read_bytes")}
    kernel = RF2ResidualNorm(torch, residual_library)
    row_kernel = RowResidualNorm(torch, library, order)
    audit = mode.endswith("audit")
    original_init = PairStr2Pair.__init__
    original_forward = PairStr2Pair.forward

    @wraps(original_init)
    def initialize(module, *args, **kwargs):
        original_init(module, *args, **kwargs)
        norm = module.row_attn.norm_pair
        if (getattr(module, "_uplifting_rf2_residual_norm", False)
                and type(norm) is torch.nn.LayerNorm and norm.normalized_shape == (WIDTH,)):
            preserve_row_norm(norm)
            module._uplifting_rf2_row_residual_norm = True

    def prepare(left, right, norm, phase):
        incoming = phase == "incoming"
        runner = row_kernel if incoming else kernel
        if not runner.eligible(left, right, norm):
            counts["fallback_residuals"] += 1
            return left + right, None
        residual, normalized = runner(left, right, norm)
        if incoming and audit:
            reference = left + right
            verify(torch, "RF2 incoming residual", reference, residual)
            verify(torch, "RF2 transposed row norm", norm(reference.transpose(1, 2)), normalized)
            counts["audited"] += 1
        counts[phase] += 1
        if incoming:
            counts["avoided_incoming_read_bytes"] += residual.nbytes
        return residual, normalized

    def consume(residual, normalized, consumer, norm, phase, *args):
        if normalized is None:
            return consumer(residual, *args)
        view = residual.transpose(1, 2) if phase == "incoming" else residual
        pending = [norm, view, normalized, False]
        token = _prepared.set(pending)
        try:
            result = consumer(residual, *args)
        finally:
            _prepared.reset(token)
        if not pending[3]:
            raise RuntimeError(f"The RF2 {phase} consumer did not request its prepared LayerNorm")
        return result

    @wraps(original_forward)
    def forward(module, pair, rbf_feat, state, strides, crop=-1):
        row_norm = module.row_attn.norm_pair
        if (not getattr(module, "_uplifting_rf2_row_residual_norm", False) or not module.training
                or module.row_attn.is_row is not True or module.col_attn.is_row is not False
                or row_norm._forward_pre_hooks or row_norm._forward_hooks
                or not rf2_pair_gate.eligible(torch, module, pair, rbf_feat, state, strides, crop)):
            counts["fallback_pairs"] += 1
            return original_forward(module, pair, rbf_feat, state, strides, crop)
        features = rf2_pair_gate.gate_features(torch, module, rbf_feat, state, "gate-65536")
        outgoing = module.drop_row(module.tri_mul_out(pair, -1))
        pair, normalized = prepare(pair, outgoing, module.tri_mul_in.norm, "triangle")
        del outgoing
        incoming = consume(pair, normalized, module.tri_mul_in, module.tri_mul_in.norm, "triangle", -1)
        del normalized
        incoming = module.drop_row(incoming)
        pair, normalized = prepare(pair, incoming, row_norm, "incoming")
        del incoming
        row = consume(pair, normalized, module.row_attn, row_norm, "incoming", features, -1)
        del normalized
        row = module.drop_row(row)
        pair, normalized = prepare(pair, row, module.col_attn.norm_pair, "row")
        del row
        col = consume(pair, normalized, module.col_attn, module.col_attn.norm_pair, "row", features, -1)
        del normalized, features
        col = module.drop_col(col)
        pair, normalized = prepare(pair, col, module.ff.norm, "col")
        del col
        return pair + consume(pair, normalized, module.ff, module.ff.norm, "col", -1)

    PairStr2Pair.__init__ = initialize
    PairStr2Pair.forward = forward
    return counts
