"""Fuse exact RF2 pair residuals with the next width-128 LayerNorm."""

from functools import wraps
import math


from .rfd_residual_norm import (
    WIDTH, ResidualNorm, _prepared, preserve_norm_call, verify,
)


PHASES = ("triangle", "row", "col")


class RF2ResidualNorm(ResidualNorm):
    def eligible(self, lhs, rhs, norm):
        torch = self.torch
        if (type(lhs) is not torch.Tensor or type(rhs) is not torch.Tensor or lhs.ndim != 4
                or lhs.shape[0] != 1 or lhs.shape[1] != lhs.shape[2] or lhs.shape[-1] != WIDTH
                or rhs.shape != lhs.shape or lhs.dtype != torch.float32 or rhs.dtype != lhs.dtype
                or not lhs.is_cuda or lhs.device != rhs.device or not lhs.is_contiguous()):
            return False
        length = lhs.shape[1]
        return (
            1 < length and length**2 * WIDTH < 2**32
            and rhs.stride()[1:] in ((length * WIDTH, WIDTH, 1), (WIDTH, length * WIDTH, 1))
            and lhs.data_ptr() % 16 == rhs.data_ptr() % 16 == 0
            and type(norm) is torch.nn.LayerNorm and norm.normalized_shape == (WIDTH,)
            and math.isfinite(norm.eps) and norm.eps > 0
            and all(t is not None and t.shape == (WIDTH,) and t.dtype == lhs.dtype and t.device == lhs.device
                    and t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (norm.weight, norm.bias))
            and torch.cuda.get_device_capability(lhs.device) == (8, 9)
        )


def install(torch, mode, library):
    """Keep RF2's training path and dropout order while reusing the P10 CUDA kernel."""
    if mode not in ("fused", "audit", *PHASES):
        raise ValueError("Choose fused, audit, triangle, row, or col")
    from rfantibody.rf2.network.Track_module import PairStr2Pair
    from . import rf2_pair_gate

    counts = {phase: 0 for phase in PHASES}
    counts.update(audited=0, fallback_pairs=0, fallback_residuals=0, avoided_residual_read_bytes=0)
    kernel = RF2ResidualNorm(torch, library)
    phases = PHASES if mode in ("fused", "audit") else (mode,)
    original_init = PairStr2Pair.__init__
    original_forward = PairStr2Pair.forward

    @wraps(original_init)
    def initialize(module, *args, **kwargs):
        original_init(module, *args, **kwargs)
        norms = (module.tri_mul_in.norm, module.col_attn.norm_pair, module.ff.norm)
        if all(type(norm) is torch.nn.LayerNorm and norm.normalized_shape == (WIDTH,) for norm in norms):
            for norm in norms:
                preserve_norm_call(norm)
            module._uplifting_rf2_residual_norm = True

    def prepare(left, right, norm, phase):
        if phase not in phases or not kernel.eligible(left, right, norm):
            counts["fallback_residuals"] += 1
            return left + right, None
        residual, normalized = kernel(left, right, norm)
        if mode == "audit":
            reference = left + right
            verify(torch, f"RF2 {phase} residual", reference, residual)
            verify(torch, f"RF2 {phase} normalization", norm(reference), normalized)
            counts["audited"] += 1
        counts[phase] += 1
        counts["avoided_residual_read_bytes"] += residual.numel() * residual.element_size()
        return residual, normalized

    def consume(residual, normalized, consumer, norm, phase, *args):
        if normalized is None:
            return consumer(residual, *args)
        pending = [norm, residual, normalized, False]
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
        if (not getattr(module, "_uplifting_rf2_residual_norm", False) or not module.training
                or module.row_attn.is_row is not True or module.col_attn.is_row is not False
                or not rf2_pair_gate.eligible(torch, module, pair, rbf_feat, state, strides, crop)):
            counts["fallback_pairs"] += 1
            return original_forward(module, pair, rbf_feat, state, strides, crop)
        features = rf2_pair_gate.gate_features(torch, module, rbf_feat, state, "gate-65536")
        outgoing = module.drop_row(module.tri_mul_out(pair, -1))
        pair, normalized = prepare(pair, outgoing, module.tri_mul_in.norm, "triangle")
        del outgoing
        incoming = consume(pair, normalized, module.tri_mul_in, module.tri_mul_in.norm, "triangle", -1)
        del normalized
        # Row attention normalizes a spatial transpose, so retain the stock
        # incoming-triangle residual and row normalization here.
        pair = pair + module.drop_row(incoming)
        del incoming
        row = module.drop_row(module.row_attn(pair, features, -1))
        pair, normalized = prepare(pair, row, module.col_attn.norm_pair, "row")
        del row
        col = consume(pair, normalized, module.col_attn, module.col_attn.norm_pair, "row", features, -1)
        del normalized, features
        col = module.drop_col(col)
        pair, normalized = prepare(pair, col, module.ff.norm, "col")
        del col
        ff = consume(pair, normalized, module.ff, module.ff.norm, "col", -1)
        return pair + ff

    PairStr2Pair.__init__ = initialize
    PairStr2Pair.forward = forward
    return counts
