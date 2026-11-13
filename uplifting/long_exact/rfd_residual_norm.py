"""Fuse two RFD pair residuals with their next stock width-128 LayerNorm."""

from contextvars import ContextVar
import ctypes
from functools import wraps
import math
from pathlib import Path


TORCH_VERSION = '2.3.1+cu118'
WIDTH = 128
_prepared = ContextVar('rfd_residual_norm_prepared', default=None)


class ResidualNorm:
    def __init__(self, torch, library):
        if torch.__version__ != TORCH_VERSION:
            raise RuntimeError(f'Expected Torch {TORCH_VERSION}')
        self.torch = torch
        self.library = ctypes.CDLL(str(Path(library).resolve(strict=True)))
        self.launch = self.library.rfd_residual_norm_launch
        self.launch.argtypes = [ctypes.c_void_p] * 8 + [
            ctypes.c_uint32, ctypes.c_float, ctypes.c_int, ctypes.c_void_p,
        ]
        self.launch.restype = ctypes.c_int

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
            and math.isfinite(norm.eps) and norm.eps > 0 and not norm.training
            and all(t is not None and t.shape == (WIDTH,) and t.dtype == lhs.dtype and t.device == lhs.device
                    and t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (norm.weight, norm.bias))
            and torch.cuda.get_device_capability(lhs.device) == (8, 9)
        )

    def __call__(self, lhs, rhs, norm, *, moments=False):
        torch = self.torch
        length = lhs.shape[1]
        residual = torch.empty_like(lhs)
        normalized = torch.empty_like(lhs)
        mean = lhs.new_empty((1, length, length, 1)) if moments else None
        inverse = lhs.new_empty((1, length, length, 1)) if moments else None
        with torch.cuda.device(lhs.device):
            status = self.launch(lhs.data_ptr(), rhs.data_ptr(), norm.weight.data_ptr(), norm.bias.data_ptr(),
                                 residual.data_ptr(), normalized.data_ptr(), None if mean is None else mean.data_ptr(),
                                 None if inverse is None else inverse.data_ptr(), length, norm.eps,
                                 int(not rhs.is_contiguous()), torch.cuda.current_stream(lhs.device).cuda_stream)
        if status:
            raise RuntimeError(f'CUDA residual LayerNorm launch failed ({status})')
        return (residual, normalized, mean, inverse) if moments else (residual, normalized)


def verify(torch, label, stock, trial):
    if (stock.shape, stock.stride(), stock.dtype) != (trial.shape, trial.stride(), trial.dtype):
        raise RuntimeError(f'{label}: output layout changed')
    a = stock.view(torch.int32).flatten()
    b = trial.view(torch.int32).flatten()
    different = stock.new_zeros((), dtype=torch.int64)
    for expected, observed in zip(a.split(2**20), b.split(2**20)):
        different += (expected != observed).sum(dtype=torch.int64)
    count = int(different.item())
    if count:
        raise RuntimeError(f'{label}: {count}/{a.numel()} values changed in FP32 raw bytes')


def preserve_norm_call(norm):
    stock = norm.forward

    @wraps(stock)
    def forward(source):
        pending = _prepared.get()
        if pending is not None and pending[0] is norm and pending[1] is source:
            if pending[3]:
                raise RuntimeError('The prepared RFD pair norm was invoked twice')
            pending[3] = True
            return pending[2]
        return stock(source)

    norm.forward = forward
