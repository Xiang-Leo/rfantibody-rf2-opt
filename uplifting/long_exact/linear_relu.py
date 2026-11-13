"""Opt-in cuBLASLt Linear+ReLU experiment for dense RFantibody pair FFNs."""
import os
from pathlib import Path


TORCH_VERSION = '2.3.1+cu118'
MIN_LENGTH = 256
LT_MAX_ROWS = 65535 * 32


def feedforward_class(model):
    if model == 'rfd':
        from rfantibody.rfdiffusion.Attention_module import FeedForwardLayer
    else:
        from rfantibody.rf2.network.Attention_module import FeedForwardLayer
    return FeedForwardLayer


def eligible(torch, module, source, *, model, stride, width=128):
    if (type(source) is not torch.Tensor or source.ndim != 4 or source.shape[0] != 1
            or source.shape[1] != source.shape[2] or source.shape[-1] != width
            or source.shape[1] < MIN_LENGTH or source.shape[1] ** 2 >= LT_MAX_ROWS
            or source.dtype != torch.float32 or not source.is_cuda):
        return False
    layer = module.linear1
    return (
        torch.__version__ == TORCH_VERSION and torch.cuda.get_device_capability(source.device) == (8, 9)
        and not torch.is_grad_enabled() and not torch.is_autocast_enabled()
        and not torch.is_autocast_cpu_enabled() and not torch.backends.cuda.matmul.allow_tf32
        and torch.get_float32_matmul_precision() == 'highest'
        and os.environ.get('DISABLE_ADDMM_CUDA_LT') != '1'
        and (model == 'rfd' or module.training or stride <= 0 or stride >= source.shape[2])
        and type(layer) is torch.nn.Linear and not layer._forward_hooks and not layer._forward_pre_hooks
        and layer.weight.shape == (2 * width, width) and layer.weight.dtype == source.dtype
        and layer.weight.device == source.device and layer.weight.is_contiguous()
        and layer.bias is not None and layer.bias.shape == (2 * width,) and layer.bias.dtype == source.dtype
        and layer.bias.device == source.device and layer.bias.is_contiguous()
    )
