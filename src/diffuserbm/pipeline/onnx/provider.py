"""Module providing ONNX Provider Objects."""
import torch


_PROVIDERS = {}


class Provider:
    """Base class for ONNX Providers."""
    __NAME = None
    __OPTIONS = None

    def __init_subclass__(cls, **kwargs):
        _PROVIDERS[kwargs['device']] = cls
        cls.__NAME = kwargs['provider']
        cls.__OPTIONS = kwargs.get('options', None)

    @property
    def name(self) -> str:
        """Property getter for provider name"""
        return self.__NAME

    @property
    def options(self) -> any:
        """Property getter for provider options"""
        return self.__OPTIONS


class CPUProvider(Provider, device='cpu', provider='CPUExecutionProvider'):
    """Default Provider for CPU Execution."""


class CUDAProvider(Provider, device='cuda', provider='CUDAExecutionProvider'):
    """Default Provider for NVIDIA CUDA Execution."""
    @property
    def options(self) -> any:
        """Property getter for NVIDIA provider options"""
        return {
            "device_id": torch.cuda.current_device(),
            "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)
        }


def get_provider(device: str) -> Provider:
    """Provider factory method"""
    return _PROVIDERS[device]()
