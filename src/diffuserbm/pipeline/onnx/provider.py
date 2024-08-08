"""Module providing ONNX Provider Objects."""
import torch
import onnxruntime as ort


_PROVIDERS = {}


class Provider:
    """Base class for ONNX Providers."""
    __NAME = None
    __OPTIONS = None

    def __init_subclass__(cls, **kwargs):
        _PROVIDERS[kwargs['device']] = cls
        cls.__NAME = kwargs['provider']
        cls.__OPTIONS = kwargs.get('options', None)

    def __init__(self, **kwargs):
        pass

    @property
    def name(self) -> str:
        """Property getter for provider name"""
        return self.__NAME

    @property
    def options(self) -> any:
        """Property getter for provider options"""
        return self.__OPTIONS

    @property
    def session_options(self):
        """Property getter for provider's session options"""
        return ort.SessionOptions()


class CPUProvider(Provider, device='cpu', provider='CPUExecutionProvider'):
    """Default Provider for CPU Execution."""


class CUDAProvider(Provider, device='cuda', provider='CUDAExecutionProvider'):
    """Default Provider for NVIDIA CUDA Execution."""
    @property
    def options(self) -> any:
        """Property getter for NVIDIA provider options"""
        return {
            "device_id": torch.cuda.current_device(),
            "user_compute_stream": str(torch.cuda.current_stream().cuda_stream),
            "do_copy_in_default_stream": "false",
            "cudnn_conv_algo_search": "DEFAULT",
            "cudnn_conv_use_max_workspace": "1",
            "cudnn_conv1d_pad_to_nc1d": "1",
        }


class DirectMLProvider(Provider, device='dml', provider='DmlExecutionProvider'):
    """Default Provider for DirectML Execution."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__device_id = kwargs.get('onnx.device_id', 0)

    @property
    def options(self) -> any:
        """Property getter for DirectML provider options"""
        return {
            'device_id': self.__device_id,
            'do_copy_in_default_stream': 'false',
        }


def get_provider(device: str, **kwargs) -> Provider:
    """Provider factory method"""
    if device not in _PROVIDERS:
        raise NotImplementedError(f'{device} type ONNX provider is not yet implemented')

    return _PROVIDERS[device](**kwargs)
