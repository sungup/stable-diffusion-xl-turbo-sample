import torch


_PROVIDERS = dict()


class Provider:
    __NAME = None
    __OPTIONS = None

    def __init_subclass__(cls, **kwargs):
        _PROVIDERS[kwargs['device']] = cls
        cls.__NAME = kwargs['provider']
        cls.__OPTIONS = kwargs.get('options', None)

    @property
    def name(self) -> str:
        return self.__NAME

    @property
    def options(self) -> any:
        return self.__OPTIONS


class CPUProvider(Provider, device='cpu', provider='CPUExecutionProvider'):
    pass


class CUDAProvider(Provider, device='cuda', provider='CUDAExecutionProvider'):
    @property
    def options(self) -> any:
        return {
            "device_id": torch.cuda.current_device(),
            "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)
        }


def get_provider(device: str) -> Provider:
    return _PROVIDERS[device]()

