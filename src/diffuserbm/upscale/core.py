"""Module providing core functionality for upscaler of diffuserbm."""

_UPSCALERS = {}


class UpScaler:
    def __init_subclass__(cls, **kwargs):
        _UPSCALERS[kwargs['name']] = cls

    def __init__(self, scale, **_):
        self.__scale_rate = scale
        pass

    @property
    def scale(self):
        return self.__scale_rate

    def __call__(self, np):
        return np


# basic upscaler
_UPSCALERS['none'] = UpScaler


def make_upscaler(upscaler, upscale_rate, upscaler_path, device):
    if upscaler not in _UPSCALERS.keys():
        upscaler = 'none'
        upscale_rate = 1

    return _UPSCALERS[upscaler](model_path=upscaler_path, scale=upscale_rate, device=device)


def supported_upscaler():
    return _UPSCALERS.keys()
