"""Module providing core functionality for upscaler of diffuserbm."""


class UpScaler:
    """Base class for Upscalers of diffuserbm."""
    UPSCALERS = {}

    def __init_subclass__(cls, **kwargs):
        UpScaler.UPSCALERS[kwargs['name']] = cls

    def __init__(self, scale, **_):
        self.__scale_rate = scale

    @property
    def scale(self):
        """Property about scale rate of the upscaler."""
        return self.__scale_rate

    def __call__(self, np):
        return np


# basic upscaler
UpScaler.UPSCALERS['none'] = UpScaler


def make_upscaler(upscaler, upscale_rate, upscaler_path, device):
    """factory method for upscaler of diffuserbm."""
    if upscaler not in UpScaler.UPSCALERS:
        upscaler = 'none'
        upscale_rate = 1

    return UpScaler.UPSCALERS[upscaler](model_path=upscaler_path, scale=upscale_rate, device=device)


def supported_upscaler():
    """Return list of supported upscalers."""
    return UpScaler.UPSCALERS.keys()
