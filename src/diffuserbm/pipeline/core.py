"""Module providing core functionality for diffuserbm pipeline."""

import os

from collections import namedtuple
from numpy import ndarray

_StableDiffusionConf = namedtuple('_StableDiffusionConf', ['model_config', 'submodel_config'])

# TODO need to change configurable
_STABLE_DIFFUSION_CONFIG = {
    "v1": _StableDiffusionConf(os.path.join("models", "v1-inference.yaml"), os.path.join("models", "stable-diffusion-v1-5")),
    "v2": _StableDiffusionConf(os.path.join("models", "v2-inference.yaml"), os.path.join("models", "stable-diffusion-2-1")),
    "xl": _StableDiffusionConf(os.path.join("models", "xl-base-inference.yaml"), os.path.join("models", "sdxl-turbo")),
}

_PIPELINES = {}


class BenchmarkPipeline:
    def __init_subclass__(cls, **kwargs):
        _PIPELINES[kwargs['name']] = cls

        config = _STABLE_DIFFUSION_CONFIG[kwargs['type']]
        cls.MODEL_CONFIG = config.model_config
        cls.SUB_MODEL_CONFIG = config.submodel_config

    def __init__(self, checkpoint, device, **_):
        pass

    def __call__(
            self,
            prompt,
            negative,
            rand_gen,
            width,
            height,
            denoising_steps,
            guidance_scale
    ) -> ndarray:
        raise Exception('please define inherited function of BenchmarkPipeline.__call__')


def make_pipeline(pipeline, checkpoint, device):
    if pipeline not in _PIPELINES.keys():
        raise ValueError('unsupported pipeline type')

    return _PIPELINES[pipeline](checkpoint=checkpoint, device=device)


def supported_pipelines():
    return _PIPELINES.keys()
