"""Module providing core functionality for diffuserbm pipeline."""

import os

from collections import namedtuple
from numpy import ndarray

_StableDiffusionConf = namedtuple('_StableDiffusionConf', ['model_config', 'submodel_config'])

# TODO need to change configurable
_STABLE_DIFFUSION_CONFIG = {
    "v1": _StableDiffusionConf(os.path.join("checkpoints/configs", "v1-inference.yaml"), os.path.join("models", "stable-diffusion-v1-5")),
    "v2": _StableDiffusionConf(os.path.join("checkpoints/configs", "v2-inference.yaml"), os.path.join("models", "stable-diffusion-2-1")),
    "xl": _StableDiffusionConf(os.path.join("checkpoints/configs", "xl-base-inference.yaml"), os.path.join("models", "sdxl-turbo")),
}

_StableDiffusionPipelineInfo = namedtuple('_StableDiffusionPipelineInfo', ['type', 'cls'])


class BenchmarkPipeline:
    PIPELINES = {}

    def __init_subclass__(cls, **kwargs):
        BenchmarkPipeline.PIPELINES[kwargs['name']] = _StableDiffusionPipelineInfo(
            type=kwargs['type'],
            cls=cls,
        )

    def __init__(self, checkpoint, device, **_):
        self.checkpoint = checkpoint
        self.device = device

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
        raise RuntimeError('please define inherited function of BenchmarkPipeline.__call__')


def make_pipeline(pipeline, checkpoint, device):
    if pipeline not in BenchmarkPipeline.PIPELINES:
        raise ValueError('unsupported pipeline type')

    return BenchmarkPipeline.PIPELINES[pipeline].cls(checkpoint=checkpoint, device=device)


def supported_pipelines():
    return BenchmarkPipeline.PIPELINES.keys()
