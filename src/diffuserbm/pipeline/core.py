"""Module providing core functionality for diffuserbm pipeline."""
from collections import namedtuple
from numpy import ndarray


_StableDiffusionPipelineInfo = namedtuple('_StableDiffusionPipelineInfo', ['type', 'cls'])


class BenchmarkPipeline:
    """Interface class for running diffuserbm pipeline."""
    PIPELINES = {}

    def __init_subclass__(cls, **kwargs):
        BenchmarkPipeline.PIPELINES[kwargs['name']] = _StableDiffusionPipelineInfo(
            type=kwargs['type'],
            cls=cls,
        )

    def __init__(self, checkpoint, device, pipeline, **_):
        self.__checkpoint = checkpoint
        self.__device = device
        self.__pipeline = pipeline

    def __call__(self, prompt, width, height, **kwargs) -> ndarray:
        """
        Functor of benchmark pipeline to generate image binary

        :param prompt: input prompt to generate image
        :param width: width of image
        :param height: height of image
        :param negative: negative prompt to avoid features not wanted in the generated image
        :param rand_gen: array of random generators for each batch
        :param denoising_steps: number of inference step to denoising
        :param guidance_scale: scale of guidance for the generating image
        :return: ndarray type image
        """
        negative = kwargs.get('negative', '')
        rand_gen = kwargs.get('rand_gen', None)
        denoising_steps = kwargs.get('denoising_steps', 8)
        guidance_scale = kwargs.get('guidance_scale', 0.0)

        return self.__pipeline(
            prompt=prompt,
            negative_prompt=negative,
            generator=rand_gen,
            num_inference_steps=denoising_steps,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            output_type='np'
        ).images

    @property
    def checkpoint(self):
        """Property function to get the checkpoint path."""
        return self.__checkpoint

    @property
    def device(self):
        """Property function to get the device type for benchmark pipeline."""
        return self.__device


def make_pipeline(pipeline, checkpoint, device):
    """Pipeline builder for target checkpoint."""
    if pipeline not in BenchmarkPipeline.PIPELINES:
        raise ValueError('unsupported pipeline type')

    return BenchmarkPipeline.PIPELINES[pipeline].cls(checkpoint=checkpoint, device=device)


def supported_pipelines():
    """Getter function for supported pipelines."""
    return BenchmarkPipeline.PIPELINES.keys()
