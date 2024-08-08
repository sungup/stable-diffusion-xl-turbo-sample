"""Module providing core functionality for diffuserbm pipeline."""
from collections import namedtuple
from numpy import ndarray


_StableDiffusionPipelineInfo = namedtuple('_StableDiffusionPipelineInfo', ['engine', 'type', 'cls'])


class BenchmarkPipeline:
    """Interface class for running diffuserbm pipeline."""
    PIPELINES = {}
    ENGINE = ""

    def __init_subclass__(cls, **kwargs):
        name = f'{kwargs["engine"]}.{kwargs["type"]}'
        BenchmarkPipeline.PIPELINES[name] = _StableDiffusionPipelineInfo(
            engine=kwargs['engine'],
            type=kwargs['type'],
            cls=cls,
        )

        cls.ENGINE = kwargs['engine']

    def __init__(self, checkpoint, device, pipeline, **_):
        self.__checkpoint = checkpoint
        self.__device = device
        self.__pipeline = pipeline

    def __call__(self, prompt, width, height, **kwargs) -> ndarray:
        """
        Functor of benchmark pipeline to generate image binary

        Args:
          prompt (str): input prompt to generate image
          width (int): width of image
          height (int): height of image

        Kwargs:
            negative (str): negative prompt to avoid features not wanted in the generated image
            denoising_steps (int): number of inference step to denoising
            guidance_scale (float): scale of guidance for the generating image

        """
        negative = kwargs.get('negative', '')
        denoising_steps = kwargs.get('denoising_steps', 8)
        guidance_scale = kwargs.get('guidance_scale', 0.0)

        return self.__pipeline(
            prompt=prompt,
            negative_prompt=negative,
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

    @property
    def engine(self):
        """Property function to get the inference engine."""
        return self.ENGINE


def make_pipeline(pipeline, checkpoint, device, **kwargs):
    """Pipeline builder for target checkpoint."""
    if pipeline not in BenchmarkPipeline.PIPELINES:
        raise ValueError('unsupported pipeline type')

    return BenchmarkPipeline.PIPELINES[pipeline].cls(checkpoint=checkpoint, device=device, **kwargs)


def supported_pipelines():
    """Getter function for supported pipelines."""
    return BenchmarkPipeline.PIPELINES.keys()
