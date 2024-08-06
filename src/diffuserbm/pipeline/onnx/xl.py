"""Module providing pipeline functionality for the Stable Diffusion XL on ONNX runtime."""
from numpy import ndarray
from optimum.onnxruntime import ORTStableDiffusionXLPipeline

from diffuserbm.pipeline.core import BenchmarkPipeline
from diffuserbm.pipeline.onnx.provider import get_provider


class StableDiffusionXLBenchmarkPipeline(BenchmarkPipeline, type='xl', name='onnx.xl'):
    """Class providing pipeline functionality for the Stable Diffusion XL on ONNX runtime."""
    def __init__(self, checkpoint, device, **_):
        super().__init__(checkpoint, device)

        self.provider = get_provider(device)

        self.pipeline = ORTStableDiffusionXLPipeline.from_single_file(
            checkpoint,
            local_files_only=True,
            provider=self.provider.name,
            provider_options=self.provider.options,
            export=True,
        )

        raise NotImplementedError('Stable Diffusion XL is not yet implemented.')

    def __call__(self, prompt, width, height, **kwargs) -> ndarray:
        """
        Functor of benchmark pipeline to generate image binary

        :param prompt: input prompt to generate image
        :param width: width of image
        :param height: height of image
        :param negative: negative prompt to avoid features not wanted in the generated image
        :param rand_gen: array of random generators for each batch
        :param denoising_steps: number of inference step to denoising, default 8 for SDXL
        :param guidance_scale: scale of guidance for the generating image, default 0.0 for SDXL
        :return: ndarray type image
        """
        negative = kwargs.get('negative', '')
        rand_gen = kwargs.get('rand_gen', None)
        denoising_steps = kwargs.get('denoising_steps', 8)
        guidance_scale = kwargs.get('guidance_scale', 0.0)

        return self.pipeline(
            prompt=prompt,
            negative_prompt=negative,
            generator=rand_gen,
            num_inference_steps=denoising_steps,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            output_type='np'
        ).images
