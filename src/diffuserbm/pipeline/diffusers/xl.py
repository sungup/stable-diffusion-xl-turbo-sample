"""Module providing pipeline functionality for the Stable Diffusion XL in benchmark"""
import torch

from diffusers import StableDiffusionXLPipeline
from numpy import ndarray

from diffuserbm.pipeline.core import BenchmarkPipeline


class StableDiffusionXLBenchmarkPipeline(BenchmarkPipeline, type='xl', name='diffusers.xl'):
    """Class providing pipeline functionality for the Stable Diffusion XL in benchmark"""
    def __init__(self, checkpoint, device, **_):
        super().__init__(checkpoint, device)

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.float16,
            # TODO this value will be uncommented after update variant download funtionality
            # variant='fp16',
            local_files_only=True,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )

        self.pipeline.to(device)

        if device == 'cuda':
            self.pipeline.enable_xformers_memory_efficient_attention()
            self.pipeline.enable_sequential_cpu_offload()

        # TODO check mps need enable_attention_slicing
        # elif device == 'mps':
        #     self.pipeline.enable_attention_slicing()

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
