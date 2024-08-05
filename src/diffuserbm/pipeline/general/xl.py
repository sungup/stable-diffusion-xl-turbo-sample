"""Module providing pipeline functionality for the Stable Diffusion XL in benchmark"""

import torch

from diffusers import StableDiffusionXLPipeline
from numpy import ndarray

from diffuserbm.pipeline.core import BenchmarkPipeline


class StableDiffusionXLBenchmarkPipeline(BenchmarkPipeline, type='xl', name='diffusers.sdxl'):
    def __init__(self, checkpoint, device, **_):
        super().__init__(checkpoint, device)

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.checkpoint,
            # TODO this value will be uncommented after update variant download funtionality
            #torch_dtype=torch.float16,
            #variant='fp16',
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
