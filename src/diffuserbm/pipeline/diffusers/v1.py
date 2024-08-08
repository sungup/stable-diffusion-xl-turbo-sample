"""Module providing pipeline functionality for the Stable Diffusion in benchmark"""
import torch

from diffusers import StableDiffusionPipeline

from diffuserbm.pipeline.core import BenchmarkPipeline


class StableDiffusionBenchmarkPipeline(BenchmarkPipeline, type='v1', engine='diffusers'):
    """Class providing pipeline functionality for the Stable Diffusion XL in benchmark"""
    def __init__(self, checkpoint, device, **_kwargs):
        pipeline = StableDiffusionPipeline.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16,
            local_files_only=True,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )

        pipeline.to(device)

        if device == 'cuda':
            pipeline.enable_xformers_memory_efficient_attention()
            pipeline.enable_sequential_cpu_offload()

        # TODO check mps need enable_attention_slicing
        # elif device == 'mps':
        #     self.pipeline.enable_attention_slicing()

        super().__init__(checkpoint, device, pipeline)
