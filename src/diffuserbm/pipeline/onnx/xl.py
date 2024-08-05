import torch

from diffuserbm.pipeline.core import BenchmarkPipeline
from diffuserbm.pipeline.onnx.provider import get_provider
from numpy import ndarray
from optimum.onnxruntime import ORTStableDiffusionXLPipeline


class StableDiffusionXLBenchmarkPipeline(BenchmarkPipeline, type='xl', name='onnx.sdxl'):
    def __init__(self, checkpoint, device, **_):
        super().__init__(checkpoint, device)

        self.provider = get_provider(device)

        self.pipeline = ORTStableDiffusionXLPipeline.from_single_file(
            checkpoint,
            config=StableDiffusionXLBenchmarkPipeline.SUB_MODEL_CONFIG,
            local_files_only=True,
            provider=self.provider.name,
            provider_options=self.provider.options,
            export=True,
            torch_dtype=torch.float16,
            variant='fp16',
            use_safetensors=True,
            low_cpu_mem_usage=True,
            original_config=StableDiffusionXLBenchmarkPipeline.MODEL_CONFIG,
        )

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
            negative_prompt=prompt,
            generator=rand_gen,
            num_inference_steps=denoising_steps,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            output_type='np'
        ).images

