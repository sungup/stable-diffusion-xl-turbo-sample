"""Module providing pipeline functionality for the Stable Diffusion on ONNX runtime."""
from optimum.onnxruntime import ORTStableDiffusionPipeline

from diffuserbm.pipeline.core import BenchmarkPipeline
from diffuserbm.pipeline.onnx.provider import get_provider


class StableDiffusionBenchmarkPipeline(BenchmarkPipeline, type='v1', name='onnx.v1'):
    """Class providing pipeline functionality for the Stable Diffusion on ONNX runtime."""
    def __init__(self, checkpoint, device, **_):
        self.provider = get_provider(device)

        pipeline = ORTStableDiffusionPipeline.from_pretrained(
            checkpoint + '.onnx',
            local_files_only=True,
            provider=self.provider.name,
            provider_options=self.provider.options,
            use_io_binding=True,
        )

        super().__init__(checkpoint, device, pipeline)
