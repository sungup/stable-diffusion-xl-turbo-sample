"""Module providing pipeline functionality for the Stable Diffusion XL on ONNX runtime."""
from optimum.onnxruntime import ORTStableDiffusionXLPipeline

from diffuserbm.pipeline.core import BenchmarkPipeline
from diffuserbm.pipeline.onnx.provider import get_provider


class StableDiffusionXLBenchmarkPipeline(BenchmarkPipeline, type='xl', name='onnx.xl'):
    """Class providing pipeline functionality for the Stable Diffusion XL on ONNX runtime."""
    def __init__(self, checkpoint, device, **_):
        self.provider = get_provider(device)

        pipeline = ORTStableDiffusionXLPipeline.from_single_file(
            checkpoint,
            local_files_only=True,
            provider=self.provider.name,
            provider_options=self.provider.options,
            export=True,
        )

        super().__init__(checkpoint, device, pipeline)

        raise NotImplementedError('Stable Diffusion XL is not yet implemented.')
