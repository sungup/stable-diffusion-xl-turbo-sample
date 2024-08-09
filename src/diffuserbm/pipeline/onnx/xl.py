"""Module providing pipeline functionality for the Stable Diffusion XL on ONNX runtime."""
from optimum.onnxruntime import ORTStableDiffusionXLPipeline

from diffuserbm.pipeline.core import BenchmarkPipeline
from diffuserbm.pipeline.onnx.provider import get_provider


class StableDiffusionXLBenchmarkPipeline(BenchmarkPipeline, type='xl', engine='onnx'):
    """Class providing pipeline functionality for the Stable Diffusion XL on ONNX runtime."""
    def __init__(self, checkpoint, device, **kwargs):
        self.provider = get_provider(device, **kwargs)

        is_int8 = kwargs.get('onnx.int8', False)

        # TODO update checkpoint path of ONNX
        pipeline = ORTStableDiffusionXLPipeline.from_single_file(
            checkpoint + f'.{"int8" if is_int8 else "fp32"}.onnx',
            local_files_only=True,
            provider=self.provider.name,
            session_options=self.provider.session_options,
            provider_options=self.provider.options,
        )

        super().__init__(checkpoint, device, pipeline)
