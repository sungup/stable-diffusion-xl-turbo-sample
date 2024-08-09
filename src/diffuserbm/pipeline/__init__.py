"""Module providing functionality for diffuserbm pipeline."""
from diffuserbm import DiffuserBMConfig
from diffuserbm.pipeline import core

import diffuserbm.pipeline.diffusers
import diffuserbm.pipeline.onnx

DEVICE_TYPES = ['cuda', 'cpu', 'npu', 'mps', 'rocm', 'dml']


def make(config: DiffuserBMConfig, pipeline: str, checkpoint: str, device: str, **kwargs):
    """Make pipeline for the stable diffusion model"""
    checkpoint = config.checkpoint(checkpoint)

    if pipeline not in core.BenchmarkPipeline.PIPELINES:
        raise ValueError(f'unsupported pipeline type: {pipeline}')

    if core.BenchmarkPipeline.PIPELINES[pipeline].type != checkpoint.type:
        raise RuntimeError('pipeline type and checkpoint type mismatch')

    return core.make_pipeline(pipeline, checkpoint.path, device, **kwargs)


def add_arguments(parser, config: DiffuserBMConfig):
    """Add arguments for diffuserbm pipeline."""
    parser.add_argument('--device', type=str, choices=DEVICE_TYPES,
                        help='Inference target device type')
    parser.add_argument('--pipeline', type=str, choices=core.supported_pipelines(),
                        help='benchmark pipeline name')
    parser.add_argument('--checkpoint', type=str, choices=config.supported_checkpoints,
                        help='checkpoint file of the target stable-diffusion pipeline')
    parser.add_argument('--onnx.device_id', type=int, default=0,
                        help='Target GPU device_id for ONNX (Windows only). '
                             'Please check running `gpus` sub-command')
    parser.add_argument('--onnx.int8', action="store_true",
                        help='Use int8 quantized model for ONNX')
