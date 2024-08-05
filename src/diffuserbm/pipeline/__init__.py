"""Module providing functionality for diffuserbm pipeline."""
from diffuserbm import DiffuserBMConfig
from diffuserbm.pipeline import core

import diffuserbm.pipeline.general
import diffuserbm.pipeline.onnx


def make(config: DiffuserBMConfig, pipeline, checkpoint, device, **_):
    """Make pipeline for the stable diffusion model"""
    checkpoint = config.checkpoint(checkpoint)

    if pipeline not in core.BenchmarkPipeline.PIPELINES:
        raise ValueError(f'unsupported pipeline type: {pipeline}')

    if core.BenchmarkPipeline.PIPELINES[pipeline].type != checkpoint.type:
        raise RuntimeError('pipeline type and checkpoint type mismatch')

    return core.make_pipeline(pipeline, checkpoint.path, device)


def add_arguments(parser, config: DiffuserBMConfig):
    """Add arguments for diffuserbm pipeline."""
    parser.add_argument('--pipeline', type=str, choices=core.supported_pipelines(),
                        help='benchmark pipeline name')
    parser.add_argument('--checkpoint', type=str, choices=config.supported_checkpoints,
                        help='checkpoint file of the target stable-diffusion pipeline')
