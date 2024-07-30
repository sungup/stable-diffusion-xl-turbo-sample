"""Module providing functionality for diffuserbm pipeline."""

import diffuserbm.pipeline.core as core
import diffuserbm.pipeline.general

from diffuserbm.pipeline import onnx

def make(pipeline, checkpoint, device, **kwargs):
    return core.make_pipeline(pipeline, checkpoint, device)


def add_arguments(parser):
    parser.add_argument('--pipeline', type=str, choices=core.supported_pipelines(),
                        help='benchmark pipeline name')
    parser.add_argument('--checkpoint', type=str,
                        help='checkpoint file of the target stable-diffusion pipeline')
