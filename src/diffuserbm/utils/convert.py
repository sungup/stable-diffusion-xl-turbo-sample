"""Module providing conversion functions."""
import os

from optimum.onnxruntime import ORTStableDiffusionPipeline, ORTStableDiffusionXLPipeline


def _to_onnx(cls, src: str, dst: str):
    if os.path.isdir(dst):
        return

    cls.from_pretrained(
        src, local_files_only=True, export=True, provider='CPUExecutionProvider'
    ).save_pretrained(dst)


def _diffuser_to_onnx(src: str, dst: str):
    _to_onnx(ORTStableDiffusionPipeline, src, dst)


def _diffuser_xl_to_onnx(src: str, dst: str):
    _to_onnx(ORTStableDiffusionXLPipeline, src, dst)


def diffuser_to_onnx(type: str, src: str, dst: str):
    """Convert Stable Diffusion checkpoints to ONNX model."""
    if type == 'xl':
        _diffuser_xl_to_onnx(src, dst)
    else:
        _diffuser_to_onnx(src, dst)
