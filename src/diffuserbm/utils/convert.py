"""Module providing conversion functions."""
from optimum.onnxruntime import ORTStableDiffusionPipeline, ORTStableDiffusionXLPipeline


def _to_onnx(cls, src, dst):
    cls.from_pretrained(
        src, local_files_only=True, export=True, provider='CPUExecutionProvider'
    ).save_pretrained(dst)


def diffuser_to_onnx(src, dst):
    """Convert Stable Diffusion checkpoints to ONNX model."""
    _to_onnx(ORTStableDiffusionPipeline, src, dst)


def diffuser_xl_to_onnx(src, dst):
    """Convert Stable Diffusion XL checkpoints to ONNX model."""
    _to_onnx(ORTStableDiffusionXLPipeline, src, dst)
