"""Module providing functions for quantization."""
import os
import shutil

import onnx

from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.transformers import float16


def to_int8(src_model: str, dst_model: str):
    """quantize to int8 (answer from ChatGPT)"""
    quantize_dynamic(src_model, dst_model, weight_type=QuantType.QInt8)


def to_fp16(src_model: str, dst_model: str):
    """reference code from Microsoft"""
    onnx.save(float16.convert_float_to_float16(onnx.load(src_model)), dst_model)

    raise NotImplementedError("not implemented yet to_fp16")


def quantize(src: str, dst: str, quant_to):
    """Quantize model to selected level"""
    if os.path.isdir(dst):
        return

    for src_dir, dirs, files in os.walk(src):
        dst_dir = src_dir.replace(src, dst)

        for directory in dirs:
            os.makedirs(os.path.join(dst_dir, directory), exist_ok=True)

        for file in files:
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)

            if file.endswith('.onnx'):
                quant_to(src_file, dst_file)
            elif not file.endswith('.onnx_data'):
                shutil.copyfile(src_file, dst_file)
