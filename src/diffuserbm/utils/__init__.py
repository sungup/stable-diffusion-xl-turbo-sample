"""Module providing utilities for benchmark"""
import os
import shutil

import yaml

from diffuserbm.utils import download, convert, quantize


def build_proj_dir(proj_path: str, config_path: str):
    """Build project directory from config on the project path."""
    with open(config_path, 'r', encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    if not os.path.isdir(proj_path):
        os.makedirs(proj_path)

    # 1. Download Stable Diffusion config
    for _, config in configs['configs'].items():
        download.get(config['source'], os.path.join(proj_path, config['dest']))

    # 2. Download stable diffusion checkpoints
    for checkpoint_type, checkpoints in configs['checkpoints'].items():
        for checkpoint in checkpoints:
            checkpoint_path = os.path.join(proj_path, checkpoint['dest'])
            download.hf_get(checkpoint['id'], checkpoint_path)

            # Convert to ONNX
            if 'use_onnx' in checkpoint:
                onnx_fp32_path = checkpoint_path + '.fp32.onnx'
                onnx_int8_path = checkpoint_path + '.int8.onnx'

                # 1. first convert to fp32 onnx
                convert.diffuser_to_onnx(checkpoint_type, checkpoint_path, onnx_fp32_path)

                # 2. Quantize ONNX model to int8
                if bool(checkpoint['use_onnx'].get('int8', 'false')):
                    quantize.quantize(onnx_fp32_path, onnx_int8_path, quantize.to_int8)

                # 3. If fp32 is not flagged, remove fp32 ONNX model directory
                if not bool(checkpoint['use_onnx'].get('fp32', 'false')):
                    shutil.rmtree(onnx_fp32_path)

    # 3. Download upscalers
    for _, upscalers in configs['upscaler'].items():
        for upscaler in upscalers:
            download.get(upscaler['source'], os.path.join(proj_path, upscaler['dest']))
