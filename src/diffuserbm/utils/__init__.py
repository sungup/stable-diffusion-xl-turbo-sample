"""Module providing utilities for benchmark"""
import os

import yaml

from diffuserbm.utils import download


def build_proj_dir(proj_path: str, config_path: str):
    """Build project directory from config on the project path."""
    with open(config_path, 'r', encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    if not os.path.isdir(proj_path):
        os.makedirs(proj_path)

    for _, config in configs['configs'].items():
        download.get(config['source'], os.path.join(proj_path, config['dest']))

    for _, checkpoints in configs['checkpoints'].items():
        for checkpoint in checkpoints:
            download.hf_get(checkpoint['id'], os.path.join(proj_path, checkpoint['dest']))

    for _, upscalers in configs['upscaler'].items():
        for upscaler in upscalers:
            download.get(upscaler['source'], os.path.join(proj_path, upscaler['dest']))
