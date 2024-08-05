"""Developing Codes for diffuserbm."""
import os

from diffuserbm import DiffuserBMConfig
from diffuserbm.bench import txt2img
from diffuserbm import utils


if __name__ == '__main__':
    proj_home = os.environ.get('DIFFUSERBM_HOME', '.')
    config_path = os.environ.get('DIFFUSERBM_CONFIG', 'configs/diffuserbm.yaml')

    utils.build_proj_dir(proj_home, config_path)

    config = DiffuserBMConfig(proj_home, yaml_config=config_path)

    # Text2Image benchmark
    txt2img.bench(config=config)
