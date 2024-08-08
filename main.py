"""Developing Codes for diffuserbm."""
import argparse
import os

from diffuserbm import utils, DiffuserBMConfig
from diffuserbm.bench import txt2img
from diffuserbm.utils import gpu


def parse_args(config: DiffuserBMConfig) -> any:
    parser = argparse.ArgumentParser(
        description='Options for Stable Diffusion Benchmarks',
        conflict_handler='resolve',
    )

    subparsers = parser.add_subparsers(dest='command')

    modules = {}

    for module in [txt2img, gpu]:
        name = module.command_name()
        help_str = module.command_help()

        modules[name] = module
        module.add_arguments(subparsers.add_parser(name, help=help_str), config)

    args = parser.parse_args()

    module = modules[args.command]

    module.post_arguments(args)

    return module, args


def main():
    proj_home = os.environ.get('DIFFUSERBM_HOME', '.')
    config_path = os.environ.get('DIFFUSERBM_CONFIG', 'configs/diffuserbm.yaml')

    utils.build_proj_dir(proj_home, config_path)

    config = DiffuserBMConfig(proj_home, yaml_config=config_path)

    module, args = parse_args(config=config)

    module.run(args=args, config=config)


if __name__ == '__main__':
    main()
