"""Module for diffuserbm core logics"""
import os

from collections import namedtuple

import yaml


Checkpoint = namedtuple("Checkpoint", ["config", "path", "type"])
Upscaler = namedtuple("Upscaler", ["rate", "path", "type"])


class DiffuserBMConfig:
    """Class for DiffuserBM configuration."""

    def __init__(self, proj_dir: str, yaml_config: str = 'configs/diffuserbm.yaml'):
        with open(yaml_config, 'r', encoding='utf8') as f:
            configs = yaml.load(f, Loader=yaml.SafeLoader)

        # build checkpoint configs
        self.__model_configs = {}
        for k, v in configs['configs'].items():
            self.__model_configs[k] = v['dest']

        # build checkpoint information
        self.__checkpoints = {}
        for k, checkpoints in configs['checkpoints'].items():
            for checkpoint in checkpoints:
                self.__checkpoints[checkpoint['id']] = Checkpoint(
                    config=self.__model_configs[k],
                    path=os.path.join(proj_dir, checkpoint['dest']),
                    type=k,
                )

        # build upscaler information
        self.__upscalers = {}
        for k, upscalers in configs['upscaler'].items():
            for upscaler in upscalers:
                name = os.path.splitext(os.path.basename(upscaler['dest']))[0]
                self.__upscalers[name] = Upscaler(
                    rate=int(upscaler['rate']),
                    path=os.path.join(proj_dir, upscaler['dest']),
                    type=k,
                )

    @property
    def supported_checkpoints(self) -> list:
        """Property to get the all supported checkpoints"""
        return list(self.__checkpoints.keys())

    @property
    def supported_upscalers(self) -> list:
        """Property to get the all supported upscalers"""
        return list(self.__upscalers)

    def checkpoint(self, hf_id) -> Checkpoint:
        """Get the local checkpoint and its config for the Hugging Face id"""
        return self.__checkpoints[hf_id]

    def upscaler(self, name) -> Upscaler:
        """Get the local upscaler"""
        return self.__upscalers[name]
