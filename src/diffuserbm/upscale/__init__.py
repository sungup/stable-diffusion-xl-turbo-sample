"""Module providing functionality for upscaling images."""

import diffuserbm.upscale.real_esrgan

from diffuserbm import DiffuserBMConfig
from diffuserbm.upscale import core


def make(config: DiffuserBMConfig, upscaler, device, **_):
    """Make upscaler object."""
    if upscaler == 'none':
        u_rate = 1
        u_path = ''
        u_type = 'none'
    else:
        upscale_conf = config.upscaler(upscaler)
        u_rate = upscale_conf.rate
        u_path = upscale_conf.path
        u_type = upscale_conf.type

    return core.make_upscaler(u_type, u_rate, u_path, device)


def add_arguments(parser, config: DiffuserBMConfig):
    """Add arguments for diffuserbm upscaler."""
    parser.add_argument('--upscaler', type=str, default='none',
                        choices=['none']+config.supported_upscalers,
                        help='Model type name of upscaler')
