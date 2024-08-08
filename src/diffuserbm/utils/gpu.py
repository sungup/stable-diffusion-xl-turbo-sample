"""Module providing list of available GPUs to run Stable Diffusion"""
import platform

from diffuserbm import DiffuserBMConfig


if platform.system() == 'Windows':
    from diffuserbm.utils import _gpu_windows
    available_gpus = _gpu_windows.available_gpus
else:
    from diffuserbm.utils import _gpu_common
    available_gpus = _gpu_common.available_gpus


def command_help() -> str:
    """Return a help string for GPU list-up sub-command."""
    return 'Find all available GPUs for benchmark on ONNX Runtime DirectML'


def command_name() -> str:
    """Returns name of GPU list-up sub-command."""
    return 'gpus'


def add_arguments(_parser, _config: DiffuserBMConfig):
    """Adds arguments for GPU list-up sub-command."""


def post_arguments(_args):
    """Check arguments after parsing."""


def run(_args, _config: DiffuserBMConfig):
    """Print GPU lists for the ONNX DirectML"""
    print('List of supported GPUs:')
    for gpu in available_gpus():
        print(f' - device_id: {gpu.device_id} - {gpu.name} ({gpu.system_id})')
