"""Private module for GPU on Windows environment."""
import wmi

from diffuserbm.utils._gpu import GPUInfo


def _parse_id(device_id: str) -> int:
    return int(device_id.removeprefix('VideoController')) - 1


def available_gpus() -> list:
    """Returns a list of available GPUs on Windows system."""
    w = wmi.WMI(namespace=r'root\CIMV2')
    return [GPUInfo(
                device_id=_parse_id(gpu.DeviceID),
                name=gpu.Name,
                system_id=gpu.DeviceID,
                processor=gpu.VideoProcessor,
            )
            for gpu in w.Win32_VideoController()]
