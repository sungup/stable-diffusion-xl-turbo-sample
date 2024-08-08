"""Module providing common functionality for GPU manipulation."""
from collections import namedtuple

GPUInfo = namedtuple('gpuInfo', ['device_id', 'system_id', 'name', 'processor'])
