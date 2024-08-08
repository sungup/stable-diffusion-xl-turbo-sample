"""Private module for GPU on Non-Windows environments."""


def available_gpus() -> list:
    """Returns a list of available GPUs on non-Windows system."""
    raise NotImplementedError('not yet implemented to the Non-Windows systems')
