""" Module providing download functionality for stable diffusion models """
import os

import git
import requests
import torch

from diffusers import DiffusionPipeline


def get(url: str, path: str):
    """Download a file from url and save it to path."""
    if os.path.exists(path):
        return

    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    response = requests.get(url, stream=True, timeout=10)
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024*1024):
            f.write(chunk)


def clone(git_url: str, path: str, branch: str = 'main'):
    """Clone a repository from url and save it to path."""
    if os.path.exists(path):
        return

    git.Repo.clone_from(
        git_url, path,
        env={
            'GIT_LFS_SKIP_SMUDGE': '1',
        },
        multi_options=[
            '--single-branch',
            '--depth', '1',
        ],
        branch=branch,
    )


def hf_get(hf_id: str, path: str, variant: str = None, dtype: torch.dtype = torch.float32):
    """Download checkpoint files from huggingface repository using diffusers module."""
    if os.path.isdir(path):
        return

    DiffusionPipeline.from_pretrained(
        hf_id, variant=variant, torch_dtype=dtype,
    ).save_pretrained(path)
