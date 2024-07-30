"""Module providing utlities for benchmark"""

import git
import os
import requests


def download(url, path):
    """Download a file from url and save it to path."""
    if os.path.exists(path):
        return

    response = requests.get(url, stream=True)
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024*1024):
            f.write(chunk)


def clone(git_url, path, branch='main'):
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

