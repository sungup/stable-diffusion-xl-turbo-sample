import os
import requests
import yaml

import diffuserbm.bench.txt2img as txt2img

from git import Repo


"""
Under developing codes
"""


def download_checkpoint(source, dest):
    if os.path.exists(dest):
        return

    response = requests.get(source, stream=True)
    with open(dest, mode='wb') as f:
        for chunk in response.iter_content(chunk_size=10*1024):
            f.write(chunk)


def download():
    with open('configs/diffuserbm.yaml', 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    for _, model in configs['configs']['model'].items():
        download_checkpoint(model['source'], model['dest'])

    for _, model in configs['configs']['sub_model'].items():
        Repo.clone_from(
            model['source'], model['dest'],
            env={
                'GIT_LFS_SKIP_SMUDGE': '1',
            },
            multi_options=[
                '--single-branch',
                '--depth', '1',
            ],
            branch='main',
        )

    for _, checkpoints in configs['checkpoints'].items():
        for checkpoint in checkpoints:
            download_checkpoint(checkpoint['source'], checkpoint['dest'])


if __name__ == '__main__':
    download()
    # Text2Image benchmark
    #txt2img.bench()
