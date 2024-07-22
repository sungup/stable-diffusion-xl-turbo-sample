import requests
import yaml

from diffuserbm.utils import download
import diffuserbm.bench.txt2img as txt2img


"""
Under developing codes
"""


def download_checkpoint():
    with open('configs/diffuserbm.yaml', 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    print(configs['checkpoints'])
    for _, checkpoints in configs['checkpoints'].items():
        for checkpoint in checkpoints:
            download(checkpoint['source'], checkpoint['dest'])


if __name__ == '__main__':
    download()
    # Text2Image benchmark
    #txt2img.bench()
