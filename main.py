import requests
import yaml

import diffuserbm.bench.txt2img as txt2img


"""
Under developing codes
"""


def download_checkpoint(source, dest):
    response = requests.get(source, stream=True)
    with open(dest, mode='wb') as f:
        for chunk in response.iter_content(chunk_size=10*1024):
            f.write(chunk)


def download():
    with open('configs/diffuserbm.yaml', 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    print(configs['checkpoints'])
    for _, checkpoints in configs['checkpoints'].items():
        for checkpoint in checkpoints:
            download_checkpoint(checkpoint['source'], checkpoint['dest'])


if __name__ == '__main__':
    download()
    # Text2Image benchmark
    #txt2img.bench()
