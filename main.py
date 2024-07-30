import yaml

import diffuserbm.utils as utils
import diffuserbm.bench.txt2img as txt2img


"""
Under developing codes
"""


def download():
    with open('configs/diffuserbm.yaml', 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    for _, model in configs['configs']['model'].items():
        utils.download(model['source'], model['dest'])

    for _, model in configs['configs']['sub_model'].items():
        utils.clone(model['source'], model['dest'])

    for _, checkpoints in configs['checkpoints'].items():
        for checkpoint in checkpoints:
            utils.download(checkpoint['source'], checkpoint['dest'])


if __name__ == '__main__':
    download()
    # Text2Image benchmark
    txt2img.bench()
