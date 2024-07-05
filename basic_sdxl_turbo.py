import argparse
import os
import random
import torch

import numpy as np

from collections import namedtuple
from PIL import Image

from diffusers import StableDiffusionXLPipeline
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


DEFAULT_DEVICE = "cuda"
DEFAULT_CHECKPOINT = "./checkpoints/sd_xl_turbo_1.0_fp16.safetensors"
DEFAULT_BATCH_SIZE = 1
DEFAULT_BATCH_COUNT = 1 #25
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 512
DEFAULT_RESULT = "result.csv"
DEFAULT_OUTPUT = "output"
DEFAULT_UPSCALE = 4
DEFAULT_DENOISING_STEPS = 4
DEFAULT_GUIDANCE_SCALE = 0.0

DEFAULT_PROMPT = "cherry blossom, bonsai, Japanese style landscape, high resolution, 8k, lush green in the background"
DEFAULT_NEGATIVE_PROMPT = "Low quality, Low resolution, disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w"

UPSCALER_PATH = "upscaler/Real-ESRGAN/RealESRGAN_x{scale}plus.pth"

_StableDiffusionConf = namedtuple('_StableDiffusionConf', ['model_config', 'submodel_config'])

STABLE_DIFFUSION_CONFIG = {
        "v1": _StableDiffusionConf("configs/v1-inference.yaml", ""),
        "v2": _StableDiffusionConf("configs/v2-inference-v.yaml", ""),
        "xl": _StableDiffusionConf("configs/sd_xl_base.yaml", "sdxl-turbo"),
        "xl_refiner": _StableDiffusionConf("configs/sd_xl_refiner.yaml", "")
}


def parse_args():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion XL Turbo Simple Benchmark", conflict_handler='resolve')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, choices=["cuda", "cpu", "npu"], help="Inference target device")
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help="Checkpoint file of SDXL Turbo")
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help="Number of images to generate in a sequence, one  at a time")
    parser.add_argument('--iteration', type=int, default=DEFAULT_BATCH_COUNT, help="Number of repeat prompt")
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT, help="Height of image to generate (must be multiple of 8")
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH, help="Width of image to generate (must be multiple of 8")
    parser.add_argument('--result', type=str, default=DEFAULT_RESULT, help="Benchmark result file path")
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT, help="A directory to save images")
    parser.add_argument('--upscale', type=int, default=DEFAULT_UPSCALE, choices=[1, 2, 4], help="Image upscale rate to generate from FHD to QHD images")
    parser.add_argument('--denoising-steps', type=int, default=DEFAULT_DENOISING_STEPS, help="Number of denoising steps")
    parser.add_argument('--guidance-scale', type=float, default=DEFAULT_GUIDANCE_SCALE, help="Number of guidance scale")
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT, help="prompt to generate image")
    parser.add_argument('--negative-prompt', type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt to aboid from image")

    args = parser.parse_args()

    if args.height < 500:
        raise Exception("height cannot smaller then 500")

    if args.width < 500:
        raise Exception("width cannot smaller than 500")

    return args


class UpScaler:
    def scale(self):
        return 1

    def __call__(self, ndarray):
        return ndarray


class RealESRGANUpScaler(UpScaler):
    def __init__(self, upscale, device):
        self.pipeline = RealESRGANer(
                scale=upscale,
                model_path=UPSCALER_PATH.format(scale=upscale),
                device=device,
                dni_weight=None,
                model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=upscale),
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True,
        )

        self.scale_rate = upscale
        pass

    def scale(self):
        return self.scale_rate

    def __call__(self, ndarray):
        return self.pipeline.enhance(ndarray, outscale=self.scale_rate)[0]


class SDXLTurboPipeline:
    def __init__(self, checkpoint, upscale, result, output, batch_size, batch_count, device):
        configs = STABLE_DIFFUSION_CONFIG["xl"]
        self.pipeline = StableDiffusionXLPipeline.from_single_file(
            checkpoint,
            torch_dtype=torch.float16,
            variant="fp16",
            local_files_only=True,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            original_config=configs.model_config,
            config=configs.submodel_config,
        )

        self.pipeline.to(device)
        self.pipeline.enable_xformers_memory_efficient_attention()
        self.pipeline.enable_sequential_cpu_offload()

        self.upscaler = RealESRGANUpScaler(upscale, device) if upscale > 1 else UpScaler()

        self.result = result
        self.output = output

        self.batch_size = batch_size
        self.batch_count = batch_count

        self.generators = [torch.Generator(device=device).manual_seed(int(random.randrange(1,9999))) for i in range(batch_size)]

    def __target_size__(self, width, height):
        return width*self.upscaler.scale(), height*self.upscaler.scale()

    def __generate__(self, prompt, negative_prompt, width, height, denoising_steps, guidance_scale):
        output = self.pipeline(
                prompt=self.batch_size * [prompt],
                negative_prompt=self.batch_size * [negative_prompt],
                generator=self.generators,
                num_inference_steps=denoising_steps,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                output_type="np"
        )

        return output.images

    def __upscale__(self, np_array):
        images = []
        for image_arr in np_array:
            images.append(Image.fromarray(self.upscaler(np.uint8(image_arr*255))))

        return images

    def __save_image__(self, images, prompt):
        basename = ''.join(set(['-'+prompt[idx].replace(' ','_')[:10] for idx in range(len(prompt))]))

        for i, image in enumerate(images):
            image.save(os.path.join(self.output, basename+'-'+str(i)+'-'+str(random.randint(1000,9999))+'.png'))

    def generate(self, prompt, negative_prompt, width, height, denoising_steps, guidance_scale):
        images = self.__generate__(prompt, negative_prompt, width, height, denoising_steps, guidance_scale)

        images = self.__upscale__(images)

        self.__save_image__(images, prompt)

    def run(
            self,
            iteration,
            prompt,
            negative_prompt,
            width,
            height,
            denoising_steps=DEFAULT_DENOISING_STEPS,
            guidance_scale=DEFAULT_GUIDANCE_SCALE
    ):
        for _ in range(iteration):
            self.generate(prompt, negative_prompt, width, height, denoising_steps, guidance_scale)


if __name__ == '__main__':
    args = parse_args()

    bench = SDXLTurboPipeline(
            args.checkpoint,
            args.upscale,
            args.result,
            args.output,
            args.batch_size,
            args.iteration,
            args.device
    )

    bench.run(
            args.iteration,
            args.prompt,
            args.negative_prompt,
            args.width,
            args.height,
            args.denoising_steps,
            args.guidance_scale
    )

