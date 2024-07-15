import argparse
import os
import random
import re
import time
import torch

import numpy as np

from collections import namedtuple, defaultdict
from PIL import Image

from diffusers import StableDiffusionXLPipeline
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


DEFAULT_DEVICE = "mps"
DEFAULT_CHECKPOINT = os.path.join("checkpoints", "sd_xl_turbo_1.0_fp16.safetensors")
DEFAULT_BATCH_SIZE = 4
DEFAULT_BATCH_COUNT = 5
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 512
DEFAULT_RESULT = "result.csv"
DEFAULT_OUTPUT = "output"
DEFAULT_UPSCALE = 4
DEFAULT_DENOISING_STEPS = 4
DEFAULT_GUIDANCE_SCALE = 0.0

DEFAULT_PROMPT = """
cherry blossom, bonsai, Japanese style landscape, high resolution, 8k, lush green in the background
""".split(' ')
DEFAULT_NEGATIVE_PROMPT = """
Low quality, Low resolution, disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w
""".split(' ')

UPSCALER_PATH = os.path.join("upscaler", "Real-ESRGAN", "RealESRGAN_x{scale}plus.pth")

_StableDiffusionConf = namedtuple('_StableDiffusionConf', ['model_config', 'submodel_config'])

STABLE_DIFFUSION_CONFIG = {
    "v1": _StableDiffusionConf(os.path.join("configs", "v1-inference.yaml"), ""),
    "v2": _StableDiffusionConf(os.path.join("configs", "v2-inference-v.yaml"), ""),
    "xl": _StableDiffusionConf(os.path.join("configs", "sd_xl_base.yaml"), "sdxl-turbo"),
    "xl_refiner": _StableDiffusionConf(os.path.join("configs", "sd_xl_refiner.yaml"), "")
}


def parse_args():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion XL Turbo Simple Benchmark",
                                     conflict_handler='resolve')

    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, choices=["cuda", "cpu", "npu"],
                        help="Inference target device")
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help="Checkpoint file of SDXL Turbo")
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help="Number of images to generate in a sequence, one  at a time")
    parser.add_argument('--iteration', type=int, default=DEFAULT_BATCH_COUNT,
                        help="Number of repeat prompt")
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT,
                        help="Height of image to generate (must be multiple of 8")
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH,
                        help="Width of image to generate (must be multiple of 8")
    parser.add_argument('--result', type=str, default=DEFAULT_RESULT,
                        help="Benchmark result file path")
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help="A directory to save images")
    parser.add_argument('--upscale', type=int, default=DEFAULT_UPSCALE, choices=[1, 2, 4],
                        help="Image upscale rate to generate from FHD to QHD images")
    parser.add_argument('--denoising-steps', type=int, default=DEFAULT_DENOISING_STEPS,
                        help="Number of denoising steps")
    parser.add_argument('--guidance-scale', type=float, default=DEFAULT_GUIDANCE_SCALE,
                        help="Number of guidance scale")
    parser.add_argument('--prompt', type=str, nargs='+', default=DEFAULT_PROMPT,
                        help="prompt to generate image")
    parser.add_argument('--negative-prompt', type=str, nargs='*', default=DEFAULT_NEGATIVE_PROMPT,
                        help="Negative prompt to avoid from image")

    arguments = parser.parse_args()

    if arguments.height < 500:
        raise Exception("height cannot smaller then 500")

    if arguments.width < 500:
        raise Exception("width cannot smaller than 500")

    return arguments


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


class Latency:
    def __init__(self, name: str, monitor):
        self.__name = name
        self.__monitor = monitor

        self.__tick = None
        self.__tock = None

    def name(self) -> str:
        return self.__name

    def latency(self) -> float:
        return (self.__tock - self.__tick) * 1000

    def __enter__(self):
        self.__tick = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__tock = time.perf_counter()
        self.__monitor.append(self)


class LatencyMonitor:
    def __init__(self):
        self.__latencies = []

    def append(self, latency: Latency):
        self.__latencies.append(latency)

    def __str__(self):
        result = defaultdict(float)

        for latency in self.__latencies:
            result[latency.name()] += latency.latency()

        return '\n'.join([f'{k}: {v:.2f} msec' for k, v in result.items()])


class SDXLTurboPipeline:
    def __init__(self, checkpoint, upscale, result, output, batch_size, batch_count, device, latencies):
        self.latencies = latencies

        configs = STABLE_DIFFUSION_CONFIG["xl"]
        with Latency("Load Model", self.latencies):
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

            if device == "cuda":
                self.pipeline.enable_xformers_memory_efficient_attention()
                self.pipeline.enable_sequential_cpu_offload()

        with Latency("Load Upscaler", self.latencies):
            self.upscaler = RealESRGANUpScaler(upscale, device) if upscale > 1 else UpScaler()

        self.result = result
        self.output = output

        self.batch_size = batch_size
        self.batch_count = batch_count

        with Latency("Load Randomizer", self.latencies):
            self.generators = [torch.Generator(device=device).manual_seed(int(random.randrange(1, 9999)))
                               for _ in range(batch_size)]

    def __target_size__(self, width, height):
        return width*self.upscaler.scale(), height*self.upscaler.scale()

    def __generate__(self, prompt, negative_prompt, width, height, denoising_steps, guidance_scale):
        with Latency("Images Generate", self.latencies):
            output = self.pipeline(
                prompt=self.batch_size * [' '.join(prompt)],
                negative_prompt=self.batch_size * [' '.join(negative_prompt)],
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
        with Latency("Images Upscale", self.latencies):
            for image_arr in np_array:
                images.append(Image.fromarray(self.upscaler(np.uint8(image_arr*255))))

        return images

    def __save_image__(self, images, prompt):
        basename = '-'.join([re.sub(r'[^a-zA-Z0-9]', '', word) for word in prompt[:5]])
        basename += '-'+str(random.randint(1000, 9999))

        with Latency("Images Save", self.latencies):
            for i, image in enumerate(images):
                image.save(os.path.join(self.output, basename+'-'+str(i)+'.png'))

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
        with Latency("End-to-end Generation", self.latencies):
            for _ in range(iteration):
                self.generate(prompt, negative_prompt, width, height, denoising_steps, guidance_scale)


if __name__ == '__main__':
    args = parse_args()

    latency_monitor = LatencyMonitor()

    with Latency("End-to-end Benchmark", latency_monitor):
        bench = SDXLTurboPipeline(
            args.checkpoint,
            args.upscale,
            args.result,
            args.output,
            args.batch_size,
            args.iteration,
            args.device,
            latency_monitor
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

    print(latency_monitor)
