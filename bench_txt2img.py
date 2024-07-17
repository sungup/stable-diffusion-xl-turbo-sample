import argparse
import numpy as np
import os
import random
import re
import torch

from PIL import Image

# module based diffuserbm
import diffuserbm.upscale as upscale
import diffuserbm.pipeline as pipelines
from diffuserbm.perf import PerfMon


DEFAULT_DEVICE = "mps"
DEFAULT_BATCH_SIZE = 4
DEFAULT_BATCH_COUNT = 5
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 512
DEFAULT_RESULT = "stdout"
DEFAULT_FORMAT = "table"
DEFAULT_OUTPUT = "output"
DEFAULT_DENOISING_STEPS = 4
DEFAULT_GUIDANCE_SCALE = 0.0

DEFAULT_PROMPT = """
cherry blossom, bonsai, Japanese style landscape, high resolution, 8k, lush green in the background
""".split(' ')
DEFAULT_NEGATIVE_PROMPT = """
Low quality, Low resolution, disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w
""".split(' ')

RESULT_FORMAT = ['table', 'text', 'csv', 'json']

MAX_FILE_WORD = 5


class Text2ImageBench:
    def __init__(self, batch_size, batch_count, device, output, perf: PerfMon):
        self.__perf = perf

        self.device = device
        self.output = output

        self.batch_size = batch_size
        self.batch_count = batch_count

        self.pipeline = None
        self.upscaler = None
        self.generators = None

    def init_bench(self, pipeline, upscaler):
        self.pipeline = pipeline
        self.upscaler = upscaler

        with self.__perf.measure_latency('Load Randomizer', 'model'):
            self.generators = [torch.Generator(device=self.device).manual_seed(int(random.randrange(1, 9999)))
                               for _ in range(self.batch_size)]

    def __generate__(self, prompt, negative_prompt, width, height, denoising_steps, guidance_scale):
        with self.__perf.measure_latency('Images Generation', 'gen'):
            return self.pipeline(
                prompt=self.batch_size * [' '.join(prompt)],
                negative=self.batch_size * [' '.join(negative_prompt)],
                rand_gen=self.generators,
                width=width,
                height=height,
                denoising_steps=denoising_steps,
                guidance_scale=guidance_scale,
            )

    def __upscale__(self, np_array):
        images = []
        for image_arr in np_array:
            with self.__perf.measure_latency('Images Upscale', 'image'):
                images.append(Image.fromarray(self.upscaler(np.uint8(image_arr*255))))

        return images

    def __save_image__(self, images, prompt):
        basename = '-'.join([re.sub(r'[^a-zA-Z0-9]', '', word) for word in prompt[:MAX_FILE_WORD]])
        basename += '-'+str(random.randint(1000, 9999))

        for i, image in enumerate(images):
            with self.__perf.measure_latency('Images Save', 'image'):
                image.save(os.path.join(self.output, basename+'-'+str(i)+'.png'))

    def generate(self, prompt, negative_prompt, width, height, denoising_steps, guidance_scale):
        self.__save_image__(
            self.__upscale__(
                self.__generate__(prompt, negative_prompt, width, height, denoising_steps, guidance_scale)
            ),
            prompt
        )

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
            with self.__perf.measure_latency('End-to-End Generation', 'iter'):
                self.generate(prompt, negative_prompt, width, height, denoising_steps, guidance_scale)


def parse_args():
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion XL Turbo Simple Benchmark",
                                     conflict_handler='resolve')

    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, choices=["cuda", "cpu", "npu"],
                        help="Inference target device")
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
    parser.add_argument('--format', type=str, default=DEFAULT_FORMAT, choices=RESULT_FORMAT,
                        help="Result file format")
    parser.add_argument('--denoising-steps', type=int, default=DEFAULT_DENOISING_STEPS,
                        help="Number of denoising steps")
    parser.add_argument('--guidance-scale', type=float, default=DEFAULT_GUIDANCE_SCALE,
                        help="Number of guidance scale")
    parser.add_argument('--prompt', type=str, nargs='+', default=DEFAULT_PROMPT,
                        help="prompt to generate image")
    parser.add_argument('--negative-prompt', type=str, nargs='*', default=DEFAULT_NEGATIVE_PROMPT,
                        help="Negative prompt to avoid from image")

    upscale.add_arguments(parser)
    pipelines.add_arguments(parser)

    arguments = parser.parse_args()

    if arguments.height < 500:
        raise Exception("height cannot smaller then 500")

    if arguments.width < 500:
        raise Exception("width cannot smaller than 500")

    return arguments


def txt2img():
    args = parse_args()

    perf_mon = PerfMon()

    with perf_mon.measure_latency('End-to-End Benchmark', 'run'):
        bench = Text2ImageBench(args.batch_size, args.iteration, args.device, args.output, perf_mon)

        with perf_mon.measure_latency('Load Checkpoint', 'model'):
            pipeline = pipelines.make(**args.__dict__)

        # make upscaler
        with perf_mon.measure_latency('Load Upscaler', 'model'):
            upscaler = upscale.make(**args.__dict__)

        bench.init_bench(pipeline=pipeline, upscaler=upscaler)

        bench.run(
            args.iteration,
            args.prompt,
            args.negative_prompt,
            args.width,
            args.height,
            args.denoising_steps,
            args.guidance_scale
        )

    if args.result == 'stdout':
        print(perf_mon.report(args.format))
    else:
        with open(args.result, 'w') as out:
            out.write(perf_mon.report(args.format))


if __name__ == '__main__':
    txt2img()
