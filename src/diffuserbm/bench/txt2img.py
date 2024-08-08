"""Module providing benchmark functionality for the stable diffusion text to image conversion."""

import os
import random
import re

import numpy as np

from PIL import Image

from diffuserbm import DiffuserBMConfig
from diffuserbm import upscale
from diffuserbm import pipeline
from diffuserbm.perf import PerfMon
from diffuserbm.bench import consts


class Bench:
    """Class for benchmarking diffusion text to image generation."""
    MAX_FILE_WORD = 5

    def __init__(self, batch_size, output, perf: PerfMon):
        self.__perf = perf

        self.__output = output

        self.__batch_size = batch_size

        self.__pipeline = None
        self.__upscaler = None

    def init_bench(self, bm_pipeline, upscaler):
        """Initialize the benchmark."""
        self.__pipeline = bm_pipeline
        self.__upscaler = upscaler

    def __generate__(self, prompt, width, height, **kwargs):
        negative_prompt = kwargs.get('negative_prompt', '')
        denoising_steps = kwargs.get('denoising_steps', 8)
        guidance_scale = kwargs.get('guidance_scale', 0.0)

        with self.__perf.measure_latency('Images Generation', 'gen'):
            return self.__pipeline(
                prompt=self.__batch_size * [' '.join(prompt)],
                negative=self.__batch_size * [' '.join(negative_prompt)],
                width=width,
                height=height,
                denoising_steps=denoising_steps,
                guidance_scale=guidance_scale,
            )

    def __upscale__(self, np_array):
        images = []
        for image_arr in np_array:
            with self.__perf.measure_latency('Images Upscale', 'image'):
                images.append(Image.fromarray(self.__upscaler(np.uint8(image_arr*255))))

        return images

    def __save_image__(self, images, prompt):
        basename = '-'.join([re.sub(r'[^a-zA-Z0-9]', '', word)
                             for word in prompt[:Bench.MAX_FILE_WORD]])
        basename += '-'+str(random.randint(1000, 9999))

        for i, image in enumerate(images):
            with self.__perf.measure_latency('Images Save', 'image'):
                image.save(os.path.join(self.__output, basename+'-'+str(i)+'.png'))

    def generate(self, prompt, width, height, **kwargs):
        """
        Generate a random image and save it as PNG files at an iteration.

        Args:
          prompt (str): input prompt to generate image
          width (int): width of image
          height (int): height of image

        Kwargs:
            negative (str): negative prompt to avoid features not wanted in the generated image
            denoising_steps (int): number of inference step to denoising
            guidance_scale (float): scale of guidance for the generating image

        """
        negative = kwargs.get('negative', '')
        denoising_steps = kwargs.get('denoising_steps', 8)
        guidance_scale = kwargs.get('guidance_scale', 0.0)

        self.__save_image__(
            self.__upscale__(
                self.__generate__(
                    prompt=prompt,
                    width=width,
                    height=height,
                    negative_prompt=negative,
                    denoising_steps=denoising_steps,
                    guidance_scale=guidance_scale,
                )
            ),
            prompt
        )

    def run(self, prompt, width, height, **kwargs):
        """
        Run benchmark for Text-to-Image generation.

        Args:
          prompt (str): input prompt to generate image
          width (int): width of image
          height (int): height of image

        Kwargs:
            negative (str): negative prompt to avoid features not wanted in the generated image
            iteration (int): number of inference step to generate image
            denoising_steps (int): number of inference step to denoising
            guidance_scale (float): scale of guidance for the generating image

        """
        negative = kwargs.get('negative', '')
        iteration = kwargs.get('iteration', 1)
        denoising_steps = kwargs.get('denoising_steps', 8)
        guidance_scale = kwargs.get('guidance_scale', 0.0)

        for _ in range(iteration):
            with self.__perf.measure_latency('End-to-End Generation', 'iter'):
                self.generate(
                    prompt=prompt,
                    width=width,
                    height=height,
                    negative=negative,
                    denoising_steps=denoising_steps,
                    guidance_scale=guidance_scale,
                )


def command_help() -> str:
    """Returns a help string for Txt2Img benchmark arguments parser."""
    return 'Text to Image Benchmark'


def command_name() -> str:
    """Returns sub-command name for Txt2Img benchmark."""
    return 'txt2img'


def add_arguments(parser, config: DiffuserBMConfig):
    """Adds arguments to the parser."""
    parser.add_argument('--batch-size', type=int, default=consts.DEFAULT_BATCH_SIZE,
                        help="Number of images to generate in a sequence, one  at a time")
    parser.add_argument('--iteration', type=int, default=consts.DEFAULT_BATCH_COUNT,
                        help="Number of repeat prompt")
    parser.add_argument('--height', type=int, default=consts.DEFAULT_HEIGHT,
                        help="Height of image to generate (must be multiple of 8")
    parser.add_argument('--width', type=int, default=consts.DEFAULT_WIDTH,
                        help="Width of image to generate (must be multiple of 8")
    parser.add_argument('--result', type=str, default=consts.DEFAULT_RESULT,
                        help="Benchmark result file path")
    parser.add_argument('--output', type=str, default=consts.DEFAULT_OUTPUT,
                        help="A directory to save images")
    parser.add_argument('--format', type=str, default=consts.DEFAULT_FORMAT,
                        choices=consts.RESULT_FORMAT,
                        help="Result file format")
    parser.add_argument('--denoising-steps', type=int, default=consts.DEFAULT_DENOISING_STEPS,
                        help="Number of denoising steps")
    parser.add_argument('--guidance-scale', type=float, default=consts.DEFAULT_GUIDANCE_SCALE,
                        help="Number of guidance scale")
    parser.add_argument('--prompt', type=str, nargs='+', default=consts.DEFAULT_PROMPT,
                        help="prompt to generate image")
    parser.add_argument('--negative-prompt', type=str, nargs='*',
                        default=consts.DEFAULT_NEGATIVE_PROMPT,
                        help="Negative prompt to avoid from image")

    upscale.add_arguments(parser, config)
    pipeline.add_arguments(parser, config)


def post_arguments(args):
    """Check arguments after parsing."""
    if args.height < 500:
        raise RuntimeError("height cannot smaller then 500")

    if args.width < 500:
        raise RuntimeError("width cannot smaller than 500")


def run(args, config: DiffuserBMConfig):
    """Run DiffuserBM Benchmark for the Text-to-Image Generative AI workload."""
    perf_mon = PerfMon()

    with perf_mon.measure_latency('End-to-End Benchmark', 'run'):
        bm = Bench(args.batch_size, args.output, perf_mon)

        with perf_mon.measure_latency('Load Checkpoint', 'model'):
            bm_pipeline = pipeline.make(config, **args.__dict__)

        # make upscaler
        with perf_mon.measure_latency('Load Upscaler', 'model'):
            upscaler = upscale.make(config, **args.__dict__)

        bm.init_bench(bm_pipeline=bm_pipeline, upscaler=upscaler)

        bm.run(
            iteration=args.iteration,
            prompt=args.prompt,
            negative=args.negative_prompt,
            width=args.width,
            height=args.height,
            denoising_steps=args.denoising_steps,
            guidance_scale=args.guidance_scale
        )

    if args.result == 'stdout':
        print(perf_mon.report(args.format))
    else:
        with open(args.result, 'w', encoding='utf-8') as out:
            out.write(perf_mon.report(args.format))
