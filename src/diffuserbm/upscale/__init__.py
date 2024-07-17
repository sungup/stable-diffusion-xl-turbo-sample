import diffuserbm.upscale.core as core
import diffuserbm.upscale.real_esrgan


def make(upscaler, upscale_rate, upscaler_path, device, **kwargs):
    return core.make_upscaler(upscaler, upscale_rate, upscaler_path, device)


def add_arguments(parser):
    parser.add_argument('--upscaler', type=str, default='none', choices=core.supported_upscaler(),
                        help='Model type name of upscaler')
    parser.add_argument('--upscale-rate', type=int, default=1, choices=[1, 2, 4],
                        help="Image upscale rate to generate from FHD to QHD images")
    parser.add_argument('--upscaler-path', type=str,
                        help='Upscale model path')
