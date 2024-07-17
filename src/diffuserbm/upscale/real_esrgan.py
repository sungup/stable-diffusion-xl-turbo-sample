from diffuserbm.upscale.core import UpScaler

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


class RealESRGANUpScaler(UpScaler, name='r-esrgan'):
    def __init__(self, model_path, scale, device):
        super().__init__(scale=scale)

        self.pipeline = RealESRGANer(
            scale=scale,
            model_path=model_path,
            device=device,
            dni_weight=None,
            model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale),
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )

    def __call__(self, np):
        return self.pipeline.enhance(np, outscale=self.scale)[0]
