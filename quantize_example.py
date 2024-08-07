import os.path

from diffuserbm.utils.convert import diffuser_to_onnx
from diffuserbm.utils.quantize import quantize, to_fp16, to_int8


if __name__ == '__main__':
    if not os.path.isdir('checkpoints/dreamshaper-8.onnx'):
        diffuser_to_onnx('checkpoints/dreamshaper-8', 'checkpoints/dreamshaper-8.onnx')

    #if not os.path.isdir('checkpoints/dreamshaper-8.fp16.onnx'):
    #    quantize('checkpoints/dreamshaper-8.onnx', 'checkpoints/dreamshaper-8.fp16.onnx', to_fp16)

    if not os.path.isdir('checkpoints/dreamshaper-8.int8.onnx'):
        quantize('checkpoints/dreamshaper-8.onnx', 'checkpoints/dreamshaper-8.int8.onnx', to_int8)
