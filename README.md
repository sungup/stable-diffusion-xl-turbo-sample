# Simple Stable Diffusion script

## Purpose of this project

Stable diffusion offline mode running script.

## Directory structure

```text
+- checkpoints
|   +- sd_xl_turbo_1.0_fp16.safetensors
+- configs
|   +- sd_xl_base.yaml
+- sdxl-turbo
+- upscaler
|   +- Real-ESRGAN
+- README.md
+- basic_sdxl_turbo.py
+- requirements.txt
```

### Making each directory

#### checkpoints

Download safetensors file from huggingface or the other repositories.

#### configs

Not yet ready to explain

#### sdxl-turbo

Clone from huggingface without git-lfs. In this project, use only configuration files so there is no need download other
model checkpoints.

#### upscaler

Not yet ready to explain

## Performance Check

In my RTX-3070 GPU + Ryzen 5 5600x 6-Core process, this code shows loading SDXL Turbo model in 20sec. I'll record the
loading and inference speed on the other system A100 and H100.