from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    #"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    "stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True
)
pipe.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    #torch_dtype=torch.float16,
    use_safetensors=True,
    #variant="fp16"
)
refiner.to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

use_refiner = True
image = pipe(prompt=prompt, output_type="latent" if use_refiner else "pil").images[0]
print(type(image))
print(type(image[None, :]))
image = refiner(prompt=prompt, image=image[None, :]).images[0]

import time
image.save("SDXL_refiner_astronaut_" + str(time.time()) + ".png")
