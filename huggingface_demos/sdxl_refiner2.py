from diffusers import DiffusionPipeline
import torch
import time
#from torchvision import transforms

randomgenerator = torch.Generator(device="cuda") # THIS IS THE OLD LINE THAT USED TO WORK.

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")

#prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed"
prompt = "A majestic lion jumping from a big stone at night"
prompt = "Photo of " + prompt + ", 8k, dslr, photorealistic, hyperrealistic, Fujifilm, film grain, wide dof"
num_inference_steps = 50
high_noise_frac = 0.8


######
# Generate base unrefined
randomgenerator.manual_seed(119)
image = pipe(prompt=prompt, output_type = "pil",
        generator=randomgenerator,
        ).images[0]
image.save("SDXL_diffusion_pipeline_generated_"+str(time.time())+".png")


#################
# Definer refiner
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

use_refiner = True
randomgenerator.manual_seed(119)
image = pipe(prompt=prompt, output_type="latent" if use_refiner else "pil",
        generator=randomgenerator,
        ).images[0]
image = refiner(prompt=prompt, image=image[None, :]).images[0]
#image = pipe(prompt=prompt, num_inference_steps = num_inference_steps, denoising_end = high_noise_frac, output_type="latent" if use_refiner else "pil").images[0]
#image = refiner(prompt=prompt, num_inference_steps = num_inference_steps, denoising_start = high_noise_frac, image=image[None, :]).images[0]
image.save("SDXL_fullbase_refined_"+str(time.time())+".png")


##########
# Generate without refiner
del refiner
from diffusers import StableDiffusionXLPipeline

use_refiner = False
randomgenerator.manual_seed(119)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")
randomgenerator.manual_seed(119)
image = pipe(prompt=prompt,
        generator=randomgenerator,
        ).images[0]
image.save("SDXL_XLpipeline_generated_"+str(time.time())+".png")
