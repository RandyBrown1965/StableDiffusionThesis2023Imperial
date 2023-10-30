from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image
import torch
randomgenerator = torch.Generator(device="cuda") 

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
#pipe.to("cuda")
pipe.enable_model_cpu_offload()

refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
#refiner.to("cuda")
refiner.enable_model_cpu_offload()

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).convert("RGB")
mask_image = load_image(mask_url).convert("RGB")

prompt = "A majestic tiger sitting on a bench"
num_inference_steps = 75
high_noise_frac = 0.7

seed = 119
randomgenerator.manual_seed(seed)
image = pipe(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
    generator=randomgenerator
).images
image = refiner(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    denoising_start=high_noise_frac,
    generator=randomgenerator
).images[0]
image.save("Tiger_inpainting_refined_highnoisefrac_" + str(seed) + ".png")

image = pipe(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    #denoising_end=high_noise_frac,
    output_type="latent",
    generator=randomgenerator
).images
image = refiner(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    #denoising_start=high_noise_frac,
    generator=randomgenerator
).images[0]
image.save("Tiger_inpainting_refined_nofrac_" + str(seed) + ".png")


seed = 33
randomgenerator.manual_seed(seed)
image = pipe(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
    generator=randomgenerator
).images
image = refiner(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    denoising_start=high_noise_frac,
    generator=randomgenerator
).images[0]
image.save("Tiger_inpainting_refined_highnoisefrac_" + str(seed) + ".png")

image = pipe(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    #denoising_end=high_noise_frac,
    output_type="latent",
    generator=randomgenerator
).images
image = refiner(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    #denoising_start=high_noise_frac,
    generator=randomgenerator
).images[0]
image.save("Tiger_inpainting_refined_nofrac_" + str(seed) + ".png")
