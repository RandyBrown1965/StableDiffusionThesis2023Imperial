from diffusers import StableDiffusionXLPipeline
import torch

randomgenerator = torch.Generator(device="cuda") 
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")

seed = 119
use_neg_prompt = True
#prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
prompt = "Photo of brunette holding a book"
negative_prompt = "low quality, disfigured, mutation, ugly, deformed, malformed, cartoon, bad anatomy, extra arm, missing arm, extra leg, missing leg, (((extra hand, extra fingers))), missing fingers, fused fingers, (((poorly drawn hands, poorly drawn fingers, poorly drawn face, malformed hands, malformed fingers))), malformed limbs, malformed face, long neck, overly-saturated, (((out of frame, cropped, text))), letters, watermark, (((closed eyes, frame, border)))" # my combination

"""
image = pipe(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=num_inference_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
    generator=randomgenerator
"""

for cfg in range(1,12,1):
    for use_neg_prompt in [False, True]:

        print(use_neg_prompt)
        randomgenerator.manual_seed(seed)
        image = pipe(prompt=prompt, guidance_scale= cfg, 
                negative_prompt=negative_prompt if use_neg_prompt else "",
                generator=randomgenerator).images[0]

        filename = "Bookworm_cfg" + str(cfg) + ".png"
        print(use_neg_prompt)
        if use_neg_prompt:
            print(filename)
            print("THIS SHOULD CHANGE THE FILENAME")
            filename = filename.replace(".png", "_negprompt.png")
            print(filename)
        print("Saving ",filename)
        image.save(filename)
    
