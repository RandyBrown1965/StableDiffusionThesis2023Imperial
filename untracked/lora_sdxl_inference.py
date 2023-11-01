from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
# load LoRA weight
pipe.unet.load_attn_procs("data/checkpoint-2000/pytorch_lora_weights.bin", use_safetensors=False) # CHANGE THIS PATH
pipe.enable_model_cpu_offload()

seed = 12345
n_steps = 50
prompt = "1girl, solo, pink hair, animal ears, mechanical ears, long hair" # CHANGE THIS PROMPT

generator = torch.Generator(device="cuda").manual_seed(seed)
image = pipe(prompt=prompt, generator=generator).images[0]
#image = pipe(prompt=prompt, generator=generator, +   cross_attention_kwargs={"scale": 2.0}).images[0]  #CHANGE THE SCALE VALUE TO ADJUST THE LORA APPLICATION (0, 1.0)
image.save("image.jpg")

