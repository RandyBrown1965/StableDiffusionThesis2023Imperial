import time
start= time.time()
print("Loading libraries")
from diffusers import StableDiffusionInpaintPipeline
import xformers
import torch
#from safetensors import safe_open
from PIL import Image
import numpy as np
from diffusers import DDIMScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, PNDMScheduler, DDPMScheduler, EulerAncestralDiscreteScheduler
from inception_score_model import get_inception_score
print("Loading time = ",time.time()-start," seconds")




"""
######################################################
#  Try an Inception Score from
# https://github.com/openai/improved-gan/tree/master/inception_score

crop_size = 32
filename_in = "input_images/PhotoOfABrunetteHoldingABookAvoidMangledFingers04.jpg"
image_in=Image.open(filename_in)
image_in_np = np.asarray(image_in)
xdim_in, ydim_in, _ = image_in_np.shape
print("Image=",xdim_in,"x",ydim_in)
xdim_out, ydim_out = xdim_in//crop_size, ydim_in//crop_size
computed_mask = np.zeros((xdim_out, ydim_out))
print("mask=",computed_mask.shape)
resized_array = np.zeros((xdim_out, ydim_out, 3),dtype=np.uint8)
image_crops = []
for x in range(0, xdim_in, crop_size):
    for y in range(0, ydim_in, crop_size):
        image_crop = image_in_np[x:x + crop_size, y:y+crop_size, :]
        if (np.max(image_crop) <= 10):
            print(image_crop)
            image_crop = image_crop *20 / np.max(image_crop)
            print(image_crop)
        image_crops.append(image_crop)
        resized_array[int(x/crop_size), int(y/crop_size), 0] = image_crop[0,0,0]
        resized_array[int(x/crop_size), int(y/crop_size), 1] = image_crop[0,0,1]
        resized_array[int(x/crop_size), int(y/crop_size), 2] = image_crop[0,0,2]

print(resized_array.shape)
resized_image = Image.fromarray(resized_array)
resized_image.save("ResizedInputImage" + str(len(computed_mask)) + ".jpg")
total_mean, total_std = get_inception_score(image_crops)
print("Overall mean=",total_mean," std=",total_std)

for x in range(xdim_out):
    for y in range(ydim_out):
        popped_image_crops = image_crops*1
        popped_image_crops.pop(x*ydim_out + y)
        print(len(popped_image_crops)," images")
        mean, std = get_inception_score(popped_image_crops)
        print(x,",",y," mean=",mean," std=",std)
        computed_mask[x, y] = mean/total_mean
print("computed_mask =\n",computed_mask)

mask_array_16x16 = np.array(computed_mask, dtype=np.uint8)
print("computed_mask =\n",mask_array_16x16)
# Computed output took 2.5 hours, saved below
computed_mask = np.array([
 [1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1],
 [1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1],
 [1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
 [1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1],
 [1,1,1,1,1,1,1,0,1,1,1,0,1,1,0,0],
 [1,1,1,1,1,0,0,0,0,0,1,1,1,0,1,1],
 [1,1,1,1,1,1,1,0,1,1,0,0,0,0,0,1],
 [1,1,1,1,1,1,1,0,1,0,0,0,0,1,1,0],
 [0,1,1,1,1,0,0,0,0,0,0,0,0,1,0,1],
 [1,1,1,1,0,1,0,0,0,0,0,0,0,0,1,1],
 [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
 [1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1],
 [0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,1],
 [0,1,1,1,1,0,1,0,0,0,0,0,0,1,0,0],
 [0,0,1,0,1,0,0,0,0,0,0,1,1,0,0,0]], dtype=np.uint8)
"""


#### PARAMETERS
dtype=torch.float16
seed = 119
#seed = 33
#seed = 0
num_inference_steps = 50
num_images_per_prompt = 2
guidance_scale = 7.5
denoising_strength=1.0 #0.7
height=512
#height = 768
width = 512


############### Make the inpainting mask for the image
#The mask structure is white for inpainting and black for keeping as is
# Mask created by hand to replace her book and screen-right hand
mask_array_16x16 = np.array([
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0],
[0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0],
[0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0],
[0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
[0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
[0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0],
[0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0]], dtype=np.uint8)
# Mask computed from Inception_score (inverted)
mask_array_16x16 = np.array([
 [1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1],
 [1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1],
 [1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
 [1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1],
 [1,1,1,1,1,1,1,0,1,1,1,0,1,1,0,0],
 [1,1,1,1,1,0,0,0,0,0,1,1,1,0,1,1],
 [1,1,1,1,1,1,1,0,1,1,0,0,0,0,0,1],
 [1,1,1,1,1,1,1,0,1,0,0,0,0,1,1,0],
 [0,1,1,1,1,0,0,0,0,0,0,0,0,1,0,1],
 [1,1,1,1,0,1,0,0,0,0,0,0,0,0,1,1],
 [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
 [1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1],
 [0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,1],
 [0,1,1,1,1,0,1,0,0,0,0,0,0,1,0,0],
 [0,0,1,0,1,0,0,0,0,0,0,1,1,0,0,0]], dtype=np.uint8)
mask_array_16x16 = 1-mask_array_16x16

# Random mask
#mask_array_16x16 = np.random.randint(0, 2, mask_array_16x16.shape, dtype=np.uint8)


xdim, ydim = mask_array_16x16.shape
mask_array = np.zeros((512,512), dtype=np.uint8)
block = int(512/16)
for x in range(xdim):
    for y in range(ydim):
        mask_array[x*block:(x+1)*block, :][:, y*block:(y+1)*block] = np.ones((block, block)) * 255 * mask_array_16x16[x][y]
mask_image = Image.fromarray(mask_array)
mask_filename = "input_images/mask_BrunetteBook04.jpg"
mask_image.save(mask_filename)


######################################
# IMAGE GENERATION MODELS
compvis_1_4="CompVis/stable-diffusion-v1-4"
runwayml_1_5="runwayml/stable-diffusion-v1-5"
dreamlikephotoreal_2_0="dreamlike-art/dreamlike-photoreal-2.0" # based on StableDiffusion_v1.5 contact@dreamlike.art
stabilityai_2_1_base = "stabilityai/stable-diffusion-2-1-base"
stabilityai_2_1 = "stabilityai/stable-diffusion-2-1" # 768x768 images
icbinp="/homes/rdb121/Thesis/models/icbinpICantBelieveIts_final.safetensors"
# INPAINTING MODELS
stabilityai_inpainting_2 = "stabilityai/stable-diffusion-2-inpainting"
stabilityai_inpainting_2_local = "/homes/rdb121/Thesis/models/stable-diffusion-2-inpainting_local"
runwayml_inpainting_1="runwayml/stable-diffusion-inpainting"
revision="fp16" # necessary for runwayml_inpainting?
#
generative_model_id = stabilityai_2_1_base
inpainting_model_id = stabilityai_inpainting_2_local

filename_prompt = "BrunetteHoldingBook"
#### PROMPTS
#prompt = "photo, brunette holding book, high quality, photo-real, cinematic lighting, ultra-real, 50mm, realistic proportions, finely detailed"
prompt = "brunette holding a book"
#prompt = "young woman, highlight hair, sitting outside restaurant, wearing dress" 
#prompt = "photo, a church in the middle of a field of crops, bright cinematic lighting, gopro, fisheye lens"

#prompt_prefix = "RAW photo, " #https://stablediffusionapi.com/models/realistic-vision-v13
prompt_prefix = "photo of "

#prompt_suffix = ", (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3" #https://stablediffusionapi.com/models/realistic-vision-v13
prompt_suffix = ", rim lighting, studio lighting, looking at the camera, dslr, ultra quality, sharp focus, tack sharp, dof, film grain, Fujifilm XT5, crystal clear, 8K UHD, highly detailed glossy eyes, high detailed skin, skin pores, hyperrealistic, pixabay, beautiful hands, detailed fingers" # stable-diffusion-art.com

# TEXT EMBEDDINGS
text_embeddings = ",Style-Empire"

prompt_total = prompt + prompt_prefix + prompt_suffix
#prompt_total = prompt + prompt_prefix + prompt_suffix + text_embeddings

##### NEGATIVE PROMPTS
#negative_prompt = "low quality disfigured bad gross disgusting mutation ugly morbid mutated deformed mutilated mangled poorly drawn face extra limb missing limb floating limbs disconnected limbs malformed limbs oversaturated duplicate bodies cloned faces low-res blurry blur out of focus out of frame extra missing"
negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 5d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, (((extra fingers))), mutated hands, (((poorly drawn hands))), poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, extra hands, fused fingers, too many fingers, long neck, overly-saturated" #https://stablediffusionapi.com/models/realistic-vision-v13
#negative_prompt = "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature" # https://huggingface.co/spaces/stabilityai/stable-diffusion/discussions/7858  Dan Mulheran
#negative_prompt = "disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w" # stable-diffusion-art.com

#### SCHEDULERS
schedulers = [DDIMScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, PNDMScheduler, DDPMScheduler, EulerAncestralDiscreteScheduler]

preload_time = time.time()

############### INPAINTING  
randomgenerator = torch.Generator(device="cuda")
randomgenerator.manual_seed(seed)
filename_in = "input_images/PhotoOfABrunetteHoldingABookAvoidMangledFingers04.jpg"
image_in=Image.open(filename_in)
#filename_mask = "input_images/mask_BrunetteBook04.jpg"
#mask_image=Image.open(filename_mask)
premodel=time.time()
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    inpainting_model_id,
    local_files_only = inpainting_model_id.startswith("/"),
    #revision="fp16",
    torch_dtype=dtype,
    ).to("cuda")
print("Model loading time = ",time.time()-premodel," seconds")


pregeneration_time = time.time()
for num_inference_steps in [50]:
    # Run the inpainting
    options = dict(
        prompt=prompt_total,
        negative_prompt=negative_prompt, 
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        guidance_scale=guidance_scale, #default = 7.5
        generator=randomgenerator,
        height = height, # default = 512
        width = width, # default = 512
        #text_input = text_input,
    
        image=image_in, # for inpainting
        mask_image=mask_image, # for inpainting
        strength= denoising_strength # 0-1, 0=no change, default=1.0
        )

    ##### RUN WITH GPU
    s = time.time()
    images_out = pipe(**options).images
    image = images_out[0]
    #print(f"Generation time with GPU: {time.time() - s} seconds")
    for i in range(len(images_out)):
        images_out[i].save("inpaint"+str(filename_prompt)+"_" + inpainting_model_id.split('/')[-1] + "_seed"+str(seed) + "_"+ str(num_inference_steps) + "steps_"+str(i)\
                +"_cfg"+str(guidance_scale)+"_strength"+str(denoising_strength) + "_"+str(time.time())+".png")
        """
        # Evaluate variation of the denoising_strength with the following
        images_out[i].save("inpaint"+str(filename_prompt)+"_" + inpainting_model_id.split('/')[-1] + "_seed"+str(seed) + "_"+ str(denoising_strength100) + "steps_"+str(i)\
                +"_cfg"+str(guidance_scale)+"_strength"+str(denoising_strength) + "_"+str(time.time())+".png")
        """
        print("Generation time = ",time.time()-pregeneration_time," seconds")
    #filename = "./BrunetteHoldingBook04_stabilityai_GPU_"+str(num_inference_steps)+"steps.png"
    #print("Writing file: ",filename)
    #image.save(filename)
    #next(randomgenerator)


print("Total time = ",time.time()-start," seconds\n")
#del pipe



"""
crop_size = 512
image_in_np = np.asarray(images_out[0])
xdim, ydim, _ = image_in_np.shape
computed_mask_out = np.zeros((xdim//crop_size, ydim//crop_size))
for x in range(0, xdim, crop_size):
    for y in range(0, ydim, crop_size):
        image_crop = image_in_np[x:x + crop_size, y:y+crop_size, :]
        if (np.max(image_crop) <= 10):
            print(image_crop)
            image_crop = image_crop *20 / np.max(image_crop)
            print(image_crop)
        mean, std = get_inception_score([image_crop, image_crop, image_crop, image_crop, image_crop, image_crop, image_crop, image_crop, image_crop, image_crop, image_crop])
        print("mean=",mean)
        computed_mask_out[int(x/crop_size), int(y/crop_size)] = mean


print(computed_mask_out)
"""

##################################################################
"""
from torchmetrics.image.inception import InceptionScore

inception_score_fn = InceptionScore(normalize=True)

def compute_metrics(images: np.ndarray, prompts: List[str]):


    inception_score_fn.update(torch.from_numpy(images).permute(0, 3, 1, 2))
    inception_score = inception_score_fn.compute()

    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(
        torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts
    ).detach()
    return {
        "inception_score (⬆️)": {
            "mean": round(float(inception_score[0]), 4),
            "std": round(float(inception_score[1]), 4),
        },
        "clip_score (⬆️)": round(float(clip_score), 4),
    }

####################################################################
# https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
import ignite.distributed as idist
from ignite.metrics import FID, InceptionScore
import PIL.Image as Image

fid_metric = FID(device=idist.device())
is_metric = InceptionScore(device=idist.device(), output_transform=lambda x: x[0])


def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299,299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)


def evaluation_step(engine, batch):
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim, 1, 1, device=idist.device())
        netG.eval()
        fake_batch = netG(noise)
        fake = interpolate(fake_batch)
        real = interpolate(batch[0])
        return fake, real

evaluator = Engine(evaluation_step)
fid_metric.attach(evaluator, "fid")
is_metric.attach(evaluator, "is")



fid_values = []
is_values = []


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    evaluator.run(test_dataloader,max_epochs=1)
    metrics = evaluator.state.metrics
    fid_score = metrics['fid']
    is_score = metrics['is']
    fid_values.append(fid_score)
    is_values.append(is_score)
    print(f"Epoch [{engine.state.epoch}/5] Metric Scores")
    print(f"*   FID : {fid_score:4f}")
    print(f"*    IS : {is_score:4f}")

"""
