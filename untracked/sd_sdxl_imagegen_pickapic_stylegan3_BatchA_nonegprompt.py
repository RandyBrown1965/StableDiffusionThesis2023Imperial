import time
start_time = time.time()
print("Loading libraries")
import os
import gc
import random
import pickapic
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
#from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionXLInpaintPipeline
#from diffusers import DDIMScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, PNDMScheduler, DDPMScheduler, EulerAncestralDiscreteScheduler
from diffusers.models import AutoencoderKL
import xformers
import torch
import numpy as np
from torchvision import transforms
import pickle
from PIL import Image
#from inception_score_model import get_inception_score
print("Libraries Loading time = ",int(time.time()-start_time)," seconds")
#schedulers = [DDIMScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, PNDMScheduler, DDPMScheduler, EulerAncestralDiscreteScheduler]


#### PARAMETERS
seed = 0 #33 #BatchA=119
num_inference_steps = 50
num_images_per_prompt = 1 # 10
num_inpainting_iterations = 0
cfg = 8.0 #7.5 # Classifier Free Guidance Scale
denoising_strength=0.7
height, width = 512, 512
crop_size = int(height/16)
evaluator = "pickapic" # ["stylegan3", "pickapic"]
high_noise_frac_generate = 0.8
high_noise_frac_inpaint = 0.7

dtype=torch.float16
randomgenerator = torch.Generator(device="cuda") 
randomgenerator.manual_seed(seed)

###### MODEL PARAMETERS
#
# IMAGE GENERATION MODELS
compvis_1_4="CompVis/stable-diffusion-v1-4"
runwayml_1_5="runwayml/stable-diffusion-v1-5" # https://huggingface.co/runwayml/stable-diffusion-v1-5  \cite{Rombach_2022_CVPR}
dreamlikephotoreal_2_0="dreamlike-art/dreamlike-photoreal-2.0" # based on StableDiffusion_v1.5 contact@dreamlike.art
stabilityai_2_1_base = "stabilityai/stable-diffusion-2-1-base"
stabilityai_2_1 = "stabilityai/stable-diffusion-2-1" # 768x768 images
icbinp="/homes/rdb121/Thesis/models/icbinpICantBelieveIts_final.safetensors" # I Can't Believe It's Not Photography
sdxl_09="stabilityai/stable-diffusion-xl-base-0.9"
sdxl_1="stabilityai/stable-diffusion-xl-base-1.0"
# REFINER (FOR SDXL, GENERATION AND INPAINTING)
sdxl_refiner_09="stabilityai/stable-diffusion-xl-refiner-0.9"
sdxl_refiner_1="stabilityai/stable-diffusion-xl-refiner-1.0"
# VAE (FOR NON-SDXL)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
# INPAINTING MODELS
runwayml_inpainting_1="runwayml/stable-diffusion-inpainting" # https://huggingface.co/runwayml/stable-diffusion-inpainting \cite{Rombach_2022_CVPR}
# NG #runwayml_inpainting_1_5_local = "/homes/rdb121/Thesis/models/sd-v1-5-inpainting.ckpt" # NG  # https://huggingface.co/runwayml/stable-diffusion-inpainting \cite{Rombach_2022_CVPR}
stabilityai_inpainting_2 = "stabilityai/stable-diffusion-2-inpainting"  # https://huggingface.co/stabilityai/stable-diffusion-2-1  \cite{Rombach_2022_CVPR}
stabilityai_inpainting_2_local = "/homes/rdb121/Thesis/models/stable-diffusion-2-inpainting_local" # https://huggingface.co/stabilityai/stable-diffusion-2-1  \cite{Rombach_2022_CVPR}
sdxl_inpainting_1 = "stabilityai/stable-diffusion-xl-base-1.0"
#
# SPECIFY WHICH MODELS TO CHOOSE
#generative_model_id = runwayml_1_5
#generative_model_id = stabilityai_2_1_base
generative_model_id = sdxl_1
refiner_model_id = sdxl_refiner_1
use_refiner = True
use_inpainting_refiner = True
inpainting_model_id = runwayml_inpainting_1
#inpainting_model_id = sdxl_inpainting_1
inpainting_refiner_model_id = sdxl_refiner_1
#
use_refiner = use_refiner and ("stable-diffusion-xl" in generative_model_id)
use_inpainting_refiner = use_refiner and ("stable-diffusion-xl" in inpainting_model_id)


###### PROMPTS
prompt = "Brunette Holding a Book"
#prompt = "a lion leaping off a rock, midnight, rim lit"
#prompt = "Spider-man swinging from a strand of web, moving towards camera"
#prompt = "young woman, highlight hair, sitting outside restaurant, wearing dress" 
#prompt = "photo, a church in the middle of a field of crops, bright cinematic lighting, gopro, fisheye lens"
#
##### PROMPT PREFIX
#prompt_prefix = "RAW photo, " #https://stablediffusionapi.com/models/realistic-vision-v13
prompt_prefix = "photo of "
#
#### PROMPT_SUFFIX
#prompt_suffix = ", high quality, photo-real, cinematic lighting, ultra-real, 50mm, realistic proportions, finely detailed"
prompt_suffix = ", (high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3" #https://stablediffusionapi.com/models/realistic-vision-v13 #BEST!
#prompt_suffix = ", rim lighting, studio lighting, looking at camera, dslr, ultra quality, sharp focus, tack sharp, dof, film grain, Fujifilm XT5, crystal clear, 8K UHD, highly detailed glossy eyes, high detailed skin, skin pores, hyperrealistic, pixabay, beautiful hands, detailed fingers" # stable-diffusion-art.com
#prompt_suffix = ", ((ultra high quality)), ((photo-realistic)), 50mm, realistic proportions, sharp focus, dof, studio lighting, 8k, dslr, film grain, Fujifilm XT5, highly detailed, skin pores, ((beautiful hands)), ((beautiful eyes)), detailed fingers, detailed background, (open eyes)" # my combination
#prompt_suffix = ", (ultra high quality), (photo-realistic), 50mm, realistic proportions, sharp focus, dof, studio lighting, 8k, dslr, film grain, Fujifilm XT3, highly detailed skin pores, ((beautiful hands)), ((beautiful eyes)), detailed background, (open eyes)" # my combination
#
# TEXT EMBEDDINGS
#text_embeddings = [" <Style-Empire> "," <kodakvision_500T>, "," <kojima> "," <doose-realistic> "," <ganyu> "," <xyz> ", " <Frank Franzetta> "]
#
filename_prompt = prompt.replace(" ","").replace(",","")
#
prompt_total = prompt_prefix + prompt + prompt_suffix
#prompt_total = prompt_prefix + prompt + prompt_suffix + text_embeddings

# NEGATIVE PROMPTS
#negative_prompt = "low quality disfigured bad gross disgusting mutation ugly morbid mutated deformed mutilated mangled poorly drawn face extra limb missing limb floating limbs disconnected limbs malformed limbs oversaturated duplicate bodies cloned faces low-res blurry blur out of focus out of frame extra missing"
#negative_prompt = "(deformed iris), (deformed pupils), semi-realistic, cgi, 5d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, (((extra fingers))), mutated hands, (((poorly drawn hands))), poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, extra hands, fused fingers, too many fingers, long neck, overly-saturated" #https://stablediffusionapi.com/models/realistic-vision-v13
#negative_prompt = "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature" # https://huggingface.co/spaces/stabilityai/stable-diffusion/discussions/7858  Dan Mulheran
#negative_prompt = "disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w" # stable-diffusion-art.com
negative_prompt = "low quality, disfigured, mutation, ugly, deformed, malformed, cartoon, bad anatomy, extra arm, missing arm, extra leg, missing leg, (((extra hand, extra fingers))), missing fingers, fused fingers, (((poorly drawn hands, poorly drawn fingers, poorly drawn face, malformed hands, malformed fingers))), malformed limbs, malformed face, long neck, overly-saturated, (((out of frame, cropped, text))), letters, watermark, (((closed eyes, frame, border)))" # my combination


# DISCRIMINATORS
pre_load_time = time.time()
stylegan_models = ['ffhq.pkl', 'cifar10.pkl', 'afhqwild.pkl'] # These models want Tensorflow
stylegan3_models = ['stylegan3-r-afhqv2-512x512.pkl', 'stylegan3-t-ffhqu-256x256.pkl', 'stylegan3-r-ffhqu-256x256.pkl']
stylegan3_models = ['stylegan2-ffhq-512x512.pkl','stylegan2-cifar10-32x32.pkl' ]  # The cifar10-32x32 model generates an error
#discriminator_model = 'stylegan2-ffhq-512x512.pkl' # This doesn't need scipy
discriminator_model = 'stylegan3-r-afhqv2-512x512.pkl' # This needs scipy
discriminator_size = int(discriminator_model.split("x")[-1].split(".")[0])
print("Discriminator = ",discriminator_model," size = ",discriminator_size)
#patch_discriminator_model = 'stylegan3-r-ffhqu-256x256.pkl'
patch_discriminator_model = discriminator_model
patch_discriminator_size = int(patch_discriminator_model.split("x")[-1].split(".")[0])
print("Patch Discriminator = ",patch_discriminator_model," size = ",patch_discriminator_size)
#
# LOAD STYLEGAN3 DISCRIMINATOR
with open('/vol/bitbucket/rdb121/models/'+discriminator_model,'rb') as f:
    #G = pickle.load(f)['Gema'].cuda()
    #G = pickle.load(f)['G'].cuda()
    D = pickle.load(f)['D']#.cuda()
with open('/vol/bitbucket/rdb121/models/'+patch_discriminator_model,'rb') as f:
    D_patch = pickle.load(f)['D']#.cuda()
c = None # class labels
#z=torch.randn([1, G.z_dim]).cuda()
#img = G(z, c)
print("Time to load discriminators = ", int(time.time()-pre_load_time), " seconds")

# LOAD REF IMAGE FOR PICKAPIC
image_reference = Image.open("images_varied/SDXL_astronaut_jungle_refined_photo.png")

#########################################
#
# SUBROUTINES
#
########################################

def resize_image_tensor(image_in_tensor, out_size: int): 
    if image_in_tensor.shape[0] == out_size: # assume square images
        return image_in_tensor
    image_unresized = transforms.ToPILImage()(image_in_tensor)
    image_resized = image_unresized.resize((out_size, out_size))
    tensor_resized = transforms.ToTensor()(image_resized)
    return tensor_resized


def enlarge_mask(mask_in, size_out= 512):
    xdim, ydim = mask_in.shape
    mask_array = np.zeros((size_out, size_out), dtype=np.uint8)
    block = int(size_out/xdim)
    for x in range(xdim):
        for y in range(ydim):
            mask_array[x*block:(x+1)*block, :][:, y*block:(y+1)*block] = np.ones((block, block)) * 255 * mask_in[x][y]
    return(mask_array)

####################################################################################

def compute_patch_mask(image_full_tensor, chunk_size: int):
    _, xdim_in, ydim_in = image_full_tensor.shape
    print("Full Image=",xdim_in,"x",ydim_in)
    xdim_out, ydim_out = xdim_in//chunk_size, ydim_in//chunk_size
    computed_mask = np.zeros((xdim_out, ydim_out))
    print("mask=",computed_mask.shape)

    for x in range(0, xdim_in, chunk_size):
        for y in range(0, ydim_in, chunk_size):
            image_crop_tensor = image_full_tensor[:, x: x+chunk_size, y: y+chunk_size]
            image_crop = transforms.ToPILImage()(image_crop_tensor)
            evaluation = calc_probs(prompt, [image_crop, image_brunette])
            computed_mask[int(x/chunk_size), int(y/chunk_size)] = int(evaluation[0] * 100)
    return computed_mask   

def compute_stride_mask(image_full_tensor,chunk_size: int):
    _, xdim_in, ydim_in = image_full_tensor.shape
    print("Full Image=",xdim_in,"x",ydim_in)
    xdim_out, ydim_out = xdim_in//chunk_size, ydim_in//chunk_size
    evaluations_total = np.zeros((xdim_out, ydim_out))
    evaluations_count = np.zeros((xdim_out, ydim_out), dtype = int)
    print("mask=",evaluations_total.shape)

    stride = chunk_size
    #crop_size_x, crop_size_y = xdim_in//2, ydim_in//2
    crop_size_x, crop_size_y = 2*chunk_size, 2*chunk_size
    ones_array = np.ones((crop_size_x // chunk_size, crop_size_y // chunk_size), dtype = int)
    for x in range(0, xdim_in - crop_size_x + 1 , chunk_size):
        for y in range(0, ydim_in - crop_size_y + 1, chunk_size):
            image_crop_tensor = image_full_tensor[:, x: x+crop_size_x, y: y+crop_size_y]
            image_crop = transforms.ToPILImage()(image_crop_tensor)
            #image_crop.save("image_crop_"+str(x)+str(y)+".png")
            evaluation = calc_probs(prompt, [image_crop, image_brunette])
            evaluations_count[x // chunk_size: (x+crop_size_x) // chunk_size, y // chunk_size: (y+crop_size_y) // chunk_size] += ones_array
            evaluations_total[x // chunk_size: (x+crop_size_x) // chunk_size, y // chunk_size: (y+crop_size_y) // chunk_size] += int(evaluation[0] * 100) * ones_array
            print("evaluations_count = \n", evaluations_count)
            print("evaluations_total = \n", (evaluations_total).astype(int))
    computed_mask = evaluations_total / evaluations_count
    return computed_mask   

####################################################################################

def pick_best_image(images: list, filename_suffix: str = ""):
    # EVALUATE THE MULTIPLE IMAGES AND CHOOSE THE BEST ONE
    evaluation_best = -np.inf
    for i in range(len(images)):
        image_tensor = transforms.ToTensor()(images[i])
        tensor_resized = resize_image_tensor(image_tensor, discriminator_size)
        tensor_resized.unsqueeze_(0)
        """
        # TRY TO PUT THE DISCRIMINATOR ON THE GPU
        if torch.cuda.is_available():
            print("Transferring to cuda")
            print("tensor is on cuda? ",tensor_resized.is_cuda)
            print("D is on cuda? ",next(D.parameters()).is_cuda)
            #D.to("cuda")
            #tensor_resized.to("cuda")
            #print("tensor is on cuda? ",tensor_resized.is_cuda)
            #print("D is on cuda? ",next(D.parameters()).is_cuda)
            D = D.to("cuda")
            tensor_resized = tensor_resized.to("cuda")
            print("tensor is on cuda? ",tensor_resized.is_cuda)
            print("D is on cuda? ",next(D.parameters()).is_cuda)
        """
        logits = D(tensor_resized, c)
        lossG = torch.nn.functional.softplus(-logits)
        lossD = torch.nn.functional.softplus(logits)
        pickapic_prob = pickapic.calc_probs("(((hyperrealistic, photographic realism)))", [images[i], image_reference])
        print("Full image ",i," logit = ",logits, "  lossG=",lossG,"  lossD=",lossD, "pickapic prob=",pickapic_prob[0])
        if evaluator == "stylegan3":
            evaluation = lossD
        elif evaluator == "pickapic":
            evaluation = pickapic_prob[0]
        if evaluation > evaluation_best:
            evaluation_best, index_best, image_best_tensor = evaluation, i, image_tensor
        # Save each image
        filename_out = filename_suffix + "_P" + str(pickapic_prob[0]) + "_D" + str(lossD.numpy()[0][0]) + ".png"
        filename_out = filename_out.replace("GENERATION-NUMBER", str(i))
        filename_out = filename_out.replace("RANDOM-NUMBER", str(random.random()))
        images[i].save(filename_out)
    return images[index_best], image_best_tensor
            
###############################################################################
#
#  MAIN PROGRAM
#  GENERATE IMAGES TO TEST DISCRIMINATOR AND PICK-A-PIC AGAINST PUBLIC OPINION
#
################################################################################
dir_name = "images_out_" + str(time.time())
os.mkdir(dir_name)

# LOOP THROUGH MODELS
for generative_model_id in [runwayml_1_5, stabilityai_2_1, sdxl_1, sdxl_refiner_1]:
  #for inpainting_model_id in [runwayml_inpainting_1, stabilityai_inpainting_2, sdxl_inpainting_1, sdxl_refiner_1]:
    if generative_model_id == sdxl_refiner_1:
        generative_model_id = sdxl_1
        use_refiner = True
    else:
        use_refiner = False
    if inpainting_model_id == sdxl_refiner_1:
        inpainting_model_id = sdxl_inpainting_1
        use_inpainting_refiner = True
    else:
        use_inpainting_refiner = False
    # PRINT MODEL SUMMARY
    print("\nGENERATOR = ",generative_model_id)
    if use_refiner:
        print("REFINER = ",refiner_model_id)
    if num_inpainting_iterations > 0:
        print("EVALUATOR = ", evaluator)
        print("INPAINTER = ",inpainting_model_id)
        if use_inpainting_refiner:
            print("INPAINTING REFINER = ",inpainting_refiner_model_id)

    
    # LOAD SD IMAGE GENERATOR 
    generator = 0
    del generator
    preload_time = time.time()
    if "xl" in generative_model_id:
        generator = DiffusionPipeline.from_pretrained(
            generative_model_id,
            local_files_only = generative_model_id.startswith("/"),
            use_safetensors=True,
            variant="fp16",
            #vae=vae,
            torch_dtype=dtype,
            )
    else:
        generator = StableDiffusionPipeline.from_pretrained(
            generative_model_id,
            local_files_only = generative_model_id.startswith("/"),
            revision="fp16",
            vae=vae,
            torch_dtype=dtype,
            )
    #NG generator.enable_xformers_memory_efficient_attention() # NG # use for torch<2.0 according to https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
    #NG generator.unet = torch.compile(generator.unet, mode="reduce-overhead", fullgraph=True)  #  Needs torch>=2.0 but NG
    #generator.to("cuda")
    generator.enable_model_cpu_offload()
    print("Loading image generation model took ",int(time.time()-preload_time)," seconds")
    
    # LOAD REFINER
    preload_time = time.time()
    if use_refiner:
        refiner = 0
        del refiner
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=generator.text_encoder_2,
            vae=generator.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        #NG refiner.enable_xformers_memory_efficient_attention() #NG# use for torch<2.0 according to https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl
        #NG refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)  #  Needs torch>=2.0 #NG
        #refiner.to("cuda")
        refiner.enable_model_cpu_offload()
    print("Loading image refiner model took ",int(time.time()-preload_time)," seconds")

    # LOAD INPAINTER
    if num_inpainting_iterations:
        premodel=time.time()
        inpainter = 0
        del inpainter
        if "runwayml" in inpainting_model_id:
            revision="fp16" # necessary for runwayml_inpainting?
        else:
            revision = None
        if "xl" in inpainting_model_id:
            inpainter = StableDiffusionXLInpaintPipeline.from_pretrained(
                inpainting_model_id,
                local_files_only = inpainting_model_id.startswith("/"),
                torch_dtype=dtype,
                variant = "fp16",
                use_safetensors = True
                )
        else:
            inpainter = StableDiffusionInpaintPipeline.from_pretrained(
                inpainting_model_id,
                local_files_only = inpainting_model_id.startswith("/"),
                torch_dtype=dtype,
                revision=revision,
                vae=vae,
                )
        #inpainter.to("cuda")
        inpainter.enable_model_cpu_offload()
        #inpainter.unet = torch.compile(inpainter.unet, mode="reduce-overhead", fullgraph=True)  #  Needs torch>=2.0 #NG
        if use_inpainting_refiner:
            inpainting_refiner = 0
            del inpainting_refiner
            inpainting_refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=inpainter.text_encoder_2,
                vae=inpainter.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            #refiner.to("cuda")
            inpainting_refiner.enable_model_cpu_offload()
        print("Loading inpainting model took = ", int(time.time()-premodel)," seconds")
    
    # LOOP THROUGH ALL THE PROMPTS FOR GENERATION/REFINEMENT/INPAINTING
    #for prompt in ["brunette holding a book", "child on a bench", "closeup of attractive man", "bedroom", "church", "cat", "dog", "car", "flowers", "horse"]:
    for prompt in ["brunette holding a book", "child on a bench", "closeup of attractive man", "bedroom", "church", "cat", "car", "horse", "steampunk lizard in lab coat and glasses"]: #BATCHA
        filename_prompt = prompt.replace(" ","").replace(",","")
        #for extend_prompt in [False, True]:
        for extend_prompt in [True]:
            if extend_prompt:
                prompt_total = prompt_prefix + prompt + prompt_suffix
            else:
                prompt_total = prompt_prefix + prompt

            ########################################
            #
            # IMAGE GENERATION 
            #
            ######################################
            pregeneration_time = time.time()
            randomgenerator.manual_seed(seed)
            """
            for scheduler in schedulers:
                generator.scheduler = scheduler.from_config(generator.scheduler.config)
            """
            for num_inference_steps in [50]:
                options = dict(
                    prompt=prompt_total,
                    #negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=num_images_per_prompt,
                    guidance_scale = cfg, #default = 8.0 for SDXL
                    generator=randomgenerator,
                    height = height, # default = 512
                    width = width, # default = 512
                    output_type="latent" if use_refiner else "pil",
                    #image=image, # for inpainting
                    #mask_image=mask_image, # for inpainting
                    #strength= denoising_strength # 0-1, 0=no change, default=1.0  for inpainting
                    )
                if use_refiner:
                    options['denoising_end'] = high_noise_frac_generate
                #print("\n\nOPTIONS = ",options)
                images_gen = generator(**options).images
                print("Generation time = ", int(time.time()-pregeneration_time)," seconds")
                print(len(images_gen)," images to choose from")
            
                #############################
                #
                # REFINER
                #
                #############################
                if use_refiner:
                    prerefiner_time = time.time()
                    images = []
                    for i in range(len(images_gen)):
                        image_gen = images_gen[i]
                        image_gen.to("cuda")
                        #image_refiner = refiner(prompt=prompt_total, image=image_gen[None, :]).images[0]
                        image_refiner = refiner(prompt=prompt_total, image=image_gen[None, :],
                                denoising_start=high_noise_frac_generate).images[0]
                        images.append(image_refiner)
                    print("Refiner time = ", int(time.time()-prerefiner_time)," seconds")
                else:
                    images = images_gen
            
                """
                for i in range(len(images)):
                    filename_out = str(filename_prompt)+"_" + generative_model_id.split('/')[-1].replace("diffusion-2","diffusion-v2")\
                            + "_seed"+str(seed) + "_"+ str(num_inference_steps) + "steps_"+str(i) +"_cfg"+str(cfg) + "_"+str(time.time())+".png"
                    if use_refiner:
                        filename_out = filename_out.replace("base","refiner")
                    images[i].save(filename_out)
                """
            
            #generator.to("cpu")
            #generator.to("cuda")
            #generator.enable_model_cpu_offload()
            """
            del generator
            if use_refiner:
                del refiner
            """
            gc.collect()
            ########################################
            #
            # GAN EVALUATION
            # https://github.com/NVlabs/stylegan3 \cite{Karras2021}
            #
            # PICKAPIC EVALUATION
            #
            ######################################
            image_out_progression = []
            mask_out_progression = []
            threshold = None
            boolean_mask = None
            #
            # Write out all the images and pick the best one.
            filename_suffix = dir_name + "/" + str(filename_prompt)+"superprompt_GENERATION-NUMBER_RANDOM-NUMBER_"\
                    + generative_model_id.split('/')[-1].replace("diffusion-2","diffusion-v2")\
                    + "_seed"+str(seed) + "_"+ str(num_inference_steps) + "steps_cfg"+str(cfg) 
            if use_refiner:
                filename_suffix = filename_suffix.replace("base","refiner")
            if not extend_prompt:
                filename_suffix = filename_suffix.replace("superprompt_", "_")
            image_out, image_out_tensor = pick_best_image(images, filename_suffix)
            image_out_progression.append(image_out)
            # OUTPUT OF IMAGE GENERATION IS PASSED TO INPAINTER AS image_out_tensor
            
            # LOOP THROUGH INPAINTING ITERATIONS    
            for inpainting_iteration in range(num_inpainting_iterations):
                print("Inpainting iteration ",inpainting_iteration+1," of ",num_inpainting_iterations)
                #######################################
                #
                # CREATE INPAINTING MASK
                # USING EVALUATOR(S)
                #
                #######################################
                pre_evaluation = time.time()
                _, xdim_in, ydim_in = image_out_tensor.shape
                print("Image=",xdim_in,"x",ydim_in)
                xdim_out, ydim_out = xdim_in//crop_size, ydim_in//crop_size
                computed_mask = np.zeros((xdim_out, ydim_out))
                print("mask=",computed_mask.shape)
                resized_array = np.zeros((xdim_out, ydim_out, 3),dtype=np.uint8)
                # CUT THE INPUT IMAGE INTO CHUNKS
                for x in range(0, xdim_in, crop_size):
                    for y in range(0, ydim_in, crop_size):
                        image_crop = image_out_tensor[:, x:x + crop_size, y:y+crop_size]
                        image_crop = resize_image_tensor(image_crop, patch_discriminator_size)
                        if evaluator == "stylegan3":
                            # STYLEGAN3 DISRIMINATOR EVALUATION
                            logit = D_patch(image_crop.unsqueeze_(0), c)
                            lossD = torch.nn.functional.softplus(logit)
                            evaluation = lossD
                        elif evaluator == "pickapic":
                            # PICKAPIC EVALUATION
                            image_crop_pil = transforms.ToPILImage()(image_crop)  # for pickapic
                            pickapic_prob = pickapic.calc_probs("(((hyperrealistic, photographic realism)))", [image_crop_pil, image_reference]) 
                            evaluation = pickapic_prob[0]
                        computed_mask[int(x/crop_size), int(y/crop_size)] = int(evaluation*1000) / 1000

                print("Mask evaluation time = ", int(time.time()-pre_evaluation)," seconds")
                mean, std = np.mean(computed_mask), np.std(computed_mask)
                print("Computed mask: mean=", mean," std =", std)
                if threshold == None: # SET THRESHOLD ON THE FIRST TIME THROUGH
                    threshold = mean 
                    #boolean_mask = np.ones((xdim_out, ydim_out))                       # THESE TWO LINES ENSURE THAT THE MASK GETS SMALLER
                #boolean_mask = (computed_mask < threshold).astype(int) * boolean_mask  # THESE TWO LINES ENSURE THAT THE MASK GETS SMALLER
                boolean_mask = (computed_mask < threshold).astype(int)
                print("Boolean mask =\n", boolean_mask)
                
                
                ########################################
                #
                # INPAINTING
                #
                ######################################
                # Define a Random mask and a solid mask (if needed)
                mask_array_16x16_random = np.random.randint(0, 2, boolean_mask.shape, dtype=np.uint8)
                mask_array_16x16_solid = np.ones(boolean_mask.shape, dtype=np.uint8)
                # Enlarge and convert our mask
                mask_array = enlarge_mask(boolean_mask, height)
                mask_image = Image.fromarray(mask_array)
                mask_out_progression.append(mask_image)
                
                #################### 
                # RUN THE INPAINTER
                #################### 
                pregeneration_time = time.time()
                options = dict(
                    prompt=prompt_total,
                    #negative_prompt=negative_prompt, 
                    num_inference_steps=num_inference_steps,
                    num_images_per_prompt=num_images_per_prompt,
                    guidance_scale = cfg, #default = 8.0 for SDXL
                    generator=randomgenerator,
                    height = height, # default = 512
                    width = width, # default = 512
                    image=image_out_progression[-1], # for inpainting
                    mask_image=mask_out_progression[-1], # for inpainting
                    strength= denoising_strength, # 0-1, 0=no change, default=1.0  for inpainting
                    output_type="latent" if use_inpainting_refiner else "pil",
                    )
                if use_inpainting_refiner:
                    options['denoising_end'] = high_noise_frac_inpaint # for inpainting with refiner
                images_inpaint = inpainter(**options).images
                print("Inpainting time = ",int(time.time()-pregeneration_time)," seconds")

                #################### 
                # RUN THE REFINER
                #################### 
                if use_inpainting_refiner:
                    prerefiner_time = time.time()
                    images = []
                    for i in range(len(images_inpaint)):
                        image_inpaint = images_inpaint[i]
                        image_inpaint.to("cuda")
                        options = dict(
                            prompt=prompt_total,
                            #negative_prompt=negative_prompt, 
                            num_inference_steps=num_inference_steps,
                            num_images_per_prompt=1,
                            guidance_scale = cfg, #default = 8.0 for SDXL
                            generator=randomgenerator,
                            height = height, # default = 512
                            width = width, # default = 512

                            image=image_inpaint[None, :], # for inpainting
                            mask_image=mask_out_progression[-1], # for inpainting
                            strength= denoising_strength, # 0-1, 0=no change, default=1.0  for inpainting
                            denoising_start=high_noise_frac_inpaint, # for inpainting refiner
                            )
                        """
                        image_inpainting_refiner = inpainting_refiner(prompt=prompt_total, image=image_inpaint[None, :],
                                denoising_start=high_noise_frac_inpaint).images[0]
                        """
                        image_inpainting_refiner = inpainting_refiner(**options).images[0]
                        images.append(image_inpainting_refiner)
                    print("Inpaint Refiner time = ", int(time.time()-prerefiner_time)," seconds")
                else:
                    images = images_inpaint

                #  Create filename
                filename_suffix = "Inpaint_" + str(filename_prompt) + "superprompt_"\
                        + generative_model_id.split('/')[-1].replace("diffusion-2","diffusion-v2")\
                        + "_GENERATION-NUMBER_RANDOM-NUMBER_inpaint_"\
                        + inpainting_model_id.split('/')[-1].replace("_","").replace("stablediffusion","sd")\
                        + "_iteration"+ str(inpainting_iteration+1)\
                        + "_seed"+str(seed) + "_"+ str(num_inference_steps) + "steps_cfg"+str(cfg) 
                if use_inpainting_refiner:
                    filename_suffix = filename_suffix.replace("_seed","refined_seed")
                if not extend_prompt:
                    filename_suffix = filename_suffix.replace("superprompt_", "_")

                # PICK THE BEST INPAINTED IMAGE
                image_out, image_out_tensor = pick_best_image(images, filename_suffix)
                image_out_progression.append(image_out)
                print("Total time = ", int(time.time()-start_time)," seconds\n")
                # EACH ITERATION OF INPAINTING IS PASSED TO THE NEXT AS image_out_tensor

            
            # Recap the progression after all the inpainting is done.
            if len(image_out_progression):
                # EVALUATE THE FINAL INPAINTED IMAGE
                for i,image in enumerate(image_out_progression):
                    image.save("Output_noneg_"+ str(filename_prompt)\
                            + "_inpainter-" + generative_model_id.split('/')[-1]\
                            + "_image_" + str(i) + "_strength"+ str(denoising_strength) + "_" + str(time.time()) + ".png")
                for i,image in enumerate(mask_out_progression):
                    image.save("Output_noneg_"+ str(filename_prompt)\
                            + "_inpainter-" + generative_model_id.split('/')[-1]\
                            + "_mask_" + str(i) + "_strength"+ str(denoising_strength) + "_" + str(time.time()) + ".png")
                """
                image_out_tensor = transforms.ToTensor()(image_out_progression[-1])
                tensor_resized = resize_image_tensor(image_out_tensor, discriminator_size)
                tensor_resized.unsqueeze_(0)
                logits = D(tensor_resized, c)
                print("\nInpainted image logit = ",logits)
                """


