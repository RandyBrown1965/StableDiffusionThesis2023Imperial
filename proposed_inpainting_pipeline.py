#
# GENERATE IMAGES WITH EACH GENERATOR AND FEED THE SAME IMAGES TO EACH INPAINTER FOR HUMAN EVALUATION
#
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--generator', choices=['sd-v1.5', 'sd-v2.1', 'sdxl-base', 'sdxl-refiner'], default=['sd-v1.5'], nargs="+")
parser.add_argument('--images_per_prompt', default=1, type = int)
parser.add_argument('--inpainter', choices=['sd-v1', 'sd-v2', 'sdxl-base', 'sdxl-refiner'], default=['sd-v2'], nargs= "+")
parser.add_argument('--num_inpainting_iterations', default=3, type = int)
parser.add_argument('--random_seed', default=0, type = int)
parser.add_argument('--num_chunks', default=16, type = int)
parser.add_argument('--chunks_per_crop', default=2, type = int)
parser.add_argument('--prompt', nargs="+")
parser.add_argument('--solid_mask', action="store_true")
parser.add_argument('--random_mask', action="store_true")
parser.add_argument('--use_pregenerated', default=None)
parser.add_argument('--mask_threshold', default=None)
args = parser.parse_args()
# Display the parameters and check for possible errors before loading libraries.
threshold = args.mask_threshold
if threshold is not None:
    threshold = float(threshold)
num_images_per_prompt = args.images_per_prompt
num_inpainting_iterations = args.num_inpainting_iterations
if (args.use_pregenerated is not None):
    print("\nUSE_PREGENERATED IMAGE ",args.use_pregenerated)
print("generator = ", args.generator)
print("inpainter = ", args.inpainter)
print("num_images_per_prompt = ", num_images_per_prompt)
print("num_inpainting_iterations = ", num_inpainting_iterations)
print("prompt = ", args.prompt)
print("num_chunks = ", args.num_chunks)
print("chunks_per_crop = ", args.chunks_per_crop)
print("random_seed = ", args.random_seed)
if args.solid_mask:
    print("SOLID MASK")
if args.random_mask:
    print("RANDOM MASK")
assert not (args.solid_mask and args.random_mask), "You cannot specify a random_mask and a solid_mask simultaneously."
if args.use_pregenerated is not None:
    assert os.path.isfile(args.use_pregenerated), args.use_pregenerated+" is not a valid path to a file."

import time
start_time = time.time()
print("Loading libraries")
import utils
#import utils2 # This variant saves out all images that are submitted to pick_best_image()
import gc
import random
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
#from diffusers import StableDiffusionXLPipeline
from diffusers import StableDiffusionXLInpaintPipeline
#from diffusers import DDIMScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, PNDMScheduler, DDPMScheduler, EulerAncestralDiscreteScheduler
from diffusers.models import AutoencoderKL
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
#import xformers
#import pickle
#import pickapic
#from inception_score_model import get_inception_score
print("Libraries Loading time = ",int(time.time()-start_time)," seconds")
#schedulers = [DDIMScheduler, LMSDiscreteScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, PNDMScheduler, DDPMScheduler, EulerAncestralDiscreteScheduler]


#### PARAMETERS
#random_seed = 0 #33 #BatchA=119  # GET THIS FROM THE PASSED ARGUMENTS
num_inference_steps = 50
#num_images_per_prompt = 1 # 10 # GET THIS FROM THE PASSED ARGUMENTS
#num_inpainting_iterations = 1 # GET THIS FROM THE PASSED ARGUMENTS
cfg = 8.0 #7.5 # Classifier Free Guidance Scale
denoising_strength=0 #0.7 THIS IS SET TO DECRASE WITH INPAINGING ITERATION 0.75, 0.5, 0.25, 0 ...
height, width = 512, 512
evaluator = "pickapic" # ["stylegan3", "pickapic"]
high_noise_frac_generate = 0.8
high_noise_frac_inpaint = 0.7

dtype=torch.float16
randomgenerator = torch.Generator(device="cuda") 
randomgenerator.manual_seed(args.random_seed)

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
generator_dict = {"sd-v1.5": (runwayml_1_5, "gen-sd-v1.5", False, None),\
        "sd-v2.1": (stabilityai_2_1_base, "gen-sd-v2.1", False, None),\
        "sdxl-base": (sdxl_1, "gen-sdxl-base", False, None),\
        "sdxl-refiner": (sdxl_1, "gen-sdxl-refiner", True, sdxl_refiner_1)
        }
inpainter_dict = {"sd-v1": (runwayml_inpainting_1, "sd-v1", False, None),\
        "sd-v2": (stabilityai_inpainting_2, "sd-v2", False, None),\
        "sdxl-base": (sdxl_inpainting_1, "sdxl-base", False, None),\
        "sdxl-refiner": (sdxl_inpainting_1, "sdxl-refiner", True, sdxl_refiner_1)
        }

"""  IGNORE THE DIRECT DESIGNATION, NOW WE CHHOSE THE GENERATOR AND INPAINTER VIA COMMAND LINE OPTIONS
# SPECIFY WHICH MODELS TO CHOOSE DIRECTLY
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
"""

"""  IGNORE THE DIRECT DESIGNATION, NOW WE CHHOSE THE PROMPT(S) VIA COMMAND LINE OPTIONS
###### PROMPTS
prompt = "Brunette Holding a Book"
#prompt = "a lion leaping off a rock, midnight, rim lit"
#prompt = "Spider-man swinging from a strand of web, moving towards camera"
#prompt = "young woman, highlight hair, sitting outside restaurant, wearing dress" 
#prompt = "photo, a church in the middle of a field of crops, bright cinematic lighting, gopro, fisheye lens"
"""
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
# NEGATIVE PROMPTS
#negative_prompt = "low quality disfigured bad gross disgusting mutation ugly morbid mutated deformed mutilated mangled poorly drawn face extra limb missing limb floating limbs disconnected limbs malformed limbs oversaturated duplicate bodies cloned faces low-res blurry blur out of focus out of frame extra missing"
#negative_prompt = "(deformed iris), (deformed pupils), semi-realistic, cgi, 5d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, (((extra fingers))), mutated hands, (((poorly drawn hands))), poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, extra hands, fused fingers, too many fingers, long neck, overly-saturated" #https://stablediffusionapi.com/models/realistic-vision-v13
#negative_prompt = "out of frame, lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature" # https://huggingface.co/spaces/stabilityai/stable-diffusion/discussions/7858  Dan Mulheran
#negative_prompt = "disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w" # stable-diffusion-art.com
negative_prompt = "low quality, disfigured, mutation, ugly, deformed, malformed, cartoon, bad anatomy, extra arm, missing arm, extra leg, missing leg, (((extra hand, extra fingers))), missing fingers, fused fingers, (((poorly drawn hands, poorly drawn fingers, poorly drawn face, malformed hands, malformed fingers))), malformed limbs, malformed face, long neck, overly-saturated, (((out of frame, cropped, text))), letters, watermark, (((closed eyes, frame, border)))" # my combination


###############################################################################
#
#  MAIN PROGRAM
#  GENERATE IMAGES TO TEST DISCRIMINATOR AND PICK-A-PIC AGAINST PUBLIC OPINION
#
################################################################################
#dir_name = "images_out_" + str(time.time())
#os.mkdir(dir_name)
generator, refiner, inpainter, inpainting_refiner = None, None, None, None
image_inpainting_refiner, images_inpaint = None, None

if (args.use_pregenerated is not None):
    generator_list, prompt_list = utils.get_generator_prompt_from_filepath(args.use_pregenerated)
else:
    generator_list = args.generator
    prompt_list = args.prompt
########################################################
#
# LOOP THROUGH MODELS
#
########################################################
for generator_choice in generator_list:
    generative_model_id, generative_model_string, use_refiner, refiner_model_id = generator_dict[generator_choice]

    if (args.use_pregenerated is None): # Don't load generator/refiner models when using a pre-generated image
        # LOAD SD IMAGE GENERATOR 
        preload_time = time.time()
        if generator is not None:
            del generator
            gc.collect()
            generator = None
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
        if use_refiner:
            preload_time = time.time()
            if refiner is not None:
                del refiner
                gc.collect()
                refiner = None
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
       
    # PRINT MODEL SUMMARY
    print("\nGENERATOR = ",generative_model_id)
    if use_refiner:
        print("REFINER = ",refiner_model_id)


    ########################################################
    #
    # LOOP THROUGH ALL THE PROMPTS FOR GENERATION/REFINEMENT/INPAINTING
    #
    ########################################################
    # DON'T DO EACH PROMPT FOR EACH GENERATOR OR WE'LL BURN OUT OUR HUMAN VOLUNTEERS
    #for prompt in ["brunette holding a book", "child on a bench", "closeup of attractive man", "bedroom", "church", "cat", "car", "horse", "steampunk lizard in lab coat and glasses"]: #BatchA
    #prompts_all = ["brunette holding a book", "child on a bench", "closeup of attractive man", "bedroom", "church", "cat", "dog", "car", "flowers", "horse", "airplane", "bird"] #BatchB
    for prompt in prompt_list:
        print("\nPROMPT = ",prompt,"\n")
        filename_prompt = prompt.replace(" ",":").replace(",","")
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
            gc.collect()
            pregeneration_time = time.time()
            randomgenerator.manual_seed(args.random_seed)
            #for scheduler in schedulers:                                                # USE THIS TO VARY THE SCHEDULERS
            #    generator.scheduler = scheduler.from_config(generator.scheduler.config) # USE THIS TO VARY THE SCHEDULERS
            #for num_inference_steps in [50]:           # USE THIS TO VARY THE NUMBER OF INFERENCE STEPS
            if (args.use_pregenerated is None): # Skip generation when using a pre-generated image.
                options = dict(
                    prompt=prompt_total,
                    negative_prompt=negative_prompt,
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
                        image_refiner = refiner(prompt=prompt_total, image=image_gen[None, :],
                                denoising_start=high_noise_frac_generate).images[0]
                        images.append(image_refiner)
                    print("Refiner time = ", int(time.time()-prerefiner_time)," seconds")
                else:
                    images = images_gen
                #generator.to("cpu")
                #generator.to("cuda")
                #generator.enable_model_cpu_offload()
                """
                del generator
                if use_refiner:
                    del refiner
                gc.collect()
                generator, refiner = None, None
                """

            # Set up lists of images and masks from the initial generation to final inpainting
            image_out_progression = []
            mask_out_progression = []
            """ Threshold is now set by input parameter, default = None -> will be set to mean of first evaluation
            #threshold = None
            #threshold = 0.23313 # CONSTANT THRESHOLD ESTABLISHED BY brunetteholdingabook_gen-sd-v1.5.png
            """
            boolean_mask = None
            #
            # Construct an informative filename for the generated output and save the image.
            if (args.use_pregenerated):
                filename_pregenerated = args.use_pregenerated
                image_out = Image.open(args.use_pregenerated)
            else:
                # Construct a filename in case we want to write out all the generated images
                filename_suffix = str(filename_prompt)+"superprompt_GENERATION-NUMBER_RANDOM-NUMBER_"\
                    + generative_model_string\
                    + "_seed"+str(args.random_seed) + "_"+ str(num_inference_steps) + "steps_cfg"+str(cfg) 
                if not extend_prompt:
                    filename_suffix = filename_suffix.replace("superprompt_", "_")
                # Pick the best image from all the generated images.
                image_out = utils.pick_best_image(images, evaluator, filename_suffix)
                # Write it out for the inpainter(s) to pick up.  We want all the inpainters working on the same input.
                directory_name = "images_out_GeneratorOutput_preThesisPaper_" + filename_prompt + "_" + str(int(time.time()))
                if not os.path.exists(directory_name):
                    os.mkdir(directory_name)
                filename_pregenerated = directory_name + "/" + filename_prompt + "_"\
                        + generative_model_string + "_" + str(random.random())[2:14]\
                        + "_seed" + str(args.random_seed) + ".png"
                # ONLY WRITE OUT THE INPUT THE FIRST TIME THROUGH. DON'T OVERWRITE.
                if not os.path.isfile(filename_pregenerated):
                    image_out.save(filename_pregenerated)
            # OUTPUT OF IMAGE GENERATION (OR PREGENERATION) IS PASSED TO INPAINTER AS AN IMAGE SAVED TO filename_pregenerated
            image_out_progression.append(image_out)


            ########################################################
            #
            # LOOP THROUGH ALL THE INPAINTERS
            #
            ########################################################
            #for inpainting_model_id in [sdxl_refiner_1, sdxl_inpainting_1, stabilityai_inpainting_2, runwayml_inpainting_1]: # Go backwards to avoid memory errors
            for inpainter_choice in args.inpainter:
                inpainting_model_id, inpainting_model_string, use_inpainting_refiner, inpainting_refiner_model_id = inpainter_dict[inpainter_choice]
            
                # PRINT MODEL SUMMARY
                print("\nGENERATOR = ",generative_model_id)
                if use_refiner:
                    print("REFINER = ",refiner_model_id)
                if num_inpainting_iterations > 0:
                    print("EVALUATOR = ", evaluator)
                    print("INPAINTER = ",inpainting_model_id)
                    if use_inpainting_refiner:
                        print("INPAINTING REFINER = ",inpainting_refiner_model_id)
            
                # LOAD INPAINTER
                if num_inpainting_iterations > 0:
                    premodel=time.time()
                    del inpainter
                    gc.collect()
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
                            revision="fp16" if ("runwayml" in inpainting_model_id) else None,
                            vae=vae,
                            )
                    #inpainter.to("cuda")
                    inpainter.enable_model_cpu_offload()
                    #inpainter.unet = torch.compile(inpainter.unet, mode="reduce-overhead", fullgraph=True)  #  Needs torch>=2.0 #NG 
                    #
                    # LOAD INPAINTER REFINER
                    if use_inpainting_refiner:
                        del inpainting_refiner
                        gc.collect()
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


                # WHEN WE WANT EACH INPAINTER WORKING ON THE SAME INPUT
                # PICK UP THE GENERATED IMAGE FROM THE SAVED FILE
                image_out = Image.open(filename_pregenerated)
                image_out_tensor = transforms.ToTensor()(image_out)
    
                
                # LOOP THROUGH INPAINTING ITERATIONS    
                for inpainting_iteration in range(num_inpainting_iterations):
                    denoising_strength = 0.75 - (0.25* inpainting_iteration)     # [0.75,  0.5, 0.25, 0.0 ...]
                    denoising_strength = min(1.0, max((1/num_inference_steps), denoising_strength)) # Keep in valid range (0, 1]
                    print("Inpainting iteration ",inpainting_iteration+1," of ",num_inpainting_iterations)
                    #######################################
                    #
                    # CREATE INPAINTING MASK
                    # USING EVALUATOR(S)
                    #
                    #######################################
                    gc.collect()
                    pre_evaluation = time.time()
                    _, xdim_in, ydim_in = image_out_tensor.shape
                    print("Image=",xdim_in,"x",ydim_in)
                    chunk_size = int(xdim_in/args.num_chunks)
                    print("chunk_size = ", chunk_size,"x",chunk_size)
                    crop_size = args.chunks_per_crop * chunk_size
                    print("crop_size = ", crop_size,"x",crop_size)
                    # The mask can be computed or random or solid
                    if args.random_mask:
                        boolean_mask = np.random.randint(0, 2, (args.num_chunks, args.num_chunks), dtype=np.uint8)
                    elif args.solid_mask:
                        boolean_mask = np.ones((args.num_chunks, args.num_chunks), dtype=np.uint8)
                    else:
                        computed_mask = utils.compute_stride_mask(image_out_tensor, chunk_size, crop_size)
                        mean, std = np.mean(computed_mask), np.std(computed_mask)
                        print("Computed mask: mean=", mean," std =", std)
                        if threshold == None: # SET THRESHOLD ON THE FIRST TIME THROUGH
                            threshold = mean 
                            #boolean_mask = np.ones((args.num_chunks, args.num_chunks))                       # THESE TWO LINES ENSURE THAT THE MASK DOESN'T GROW
                        #boolean_mask = (computed_mask < threshold).astype(int) * boolean_mask  # THESE TWO LINES ENSURE THAT THE MASK DOESN'T GROW
                        boolean_mask = (computed_mask < threshold).astype(int)
                    print("Mask evaluation time = ", int(time.time()-pre_evaluation)," seconds")
                    print("mask=", boolean_mask.shape)
                    print("Boolean mask =\n", boolean_mask)
                    # Enlarge and convert our mask
                    mask_array = utils.enlarge_mask(boolean_mask, height)
                    mask_image = Image.fromarray(mask_array)
                    mask_out_progression.append(mask_image)
                    
                    ########################################
                    #
                    # INPAINTING
                    #
                    ######################################
                    
                    #################### 
                    # RUN THE INPAINTER
                    #################### 
                    gc.collect()
                    pregeneration_time = time.time()
                    options = dict(
                        prompt=prompt_total,
                        negative_prompt=negative_prompt, 
                        num_inference_steps=num_inference_steps,
                        num_images_per_prompt=num_images_per_prompt,
                        guidance_scale = cfg, #default = 8.0 for SDXL
                        generator=randomgenerator,
                        #height = height, # default = 512
                        #width = width, # default = 512
                        image=image_out, # for inpainting
                        mask_image=mask_image, # for inpainting
                        strength= denoising_strength, # 0-1, 0=no change, default=1.0  for inpainting
                        output_type="latent" if use_inpainting_refiner else "pil",
                        )
                    if use_inpainting_refiner:
                        options['denoising_end'] = high_noise_frac_inpaint # for inpainting with refiner
                    del images_inpaint
                    gc.collect()
                    images_inpaint = inpainter(**options).images
                    print("Inpainting time = ",int(time.time()-pregeneration_time)," seconds")
    
                    ############################ 
                    # RUN THE INPAINTING REFINER
                    ############################ 
                    if use_inpainting_refiner:
                        prerefiner_time = time.time()
                        images = []
                        for i in range(len(images_inpaint)):
                            image_inpaint = images_inpaint[i]
                            image_inpaint.to("cuda")
                            options = dict(
                                prompt=prompt_total,
                                negative_prompt=negative_prompt, 
                                num_inference_steps=num_inference_steps,
                                num_images_per_prompt=1,
                                guidance_scale = cfg, #default = 8.0 for SDXL
                                generator=randomgenerator,
                                #height = height, # default = 512
                                #width = width, # default = 512
                                image=image_inpaint[None, :], # for inpainting
                                mask_image=mask_out_progression[-1], # for inpainting
                                strength= denoising_strength, # 0-1, 0=no change, default=1.0  for inpainting
                                denoising_start=high_noise_frac_inpaint, # for inpainting refiner
                                )
                            del image_inpainting_refiner
                            gc.collect()
                            image_inpainting_refiner = inpainting_refiner(**options).images[0]

                            image_inpainting_refiner = image_inpainting_refiner.resize((width, height))
                            images.append(image_inpainting_refiner)
                        print("Inpaint Refiner time = ", int(time.time()-prerefiner_time)," seconds")
                    else:
                        images = [image.resize((width, height)) for image in images_inpaint]
    
                    #  Create filename
                    filename_suffix = "Inpaint_" + str(filename_prompt) + "superprompt_"\
                            + generative_model_string\
                            + "_GENERATION-NUMBER_RANDOM-NUMBER_inpainter_"\
                            + inpainting_model_string\
                            + "_iteration"+ str(inpainting_iteration+1)\
                            + "_seed"+str(args.random_seed) + "_"+ str(num_inference_steps) + "steps_cfg"+str(cfg) 
                    if not extend_prompt:
                        filename_suffix = filename_suffix.replace("superprompt_", "_")
                    if args.random_mask:
                        filename_suffix = filename_suffix.replace("_inpainter_", "_randommask_inpainter_")
                    elif args.solid_mask:
                        filename_suffix = filename_suffix.replace("_inpainter_", "_solidmask_inpainter_")

                    # PICK THE BEST INPAINTED IMAGE
                    image_out = utils.pick_best_image(images, evaluator, filename_suffix)
                    image_out_progression.append(image_out)
                    image_out_tensor = transforms.ToTensor()(image_out)
                    print("Total time = ", int(time.time()-start_time)," seconds\n")
                    # EACH ITERATION OF INPAINTING IS PASSED TO THE NEXT AS image_out_tensor
    
                
                ###########################################################
                #
                # Recap the progression after all the inpainting is done.
                #
                ###########################################################
                if len(image_out_progression) > 1:
                    output_dir = "images_out_progression_numchunks" +str(args.num_chunks) + "_chunkspercrop"\
                            + str(args.chunks_per_crop) + "_" + filename_prompt + "_" + str(int(time.time()))
                    if args.random_mask:
                        output_dir = output_dir + "_randommask"
                    elif args.solid_mask:
                        output_dir = output_dir + "_solidmask"
                    os.mkdir(output_dir)
                    # SAVE THE PROGRESSION TO THE FINAL IMAGE
                    for i,image in enumerate(image_out_progression):
                        image.save(output_dir + "/Output_"+ str(filename_prompt)+ "_" + generative_model_string\
                                + "_inpainter_" + inpainting_model_string\
                                + "_numchunks" +str(args.num_chunks) + "_chunkspercrop" + str(args.chunks_per_crop)\
                                + "_image_" + str(i) + "_strength"+ str(denoising_strength) + "_" + str(time.time()) + ".png")
                    for i,mask in enumerate(mask_out_progression):
                        mask.save(output_dir + "/Output_"+ str(filename_prompt)+ "_" + generative_model_string\
                                + "_inpainter_" + inpainting_model_string\
                                + "_numchunks" +str(args.num_chunks) + "_chunkspercrop" + str(args.chunks_per_crop)\
                                + "_mask_" + str(i) + "_strength"+ str(denoising_strength) + "_" + str(time.time()) + ".png")
                    # SAVE THE FINAL IMAGE
                    output_filename = str(filename_prompt)+ "_" + generative_model_string + "_"\
                            + str(random.random())[2:14] + "_inpainter_" + inpainting_model_string\
                            + "_numchunks" +str(args.num_chunks) + "_chunkspercrop" + str(args.chunks_per_crop)\
                            + "_image_" + str(len(image_out_progression)-1) + "_strength"+ str(denoising_strength) + "_" + str(time.time()) + ".png"
                    if args.random_mask:
                        output_filename = output_filename.replace("_inpainter_", "_randommask_inpainter_")
                    elif args.solid_mask:
                        output_filename = output_filename.replace("_inpainter_", "_solidmask_inpainter_")
                    if args.use_pregenerated is not None:
                        output_dir = "images_out_ThesisPaper_" + args.use_pregenerated.split("/")[0].split("_")[-1]
                        output_dir = output_dir.replace(".png","").replace(".jpg","").replace(".jpeg","").replace(".tif","")
                    else:   
                        output_dir = "images_out_ThesisPaper"
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    image_out_progression[-1].save(output_dir + "/" + output_filename)
