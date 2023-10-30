##########################
# UTILITY FUNCTIONS
##########################
import numpy as np
from torchvision import transforms
import torch
#from torch import FloatTensor as torchFloatTensor
#from torch import nn as torchnn
#
# For compute_stride_mask()
from torchvision import transforms
from PIL import Image
import pickapic
# For random tagging of filenames when pick_best_image writes out files
#import random
"""  WE DON'T NEED TO LOAD THE DISCIMINATOR.  WE ARE NOW USING PICKAPIC.
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
print("Time to load discriminator = ", int(time.time()-pre_load_time), " seconds")
"""

#########################################

# RESIZE THE INPUT TENSOR TO THE SIZE SPECIFIED BY out_size
#
def resize_image_tensor(image_in_tensor: torch.FloatTensor, out_size: int) -> torch.FloatTensor: 
    if image_in_tensor.shape[0] == out_size: # assume square images
        return image_in_tensor
    image_unresized = transforms.ToPILImage()(image_in_tensor)
    image_resized = image_unresized.resize((out_size, out_size))
    tensor_resized = transforms.ToTensor()(image_resized)
    return tensor_resized


# CHECK THE TWO SEQUENCES, AND RETURN NUMBER OF GROUP-MATCHES
#
# Assert that the sequences are the same length.
# Assert that the lengths are divisible by "group_length."
# Divide each sequence into groups of length "group_length,"
# and find the index of the largest entry in each gropu.
# If the corresponding groups from the two sequences have matching indices,
# that is a group-match.
# Return the total number of group-matches.
#
def get_group_matches(sequence_a: list[float], sequence_b: list[float], group_length: int = 4) -> int:
    assert len(sequence_a) == len(sequence_b)
    assert (len(sequence_a) % group_length) == 0
    # Just in case sequences are np.array
    sequence_a, sequence_b = list(sequence_a), list(sequence_b)
    #
    group_matches = 0
    for index in range(0, len(sequence_a), group_length):
        group_a = sequence_a[index : index+group_length]
        group_b = sequence_b[index : index+group_length]
        index_a = group_a.index(np.max(group_a)) 
        index_b = group_b.index(np.max(group_b)) 
        print(group_a, index_a, group_b, index_b)
        if index_a == index_b:
            group_matches +=1
            print("group_matches = ",group_matches)
    return group_matches


# ENLARGE THE INPUT MASK TO THE SIZE SPECIFIED BY SIZE_OUT, MULTIPLY MASK BY 255
# Input mask range [0,1] -> output mask range [0,255]
# Used to make the mask size match the image size before sending to the inpainter
def enlarge_mask(mask_in: np.ndarray, size_out:int = 512) -> np.ndarray:
    xdim, ydim = mask_in.shape
    mask_array = np.zeros((size_out, size_out), dtype=np.uint8)
    block = int(size_out/xdim)
    for x in range(xdim):
        for y in range(ydim):
            mask_array[x*block:(x+1)*block, :][:, y*block:(y+1)*block] = np.ones((block, block)) * 255 * mask_in[x][y]
    return(mask_array)



# EVALUATE ALL THE IMAGES IN THE INPUT LIST AND RETURN THE BEST ONE
# The evaluator specifies either "stylegan3" or "pickapic" evaluations
# Optionally, we can also write out all the images to disk using the specified filename_suffix
def pick_best_image(images: list[Image.Image], evaluator, filename_suffix: str = "") -> tuple[Image.Image, torch.FloatTensor]:
    # EVALUATE THE MULTIPLE IMAGES AND CHOOSE THE BEST ONE
    evaluation_best = -np.inf
    # LOAD REF IMAGE FOR PICKAPIC
    image_pickscore_reference = Image.open("/homes/rdb121/Thesis/images_varied/SDXL_astronaut_jungle_refined_photo.png")
    for i in range(len(images)):
        if evaluator == "stylegan3":
            image_tensor = transforms.ToTensor()(images[i])
            tensor_resized = utils.resize_image_tensor(image_tensor, discriminator_size)
            tensor_resized.unsqueeze_(0)
            """  DISABLE UNTIL WE ARE ABLE TO PUT THE DISCRIMINATOR ON THE GPU # TRY TO PUT THE DISCRIMINATOR ON THE GPU
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
            print("Full image ",i," logit = ",logits, "  lossG=",lossG,"  lossD=",lossD, "pickapic prob=",pickapic_prob[0])
            evaluation = lossD
        elif evaluator == "pickapic":
            pickapic_prob = pickapic.calc_probs("(((hyperrealistic, photographic realism)))", [images[i], image_pickscore_reference])
            evaluation = pickapic_prob[0]
        if evaluation > evaluation_best:
            evaluation_best, index_best = evaluation, i

        """     DISABLE SAVING EACH CONSIDERED IMAGE
        # Save each image
        #filename_out = filename_suffix + "_P" + str(pickapic_prob[0]) + "_D" + str(lossD.numpy()[0][0]) + ".png" # TEMPORARY IGNORE DISCRIMINATOR
        filename_out = filename_suffix + "_P" + str(pickapic_prob[0]) + ".png" # TEMPORARY IGNORE DISCRIMINATOR
        filename_out = filename_out.replace("GENERATION-NUMBER", str(i))
        filename_out = filename_out.replace("RANDOM-NUMBER", str(random.random()))
        images[i].save(filename_out)
        """
    image_best_tensor = transforms.ToTensor()(images[index_best])
    return images[index_best]


# EVALUATE image_full_tensor AND RETURN AN INPAINTING MASK
# Divide the input tensor into square chunks of size chunk_size x chunk_size
# Evaluate larger crops of size crop_size x crop_size (containing multiple chunks)
# Each chunk receives the evaluation of each crop that the chunk is in
# At the end all of the evaluations of each chunk are averaged.
def compute_stride_mask(image_full_tensor: torch.FloatTensor, chunk_size: int, crop_size:int =None) -> np.ndarray:
    _, xdim_in, ydim_in = image_full_tensor.shape
    print("Full Image=",xdim_in,"x",ydim_in)
    xdim_out, ydim_out = xdim_in//chunk_size, ydim_in//chunk_size
    evaluations_total = np.zeros((xdim_out, ydim_out))
    evaluations_count = np.zeros((xdim_out, ydim_out), dtype = int)
    print("mask=",evaluations_total.shape)
    #
    image_pickscore_reference = Image.open("images_varied/SDXL_astronaut_jungle_refined_photo.png")
    #
    if crop_size == None:
        crop_size = chunk_size
    crop_size_x, crop_size_y = crop_size, crop_size
    ones_array = np.ones((crop_size_x // chunk_size, crop_size_y // chunk_size), dtype = int)
    for x in range(0, xdim_in - crop_size_x + 1 , chunk_size):
        for y in range(0, ydim_in - crop_size_y + 1, chunk_size):
            image_crop_tensor = image_full_tensor[:, x: x+crop_size_x, y: y+crop_size_y]
            image_crop = transforms.ToPILImage()(image_crop_tensor)
            #image_crop.save("image_crop_"+str(x)+str(y)+".png")
            evaluation = pickapic.calc_probs("(((hyperrealistic, photographic realism)))", [image_crop, image_pickscore_reference])
            evaluations_count[x // chunk_size: (x+crop_size_x) // chunk_size, y // chunk_size: (y+crop_size_y) // chunk_size] += ones_array
            evaluations_total[x // chunk_size: (x+crop_size_x) // chunk_size, y // chunk_size: (y+crop_size_y) // chunk_size] += evaluation[0] * ones_array #only for display
            #evaluations_total[x // chunk_size: (x+crop_size_x) // chunk_size, y // chunk_size: (y+crop_size_y) // chunk_size] += int(evaluation[0] * 100) * ones_array #only for display
            #print("evaluations_count = \n", evaluations_count)
            #print("evaluations_total = \n", (evaluations_total).astype(int))
    computed_mask = evaluations_total / evaluations_count
    return computed_mask   


# USED WHEN WE USE A PRE-GENERATED IMAGE TO INPAINT
# RETURN THE NAME OF THE GENERATOR AND THE PROMPT FROM THE FILEPATH
def get_generator_prompt_from_filepath(pathname):
    filename = pathname.split("/")[-1]
    prompt = filename.split("_")[0].replace(":"," ")
    suffix_len = len(filename.split(".")[-1])
    filename_nosuffix = filename[: -1*(suffix_len+1)]
    generator = filename_nosuffix.split("_")[1].replace("gen-", "")
    generator = generator.replace("stable-diffusion", "sd")  # Accommodate two different naming conventions
    generator = generator.replace("v1-5", "v1.5").replace("v2-1","v2.1")  # Accommodate two different naming conventions
    if "xl" in generator:
        generator = generator.replace("stable-diffusion-xl","sdxl").replace("-1.0", "")
    print("\nPRE-GENERATED IMAGE OVERRIDES SPECIFIED GENERATOR(S) AND PROMPT(S)")
    print(pathname)
    print("Generator = ", generator)
    print("Prompt = ", prompt)
    return [generator], [prompt]
