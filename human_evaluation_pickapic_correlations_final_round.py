import argparse
parser = argparse.ArgumentParser()
parser.add_argument('image_dir') 
parser.add_argument("--image_ref", default = "SDXL_astronaut_jungle_refined_photo.png")
args = parser.parse_args()
assert("renumbered" in args.image_dir or (("BatchA" not in args.image_dir) and ("BatchB" not in args.image_dir)))

import time
start_time = time.time()
print("Loading libraries")
import utils
import pickle
import os
import numpy as np
from PIL import Image
#from inception_score_imagenet.inception_score_model import get_inception_score
print("Loading pickapic")
import pickapic
print("Loading torch")
from torch import FloatTensor as torchFloatTensor
from torch import nn as torchnn
print("Loading torchvision")
from torchvision import transforms
print("Libraries Loading time = ",int(time.time()-start_time)," seconds")

image_dir = "/homes/rdb121/Thesis/" + args.image_dir
print("Evaluating images in ",image_dir)
list_of_imagenames = sorted(os.listdir(image_dir))
list_of_images = [Image.open(image_dir + "/" + filename) for filename in list_of_imagenames]
# LOAD REF IMAGE FOR PICKAPIC
image_reference = Image.open("images_varied/SDXL_astronaut_jungle_refined_photo.png") # Very High quality image
#image_reference = Image.open("images_varied/SDXL_lion_refined.png") # High quality image
#image_reference = Image.open("images_varied/SDXL_brunette_holding_book_refined3.png") # High quality image
#image_reference = Image.open("images_varied/BrunetteHoldingaBook_stable-diffusion-v1-5_seed119_50steps_5_cfg7.5_1690387105.845417.png") # Moderate quality image
#image_reference = Image.open("images_varied/PhotoOfABrunetteHoldingABookAvoidMangledFingers04.jpg") # Low quality image
#image_reference = Image.open("images_varied/queens_tower.jpg") # Real image
#image_reference = Image.open("images_varied/Cantebury_DadJudy.jpg") # Real image
#image_reference = Image.open("images_varied/GuysAndDolls_RandyAndFayeAtTheBridgeTheatre.jpg") # Real image
#image_reference = Image.open("images_varied/MSc_bowling_28June.jpg") # Real image
#image_reference = Image.open("images_varied/www.Panayispicture.com-332.jpg") # Real image
image_reference = Image.open("images_varied/" + args.image_ref) # Very High quality image


for imagename in list_of_imagenames:
    image = Image.open(image_dir + "/" + imagename)
    evaluation = pickapic.calc_probs("(((hyperrealistic, photographic realism)))", [image, image_reference])[0]
    print(imagename, evaluation)




################################
# PROCESS HUMAN EVALUATIONS
################################
#eval_batchA_old = [2,0,10,22,2,4,15,11,0,0,23,6,18,6,0,6,3,1,16,13,0,0,20,4,0,24,1,1,15,0,7,1,9,14,2,0,11,6,3,1,16,5,4,4,11,10,3,0,0,3,4,18,0,0,18,4,0,17,0,6,9,0,8,6,3,1,9,12,2,0,0,18]
eval_batchA_new = [2,0,10,27,3,5,17,12,0,1,27,6,21,6,0,8,4,1,17,16,1,0,21,5,0,27,2,1,17,0,9,1,10,15,3,0,12,8,4,1,18,6,5,4,11,11,4,0,0,3,5,21,0,0,21,4,0,21,0,6,10,0,9,8,4,1,10,14,3,0,0,21]
eval_batchB = [8,1,1,9,5,1,14,0,0,15,0,5,0,4,16,0,8,4,9,0,12,1,0,0,4,11,0,0,5,1,9,2,5,8,1,2,2,0,8,3,1,2,7,2,3,2,2,6,5,6,0,0,0,8,4,0,11,2,1,0,5,2,3,0,0,4,0,7,0,10,0,0,9,0,1,0,0,0,6,1,6,2,1,0,3,1,5,1,4,5,0,0,0,5,2,4]
#eval_batchD_old = [1,0,3,28, 19,5,10,0, 0,2,23,3, 1,9,16,1, 0,19,8,0, 7,18,2,1, 17,6,0,5, 0,9,3,17, 0,12,9,9, 4,4,15,5, 6,2,8,10, 5,16,1,1, 0,24,0,0, 6,16,1,0, 0,22,0,0, 11,13,1,0, 1,15,1,1, 1,6,0,13, 9,11,4,3, 6,2,12,0, 1,16,0,2, 1,1,18,0, 0,5,3,12, 1,15,1,1]
eval_batchD = [1,0,3,29, 19,6,10,0, 0,2,24,3, 1,9,17,1, 0,20,8,0, 8,18,2,1, 18,6,0,5, 0,9,3,18, 0,12,9,10, 4,4,15,6, 6,2,9,10, 6,16,1,1, 0,25,0,0, 6,17,1,0, 0,23,0,0, 11,14,1,0, 2,16,7,1, 2,6,0,13, 10,11,4,3, 7,2,12,0, 1,17,0,2, 1,1,19,0, 0,6,3,12, 1,16,1,1]
batch_dict = {"images_out_BatchA_renumbered": eval_batchA_new, "images_out_BatchB_renumbered": eval_batchB, "images_out_BatchD_seed02005556": eval_batchD}
eval_human = batch_dict[args.image_dir]

quad_sums = []
for index in range(0, len(eval_human), 4):
    quad = eval_human[index : index+4]
    quad_sum = np.sum(quad)
    for i in range(4):
        quad_sums.append(quad_sum)
# Calculate distribution percentages for each quad
eval_human_percent = eval_human / np.array(quad_sums)
# Calculate total votes for each generator
votes_v1, votes_v2, votes_sdxl_base, votes_sdxl_refiner = 0,0,0,0
votes_original, votes_solid, votes_random, votes_calculated = 0,0,0,0
votes_total = 0
if "BatchA" in args.image_dir:
    for filename, votes in zip(list_of_imagenames, eval_human):
        if "v1-5" in filename:
            votes_v1 += votes
        if "v2-1" in filename:
            votes_v2 += votes
        if "xl-base" in filename:
            votes_sdxl_base += votes
        if "xl-refiner" in filename:
            votes_sdxl_refiner += votes
elif "BatchB" in args.image_dir:
    for filename, votes in zip(list_of_imagenames, eval_human):
        if "inpaint_stable-diffusion-1" in filename:
            votes_v1 += votes
        if "inpaint_stable-diffusion-2" in filename:
            votes_v2 += votes
        if "inpaint_stable-diffusion-xl-base" in filename:
            votes_sdxl_base += votes
        if "inpaint_stable-diffusion-xl-refiner" in filename:
            votes_sdxl_refiner += votes
elif "BatchD" in args.image_dir:
    for filename, votes in zip(list_of_imagenames, eval_human):
        votes_total += votes
        if "solid" in filename:
            votes_solid += votes
        elif "random" in filename:
            votes_random += votes
        elif "inpaint" in filename:
            votes_calculated += votes
        else: # original image
            votes_original += votes



if "BatchD" in args.image_dir:
    print("original_image=",votes_original," solid_mask =",votes_solid," random_mask=", votes_random, " calculated_pipeline ", votes_calculated)
    print("original_image=",votes_original/votes_total," solid_mask =",votes_solid/votes_total," random_mask=", votes_random/votes_total, " calculated_pipeline ", votes_calculated/votes_total)

else:
    print("v1=",votes_v1," v2 =",votes_v2," sdxl_base=", votes_sdxl_base, " sdxl_refiner= ", votes_sdxl_refiner)
    # BATCHA RESULTS: v1-5= 150  v2-1 = 145 sdxl_base= 113  sdxl_refiner=  132
    votes_total = votes_v1 + votes_v2 + votes_sdxl_base + votes_sdxl_refiner
    print("v1=",votes_v1/votes_total," v2 =",votes_v2/votes_total," sdxl_base=", votes_sdxl_base/votes_total, " sdxl_refiner= ", votes_sdxl_refiner/votes_total)


"""
################################
# INCEPTION SCORES
################################
list_of_images_np = [np.asarray(image) for image in list_of_images]
inception_score_mean, inception_score_std = get_inception_score(list_of_images_np)
#print("Batch A inception_score = ",inception_score_mean)
#
# CALCULATE INCEPTION SCORES FOR EACH IMAGE 
inverted_eval_inception_score = []
for image_number in range(len(list_of_images_np)):
    popped_list_of_images_np = list_of_images_np * 1
    popped_list_of_images_np.pop(image_number)
    mean, std = get_inception_score(popped_list_of_images_np)
    #print("Image", image_number," inception score = ", mean)
    inverted_eval_inception_score.append(mean)
inverted_eval_inception_score = np.array(inverted_eval_inception_score)
#print("Inverted Inception scores = ", inverted_eval_inception_score)
#BATCHA RESULTS: inverted_eval_inception_score = np.array([3.8750293, 3.8783443, 3.8643754, 3.8647294, 3.838176, 3.868359, 3.810842, 3.819488, 3.8177986, 3.8230846, 3.8871098, 3.8630512, 3.8687088, 3.871879, 3.8207734, 3.8345852, 3.7834516, 3.794244, 3.7910476, 3.7733154, 3.8163204, 3.814888, 3.8499699, 3.8218408, 3.8591244, 3.883424, 3.9180634, 3.896566, 3.8612804, 3.8464122, 3.841914, 3.8714848, 3.8474574, 3.8475583, 3.8053563, 3.8532052, 3.8331623, 3.8412662, 3.8788152, 3.8710847, 3.8618836, 3.8731697, 3.8557854, 3.8438842, 3.8708835, 3.873597, 3.8702686, 3.87606, 3.8765385, 3.8795998, 3.9426312, 3.9218717, 3.8722293, 3.875833, 3.9059021, 3.9163678, 3.8762689, 3.901252, 3.8959491, 3.9014504, 3.9171474, 3.8987107, 3.8969429, 3.9019933, 3.94852, 3.963456, 3.9605916, 3.9737294, 3.9749348, 3.9693253, 3.9190354, 3.934035])
inverted_inception_score_correlation = np.corrcoef(inverted_eval_inception_score, eval_human_percent)[0, 1]
#
# UN-INVERT THE INCEPTION SCORES USING BOTH ADDITIVE AND MULTIPLICATIVE INVERSIONS, THEN CALCULATE CORRELATION COEF
inception_score_correlation = np.corrcoef(-1*inverted_eval_inception_score, eval_human_percent)[0, 1]
print("Correlation between inception score and human evaluation is ", inception_score_correlation)
#BATCHA RESULTS: Correlation between inception score and human evaluation is  -0.08467408578708835
inception_score_correlation = np.corrcoef(1/inverted_eval_inception_score, eval_human_percent)[0, 1]
print("Correlation between inception score and human evaluation is ", inception_score_correlation)
#BATCHA RESULTS: Correlation between inception score and human evaluation is  -0.08569879098079057
"""


################################
# Evaluate each image in the list using pickapic
################################
eval_pickapic = [pickapic.calc_probs("(((hyperrealistic, photographic realism)))", [image, image_reference])[0]\
        for image in list_of_images]
eval_pickapic = np.array(eval_pickapic)
print("eval_pickapic = ",eval_pickapic)
print("mean = ",np.mean(eval_pickapic), "  std = ", np.std(eval_pickapic))
pickapic_correlation = np.corrcoef(eval_pickapic, eval_human_percent)[0, 1]
print("Reference picture = ", args.image_ref)
print("\nCorrelation between pickapic and human evaluation is ", pickapic_correlation,"\n")
# BATCHA RESULTS: Correlation between pickapic and human evaluation is 0.095  (very high quality image, astro refined)  
# BATCHA RESULTS: Correlation between pickapic and human evaluation is   (high quality image, book refined)        
# BATCHA RESULTS: Correlation between pickapic and human evaluation is   (medium-high quality image, lion refined) 
# BATCHA RESULTS: Correlation between pickapic and human evaluation is   (moderate quality image, book sd1.5)      
# BATCHA RESULTS: Correlation between pickapic and human evaluation is 0.08324672660521694  (low quality image, Brunette mangled fingers)                       
# BATCHA RESULTS: Correlation between pickapic and human evaluation is 0.0707  (real image, Queen's tower)               
# BATCHA RESULTS: Correlation between pickapic and human evaluation is   (real image, DadJudyCantebury) 
# BATCHA RESULTS: Correlation between pickapic and human evaluation is   (real image, FayeRandy_GuysAndDolls) 
# BATCHA RESULTS: Correlation between pickapic and human evaluation is 0.0696  (real image, Msc_Bowling) 
# BATCHA RESULTS: Correlation between pickapic and human evaluation is 0.0702  (real image, FayeRandy_Wedding) 

# BATCHB RESULTS: Correlation between pickapic and human evaluation is   (very high quality image, astro refined)
#
# Check how many predictions match human evaluations
group_matches = utils.get_group_matches(eval_pickapic, eval_human_percent, 4)
print(group_matches," correct predictions out of ",len(eval_human_percent)//4)


"""
################################
# Evaluate each image in the list using the discriminator
################################
# DISCRIMINATORS
pre_load_time = time.time()
stylegan_models = ['ffhq.pkl', 'cifar10.pkl', 'afhqwild.pkl'] # These models want Tensorflow
stylegan3_models = ['stylegan3-r-afhqv2-512x512.pkl', 'stylegan3-t-ffhqu-256x256.pkl', 'stylegan3-r-ffhqu-256x256.pkl']
stylegan3_models = ['stylegan2-ffhq-512x512.pkl','stylegan2-cifar10-32x32.pkl' ]  # The cifar10-32x32 model generates an error
#discriminator_model = 'stylegan2-ffhq-512x512.pkl' # This doesn't need scipy
discriminator_model = 'stylegan3-r-afhqv2-512x512.pkl' # This needs scipy
discriminator_size = int(discriminator_model.split("x")[-1].split(".")[0])
print("Discriminator = ",discriminator_model," size = ",discriminator_size)
#
# LOAD STYLEGAN3 DISCRIMINATOR
with open('/vol/bitbucket/rdb121/models/'+discriminator_model,'rb') as f:
    D = pickle.load(f)['D']#.cuda()
c = None # class labels
print("Time to load discriminators = ", int(time.time()-pre_load_time), " seconds")
#
def resize_image_tensor(image_in_tensor: torchFloatTensor, out_size: int) -> torchFloatTensor: 
    # Resize the input tensor to the size specified by out_size
    if image_in_tensor.shape[0] == out_size: # assume square images
        return image_in_tensor
    image_unresized = transforms.ToPILImage()(image_in_tensor)
    image_resized = image_unresized.resize((out_size, out_size))
    tensor_resized = transforms.ToTensor()(image_resized)
    return tensor_resized
#
i = 0
eval_stylegan3 = []
for image in list_of_images:
    image_tensor = transforms.ToTensor()(image)
    tensor_resized = resize_image_tensor(image_tensor, discriminator_size)
    tensor_resized.unsqueeze_(0)
    logits = D(tensor_resized, c)
    lossG = torchnn.functional.softplus(-logits)
    lossD = torchnn.functional.softplus(logits)
    eval_stylegan3.append(lossD.item())
    print(".", end="")
    #print("Full image ",i," logit = ",logits, "  lossG=",lossG,"  lossD=",lossD.item())
    i+=1
eval_stylegan3 = np.array(eval_stylegan3)
stylegan3_correlation = np.corrcoef(eval_stylegan3, eval_human_percent)[0, 1]
print("Discriminator model = ",discriminator_model)
print("\n\nCorrelation between Stylegan3 and human evaluation is ", stylegan3_correlation)
# BATCHA RESULTS: Correlation between Stylegan3(re-evaluated) and human evaluation is  -0.19
"""
