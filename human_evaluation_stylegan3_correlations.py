import argparse
parser = argparse.ArgumentParser()
parser.add_argument('image_dir') 
parser.add_argument('--model', choices = ['stylegan3-r-ffhq-1024x1024', 'stylegan3-r-ffhqu-256x256', 'stylegan3-t-ffhqu-256x256',\
        'stylegan3-r-afhqv2-512x512', 'stylegan2-ffhq-512x512', 'stylegan2-cifar10-32x32']) 
args = parser.parse_args()
assert("renumbered" in args.image_dir)

import time
start_time = time.time()
print("Loading libraries")
#import dnnlib.tflib.tfutil
#import tensorflow as tf
import utils
import pickle
import os
import numpy as np
from PIL import Image
#from inception_score_imagenet.inception_score_model import get_inception_score
print("Loading pickapic")
#import pickapic
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


################################
# PROCESS HUMAN EVALUATIONS
################################
#eval_batchA_old = [2,0,10,22,2,4,15,11,0,0,23,6,18,6,0,6,3,1,16,13,0,0,20,4,0,24,1,1,15,0,7,1,9,14,2,0,11,6,3,1,16,5,4,4,11,10,3,0,0,3,4,18,0,0,18,4,0,17,0,6,9,0,8,6,3,1,9,12,2,0,0,18]
eval_batchA_new = [2,0,10,27,3,5,17,12,0,1,27,6,21,6,0,8,4,1,17,16,1,0,21,5,0,27,2,1,17,0,9,1,10,15,3,0,12,8,4,1,18,6,5,4,11,11,4,0,0,3,5,21,0,0,21,4,0,21,0,6,10,0,9,8,4,1,10,14,3,0,0,21]
eval_batchB = [8,1,1,9,5,1,14,0,0,15,0,5,0,4,16,0,8,4,9,0,12,1,0,0,4,11,0,0,5,1,9,2,5,8,1,2,2,0,8,3,1,2,7,2,3,2,2,6,5,6,0,0,0,8,4,0,11,2,1,0,5,2,3,0,0,4,0,7,0,10,0,0,9,0,1,0,0,0,6,1,6,2,1,0,3,1,5,1,4,5,0,0,0,5,2,4]
batch_dict = {"images_out_BatchA_renumbered": eval_batchA_new, "images_out_BatchB_renumbered": eval_batchB}
"""
if args.image_dir == "images_out_BatchA":
    eval_human = eval_batchA_new
"""
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
votes_v1_5, votes_v2_1, votes_sdxl_base, votes_sdxl_refiner = 0,0,0,0
for filename, votes in zip(list_of_imagenames, eval_human):
    if "v1-5" in filename:
        votes_v1_5 += votes
    if "v2-1" in filename:
        votes_v2_1 += votes
    if "xl-base" in filename:
        votes_sdxl_base += votes
    if "xl-refiner" in filename:
        votes_sdxl_refiner += votes
print("v1-5=",votes_v1_5," v2-1 =",votes_v2_1," sdxl_base=", votes_sdxl_base, " sdxl_refiner= ", votes_sdxl_refiner)
# BATCHA RESULTS: v1-5= 150  v2-1 = 145 sdxl_base= 113  sdxl_refiner=  132
votes_total = votes_v1_5 + votes_v2_1 + votes_sdxl_base + votes_sdxl_refiner
print("v1-5=",votes_v1_5/votes_total," v2-1 =",votes_v2_1/votes_total," sdxl_base=", votes_sdxl_base/votes_total, " sdxl_refiner= ", votes_sdxl_refiner/votes_total)



"""
################################
# Evaluate each image in the list using pickapic
################################
eval_pickapic = [pickapic.calc_probs("(((hyperrealistic, photographic realism)))", [image, image_reference])[0]\
        for image in list_of_images]
for index in range(0, len(eval_pickapic), 4):
    quad = eval_pickapic[index : index+4]
    print(quad, quad.index(np.max(quad))+1)
eval_pickapic = np.array(eval_pickapic)
print("eval_pickapic = ",eval_pickapic)
print("mean = ",np.mean(eval_pickapic), "  std = ", np.std(eval_pickapic))
pickapic_correlation = np.corrcoef(eval_pickapic, eval_human_percent)[0, 1]
print("Reference picture = ", image_reference)
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
"""


################################
# Evaluate each image in the list using the stylegan3 discriminator
################################
# DISCRIMINATORS
pre_load_time = time.time()
stylegan_models = ['ffhq.pkl', 'cifar10.pkl', 'afhqwild.pkl'] # These models want Tensorflow
stylegan3_models_ffhq = ['stylegan3-r-ffhq-1024x1024.pkl', 'stylegan3-r-ffhqu-256x256.pkl', 'stylegan3-t-ffhqu-256x256.pkl']
stylegan3_models_afhq = ['stylegan3-r-afhqv2-512x512.pkl'] # This needs scipy
#stylegan3_models = ['stylegan3-r-ffhq-1024x1024.pkl2', 'stylegan3-r-ffhqu-1024x1024.pkl', 'stylegan3-r-ffhqu-256x256.pkl'] # not downloaded
stylegan2_models = ['stylegan2-ffhq-512x512.pkl','stylegan2-cifar10-32x32.pkl' ]  # The cifar10-32x32 model generates an error
discriminator_model = args.model + '.pkl' 
discriminator_size = int(discriminator_model.split("x")[-1].split(".")[0])
print("Discriminator = ",discriminator_model," size = ",discriminator_size)
#
# LOAD STYLEGAN3 DISCRIMINATOR
#tfutil.init_tf()
with open('/vol/bitbucket/rdb121/models/'+discriminator_model,'rb') as f:
    D = pickle.load(f)['D']#.cuda()
c = None # class labels
print("Time to load discriminators = ", int(time.time()-pre_load_time), " seconds")
#
i = 0
eval_stylegan3 = []
for image in list_of_images:
    image_tensor = transforms.ToTensor()(image)
    tensor_resized = utils.resize_image_tensor(image_tensor, discriminator_size)
    tensor_resized.unsqueeze_(0)
    logits = D(tensor_resized, c)
    lossG = torchnn.functional.softplus(-logits)
    lossD = torchnn.functional.softplus(logits)
    eval_stylegan3.append(lossD.item())
    print(".", end="")
    #print("Full image ",i," logit = ",logits, "  lossG=",lossG,"  lossD=",lossD.item())
    i+=1
"""
# Print winner of each quad
for index in range(0, len(eval_stylegan3), 4):
    quad = eval_stylegan3[index : index+4]
    quad = eval_stylegan3[index : index+4]
    print(quad, quad.index(np.max(quad))+1)
"""
eval_stylegan3 = np.array(eval_stylegan3)
stylegan3_correlation = np.corrcoef(eval_stylegan3, eval_human_percent)[0, 1]
print("Discriminator model = ",discriminator_model)
print("\n\nCorrelation between Stylegan3 and human evaluation is ", stylegan3_correlation)
# BATCHA RESULTS: Correlation between Stylegan3(re-evaluated) and human evaluation is  -0.19
#
# Check how many predictions match human evaluations
group_matches = utils.get_group_matches(eval_stylegan3, eval_human_percent, 4)
print(group_matches," correct predictions out of ",len(eval_human_percent)//4)
