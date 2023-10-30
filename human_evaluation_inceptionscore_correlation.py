import argparse
parser = argparse.ArgumentParser()
parser.add_argument('image_dir') 
args = parser.parse_args()
assert ("renumbered" in args.image_dir)

print("importing utils")
import utils
print("importing os")
import os
import numpy as np
from PIL import Image
print("importing get_inception_score")
from inception_score_imagenet.inception_score_model import get_inception_score

image_dir = "/homes/rdb121/Thesis/" + args.image_dir
print("Evaluating images in ",image_dir)
list_of_imagenames = sorted(os.listdir(image_dir))

"""
eval_stylegan3 = [float(imagename.replace("[[","]]"). split("]]")[-2]) for imagename in list_of_imagenames]
eval_stylegan3 = np.array(eval_stylegan3)
#print("stylegan3 = ",eval_stylegan3)
eval_pickapic = [float(imagename. split("_")[-2].replace("P","")) for imagename in list_of_imagenames]
eval_pickapic = np.array(eval_pickapic)
#print("pickapic = ",eval_pickapic)
correlation = np.corrcoef(eval_pickapic, eval_stylegan3)[0, 1]
print("Correlation between Stylegan3 and pickapic is ",correlation)
"""

# HUMAN EVALUATIONS
#eval_batchA_old = [2,0,10,22,2,4,15,11,0,0,23,6,18,6,0,6,3,1,16,13,0,0,20,4,0,24,1,1,15,0,7,1,9,14,2,0,11,6,3,1,16,5,4,4,11,10,3,0,0,3,4,18,0,0,18,4,0,17,0,6,9,0,8,6,3,1,9,12,2,0,0,18]
eval_batchA_new = [2,0,10,27,3,5,17,12,0,1,27,6,21,6,0,8,4,1,17,16,1,0,21,5,0,27,2,1,17,0,9,1,10,15,3,0,12,8,4,1,18,6,5,4,11,11,4,0,0,3,5,21,0,0,21,4,0,21,0,6,10,0,9,8,4,1,10,14,3,0,0,21]
eval_batchB = [8,1,1,9,5,1,14,0,0,15,0,5,0,4,16,0,8,4,9,0,12,1,0,0,4,11,0,0,5,1,9,2,5,8,1,2,2,0,8,3,1,2,7,2,3,2,2,6,5,6,0,0,0,8,4,0,11,2,1,0,5,2,3,0,0,4,0,7,0,10,0,0,9,0,1,0,0,0,6,1,6,2,1,0,3,1,5,1,4,5,0,0,0,5,2,4]
batch_dict = {"images_out_BatchA_renumbered": eval_batchA_new, "images_out_BatchB_renumbered": eval_batchB}
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
print("v1=",votes_v1," v2 =",votes_v2," sdxl_base=", votes_sdxl_base, " sdxl_refiner= ", votes_sdxl_refiner)
# BATCHA RESULTS: v1-5= 146  v2-1 = 141 sdxl_base= 58  sdxl_refiner=  195
# BATCHB RESULTS: v1= 60  v2 = 135 sdxl_base= 46  sdxl_refiner= 82 
votes_total = votes_v1 + votes_v2 + votes_sdxl_base + votes_sdxl_refiner
print("v1=",votes_v1/votes_total," v2 =",votes_v2/votes_total," sdxl_base=", votes_sdxl_base/votes_total, " sdxl_refiner= ", votes_sdxl_refiner/votes_total)


# CALCULATE INCEPTION SCORES FOR EACH IMAGE 
list_of_images = [Image.open(image_dir + "/" + filename) for filename in list_of_imagenames]
list_of_images_np = [np.asarray(image) for image in list_of_images]
inception_score_mean, inception_score_std = get_inception_score(list_of_images_np)
#print("Batch A inception_score = ",inception_score_mean)
#
eval_inception_score = []
for image_number in range(len(list_of_images_np)):
    popped_list_of_images_np = list_of_images_np * 1
    popped_list_of_images_np.pop(image_number)
    mean, std = get_inception_score(popped_list_of_images_np)
    #print("Image", image_number," inception score = ", mean)
    eval_inception_score.append(-1 * mean)
"""
for index in range(0, len(eval_inception_score), 4):
    quad = eval_inception_score[index : index+4]
    print(quad, quad.index(np.max(quad))+1)
"""
eval_inception_score = np.array(eval_inception_score)
#print("Inverted Inception scores = ", inverted_eval_inception_score)
inception_score_correlation = np.corrcoef(eval_inception_score, eval_human_percent)[0, 1]
print("Correlation between inception score and human evaluation is ", inception_score_correlation)
#
# Check how many predictions match human evaluations
group_matches = utils.get_group_matches(eval_inception_score, eval_human_percent, 4)
print(group_matches," correct predictions out of ",len(eval_human_percent)//4)
