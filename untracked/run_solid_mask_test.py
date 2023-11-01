#
# Run the solid_mask on the pregenerated image
#
# But put a blank square in the upper-left
#
import argparse
import os
import shutil
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument('use_pregenerated')
parser.add_argument('--random_seed', default=0, type = int)
args = parser.parse_args()
# Display the parameters and check for possible errors before loading libraries.
print("\nPROCESSING PREGENERATED IMAGE ",args.use_pregenerated)
print("random_seed = ", args.random_seed)

print("No Blank spot ")
os.system("python solid_mask_test.py --use_pregenerated " + str(args.use_pregenerated) + " --solid_mask --num_inpainting_iterations 1 --num_pixels_blank 0 --images_per_prompt 4")

for i in range(9):
    num_pixels_blank = int(2**i)
    print("Blank spot ",num_pixels_blank," by ", num_pixels_blank)
    os.system("python solid_mask_test.py --use_pregenerated " + str(args.use_pregenerated) + " --solid_mask --num_inpainting_iterations 1 --num_pixels_blank " + str(num_pixels_blank) + " --images_per_prompt 4")
