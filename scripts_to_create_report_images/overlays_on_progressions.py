# Utility for making overlays for the Thesis paper
# Takes a user-given DIR
# Looks in DIR for subdirs with "progression" in the name
# Looks in the subdirs for files with "mask" in the filename, and corresponding files with "image" in the filename
# Overlays the mask on the image and saves it as an "overlay" file in the same directory.
#
print("Loading libraries")
import argparse
import os
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser()
#parser.add_argument('image_in_name') 
parser.add_argument('dir_in_name') 
#parser.add_argument('--num_chunks', default = 4, type=int) 
#parser.add_argument('--thickness', default = 4, type=int) 
args = parser.parse_args()


print("Checking ", args.dir_in_name)
assert os.path.isdir(args.dir_in_name), args.dir_in_name+" is not a valid directory."
list_of_progression_subdirs = os.listdir(args.dir_in_name)
list_of_progression_paths = [os.path.join(args.dir_in_name, subdir) for subdir in list_of_progression_subdirs if "progression" in subdir]
#print(list_of_progression_paths)

for path in list_of_progression_paths:
    list_of_filenames = os.listdir(path)
    list_of_mask_filenames = [filename for filename in list_of_filenames if "mask" in filename]
    #print(list_of_mask_files)
    for mask_filename in list_of_mask_filenames:
        mask_number = mask_filename.split("_")[-3]
        image_filename = [filename for filename in list_of_filenames if (("image_" + mask_number) in filename)][0]
        mask_in = np.array(Image.open(os.path.join(path, mask_filename)))
        image_in = np.array(Image.open(os.path.join(path, image_filename)))
        mask_red = np.zeros(image_in.shape, dtype = np.uint8)
        mask_red[:, :, 0] = mask_in
        overlay_image = np.maximum(image_in//2, mask_red)# HALF STRENGTH BG
        #overlay_image = np.maximum(image_in, mask_red)  # FULL STRENGTH BG
        overlay_filename = os.path.join(path, mask_filename.replace("mask", "overlay"))
        print("writing out ",overlay_filename)
        Image.fromarray(overlay_image).save(overlay_filename)
    





"""
assert os.path.isfile(args.image_in_name), args.image_in_name+" is not a valid file."
assert os.path.isdir(args.dir_in_name), args.dir_in_name+" is not a valid directory."
image_in = Image.open(args.image_in_name)
image_in = np.array(image_in)
print("image_in = ",image_in.shape)
print(type(image_in[0,0,0]))
list_of_masks = os.listdir(args.dir_in_name)
list_of_masks = [filename for filename in list_of_masks if "overlay" not in filename]
for mask_name in list_of_masks:
    mask_path = args.dir_in_name + "/" + mask_name
    print("processing ",mask_path)
    output_path = args.dir_in_name + "/overlay_" + mask_name
    mask_in = np.array(Image.open(mask_path))
    mask_red = image_blank
    mask_red[:, :, 0] = mask_in
    overlay_image = np.maximum(image_in//2, mask_red)
    print("writing out",output_path)
    Image.fromarray(overlay_image).save(output_path)
"""
