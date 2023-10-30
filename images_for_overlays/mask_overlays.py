import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('image_in_name') 
parser.add_argument('dir_in_name') 
#parser.add_argument('--num_chunks', default = 4, type=int) 
#parser.add_argument('--thickness', default = 4, type=int) 
args = parser.parse_args()

import numpy as np
from PIL import Image

array_solid = 255*np.ones((512,512, 3), dtype = np.uint8)
#Image.fromarray(array_solid).save("solid_white.png")

image_blank = np.zeros((512,512, 3), dtype = np.uint8)
#Image.fromarray(image_blank).save("blank.png")

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
image_np = np.zeros((512,512, 3), dtype = np.uint8)
image_size = image_np.shape[0] # Assume a square image
for i in range(args.num_chunks+1):
    chunk_size = image_size // args.num_chunks
    minimum = (i * chunk_size) - 2
    maximum = (i * chunk_size) + 1
    minimum = max(minimum,0)
    maximum = min(maximum,image_size-1)
    image_np[minimum:maximum, :] = 255
    image_np[:, minimum:maximum] = 255

image_in_name = "blank.png"
image_out = Image.fromarray(image_np)
image_out_name = "grid"+ str(args.num_chunks) + "_" + image_in_name
image_out.save(image_out_name)
"""

"""
# Make images for chunk evaluation figure
for vert in range(2):
    for horiz in range(args.num_chunks):
        image_out_np = image_np * 1
        image_out_np[vert*chunk_size: (vert+1)*chunk_size, horiz*chunk_size: (horiz+1)*chunk_size, 0] = 255
        image_out = Image.fromarray(image_out_np)
        image_out_name = "chunk"+ str(args.num_chunks) + "_eval_"+ str(vert) + "_" + str(horiz) + ".png"
        print("Saving ",image_out_name)
        image_out.save(image_out_name)
"""

"""
# Make images for stride evaluation figures, 4x4 chunks.
num_chunks = 4
stride_size = chunk_size
for crop_size in [chunk_size * 2]:
  for vert in range(3):
    horiz = 0
    while ((horiz*chunk_size) + crop_size) <= image_size:
        image_out_np = image_np * 1
        image_out_np[vert*chunk_size: (vert*chunk_size)+crop_size, horiz*chunk_size: (horiz*chunk_size) + crop_size, 0] = 255
        image_out = Image.fromarray(image_out_np)
        image_out_name = "stride"+ str(args.num_chunks) + "_crop" + str(crop_size//chunk_size) + "_eval_"+ str(vert) + "_" + str(horiz) + ".png"
        print("Saving ",image_out_name)
        image_out.save(image_out_name)
        horiz += 1 
"""

"""
# Make images for stride evaluation figures, 8x8 chunks.
stride_size = chunk_size
#for crop_size in [chunk_size, chunk_size * 2, chunk_size *3]:
for crop_size in [chunk_size *4]:
    vert = 0
    horiz = 0
    while ((horiz*chunk_size) + crop_size) <= image_size:
        image_out_np = image_np * 1
        image_out_np[vert*chunk_size: (vert*chunk_size)+crop_size, horiz*chunk_size: (horiz*chunk_size) + crop_size, 0] = 255
        image_out = Image.fromarray(image_out_np)
        image_out_name = "chunk"+ str(args.num_chunks) + "_crop" + str(crop_size//chunk_size) + "_eval_"+ str(vert) + "_" + str(horiz) + ".png"
        print("Saving ",image_out_name)
        image_out.save(image_out_name)
        horiz += 1 
"""
