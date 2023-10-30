import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--image_in_name') 
parser.add_argument('--num_chunks', default = 4, type=int) 
parser.add_argument('--thickness', default = 4, type=int) 
args = parser.parse_args()

import numpy as np
from PIL import Image
import os



mask_filename = "Output_bedroom_gen-sd-v1.5_inpainter_sdxl-base_numchunks16_chunkspercrop1_mask_0_strength0.75_1693777718.1127644.png"
image_filename = "Output_bedroom_gen-sd-v1.5_inpainter_sdxl-base_numchunks16_chunkspercrop1_image_1_strength0.75_1693777718.0038412.png"
mask_in = np.array(Image.open(mask_filename))
image_in = np.array(Image.open(image_filename))
mask_red = np.zeros(image_in.shape, dtype = np.uint8)
mask_red[:, :, 0] = mask_in
overlay_image = np.maximum(image_in, mask_red)    # FULL STRENGTH BG
overlay_filename = image_filename.replace("image", "overlay_on_inpainting")
print("writing out ",overlay_filename)
Image.fromarray(overlay_image).save(overlay_filename)

y1 = int(7.25*32)
y2 = int(y1 + 4.5*32)
x1 = 0
x2 = int(x1 + 4.5*32)
inpainted_image_crop1 = image_in[x1:x2, y1:y2, :]
overlay_image_crop1 = overlay_image[x1:x2, y1:y2, :]
Image.fromarray(inpainted_image_crop1).save(image_filename.replace("image_1", "image_1_crop1"))
Image.fromarray(overlay_image_crop1).save(overlay_filename.replace("overlay_on_inpainting_1", "overlay_on_inpainting_1_crop1"))


y1 = int(7.5*32)
y2 = int(y1 + 5*32)
x1 = int(8.5*32)
x2 = int(x1 + 5*32)
inpainted_image_crop1 = image_in[x1:x2, y1:y2, :]
overlay_image_crop1 = overlay_image[x1:x2, y1:y2, :]
Image.fromarray(inpainted_image_crop1).save(image_filename.replace("image_1", "image_1_crop2"))
Image.fromarray(overlay_image_crop1).save(overlay_filename.replace("overlay_on_inpainting_1", "overlay_on_inpainting_1_crop2"))




"""
#############################################
# WOMAN ON HORSE FACIAL INPAINTING SMALL VS CLOSEUP
image_in_name = "images_out_GeneratorOutput_preBatchC_seed02005556/images_out_GeneratorOutput_preBatchC_medium:shot:woman:riding:a:horse_1693170047/medium:shot:woman:riding:a:horse_gen-sd-v2.1.png"
image_in = Image.open(image_in_name)
image_in.save("medium:shot:woman:riding:a:horse_gen-sd-v2.1.png")
image_in_np = np.array(image_in)
image_crop_np = image_in_np[128:128+64, 160:160+64, :]
image_crop = Image.fromarray(image_crop_np)
image_crop.save("medium:shot:woman:riding:a:horse_gen-sd-v2.1_crop.png")
image_resized = image_crop.resize((512, 512))
image_resized.save("medium:shot:woman:riding:a:horse_gen-sd-v2.1_resized.png")

image_inpainted = Image.open(args.image_in_name)
image_inpainted_shrunk = image_inpainted.resize((64,64))
image_inpainted_shrunk_np = np.array(image_inpainted_shrunk)
image_out_np = image_in_np*1
#image_out_np[128:128+64, 160:160+64, :] = image_inpainted_shrunk_np
n= 6
image_out_np[128+n:128+64-n, 160+n:160+64-n, :] = image_inpainted_shrunk_np[n:-n, n:-n, :]
image_out = Image.fromarray(image_out_np)
image_out.save("medium:shot:woman:riding:a:horse_gen-sd-v2.1_face_inpainted.png")
"""

"""
image_in = Image.open(args.image_in_name)
image_np = np.array(image_in)
"""

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

chunk_image_path = "images_varied/BrunetteHoldingaBook_stable-diffusion-v1-5_seed119_50steps_5_cfg7.5_1690387105.845417.png"
chunk_image_np = np.array(Image.open(chunk_image_path))
"""


"""
# Make images for chunk evaluation figure
for vert in range(args.num_chunks):
    for horiz in range(args.num_chunks):
        image_out_np = image_np * 1
        image_out_np[vert*chunk_size: (vert+1)*chunk_size, horiz*chunk_size: (horiz+1)*chunk_size, :] =\
            chunk_image_np[vert*chunk_size: (vert+1)*chunk_size, horiz*chunk_size: (horiz+1)*chunk_size, :] # Image cropped by mask
        #image_out_np[vert*chunk_size: (vert+1)*chunk_size, horiz*chunk_size: (horiz+1)*chunk_size, 0] = 255 # red mask
        image_out = Image.fromarray(image_out_np)
        image_out_name = "chunk"+ str(args.num_chunks) + "_eval_"+ str(vert) + "_" + str(horiz) + ".png"
        print("Saving ",image_out_name)
        image_out.save(image_out_name)

"""
"""
# Make images for stride evaluation figures, 4x4 chunks.
num_chunks = 4
for crop_size in [chunk_size * 2]:
  for vert in range(3):
    horiz = 0
    while ((horiz*chunk_size) + crop_size) <= image_size:
        image_out_np = image_np * 1
        #image_out_np[vert*chunk_size: (vert*chunk_size)+crop_size, horiz*chunk_size: (horiz*chunk_size) + crop_size, 0] = 255 # red mask
        image_out_np[vert*chunk_size: (vert*chunk_size)+crop_size, horiz*chunk_size: (horiz*chunk_size) + crop_size, :] =\
            chunk_image_np[vert*chunk_size: (vert*chunk_size)+crop_size, horiz*chunk_size: (horiz*chunk_size) + crop_size, :] # Crop of image
        image_out = Image.fromarray(image_out_np)
        image_out_name = "chunk"+ str(args.num_chunks) + "_crop" + str(crop_size//chunk_size) + "_eval_"+ str(vert) + "_" + str(horiz) + ".png"
        print("Saving ",image_out_name)
        image_out.save(image_out_name)
        horiz += 1 
"""
"""
# Make images for stride evaluation figures, 8x8 chunks.
#for crop_size in [chunk_size, chunk_size * 2, chunk_size *3]:
for crop_size in [chunk_size*2, chunk_size *4]:
    vert = 0
    horiz = 0
    while ((horiz*chunk_size) + crop_size) <= image_size:
        image_out_np = image_np * 1
        #image_out_np[vert*chunk_size: (vert*chunk_size)+crop_size, horiz*chunk_size: (horiz*chunk_size) + crop_size, 0] = 255 # red mask
        image_out_np[vert*chunk_size: (vert*chunk_size)+crop_size, horiz*chunk_size: (horiz*chunk_size) + crop_size, :] =\
            chunk_image_np[vert*chunk_size: (vert*chunk_size)+crop_size, horiz*chunk_size: (horiz*chunk_size) + crop_size, :] # Crop of image
        image_out = Image.fromarray(image_out_np)
        image_out_name = "chunk"+ str(args.num_chunks) + "_crop" + str(crop_size//chunk_size) + "_eval_"+ str(vert) + "_" + str(horiz) + ".png"
        print("Saving ",image_out_name)
        image_out.save(image_out_name)
        horiz += 1 
"""
"""
# Make images to go next to table showing numbers of evaluations
for crop_size in [2, 4]:
    image_out_np = image_np * 1
    image_out_np[0: crop_size*chunk_size, 0: crop_size*chunk_size, :] = chunk_image_np[0: crop_size*chunk_size, 0: crop_size*chunk_size, :] # Crop of image
    image_out.save("chunk"+str(args.num_chunks)+"_crop"+str(crop_size)+"_eval_0_0.png")
"""
