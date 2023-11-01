#
# Run the inpainter on all the test images in the specified directory
#
import argparse
import os
import shutil
import time
import random
parser = argparse.ArgumentParser()
parser.add_argument('--use_pregenerated_dir')
parser.add_argument('--images_per_prompt', default=4, type = int)
parser.add_argument('--num_inpainting_iterations', default=3, type = int)
parser.add_argument('--random_seed', default=0, type = int)
parser.add_argument('--num_chunks', default=16, type = int)
parser.add_argument('--chunks_per_crop', default=4, type = int)
parser.add_argument('--solid_mask', action="store_true")
parser.add_argument('--random_mask', action="store_true")
args = parser.parse_args()
# Display the parameters and check for possible errors before loading libraries.
if (args.use_pregenerated_dir is not None):
    print("\nPROCESSING PREGENERATED IMAGES ",args.use_pregenerated_dir)
print("num_images_per_prompt = ", args.images_per_prompt)
print("num_inpainting_iterations = ", args.num_inpainting_iterations)
print("num_chunks = ", args.num_chunks)
print("chunks_per_crop = ", args.chunks_per_crop)
print("random_seed = ", args.random_seed)
if args.solid_mask:
    print("SOLID MASK")
if args.random_mask:
    print("RANDOM MASK")
assert not (args.solid_mask and args.random_mask), "You cannot specify a random_mask and a solid_mask simultaneously."
if args.use_pregenerated_dir is not None:
    assert os.path.isdir(args.use_pregenerated_dir), args.use_pregenerated_dir+" is not a valid directory."


list_of_test_pathnames = []
for root, dirs, files in os.walk(args.use_pregenerated_dir):
    for name in files:
        if "Output" not in name:
            list_of_test_pathnames.append(os.path.join(root, name))
        #print(os.path.join(root, name))
    #for name in dirs:
    #    print("D  ",os.path.join(root, name))

print(list_of_test_pathnames)
output_dir = "images_out_BatchC_" + args.use_pregenerated_dir.split("_")[-1]
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
for image_pathname in list_of_test_pathnames:
    #image_pathname = args.use_pregenerated_dir + "/" + image_filename
    print("Processing ", image_pathname)
    print("solid mask")
    # THE SOLID MASK IS OPTIMIZED AT 1 ITERATION
    """
    os.system("python sd_sdxl_imagegen_inpaint.py  --images_per_prompt " + str(args.images_per_prompt)\
            + " --num_inpainting_iterations " + str(1) + " --num_chunks "\
            + str(args.num_chunks) + " --chunks_per_crop " + str(args.chunks_per_crop)\
            + " --use_pregenerated " + image_pathname + " --solid_mask")
    print("random mask")
    os.system("python sd_sdxl_imagegen_inpaint.py  --images_per_prompt " + str(args.images_per_prompt)\
            + " --num_inpainting_iterations " + str(args.num_inpainting_iterations) + " --num_chunks "\
            + str(args.num_chunks) + " --chunks_per_crop " + str(args.chunks_per_crop)\
            + " --use_pregenerated " + image_pathname + " --random_mask")
    print("computed mask")
    os.system("python sd_sdxl_imagegen_inpaint.py  --images_per_prompt " + str(args.images_per_prompt)\
            + " --num_inpainting_iterations " + str(args.num_inpainting_iterations) + " --num_chunks "\
            + str(args.num_chunks) + " --chunks_per_crop " + str(args.chunks_per_crop)\
            + " --use_pregenerated " + image_pathname)

    """
    image_pathname_fix = image_pathname.replace("_seed2005556.png", ".png").replace(".png","_seed02005556.png")
    input_copy_filename = image_pathname_fix.split("/")[-1].replace("_seed", "_" +str(random.random())[2:14] + "_seed")
    shutil.copyfile(image_pathname, output_dir + "/" + input_copy_filename)
