# StableDiffusionThesis2023Imperial
A copy of my 2023 MSc AI Thesis for Imperial College London, plus some additional experiments done afterwards.


To run the inpainting pipeline, install the pip environment:
pip install -r requirements.txt


The main image generation pipeline is:
proposed_inpainting_pipeline.py

usage: proposed_inpainting_pipeline.py [-h] [--generator {sd-v1.5,sd-v2.1,sdxl-base,sdxl-refiner} [{sd-v1.5,sd-v2.1,sdxl-base,sdxl-refiner} ...]] [--images_per_prompt IMAGES_PER_PROMPT]
                                       [--inpainter {sd-v1,sd-v2,sdxl-base,sdxl-refiner} [{sd-v1,sd-v2,sdxl-base,sdxl-refiner} ...]] [--num_inpainting_iterations NUM_INPAINTING_ITERATIONS]
                                       [--random_seed RANDOM_SEED] [--num_chunks NUM_CHUNKS] [--chunks_per_crop CHUNKS_PER_CROP] [--prompt PROMPT [PROMPT ...]] [--solid_mask]
                                       [--random_mask] [--use_pregenerated USE_PREGENERATED] [--mask_threshold MASK_THRESHOLD]

options:
  -h, --help            show this help message and exit

  --prompt PROMPT [PROMPT ...] 
	REQUIRED (unless using a pregenerated image)
	a short description of the desired image(s)
	multiple prompts are accepted
	multi-word prompts must be encased in quotes "" or they will be interpretted as multiple single-word prompts

  --generator {sd-v1.5,sd-v2.1,sdxl-base,sdxl-refiner} [{sd-v1.5,sd-v2.1,sdxl-base,sdxl-refiner} ...]
	Pick one (or more) of the available generators.
	Multiple generators are accepted, which will each generate an image based on the other parameters.
	DEFAULT = sd-v1.5

  --images_per_prompt IMAGES_PER_PROMPT
	Although only a single image is output at the end, this option generates multiple images (at each generation and inpainting stage) from which the best is chosen.
	DEFAULT = 1

  --inpainter {sd-v1,sd-v2,sdxl-base,sdxl-refiner} [{sd-v1,sd-v2,sdxl-base,sdxl-refiner} ...]
	Pick one (or more) of the available inpainters.
	Multiple inpainters are accepted, which will each generate an image based on the other parameters.
	DEFAULT = sd-v2

  --num_inpainting_iterations NUM_INPAINTING_ITERATIONS
	Number of times to iteratively apply inpainting.  A separate mask is determined for each iteration.
	Can be set to 0 to bypass the inpainting completely.
	DEFAULT = 3

  --random_seed RANDOM_SEED
	Random seed for the inpainting generator(s) and inpainter(s).
	DEFAULT = 0

  --num_chunks NUM_CHUNKS
	Resolution of the inpainting mask.  How many horizontal and vertical blocks in the mask?  
	Chunk resolution = (512 / num_chunks) by (512 / num_chunks)
	DEFAULT = 16 (which corresponds to a chunk size of 32x32 pixels within the 512x512 image) 

  --chunks_per_crop CHUNKS_PER_CROP
	Size of the image crops to be evaluated when creating the inpainting mask.
	Evaluation crop resolution = (chunks_per_crop * 512 / num_chunks) by (chunks_per_crop * 512 / num_chunks)
	DEFAULT = 2 (which corresponds to an evaluation crop size of 64x64 pixels if num_chunks is also at the default value (16))

  --solid_mask 
	FLAG
	Use a solid mask for inpainting.  Inpaint the entire image.
	DEFAULT = FALSE (when the solid_mask flag is not asserted)

  --random_mask 
	FLAG
	Use a random mask for inpainting.  The resolution of the mask is determined by the parameter, num_chunks)
	DEFAULT = FALSE (when the random_mask flag is not asserted)

  --use_pregenerated USE_PREGENERATED
	Skip the image generation and feed the specified image directly to the inpainter(s).
	Useful for comparing inpainters on identical inputs or when memory constraints break the generation and inpainting into separate parts.
	DEFAULT = FALSE

  --mask_threshold MASK_THRESHOLD
	Set the threshold value for the inpainting mask.  Chunks below the threshold will be inpainted, and chunks above the threshold won't.
	For example, experimentation established a mask_threshold value of 0.23313 to be used during testing.
	DEFAULT = <The mean value of the crop evaluations done during the first inpainting iteration>

#####

EXAMPLES:
python proposed_inpainting_pipeline.py --generator sdxl-base --images_per_prompt 2 --inpainter sd-v1 --num_inpainting_iterations 2 --random seed 119 --num_chunks 8 --chunks_per_crop 4 --prompt "pumpkin pie" --mask_threshold 0.8

This would generate an image of a pumpkin pie, generated by the sdxl-base generator, followed by 2 inpainting iterations using the sd-v1 inpainter.  
An 8x8 inpainting mask would be determined and upsized to 512x512 pixels (so the inpainting mask would be composed of 64x64 pixel chunks).  These chunks would be determined by evaluating image crops of size 256x256 pixels, and those chunks that evaluated below the 0.8 threshold at each iteration would be inpainted. (0.8 is rather high so nearly all the chunks would likely be inpainted at each iteration.)  At each generation and inpainting step, two images would be rendered and evaluated, the best one chosen to continue through the pipeline, and the other discarded.

---

python proposed_inpainting_pipeline.py --generator sdxl-base --images_per_prompt 2 --inpainter sd-v1 --num_inpainting_iterations 2 --random_seed 119 --num_chunks 8 --chunks_per_crop 4 --prompt "pumpkin pie" --mask_threshold 0.8 --random_mask

The same as above, except that the random_mask flag would override the --chunks_per_crop and --mask_threshold parameters.  At each iteration, a random mask of size 8x8 would be created and upsized to 512x512 pixels for use as an inpainting mask.

---

python proposed_inpainting_pipeline.py --generator sdxl-base --images_per_prompt 2 --inpainter sd-v1 --num_inpainting_iterations 2 --random_seed 119 --num_chunks 8 --chunks_per_crop 4 --prompt "pumpkin pie" --mask_threshold 0.8 --solid_mask

The same as above, except that the solid_mask flag would override the --num_chunks and --chunks_per_crop and --mask_threshold parameters.  At each iteration, a 512x512 pixel mask of all ones would be applied and the entire image would be inpainted.

---

python proposed_inpainting_pipeline.py --prompt "pumpkin pie"

Generate one image of a "pumpkin pie" using all the default values, the (default) sd-v1.5 generator, and the (default) sd-v2 inpainter.

---

python proposed_inpainting_pipeline.py --prompt pumpkin pie

Two images would be generated using all the default values, one image of a "pumpkin" and one image of a "pie."

---

python proposed_inpainting_pipeline.py --prompt "pumpkin pie" --generator sdxl-refiner

Generate one image of a "pumpkin pie" using the sdxl-refiner generator and the (default) sd-v2 inpainter.

---

python proposed_inpainting_pipeline.py --prompt "pumpkin pie" --inpainter sd-v1

Generate one image of a "pumpkin pie" using the (default) sd-v1.5 generator and the sd-v1 inpainter.

---

python proposed_inpainting_pipeline.py --prompt "pumpkin pie" --generator sd-v2.1 sdxl-base

Generate two images of a "pumpkin pie", one using the sd-v1.5 generator and the (default) sd-v2 inpainter, the other using the sdxl-base generator and the (default) sd-v2 inpainter.

---

python proposed_inpainting_pipeline.py --prompt "pumpkin pie" --inpainter sd-v1 sdxl-base

Generate two images of a "pumpkin pie", one using the (defaulat) sd-v1.5 generator and the sd-v1 inpainter, the other using the (default) sd-v1.5 generator and the sdxl-refiner inpainter.

---

python proposed_inpainting_pipeline.py --prompt "pumpkin pie" --generator sd-v2.1 sdxl-base --inpainter sd-v1 sdxl-base

Generate four images of a "pumpkin pie", using each of the generators (sd-v2.1 and sdxl-base) with each of the inpainters (sd-v1 and sdxl-base).

#####

OUTPUT FILES:
Each of the images output will be saved in to the directory images_out_ThesisPaper.  
Currently, the pipeline also saves out the output of the generator into a directory named images_outGeneratorOutput_preThesisPaper_<prompt>_<timestamp>.
It also saves the output of the generator and the mask and output of each inpainting iteration into a folder named images_out_progression_<num_chunks value>_<chunks_per_crop value>_<prompt>_<timestamp>.







