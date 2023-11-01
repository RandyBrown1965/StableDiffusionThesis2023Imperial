import os

os.mkdir("generator_pre_headtohead")
os.mkdir("inpainter_headtohead")
os.mkdir("refiner_headtohead")

base_generator_string = "python refiner_after_complete_sdxl_base_pipeline.py --images_per_prompt 1 --generator sdxl-base --num_inpainting_iterations 0 --random_seed 2005556 --prompt 'child eating candy'"
os.system(base_generator_string)
os.system("mv  images_out_GeneratorOutput_preThesisPaper_* generator_pre_headtohead") 

refiner_generator_string = base_generator_string.replace("sdxl-base", "sdxl-refiner")
os.system(refiner_generator_string)
os.system("mv  images_out_GeneratorOutput_preThesisPaper_* refiner_pre_headtohead") 





base_generator_string = "python refiner_after_complete_sdxl_base_pipeline.py --images_per_prompt 1 --generator sdxl-base --num_inpainting_iterations 0 --random_seed 2005556 --prompt 'busy city street' 'children playing in playground' 'close up of blond woman' dancing farm fireplace foxes 'laundry room' shark shipyard truck"
os.system(base_generator_string)
os.system("mv  images_out_GeneratorOutput_preThesisPaper_* generator_pre_headtohead") 

refiner_generator_string = base_generator_string.replace("sdxl-base", "sdxl-refiner")
os.system(refiner_generator_string)
os.system("mv  images_out_GeneratorOutput_preThesisPaper_* refiner_headtohead") 
