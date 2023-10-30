# import
from transformers import AutoProcessor, AutoModel
from PIL import Image
from torchvision import transforms
import numpy as np
import torch

# load model
device = "cuda"
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

def calc_probs(prompt, images):
    
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)
    
    return probs.cpu().tolist()

############################################################################################
def compute_patch_mask(image_full_tensor, chunk_size: int):
    _, xdim_in, ydim_in = image_full_tensor.shape
    print("Full Image=",xdim_in,"x",ydim_in)
    xdim_out, ydim_out = xdim_in//chunk_size, ydim_in//chunk_size
    computed_mask = np.zeros((xdim_out, ydim_out))
    print("mask=",computed_mask.shape)

    for x in range(0, xdim_in, chunk_size):
        for y in range(0, ydim_in, chunk_size):
            image_crop_tensor = image_full_tensor[:, x: x+chunk_size, y: y+chunk_size]
            image_crop = transforms.ToPILImage()(image_crop_tensor)
            evaluation = calc_probs(prompt, [image_crop, image_brunette])
            computed_mask[int(x/chunk_size), int(y/chunk_size)] = int(evaluation[0] * 100)
    return computed_mask   

def compute_stride_mask(image_full_tensor,chunk_size: int):
    _, xdim_in, ydim_in = image_full_tensor.shape
    print("Full Image=",xdim_in,"x",ydim_in)
    xdim_out, ydim_out = xdim_in//chunk_size, ydim_in//chunk_size
    evaluations_total = np.zeros((xdim_out, ydim_out))
    evaluations_count = np.zeros((xdim_out, ydim_out), dtype = int)
    print("mask=",evaluations_total.shape)

    stride = chunk_size
    #crop_size_x, crop_size_y = xdim_in//2, ydim_in//2
    crop_size_x, crop_size_y = 2*chunk_size, 2*chunk_size
    ones_array = np.ones((crop_size_x // chunk_size, crop_size_y // chunk_size), dtype = int)
    for x in range(0, xdim_in - crop_size_x + 1 , chunk_size):
        for y in range(0, ydim_in - crop_size_y + 1, chunk_size):
            image_crop_tensor = image_full_tensor[:, x: x+crop_size_x, y: y+crop_size_y]
            image_crop = transforms.ToPILImage()(image_crop_tensor)
            #image_crop.save("image_crop_"+str(x)+str(y)+".png")
            evaluation = calc_probs(prompt, [image_crop, image_brunette])
            evaluations_count[x // chunk_size: (x+crop_size_x) // chunk_size, y // chunk_size: (y+crop_size_y) // chunk_size] += ones_array
            evaluations_total[x // chunk_size: (x+crop_size_x) // chunk_size, y // chunk_size: (y+crop_size_y) // chunk_size] += int(evaluation[0] * 100) * ones_array
            print("evaluations_count = \n", evaluations_count)
            print("evaluations_total = \n", (evaluations_total).astype(int))
    computed_mask = evaluations_total / evaluations_count
    return computed_mask   
####################################################################################

if __name__ == "__main__":
    #prompt = "a lion leaping off a rock, midnight, rim lit"
    prompt = "Photorealistic"
    prompt = "(((Hyperrealistic, photographic realism)))"
    
    print("Pick the best initial image")
    best_index=7
    for i in range(8):
        pil_images = [Image.open("images_lions/SDXLalionleapingoffarockmidnightrimlit_stable-diffusion-xl-base-1.0_seed119_50steps_"+str(best_index)+"_cfg7.5.png"),
                Image.open("images_lions/SDXLalionleapingoffarockmidnightrimlit_stable-diffusion-xl-base-1.0_seed119_50steps_"+str(i)+"_cfg7.5.png")]
        preference_probs = calc_probs(prompt, pil_images)
        print("Image",best_index," vs Image",i,"   ", preference_probs)
        if preference_probs[1] > 0.5:
            best_index = i
    print("People will probably prefer image ",best_index)
    
    print("\nJudge the inpainting iterations")
    for i in range(9):
        pil_images = [Image.open("images_lions/SDXLOutput_image_strength0.7_"+str(i)+".png"), Image.open("images_lions/SDXLOutput_image_strength0.7_"+str(i+1)+".png")]
        print("Image",i," vs Image",i+1,"   ",calc_probs(prompt, pil_images))
    a,b = 2,6
    pil_images = [Image.open("images_lions/SDXLOutput_image_strength0.7_"+str(a)+".png"), Image.open("images_lions/SDXLOutput_image_strength0.7_"+str(b)+".png")]
    print("Image",a," vs Image",b,"   ",calc_probs(prompt, pil_images))
    
    
    print("\nJudge the image content")
    image_astronaut = Image.open("images_varied/SDXL_astronaut_jungle_refined_photo.png")
    #image_brunette = Image.open("images_varied/SDXL_brunette_holding_book_refined2.png")
    image_brunette = Image.open("images_varied/SDXL_brunette_holding_book_refined3.png")
    image_lion = Image.open("images_varied/SDXL_lion_refined.png")
    prompt = "a lion leaping off a rock, midnight, rim lit"
    print("Lion correct prediction = ",calc_probs(prompt, [image_lion, image_astronaut]))
    print("Lion correct prediction = ",calc_probs(prompt, [image_lion, image_brunette]))
    prompt = "Brunette holding a book"
    print("Brunette correct prediction = ",calc_probs(prompt, [image_brunette, image_astronaut]))
    print("Brunette correct prediction = ",calc_probs(prompt, [image_brunette, image_lion]))
    prompt = "Astronaut in a jungle"
    print("Astronaut correct prediction = ",calc_probs(prompt, [image_astronaut, image_brunette]))
    print("Astronaut correct prediction = ",calc_probs(prompt, [image_astronaut, image_lion]))
    prompt = "Photorealistic"
    print("Photorealism: Astronaut vs Brunette = ",calc_probs(prompt, [image_astronaut, image_brunette]))
    print("Photorealism: Astronaut vs Lion = ",calc_probs(prompt, [image_astronaut, image_lion]))
    print("Photorealism: Brunette vs Lion = ",calc_probs(prompt, [image_brunette, image_lion]))
    
    
    
    # CUT THE INPUT IMAGE INTO CROPS
    chunk_size = 64
    prompt = "Photorealistic"
    image_full_tensor = transforms.ToTensor()(image_brunette)
    #image_full_tensor = transforms.ToTensor()(image_lion)
    

    computed_mask = compute_patch_mask(image_full_tensor, chunk_size)
    boolean_mask50 = (computed_mask < 50).astype(int)
    boolean_mask45 = (computed_mask < 45).astype(int)
    boolean_mask40 = (computed_mask < 40).astype(int)
    boolean_mask35 = (computed_mask < 35).astype(int)
    print("computed mask = \n", computed_mask)
    print("Boolean mask50 = \n", boolean_mask50)
    print("Boolean mask45 = \n", boolean_mask45)
    print("Boolean mask40 = \n", boolean_mask40)
    print("Boolean mask35 = \n", boolean_mask35)
    
    computed_mask = compute_stride_mask(image_full_tensor, chunk_size)
    print("computed mask = \n", computed_mask.astype(int))
    boolean_mask50 = (computed_mask < 50).astype(int)
    boolean_mask45 = (computed_mask < 45).astype(int)
    boolean_mask40 = (computed_mask < 40).astype(int)
    boolean_mask35 = (computed_mask < 35).astype(int)
    boolean_mask30 = (computed_mask < 30).astype(int)
    boolean_mask25 = (computed_mask < 25).astype(int)
    boolean_mask20 = (computed_mask < 20).astype(int)
    print("Boolean mask50 = \n", boolean_mask50)
    print("Boolean mask45 = \n", boolean_mask45)
    print("Boolean mask40 = \n", boolean_mask40)
    print("Boolean mask35 = \n", boolean_mask35)
    print("Boolean mask30 = \n", boolean_mask30)
    print("Boolean mask25 = \n", boolean_mask25)
    print("Boolean mask20 = \n", boolean_mask20)
