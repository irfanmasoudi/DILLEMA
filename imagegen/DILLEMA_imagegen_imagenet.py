from tqdm import tqdm
import pathlib
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import HEDdetector
from diffusers.utils import load_image
import torch
from torchvision.datasets import ImageNet

dataset = ImageNet(root="/home/jovyan/work/dataset/ILSVRC2012")
gen_image_folder = pathlib.Path("imagenet-gen/")
counterfactuals_folder = pathlib.Path("lim25-imagenet-counterfactuals/")

hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
generator = torch.Generator("cuda").manual_seed(1024)
controlnet = ControlNetModel.from_pretrained(
    "fusing/stable-diffusion-v1-5-controlnet-hed", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

def generate_image(image, prompt):
    image_hed = hed(image)
    negative_prompt = 'low quality, painting, cartoon, greyscale image, (saturated color: 1.9), bright colors'
    image_out = pipe(prompt, 
                     image_hed, 
                     num_inference_steps=5, 
                     negative_prompt=negative_prompt, 
                     generator = generator
                    ).images[0]

    return image_out 

count = 0
last_dir = " "
check_point = 'n01644900'
t_next = True
for i, elem in tqdm(enumerate(dataset)):
    img, y = elem
    img_name = dataset.imgs[i][0].split('/')
    img_file_name = img_name[-1]
    img_dir = img_name[-2]
    
    if img_dir == check_point:
        t_next = False
    
    if img_dir == 'n02843684':
        print('finish')
        break
    
    if (img_dir != last_dir) and (t_next == False):
        print(img_dir, img_file_name, count)
        gen_image_file = gen_image_folder.joinpath(img_dir).joinpath(img_file_name.replace('.JPEG', ''))
        counterfactual_file = counterfactuals_folder.joinpath(img_dir).joinpath(img_file_name.replace('JPEG', 'txt'))
        if count > 124 :
            count = 0
            last_dir = img_dir
            
        else :
            if counterfactual_file.exists():
                with open(counterfactual_file, 'r') as f:
                    counterfactual = (', ').join(f.readlines())
                    for gen in range(5):
                        gen_image = generate_image(img, counterfactual)
                        gen_image_file.parent.mkdir(parents=True, exist_ok=True)
                        gen_image.save(f"{gen_image_file}_{gen}.JPEG")
                        count += 1
            else:
                continue