# from lavis.models import load_model_and_preprocess
import time
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import pathlib
# Dataset imports
from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import FileBackend, ZipBackend

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import HEDdetector
from diffusers.utils import load_image
import torch

gen_image_folder = pathlib.Path("imagegen/")
counterfactuals_folder = pathlib.Path("counterfactuals/")

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

dataset = ds = SHIFTDataset(
    data_root="../../dataset/shift",
    split="train",
    keys_to_load=[
        Keys.images,
        Keys.intrinsics,
        Keys.boxes2d,
        Keys.boxes2d_classes,
        Keys.boxes2d_track_ids,
        Keys.segmentation_masks,
    ],
    views_to_load=["front"],
    framerate="images",
    shift_type="continuous/1x",
    backend=FileBackend(),
    verbose=True,
)

tensor_to_image = T.ToPILImage()

for i, x in tqdm(enumerate(dataset)):
    # Retrieve information from dataset
    img_name = x['front']['name']
    img_dir = x['front']['videoName']
    # Where the generated results are saved
    gen_image_file = gen_image_folder.joinpath(img_dir).joinpath(img_name)
    
    counterfactual_file = counterfactuals_folder.joinpath(img_dir).joinpath(img_name.replace('jpg', 'txt'))

    if gen_image_file.exists():
        continue
    # If the result already exists, skip to next image
    
    with open(counterfactual_file, 'r') as f:
        counterfactual = (', ').join(f.readlines())
        img = tensor_to_image(x['front']['images'][0]/255)
        gen_image = generate_image(img, counterfactual)
        # print(counterfactual)
        gen_image_file.parent.mkdir(parents=True, exist_ok=True)
        gen_image.save(gen_image_file)
        # if i > 10:
        #     end = time.time()
        #     break
    
