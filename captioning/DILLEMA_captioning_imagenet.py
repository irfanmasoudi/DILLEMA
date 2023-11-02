from lavis.models import load_model_and_preprocess
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import pathlib
# Dataset imports
from torchvision.datasets import ImageNet

# direction path folder
captions_folder = pathlib.Path("/home/ubuntu/imagenet-captions/")

# dataset root path
dataset = ImageNet(root="/data/")

model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_opt",
    model_type="caption_coco_opt6.7b",
    is_eval=True,
    device="cuda",
)

def get_caption(image):
    image = vis_processors["eval"](image).unsqueeze(0).to("cuda")  
    generated_caption = model.generate(
        {"image": image},
        use_nucleus_sampling=True,
        num_captions=2
    )
    generated_caption = '. '.join(generated_caption)
    return generated_caption

# add the usual imports
for i, elem in tqdm(enumerate(dataset)):
    x, y = elem
    img_name = dataset.imgs[i][0].split('/')
    # Retrieve information from dataset
    img_file_name = img_name[-1]
    img_dir = img_name[-2]
    # Where the generated caption is saved
    caption_file = captions_folder.joinpath(img_dir).joinpath(img_file_name.replace('JPEG', 'txt'))
    # If the result already exists, skip to next image
    if caption_file.exists():
        continue
    img = x
    caption = get_caption(img)
    # Write result file
    caption_file.parent.mkdir(parents=True, exist_ok=True)
    with open(caption_file, 'w') as f:
        f.write(caption)
