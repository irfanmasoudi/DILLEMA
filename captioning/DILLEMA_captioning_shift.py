from lavis.models import load_model_and_preprocess
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import pathlib
# Dataset imports
from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import FileBackend, ZipBackend

captions_folder = pathlib.Path("/home/ubuntu/captions/")

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

dataset = ds = SHIFTDataset(
    data_root="/home/ubuntu/shift",
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
    # Where the generated caption is saved
    caption_file = captions_folder.joinpath(img_dir).joinpath(img_name.replace('jpg', 'txt'))
    # If the result already exists, skip to next image
    if caption_file.exists():
        continue
    img = tensor_to_image(x['front']['images'][0]/255)
    caption = get_caption(img)
    # Write result file
    caption_file.parent.mkdir(parents=True, exist_ok=True)
    with open(caption_file, 'w') as f:
        f.write(caption)
