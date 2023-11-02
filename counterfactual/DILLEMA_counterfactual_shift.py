from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
import pathlib
from ctransformers import AutoModelForCausalLM
import re
import ast
import random
# Dataset imports
from shift_dev import SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import FileBackend, ZipBackend

captions_folder = pathlib.Path("/home/ubuntu/captions/")
counterfactuals_folder = pathlib.Path("/home/ubuntu/counterfactuals/")
word_lists_folder = pathlib.Path("/home/ubuntu/word_lists/")
alternatives_folder = pathlib.Path("/home/ubuntu/alternatives/")
applied_alternatives_folder = pathlib.Path("/home/ubuntu/applied_alternatives/")
task = "semantic segmentation for autonomous driving"

config = {"temperature": 0.1, "top_p": 0.85, "repetition_penalty": 1, "max_new_tokens": 2048, "context_length": 2048}
llm_model = AutoModelForCausalLM.from_pretrained(
    "models/llama-2-13b-chat.ggmlv3.q5_1.bin",
    model_type="llama",
    **config,
    gpu_layers=500
)
llm_begin_inst, llm_end_inst = "[INST]", "[/INST]"
llm_begin_sys, llm_end_sys = "<<SYS>>\n", "\n<</SYS>>\n\n"

default_system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully 
as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, 
dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in 
nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering 
something not correct. If you don't know the answer to a question, please don't share false information."""

llm_system_prompt = llm_begin_sys + default_system_prompt + llm_end_sys

def get_word_list(caption):
    instruction = f"""What are word related to the image described by the [{caption}] that can be modified without 
    modifying the label or class of the image if the task is {task}. For example, if the task is related to cars, 
    then you cannot put the word car in the list. Please just give me only the result as a list of words that formatted 
    in python list, without any additional response from you!. For example, list = ['color', 'people', 'rainy']"""
    template = llm_begin_inst + llm_system_prompt + instruction + llm_end_inst
    result = llm_model(template)
    pattern = r'\[([^\]]+)\]'
    matches = re.search(pattern, result)
    word_list_text = matches.group(1)
    words = [word.strip() for word in word_list_text.split(',')]
    words = [word.strip(" '") for word in words]
    word_list = (', ').join(words)
    return word_list

def get_alternatives(caption, word_list):
    """
    Class method to get counterfactual as a JSON formatted based on word_list and context from caption

    Args:
        caption : (str) list of generated_caption
        word_list : (str) generated word
    """
    instruction = f"""Give me some possible words which is composed of counterfactual for each word in this list: 
    [{word_list}]. For example car: [bike, motorcycle] which is different type of vehicle, person: 
    [woman, man, boy, girl], rainy: [sunny, snowy, clear] which is different condition of weather, red: 
    [green, yellow] which is possible traffic light color, night: [day, lights]). The contexts are [{caption}]. 
    Please give me only the result which is formatted in JSON. Do not provide me any additional responses."""
    template = llm_begin_inst + llm_system_prompt + instruction + llm_end_inst
    result = llm_model(template)
    start_index = result.find("{")
    end_index = result.rfind("}")
    alternatives = result[start_index:end_index + 1]
    return alternatives

def get_counterfactual(caption, alternatives):
    """
    Class method to get the final prompt

    Args:
        caption: (str) list of generated_caption
        counterfactual: (str) JSON generated counterfactual
    """
    while True:
        data = ast.literal_eval(alternatives)
        keys = random.sample(list(data.keys()), min(len(data), 3))
        data = {k: data[k] for k in keys}
        lst_data = []
        for key in data.keys():
            # res = "'"+'Replace word ' + key +' become ' + random.choice(data[key])+"'"
            res = '"' + key + '": "' + random.choice(data[key]) + '"'
            lst_data.append(res)
        applied_alternatives = ", ".join(lst_data)
        caption_lst = caption.split('. ')
        instruction = f"""This is a list of image captions: {caption_lst}. Please replace some words of the image captions with 
        the word from this dictionary: [{applied_alternatives}]. Provide me the just result with formatted as a python list!. 
        Do not compare the revised result with the original one. 
        If the input sentence is "a street in the night, a dark road" and the alternatives are {{"night": "morning", "afternoon", "dark": "red"}} 
        your output can be result = ["a beautiful road in the morning", "a red street"]. 
        This is the expected result example, result = ["sentence A", "sentence B"]. Provide only the result.\n\n."""
        template = llm_begin_inst + llm_system_prompt + instruction + llm_end_inst
        result = llm_model(template)
        sentences = re.findall(r"['\"](.*?)['\"]", result)
        counterfactual = ", ".join(sentences)
        if len(counterfactual) > 0 and len(sentences) == 2:
            break
        else:
            print(counterfactual)
            print(sentences)
            print(result)
    return counterfactual, applied_alternatives

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
    # Where the generated results are saved
    counterfactual_file = counterfactuals_folder.joinpath(img_dir).joinpath(img_name.replace('jpg', 'txt'))
    word_list_file = word_lists_folder.joinpath(img_dir).joinpath(img_name.replace('jpg', 'txt'))
    alternatives_file = alternatives_folder.joinpath(img_dir).joinpath(img_name.replace('jpg', 'txt'))
    applied_alternatives_file = applied_alternatives_folder.joinpath(img_dir).joinpath(img_name.replace('jpg', 'txt'))
    # If the result already exists, skip to next image
    if counterfactual_file.exists() and word_list_file.exists() and alternatives_file.exists() and applied_alternatives_file.exists():
        continue
    caption_file = captions_folder.joinpath(img_dir).joinpath(img_name.replace('jpg', 'txt'))
    with open(caption_file, 'r') as f:
        caption = '\n'.join(f.readlines())
        word_list = get_word_list(caption)
        alternatives = get_alternatives(caption, word_list)
        counterfactual, applied_alternatives = get_counterfactual(caption, alternatives)
        if len(counterfactual) > 0:
            print(counterfactual_file)
            counterfactual_file.parent.mkdir(parents=True, exist_ok=True)
            with open(counterfactual_file, 'w') as f:
                f.write(counterfactual)
            word_list_file.parent.mkdir(parents=True, exist_ok=True)
            with open(word_list_file, 'w') as f:
                f.write(word_list)
            alternatives_file.parent.mkdir(parents=True, exist_ok=True)
            with open(alternatives_file, 'w') as f:
                f.write(alternatives)
            applied_alternatives_file.parent.mkdir(parents=True, exist_ok=True)
            with open(applied_alternatives_file, 'w') as f:
                f.write(applied_alternatives)
        else:
            print(f"failed generation for {i}")
            print(caption)
            print(word_list)
            print(alternatives)
            print(counterfactual)
            print(applied_alternatives)
