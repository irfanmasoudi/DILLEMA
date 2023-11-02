from tqdm import tqdm
from PIL import Image
# import torchvision.transforms as T
import pathlib
from ctransformers import AutoModelForCausalLM
import re
import ast
import random

from torchvision.datasets import ImageNet

import pandas as pd

# get label for prompt
url = "https://gist.githubusercontent.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57/raw/aa66dd9dbf6b56649fa3fab83659b2acbf3cbfd1/map_clsloc.txt"
df = pd.read_csv(url, sep=" ", names=["nwid", "id", "class"]).set_index('nwid')
mapping = df.to_dict()

dataset = ImageNet(root="/data/")


# path setting
captions_folder = pathlib.Path("/home/ubuntu/imagenet-captions/")
counterfactuals_folder = pathlib.Path("/home/ubuntu/imagenet-counterfactuals/")
word_lists_folder = pathlib.Path("/home/ubuntu/imagenet-word_lists/")
alternatives_folder = pathlib.Path("/home/ubuntu/imagenet-alternatives/")
applied_alternatives_folder = pathlib.Path("/home/ubuntu/imagenet-applied_alternatives/")

# task specification
task = "ImageNet image classification with 1000 labels"

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

def get_word_list(caption, label_name):
    instruction = f"""What are word related to the image described by sentences:  [{caption}] that can be modified without modifying the label or class of the image if the task is {task} and the label is {label_name}. For example, the sentence is: "the man with a fish", if the label is fish", then fish should not be included in the result). Please just give me only the result as a list of words that formatted in python list, without any additional response from you!. For example, list= ['color', 'man', 'car']"""
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
    function to get counterfactual as a JSON formatted based on word_list and context from caption

    Args:
        caption : (str) list of generated_caption
        word_list : (str) generated word
    """
    instruction = f"""Give me some possible words which is composed of counterfactual for each word in this list: [{word_list}]. For example "small": ["big", "medium"] which is different type of size, "person": ["woman", "man", "boy", "girl"], "rainy": ["sunny", "snowy", "clear"] which is different condition of weather, "red": ["green", "yellow", "blue"] which is a possible color, or "night": ["day", "lights"]. The contexts are [{caption}]. Please give me only the result which is formatted in JSON. Do not provide me any additional responses. For example, result = {{"small": ["big", "medium"], "person": ["woman", "man", "boy", "girl"], "rainy": ["sunny", "snowy", "clear"], "red": ["green", "yellow", "blue"], "night": ["day", "lights"]}}."""
    template = llm_begin_inst + llm_system_prompt + instruction + llm_end_inst
    result = llm_model(template)
    start_index = result.find("{")
    end_index = result.rfind("}")
    alternatives = result[start_index:end_index + 1]
    return alternatives

def get_counterfactual(caption, alternatives):
    """
    function to get the final prompt

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
            res = '"' + key + '": "' + random.choice(data[key]) + '"'
            lst_data.append(res)
        applied_alternatives = ", ".join(lst_data)
        caption_lst = caption.split('. ')
        instruction = f"""This is a list of image captions: {caption_lst}. Please replace some words of the image captions with the word from this dictionary: [{applied_alternatives}]. And the label Provide me the just the result with formatted as a python list!. Do not compare the revised result with the original one. If the input sentence is "a street in the night, a dark road" and the alternatives are {{"night": "morning", "afternoon", "dark": "red"}} your output can be result = ["a beautiful road in the morning", "a red street"]. Provide me only the result!\n\n."""
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


for i, elem in tqdm(enumerate(dataset)):
    img_name = dataset.imgs[i][0].split('/')
    # Retrieve information from dataset
    img_file_name = img_name[-1]
    img_dir = img_name[-2]
    # Where the generated caption is saved
    counterfactual_file = counterfactuals_folder.joinpath(img_dir).joinpath(img_file_name.replace('JPEG', 'txt'))
    caption_file = captions_folder.joinpath(img_dir).joinpath(img_file_name.replace('JPEG', 'txt'))
    word_list_file = word_lists_folder.joinpath(img_dir).joinpath(img_file_name.replace('JPEG', 'txt'))
    alternatives_file = alternatives_folder.joinpath(img_dir).joinpath(img_file_name.replace('JPEG', 'txt'))
    applied_alternatives_file = applied_alternatives_folder.joinpath(img_dir).joinpath(img_file_name.replace('JPEG', 'txt'))
    
    # If the result already exists, skip to next image
    if counterfactual_file.exists() and word_list_file.exists() and alternatives_file.exists() and applied_alternatives_file.exists():
        continue
    caption_file = captions_folder.joinpath(img_dir).joinpath(img_file_name.replace('JPEG', 'txt'))
    with open(caption_file, 'r') as f:
        caption = '\n'.join(f.readlines())
        word_list = get_word_list(caption, mapping['class'][img_dir])
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
