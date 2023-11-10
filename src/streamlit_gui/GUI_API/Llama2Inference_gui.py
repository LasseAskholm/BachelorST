import os
import textwrap

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel

from huggingface_hub import login
from utils_gui.CommonVariables import (
    COMMON_HUGGINGFACE_ACCESS_TOKEN
)

from utils_gui.prompt import generate_prompt_ner_inference, generate_single_label_prompt_ner_inference

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# The model to load from Huggingface
model_name6 = "LazzeKappa/L06"   #alle
# model_name7 = "LazzeKappa/L07"   #single
model_name8 = "LazzeKappa/L08"   #alle
model_name9 = "LazzeKappa/L09"   #single


# Reload model in FP16 and merge it with LoRA weights
model6 = AutoModelForCausalLM.from_pretrained(
    model_name6,
    token=COMMON_HUGGINGFACE_ACCESS_TOKEN,
    load_in_8bit= True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model8 = AutoModelForCausalLM.from_pretrained(
    model_name8,
    token=COMMON_HUGGINGFACE_ACCESS_TOKEN,
    load_in_8bit= True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model9 = AutoModelForCausalLM.from_pretrained(
    model_name9,
    token=COMMON_HUGGINGFACE_ACCESS_TOKEN,
    load_in_8bit= True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Reload tokenizer to save it
tokenizer6 = AutoTokenizer.from_pretrained(model_name6, token=COMMON_HUGGINGFACE_ACCESS_TOKEN, add_eos_token= True)
tokenizer6.pad_token_id = 0
tokenizer6.padding_side = "left"

tokenizer8 = AutoTokenizer.from_pretrained(model_name8, token=COMMON_HUGGINGFACE_ACCESS_TOKEN, add_eos_token= True)
tokenizer8.pad_token_id = 0
tokenizer8.padding_side = "left"

tokenizer9 = AutoTokenizer.from_pretrained(model_name9, token=COMMON_HUGGINGFACE_ACCESS_TOKEN, add_eos_token= True)
tokenizer9.pad_token_id = 0
tokenizer9.padding_side = "left"


def get_prompt(context, model):
    if model == "Llama2_v6":
        return generate_prompt_ner_inference(context)
    elif model == "Llama2_v7":
        return generate_prompt_ner_inference(context)
    elif model == "Llama2_v9":
        return generate_single_label_prompt_ner_inference(context)


def get_pipeline(model_name):
   
    if model_name == "Llama2_v6":
        return pipeline(
            task="text-generation", 
            model=model6, 
            tokenizer=tokenizer6, 
            return_full_text=False,
            max_length=1024
            )
    elif model_name == "Llama2_v8":
        return pipeline(
            task="text-generation", 
            model=model8, 
            tokenizer=tokenizer8, 
            return_full_text=False,
            max_length=1024
            )
    elif model_name == "Llama2_v9":
        return pipeline(
            task="text-generation", 
            model=model9, 
            tokenizer=tokenizer9, 
            return_full_text=False,
            max_length=1024
            )

def ask_alpacha(context: str, model_name):

    prompt = get_prompt(context, model_name)

    pipe = get_pipeline(model_name)

    result = pipe(prompt) 

    print(result[0]['generated_text'])

    return(result[0]['generated_text'])

# def ask_alpacha_single_label(context: str, label: str):
#     #TODO: Create validation of chosen label. 

#     prompt = generate_single_label_prompt_ner_inference(label, context)

#     pipe = pipeline(
#     task="text-generation", 
#     model=model, 
#     tokenizer=tokenizer, 
#     return_full_text=False,
#     max_length=1024
#     )

#     result = pipe(prompt) 

#     print(result[0]['generated_text'])



# if __name__ == '__main__':
#     path = "../../data/selv-labeled-data/ValData/ValGEN.txt"
#     with open (path, "r", encoding="utf-8") as file:
#         for line in file:
#             ask_alpacha(line)
#             print("--------")