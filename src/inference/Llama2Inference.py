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
from utils.CommonVariables import (
    COMMON_HUGGINGFACE_ACCESS_TOKEN
)

from utils.prompt import generate_prompt_ner_inference

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# The model to load from Huggingface
model_name = "LazzeKappa/L06"


# Reload model in FP16 and merge it with LoRA weights
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=COMMON_HUGGINGFACE_ACCESS_TOKEN,
    load_in_8bit= True,
    torch_dtype=torch.float16,
    device_map="auto",
)


# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, token=COMMON_HUGGINGFACE_ACCESS_TOKEN, add_eos_token= True)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

def ask_alpacha(context: str):
    prompt = generate_prompt_ner_inference(context)

    pipe = pipeline(
    task="text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_length=800
    )

    result = pipe(prompt) 

    print(result[0]['generated_text'])



if __name__ == '__main__':
    path = "../../data/self-labeled-data/ValData/ValGEN.txt"
    with open (path, "r", encoding="utf-8") as file:
        context = file.read().replace('\n', '')
    ask_alpacha(context)