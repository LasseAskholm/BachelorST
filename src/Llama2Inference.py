import os
import itertools
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from huggingface_hub import login
from ner_training import load_labels

import torch
from peft import PeftModel
import transformers
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

access_token = "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"
_, label_list = load_labels("../resources/labels.txt")

login(token = access_token)

PROMT_TEMPLATE = f""" Your task is to harness the capabilities of a robust entity extraction model. Equipped with the knowledge of various entity types, your mission is to analyze a provided text from the military context, which includes both a question and context, and identify entities within it. Your goal is to generate a comprehensive, comma-separated list that presents the identified entities alongside their respective labels. The entity types at your disposal include:
Organisation,
Person,
Location,
Money,
Temporal,
Weapon,
MilitaryPlatform

### Instruction:
[INSTRUCTION]

### Response:
"""

question = '''Extract all entities in the following context along with their label from the entities at your disposal: 
Military personnel from the United States and the Kingdom of Bahrain began a 10-day naval exercise Jan. 15 in and off the coast of Bahrain.

Exercise Neon Defender is an annual bilateral training event that enhances collaboration and interoperability among the Bahrain Defence Force, Ministry of Interior and U.S. Naval Forces Central Command (NAVCENT). NAVCENT is headquartered in Manama, Bahrain.

“We are very excited to begin the new year training side by side with the Kingdom of Bahrain, a longstanding, strategic partner,” said Vice Adm. Brad Cooper, commander of NAVCENT and U.S. 5th Fleet. Cooper also leads the Combined Maritime Forces and the International Maritime Security Construct, two U.S.-led international naval coalitions hosted in Bahrain.

“Each year, our mutual commitment to regional maritime security and stability strengthens and expands, and this year is no different,” said Cooper. “This is a great start to meaningful bilateral cooperation we will demonstrate together in 2023.”

The exercise will focus on maritime operations, installation defense, expeditionary operations, tactical combat, medical response and search and rescue training.

Approximately 200 personnel from the U.S. Navy, Marine Corps and Coast Guard are participating in addition to coastal patrol ships USS Monsoon (PC 4) and USS Chinook (PC 9).

“We appreciate the opportunity to sharpen our skills alongside our Bahraini partners,” said Lt. Cmdr. Antoni Wyszynski, the lead exercise planner. “This event brings us together and enables us to learn from each another.”

NAVCENT includes maritime forces stationed in Bahrain and operating in the Arabian Gulf, Gulf of Oman, Red Sea, parts of the Indian Ocean and three critical choke points at the Strait of Hormuz, Suez Canal and Bab al-Mandeb.'''

tokenizer = AutoTokenizer.from_pretrained('LazzeKappa/llama2', token=access_token)

model = AutoModelForCausalLM.from_pretrained('LazzeKappa/llama2', 
                                            device_map='auto',
                                            token=access_token)

model = PeftModel.from_pretrained(model, "LazzeKappa/llama2", torch_dtype=torch.float16, offload_dir="../inferenceLlama2")

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model = model.eval()
model = torch.compile(model)

def create_prompt(instruction: str) -> str:
    return PROMT_TEMPLATE.replace("[INSTRUCTION]", instruction)


def generate_response(prompt: str, model: PeftModel) -> GreedySearchDecoderOnlyOutput:
    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"].to(DEVICE)
 
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        repetition_penalty=1.1,
    )
    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
        )
    
def format_response(response: GreedySearchDecoderOnlyOutput) -> str:
    decoded_output = tokenizer.decode(response.sequences[0])
    response = decoded_output.split("### Response:")[1].strip()
    return "\n".join(textwrap.wrap(response))

def ask_alpaca(prompt: str, model: PeftModel = model) -> str:
    prompt = create_prompt(prompt)
    response = generate_response(prompt, model)
    print(format_response(response))
    

if __name__ == '__main__':
    ask_alpaca(question)