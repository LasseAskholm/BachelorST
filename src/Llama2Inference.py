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
import torch
from ner_training import load_labels
import textwrap

access_token = "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"
_, label_list = load_labels("../resources/labels.txt")

login(token = access_token)

torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained('LazzeKappa/llama2', token=access_token)

model = AutoModelForCausalLM.from_pretrained('LazzeKappa/llama2', 
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             token=access_token)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
Your task is to harness the capabilities of a robust entity extraction model. 
Equipped with the knowledge of various entity types, your mission is to analyze a provided text from the military context, which includes both a question and context, and identify entities within it. 
Your goal is to generate a comprehensive, comma-separated list that presents the identified entities alongside their respective labels. 
The entity types at your disposal include:

Organisation,
Person,
Location,
Money,
Temporal,
Weapon,
MilitaryPlatform"""
SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

prompt = '''Extract all entities in the following context along with their label from the entities at your disposal: 
Military personnel from the United States and the Kingdom of Bahrain began a 10-day naval exercise Jan. 15 in and off the coast of Bahrain.

Exercise Neon Defender is an annual bilateral training event that enhances collaboration and interoperability among the Bahrain Defence Force, Ministry of Interior and U.S. Naval Forces Central Command (NAVCENT). NAVCENT is headquartered in Manama, Bahrain.

“We are very excited to begin the new year training side by side with the Kingdom of Bahrain, a longstanding, strategic partner,” said Vice Adm. Brad Cooper, commander of NAVCENT and U.S. 5th Fleet. Cooper also leads the Combined Maritime Forces and the International Maritime Security Construct, two U.S.-led international naval coalitions hosted in Bahrain.

“Each year, our mutual commitment to regional maritime security and stability strengthens and expands, and this year is no different,” said Cooper. “This is a great start to meaningful bilateral cooperation we will demonstrate together in 2023.”

The exercise will focus on maritime operations, installation defense, expeditionary operations, tactical combat, medical response and search and rescue training.

Approximately 200 personnel from the U.S. Navy, Marine Corps and Coast Guard are participating in addition to coastal patrol ships USS Monsoon (PC 4) and USS Chinook (PC 9).

“We appreciate the opportunity to sharpen our skills alongside our Bahraini partners,” said Lt. Cmdr. Antoni Wyszynski, the lead exercise planner. “This event brings us together and enables us to learn from each another.”

NAVCENT includes maritime forces stationed in Bahrain and operating in the Arabian Gulf, Gulf of Oman, Red Sea, parts of the Indian Ocean and three critical choke points at the Strait of Hormuz, Suez Canal and Bab al-Mandeb.'''


def run_validation():
    tokens = tokenizer(paragraph)
    torch.tensor(tokens['input_ids']).unsqueeze(0).size()

    predictions = model.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0), attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
    predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
    
    words = tokenizer.batch_decode(tokens['input_ids'])
    print(pd.DataFrame({'ner': predictions, 'words': words}))

def get_prompt(instruction):
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

def remove_substring(string, substring):
    return string.replace(substring, "")



def generate(text):
    prompt = get_prompt(text)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs,
                                 max_new_tokens=512,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id,
                                 )
        final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_outputs = cut_off_text(final_outputs, '</s>')
        final_outputs = remove_substring(final_outputs, prompt)

    return final_outputs#, outputs

def parse_text(text):
        wrapped_text = textwrap.fill(text, width=100)
        print(wrapped_text +'\n\n')
        # return assistant_text

if __name__ == '__main__':
    generated_text = generate(prompt)
    parse_text(generated_text)