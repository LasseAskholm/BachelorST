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

access_token = "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"


def run_validation():
    login(token = access_token)
    tokenizer = AutoTokenizer.from_pretrained('LazzeKappa/llama2', token=access_token)

    paragraph = '''Extract all entities in the following context along with their label from the entities at your disposal: 
    Military personnel from the United States and the Kingdom of Bahrain began a 10-day naval exercise Jan. 15 in and off the coast of Bahrain.

    Exercise Neon Defender is an annual bilateral training event that enhances collaboration and interoperability among the Bahrain Defence Force, Ministry of Interior and U.S. Naval Forces Central Command (NAVCENT). NAVCENT is headquartered in Manama, Bahrain.

    “We are very excited to begin the new year training side by side with the Kingdom of Bahrain, a longstanding, strategic partner,” said Vice Adm. Brad Cooper, commander of NAVCENT and U.S. 5th Fleet. Cooper also leads the Combined Maritime Forces and the International Maritime Security Construct, two U.S.-led international naval coalitions hosted in Bahrain.

    “Each year, our mutual commitment to regional maritime security and stability strengthens and expands, and this year is no different,” said Cooper. “This is a great start to meaningful bilateral cooperation we will demonstrate together in 2023.”

    The exercise will focus on maritime operations, installation defense, expeditionary operations, tactical combat, medical response and search and rescue training.

    Approximately 200 personnel from the U.S. Navy, Marine Corps and Coast Guard are participating in addition to coastal patrol ships USS Monsoon (PC 4) and USS Chinook (PC 9).

    “We appreciate the opportunity to sharpen our skills alongside our Bahraini partners,” said Lt. Cmdr. Antoni Wyszynski, the lead exercise planner. “This event brings us together and enables us to learn from each another.”

    NAVCENT includes maritime forces stationed in Bahrain and operating in the Arabian Gulf, Gulf of Oman, Red Sea, parts of the Indian Ocean and three critical choke points at the Strait of Hormuz, Suez Canal and Bab al-Mandeb.'''

    tokens = tokenizer(paragraph)
    torch.tensor(tokens['input_ids']).unsqueeze(0).size()

    model = AutoModelForCausalLM.from_pretrained('LazzeKappa/llama2', token=access_token)
    predictions = model.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0), attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
    print(predictions)
    exit()
    predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
    
    words = tokenizer.batch_decode(tokens['input_ids'])
    pd.DataFrame({'ner': predictions, 'words': words})
    pd.DataFrame({'ner': predictions, 'words': words}).to_csv('danish_legal_ner.csv')

if __name__ == '__main__':
    run_validation()