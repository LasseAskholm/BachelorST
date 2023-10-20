import os

#os.environ["CUDA_VISIBLE_DEVICES"] = str(0) #Limits to use only one GPU

from datasets import Dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM
from transformers import DataCollatorForTokenClassification
from transformers import BitsAndBytesConfig
from LoadTestData import generate_prompt_data, construct_global_docMap, generate_prompt_data_sentence, generate_df_from_additional_data
import torch
from huggingface_hub import login
from loguru import logger
from prompt import generate_prompt_ner
from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_int8_training,
        set_peft_model_state_dict,
    )



access_token = "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"
write_token = "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"
#tokenizer
logger.info("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token, add_eos_token= True)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"
#data collator
data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding= True)

#model
logger.info("Loading model....")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", 
                                             token=access_token,
                                             load_in_8bit= True,
                                             torch_dtype = torch.float16,
                                             device_map ="auto")

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

def load_data_sets():
   
    entities = construct_global_docMap("../data/re3d-master/*/entities_cleaned_sorted_and_filtered.json")
    df = generate_prompt_data_sentence(entities, "../data/re3d-master/*/documents.json")
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset['train'].map(tokenize_and_generate_prompt)
    test_dataset = dataset['test'].map(tokenize_and_generate_prompt)

    return (train_dataset, test_dataset)

def tokenize_and_generate_prompt(data_point):
    full_prompt = generate_prompt_ner(context =data_point['context'], output= data_point['answers'])
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt

def tokenize(prompt, add_eos_token = True):
    result = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors=None)
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < 512
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result

def main ():
    login(token = write_token)
    logger.info("Prepping Data")
    #Loading and tokenizing datasets
    train_data, test_data = load_data_sets()
    model.print_trainable_parameters()

    ### define training args here
    ### we should experiment with these hyperparameters.
    ### Leaning rates from bert documentation = (among 5e-5, 4e-5, 3e-5, and 2e-5)
    logger.info("Setting training args....")
    training_args = TrainingArguments(
        output_dir="../llama2",
        evaluation_strategy = "epoch",
        learning_rate = 3e-4,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        num_train_epochs = 10,
        weight_decay = 1e-5,
        logging_dir = "../logging",
        logging_steps = 10,
        fp16= True,
        optim = "adamw_torch"
    )
    ### define trainer here
    logger.info("Defining Trainer...")
    trainer = Trainer (
        model,
        training_args,
        train_dataset = train_data,
        eval_dataset = test_data,
        data_collator = data_collator,
        tokenizer = tokenizer,
    )
    #start training model
    logger.info("STARTING TRAINING OF PROMPTING NER_MODEL")
    trainer.train()
    trainer.push_to_hub()

if __name__ == '__main__':
    main()


#akjen