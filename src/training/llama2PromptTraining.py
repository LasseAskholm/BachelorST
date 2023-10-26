import os

#os.environ["CUDA_VISIBLE_DEVICES"] = str(0) #Limits to use only one GPU

from datasets import Dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM
from transformers import DataCollatorForTokenClassification
from transformers import BitsAndBytesConfig
from utils.Llama2DataLoader import generate_prompt_data_entire_text, generate_prompt_data_sentence
from utils.CommonDataLoader import construct_global_docMap
import torch
from huggingface_hub import login
from loguru import logger
from utils.prompt import generate_prompt_ner
from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_int8_training,
        set_peft_model_state_dict,
    )
from utils.CommonVariables import (
    COMMON_HUGGINGFACE_ACCESS_TOKEN, 
    COMMON_HUGGINGFACE_WRITE_TOKEN, 
    COMMON_DSTL_DOCUMENTS, 
    COMMON_DSTL_ENTITIES,
    COMMON_LLAMA2_MODEL_NAME,
    COMMON_LLAMA2_SELF_LABELED_DATA,
    COMMON_LLAMA2_OUTPUT_DIR,
    COMMON_LLAMA2_LEARNING_RATE,
    COMMON_LLAMA2_TRAIN_BATCH_SIZE,
    COMMON_LLAMA2_EVAL_BATCH_SIZE,
    COMMON_LLAMA2_EPOCHS,
    COMMON_LLAMA2_WEIGHT_DECAY,
    COMMON_LLAMA2_LOGGING_DIR,
    COMMON_LLAMA2_LOGGING_STEPS
    )

#Reduced labeles
reducedLabeles = True

#tokenizer
logger.info("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(COMMON_LLAMA2_MODEL_NAME, token=COMMON_HUGGINGFACE_ACCESS_TOKEN, add_eos_token= True)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

#data collator
data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding= True)

#model
logger.info("Loading model....")
model = AutoModelForCausalLM.from_pretrained(COMMON_LLAMA2_MODEL_NAME, 
                                             token=COMMON_HUGGINGFACE_ACCESS_TOKEN,
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
   
    entities = construct_global_docMap(COMMON_DSTL_ENTITIES)
    df = generate_prompt_data_sentence(entities, COMMON_DSTL_DOCUMENTS, COMMON_LLAMA2_SELF_LABELED_DATA, reducedLabeles)
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
    login(token = COMMON_HUGGINGFACE_WRITE_TOKEN)
    logger.info("Prepping Data")
    #Loading and tokenizing datasets
    train_data, test_data = load_data_sets()
    model.print_trainable_parameters()

    ### define training args here
    ### we should experiment with these hyperparameters.
    ### Leaning rates from bert documentation = (among 5e-5, 4e-5, 3e-5, and 2e-5)
    logger.info("Setting training args....")
    training_args = TrainingArguments(
        output_dir=COMMON_LLAMA2_OUTPUT_DIR,
        evaluation_strategy = "epoch",
        learning_rate = COMMON_LLAMA2_LEARNING_RATE,
        per_device_train_batch_size = COMMON_LLAMA2_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size = COMMON_LLAMA2_EVAL_BATCH_SIZE,
        num_train_epochs = COMMON_LLAMA2_EPOCHS,
        weight_decay = COMMON_LLAMA2_WEIGHT_DECAY,
        logging_dir = COMMON_LLAMA2_LOGGING_DIR,
        logging_steps = COMMON_LLAMA2_LOGGING_STEPS,
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
