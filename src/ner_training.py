import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0) #Limits to use only one GPU

import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM, TrainingArguments, Trainer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import BitsAndBytesConfig
from LoadTestData import construct_global_docMap, map_all_entities
from SortEntities import sort_entities_assending
import torch
from huggingface_hub import login
from loguru import logger
import evaluate

access_token = "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"
write_token = "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"
seqeval = evaluate.load("seqeval")

def load_labels(labels_path):
    with open(labels_path, encoding='utf-8') as file:
        labels = file.readlines()
        
    labels = [label if label[-1:] != '\n' else label[:-1] for label in labels]
    
    mapped_labels = {}
    
    for i, label in enumerate(labels):
        mapped_labels[label] = i
    
    return mapped_labels, labels

def load_data_sets():
    
    entities = construct_global_docMap("../data/re3d-master/*/entities_cleaned_sorted.json")
    
    df_word_weights, train_data, test_data = map_all_entities(entities,"../data/re3d-master/*/documents.json")
    class_weights = (1 - (df_word_weights["ner_tags"].value_counts().sort_index() / len(df_word_weights))).values
    class_weights = torch.from_numpy(class_weights).float().to("cuda")

    return (train_data, test_data)

def tokenize_and_align_labels(examples, tokenizer, mapped_labels):
    tokenized_inputs = tokenizer(examples["text"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(mapped_labels[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def tokenize_labels_old(examples, tokenizer, mapped_labels):
    tokenized_inputs = tokenizer(list(examples["text"]), truncation = True, is_split_into_words = True, max_length = 512)
    
    label_all_tokens = False
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(mapped_labels[label[word_idx]])
            else:
                label_ids.append(mapped_labels[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
        
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    '''
    Function for computing evaluation metrics.
    '''
    predictions, labels = p    
    predictions = np.argmax(predictions, axis=2)
    _ , label_list = load_labels('../resources/labels.txt')

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
                                                                   
def main ():
    login(token = write_token)
    logger.info("Prepping Data")
    #Loading and tokenizing datasets
    train_data, test_data = load_data_sets()
    mapped_labels, labels = load_labels("../resources/labels.txt")
    
    #tokenizer
    logger.info("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased", token=access_token)
    tokenizer.pad_token_id = 0

    train_tokenized_dataset = train_data.map(tokenize_labels_old, batched=True, fn_kwargs={"tokenizer": tokenizer, "mapped_labels": mapped_labels})
    test_tokenized_dataset = test_data.map(tokenize_labels_old, batched=True, fn_kwargs={"tokenizer": tokenizer, "mapped_labels": mapped_labels})
 
    #data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    #model
    logger.info("Loading model....")
    model = AutoModelForTokenClassification.from_pretrained("distilbert-base-multilingual-cased"
            , num_labels = len(labels)
            , token=access_token
            )
            #, load_in_8bit = True
            #, torch_dtype=torch.float32

    #if tokenizer.pad_token is None:
    #    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #    model.resize_token_embeddings(len(tokenizer))

    ### define training args here
    ### we should experiment with these hyperparameters.
    ### Leaning rates from bert documentation = (among 5e-5, 4e-5, 3e-5, and 2e-5)
    logger.info("Setting training args....")
    training_args = TrainingArguments(
        output_dir="../models",
        evaluation_strategy = "epoch",
        learning_rate = 2e-5,
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 32,
        num_train_epochs = 10,
        weight_decay = 1e-5,
        logging_dir = "../logging",
        logging_steps = 10
        
    )
    
    ### define trainer here
    logger.info("Defining Trainer...")
    trainer = Trainer (
        model,
        training_args,
        train_dataset = train_tokenized_dataset,
        eval_dataset = test_tokenized_dataset,
        data_collator = data_collator,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics   
    )
    
    #start training model
    logger.info("STARTING TRAINING OF NER_MODEL")
    trainer.train()
    trainer.push_to_hub()
    

if __name__ == '__main__':
    main()
    

