import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0) #Limits to use only one GPU

import numpy as np
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from utils.BERTDataLoader import fetch_train_test_data
from utils.CommonDataLoader import construct_global_docMap
import torch
from huggingface_hub import login
from loguru import logger
import evaluate
from utils.CommonVariables import (
    COMMON_HUGGINGFACE_ACCESS_TOKEN, 
    COMMON_HUGGINGFACE_WRITE_TOKEN, 
    COMMON_DSTL_DOCUMENTS, 
    COMMON_BERT_OUTPUT_DIR,
    COMMON_BERT_LABELS, 
    COMMON_DSTL_ENTITIES,
    COMMON_BERT_MODEL_NAME,
    COMMON_BERT_LABEL2ID,
    COMMON_BERT_ID2LABEL,
    COMMON_BERT_REDUCE_LABELS,
    COMMON_BERT_LEARNING_RATE,
    COMMON_BERT_TRAIN_BATCH_SIZE,
    COMMON_BERT_EVAL_BATCH_SIZE,
    COMMON_BERT_EPOCHS,
    COMMON_BERT_WEIGHT_DECAY,
    COMMON_BERT_LOGGING_DIR,
    COMMON_BERT_LOGGING_STEPS
    )

seqeval = evaluate.load("seqeval")

def load_labels(labels_path):
    '''
    Function to load the labels.
    '''
    with open(labels_path, encoding='utf-8') as file:
        labels = file.readlines()
        
    labels = [label if label[-1:] != '\n' else label[:-1] for label in labels]
    
    mapped_labels = {}
    
    for i, label in enumerate(labels):
        mapped_labels[label] = i
    
    return mapped_labels, labels

def load_data_sets():
    '''
    Function to load the test data and adjust importance for each label.
    '''
    entities = construct_global_docMap(COMMON_DSTL_ENTITIES)
    
    df_word_weights, train_data, test_data = fetch_train_test_data(entities,
                                                              COMMON_DSTL_DOCUMENTS, 
                                                              COMMON_BERT_REDUCE_LABELS)
    
    class_weights = (1 - (df_word_weights["ner_tags"].value_counts().sort_index() / len(df_word_weights))).values
    class_weights = torch.from_numpy(class_weights).float().to("cuda")

    return (train_data, test_data)

def tokenize_and_align_labels(examples, tokenizer, mapped_labels):
    '''
    Function To realing labels after subtokenization.
    '''
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
    _ , label_list = load_labels(COMMON_BERT_LABELS)

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
    login(token = COMMON_HUGGINGFACE_WRITE_TOKEN)
    logger.info("Prepping Data")

    #Loading and tokenizing datasets
    train_data, test_data = load_data_sets()
    mapped_labels, labels = load_labels(COMMON_BERT_LABELS)
    
    #tokenizer
    logger.info("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(COMMON_BERT_MODEL_NAME, 
                                              token=COMMON_HUGGINGFACE_ACCESS_TOKEN)
    tokenizer.pad_token_id = 0

    logger.info("Creating tokenized dataset")
    train_tokenized_dataset = train_data.map(tokenize_and_align_labels, 
                                             batched=True, 
                                             fn_kwargs={"tokenizer": tokenizer, 
                                                        "mapped_labels": mapped_labels})
    
    test_tokenized_dataset = test_data.map(tokenize_and_align_labels, 
                                           batched=True, 
                                           fn_kwargs={"tokenizer": tokenizer, 
                                                      "mapped_labels": mapped_labels})
 
    #data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    #model
    logger.info("Loading model")
    model = AutoModelForTokenClassification.from_pretrained(COMMON_BERT_MODEL_NAME, 
                                                            num_labels = len(labels), 
                                                            token=COMMON_HUGGINGFACE_ACCESS_TOKEN,
                                                            id2label = COMMON_BERT_ID2LABEL,
                                                            label2id = COMMON_BERT_LABEL2ID,
            )


    # define training args here
    # Leaning rates from bert documentation = (among 5e-5, 4e-5, 3e-5, and 2e-5)
    logger.info("Setting training args")
    training_args = TrainingArguments(
        output_dir=COMMON_BERT_OUTPUT_DIR,
        evaluation_strategy = "epoch",
        learning_rate = COMMON_BERT_LEARNING_RATE,
        per_device_train_batch_size = COMMON_BERT_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size = COMMON_BERT_EVAL_BATCH_SIZE,
        num_train_epochs = COMMON_BERT_EPOCHS,
        weight_decay = COMMON_BERT_WEIGHT_DECAY,
        logging_dir = COMMON_BERT_LOGGING_DIR,
        logging_steps = COMMON_BERT_LOGGING_STEPS,
        label_smoothing_factor = 0.3
        
    )
    
    ### define trainer here
    logger.info("Defining Trainer")
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
    
    logger.info("Pushing trained model to hub")
    trainer.push_to_hub()
    

if __name__ == '__main__':
    main()
    

