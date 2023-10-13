import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(0) #Limits to use only one GPU

import numpy as np
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from LoadTestData import construct_global_docMap, map_all_entities, generate_df_from_additional_data
import torch
from huggingface_hub import login
from loguru import logger
import evaluate

access_token = "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"
write_token = "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"
seqeval = evaluate.load("seqeval")

entities_file_ptah = "../data/re3d-master/*/entities_cleaned_sorted_and_filtered.json"
documens_file_path = "../data/re3d-master/*/documents.json"
label_file_path = '../resources/labelsReduced.txt'

model_name = "distilbert-base-multilingual-cased"

label2id = {"O": 0, 
            "B-Organisation": 1, 
            "I-Organisation": 2, 
            "B-Nationality": 3, 
            "I-Nationality": 4, 
            "B-Person": 5,
            "I-Person": 6, 
            "B-DocumentReference": 7,
            "I-DocumentReference": 8,
            "B-Location": 9,
            "I-Location": 10,
            "B-Money": 11,
            "I-Money": 12,
            "B-Vehicle": 13,
            "I-Vehicle": 14,
            "B-Temporal": 15,
            "I-Temporal": 16,
            "B-Weapon": 17,
            "I-Weapon": 18,
            "B-Quantity": 19,
            "I-Quantity": 20,
            "B-CommsIdentifier": 21,
            "B-MilitaryPlatform": 22,
            "I-MilitaryPlatform": 23,
            "B-Coordinate": 24,
            "I-Coordinate": 25,
            "B-Frequency": 26,
            "I-Frequency": 27}
id2label = {0 : "O", 
            1 : "B-Organisation", 
            2 : "I-Organisation", 
            3 : "B-Nationality", 
            4 : "I-Nationality", 
            5 : "B-Person",
            6 : "I-Person", 
            7 : "B-DocumentReference",
            8 : "I-DocumentReference",
            9 : "B-Location",
            10 : "I-Location",
            11 : "B-Money",
            12 : "I-Money",
            13 : "B-Vehicle",
            14 : "I-Vehicle",
            15 : "B-Temporal",
            16 : "I-Temporal",
            17 : "B-Weapon",
            18 : "I-Weapon",
            19 : "B-Quantity",
            20 : "I-Quantity",
            21 : "B-CommsIdentifier",
            22 : "B-MilitaryPlatform",
            23 : "I-MilitaryPlatform",
            24 : "B-Coordinate",
            25 : "I-Coordinate",
            26 : "B-Frequency",
            27 : "I-Frequency"}

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
    entities = construct_global_docMap(entities_file_ptah)
    
    df_word_weights, train_data, test_data = map_all_entities(entities,
                                                              documens_file_path)
    
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
    _ , label_list = load_labels(label_file_path)

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
    mapped_labels, labels = load_labels("../resources/labelsReduced.txt")
    
    #tokenizer
    logger.info("Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              token=access_token)
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
    model = AutoModelForTokenClassification.from_pretrained("distilbert-base-multilingual-cased", 
                                                            num_labels = len(labels), 
                                                            token=access_token,
            )


    # define training args here
    # Leaning rates from bert documentation = (among 5e-5, 4e-5, 3e-5, and 2e-5)
    logger.info("Setting training args")
    training_args = TrainingArguments(
        output_dir="../models",
        evaluation_strategy = "epoch",
        learning_rate = 2e-5,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 16,
        num_train_epochs = 10,
        weight_decay = 1e-5,
        logging_dir = "../logging",
        logging_steps = 10
        
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
    

