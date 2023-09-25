import os
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoModelForCausalLM, GPTQConfig
from transformers import DataCollatorForTokenClassification
from LoadTestData import construct_global_docMap, map_all_entities

def load_labels(labels_path):
    with open(labels_path, encoding='utf-8') as file:
        labels = file.readlines()
        
    labels = [label if label[-1:] != '\n' else label[:-1] for label in labels]
    
    mapped_labels = {}
    
    for i, label in enumerate(labels):
        mapped_labels[label] = i
    
    return mapped_labels, labels

def load_data_sets():
    
    entities = construct_global_docMap("../data/re3d-master/Australian Department of Foreign Affairs/entities_cleaned.json")
    
    train_data, test_data = map_all_entities(entities,"../data/re3d-master/Australian Department of Foreign Affairs/documents.json")
    
    return (train_data, test_data)
    
def tokenize_labels(examples, tokenizer, mapped_labels):
    tokenized_inputs = tokenizer(list(examples["Words"]), truncation = True)
    labels = []

    for i, label in enumerate (examples["Label"]):
        word_ids = tokenized_inputs.word_ids(batch_index = i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(mapped_labels[label])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    print(labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs 

def compute_metrics(p):
    '''
    Function for computing evaluation metrics.
    '''
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    label_list, _ = load_labels('../resources/labels.txt')
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    metric = load_metric("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

    
def main ():
    
    #Loading and tokenizing datasets
    train_data, test_data = load_data_sets()
    mapped_labels, labels = load_labels("../resources/labels.txt")
    
    #tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token)

    train_tokenized_dataset = train_data.map(tokenize_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "mapped_labels": mapped_labels})
    
    test_tokenized_dataset = test_data.map(tokenize_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "mapped_labels": mapped_labels})
    
    #data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    #Configure quantization
    #gptq_config = GPTQConfig(bits = 4 , dataset = train_tokenized_dataset, tokenizer = tokenizer)
    
    #model
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",num_labels = len(labels), token=access_token)
    
    ### define training args here
    ### we should experiment with these hyperparameters.
    ### Leaning rates from bert documentation = (among 5e-5, 4e-5, 3e-5, and 2e-5)
    training_args = TrainingArguments(
        output_dir="../models",
        evaluation_strategy = "epoch",
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=30,
        weight_decay=1e-5,
        load_best_model_at_end=True,
        push_to_hub=True
    )
    
    ### define trainer here
    
    trainer = Trainer (
        model,
        training_args,
        train_dataset = train_tonkenized_dataset,
        eval_dataset = test_tonkenized_dataset,
        data_collator = data_collator,
        tokenizer = tokenzier,
        compute_metrics = compute_metrics
        
    )
    

    #start training model
    
    trainer.train()
    

if __name__ == '__main__':
    main()
    

