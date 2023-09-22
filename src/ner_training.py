import os
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from LoadTestData import construct_global_docMap, map_all_entities

def load_labels(labels_path):
    
    with open(labels_path, encoding='utf-8') as file:
        labels = file.readlines()
        
    labels = [label if label[-1:] != '\n' else label[:-1] for label in labels]
    
    mapped_labels = {}
    
    for i, labels in enumerate(labels):
        mapped_labels[label] = i
    
    return mapped_labels, labels

def load_data_sets():
    
    entities = construct_global_docMap("../data/re3d-master/*/entities.json")
    
    train_data, test_data = map_all_entities(entities,"../data/re3d-master/*/documents.json")
    
    return (train_data, test_data)
    
def tokenize_labels(examples, tokenizer, mapped_labels):
    tokenized_inputs = tokenizer(list(examples["Words"]),truncation = true, is_split_into_words = true)
    label_all_tokens = False
    labels = []
    
    for i , label in enumerate (examples["Label"]):
        word_ids  = tokenized_inputs.word_ids(batch_index = i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label [word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(mapped_labels[label[word_idx]])
            else:
                label_ids.append(mapped_labels[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)
        tokenized_inputs["Label"] = labels
        
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
    labels, mapped_labels = load_labels("../resources/labels.txt")
    #TODO: Verificer størelse på modellen med henrik eller noget
    #model
    model = AutoModelForTokenClassification.from_pretrained("meta-llama/Llama-2-70b-hf",num_labels = len(labels))
    #tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
    #data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    train_data, test_data = load_data_sets()
    
    train_tokenized_dataset = train_data.map(tokenize_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "label_encoding_dict": mapped_labels})
    
    test_tokenized_dataset = test_data.map(tokenize_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "label_encoding_dict": mapped_labels})
    
    ### define training args here
    ### we should experiment with these hyperparameters.
    training_args = TrainingArguments(
        os.path.join(get_root_path(), 'models',"/outputDir"),
        evaluation_strategy = "epoch",
        save_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=30,
        weight_decay=1e-5,
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
    
    #trainer.train()
    

if __name__ == '__main__':
    main()
    

