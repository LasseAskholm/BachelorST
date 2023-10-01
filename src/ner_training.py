import os
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoModelForCausalLM, GPTQConfig
from transformers import DataCollatorForTokenClassification
from transformers import BitsAndBytesConfig
from LoadTestData import construct_global_docMap, map_all_entities
from SortEntities import sort_entities_assending
import torch
from huggingface_hub import login

access_token = "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"
write_token = "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"

def load_labels(labels_path):
    with open(labels_path, encoding='utf-8') as file:
        labels = file.readlines()
        
    labels = [label if label[-1:] != '\n' else label[:-1] for label in labels]
    
    mapped_labels = {}
    
    for i, label in enumerate(labels):
        mapped_labels[label] = i
    
    return mapped_labels, labels

def load_data_sets():
    
    entities = construct_global_docMap("../data/re3d-master/Australian Department of Foreign Affairs/entities_cleaned_sorted.json")
    
    train_data, test_data = map_all_entities(entities,"../data/re3d-master/Australian Department of Foreign Affairs/documents.json")

    return (train_data, test_data)
    
def tokenize_labels(examples, tokenizer, mapped_labels):
    tokenized_inputs = tokenizer(list(examples["text"]), truncation = True, is_split_into_words = True, max_length = 512)
    
    label_all_tokens = False
    labels = []
    for i, label in enumerate(examples["ner-tags"]):
        
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

    #labels = []
    #for i, label in enumerate (examples["labels"]):
        
    #    word_ids = tokenized_inputs.word_ids(batch_index = i)
    #    previous_word_idx = None
    #    label_ids = []
    #    for word_idx in word_ids:
    #        if word_idx is None:
    #             label_ids.append(-100)
    #        elif word_idx != previous_word_idx:
    #            label_ids.append(mapped_labels[label])
    #        else:
    #            label_ids.append(-100)
    #            # label_ids.append(-101)
    #        previous_word_idx = word_idx
    #    labels.append(label_ids)
    #tokenized_inputs["labels"] = labels

# {'text': 'The', 'labels': [-100, 1], 'input_ids': [1, 450], 'attention_mask': [1, 1]}
# {'text': 'Members', 'labels': [-100, 2, -101], 'input_ids': [1, 341, 13415], 'attention_mask': [1, 1, 1]}

    #return tokenized_inputs 

def compute_metrics(p):
    '''
    Function for computing evaluation metrics.
    '''
    predictions, labels = p
    print("METRICS!!")
    
    predictions = np.argmax(predictions, axis=2)
    #print(predictions)
    
    _ , label_list = load_labels('../resources/labels.txt')

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    print("LABELS;")
    print(true_labels)

    print("PREDICTIONS!")
    print(true_predictions)
    metric = load_metric("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

def main ():
    #sort_entities_assending("../data/re3d-master/Australian Department of Foreign Affairs/entities_cleaned.json")
    #exit()
    login(token = write_token)
    print("Prepping Data....")
    #Loading and tokenizing datasets
    train_data, test_data = load_data_sets()
    print(train_data["text"])
    mapped_labels, labels = load_labels("../resources/labels.txt")
    
    #tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token)

    train_tokenized_dataset = train_data.map(tokenize_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "mapped_labels": mapped_labels})
   # print(train_tokenized_dataset['Words'])
    #print("LABELS:")
    #print(train_tokenized_dataset[0])
    #exit()

    test_tokenized_dataset = test_data.map(tokenize_labels, batched=True, fn_kwargs={"tokenizer": tokenizer, "mapped_labels": mapped_labels})
    
    #data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    quantization_config  = BitsAndBytesConfig (
        load_in_4bit = True,
        bnb_4bit_compute_dtype = torch.bfloat16
    )

    # sentence = "The Members of the Security Council are "
    # text_seq = tokenizer.tokenize(sentence)
    # print(text_seq)
    # sq = tokenizer(sentence)
    # print(sq)
    # encoded = sq['input_ids']
    # print(encoded)
    # exit()

    #for i in range(20):
    #    print(train_tokenized_dataset[i])

    #exit()


    #model
    print("Loading model....")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",num_labels = len(labels), token=access_token, quantization_config = quantization_config)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    
    ### define training args here
    ### we should experiment with these hyperparameters.
    ### Leaning rates from bert documentation = (among 5e-5, 4e-5, 3e-5, and 2e-5)
    print("Setting training args....")
    training_args = TrainingArguments(
        output_dir="../models",
        evaluation_strategy = "epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=30,
        weight_decay=1e-5

    )
    
    ### define trainer here
    print ("Defining Trainer...")
    trainer = Trainer (
        model,
        training_args,
        train_dataset = train_tokenized_dataset,
        eval_dataset = test_tokenized_dataset,
        data_collator = data_collator,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics,
        
    )
    
    #start training model
    print ("STARTING TRAINING OF NER_MODEL")
    trainer.train()
    

if __name__ == '__main__':
    main()
    

