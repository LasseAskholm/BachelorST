import pandas as pd
import numpy as np
import json 
import glob
import itertools
from datasets import Dataset
from CommonVariables import COMMON_SKIPPED_LABELES, CONMON_BERT_SELF_LABELED_DATA

def fetch_train_test_data(dict, dirPath, reduceLabels):
    df_word_weights, df = map_entities(dict, dirPath, reduceLabels)

    df.reindex(np.random.permutation(df.index))

    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    return (df_word_weights, train_dataset, test_dataset)

def map_entities(entities_dict, dirPath, reducedLabels):
    '''
    Function to generate a word/label dataframe and sentence/label dataframe. 
    Includes both json and conll data. 
    '''
    # Initialize lists to store words and labels
    words = []
    labels = []
    
    for document_path in glob.glob(dirPath):
        # Read the document JSON
        with open(document_path, 'r', encoding = "utf-8") as doc_file:
            document_data = [json.loads(line) for line in doc_file]

        words_temp, labels_temp = construct_ner_tags_from_document(document_data, entities_dict, reducedLabels)    
        # Appends the temp dicts to the final dicts
        words += words_temp
        labels += labels_temp

    df_word = pd.DataFrame({"text": words, "ner_tags": labels})

    df_sentence = construct_sentence_from_words_dict(words, labels)
    df_self_labeled_data = load_data_from_conll(CONMON_BERT_SELF_LABELED_DATA)

    df_sentence_merged = pd.concat([df_sentence, df_self_labeled_data], ignore_index=True, sort=False)
    
    return df_word, df_sentence_merged

def construct_ner_tags_from_document(document_data, entities_dict, reducedLabels):
    '''
    Function to generate the BIO tagging on the text. 
    '''
    words = []
    labels = []

    # Iterate through the document data
    for doc_entry in document_data:

        text = doc_entry['text']  # Get the full text

        # Get the entities for this document entry from the provided dictionary
        if doc_entry['_id'] in entities_dict:
            entities_data = entities_dict[doc_entry['_id']]
        else:
            continue

        current_char_index = 0  # Track the current character index
        for entity_id, entity_info in entities_data.items():
            begin = entity_info['begin']
            end = entity_info['end']
            entity_type = entity_info['type']

            # Append words before the entity
            while current_char_index < begin:
                entity_text = text[current_char_index:begin]
                entity_words = entity_text.split()
                for i, word in enumerate(entity_words):
                    words.append(word)
                    labels.append('O')  # Outside an entity
                current_char_index = begin

            # Append the entity
            entity_text = text[begin:end]
            entity_words = entity_text.split()
            for i, word in enumerate(entity_words):
                if i == 0:
                    words.append(word)
                    if not reducedLabels:
                        labels.append(f'B-{entity_type}')  # Beginning of an entity
                    else:
                        if entity_type in COMMON_SKIPPED_LABELES:
                            labels.append('O')
                        elif entity_type == 'Vehicle':
                            labels.append(f'B-MilitaryPlatform')
                        else:
                            labels.append(f'B-{entity_type}')  # Beginning of an entity   
                else:
                    words.append(word)
                    if not reducedLabels:
                        labels.append(f'I-{entity_type}')  # Inside an entity
                    else:
                        if entity_type in COMMON_SKIPPED_LABELES:
                            labels.append('O')
                        elif entity_type == 'Vehicle':
                            labels.append(f'I-MilitaryPlatform')
                        else:
                            labels.append(f'I-{entity_type}')  # Beginning of an entity  
                    
            current_char_index = end

        # Append words after the last entity
        while current_char_index < len(text):
            entity_text = text[current_char_index:len(text)]
            entity_words = entity_text.split()
            for i, word in enumerate(entity_words):
                words.append(word)
                labels.append('O')  # Outside an entity
            current_char_index = len(text)

    return words, labels


def construct_sentence_from_words_dict(words_dict, labels_dict):
    '''
    Function to construct a sentence from the entire dict of words.
    '''
    # Initialize list to store words and labels in sentences
    words_in_sentence = []
    labels_in_sentence = []

    #Format the list to be per sentence. 
    word_sentence_index = []
    word_index_counter = 0
    words_to_skip = ["Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec."]
    for word in words_dict:
        if word[-1] == '.' or word[-1] == "?" or word[-1] == "!":
            if not words_to_skip.__contains__(word):
                word_sentence_index.append(word_index_counter)
        word_index_counter += 1
    
    for x in range (len(word_sentence_index)):
        if x == 0:
            words_in_sentence.append(words_dict[:word_sentence_index[x] + 1])
            labels_in_sentence.append(labels_dict[:word_sentence_index[x] + 1])
        else:
            words_in_sentence.append(words_dict[word_sentence_index[x-1] + 1:word_sentence_index[x] + 1])
            labels_in_sentence.append(labels_dict[word_sentence_index[x-1] + 1:word_sentence_index[x] + 1])
    
    return pd.DataFrame({"text": words_in_sentence, "ner_tags": labels_in_sentence})

   
def load_data_from_conll(filename):
    '''
    Function for loading tokens and ner tags from a conll file.
    '''
    with open(filename, 'r', encoding="utf8") as f:
        lines = f.readlines()
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
        tokens = [[x.split('\t')[0] for x in y] for y in split_list]
        entities = [[x.split('\t')[1][:-1] for x in y] for y in split_list]

    df = pd.DataFrame({'text': tokens, 'ner_tags': entities})
    return df