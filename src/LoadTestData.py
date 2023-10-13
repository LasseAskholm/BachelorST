import pandas as pd
import re
import json 
import glob
import itertools
from datasets import Dataset

self_labeled_data = "../data/selfLabeledWithReducedLabels.conll"

skippedLabelsList = ["DocumentReference", "Nationality", "Quantity", "CommsIdentifier", "Coordinate", "Frequency"]

def construct_docMap(path):
    dict = {}
    with open (path, 'r', encoding = "utf-8") as json_file:
        for line in json_file:
            record = json.loads(line)
            documentid = record['_id'].split('-')[0]
            if documentid in dict:
                dict[documentid][record['_id']] = record
            else:
                dict[documentid] = {record['_id']: record}
    return dict

def map_entities(entities_dict, dirPath, reducedLabels):
    # Initialize lists to store words and labels
    words = []
    labels = []
    # Initialize list to store words and labels in sentences
    words_in_sentence = []
    labels_in_sentence = []
    for document_path in glob.glob(dirPath):
        # Read the document JSON
        with open(document_path, 'r', encoding = "utf-8") as doc_file:
            document_data = [json.loads(line) for line in doc_file]

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
                            if entity_type in skippedLabelsList:
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
                            if entity_type in skippedLabelsList:
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
    
    #Format the list to be per sentence. 
    word_sentence_index = []
    word_index_counter = 0
    words_to_skip = ["Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec."]
    for word in words:
        if word[-1] == '.' or word[-1] == "?" or word[-1] == "!":
            if not words_to_skip.__contains__(word):
                word_sentence_index.append(word_index_counter)
        word_index_counter += 1
    
    for x in range (len(word_sentence_index)):
        if x == 0:
            words_in_sentence.append(words[:word_sentence_index[x] + 1])
            labels_in_sentence.append(labels[:word_sentence_index[x] + 1])
        else:
            words_in_sentence.append(words[word_sentence_index[x-1] + 1:word_sentence_index[x] + 1])
            labels_in_sentence.append(labels[word_sentence_index[x-1] + 1:word_sentence_index[x] + 1])

    df_word = pd.DataFrame({"text": words, "ner_tags": labels})
    df_sentence = pd.DataFrame({"text": words_in_sentence, "ner_tags": labels_in_sentence})
    df_self_labeled_data = get_tokens_and_ner_tags(self_labeled_data)
    df_merged = pd.concat([df_sentence, df_self_labeled_data], ignore_index=True, sort=False)
    return df_word, df_merged, labels_in_sentence
        
def construct_global_docMap(dirPath):
    dict = {}
    for path in glob.glob(dirPath):
        temp = construct_docMap(path)
        dict.update(temp)
    return dict

def map_all_entities(dict,dirPath):
    df_word_weights, df, _ = map_entities(dict,dirPath, True)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    return (df_word_weights, train_dataset, test_dataset)

def generate_df_from_additional_data():
    data_set = []
    path = "../data/additional.json"
    file = open (path)
    obj = json.load(file)
    for i in range (len(obj)):
        data_point = {}
        response = []
        contexts = []
        json_obj = obj[i]
        contexts.append(json_obj["text"])
        for label in json_obj["label"]:
            response.append(label["text"]+ " - " + label["labels"][0])
        data_point["context"] = contexts
        data_point["answers"] = response
        data_set.append(data_point)
    
    return data_set


def generate_prompt_data(entities_dict, dirPath):
    # Initialize lists to store the promts
    
    prompt_data_set = []    
    
    for document_path in glob.glob(dirPath):
        # Read the document JSON
        with open(document_path, 'r', encoding = "utf-8") as doc_file:
            document_data = [json.loads(line) for line in doc_file]

        # Iterate through the document data
        for doc_entry in document_data:
            data_point = {}
            response = []
            contexts = []
            contexts.append( doc_entry['text'])  # Get the full text
            
            # Get the entities for this document entry from the provided dictionary
            if doc_entry['_id'] in entities_dict:
                entities_data = entities_dict[doc_entry['_id']]
            else:
                continue

            for entity_id, entity_info in entities_data.items():
                response.append({"WordLabel": entity_info['value'] + " - " + entity_info['type'], "Placement": entity_info['end']})
            data_point["context"] = contexts
            data_point["answers"] = response
            prompt_data_set.append(data_point)

    return pd.DataFrame(prompt_data_set, columns=['context','answers'])


def generate_prompt_data_sentence(entities_dict, dirPath):
    # Initialize list to store the promts
    prompt_data_set = []

    pattern = "(?<=[.!?])\s+(?![A-Z][a-z]+\.\s+[A-Z]|[a-z]|[1-9])"

    for document_path in glob.glob(dirPath):
        # Read the document JSON
        with open(document_path, 'r', encoding = "utf-8") as doc_file:
            document_data = [json.loads(line) for line in doc_file]

        # Iterate through the document data
        for doc_entry in document_data:
            text_idx = 0
            doc_text = doc_entry['text']
            
            # Get the entities for this sentence entry from the provided dictionary
            if doc_entry['_id'] in entities_dict:
                entities_data = entities_dict[doc_entry['_id']]
            else:
                continue
            
            for sentence in re.split(pattern, doc_text):
                if(sentence == ""):
                    continue

                data_point = {}
                response = []
                contexts = []
                contexts.append(sentence)

                #Find entities in current sentence:
                for entity_id, entity_info in entities_data.items():
                    if(entity_info['begin']>= text_idx and entity_info['end']<= text_idx+len(sentence)+1):
                        response.append(entity_info['value'] + " - " + entity_info['type'])
                text_idx += len(sentence)+1
                data_point["context"] = contexts
                data_point["answers"] = response
                prompt_data_set.append(data_point)

    df = pd.DataFrame(prompt_data_set, columns=['context',''])
    df_self_labeled_data = get_tokens_and_ner_tags(self_labeled_data)
    df_merged = pd.concat([df, df_self_labeled_data], ignore_index=True, sort=False)
    print(df_merged)
    exit()
    
    return pd.DataFrame(prompt_data_set, columns=['context','answers'])
    
def get_tokens_and_ner_tags(filename):
    '''
    Function for loading tokens and ner tags.
    '''
    with open(filename, 'r', encoding="utf8") as f:
        lines = f.readlines()
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
        tokens = [[x.split('\t')[0] for x in y] for y in split_list]
        entities = [[x.split('\t')[1][:-1] for x in y] for y in split_list]

    return pd.DataFrame({'text': tokens, 'ner_tags': entities})

if __name__ == '__main__':
    dict = construct_global_docMap("../data/re3d-master/*/entities.json")
    train,test = map_all_entities(dict, "../data/re3d-master/*/documents.json")
   