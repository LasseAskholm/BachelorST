import pandas as pd
import json 
import DuplicateCheck
import glob
from datasets import Dataset

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

def map_entities_old(dict,path):
    print ("mapping entities in:" , path)
    tags = []
    words = []
    document_ids = []
    o_tags_idx = []

    with open (path, 'r', encoding = "utf-8") as json_file:
        for line in json_file:
            document = json.loads(line)
            if document['_id'] in dict:
                entities = dict[document['_id']]
                text = document['text']
                masked_text = document['text']
                for k,v in entities.items():
                    entity = text[v["begin"]:v["end"]]
                    masked_text = masked_text[:v["begin"]] + "".join(["*" for _ in range(len(entity))]) + masked_text[v["end"]:]
                begin_idx = 0
                for word in masked_text.split(" "):
                    use_word = True
                    for char in word:
                        if char == '*':
                            use_word = False
                            break
                    if use_word:
                        o_tags_idx.append((begin_idx,begin_idx+len(word)+1))    
                    begin_idx += len(word)+1

                current_o_tags_idx = 0
                i = 0
                while (i< len(text)):
                    entity_found = False
                    for k,v in entities.items():
                        if i == v["begin"]:
                            entity_found = True
                            entity = text[v["begin"]:v["end"]]
                            for j, secquence in enumerate(entity.split (" ")):
                                if (j == 0):
                                    words.append(secquence)
                                    tags.append(f"B-{v['type']}")
                                else:
                                    words.append(secquence)
                                    tags.append(f"I-{v['type']}")
                        
                            i += 1
                    if not entity_found:
                        if (i < o_tags_idx[current_o_tags_idx][0]):
                            i += 1
                            continue
                        word = text[o_tags_idx[current_o_tags_idx][0]:o_tags_idx[current_o_tags_idx][1]]

                        words.append(word)
                        tags.append("O")
                        if current_o_tags_idx == len(o_tags_idx)-1:
                            break
                        else:
                            i += 1
                            current_o_tags_idx+=1

    df = pd.DataFrame({"text": words, "labels": tags})
    return df

def map_entities(entities_dict, dirPath):
    # Initialize lists to store words and labels
    words = []
    labels = []

    # Initialize list to store words and labels in sentences
    words_in_sentence = []
    labels_in_sentence = []
    for document_path in glob.glob(dirPath):
        # Read the document JSON
        with open(document_path, 'r') as doc_file:
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
                        labels.append(f'B-{entity_type}')  # Beginning of an entity
                    else:
                        words.append(word)
                        labels.append(f'I-{entity_type}')  # Inside an entity
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
    return df_word, df_sentence
        
def construct_global_docMap(dirPath):
    dict = {}
    for path in glob.glob(dirPath):
        temp = construct_docMap(path)
        dict.update(temp)
    return dict

def map_all_entities(dict,dirPath):
    df_word_weights, df = map_entities(dict,dirPath)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    return (df_word_weights, train_dataset,test_dataset)


if __name__ == '__main__':
    dict = construct_global_docMap("../data/re3d-master/*/entities.json")
    train,test = map_all_entities(dict, "../data/re3d-master/*/documents.json")
   

    #print(df.to_string())
    
    #find_all_entities_files("data/re3d-master/*/entities.json")