import pandas as pd
import re
import json 
import glob
from CommonVariables import COMMON_SKIPPED_LABELES

def generate_prompt_data_entire_text(entities_dict, dirPath):
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


def generate_prompt_data_sentence(entities_dict, dirPath, self_labeled_data_path, reducedLabeles):
    '''
    Function to generate the promt sentence for llama2.
    '''
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
                for _, entity_info in entities_data.items():
                    if(entity_info['begin']>= text_idx and entity_info['end']<= text_idx+len(sentence)+1):
                        if not reducedLabeles:
                            response.append(entity_info['value']+ " - " + entity_info['type'])
                        else:
                            if entity_info['type'] in COMMON_SKIPPED_LABELES:
                                continue
                            elif entity_info['type'] == 'Vehicle':
                                response.append(entity_info['value']+ " - " + "MilitaryPlatform")
                            else:
                                response.append(entity_info['value']+ " - " + entity_info['type'])
                        response.append(entity_info['value'] + " - " + entity_info['type'])

                text_idx += len(sentence)+1
                data_point["context"] = contexts
                data_point["answers"] = response
                prompt_data_set.append(data_point)

    df = pd.DataFrame(prompt_data_set, columns=['context','answers'])
    df_self_labeled_data = generate_df_from_selflabeled_data(self_labeled_data_path, reducedLabeles)
    df_merged = pd.concat([df, df_self_labeled_data], ignore_index=True, sort=False)
    return df_merged

def generate_df_from_selflabeled_data(self_labeled_data_path, reducedLabeles):
    data_set = []
    path = self_labeled_data_path
    file = open (path)
    obj = json.load(file)
    print(len(obj))
    for i in range (len(obj)):
        labels_present = True
        data_point = {}
        response = []
        contexts = []
        json_obj = obj[i]

        if json_obj.get("label") == None:
            labels_present = False
    
        contexts.append(json_obj["text"])
        
        if labels_present:
            label_map = json_obj["label"]
            for label in label_map:
                type = label["labels"][0]
                if not reducedLabeles:
                    response.append(label["text"]+ " - " + label["labels"][0])
                else:
                    if type in COMMON_SKIPPED_LABELES:
                        continue
                    elif type == 'Vehicle':
                        response.append(label["text"]+ " - " + "MilitaryPlatform")
                    else:
                        response.append(label["text"]+ " - " + label["labels"][0])
        data_point["context"] = contexts
        data_point["answers"] = response
        data_set.append(data_point)
    
    return pd.DataFrame(data_set, columns=['context','answers'])