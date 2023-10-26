import pandas as pd
import json 

base_file_name = "entities.json"

def clean_entities(path):
    dict = check_for_duplicates(path + base_file_name)
    extract_longest_from_duplicates(dict, path)

def check_for_duplicates(path):
    dict = {}
    dict_with_duplicates = {}
    with open (path, 'r', encoding = "utf-8") as json_file:
        for line in json_file:
            record = json.loads(line)
            id = record['_id']
            for key in dict:
                if(dict[key]["documentId"] == record['documentId']):
                    if(record['begin'] in range(dict[key]['begin'], dict[key]['end'] + 1) or record['end'] in range(dict[key]['begin'], dict[key]['end'] + 1)):    
                        dict_with_duplicates[record["_id"]] = {"first entry id": key, "first entry": dict[key], "offending id": record["_id"], "begin": record['begin'], "end": record['end'], "confidence": record['confidence']}  
            dict[id] = {"documentId": record['documentId'], "begin": record['begin'], "end": record['end'], "confidence": record['confidence']}
    return dict_with_duplicates

def extract_longest_from_duplicates(dict_with_duplicates, path):
    uncleaned_dict = {}
    cleaned_dict = {}
    keys_to_remove = []

    for key in dict_with_duplicates:
        if(dict_with_duplicates[key]['first entry']['confidence'] == 0):
            keys_to_remove.append(dict_with_duplicates[key]['first entry id'])
        else:
            keys_to_remove.append(key)
    keys_to_remove = list(dict.fromkeys(keys_to_remove))

    with open(path + base_file_name, 'r', encoding='utf-8') as json_file:
        for line in json_file:
            record = json.loads(line)
            id = record['_id']
            uncleaned_dict[id] = record
    
    for key in keys_to_remove:
        uncleaned_dict.pop(key)
        cleaned_dict = uncleaned_dict
    
    newPath = path + '/entities_cleaned.json'

    with open(newPath, 'w', encoding='utf-8') as f:
        for key in cleaned_dict:
            json.dump(cleaned_dict[key], f)
            f.write("\n")
        f.close
            


#Dup     Total   Left
#27     | 70    | 43
#167    | 913   | 746
#366    | 1439  | 1283
#29     | 110   | 81
#271    | 945   | 674
#198    | 720   | 522
#168    | 641   | 473

#TOTAL
#1226   | 4838  | 3613