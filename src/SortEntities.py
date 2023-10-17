import pandas as pd
import json 

def sort_entities_assending(path):
    dict_temp = {}
    # Load the JSON data from the file
    with open (path, 'r', encoding = "utf-8") as json_file:
        for line in json_file:
            record = json.loads(line)
            id = record['_id']
            dict_temp[id] = record
        
    sorted_data = dict(sorted(dict_temp.items(), key=lambda item: (item[1]['documentId'], item[1]['begin'])))

    
    with open('../data/re3d-master/Wikipedia/entities_cleaned_sorted.json', 'w', encoding='utf-8') as f:
        for key in sorted_data:
            json.dump(sorted_data[key], f)
            f.write("\n")
        f.close

def remove_misleading_words(path):
    black_list = ["He", "he", "I", "We", "we", "Her", "her", "us", ]
    dict_temp = {}
    with open (path, 'r', encoding = "utf-8") as json_file:
        for line in json_file:
            record = json.loads(line)
            id = record['_id']
            if(record['value'] not in black_list): 
                dict_temp[id] = record
        
    sorted_data = dict(sorted(dict_temp.items(), key=lambda item: (item[1]['documentId'], item[1]['begin'])))

    
    with open('../data/re3d-master/Australian Department of Foreign Affairs/entities_cleaned_sorted_and_filtered.json', 'w', encoding='utf-8') as f:
        for key in sorted_data:
            json.dump(sorted_data[key], f)
            f.write("\n")
        f.close

if __name__ == '__main__':
    remove_misleading_words("../data/re3d-master/Australian Department of Foreign Affairs/entities_cleaned_sorted.json")