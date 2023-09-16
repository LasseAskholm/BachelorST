import pandas as pd
import json 

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
                        dict_with_duplicates[record["_id"]] = {"first entry id": key, "first entry": dict[key], "offending id": record["_id"], "begin": record['begin'], "end": record['end']}  
            dict[id] = {"documentId": record['documentId'], "begin": record['begin'], "end": record['end']}
    print(len(dict_with_duplicates))

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