import json 
import glob

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

def construct_global_docMap(dirPath):
    dict = {}
    for path in glob.glob(dirPath):
        temp = construct_docMap(path)
        dict.update(temp)
    return dict