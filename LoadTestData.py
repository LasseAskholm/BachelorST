import pandas as pd
import json 

def json_to_ner_dataset(json_data):
    # Initialize empty lists to store tokens and labels
    tokens = []
    labels = []
    document_ids = []
    
    for record in json_data:
        # Extract relevant fields from the JSON record
        begin = record["begin"]
        end = record["end"]
        entity_type = record["type"]
        value = record["value"]
        document_id = record['documentId']

        value_tokens = value.split()
        
        token_position = 0

        for token in value_tokens:
            if token_position == 0:
                labels.append(f"B-{entity_type}")
            else:
                labels.append(f"I-{entity_type}")

            tokens.append(token)
            document_ids.append(document_id)
            token_position += 1

    df = pd.DataFrame({"DocumentID":document_ids,"Token": tokens, "Label": labels})

    return df

def parse_json(path):
    parsed_json=[]

    with open (path, 'r') as json_file:
        for line in json_file:
            parsed_record = json.loads(line)
            parsed_json.append(parsed_record)
    
    return parsed_json

new_data = parse_json('C:/Git/Bachelores/BachelorST/data/re3d-master/US State Department/entities.json') 
df = json_to_ner_dataset(new_data)


def construct_docMap(path):
    dict = {}
    with open (path, 'r') as json_file:
        for line in json_file:
            record = json.loads(line)
            documentid = record['_id'].split('-')[0]
            if documentid in dict:
                dict[documentid][record['_id']] = record
            else:
                dict[documentid] = {record['_id']: record}
    return dict

def map_entities(dict,path):
    tags = []
    words = []
    document_ids = []
    
    with open (path, 'r') as json_file:
        for line in json_file:
            document = json.loads(line)
            entities = dict[document["_id"]]
            text = document['text']
            for k,v in entities.items():
                entity = text[v["begin"]:v["end"]]
                for i,word in enumerate (entity.split(" ")):
                    if(i == 0):
                        words.append(word)
                        tags.append(f"B-{v['type']}")
                    else:
                        words.append(word)
                        tags.append(f"I-{v['type']}")

            
    df = pd.DataFrame({"Words": words, "Label": tags})
    return df   

            
if __name__ == '__main__':            
    dict = construct_docMap("C:/Git/Bachelores/BachelorST/data/re3d-master/US State Department/entities.json")
    df = map_entities(dict, "C:/Git/Bachelores/BachelorST/data/re3d-master/US State Department/documents.json")
    print(df.to_string())
    
    

    
   
