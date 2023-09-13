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

def map_entities(dict,path):
    tags = []
    words = []
    document_ids = []
    o_tags_idx = []

    with open (path, 'r', encoding = "utf-8") as json_file:
        for line in json_file:
            document = json.loads(line)
            entities = dict[document["_id"]]
            text = document['text']
            masked_text= document['text']
            for k,v in entities.items():
                entity = text[v["begin"]:v["end"]]
                for i,word in enumerate (entity.split(" ")):
                    if(i == 0):
                        words.append(word)
                        tags.append(f"B-{v['type']}")
                    else:
                        words.append(word)
                        tags.append(f"I-{v['type']}")
                masked_text = masked_text[:v["begin"]] + "".join(["*" for _ in range(len(entity))]) + masked_text[v["end"]:]
            begin_idx = 0
            for word in masked_text.split(" "):
                use_word = False
                for char in word:
                    if char != '*':
                        use_word = True
                        break
                if use_word:
                    o_tags_idx.append((begin_idx,begin_idx+len(word)+1))
                begin_idx += len(word)+1
            print(o_tags_idx)

            current_idx_in_text = 0
            current_o_tags_idx = 0
            for word in text.split(" "):
                for k,v in entities.items():
                    if current_idx_in_text == v["begin"]:
                        entity = text[v["begin"]:v["end"]]
                        for i, secquence in enumerate(entity.split (" ")):
                            if (i == 0):
                                words.append(secquence)
                                tags.append(f"B-{v['type']}")
                            else:
                                words.append(secquence)
                                tags.append(f"I-{v['type']}")
                        current_idx_in_text+=v["begin"]+1
                words.append(word)
                tags.append("O")
                current_idx_in_text = o_tags_idx[current_o_tags_idx][1]+1
                current_o_tags_idx+=1


    df = pd.DataFrame({"Words": words, "Label": tags})
    return df


if __name__ == '__main__':
    dict = construct_docMap("data/re3d-master/US State Department/entities.json")
    df = map_entities(dict, "data/re3d-master/US State Department/documents.json")
    #print(df.to_string())



