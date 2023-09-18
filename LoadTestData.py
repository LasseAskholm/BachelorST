import pandas as pd
import json 
import DuplicateCheck
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
            break


    df = pd.DataFrame({"Words": words, "Label": tags})
    return df

#def find_all_entities_files(dirPath):
    #for path in glob.glob(dirPath):
        #DuplicateCheck.clean_entities(path)
        #print(len(DuplicateCheck.check_for_duplicates(path)))

if __name__ == '__main__':
    dict = construct_docMap("data/re3d-master/US State Department/entities.json")
    #df = map_entities(dict, "data/re3d-master/US State Department/documents.json")
    #print(df.to_string())

