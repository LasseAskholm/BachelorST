import requests
import json
import pandas as pd

API_URL = "https://api-inference.huggingface.co/models/LazzeKappa/BERT_B08"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json={"inputs": payload, "options":{"wait_for_model":True}})
	return response.json()

def getValData():
    path = "../../data/selv-labeled-data/ValData/ValGEN.txt"
    with open (path, "r", encoding='utf-8') as f:
         input = f.read().replace('\n', '')
    return input
    
def jsonDump_pred(path):
    with open (path,'w') as f:
        json.dump(output, f, indent=4)
    
def getPredictions(data):
    words = []
    labels = []
    predictions = {}
    rawPreds = query(data)
    entityidx = 0
    entityOidx = 0
    i = 0
    text_idx = 0
    o_tag_end = 0
    print("LENGTH:"+ str(len(rawPreds)))
    while i < len(data):
        
        
        if entityidx == len(rawPreds):
            remaining = data[i:]
            for word in remaining.split(" "):
                predictions[word] = "O"
            break
            
        elif entityidx < len(rawPreds):            
            entity = rawPreds[entityidx]
            
        #print(entity['word'])
        next_paragraph = data [text_idx:entity['start']]
        
        #if we are in an entity we get the predicted label
        if i<=entity['end'] and i>= entity['start']:
            entity_text = entity['word'].split(" ")
            #print(entity['word'])
            for j, word in enumerate(entity_text):
                if j == 0:
                    words.append(word)
                    labels.append("B-"+str(entity['entity_group']))
                else:
                    words.append(word)
                    labels.append("I-"+str(entity['entity_group']))
            i = entity['end']
            entityidx+=1 
            continue
        #Else all the words in the next paragraph leading up to the next entity gets O tagged
        for word in next_paragraph.split(" "):
            if word == "":
                continue
            words.append(word)
            labels.append("O")
        text_idx = entity['end']+1
        i = entity['start']
        entityOidx += 1
         
    return pd.DataFrame({"words": words, "ner_tags": labels})    

    

