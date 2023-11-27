import requests
import json
import pandas as pd

API_URL = "https://api-inference.huggingface.co/models/LazzeKappa/BERT_B07"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    print(headers)
    response = requests.post(API_URL, headers=headers, json={"inputs": payload, "options":{"wait_for_model":True}})
    return response.json()

def getValData(path):
    with open (path, "r", encoding='utf-8') as f:
         input = f.read().replace('\n', '')
    return input
    
def jsonDump_pred(path, output):
    with open (path,'w') as f:
        json.dump(output, f, indent=4)
        
        
def checkForDumbWords(word):
    dumb_things = ["“",".",","," "]
    for dumb_thing in dumb_things:
        if word == dumb_thing:
            return True
    return False
    
def getPredictions(preds,data):
    dumb_things = ["“",".",","," ","","”","(",")",":"]
    words = []
    labels = []
    predictions = {}
    with open (preds, 'r', encoding = "utf-8") as f : 
        rawPreds = json.load(f)
        
    entityidx = 0
    i = 1
    text_idx = 0
    
    while i < len(data):
        
        if entityidx == len(rawPreds):
            remaining = data[i:]
            for word in remaining.split(" "):
                if word in dumb_things:
                    continue
                words.append(word)
                labels.append("O")
            break
            
        elif entityidx < len(rawPreds):            
            entity = rawPreds[entityidx]
            
        next_paragraph = data[text_idx:entity['start']]
        
        #if we are in an entity we get the predicted label
        if i<=entity['end'] and i>= entity['start']:
            entity_text = entity['word'].split(" ")
            #print(entity['word'])
            for word in (entity_text):
                words.append(word)
                labels.append(entity['entity_group'])
            i = entity['end']
            entityidx+=1
            if(text_idx == 0):
                text_idx = entity['end']+1
            continue
        #Else all the words in the next paragraph leading up to the next entity gets O tagged
        for word in next_paragraph.split(" "):
            if word in dumb_things:
                continue
            words.append(word)
            labels.append("O")
            
        text_idx = entity['end']+1
        i = entity['start']
         
    return pd.DataFrame({"words": words, "ner_tags": labels})    

def main():
    path = "../../data/selv-labeled-data/ValData/ValGEN.txt"
    data = getValData(path)
    predictions = query(data)
    jsonDump_pred("./bertPredictionsB07.json",predictions)
    
if __name__ == '__main__':
    main()

