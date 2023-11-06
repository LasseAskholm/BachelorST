import requests
import json

API_URL = "https://api-inference.huggingface.co/models/LazzeKappa/BERT_B08"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json={"inputs": payload, "options":{"wait_for_model":True}})
	return response.json()

if __name__ == '__main__':
    path = "../../data/selv-labeled-data/ValData/ValGEN.txt"
    
    with open (path, "r", encoding='utf-8') as f:
         input = f.read().replace('\n', '')
    
    output = query(input)
    #json_string = json.dumps(output)
    
    with open ('bertPredictionsB08.json','w') as f:
        json.dump(output, f, indent=4)
