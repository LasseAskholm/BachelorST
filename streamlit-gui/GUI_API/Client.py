import requests

API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/LazzeKappa/BERT_B07"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def inference_bert(payload, URL):
	response = requests.post(URL, headers=headers, json={"inputs": payload, "options":{"wait_for_model":True}})
	return response.json()
	
# output = inference_bert({
# 	"inputs": "My name is Sarah Jessica Parker but you can call me Jessica",
# })

# OUTPUT_Format = [
# 	{'entity_group': 'Person', 'score': 0.8346006274223328, 'word': 'Sarah Jessica Parker', 'start': 11, 'end': 31}, 
# 	{'entity_group': 'Person', 'score': 0.7973623275756836, 'word': 'Jessica', 'start': 52, 'end': 59}
# 	]

def print_output(response):
	for i in response:
		print(i['word'] + ": " + i['entity_group'] + "  => Score: " + str(i['score']))


def inference_llama2_test(payload):
	url = "http://127.0.0.1:5000/llama2"
	res = requests.post(url, json=payload)
	return res.json()

output2 = inference_llama2_test({
	"inputs": "My name is Sarah Jessica Parker but you can call me Jessica",
})
