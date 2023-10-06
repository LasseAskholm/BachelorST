import requests
import json

def get_API_data():
    API_URL = "https://api-inference.huggingface.co/models/LazzeKappa/models"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
        
    output = query({
        "inputs": "My name is Sarah Jessica Parker but you can call me Jessica",
    })
    return output


def get_sample_data():
	# hack to test gui
    f = open('gui/sample.json')
    sample_results = json.load(f)
    f.close()
    return sample_results