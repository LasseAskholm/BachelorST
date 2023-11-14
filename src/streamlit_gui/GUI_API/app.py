from flask import Flask, request
import time

#TODO remove comment line 5 and 20. Comment line 21

from Llama2Inference_gui import *

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

#dc1-proj287:5000/llama2
@app.route("/api/llama2", methods=["POST", "GET"])
def llama2():
    if request.method == "GET":
        return "<p>Llama2</p>"
    elif request.method == "POST":
        start = time.time()
        payload = request.json
        payload['response'] = ask_alpacha(payload['text'], payload['model'], payload['label'])
        #payload['text'] = "manually"
        end = time.time()
        payload['time'] = end-start
        return payload

if __name__ == '__main__':  
   app.run(host="0.0.0.0")
   