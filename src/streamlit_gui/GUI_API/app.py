from flask import Flask, request

#TODO remove comment line 5 and 20. Comment line 21

#from inference.Llama2Inference import *

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
        payload = request.json
        # payload['text'] = ask_alpacha(text, payload['model'])
        payload['text'] = "manually"
        return payload

if __name__ == '__main__':  
   app.run(host="0.0.0.0")
   