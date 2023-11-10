from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

#dc1-proj287:5000/llama2
@app.route("/llama2", methods=["POST", "GET"])
def llama2():
    if request.method == "GET":
        return "<p>Llama2</p>"
    else:
        data = request.json

        # Call Llama 2 inference
        # Return inference results

        
        return data

if __name__ == '__main__':  
   app.run(host="0.0.0.0")
   