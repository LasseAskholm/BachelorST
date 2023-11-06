from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/llama2", methods=["POST", "GET"])
def llama2():
    if request.method == "GET":
        return "<p>Llama2</p>"
    else:
        data = request.json()
        print(data)
        return request.form

if __name__ == '__main__':  
   app.run()