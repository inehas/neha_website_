from flask import Flask 
#app is object of Flask class
app = Flask(__name__)
## what should we return when a app i run
@app.route("/")
#when url / return hellow owrld
def hello_world():
    return "<p>Hello, World!</p>"
if(__name__ == "__main__") :
    app.run(host = '0.0.0.0',debug=True)
