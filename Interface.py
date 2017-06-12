
# -*- coding: utf-8 -*-
import json
import sys
from flask import Flask
from flask import request
from flask_cors import CORS


reload(sys)
sys.setdefaultencoding('utf8')

# Defining Flask server
app = Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy dog'
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)


@app.route("/getRating", methods=['POST', 'GET'])
def getRating():
    import Movie as M 
    M.getFinalRating()
    data = ""
    with open("/home/tharindra/PycharmProjects/WorkBench/DataMiningAssignment/hello.txt") as file:
        data = file.read()
    print(data)
    return data

def printLog(level, message):
    if level == 'info':
        print "[INFO]: ", message
    elif level == 'error':
        print "[ERROR]: ", message
    elif level == 'process':
        print "[PROCESS]: ", message


if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', debug=True, port=3030)
    except Exception as e:
        print(e)
