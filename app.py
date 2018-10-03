from flask import Flask, render_template, json, request
from flask import jsonify

import nltk

from prediction.prediction import prediction
obj=prediction()
app = Flask(__name__)
pos=0
neg=0

@app.route('/predict',methods=['POST'])
def predict():
    text = request.json['feedback']
    global pos,neg
    result=obj.predict([text])
    print(result)


    word_freqs, max_freq=obj.word_cloud()

    response = {}
    response["word_freqs"] = str(word_freqs)
    response["max_freq"]= max_freq
    response["result"]=  str(result)
    response["pos"]= pos
    response["neg"]= neg

    return jsonify(response)
@app.route('/')
def ren():
    return render_template("main.html")

if __name__ == '__main__':



    app.run(debug=True)