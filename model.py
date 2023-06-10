import numpy as np
import pandas as pd
import pickle
import json
from flask import Flask,render_template,url_for,jsonify,request

#create a Flask object
app = Flask(__name__)
#Loading the pickle files (model)
regmodel = pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/',methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template('home.html',
                           prediction_text = "The House Price is {}".format(output))
