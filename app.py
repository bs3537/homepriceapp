from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import pickle 
import joblib
import json
import numpy as np
import requests
import pandas as pd

#https://www.kdnuggets.com/2019/10/easily-deploy-machine-learning-models-using-flask.html

app = Flask(__name__)

def create_app():
    model = joblib.load(open('ameshousing_model.pkl', 'rb'))

#################APP ROUTES####################

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/predict', methods=['POST', 'GET'])
    def predict():
        model = joblib.load(open('ameshousing_model.pkl', 'rb'))
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        output = 2.4 * round(prediction[0])
        return render_template('index.html', prediction_text='The predicted home price is = $ {}'.format(output))



    @app.route('/results',methods=['POST'])
    def results():

        data = request.get_json(force=True)
        model = joblib.load(open('ameshousing_model.pkl', 'rb'))
        prediction = model.predict([np.array(list(data.values()))])

        output = prediction[0]
        return jsonify(output)
        
    return app

if __name__ == "__main__":
    app.run(debug=True)
