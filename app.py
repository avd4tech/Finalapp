from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# loading model
model = pickle.load(open('model.pkl','rb'))

# creating app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #  Quantity, Catalyst, Ash, Water,Plasticizer,Mod_Aggregator,Ref_Aggregator,Duration
    Quantity = float(request.form['Quantity'])
    Catalyst = float(request.form['Catalyst'])
    Ash = float(request.form['Ash'])
    Water = float(request.form['Water'])
    Plasticizer = float(request.form['Plasticizer'])
    Mod_Aggregator = float(request.form['Mod_Aggregator'])
    Ref_Aggregator = float(request.form['Ref_Aggregator'])
    Duration = int(request.form['Duration'])

    # transform input features
    features = np.array([Quantity, Catalyst, Ash, Water,Plasticizer,Mod_Aggregator,Ref_Aggregator,Duration]).reshape(1, -1)
    prediction = model.predict(features).reshape(1, -1)

    return render_template('index.html', strength=prediction[0][0])

# python main
if __name__ == "__main__":
    app.run(debug=True)
