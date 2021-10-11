from flask import Flask, app, request, jsonify, render_template
import numpy as np
import pickle

app=Flask(__name__)
model = pickle.load(open('predictmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    inputs_data = [int(data) for data in request.form.values()]
    features = [np.array(inputs_data)]
    prediction = model.predict(features)

    output = round(prediction[0],2)

    if output == 1:
        return render_template('index.html',prediction_text = 'The chance of heart attack is high!')
    if output == 0:
        return render_template('index.html',prediction_text = 'The chance of heart attack is Low! You are safe!')

if __name__ == "__main__":
    app.run(debug=True)
