import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import sklearn


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = (prediction[0]).round(2)

    return render_template('output.html', prediction_text='Property price will be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)