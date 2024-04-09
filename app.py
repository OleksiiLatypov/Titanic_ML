from flask import Flask, request, render_template, redirect, flash
import pickle
import numpy as np
import pandas as pd
from utils.dataloader import DataLoader
from settings.constants import MODELS_DIRECTORY
from sklearn.svm import SVC
from pathlib import Path

app = Flask(__name__)

print('model_directory')
print(MODELS_DIRECTORY.joinpath('svc_model.pkl'))
with open(MODELS_DIRECTORY.joinpath('svc_model.pkl'), 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = int(request.form['age'])
        name = request.form['name']
        siblings_spouses = int(request.form['siblings_spouses'])
        parents_children = int(request.form['parents_children'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked']

        # Preprocess input data
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'Name': [name],
            'SibSp': [siblings_spouses],
            'Parch': [parents_children],
            'Fare': [fare],
            'Embarked': [embarked]
        })
        print(input_data)
        dl = DataLoader(input_data)
        preprocessed_data = dl.load_data()
        print(preprocessed_data)
        prediction = model.predict(preprocessed_data[['Pclass',	'Sex',	'Age', 'SibSp', 'Parch','Fare',	'Embarked', 'Title', 'IsAlone']])
        print(prediction)
        # return render_template('index.html', prediction=prediction[0])
        if prediction[0] == 0:
            #flash("No Survived", 'danger')
            return render_template('not_survived.html')
        else:
            flash("Survived", 'success')
    return render_template('index.html')


if __name__ == '__main__':
    app.secret_key='secret_key'
    app.run(debug=True)
