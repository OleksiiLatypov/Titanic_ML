from flask import Flask, request, render_template, redirect, flash
import pickle
import numpy as np
import pandas as pd
from utils.dataloader import DataLoader
from settings.constants import MODELS_DIRECTORY
from forms.form import PassengerForm

app = Flask(__name__)

with open(MODELS_DIRECTORY.joinpath('svc_model.pkl'), 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    form = PassengerForm()
    return render_template('index.html', form=form)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = PassengerForm(request.form)
    if request.method == 'POST' and form.validate():
        pclass = int(form.pclass.data)
        sex = form.sex.data
        age = int(form.age.data)
        name = form.name.data
        siblings_spouses = int(form.siblings_spouses.data)
        parents_children = int(form.parents_children.data)
        fare = float(form.fare.data)
        embarked = form.embarked.data

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
        dl = DataLoader(input_data)
        preprocessed_data = dl.load_data()
        data_to_predict = preprocessed_data[
            ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'IsAlone']]
        prediction = model.predict(data_to_predict)
        probability = model.predict_proba(data_to_predict)[:, 1] * 100

        if prediction[0] == 0:
            return render_template('not_survived.html', probability=round(probability[0]))
        else:
            return render_template('survive.html', probability=round(probability[0]))
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.secret_key = 'secret_key'
    app.run(debug=True, host='0.0.0.0', port=8000)
