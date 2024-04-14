from flask import Flask, request, render_template, redirect, flash
import pickle
import numpy as np
import pandas as pd
from utils.dataloader import DataLoader
from settings.constants import MODELS_DIRECTORY
from sklearn.svm import SVC
from pathlib import Path
from forms.form import PassengerForm


app = Flask(__name__)

print('model_directory')
print(MODELS_DIRECTORY.joinpath('svc_model.pkl'))
with open(MODELS_DIRECTORY.joinpath('svc_model.pkl'), 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    form = PassengerForm()
    return render_template('index.html', form=form)


# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
    # form = PassengerForm()
    # if request.method == 'POST':
    #     pclass = int(request.form['pclass'])
    #     sex = request.form['sex']
    #     age = int(request.form['age'])
    #     name = request.form['name']
    #     siblings_spouses = int(request.form['siblings_spouses'])
    #     parents_children = int(request.form['parents_children'])
    #     fare = float(request.form['fare'])
    #     embarked = request.form['embarked']
    #
    #     # Preprocess input data
    #     input_data = pd.DataFrame({
    #         'Pclass': [pclass],
    #         'Sex': [sex],
    #         'Age': [age],
    #         'Name': [name],
    #         'SibSp': [siblings_spouses],
    #         'Parch': [parents_children],
    #         'Fare': [fare],
    #         'Embarked': [embarked]
    #     })
    #     print(input_data)
    #     dl = DataLoader(input_data)
    #     preprocessed_data = dl.load_data()
    #     print(preprocessed_data)
    #     prediction = model.predict(preprocessed_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
    #                                                   'Title', 'IsAlone']])
    #
    #     probability = model.predict_proba(preprocessed_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
    #                                                   'Title', 'IsAlone']])[:, 1] * 100
    #     print(probability)
    #     print(prediction)
    #     # return render_template('index.html', prediction=prediction[0])
    #     if prediction[0] == 0:
    #         # flash("No Survived", 'danger')
    #         return render_template('not_survived.html', probability=probability[0].round())
    #     else:
    #         return render_template('survive.html', probability=probability[0].round())
    # return render_template('index.html')


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

        prediction = model.predict(
            preprocessed_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'IsAlone']])
        probability = model.predict_proba(
            preprocessed_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'IsAlone']])[
                      :, 1] * 100

        if prediction[0] == 0:
            return render_template('not_survived.html', probability=round(probability[0]))
        else:
            return render_template('survive.html', probability=round(probability[0]))
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.secret_key = 'secret_key'
    app.run(debug=True)
