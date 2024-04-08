import re
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from settings.constants import TRAIN_DATA, MODELS_DIRECTORY


# current_script_dir = Path(__file__).resolve().parent.parent
# print(current_script_dir)


class DataLoader:
    def __init__(self, data, model=None):
        self.data = data.copy()
        self.model = model

    @staticmethod
    def map_fare_to_interval(data) -> list:
        res = []
        for value in data:
            if value < 7.91:
                res.append(0)
            elif value < 14.454:
                res.append(1)
            elif value < 31.0:
                res.append(2)
            else:
                res.append(3)
        return res

    @staticmethod
    def get_title(name: str) -> str:
        pattern = ' ([A-Za-z]+)\.'
        title_search = re.search(pattern, name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""

    def load_data(self) -> pd.DataFrame:

        self.data['Title'] = self.data['Name'].apply(self.get_title)

        # replace
        self.data['Title'] = self.data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                                         'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
                                                        'Rare')
        # replace
        self.data['Title'] = self.data['Title'].replace(['Mlle', 'Ms'], 'Miss')
        # replace
        self.data['Title'] = self.data['Title'].replace('Mme', 'Mrs')

        title_map = {'Rare': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Master': 4}
        self.data['Title'] = self.data['Title'].map(title_map)

        # fill nans
        self.data['Title'] = self.data['Title'].fillna(0)

        # drop columns
        drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',
                         'Parch', 'FamilySize']

        self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())

        self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode()[0])

        sex_mapping = {'female': 0, 'male': 1}

        self.data['Sex'] = self.data['Sex'].map(sex_mapping)

        embark_mapping = {'S': 0, 'C': 1, 'Q': 2}

        self.data['Embarked'] = self.data['Embarked'].map(embark_mapping)

        age_intervals = [0, 16, 32, 48, 64, 80]

        self.data['Age'] = pd.cut(self.data['Age'], bins=age_intervals, labels=False).astype(int)

        self.data['Fare'] = self.map_fare_to_interval(self.data['Fare'])

        return self.data

    def train_model(self):
        X = self.data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title']]
        y = self.data['Survived']
        print(X.head())

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

        svc_classifier = SVC(probability=True)
        svc_classifier.fit(X_train, y_train)
        self.model = svc_classifier

        train_predictions = svc_classifier.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        print(train_accuracy)
        predictions = svc_classifier.predict(X_test)
        rfc_accuracy = accuracy_score(y_test, predictions)
        print(rfc_accuracy)
        return predictions

    def save_model(self):

        with open(MODELS_DIRECTORY.joinpath('svc_model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)


df = pd.read_csv(TRAIN_DATA)

dp = DataLoader(df)
res = dp.load_data()
# print(res.head())
# display(res)
split = dp.train_model()
dp.save_model()

# print(split)
