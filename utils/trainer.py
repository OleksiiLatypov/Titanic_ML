import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from settings.constants import TRAIN_DATA, MODELS_DIRECTORY
from dataloader import DataLoader


class Trainer:
    def __init__(self, model=None):
        self.df = pd.read_csv(TRAIN_DATA)
        self.data = DataLoader(self.df).load_data()
        self.model = model

    def train_model(self, data):
        X = self.data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'IsAlone']]
        y = self.data['Survived']
        print(X.head())

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

        svc_classifier = SVC(probability=True)
        svc_classifier.fit(X_train, y_train)
        self.model = svc_classifier

        train_predictions = svc_classifier.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        print(train_accuracy)
        predictions = svc_classifier.predict(X_test)
        print(predictions)
        svc_accuracy = accuracy_score(y_test, predictions)
        print(svc_accuracy)

        return predictions

    def save_model(self):
        with open(MODELS_DIRECTORY.joinpath('svc_model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)


if __name__ == '__main__':
    t = Trainer()
    result = t.train_model(TRAIN_DATA)
    t.save_model()
    print(result)
