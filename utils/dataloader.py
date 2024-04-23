import re
import pandas as pd


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
    def get_title(name):
        pattern = ' ([A-Za-z]+)\.'
        title_search = re.search(pattern, name)
        # If the title exists, extract and return it.
        if title_search:
            title = title_search.group(1)
            title_map = {'Rare': 4, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Master': 0}
            return title_map.get(title, 0)  # Return 0 for unknown titles
        return 0

    def load_data(self) -> pd.DataFrame:
        self.data['FamilySize'] = self.data['SibSp'] + self.data['Parch'] + 1

        # replace value
        self.data['IsAlone'] = 0
        self.data.loc[self.data['FamilySize'] == 1, 'IsAlone'] = 1

        # apply regex
        self.data['Title'] = self.data['Name'].apply(self.get_title)
        # replace
        self.data['Title'] = self.data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                                         'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
                                                        'Rare')
        # replace
        self.data['Title'] = self.data['Title'].replace(['Mlle', 'Ms'], 'Miss')
        # replace
        self.data['Title'] = self.data['Title'].replace('Mme', 'Mrs')
        # fill nans
        self.data['Title'] = self.data['Title'].fillna(0)

        self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())

        self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode()[0])

        sex_mapping = {'female': 0, 'male': 1}

        self.data['Sex'] = self.data['Sex'].map(sex_mapping)

        embark_mapping = {'C': 0, 'Q': 1, 'S': 2}

        self.data['Embarked'] = self.data['Embarked'].map(embark_mapping)

        age_intervals = [0, 16, 32, 48, 64, 80]

        self.data['Age'] = pd.cut(self.data['Age'], bins=age_intervals, labels=False).astype(int)

        self.data['Fare'] = self.map_fare_to_interval(self.data['Fare'])

        return self.data

