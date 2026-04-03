import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

import os

print("\"\"\"")
for dirname, _, filenames in os.walk('ExploratoryDataAnalysis-TitanicDataset-Kaggle\\datasets'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
print("\"\"\"")

train_data = pd.read_csv("ExploratoryDataAnalysis-TitanicDataset-Kaggle\\datasets\\train.csv")
print(train_data.head())

test_data = pd.read_csv("ExploratoryDataAnalysis-TitanicDataset-Kaggle\\datasets\\test.csv")
print(test_data.head())

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("\npercentage of women who survived:", rate_women, "\n")

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("\npercentage of men who survived:", rate_men, "\n")

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('ExploratoryDataAnalysis-TitanicDataset-Kaggle\\datasets\\submission.csv', index=False)