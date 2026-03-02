# Can be used for regression and classification, data structure, decision tree can interpret more variety in 

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score

cars_df = pd.read_csv ('/Users/arjunbubbar/Desktop/Jetlearn/Data Science/Datasets/car.csv')

X = cars_df.drop (columns=["class"])
y = cars_df ["class"]

le = LabelEncoder()
y = le.fit_transform(y)

print (cars_df.info ())

xtrain, xtest, ytrain, ytest = train_test_split (X, y, test_size=0.2, random_state=20)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 1, 2, 3, 4, 5])],remainder='passthrough')

xtrain_enc = ct.fit_transform(xtrain)
xtest_enc = ct.transform(xtest)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier ()

dtc.fit (xtrain_enc, ytrain)

predy = dtc.predict (xtest_enc)

print (predy)

print(confusion_matrix(ytest, predy))
print(classification_report(ytest, predy))
