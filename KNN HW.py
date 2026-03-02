import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier
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

nrows = round(np.sqrt(xtrain_enc.shape[0]))
scores = []

for k in range(1, nrows + 1):
    model = KNeighborsClassifier(k)
    model.fit(xtrain_enc, ytrain)
    predy = model.predict(xtest_enc)
    scores.append(f1_score(ytest, predy, average='macro'))

best_score = max(scores)
bestk = scores.index(best_score) + 1
print(bestk, best_score)

model = KNeighborsClassifier(bestk)
model.fit(xtrain_enc, ytrain)
predy = model.predict(xtest_enc)

print(confusion_matrix(ytest, predy))
print(classification_report(ytest, predy))
