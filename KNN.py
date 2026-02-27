# k denotes a number, NN - nearest numbers, used for regression and classification, kneighborsclassifier or kneighborsregressor
# important to scale before building the model, since its easier to close the distance then analyse data
# min max scales, sets between 0 and 1

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

iris_df = pd.read_csv ('/Users/arjunbubbar/Desktop/Jetlearn/Data Science/Datasets/iris.csv')

le = LabelEncoder ()

iris_df ['species'] = le.fit_transform (iris_df['species'])

print (iris_df.info ())

x = iris_df.drop (columns=['species'])
y = iris_df ['species']

mms = MinMaxScaler ()

x = mms.fit_transform (x)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score

xtrain, xtest, ytrain, ytest = train_test_split (x,y, test_size=0.2, random_state=19)

model = KNeighborsClassifier (1)

model.fit (xtrain, ytrain)

predy = model.predict (xtest)

print (predy)

cm = confusion_matrix (ytest, predy)

print (cm)

cr = classification_report (ytest, predy)

print (cr)

# numbers of neighbours can range from 1 - sqrt (n), so here its 11 for example

nrows = round (np.sqrt (xtrain.shape [0]))

print (nrows)

scores = []

for i in range (1,nrows+1):
    model = KNeighborsClassifier ()
    model.fit (xtrain, ytrain)
    predy = model.predict (xtest)
    f = f1_score (ytest, predy, average = 'macro')
    scores.append (f)

print (scores)

max = max (scores)
bestk = scores.index (max) + 1
print (bestk)
    









