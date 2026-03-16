# multiple decision trees, several opinions from sources, better predictions
# regression and classifications - different trees in different ways

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from imblearn.under_sampling import RandomUnderSampler

aidf = pd.read_csv ('/Users/arjunbubbar/Desktop/Jetlearn/Data Science/Datasets/adult_income.csv',sep= ', ')

nonnumericalcols = aidf.select_dtypes (include = 'object').columns 

le = LabelEncoder ()

for i in nonnumericalcols:
    aidf [i] = le.fit_transform (aidf [i])

print (aidf.info ())

x = aidf.drop (columns = 'income')
y = aidf ['income']

rus = RandomUnderSampler (sampling_strategy='auto')

x,y = rus.fit_resample (x,y)

print (y.value_counts ())

xtrain, xtest, ytrain, ytest = train_test_split (x,y, test_size= 0.2, random_state= 20)

rfc = RandomForestClassifier ()

rfc.fit (xtrain, ytrain)

predy = rfc.predict (xtest)

print (predy)

print(f1_score(ytest, predy))
print(confusion_matrix(ytest, predy))
print(classification_report(ytest, predy))

print (ytrain.value_counts ())
print (ytest.value_counts ())

